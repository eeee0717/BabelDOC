import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
from babeldoc.assets import embedding_assets_metadata
from babeldoc.assets.embedding_assets_metadata import CMAP_METADATA
from babeldoc.assets.embedding_assets_metadata import CMAP_URL_BY_UPSTREAM
from babeldoc.assets.embedding_assets_metadata import DOC_LAYOUT_ONNX_MODEL_URL
from babeldoc.assets.embedding_assets_metadata import (
    DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
)
from babeldoc.assets.embedding_assets_metadata import (
    DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SIZE,
)
from babeldoc.assets.embedding_assets_metadata import EMBEDDING_FONT_METADATA
from babeldoc.assets.embedding_assets_metadata import FONT_METADATA_URL
from babeldoc.assets.embedding_assets_metadata import FONT_URL_BY_UPSTREAM
from babeldoc.assets.embedding_assets_metadata import TIKTOKEN_CACHE_SIZES
from babeldoc.assets.embedding_assets_metadata import TIKTOKEN_CACHES
from babeldoc.assets.embedding_assets_metadata import TIKTOKEN_URL_BY_UPSTREAM
from babeldoc.const import get_cache_file_path
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


_FASTEST_FONT_UPSTREAM_LOCK = asyncio.Lock()
_FASTEST_FONT_UPSTREAM: str | None = None
_FASTEST_FONT_METADATA: dict | None = None
_ASSET_PROGRESS_CALLBACK: Callable[[str, str, float | None], None] | None = None
_ASSET_UPSTREAM_ENV = "BABELDOC_ASSET_UPSTREAM"
# ModelScope resolves every file through a signed CDN redirect and needs 5-10s
# even for a 13 KB metadata file, so httpx's 5s default times out on the very
# first request and pins the whole run to an overseas upstream.
_ASSET_HTTP_TIMEOUT = httpx.Timeout(30.0)


def set_asset_progress_callback(
    callback: Callable[[str, str, float | None], None] | None,
) -> Callable[[str, str, float | None], None] | None:
    """Set the process-local asset transfer reporter and return its previous value."""
    global _ASSET_PROGRESS_CALLBACK
    previous = _ASSET_PROGRESS_CALLBACK
    _ASSET_PROGRESS_CALLBACK = callback
    return previous


def _report_asset_progress(stage: str, asset_id: str, progress: float | None) -> None:
    if _ASSET_PROGRESS_CALLBACK is not None:
        _ASSET_PROGRESS_CALLBACK(stage, asset_id, progress)


def _configured_asset_upstream() -> str | None:
    value = os.getenv(_ASSET_UPSTREAM_ENV)
    if value is None or not value.strip():
        return None
    upstream = value.strip().lower()
    if upstream not in FONT_METADATA_URL:
        choices = ", ".join(FONT_METADATA_URL)
        raise ValueError(f"Invalid {_ASSET_UPSTREAM_ENV}: {value!r}; choose {choices}")
    return upstream


def _log_asset_request(
    *,
    source: str,
    original_url: str,
    final_url: str,
    byte_count: int,
    elapsed_ms: int,
    verification: str,
    result: str,
) -> None:
    logger.info(
        "Asset request audit %s",
        json.dumps(
            {
                "source": source,
                "original_url": original_url,
                "final_url": final_url,
                "bytes": byte_count,
                "elapsed_ms": elapsed_ms,
                "verification": verification,
                "result": result,
            },
            sort_keys=True,
        ),
    )


class AssetProgressBatch:
    """Aggregate byte progress for a fixed set of assets."""

    def __init__(self, asset_sizes: dict[str, int]):
        self.asset_sizes = {
            asset_id: size for asset_id, size in asset_sizes.items() if size > 0
        }
        self.batch_id = f"asset-batch:{id(self)}"
        self._progress_by_stage: dict[str, dict[str, int]] = {}
        self._last_by_stage: dict[str, float] = {}
        self._lock = threading.Lock()

    def report_bytes(self, stage: str, asset_id: str, completed: int) -> None:
        expected = self.asset_sizes.get(asset_id)
        if expected is None:
            return
        with self._lock:
            stage_progress = self._progress_by_stage.setdefault(stage, {})
            stage_progress[asset_id] = max(
                stage_progress.get(asset_id, 0), min(expected, max(0, completed))
            )
            total = sum(self.asset_sizes.values())
            progress = sum(stage_progress.values()) * 100 / total
            progress = max(self._last_by_stage.get(stage, 0.0), progress)
            self._last_by_stage[stage] = progress
            _report_asset_progress(stage, self.batch_id, progress)

    def complete(self, stage: str, asset_id: str) -> None:
        expected = self.asset_sizes.get(asset_id)
        if expected is not None:
            self.report_bytes(stage, asset_id, expected)

    def report_indeterminate(self, stage: str) -> None:
        _report_asset_progress(stage, self.batch_id, None)


class ResultContainer:
    def __init__(self):
        self.result = None
        self.exception: BaseException | None = None

    def set_result(self, result):
        self.result = result

    def set_exception(self, exc: BaseException):
        self.exception = exc


def run_in_another_thread(coro):
    result_container = ResultContainer()

    def _wrapper():
        try:
            result_container.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            result_container.set_exception(exc)

    thread = threading.Thread(target=_wrapper)
    thread.start()
    thread.join()
    if result_container.exception is not None:
        msg = (
            "asset coroutine failed: "
            f"{type(result_container.exception).__name__}: {result_container.exception}"
        )
        raise RuntimeError(msg) from result_container.exception
    return result_container.result


def run_coro(coro):
    return run_in_another_thread(coro)


def _retry_if_not_cancelled_and_failed(retry_state):
    """Only retry if the exception is not CancelledError and the attempt failed."""
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        # Don't retry on CancelledError
        if isinstance(exception, asyncio.CancelledError):
            logger.debug("Operation was cancelled, not retrying")
            return False
        # Retry on network related errors
        if isinstance(
            exception, httpx.HTTPError | ConnectionError | ValueError | TimeoutError
        ):
            logger.warning(f"Network error occurred: {exception}, will retry")
            return True
    # Don't retry on success
    return False


def verify_file(
    path: Path,
    sha3_256: str,
    progress_batch: AssetProgressBatch | None = None,
    *,
    report_progress: bool = True,
):
    asset_id = str(path)
    if not path.exists():
        if report_progress and progress_batch is not None:
            progress_batch.complete("checking_assets", asset_id)
        return False
    total = path.stat().st_size
    if report_progress:
        if progress_batch is not None:
            progress_batch.report_bytes("checking_assets", asset_id, 0)
        else:
            _report_asset_progress("checking_assets", asset_id, 0.0 if total else None)
    checked = 0
    hash_ = hashlib.sha3_256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hash_.update(chunk)
            checked += len(chunk)
            if report_progress:
                if progress_batch is not None:
                    progress_batch.report_bytes("checking_assets", asset_id, checked)
                else:
                    _report_asset_progress(
                        "checking_assets",
                        asset_id,
                        min(100.0, checked * 100 / total) if total else None,
                    )
    if report_progress and progress_batch is not None:
        progress_batch.complete("checking_assets", asset_id)
    return hash_.hexdigest() == sha3_256


@retry(
    retry=_retry_if_not_cancelled_and_failed,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    before_sleep=lambda retry_state: logger.warning(
        f"Download file failed, retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number}/3)"
    ),
)
async def download_file(
    client: httpx.AsyncClient | None = None,
    url: str = None,
    path: Path = None,
    sha3_256: str = None,
    progress_batch: AssetProgressBatch | None = None,
    source: str = "unknown",
):
    async def transfer(active_client: httpx.AsyncClient) -> None:
        asset_id = str(path)
        if progress_batch is not None:
            progress_batch.report_indeterminate("downloading_assets")
        else:
            _report_asset_progress("downloading_assets", asset_id, None)
        temp_path = path.with_name(f"{path.name}.part")
        response: httpx.Response | None = None
        started_at = time.perf_counter()
        final_url = url
        downloaded = 0
        verification = "not_run"
        result = "failed"
        try:
            request = active_client.build_request("GET", url)
            response = await active_client.send(
                request, follow_redirects=True, stream=True
            )
            final_url = str(response.url)
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            try:
                total = int(content_length) if content_length is not None else None
            except ValueError:
                total = None
            if progress_batch is not None:
                progress_batch.report_bytes("downloading_assets", asset_id, 0)
            else:
                _report_asset_progress(
                    "downloading_assets",
                    asset_id,
                    0.0 if total and total > 0 else None,
                )
            with temp_path.open("wb") as file:
                async for chunk in response.aiter_bytes(64 * 1024):
                    file.write(chunk)
                    downloaded += len(chunk)
                    if progress_batch is not None:
                        progress_batch.report_bytes(
                            "downloading_assets", asset_id, downloaded
                        )
                    else:
                        progress = (
                            min(100.0, downloaded * 100 / total)
                            if total and total > 0
                            else None
                        )
                        _report_asset_progress("downloading_assets", asset_id, progress)
            if not verify_file(temp_path, sha3_256, report_progress=False):
                verification = "failed"
                raise ValueError(f"File {path} is corrupted")
            verification = "passed"
            temp_path.replace(path)
            result = "success"
            if progress_batch is not None:
                progress_batch.complete("downloading_assets", asset_id)
            else:
                _report_asset_progress("downloading_assets", asset_id, 100.0)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        finally:
            if response is not None:
                await response.aclose()
            _log_asset_request(
                source=source,
                original_url=url,
                final_url=final_url,
                byte_count=downloaded,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000),
                verification=verification,
                result=result,
            )

    if client is None:
        async with httpx.AsyncClient(timeout=_ASSET_HTTP_TIMEOUT) as owned_client:
            await transfer(owned_client)
    else:
        await transfer(client)


def _remember_fastest_upstream(upstream: str, font_metadata: dict | None) -> None:
    """Keep the cached upstream and its metadata consistent."""
    global _FASTEST_FONT_METADATA, _FASTEST_FONT_UPSTREAM
    if upstream in FONT_METADATA_URL and font_metadata is not None:
        _FASTEST_FONT_UPSTREAM = upstream
        _FASTEST_FONT_METADATA = font_metadata


async def download_file_with_fallback(
    client: httpx.AsyncClient | None,
    urls: dict[str, str],
    path: Path,
    sha3_256: str,
    preferred_upstream: str,
    font_metadata: dict | None = None,
    progress_batch: AssetProgressBatch | None = None,
    enabled_upstreams: set[str] | None = None,
) -> str:
    """Download from the preferred upstream, then the others in turn.

    A reachable metadata host does not imply a reachable download host: the
    HuggingFace LFS redirect lands on a separate CDN domain that a proxy rule
    matching only huggingface.co never covers. Retrying one URL cannot recover
    from that, so exhaust the other upstreams before giving up.
    """
    allowed = enabled_upstreams or set(FONT_METADATA_URL)
    enabled_urls = {
        upstream: url for upstream, url in urls.items() if upstream in allowed
    }
    ordered = [preferred_upstream] if preferred_upstream in enabled_urls else []
    ordered.extend(
        upstream for upstream in enabled_urls if upstream != preferred_upstream
    )

    last_exception: Exception = ValueError(f"No upstream URL for {path.name}")
    for upstream in ordered:
        try:
            await download_file(
                client,
                enabled_urls[upstream],
                path,
                sha3_256,
                progress_batch,
                upstream,
            )
        except Exception as e:  # noqa: BLE001
            last_exception = e
            logger.warning(f"Download {path.name} from {upstream} failed: {e}")
            continue
        _remember_fastest_upstream(upstream, font_metadata)
        return upstream
    raise last_exception


@retry(
    retry=_retry_if_not_cancelled_and_failed,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    before_sleep=lambda retry_state: logger.warning(
        f"Get font metadata failed, retrying in {retry_state.next_action.sleep} seconds... "
        f"(Attempt {retry_state.attempt_number}/3)"
    ),
)
async def get_font_metadata(
    client: httpx.AsyncClient | None = None, upstream: str = None
):
    if upstream not in FONT_METADATA_URL:
        logger.critical(f"Invalid upstream: {upstream}")
        exit(1)

    async def request_metadata(active_client: httpx.AsyncClient):
        url = FONT_METADATA_URL[upstream]
        started_at = time.perf_counter()
        final_url = url
        byte_count = 0
        result = "failed"
        try:
            response = await active_client.get(url, follow_redirects=True)
            final_url = str(response.url)
            byte_count = len(response.content)
            response.raise_for_status()
            metadata = response.json()
            result = "success"
            logger.debug(f"Get font metadata from {upstream} success")
            return upstream, metadata
        finally:
            _log_asset_request(
                source=upstream,
                original_url=url,
                final_url=final_url,
                byte_count=byte_count,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000),
                verification="not_applicable",
                result=result,
            )

    if client is None:
        async with httpx.AsyncClient(timeout=_ASSET_HTTP_TIMEOUT) as owned_client:
            return await request_metadata(owned_client)
    return await request_metadata(client)


async def _get_fastest_upstream_for_font_internal(
    client: httpx.AsyncClient | None = None, exclude_upstream: list[str] | None = None
) -> tuple[str | None, dict | None]:
    """Find the fastest upstream for font metadata without using cached result."""
    excluded = set(exclude_upstream or [])
    configured = _configured_asset_upstream()
    if configured is not None and configured not in excluded:
        ordered = [configured]
        ordered.extend(
            upstream
            for upstream in FONT_METADATA_URL
            if upstream != configured and upstream not in excluded
        )
        for upstream in ordered:
            try:
                return await get_font_metadata(client, upstream)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error getting font metadata from {upstream}: {e}")
        logger.error("All upstreams failed")
        return None, None

    tasks: list[asyncio.Task[tuple[str, dict]]] = []
    for upstream in FONT_METADATA_URL:
        if upstream in excluded:
            continue
        tasks.append(asyncio.create_task(get_font_metadata(client, upstream)))
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            for task in tasks:
                if not task.done():
                    task.cancel()
            return result
        except Exception as e:
            logger.exception(f"Error getting font metadata: {e}")
    logger.error("All upstreams failed")
    return None, None


async def get_fastest_upstream_for_font(
    client: httpx.AsyncClient | None = None, exclude_upstream: list[str] | None = None
) -> tuple[str | None, dict | None]:
    """Get the fastest upstream for font metadata with cached result.

    The cached upstream is only used when exclude_upstream is None.
    """
    global _FASTEST_FONT_UPSTREAM, _FASTEST_FONT_METADATA

    if exclude_upstream is None and _FASTEST_FONT_UPSTREAM is not None:
        return _FASTEST_FONT_UPSTREAM, _FASTEST_FONT_METADATA

    if exclude_upstream is not None:
        # Do not use or update cache when exclude_upstream is provided.
        return await _get_fastest_upstream_for_font_internal(client, exclude_upstream)

    async with _FASTEST_FONT_UPSTREAM_LOCK:
        if _FASTEST_FONT_UPSTREAM is not None:
            return _FASTEST_FONT_UPSTREAM, _FASTEST_FONT_METADATA

        upstream, metadata = await _get_fastest_upstream_for_font_internal(client)
        if upstream is not None:
            _FASTEST_FONT_UPSTREAM = upstream
            _FASTEST_FONT_METADATA = metadata
            logger.info(f"Fastest font upstream determined: {upstream}")
        return upstream, metadata


async def get_fastest_upstream_for_model(client: httpx.AsyncClient | None = None):
    return await get_fastest_upstream_for_font(client, exclude_upstream=["github"])


async def get_fastest_upstream(client: httpx.AsyncClient | None = None):
    (
        fastest_upstream_for_font,
        online_font_metadata,
    ) = await get_fastest_upstream_for_font(client)
    if fastest_upstream_for_font is None:
        logger.error("Failed to get fastest upstream")
        exit(1)

    if fastest_upstream_for_font == "github":
        # since github is only store font, we need to get the fastest upstream for model
        fastest_upstream_for_model, _ = await get_fastest_upstream_for_model(client)
        if fastest_upstream_for_model is None:
            logger.error("Failed to get fastest upstream")
            exit(1)
    else:
        fastest_upstream_for_model = fastest_upstream_for_font

    return online_font_metadata, fastest_upstream_for_font, fastest_upstream_for_model


async def get_doclayout_onnx_model_path_async(
    client: httpx.AsyncClient | None = None,
    progress_batch: AssetProgressBatch | None = None,
):
    onnx_path = get_cache_file_path(
        "doclayout_yolo_docstructbench_imgsz1024.onnx", "models"
    )
    progress_batch = progress_batch or AssetProgressBatch(
        {str(onnx_path): DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SIZE}
    )
    if verify_file(
        onnx_path,
        DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
        progress_batch,
    ):
        return onnx_path

    logger.info("doclayout onnx model not found or corrupted, downloading...")
    fastest_upstream, font_metadata = await get_fastest_upstream_for_model(client)
    if fastest_upstream is None:
        logger.error("Failed to get fastest upstream")
        exit(1)

    upstream = await download_file_with_fallback(
        client,
        DOC_LAYOUT_ONNX_MODEL_URL,
        onnx_path,
        DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
        fastest_upstream,
        font_metadata,
        progress_batch,
    )
    logger.info(f"Download doclayout onnx model from {upstream} success")
    return onnx_path


async def get_table_detection_rapidocr_model_path_async(
    client: httpx.AsyncClient | None = None,
):
    _ = client
    dummy_path = get_cache_file_path("rapidocr-retired-dummy.txt", "models")
    dummy_path.write_text("RapidOCR table detection is retired.\n", encoding="utf-8")
    return dummy_path


def get_doclayout_onnx_model_path():
    return run_coro(get_doclayout_onnx_model_path_async())


def get_table_detection_rapidocr_model_path():
    return run_coro(get_table_detection_rapidocr_model_path_async())


def get_font_url_by_name_and_upstream(font_file_name: str, upstream: str):
    if upstream not in FONT_URL_BY_UPSTREAM:
        logger.critical(f"Invalid upstream: {upstream}")
        exit(1)

    return FONT_URL_BY_UPSTREAM[upstream](font_file_name)


async def get_font_and_metadata_async(
    font_file_name: str,
    client: httpx.AsyncClient | None = None,
    fastest_upstream: str | None = None,
    font_metadata: dict | None = None,
    progress_batch: AssetProgressBatch | None = None,
):
    cache_file_path = get_cache_file_path(font_file_name, "fonts")
    embedded_metadata = EMBEDDING_FONT_METADATA.get(font_file_name)
    progress_batch = progress_batch or AssetProgressBatch(
        {str(cache_file_path): embedded_metadata.get("size", 1)}
        if embedded_metadata is not None
        else {}
    )
    if font_file_name in EMBEDDING_FONT_METADATA and verify_file(
        cache_file_path,
        EMBEDDING_FONT_METADATA[font_file_name]["sha3_256"],
        progress_batch,
    ):
        return cache_file_path, EMBEDDING_FONT_METADATA[font_file_name]

    logger.info(f"Font {cache_file_path} not found or corrupted, downloading...")
    if fastest_upstream is None:
        fastest_upstream, font_metadata = await get_fastest_upstream_for_font(client)
        if fastest_upstream is None:
            logger.critical("Failed to get fastest upstream")
            exit(1)

        if font_file_name not in font_metadata:
            logger.critical(f"Font {font_file_name} not found in {font_metadata}")
            exit(1)

        if verify_file(
            cache_file_path,
            font_metadata[font_file_name]["sha3_256"],
            progress_batch,
        ):
            return cache_file_path, font_metadata[font_file_name]

    assert font_metadata is not None
    logger.info(f"download {font_file_name} from {fastest_upstream}")

    if "sha3_256" not in font_metadata[font_file_name]:
        logger.critical(f"Font {font_file_name} not found in {font_metadata}")
        exit(1)
    await download_file_with_fallback(
        client,
        {
            upstream: url_of(font_file_name)
            for upstream, url_of in FONT_URL_BY_UPSTREAM.items()
        },
        cache_file_path,
        font_metadata[font_file_name]["sha3_256"],
        fastest_upstream,
        font_metadata,
        progress_batch,
    )
    return cache_file_path, font_metadata[font_file_name]


def get_font_and_metadata(font_file_name: str):
    return run_coro(get_font_and_metadata_async(font_file_name))


async def get_fonts_and_metadata_async(
    font_file_names: list[str], client: httpx.AsyncClient | None = None
) -> dict[str, tuple[Path, dict]]:
    unique_names = list(dict.fromkeys(font_file_names))
    missing_names = [
        name
        for name in unique_names
        if not verify_file(
            get_cache_file_path(name, "fonts"),
            EMBEDDING_FONT_METADATA[name]["sha3_256"],
            report_progress=False,
        )
    ]
    results = {
        name: (
            get_cache_file_path(name, "fonts"),
            EMBEDDING_FONT_METADATA[name],
        )
        for name in unique_names
        if name not in missing_names
    }
    if not missing_names:
        return results

    progress_batch = AssetProgressBatch(
        {
            str(get_cache_file_path(name, "fonts")): EMBEDDING_FONT_METADATA[name][
                "size"
            ]
            for name in missing_names
        }
    )
    fastest_upstream, font_metadata = await get_fastest_upstream_for_font(client)
    if fastest_upstream is None or font_metadata is None:
        logger.critical("Failed to get fastest upstream for fonts")
        exit(1)

    downloaded = await asyncio.gather(
        *(
            get_font_and_metadata_async(
                name,
                client,
                fastest_upstream,
                font_metadata,
                progress_batch,
            )
            for name in missing_names
        )
    )
    results.update(zip(missing_names, downloaded, strict=True))
    return results


def get_fonts_and_metadata(font_file_names: list[str]):
    return run_coro(get_fonts_and_metadata_async(font_file_names))


async def get_cmap_file_path_async(
    name: str,
    client: httpx.AsyncClient | None = None,
    progress_batch: AssetProgressBatch | None = None,
) -> Path:
    """Get cached cmap file path, downloading it if necessary."""
    if name.endswith(".json"):
        file_name = name
    else:
        file_name = f"{name}.json"

    if file_name not in CMAP_METADATA:
        logger.critical(f"CMap {file_name} not found in CMAP_METADATA")
        exit(1)

    meta = CMAP_METADATA[file_name]
    cache_file_path = get_cache_file_path(file_name, "cmap")
    progress_batch = progress_batch or AssetProgressBatch(
        {str(cache_file_path): meta.get("size", 1)}
    )
    if verify_file(cache_file_path, meta["sha3_256"], progress_batch):
        return cache_file_path

    logger.info(f"CMap {cache_file_path} not found or corrupted, downloading...")
    await download_cmap_file_async(file_name, client, progress_batch)
    if not verify_file(cache_file_path, meta["sha3_256"], report_progress=False):
        logger.critical(f"Failed to verify downloaded cmap file: {cache_file_path}")
        exit(1)
    return cache_file_path


async def download_cmap_file_async(
    file_name: str,
    client: httpx.AsyncClient | None = None,
    progress_batch: AssetProgressBatch | None = None,
) -> Path:
    """Download a single cmap file to cache directory."""
    if file_name not in CMAP_METADATA:
        logger.critical(f"CMap {file_name} not found in CMAP_METADATA")
        exit(1)

    fastest_upstream, font_metadata = await get_fastest_upstream_for_font(client)
    if fastest_upstream is None:
        logger.critical("Failed to get fastest upstream for cmap")
        exit(1)

    if fastest_upstream not in CMAP_URL_BY_UPSTREAM:
        logger.critical(f"Invalid fastest upstream for cmap: {fastest_upstream}")
        exit(1)

    cache_file_path = get_cache_file_path(file_name, "cmap")
    sha3_256 = CMAP_METADATA[file_name]["sha3_256"]
    progress_batch = progress_batch or AssetProgressBatch(
        {str(cache_file_path): CMAP_METADATA[file_name].get("size", 1)}
    )
    await download_file_with_fallback(
        client,
        {
            upstream: url_of(file_name)
            for upstream, url_of in CMAP_URL_BY_UPSTREAM.items()
        },
        cache_file_path,
        sha3_256,
        fastest_upstream,
        font_metadata,
        progress_batch,
    )
    return cache_file_path


async def get_cmap_data_async(
    name: str, client: httpx.AsyncClient | None = None
) -> dict:
    """Load cmap json data from cached file, downloading it if necessary."""
    path = await get_cmap_file_path_async(name, client)
    return json.loads(path.read_text())


def get_cmap_file_path(name: str):
    return run_coro(get_cmap_file_path_async(name))


def get_cmap_data(name: str):
    return run_coro(get_cmap_data_async(name))


def get_font_family(lang_code: str):
    font_family = embedding_assets_metadata.get_font_family(lang_code)
    return font_family


async def download_all_fonts_async(
    client: httpx.AsyncClient | None = None,
    progress_batch: AssetProgressBatch | None = None,
):
    missing_fonts = [
        font_file_name
        for font_file_name, metadata in EMBEDDING_FONT_METADATA.items()
        if not verify_file(
            get_cache_file_path(font_file_name, "fonts"),
            metadata["sha3_256"],
            report_progress=False,
        )
    ]
    if not missing_fonts:
        logger.debug("All fonts are already downloaded")
        return

    progress_batch = progress_batch or AssetProgressBatch(
        {
            str(get_cache_file_path(name, "fonts")): EMBEDDING_FONT_METADATA[name][
                "size"
            ]
            for name in missing_fonts
        }
    )

    fastest_upstream, font_metadata = await get_fastest_upstream_for_font(client)
    if fastest_upstream is None:
        logger.error("Failed to get fastest upstream")
        exit(1)
    logger.info(f"Downloading fonts from {fastest_upstream}")

    font_tasks = [
        asyncio.create_task(
            get_font_and_metadata_async(
                font_file_name,
                client,
                fastest_upstream,
                font_metadata,
                progress_batch,
            )
        )
        for font_file_name in missing_fonts
    ]
    await asyncio.gather(*font_tasks)


async def download_all_cmaps_async(
    client: httpx.AsyncClient | None = None,
    progress_batch: AssetProgressBatch | None = None,
):
    """Download all cmap files defined in CMAP_METADATA."""
    missing_cmaps = [
        cmap_file_name
        for cmap_file_name, meta in CMAP_METADATA.items()
        if not verify_file(
            get_cache_file_path(cmap_file_name, "cmap"),
            meta["sha3_256"],
            report_progress=False,
        )
    ]
    if not missing_cmaps:
        logger.debug("All cmaps are already downloaded")
        return

    progress_batch = progress_batch or AssetProgressBatch(
        {
            str(get_cache_file_path(name, "cmap")): CMAP_METADATA[name]["size"]
            for name in missing_cmaps
        }
    )

    fastest_upstream, _ = await get_fastest_upstream_for_font(client)
    if fastest_upstream is None:
        logger.error("Failed to get fastest upstream for cmap")
        exit(1)
    logger.info(f"Downloading cmaps from {fastest_upstream}")

    cmap_tasks = [
        asyncio.create_task(
            get_cmap_file_path_async(cmap_file_name, client, progress_batch)
        )
        for cmap_file_name in missing_cmaps
    ]
    await asyncio.gather(*cmap_tasks)


async def download_tiktoken_caches_async(
    client: httpx.AsyncClient | None = None,
    progress_batch: AssetProgressBatch | None = None,
) -> None:
    preferred_upstream = _configured_asset_upstream() or "openai"
    for cache_name, sha3_256 in TIKTOKEN_CACHES.items():
        cache_path = get_cache_file_path(cache_name, "tiktoken")
        if verify_file(cache_path, sha3_256, progress_batch):
            continue
        await download_file_with_fallback(
            client,
            TIKTOKEN_URL_BY_UPSTREAM,
            cache_path,
            sha3_256,
            preferred_upstream,
            progress_batch=progress_batch,
            enabled_upstreams=set(TIKTOKEN_URL_BY_UPSTREAM),
        )


async def async_warmup():
    logger.info("Downloading all assets...")
    onnx_path = get_cache_file_path(
        "doclayout_yolo_docstructbench_imgsz1024.onnx", "models"
    )
    asset_specs = {
        str(onnx_path): (
            DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SIZE,
            DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
        ),
        **{
            str(get_cache_file_path(name, "tiktoken")): (size, TIKTOKEN_CACHES[name])
            for name, size in TIKTOKEN_CACHE_SIZES.items()
        },
        **{
            str(get_cache_file_path(name, "fonts")): (
                metadata["size"],
                metadata["sha3_256"],
            )
            for name, metadata in EMBEDDING_FONT_METADATA.items()
        },
        **{
            str(get_cache_file_path(name, "cmap")): (
                metadata["size"],
                metadata["sha3_256"],
            )
            for name, metadata in CMAP_METADATA.items()
        },
    }
    missing_asset_sizes = {
        asset_id: size
        for asset_id, (size, sha3_256) in asset_specs.items()
        if not verify_file(Path(asset_id), sha3_256, report_progress=False)
    }
    progress_batch = AssetProgressBatch(missing_asset_sizes)
    async with httpx.AsyncClient(timeout=_ASSET_HTTP_TIMEOUT) as client:
        await download_tiktoken_caches_async(client, progress_batch)
        from tiktoken import encoding_for_model

        _ = encoding_for_model("gpt-4o")
        onnx_task = asyncio.create_task(
            get_doclayout_onnx_model_path_async(client, progress_batch)
        )
        font_tasks = asyncio.create_task(
            download_all_fonts_async(client, progress_batch)
        )
        cmap_tasks = asyncio.create_task(
            download_all_cmaps_async(client, progress_batch)
        )
        await asyncio.gather(onnx_task, font_tasks, cmap_tasks)


def warmup():
    run_coro(async_warmup())


def generate_all_assets_file_list():
    result: dict[str, list[dict[str, str]]] = {}
    result["fonts"] = []
    result["models"] = []
    result["tiktoken"] = []
    result["cmap"] = []
    for font_file_name in EMBEDDING_FONT_METADATA:
        result["fonts"].append(
            {
                "name": font_file_name,
                "sha3_256": EMBEDDING_FONT_METADATA[font_file_name]["sha3_256"],
            }
        )
    for cmap_file_name in CMAP_METADATA:
        result["cmap"].append(
            {
                "name": cmap_file_name,
                "sha3_256": CMAP_METADATA[cmap_file_name]["sha3_256"],
            }
        )
    for tiktoken_file, sha3_256 in TIKTOKEN_CACHES.items():
        result["tiktoken"].append(
            {
                "name": tiktoken_file,
                "sha3_256": sha3_256,
            }
        )
    result["models"].append(
        {
            "name": "doclayout_yolo_docstructbench_imgsz1024.onnx",
            "sha3_256": DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256,
        },
    )
    return result


async def generate_offline_assets_package_async(output_directory: Path | None = None):
    await async_warmup()
    logger.info("Generating offline assets package...")
    file_list = generate_all_assets_file_list()
    offline_assets_tag = get_offline_assets_tag(file_list)
    if output_directory is None:
        output_path = get_cache_file_path(
            f"offline_assets_{offline_assets_tag}.zip", "assets"
        )
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / f"offline_assets_{offline_assets_tag}.zip"
    with zipfile.ZipFile(
        output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zipf:
        for file_type, file_descs in file_list.items():
            # zipf.mkdir(file_type)
            for file_desc in file_descs:
                file_name = file_desc["name"]
                sha3_256 = file_desc["sha3_256"]
                file_path = get_cache_file_path(file_name, file_type)
                if not verify_file(file_path, sha3_256):
                    logger.error(f"File {file_path} is corrupted")
                    exit(1)

                with file_path.open("rb") as f:
                    zipf.writestr(f"{file_type}/{file_name}", f.read())
    logger.info(f"Offline assets package generated at {output_path}")


async def restore_offline_assets_package_async(input_path: Path | None = None):
    file_list = generate_all_assets_file_list()
    offline_assets_tag = get_offline_assets_tag(file_list)
    if input_path is None:
        input_path = get_cache_file_path(
            f"offline_assets_{offline_assets_tag}.zip", "assets"
        )
    else:
        if input_path.exists() and input_path.is_dir():
            input_path = input_path / f"offline_assets_{offline_assets_tag}.zip"
        if not input_path.exists():
            logger.critical(f"Offline assets package not found: {input_path}")
            exit(1)

        import re

        offline_assets_tag_from_input_path = re.match(
            r"offline_assets_(.*)\.zip", input_path.name
        ).group(1)
        if offline_assets_tag != offline_assets_tag_from_input_path:
            logger.critical(
                f"Offline assets tag mismatch: {offline_assets_tag} != {offline_assets_tag_from_input_path}"
            )
            exit(1)
    nothing_changed = True
    with zipfile.ZipFile(input_path, "r") as zipf:
        for file_type, file_descs in file_list.items():
            for file_desc in file_descs:
                file_name = file_desc["name"]
                file_path = get_cache_file_path(file_name, file_type)

                if verify_file(file_path, file_desc["sha3_256"]):
                    continue
                nothing_changed = False
                with zipf.open(f"{file_type}/{file_name}", "r") as f:
                    with file_path.open("wb") as f2:
                        f2.write(f.read())
                if not verify_file(file_path, file_desc["sha3_256"]):
                    logger.critical(
                        "Offline assets package is corrupted, please delete it and try again"
                    )
                    exit(1)
    if not nothing_changed:
        logger.info(f"Offline assets package restored from {input_path}")


def get_offline_assets_tag(file_list: dict | None = None):
    if file_list is None:
        file_list = generate_all_assets_file_list()
    import orjson

    # noinspection PyTypeChecker
    offline_assets_tag = hashlib.sha3_256(
        orjson.dumps(
            file_list,
            option=orjson.OPT_APPEND_NEWLINE
            | orjson.OPT_INDENT_2
            | orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()
    return offline_assets_tag


def generate_offline_assets_package(output_directory: Path | None = None):
    return run_coro(generate_offline_assets_package_async(output_directory))


def restore_offline_assets_package(input_path: Path | None = None):
    return run_coro(restore_offline_assets_package_async(input_path))


if __name__ == "__main__":
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.DEBUG, handlers=[RichHandler()])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # warmup()
    # generate_offline_assets_package()
    # warmup()
