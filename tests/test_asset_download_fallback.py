import asyncio
import hashlib
import json
import logging
import sys
from types import SimpleNamespace

import babeldoc.assets.assets as assets
import httpx
import pytest
from babeldoc.assets.assets import download_file_with_fallback
from tenacity import stop_after_attempt
from tenacity import wait_none

CONTENT = b"doclayout-model-bytes"
DIGEST = hashlib.sha3_256(CONTENT).hexdigest()
URLS = {
    "huggingface": "https://us.aws.cdn.hf.co/xet-bridge-us/model.onnx",
    "modelscope": "https://www.modelscope.cn/models/AI-ModelScope/model.onnx",
}


@pytest.fixture(autouse=True)
def _restore_upstream_cache():
    """The pinned upstream is module-global; keep tests independent of order."""
    previous_upstream = assets._FASTEST_FONT_UPSTREAM
    previous_metadata = assets._FASTEST_FONT_METADATA
    yield
    assets._FASTEST_FONT_UPSTREAM = previous_upstream
    assets._FASTEST_FONT_METADATA = previous_metadata


def _transport(failing_hosts):
    def handler(request):
        if request.url.host in failing_hosts:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200, headers={"content-length": str(len(CONTENT))}, content=CONTENT
        )

    return httpx.MockTransport(handler)


def _run(coro_factory, failing_hosts):
    async def run():
        async with httpx.AsyncClient(transport=_transport(failing_hosts)) as client:
            return await coro_factory(client)

    return asyncio.run(run())


def test_falls_back_to_next_upstream_when_preferred_download_fails(tmp_path):
    """The HF LFS CDN can be unreachable while huggingface.co itself resolves,
    so a dead preferred upstream must not fail the whole download."""
    destination = tmp_path / "model.onnx"

    upstream = _run(
        lambda client: download_file_with_fallback(
            client, URLS, destination, DIGEST, "huggingface"
        ),
        failing_hosts={"us.aws.cdn.hf.co"},
    )

    assert upstream == "modelscope"
    assert destination.read_bytes() == CONTENT
    assert not destination.with_name("model.onnx.part").exists()


def test_fallback_keeps_upstream_and_metadata_cache_consistent(tmp_path):
    """The cached upstream and its font metadata must change together."""
    assets._FASTEST_FONT_UPSTREAM = "huggingface"
    metadata = {"font.ttf": {"sha3_256": DIGEST}}

    _run(
        lambda client: download_file_with_fallback(
            client,
            URLS,
            tmp_path / "font.ttf",
            DIGEST,
            "huggingface",
            metadata,
        ),
        failing_hosts={"us.aws.cdn.hf.co"},
    )

    assert assets._FASTEST_FONT_UPSTREAM == "modelscope"
    assert assets._FASTEST_FONT_METADATA is metadata


def test_disabled_upstream_is_not_used_as_fallback(tmp_path):
    """hf-mirror has a URL mapping but is intentionally absent from the active
    metadata sources, so fallback must not send requests to it."""
    attempted_hosts = []
    urls = {
        "huggingface": URLS["huggingface"],
        "hf-mirror": "https://hf-mirror.com/model.onnx",
        "modelscope": URLS["modelscope"],
    }

    def handler(request):
        attempted_hosts.append(request.url.host)
        if request.url.host == "us.aws.cdn.hf.co":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200, headers={"content-length": str(len(CONTENT))}, content=CONTENT
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await download_file_with_fallback(
                client, urls, tmp_path / "model.onnx", DIGEST, "huggingface"
            )

    assert asyncio.run(run()) == "modelscope"
    assert "hf-mirror.com" not in attempted_hosts


def test_model_fallback_leaves_font_cache_usable(monkeypatch, tmp_path):
    """The model is downloaded before fonts during translation. Its fallback
    must leave both values required by the next font lookup."""
    metadata = {"font.ttf": {"sha3_256": DIGEST}}
    assets._FASTEST_FONT_UPSTREAM = None
    assets._FASTEST_FONT_METADATA = None

    async def fastest_model_upstream(_client):
        return "huggingface", metadata

    monkeypatch.setattr(
        assets, "get_fastest_upstream_for_model", fastest_model_upstream
    )
    monkeypatch.setattr(assets, "DOC_LAYOUT_ONNX_MODEL_URL", URLS)
    monkeypatch.setattr(
        assets, "DOCLAYOUT_YOLO_DOCSTRUCTBENCH_IMGSZ1024ONNX_SHA3_256", DIGEST
    )
    monkeypatch.setattr(
        assets, "get_cache_file_path", lambda name, _category: tmp_path / name
    )

    async def run():
        async with httpx.AsyncClient(
            transport=_transport({"us.aws.cdn.hf.co"})
        ) as client:
            await assets.get_doclayout_onnx_model_path_async(client)
            return await assets.get_fastest_upstream_for_font(client)

    upstream, cached_metadata = asyncio.run(run())

    assert upstream == "modelscope"
    assert cached_metadata is metadata


def test_preferred_upstream_stays_pinned_when_it_works(tmp_path):
    assets._FASTEST_FONT_UPSTREAM = "huggingface"

    upstream = _run(
        lambda client: download_file_with_fallback(
            client, URLS, tmp_path / "font.ttf", DIGEST, "huggingface"
        ),
        failing_hosts=set(),
    )

    assert upstream == "huggingface"
    assert assets._FASTEST_FONT_UPSTREAM == "huggingface"


def test_raises_when_every_upstream_fails(tmp_path):
    destination = tmp_path / "model.onnx"

    with pytest.raises(Exception) as excinfo:
        _run(
            lambda client: download_file_with_fallback(
                client, URLS, destination, DIGEST, "huggingface"
            ),
            failing_hosts={"us.aws.cdn.hf.co", "www.modelscope.cn"},
        )

    assert not isinstance(excinfo.value, ValueError), (
        "must surface the real network error"
    )
    assert not destination.exists()


def test_tries_every_upstream_when_preferred_is_unknown(tmp_path):
    """`get_fastest_upstream_for_font` can return `github`, which has no model
    URL — that must degrade to trying the others, not skip the download."""
    destination = tmp_path / "model.onnx"

    upstream = _run(
        lambda client: download_file_with_fallback(
            client, URLS, destination, DIGEST, "github"
        ),
        failing_hosts={"us.aws.cdn.hf.co"},
    )

    assert upstream == "modelscope"
    assert destination.read_bytes() == CONTENT


def test_cancellation_is_not_swallowed_by_fallback(tmp_path):
    """A user cancelling the translation must stop the download, not make it
    walk the remaining upstreams."""
    attempted = []

    def handler(request):
        attempted.append(request.url.host)
        raise asyncio.CancelledError

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await download_file_with_fallback(
                client, URLS, tmp_path / "model.onnx", DIGEST, "huggingface"
            )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert attempted == ["us.aws.cdn.hf.co"]


def test_pinned_modelscope_metadata_does_not_probe_overseas(monkeypatch):
    attempted_hosts = []
    monkeypatch.setenv("BABELDOC_ASSET_UPSTREAM", "modelscope")
    assets._FASTEST_FONT_UPSTREAM = None
    assets._FASTEST_FONT_METADATA = None

    def handler(request):
        attempted_hosts.append(request.url.host)
        return httpx.Response(200, json={"font.ttf": {"sha3_256": DIGEST}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await assets.get_fastest_upstream_for_font(client)

    upstream, metadata = asyncio.run(run())

    assert upstream == "modelscope"
    assert metadata == {"font.ttf": {"sha3_256": DIGEST}}
    assert attempted_hosts == ["www.modelscope.cn"]


def test_modelscope_download_failure_falls_back_sequentially(monkeypatch, tmp_path):
    attempted_hosts = []
    urls = {
        "huggingface": URLS["huggingface"],
        "modelscope": URLS["modelscope"],
    }

    def handler(request):
        attempted_hosts.append(request.url.host)
        if request.url.host == "www.modelscope.cn":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200, headers={"content-length": str(len(CONTENT))}, content=CONTENT
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await download_file_with_fallback(
                client, urls, tmp_path / "model.onnx", DIGEST, "modelscope"
            )

    monkeypatch.setattr(
        assets,
        "download_file",
        assets.download_file.retry_with(wait=wait_none()),
    )
    assert asyncio.run(run()) == "huggingface"
    assert attempted_hosts == [
        "www.modelscope.cn",
        "www.modelscope.cn",
        "www.modelscope.cn",
        "us.aws.cdn.hf.co",
    ]


def test_download_audit_records_source_redirect_bytes_and_timing(caplog, tmp_path):
    destination = tmp_path / "model.onnx"

    async def run():
        def handler(request):
            if request.url.path == "/redirect":
                return httpx.Response(302, headers={"location": "/asset"})
            return httpx.Response(200, content=CONTENT)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="https://mirror.test"
        ) as client:
            await assets.download_file(
                client,
                "https://mirror.test/redirect",
                destination,
                DIGEST,
                source="modelscope",
            )

    with caplog.at_level(logging.INFO, logger="babeldoc.assets.assets"):
        asyncio.run(run())

    message = next(
        record.message
        for record in caplog.records
        if record.message.startswith("Asset request audit ")
    )
    audit = json.loads(message.removeprefix("Asset request audit "))
    assert audit["bytes"] == len(CONTENT)
    assert audit["final_url"] == "https://mirror.test/asset"
    assert audit["original_url"] == "https://mirror.test/redirect"
    assert audit["result"] == "success"
    assert audit["source"] == "modelscope"
    assert audit["verification"] == "passed"
    assert audit["elapsed_ms"] >= 0


def test_checksum_failure_removes_partial_file(tmp_path):
    destination = tmp_path / "model.onnx"
    single_attempt = assets.download_file.retry_with(
        stop=stop_after_attempt(1), wait=wait_none(), reraise=True
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=CONTENT)
            )
        ) as client:
            await single_attempt(
                client,
                URLS["modelscope"],
                destination,
                "0" * 64,
                source="modelscope",
            )

    with pytest.raises(ValueError, match="corrupted"):
        asyncio.run(run())

    assert not destination.exists()
    assert not destination.with_name("model.onnx.part").exists()


def test_warmup_prefetches_tiktoken_before_loading_encoding(monkeypatch):
    events = []

    monkeypatch.setattr(assets, "verify_file", lambda *_args, **_kwargs: True)
    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        SimpleNamespace(
            encoding_for_model=lambda _model: events.append("encoding") or object()
        ),
    )

    async def tiktoken(_client, _progress_batch):
        events.append("tiktoken")

    async def no_op(_client, _progress_batch):
        return None

    monkeypatch.setattr(assets, "download_tiktoken_caches_async", tiktoken)
    monkeypatch.setattr(assets, "get_doclayout_onnx_model_path_async", no_op)
    monkeypatch.setattr(assets, "download_all_fonts_async", no_op)
    monkeypatch.setattr(assets, "download_all_cmaps_async", no_op)

    asyncio.run(assets.async_warmup())

    assert events[:2] == ["tiktoken", "encoding"]


def test_pinned_modelscope_prefetch_never_requests_openai(monkeypatch, tmp_path):
    cache_name = "tiktoken-cache"
    attempted_hosts = []
    monkeypatch.setenv("BABELDOC_ASSET_UPSTREAM", "modelscope")
    monkeypatch.setattr(assets, "TIKTOKEN_CACHES", {cache_name: DIGEST})
    monkeypatch.setattr(
        assets,
        "TIKTOKEN_URL_BY_UPSTREAM",
        {
            "openai": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
            "modelscope": "https://www.modelscope.cn/tiktoken-cache",
        },
    )
    monkeypatch.setattr(
        assets, "get_cache_file_path", lambda _name, _category: tmp_path / cache_name
    )

    def handler(request):
        attempted_hosts.append(request.url.host)
        return httpx.Response(200, content=CONTENT)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await assets.download_tiktoken_caches_async(client)

    asyncio.run(run())

    assert attempted_hosts == ["www.modelscope.cn"]
