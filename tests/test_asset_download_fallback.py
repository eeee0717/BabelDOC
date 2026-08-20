import asyncio
import hashlib

import babeldoc.assets.assets as assets
import httpx
import pytest
from babeldoc.assets.assets import download_file_with_fallback

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
