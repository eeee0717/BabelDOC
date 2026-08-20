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
    previous = assets._FASTEST_FONT_UPSTREAM
    yield
    assets._FASTEST_FONT_UPSTREAM = previous


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


def test_fallback_pins_working_upstream_for_later_downloads(tmp_path):
    """184 fonts/cmaps follow the first download. Without pinning, every one of
    them would re-run the full retry budget against the dead upstream first."""
    assets._FASTEST_FONT_UPSTREAM = "huggingface"

    _run(
        lambda client: download_file_with_fallback(
            client, URLS, tmp_path / "font.ttf", DIGEST, "huggingface"
        ),
        failing_hosts={"us.aws.cdn.hf.co"},
    )

    assert assets._FASTEST_FONT_UPSTREAM == "modelscope"


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

    assert not isinstance(excinfo.value, ValueError), "must surface the real network error"
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
