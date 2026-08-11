import asyncio
import hashlib

import httpx
from babeldoc.assets.assets import download_file
from babeldoc.assets.assets import set_asset_progress_callback


def test_download_file_streams_progress_and_replaces_destination(tmp_path):
    content = b"progressive-download"
    destination = tmp_path / "asset.bin"
    progress = []

    async def run():
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-length": str(len(content))},
                content=content,
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            previous = set_asset_progress_callback(
                lambda stage, asset_id, value: progress.append((stage, asset_id, value))
            )
            try:
                await download_file(
                    client,
                    "https://example.test/asset.bin",
                    destination,
                    hashlib.sha3_256(content).hexdigest(),
                )
            finally:
                set_asset_progress_callback(previous)

    asyncio.run(run())

    assert destination.read_bytes() == content
    assert not destination.with_name("asset.bin.part").exists()
    assert progress[-1] == ("downloading_assets", str(destination), 100.0)
    assert any(stage == "checking_assets" for stage, _asset_id, _value in progress)


def test_download_file_reports_indeterminate_without_content_length(tmp_path):
    content = b"unknown-length"
    destination = tmp_path / "asset.bin"
    progress = []

    class UnknownLengthStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield content

    async def run():
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=UnknownLengthStream())
        )
        async with httpx.AsyncClient(transport=transport) as client:
            previous = set_asset_progress_callback(
                lambda stage, asset_id, value: progress.append((stage, asset_id, value))
            )
            try:
                await download_file(
                    client,
                    "https://example.test/asset.bin",
                    destination,
                    hashlib.sha3_256(content).hexdigest(),
                )
            finally:
                set_asset_progress_callback(previous)

    asyncio.run(run())

    assert destination.read_bytes() == content
    assert progress[0] == ("downloading_assets", str(destination), None)
