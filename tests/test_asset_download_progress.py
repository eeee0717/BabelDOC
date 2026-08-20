import asyncio
import hashlib

import babeldoc.assets.assets as assets
import httpx
from babeldoc.assets.assets import AssetProgressBatch
from babeldoc.assets.assets import download_file
from babeldoc.assets.assets import set_asset_progress_callback


def test_asset_batch_reports_byte_weighted_progress_without_resetting():
    progress = []
    previous = set_asset_progress_callback(
        lambda stage, asset_id, value: progress.append((stage, asset_id, value))
    )
    try:
        batch = AssetProgressBatch({"large": 80, "small": 20})
        batch.report_bytes("downloading_assets", "large", 40)
        batch.report_bytes("downloading_assets", "small", 10)
        batch.report_bytes("downloading_assets", "large", 20)
        batch.complete("downloading_assets", "large")
        batch.complete("downloading_assets", "small")
    finally:
        set_asset_progress_callback(previous)

    determinate = [value for _stage, _asset_id, value in progress if value is not None]
    assert determinate == [40.0, 50.0, 50.0, 90.0, 100.0]


def test_asset_batch_reports_indeterminate_retry_then_resumes_monotonically():
    progress = []
    previous = set_asset_progress_callback(
        lambda stage, asset_id, value: progress.append((stage, asset_id, value))
    )
    try:
        batch = AssetProgressBatch({"model": 100})
        batch.report_bytes("downloading_assets", "model", 60)
        batch.report_indeterminate("downloading_assets")
        batch.report_bytes("downloading_assets", "model", 10)
        batch.report_bytes("downloading_assets", "model", 80)
    finally:
        set_asset_progress_callback(previous)

    assert [value for _stage, _asset_id, value in progress] == [60.0, None, 60.0, 80.0]


def test_required_fonts_share_one_byte_weighted_batch(monkeypatch, tmp_path):
    metadata = {
        "small.ttf": {"size": 100, "sha3_256": "small"},
        "large.ttf": {"size": 300, "sha3_256": "large"},
    }
    progress = []

    monkeypatch.setattr(assets, "EMBEDDING_FONT_METADATA", metadata)
    monkeypatch.setattr(
        assets, "get_cache_file_path", lambda name, _category: tmp_path / name
    )
    monkeypatch.setattr(assets, "verify_file", lambda *_args, **_kwargs: False)

    async def fastest_upstream(_client):
        return "modelscope", metadata

    async def download_font(name, _client, _upstream, _metadata, progress_batch):
        path = tmp_path / name
        size = metadata[name]["size"]
        progress_batch.report_bytes("downloading_assets", str(path), size // 2)
        progress_batch.complete("downloading_assets", str(path))
        return path, metadata[name]

    monkeypatch.setattr(assets, "get_fastest_upstream_for_font", fastest_upstream)
    monkeypatch.setattr(assets, "get_font_and_metadata_async", download_font)
    previous = set_asset_progress_callback(
        lambda stage, asset_id, value: progress.append((stage, asset_id, value))
    )
    try:
        result = asyncio.run(
            assets.get_fonts_and_metadata_async(["small.ttf", "large.ttf"])
        )
    finally:
        set_asset_progress_callback(previous)

    determinate = [
        value
        for stage, _asset_id, value in progress
        if stage == "downloading_assets" and value is not None
    ]
    assert determinate == [12.5, 25.0, 62.5, 100.0]
    assert set(result) == set(metadata)


def test_warmup_shares_one_batch_across_model_fonts_and_cmaps(monkeypatch):
    batches = []
    batch = object()

    monkeypatch.setattr(assets, "verify_file", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(assets, "AssetProgressBatch", lambda _sizes: batch)

    async def capture_batch(_client, progress_batch):
        batches.append(progress_batch)

    monkeypatch.setattr(assets, "get_doclayout_onnx_model_path_async", capture_batch)
    monkeypatch.setattr(assets, "download_all_fonts_async", capture_batch)
    monkeypatch.setattr(assets, "download_all_cmaps_async", capture_batch)

    asyncio.run(assets.async_warmup())

    assert batches == [batch, batch, batch]


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


def test_download_file_uses_batch_size_when_response_has_no_content_length(tmp_path):
    content = b"x" * (128 * 1024)
    destination = tmp_path / "asset.bin"
    progress = []

    class UnknownLengthStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield content[: 64 * 1024]
            yield content[64 * 1024 :]

    async def run():
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=UnknownLengthStream())
        )
        batch = AssetProgressBatch({str(destination): len(content)})
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
                    batch,
                )
            finally:
                set_asset_progress_callback(previous)

    asyncio.run(run())

    determinate = [
        value
        for stage, _asset_id, value in progress
        if stage == "downloading_assets" and value is not None
    ]
    assert determinate[-1] == 100.0
    assert any(0 < value < 100 for value in determinate)


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
