import asyncio
import io
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from babeldoc.progress_json import JsonProgressEmitter
from babeldoc.progress_json import JsonProgressEmitterV2
from babeldoc.progress_json import cancel_translation_on_signal
from babeldoc.progress_json import stream_translation_events


def emitted(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_emits_clamped_progress_and_finish_result():
    stream = io.StringIO()
    emitter = JsonProgressEmitter(stream)
    result = SimpleNamespace(
        original_pdf_path="input.pdf",
        mono_pdf_path=Path("mono.pdf"),
        dual_pdf_path=None,
        total_seconds=12.5,
    )

    emitter.handle(
        {
            "type": "progress_update",
            "stage": "Translate Paragraphs",
            "overall_progress": 105,
        }
    )
    emitter.handle({"type": "finish", "translate_result": result})

    assert emitted(stream) == [
        {
            "schema": "babeldoc-stream/v1",
            "type": "progress",
            "stage": "Translate Paragraphs",
            "progress": 100.0,
        },
        {
            "schema": "babeldoc-stream/v1",
            "type": "finish",
            "result": {
                "original_pdf_path": "input.pdf",
                "mono_pdf_path": "mono.pdf",
                "dual_pdf_path": None,
                "total_seconds": 12.5,
            },
        },
    ]


def test_ignores_invalid_progress_and_events_after_terminal_error():
    stream = io.StringIO()
    emitter = JsonProgressEmitter(stream)

    emitter.handle(
        {
            "type": "progress_update",
            "stage": "Parse PDF",
            "overall_progress": float("nan"),
        }
    )
    emitter.handle({"type": "error", "error": ValueError("bad input")})
    emitter.handle({"type": "error", "error": RuntimeError("duplicate")})
    emitter.handle(
        {"type": "progress_end", "stage": "Parse PDF", "overall_progress": 100}
    )

    assert emitted(stream) == [
        {
            "schema": "babeldoc-stream/v1",
            "type": "error",
            "name": "ValueError",
            "message": "bad input",
        }
    ]


def test_cancelled_error_class_gets_stable_message():
    stream = io.StringIO()
    emitter = JsonProgressEmitter(stream)

    emitter.handle({"type": "error", "error": asyncio.CancelledError})

    assert emitted(stream)[0] == {
        "schema": "babeldoc-stream/v1",
        "type": "error",
        "name": "CancelledError",
        "message": "Translation cancelled",
    }


def test_v2_emits_initialization_and_structured_translation_progress():
    stream = io.StringIO()
    emitter = JsonProgressEmitterV2(stream)

    emitter.emit_progress("checking_assets", None, 0)
    emitter.handle_asset_progress("checking_assets", "layout-model", 50)
    emitter.handle_asset_progress("downloading_assets", "layout-model", 50)
    emitter.emit_progress("loading_model", None, 8)
    emitter.handle(
        {
            "type": "progress_start",
            "stage": "Parse PDF and Create Intermediate Representation",
            "stage_progress": 0,
            "overall_progress": 0,
        }
    )
    emitter.handle(
        {
            "type": "progress_update",
            "stage": "Translate Paragraphs",
            "stage_progress": 25,
            "overall_progress": 50,
        }
    )

    assert emitted(stream) == [
        {
            "schema": "babeldoc-stream/v2",
            "type": "progress",
            "stage": "checking_assets",
            "stage_progress": None,
            "overall_progress": 0.0,
        },
        {
            "schema": "babeldoc-stream/v2",
            "type": "progress",
            "stage": "checking_assets",
            "stage_progress": 50.0,
            "overall_progress": 0.5,
        },
        {
            "schema": "babeldoc-stream/v2",
            "type": "progress",
            "stage": "downloading_assets",
            "stage_progress": 50.0,
            "overall_progress": 4.5,
        },
        {
            "schema": "babeldoc-stream/v2",
            "type": "progress",
            "stage": "loading_model",
            "stage_progress": None,
            "overall_progress": 8.0,
        },
        {
            "schema": "babeldoc-stream/v2",
            "type": "progress",
            "stage": "parsing",
            "stage_progress": 0.0,
            "overall_progress": 10.0,
        },
        {
            "schema": "babeldoc-stream/v2",
            "type": "progress",
            "stage": "translating",
            "stage_progress": 25.0,
            "overall_progress": 55.0,
        },
    ]


def test_v2_keeps_overall_progress_monotonic_and_resets_per_asset():
    stream = io.StringIO()
    emitter = JsonProgressEmitterV2(stream)

    emitter.handle_asset_progress("checking_assets", "font-a", 80)
    emitter.handle_asset_progress("checking_assets", "font-a", 40)
    emitter.handle_asset_progress("checking_assets", "font-b", 10)
    emitter.handle(
        {
            "type": "progress_update",
            "stage": "Parse Page Layout",
            "stage_progress": 20,
            "overall_progress": 10,
        }
    )
    emitter.handle_asset_progress("downloading_assets", "font-c", None)

    events = emitted(stream)
    assert [event["overall_progress"] for event in events] == sorted(
        event["overall_progress"] for event in events
    )
    assert [event["stage_progress"] for event in events[:2]] == [80.0, 10.0]
    assert events[-1] == {
        "schema": "babeldoc-stream/v2",
        "type": "progress",
        "stage": "downloading_assets",
        "stage_progress": None,
        "overall_progress": 19.0,
    }


def test_v2_terminal_events_use_v2_schema():
    stream = io.StringIO()
    emitter = JsonProgressEmitterV2(stream)

    emitter.handle({"type": "error", "error": ValueError("bad input")})

    assert emitted(stream) == [
        {
            "schema": "babeldoc-stream/v2",
            "type": "error",
            "name": "ValueError",
            "message": "bad input",
        }
    ]


def test_stream_translation_events_requires_terminal_event():
    async def events():
        yield {"type": "progress_start", "stage": "Parse PDF"}

    with pytest.raises(RuntimeError, match="without a terminal event"):
        asyncio.run(
            stream_translation_events(events(), JsonProgressEmitter(io.StringIO()))
        )


def test_stream_translation_events_returns_false_for_error():
    async def events():
        yield {"type": "error", "error": "failed"}

    assert not asyncio.run(
        stream_translation_events(events(), JsonProgressEmitter(io.StringIO()))
    )


def test_signal_handler_requests_cancellation_and_restores_handlers(monkeypatch):
    installed = {}
    restored = {}
    config = SimpleNamespace(
        cancel_translation=lambda: installed.setdefault("cancelled", True)
    )

    monkeypatch.setattr(signal, "getsignal", lambda value: f"previous-{value.name}")

    def set_signal(value, handler):
        if callable(handler):
            installed[value] = handler
        else:
            restored[value] = handler

    monkeypatch.setattr(signal, "signal", set_signal)

    with cancel_translation_on_signal(config):
        installed[signal.SIGINT](signal.SIGINT, None)

    assert installed["cancelled"] is True
    assert restored == {
        signal.SIGINT: "previous-SIGINT",
        signal.SIGTERM: "previous-SIGTERM",
    }
