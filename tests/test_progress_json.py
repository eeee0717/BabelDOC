import asyncio
import io
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from babeldoc.progress_json import JsonProgressEmitter
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
