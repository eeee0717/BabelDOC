from __future__ import annotations

import io
import json
import math
import os
import signal
import sys
import threading
from collections.abc import AsyncIterable
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path
from types import FrameType
from typing import Any
from typing import TextIO

PROTOCOL_SCHEMA = "babeldoc-stream/v1"


@contextmanager
def reserve_stdout_for_protocol(enabled: bool) -> Iterator[TextIO | None]:
    """Reserve the original stdout fd for JSON and redirect other output to stderr."""
    if not enabled:
        yield None
        return

    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation):
        protocol_stream = sys.stdout
        with redirect_stdout(sys.stderr):
            yield protocol_stream
        return

    sys.stdout.flush()
    sys.stderr.flush()
    protocol_fd = os.dup(stdout_fd)
    restore_fd = os.dup(stdout_fd)
    protocol_stream = os.fdopen(
        protocol_fd,
        "w",
        encoding=sys.stdout.encoding or "utf-8",
        errors=sys.stdout.errors or "strict",
        buffering=1,
    )
    os.dup2(stderr_fd, stdout_fd)
    try:
        yield protocol_stream
    finally:
        protocol_stream.flush()
        sys.stdout.flush()
        os.dup2(restore_fd, stdout_fd)
        os.close(restore_fd)
        protocol_stream.close()


class JsonProgressEmitter:
    """Serialize BabelDOC translation events as a stable JSON Lines protocol."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stdout
        self.terminal_emitted = False
        self.result: Any = None

    def handle(self, event: dict[str, Any]) -> str | None:
        """Emit a supported event and return its normalized protocol type."""
        if self.terminal_emitted:
            return None

        event_type = event.get("type")
        if event_type in ("progress_update", "progress_end"):
            stage = event.get("stage")
            progress = event.get("overall_progress")
            if not isinstance(stage, str) or not isinstance(progress, int | float):
                return None
            progress = float(progress)
            if not math.isfinite(progress):
                return None
            self._emit(
                {
                    "schema": PROTOCOL_SCHEMA,
                    "type": "progress",
                    "stage": stage,
                    "progress": max(0.0, min(100.0, progress)),
                }
            )
            return "progress"

        if event_type == "error":
            self.emit_error(event.get("error"))
            return "error"

        if event_type == "finish":
            self.result = event.get("translate_result")
            self._emit(
                {
                    "schema": PROTOCOL_SCHEMA,
                    "type": "finish",
                    "result": _translate_result_payload(self.result),
                }
            )
            self.terminal_emitted = True
            return "finish"

        return None

    def emit_error(self, error: Any) -> None:
        """Emit one terminal error event."""
        if self.terminal_emitted:
            return
        name, message = _error_details(error)
        self._emit(
            {
                "schema": PROTOCOL_SCHEMA,
                "type": "error",
                "name": name,
                "message": message,
            }
        )
        self.terminal_emitted = True

    def _emit(self, payload: dict[str, Any]) -> None:
        self.stream.write(
            json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        )
        self.stream.flush()


async def stream_translation_events(
    events: AsyncIterable[dict[str, Any]], emitter: JsonProgressEmitter
) -> bool:
    """Forward translation events until a terminal event is received."""
    async for event in events:
        normalized_type = emitter.handle(event)
        if normalized_type == "finish":
            return True
        if normalized_type == "error":
            return False
    raise RuntimeError("BabelDOC async_translate ended without a terminal event")


@contextmanager
def cancel_translation_on_signal(config: Any) -> Iterator[None]:
    """Request graceful cancellation on the first SIGINT or SIGTERM."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handlers: dict[signal.Signals, Any] = {}
    cancellation_requested = False

    def handle_signal(received_signal: int, frame: FrameType | None) -> None:
        nonlocal cancellation_requested
        signal_value = signal.Signals(received_signal)
        if cancellation_requested:
            previous = previous_handlers.get(signal_value)
            if callable(previous):
                previous(received_signal, frame)
                return
            raise KeyboardInterrupt
        cancellation_requested = True
        config.cancel_translation()

    for signal_value in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signal_value] = signal.getsignal(signal_value)
            signal.signal(signal_value, handle_signal)
        except (OSError, ValueError):
            continue

    try:
        yield
    finally:
        for signal_value, previous in previous_handlers.items():
            signal.signal(signal_value, previous)


def _error_details(error: Any) -> tuple[str, str]:
    if isinstance(error, type) and issubclass(error, BaseException):
        name = error.__name__
        message = "Translation cancelled" if name == "CancelledError" else name
        return name, message
    if isinstance(error, BaseException):
        return type(error).__name__, str(error) or type(error).__name__
    return "BabelDOCError", str(error) or "Unknown BabelDOC error"


def _translate_result_payload(result: Any) -> dict[str, Any]:
    return {
        "original_pdf_path": _path_or_none(getattr(result, "original_pdf_path", None)),
        "mono_pdf_path": _path_or_none(getattr(result, "mono_pdf_path", None)),
        "dual_pdf_path": _path_or_none(getattr(result, "dual_pdf_path", None)),
        "total_seconds": _finite_number_or_zero(getattr(result, "total_seconds", None)),
    }


def _path_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str | Path):
        return str(value)
    return None


def _finite_number_or_zero(value: Any) -> float:
    if not isinstance(value, int | float):
        return 0.0
    value = float(value)
    return value if math.isfinite(value) else 0.0
