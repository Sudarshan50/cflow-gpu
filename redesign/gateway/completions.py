"""Bounded, content-free observations of OpenAI-compatible completions.

Only the current SSE event (or the non-streaming JSON body) is buffered. A size
limit disables further observation, not the caller's relay. Parsed documents,
error messages, IDs and generated text are never kept; ``finish`` also discards
any unfinished raw input. The summary describes protocol completion, not answer
quality: exhausting an output budget with ``length`` is a valid terminal reason.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass

DEFAULT_MAX_BYTES = 1024 * 1024
FINISH_REASONS = frozenset({
    "stop", "length", "tool_calls", "content_filter", "function_call",
})
_LINE_END = re.compile(br"\r\n?|\n")


@dataclass(frozen=True)
class CompletionSummary:
    first_token_seconds: float | None = None
    prompt_tokens: int | float | None = None
    completion_tokens: int | float | None = None
    cached_tokens: int | float | None = None
    reasoning_tokens: int | float | None = None
    finish_reasons: tuple[str, ...] = ()
    protocol_complete: bool | None = None
    protocol_error: bool = False


def _has_token(delta: dict) -> bool:
    for name in ("content", "reasoning", "reasoning_content"):
        value = delta.get(name)
        if isinstance(value, str) and value:
            return True
    calls = delta.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and any(
                isinstance(function.get(key), str) and function[key]
                for key in ("name", "arguments")
            ):
                return True
    return False


class CompletionObserver:
    """Observe bytes without consuming, changing, or retaining the relay body.

    ``started`` uses ``time.monotonic``. TTFT is the arrival of a complete SSE
    event containing a nonempty text/reasoning or tool-function delta, never a
    role, usage record, or non-streaming response.

    ``protocol_complete`` is unknown (None) if no bytes arrived or the buffer
    limit was exceeded. Otherwise a stream needs a dispatched [DONE] event and
    no unfinished event; JSON needs nonempty choices with terminal reasons.
    Malformed data and upstream error events/documents set ``protocol_error``
    and prevent a True result. Missing [DONE] alone is incompleteness, not an
    explicit protocol error. Counters remain None until actually reported.
    """

    def __init__(
        self, stream: bool, started: float, max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self._stream = stream
        self._started = started
        self._max_bytes = max_bytes
        self._buffer = bytearray()
        self._line_bytes = 0
        self._skip_lf = False
        self._saw_bytes = False
        self._dropped = False
        self._done = False
        self._protocol_error = False
        self._first_token_seconds: float | None = None
        self._counts: dict[str, int | float | None] = dict.fromkeys((
            "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
        ))
        self._finish_reasons: set[str] = set()
        self._summary: CompletionSummary | None = None

    @property
    def buffered_bytes(self) -> int:
        """Raw response bytes currently retained, always at most max_bytes."""
        return len(self._buffer)

    def feed(self, chunk: bytes) -> None:
        if self._summary is not None or self._dropped or not chunk:
            return
        self._saw_bytes = True
        if not self._stream:
            self._append(chunk, 0, len(chunk))
            return

        # Work one line at a time even when a relay read contains many events.
        # Normalizing CR/CRLF here handles delimiters split across feed calls;
        # decoding only whole events handles split UTF-8 code points too.
        start = int(self._skip_lf and chunk.startswith(b"\n"))
        self._skip_lf = False
        for ending in _LINE_END.finditer(chunk, start):
            end = ending.start()
            if not self._append(chunk, start, end):
                return
            self._line_bytes += end - start
            if not self._append(b"\n", 0, 1):
                return
            if self._line_bytes == 0:
                self._event()
                self._buffer.clear()
            self._line_bytes = 0
            start = ending.end()
            self._skip_lf = chunk[ending.end() - 1] == 13
        if self._append(chunk, start, len(chunk)):
            self._line_bytes += len(chunk) - start
            if start < len(chunk):
                self._skip_lf = False

    def _append(self, chunk: bytes, start: int, end: int) -> bool:
        # Check before slicing/copying: a single enormous relay chunk must not
        # briefly become an enormous observation buffer.
        if len(self._buffer) + end - start > self._max_bytes:
            self._dropped = True
            self._buffer.clear()
            self._line_bytes = 0
            self._skip_lf = False
            return False
        self._buffer.extend(memoryview(chunk)[start:end])
        return True

    def _event(self) -> None:
        try:
            text = self._buffer.decode("utf-8-sig")
        except UnicodeError:
            self._protocol_error = True
            return
        data = []
        error_event = False
        for line in text.split("\n"):
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                error_event = value == "error"
        if error_event:
            self._protocol_error = True
            return
        if not data:
            return
        payload = "\n".join(data)
        if payload.strip() == "[DONE]":
            self._done = True
        elif self._done:
            self._protocol_error = True
        else:
            self._parse(payload)

    def _parse(self, raw: str | bytearray) -> None:
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            # Never keep an exception: JSONDecodeError includes the raw body.
            self._protocol_error = True
            return
        if not isinstance(document, dict):
            self._protocol_error = True
            return
        if "error" in document or document.get("type") == "error":
            self._protocol_error = True
            return
        self._usage(document.get("usage"))
        choices = document.get("choices")
        if self._stream and choices is None and isinstance(document.get("usage"), dict):
            return
        if not isinstance(choices, list) or (not self._stream and not choices):
            self._protocol_error = True
            return
        for choice in choices:
            if not isinstance(choice, dict):
                self._protocol_error = True
                continue
            reason = choice.get("finish_reason")
            if isinstance(reason, str) and reason in FINISH_REASONS:
                self._finish_reasons.add(reason)
            elif reason is not None or not self._stream:
                self._protocol_error = True
            if self._stream:
                delta = choice.get("delta")
                if delta is not None and not isinstance(delta, dict):
                    self._protocol_error = True
                elif (
                    isinstance(delta, dict) and self._first_token_seconds is None
                    and _has_token(delta)
                ):
                    elapsed = time.monotonic() - self._started
                    if math.isfinite(elapsed):
                        self._first_token_seconds = max(0.0, elapsed)

    def _usage(self, usage: object) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            self._protocol_error = True
            return
        for name in ("prompt_tokens", "completion_tokens"):
            self._counter(name, usage.get(name))
        for name, field in (
            ("cached_tokens", "prompt_tokens_details"),
            ("reasoning_tokens", "completion_tokens_details"),
        ):
            details = usage.get(field)
            if details is None:
                continue
            if isinstance(details, dict):
                self._counter(name, details.get(name))
            else:
                self._protocol_error = True

    def _counter(self, name: str, value: object) -> None:
        if value is None:
            return
        if (
            type(value) is int and value >= 0
            or type(value) is float and math.isfinite(value) and value >= 0
        ):
            # Usage snapshots are cumulative; repeated/interim records must
            # replace counters rather than double-counting them.
            self._counts[name] = value
        else:
            self._protocol_error = True

    def finish(self) -> CompletionSummary:
        if self._summary is not None:
            return self._summary
        try:
            complete = None
            if self._saw_bytes and not self._dropped:
                if self._stream:
                    complete = self._done and bool(self._finish_reasons) and not self._buffer and not self._protocol_error
                else:
                    self._parse(self._buffer)
                    complete = not self._protocol_error
            self._summary = CompletionSummary(
                first_token_seconds=self._first_token_seconds,
                **self._counts,
                finish_reasons=tuple(sorted(self._finish_reasons)),
                protocol_complete=complete,
                protocol_error=self._protocol_error,
            )
            return self._summary
        finally:
            # This also covers an unfinished SSE event or truncated JSON body.
            self._buffer.clear()
            self._line_bytes = 0
            self._skip_lf = False
