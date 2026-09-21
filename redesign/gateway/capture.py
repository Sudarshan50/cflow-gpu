"""Records request shapes so Z3 has a trace to size against.

The production logs did not survive the VM teardown, so the distribution every
admission number depends on has to be rebuilt from first boot. The tenancy layer
already sees every request, which makes it the cheapest place to capture it.

Records shapes, never content. No prompt text, no completion text, no headers.
A customer label is hashed unless the caller opts out, so a trace can be shared
without exposing which customer is which.
"""

from __future__ import annotations

import abc
import hashlib
import json
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from .completions import CompletionSummary
from .models import Decision

SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceRecord:
    schema: int
    timestamp: float
    customer: str
    path: str
    prompt_tokens: int
    requested_max_tokens: int | None
    granted_max_tokens: int
    clamp_reason: str
    traffic_class: str
    priority: int
    outcome: str
    rescued_from_rejection: bool
    streaming: bool
    has_tools: bool


@dataclass(frozen=True)
class CompletionRecord:
    """A terminal observation, written separately from admission shapes.

    requested_max_tokens is this gateway hop's incoming limit, possibly already
    defaulted/clamped by tenancy. granted_max_tokens is this hop's effective
    limit. Neither field reconstructs a proxy client's original limit.
    counted_prompt_tokens is admission accounting; prompt_tokens and the other
    usage counters are only what the upstream response actually reported.
    """

    schema: int
    timestamp: float
    customer: str
    path: str
    traffic_class: str
    priority: int
    counted_prompt_tokens: int
    prompt_count_source: str
    prompt_count_cached: bool
    model_context_limit: int | None
    requested_max_tokens: int | None
    granted_max_tokens: int
    default_output_tokens: int | None
    default_applied: bool
    clamp_reason: str
    prompt_tokens: int | float | None
    completion_tokens: int | float | None
    cached_tokens: int | float | None
    reasoning_tokens: int | float | None
    finish_reasons: tuple[str, ...]
    protocol_complete: bool | None
    protocol_error: bool
    elapsed_seconds: float
    first_token_seconds: float | None
    relay_outcome: str
    upstream_status: int | None
    borrowed: bool
    streaming: bool
    has_tools: bool
    has_images: bool
    modalities: tuple[str, ...]


class TraceSink(abc.ABC):
    @abc.abstractmethod
    def write(self, record: TraceRecord | CompletionRecord) -> None: ...


class JsonlSink(TraceSink):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TraceRecord | CompletionRecord) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")


class MemorySink(TraceSink):
    def __init__(self) -> None:
        self.records: list[TraceRecord | CompletionRecord] = []

    def write(self, record: TraceRecord | CompletionRecord) -> None:
        self.records.append(record)


class RotatingJsonlSink(JsonlSink):
    """Bound terminal trace retention for the single-process gateway."""

    def __init__(self, path: Path, max_bytes: int = 16 * 1024 * 1024, backups: int = 3) -> None:
        super().__init__(path)
        self._max_bytes, self._backups = max_bytes, backups
        self._lock = threading.Lock()

    def write(self, record: TraceRecord | CompletionRecord) -> None:
        body = (json.dumps(asdict(record)) + "\n").encode()
        with self._lock:
            if self._path.exists() and self._path.stat().st_size + len(body) > self._max_bytes:
                for number in range(self._backups, 0, -1):
                    source = self._path if number == 1 else Path(str(self._path) + f".{number-1}")
                    if source.exists():
                        source.replace(Path(str(self._path) + f".{number}"))
            with self._path.open("ab") as handle:
                handle.write(body)


class TraceRecorder:
    def __init__(
        self, sink: TraceSink, hash_customers: bool = True,
        completion_sink: TraceSink | None = None,
    ) -> None:
        self._sink = sink
        self._hash_customers = hash_customers
        self._completion_sink = completion_sink

    def record(self, decision: Decision, timestamp: float | None = None) -> TraceRecord:
        envelope = decision.envelope
        record = TraceRecord(
            schema=SCHEMA_VERSION,
            timestamp=timestamp if timestamp is not None else time.time(),
            customer=self._label(envelope.customer),
            path=envelope.path,
            prompt_tokens=envelope.prompt_tokens,
            requested_max_tokens=envelope.requested_max_tokens,
            granted_max_tokens=decision.clamp.granted,
            clamp_reason=decision.clamp.reason,
            traffic_class=decision.traffic_class,
            priority=int(decision.priority),
            outcome=decision.outcome.name,
            rescued_from_rejection=decision.rescued_from_rejection,
            streaming=envelope.streaming,
            has_tools=envelope.has_tools,
        )
        self._sink.write(record)
        return record

    def record_completion(
        self, decision: Decision, summary: CompletionSummary,
        elapsed_seconds: float, relay_outcome: str, upstream_status: int | None,
        *, timestamp: float | None = None,
    ) -> CompletionRecord:
        envelope, clamp = decision.envelope, decision.clamp
        has_images = bool(getattr(envelope, "has_images", False))
        record = CompletionRecord(
            schema=COMPLETION_SCHEMA_VERSION,
            timestamp=timestamp if timestamp is not None else time.time(),
            customer=self._label(envelope.customer),
            path=envelope.path,
            traffic_class=decision.traffic_class,
            priority=int(decision.priority),
            counted_prompt_tokens=envelope.prompt_tokens,
            prompt_count_source=getattr(envelope, "prompt_count_source", "heuristic"),
            prompt_count_cached=bool(getattr(envelope, "prompt_count_cached", False)),
            model_context_limit=getattr(envelope, "model_context_limit", None),
            requested_max_tokens=envelope.requested_max_tokens,
            granted_max_tokens=clamp.granted,
            default_output_tokens=getattr(clamp, "default_output_tokens", None),
            default_applied=getattr(clamp, "default_applied", envelope.requested_max_tokens is None),
            clamp_reason=clamp.reason,
            prompt_tokens=summary.prompt_tokens,
            completion_tokens=summary.completion_tokens,
            cached_tokens=summary.cached_tokens,
            reasoning_tokens=summary.reasoning_tokens,
            finish_reasons=summary.finish_reasons,
            protocol_complete=summary.protocol_complete,
            protocol_error=summary.protocol_error,
            elapsed_seconds=elapsed_seconds,
            first_token_seconds=summary.first_token_seconds,
            relay_outcome=relay_outcome,
            upstream_status=upstream_status,
            borrowed=bool((getattr(decision, "metadata", None) or {}).get("borrowed", False)),
            streaming=envelope.streaming,
            has_tools=envelope.has_tools,
            has_images=has_images,
            modalities=("text", "image") if has_images else ("text",),
        )
        if self._completion_sink is not None:
            self._completion_sink.write(record)
        return record

    def _label(self, customer: str) -> str:
        if not self._hash_customers:
            return customer
        return hashlib.sha256(customer.encode("utf-8")).hexdigest()[:12]
