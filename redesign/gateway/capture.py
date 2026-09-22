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
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import Decision

SCHEMA_VERSION = 1


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
    policy_reason: str = ""
    admission_reason: str = ""
    admission_wait_seconds: float = 0.0


class TraceSink(abc.ABC):
    @abc.abstractmethod
    def write(self, record: TraceRecord) -> None: ...


class JsonlSink(TraceSink):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TraceRecord) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")


class MemorySink(TraceSink):
    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def write(self, record: TraceRecord) -> None:
        self.records.append(record)


class TraceRecorder:
    def __init__(self, sink: TraceSink, hash_customers: bool = True) -> None:
        self._sink = sink
        self._hash_customers = hash_customers

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
            policy_reason=decision.reason,
            admission_reason=decision.admission_reason,
            admission_wait_seconds=decision.admission_wait_seconds,
        )
        self._sink.write(record)
        return record

    def _label(self, customer: str) -> str:
        if not self._hash_customers:
            return customer
        return hashlib.sha256(customer.encode("utf-8")).hexdigest()[:12]
