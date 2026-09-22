"""Value types crossing the gateway boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    """Engine scheduling priority. Lower runs earlier."""

    INTERACTIVE = 0
    SHORT_CHAT = 1
    LONG_CONTEXT = 2
    BATCH = 3


class Outcome(IntEnum):
    ADMIT = 0
    REJECT_BUDGET = 1
    REJECT_SHED = 2


@dataclass(frozen=True)
class RequestEnvelope:
    """What the gateway knows before the engine sees the request."""

    customer: str
    prompt_tokens: int
    requested_max_tokens: int | None = None
    streaming: bool = False
    has_tools: bool = False
    has_images: bool = False
    batch_hint: bool = False
    path: str = "/v1/chat/completions"


@dataclass(frozen=True)
class EngineSnapshot:
    """Engine-published health. The gateway reads this; it never models KV."""

    kv_usage: float
    running: int
    waiting: int
    preemptions_per_minute: float
    kv_capacity_tokens: int | None = None
    cache_epoch: float | None = None
    sampled_at: float | None = None
    mean_itl_seconds: float | None = None
    mean_ttft_seconds: float | None = None
    mean_prefill_seconds: float | None = None

    @classmethod
    def healthy(cls) -> "EngineSnapshot":
        return cls(kv_usage=0.0, running=0, waiting=0, preemptions_per_minute=0.0)


@dataclass(frozen=True)
class ClampResult:
    granted: int
    requested: int | None
    reason: str
    exceeded_window: bool = False
    """True when prompt + requested would not have fit the engine's window.

    Independent of which clamp path applied: a class ceiling and a window
    ceiling both rescue such a request, so keying on the reason undercounts.
    """

    @property
    def clamped(self) -> bool:
        return self.requested is not None and self.granted < self.requested


@dataclass(frozen=True)
class Decision:
    envelope: RequestEnvelope
    traffic_class: str
    priority: Priority
    clamp: ClampResult
    outcome: Outcome
    reason: str
    retry_after_seconds: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    workload_lease: int | None = None
    admission_wait_seconds: float = 0.0
    admission_queued: bool = False
    admission_reason: str = ""

    @property
    def admitted(self) -> bool:
        return self.outcome is Outcome.ADMIT

    @property
    def rescued_from_rejection(self) -> bool:
        """Would an unguarded engine have returned 400 for this request?

        The metric behind register item D2. It counts only what the clamp
        actually recovers, which is a subset of the observed 400s -- the rest
        have another cause the surviving logs do not identify.
        """
        return self.admitted and self.clamp.exceeded_window
