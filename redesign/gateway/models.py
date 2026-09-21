"""Value types crossing the gateway boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


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


class CountSource(str, Enum):
    """Only ENGINE_RENDERED counts the full prompt including expanded images."""

    HEURISTIC = "heuristic"
    ENGINE_RENDERED = "engine_rendered"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class AdmissionReason(str, Enum):
    """Stable rejection codes, also usable as JSON metadata and metric labels."""

    GLOBAL = "global"
    CLASS = "class"
    TOKEN = "token"
    PRESSURE = "pressure"
    PROTECTED_RESERVE = "protected_reserve"
    ENCODER = "encoder"
    ENCODER_COUNT = "encoder_count"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RequestEnvelope:
    """What the gateway knows before the engine sees the request.

    requested_max_tokens is this hop's input, possibly already defaulted or
    clamped upstream; it is not necessarily the original caller's limit.
    """

    customer: str
    prompt_tokens: int
    requested_max_tokens: int | None = None
    streaming: bool = False
    has_tools: bool = False
    has_images: bool = False
    batch_hint: bool = False
    path: str = "/v1/chat/completions"
    tools_disabled: bool = False
    # Strings from existing estimators remain accepted. Only engine_rendered
    # can bound encoder work; a cached engine-rendered count is still exact.
    prompt_count_source: CountSource | str = CountSource.HEURISTIC
    prompt_count_cached: bool = False
    model_context_limit: int | None = None


@dataclass(frozen=True)
class EngineSnapshot:
    """Engine-published health, not a physical KV reservation model.

    Sources must set ``known=False`` for incomplete, unavailable or stale
    health. Capacity is an optional engine-reported token count, not an
    estimate derived from request counts or cache-prefix sharing.
    """

    kv_usage: float
    running: int
    waiting: int
    preemptions_per_minute: float
    known: bool = True
    kv_capacity_tokens: int | None = None

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
    default_output_tokens: int | None = None

    @property
    def default_applied(self) -> bool:
        """This hop supplied the budget, even if a ceiling then reduced it."""
        return self.requested is None

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
    admission_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

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
