"""Refuses work when the engine says it is in trouble.

The gateway does not model KV. It reads what the engine publishes and decides
which classes to stop admitting. Ownership is set out in
docs/SYSTEM-DESIGN.md 4: the engine owns scheduling, the gateway owns tenancy,
and this module is the seam.

Fails open. A broken breaker must degrade admission, never deny service.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from .classification import TrafficClass
from .models import EngineSnapshot, Priority

KV_DISTRESS_THRESHOLD = 0.95
PREEMPTION_DISTRESS_THRESHOLD = 0.0

# Classes shed first when the engine is distressed. P0 is never shed; if it
# cannot be served the box is down, which is an availability problem.
SHEDDABLE_PRIORITIES = (Priority.BATCH, Priority.LONG_CONTEXT)

DEFAULT_RETRY_AFTER_SECONDS = 30


class EngineHealthSource(abc.ABC):
    @abc.abstractmethod
    def snapshot(self) -> EngineSnapshot: ...


class StaticHealthSource(EngineHealthSource):
    """For tests and dry runs."""

    def __init__(self, snapshot: EngineSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> EngineSnapshot:
        return self._snapshot


@dataclass(frozen=True)
class DistressVerdict:
    distressed: bool
    reasons: tuple[str, ...] = ()


class CircuitBreaker:
    def __init__(
        self,
        source: EngineHealthSource,
        kv_threshold: float = KV_DISTRESS_THRESHOLD,
        preemption_threshold: float = PREEMPTION_DISTRESS_THRESHOLD,
    ) -> None:
        self._source = source
        self._kv_threshold = kv_threshold
        self._preemption_threshold = preemption_threshold

    def evaluate(self) -> DistressVerdict:
        try:
            snapshot = self._source.snapshot()
        except Exception:  # noqa: BLE001 -- fail open, see module docstring
            return DistressVerdict(False, ("health source unavailable",))

        reasons = []
        if snapshot.kv_usage > self._kv_threshold:
            reasons.append(f"kv_usage {snapshot.kv_usage:.0%}")
        if snapshot.preemptions_per_minute > self._preemption_threshold:
            reasons.append(f"preemptions {snapshot.preemptions_per_minute:.1f}/min")

        return DistressVerdict(bool(reasons), tuple(reasons))

    def should_shed(self, traffic_class: TrafficClass) -> DistressVerdict:
        verdict = self.evaluate()
        if not verdict.distressed:
            return DistressVerdict(False)
        if traffic_class.priority not in SHEDDABLE_PRIORITIES:
            return DistressVerdict(False)
        return verdict


@dataclass
class ClassBudget:
    """Concurrency slots per class, as a share of a server-wide ceiling."""

    ceiling: int
    shares: dict[str, float]
    _in_flight: dict[str, int] = field(default_factory=dict)

    def limit_for(self, traffic_class: TrafficClass) -> int:
        return int(self.ceiling * self.shares.get(traffic_class.name, 0.0))

    def in_flight(self, traffic_class: TrafficClass) -> int:
        return self._in_flight.get(traffic_class.name, 0)

    def try_acquire(self, traffic_class: TrafficClass) -> bool:
        limit = self.limit_for(traffic_class)
        if limit <= 0:
            return False
        current = self.in_flight(traffic_class)
        if current >= limit:
            return False
        self._in_flight[traffic_class.name] = current + 1
        return True

    def release(self, traffic_class: TrafficClass) -> None:
        current = self.in_flight(traffic_class)
        if current:
            self._in_flight[traffic_class.name] = current - 1

    @classmethod
    def from_classes(cls, ceiling: int, classes) -> "ClassBudget":
        return cls(
            ceiling=ceiling,
            shares={c.name: c.kv_budget_share for c in classes if not c.served_off_box},
        )
