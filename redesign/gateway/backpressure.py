"""Engine backpressure and atomic local admission accounting.

The breaker fails open for ordinary admission. Opt-in borrowing requires
known healthy metrics; concurrency and context-work budgets always apply.
Context work is estimated prompt + granted output, without a prefix-sharing
discount. It is deliberately not an exact physical KV reservation.
Image-prefill work reserves the full engine-rendered prompt: a conservative
upper bound on encoder embeddings, independent of the encoder compute budget.
"""

from __future__ import annotations

import abc
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from .classification import TrafficClass
from .models import AdmissionReason, CountSource, EngineSnapshot, Priority

# KV alone this high means the pool is gone whatever the queue looks like.
KV_DISTRESS_THRESHOLD = 0.97

# Earlier, and more precise: a full-ish pool *plus* a queue is the live
# signature of over-admission (KV 96%, 21 waiting, queue p90 114 s on
# opt-2026-09-20/baseline.json). Either symptom alone is survivable.
KV_PRESSURE_THRESHOLD = 0.90
QUEUE_DISTRESS_DEPTH = 8

# One preemption in a long window is not distress; a sustained rate is.
# A zero threshold sheds on the first counter tick.
PREEMPTION_DISTRESS_THRESHOLD = 1.0

# Classes shed first when the engine is distressed. P0 is never shed; if it
# cannot be served the box is down, which is an availability problem.
SHEDDABLE_PRIORITIES = (Priority.BATCH, Priority.LONG_CONTEXT)

DEFAULT_RETRY_AFTER_SECONDS = 30
ENCODER_RETRY_AFTER_SECONDS = 2

# Share for an off-box class with no target yet, so it is bounded not unlimited.
LOCAL_FALLBACK_SHARE = 0.25

DEFAULT_P0_RESERVE = 4
BORROWING_KV_THRESHOLD = 0.80
CONTEXT_KV_CAPACITY_PERCENT = 80


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
        kv_pressure_threshold: float = KV_PRESSURE_THRESHOLD,
        queue_depth: int = QUEUE_DISTRESS_DEPTH,
    ) -> None:
        self._source = source
        self._kv_threshold = kv_threshold
        self._preemption_threshold = preemption_threshold
        self._kv_pressure_threshold = kv_pressure_threshold
        self._queue_depth = queue_depth

    def snapshot(self) -> EngineSnapshot:
        """Collect once; unavailable health is explicit rather than healthy."""
        try:
            return self._source.snapshot()
        except Exception:  # noqa: BLE001 -- fail open, see module docstring
            return EngineSnapshot(0.0, 0, 0, 0.0, known=False)

    def evaluate(self, snapshot: EngineSnapshot | None = None) -> DistressVerdict:
        """An explicit snapshot avoids another scrape (including on failure)."""
        if snapshot is None:
            snapshot = self.snapshot()
        if not snapshot.known:
            return DistressVerdict(False, ("health source unknown or stale",))

        reasons = []
        if snapshot.kv_usage > self._kv_threshold:
            reasons.append(f"kv_usage {snapshot.kv_usage:.0%}")
        elif (
            snapshot.kv_usage > self._kv_pressure_threshold
            and snapshot.waiting > self._queue_depth
        ):
            reasons.append(
                f"kv_usage {snapshot.kv_usage:.0%} with {snapshot.waiting} queued"
            )
        if snapshot.preemptions_per_minute > self._preemption_threshold:
            reasons.append(f"preemptions {snapshot.preemptions_per_minute:.1f}/min")

        return DistressVerdict(bool(reasons), tuple(reasons))

    def should_shed(
        self, traffic_class: TrafficClass, snapshot: EngineSnapshot | None = None,
    ) -> DistressVerdict:
        verdict = self.evaluate(snapshot)
        if not verdict.distressed:
            return DistressVerdict(False)
        if traffic_class.priority not in SHEDDABLE_PRIORITIES:
            return DistressVerdict(False)
        return verdict


@dataclass(frozen=True)
class AdmissionLease:
    """Immutable acquisition facts; release never trusts request metadata."""

    admission_id: str
    traffic_class: str
    priority: Priority
    context_tokens: int
    borrowed: bool
    encoder_tokens: int = 0


@dataclass(frozen=True)
class BudgetSnapshot:
    """Atomic, detached gauges. Contains no request IDs or customer labels.

    Borrowed counts describe live leases acquired above their class share,
    even if an ordinary lease subsequently finishes. Capacity/limit reflect
    the last admission attempt; reading gauges never scrapes engine health.
    Encoder work counts remaining prefill reservations, independently of the
    context work and concurrency retained during streaming decode.
    """

    ceiling: int
    in_flight: int
    by_class: Mapping[str, int]
    class_limits: Mapping[str, int]
    borrowed_in_flight: int
    borrowed_by_class: Mapping[str, int]
    context_tokens_in_flight: int
    context_token_budget: int | None
    effective_context_token_budget: int | None
    kv_capacity_tokens: int | None
    borrowing_enabled: bool
    p0_reserve: int
    p0_in_flight: int
    encoder_tokens_in_flight: int = 0
    encoder_token_budget: int | None = None


@dataclass(frozen=True)
class AdmissionResult:
    lease: AdmissionLease | None
    reason: AdmissionReason | None
    detail: str
    snapshot: BudgetSnapshot

    @property
    def admitted(self) -> bool:
        return self.lease is not None


@dataclass
class ClassBudget:
    """One lock covers slots, protected P0, context work and image-prefill work.

    Defaults retain static class limits. With borrowing enabled, a class with
    a positive local share may exceed it only under healthy engine metrics.
    Non-P0 occupancy is capped at ceiling - p0_reserve, including ordinary
    admissions. P0 can consume the reserve, subject to the global/token limits.

    ``context_token_budget=None`` disables the token limit. When enabled it
    is capped at 80% of reported KV capacity. Missing capacity retains the last
    report; unknown/stale health can tighten that cap but cannot relax it.

    ``encoder_token_budget=None`` disables the encoder limit. When enabled,
    image requests require an engine-rendered prompt count, reserved in full
    until release_encoder_lease or release_lease. No image/prefix-cache sharing
    discount applies. Known rendered image costs are tracked even without a
    limit; legacy heuristic images remain admissible only with it disabled.

    New callers use try_acquire_lease/release_lease. The anonymous legacy
    try_acquire/release pair remains concurrency-only and cannot acquire when
    either token limit is enabled, since it supplies no request cost/image flag.
    """

    ceiling: int
    shares: dict[str, float]
    borrowing_enabled: bool = field(default=False, kw_only=True)
    p0_reserve: int = field(default=DEFAULT_P0_RESERVE, kw_only=True)
    context_token_budget: int | None = field(default=None, kw_only=True)
    encoder_token_budget: int | None = field(default=None, kw_only=True)
    _in_flight: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _borrowed: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _leases: dict[str, AdmissionLease] = field(default_factory=dict, init=False, repr=False)
    _legacy: dict[str, deque[str]] = field(default_factory=dict, init=False, repr=False)
    _context_tokens: int = field(default=0, init=False, repr=False)
    _encoder_tokens: int = field(default=0, init=False, repr=False)
    _encoder_leases: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _p0_in_flight: int = field(default=0, init=False, repr=False)
    _kv_capacity_tokens: int | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.p0_reserve) is not int or self.p0_reserve < 0:
            raise ValueError("p0_reserve must be a non-negative integer")
        if self.context_token_budget is not None and (
            type(self.context_token_budget) is not int or self.context_token_budget <= 0
        ):
            raise ValueError("context_token_budget must be a positive integer or None")
        if self.encoder_token_budget is not None and (
            type(self.encoder_token_budget) is not int or self.encoder_token_budget <= 0
        ):
            raise ValueError("encoder_token_budget must be a positive integer or None")

    def limit_for(self, traffic_class: TrafficClass) -> int:
        return int(self.ceiling * self.shares.get(traffic_class.name, 0.0))

    def in_flight(self, traffic_class: TrafficClass) -> int:
        with self._lock:
            return self._in_flight.get(traffic_class.name, 0)

    def try_acquire(self, traffic_class: TrafficClass) -> bool:
        """Legacy anonymous slot; no cost or healthy metrics are assumed."""
        return self._try_acquire(traffic_class, None, None, legacy=True).admitted

    def try_acquire_lease(
        self,
        traffic_class: TrafficClass,
        context_tokens: int,
        snapshot: EngineSnapshot | None = None,
        *,
        has_images: bool = False,
        prompt_tokens: int | None = None,
        prompt_count_source: CountSource | str = CountSource.HEURISTIC,
    ) -> AdmissionResult:
        """Atomically acquire all budgets for estimated prompt + granted output.

        Image encoder cost is the FULL engine-rendered prompt, not context work
        (which includes output) or a per-image estimate. Text has zero cost.
        Failed attempts consume nothing. The caller owns a successful lease
        until release_lease, which is safe to call repeatedly/concurrently.
        """
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be a non-negative integer")
        return self._try_acquire(
            traffic_class, context_tokens, snapshot, legacy=False,
            has_images=has_images, prompt_tokens=prompt_tokens,
            prompt_count_source=prompt_count_source,
        )

    def _try_acquire(
        self, traffic_class: TrafficClass, context_tokens: int | None,
        snapshot: EngineSnapshot | None, *, legacy: bool,
        has_images: bool = False, prompt_tokens: int | None = None,
        prompt_count_source: CountSource | str = CountSource.HEURISTIC,
    ) -> AdmissionResult:
        with self._lock:
            self._observe_capacity(snapshot)
            rendered_count = (
                prompt_count_source == CountSource.ENGINE_RENDERED
                and type(prompt_tokens) is int and prompt_tokens >= 0
            )
            if self.encoder_token_budget is not None and (
                legacy or (has_images and not rendered_count)
            ):
                return self._refuse(
                    AdmissionReason.ENCODER_COUNT,
                    "image-prefill budget requires an image-aware request with a full "
                    "engine_rendered non-negative integer prompt count for images",
                )
            encoder_cost = prompt_tokens if has_images and rendered_count else 0
            total = len(self._leases)
            limit = self.limit_for(traffic_class)
            current = self._in_flight.get(traffic_class.name, 0)
            if total >= self.ceiling:
                return self._refuse(
                    AdmissionReason.GLOBAL, f"global concurrency limit of {self.ceiling}",
                )
            if limit <= 0 or (current >= limit and not self.borrowing_enabled):
                return self._refuse(
                    AdmissionReason.CLASS, f"{traffic_class.name} at its concurrency limit of {limit}",
                )
            reserve = self._effective_p0_reserve()
            if (
                traffic_class.priority != Priority.INTERACTIVE
                and total - self._p0_in_flight >= self.ceiling - reserve
            ):
                return self._refuse(
                    AdmissionReason.PROTECTED_RESERVE, f"protected P0 reserve of {reserve} slots",
                )
            token_limit = self._token_limit()
            if token_limit is not None and context_tokens is None:
                return self._refuse(
                    AdmissionReason.TOKEN, "context-work budget requires a request cost",
                )
            cost = context_tokens if context_tokens is not None else 0
            if token_limit is not None and self._context_tokens + cost > token_limit:
                return self._refuse(
                    AdmissionReason.TOKEN, f"in-flight context-work budget of {token_limit} tokens "
                    f"({self._context_tokens} in use, {cost} requested)",
                )
            if (
                self.encoder_token_budget is not None
                and encoder_cost > 0
                and self._encoder_tokens + encoder_cost > self.encoder_token_budget
            ):
                return self._refuse(
                    AdmissionReason.ENCODER,
                    f"in-flight image-prefill encoder budget of {self.encoder_token_budget} tokens "
                    f"({self._encoder_tokens} in use, {encoder_cost} requested)",
                )
            borrowed = current >= limit
            if borrowed:
                pressure = self._borrowing_pressure(snapshot)
                if pressure:
                    return self._refuse(AdmissionReason.PRESSURE, "borrowing blocked: " + pressure)

            lease = AdmissionLease(
                uuid4().hex, traffic_class.name, traffic_class.priority, cost, borrowed,
                encoder_tokens=encoder_cost,
            )
            self._leases[lease.admission_id] = lease
            self._in_flight[traffic_class.name] = current + 1
            self._context_tokens += cost
            if encoder_cost:
                self._encoder_leases[lease.admission_id] = encoder_cost
                self._encoder_tokens += encoder_cost
            if lease.priority == Priority.INTERACTIVE:
                self._p0_in_flight += 1
            if borrowed:
                self._borrowed[lease.traffic_class] = (
                    self._borrowed.get(lease.traffic_class, 0) + 1
                )
            if legacy:
                self._legacy.setdefault(lease.traffic_class, deque()).append(lease.admission_id)
            return AdmissionResult(lease, None, "admitted", self._snapshot())

    def release(self, traffic_class: TrafficClass) -> None:
        """Release one legacy acquisition only; cannot consume a named lease."""
        with self._lock:
            legacy = self._legacy.get(traffic_class.name)
            if legacy:
                self._release(legacy.popleft())

    def release_lease(self, admission_id: str) -> bool:
        """Release all remaining costs, including encoder work, exactly once."""
        with self._lock:
            return self._release(admission_id)

    def release_encoder_lease(self, admission_id: str) -> bool:
        """Release only image-prefill work; retain the slot and full context cost.

        True only when a positive live encoder reservation was released. Unknown
        IDs, text, and duplicate/late callbacks (including after cancel) are no-ops.
        The caller must have observed completion of prefill before releasing.
        """
        with self._lock:
            return self._release_encoder(admission_id)

    def _release_encoder(self, admission_id: str) -> bool:
        cost = self._encoder_leases.pop(admission_id, 0)
        self._encoder_tokens -= cost
        return cost > 0

    def _release(self, admission_id: str) -> bool:
        lease = self._leases.pop(admission_id, None)
        if lease is None:
            return False
        self._release_encoder(admission_id)
        self._in_flight[lease.traffic_class] -= 1
        self._context_tokens -= lease.context_tokens
        if lease.priority == Priority.INTERACTIVE:
            self._p0_in_flight -= 1
        if lease.borrowed:
            self._borrowed[lease.traffic_class] -= 1
        return True

    @staticmethod
    def _borrowing_pressure(snapshot: EngineSnapshot | None) -> str:
        if snapshot is None or not snapshot.known:
            return "engine health unknown or stale"
        if any(not math.isfinite(value) or value < 0 for value in (
            snapshot.kv_usage, snapshot.running, snapshot.waiting,
            snapshot.preemptions_per_minute,
        )):
            return "engine health invalid"
        if snapshot.kv_usage >= BORROWING_KV_THRESHOLD:
            return f"kv_usage {snapshot.kv_usage:.0%} >= {BORROWING_KV_THRESHOLD:.0%}"
        if snapshot.waiting > 0:
            return f"{snapshot.waiting} queued"
        if snapshot.preemptions_per_minute > 0:
            return f"preemptions {snapshot.preemptions_per_minute:g}/min"
        return ""

    def _observe_capacity(self, snapshot: EngineSnapshot | None) -> None:
        if snapshot is None:
            return
        capacity = snapshot.kv_capacity_tokens
        if type(capacity) is int and capacity >= 0:
            if snapshot.known or self._kv_capacity_tokens is None:
                self._kv_capacity_tokens = capacity
            else:
                self._kv_capacity_tokens = min(self._kv_capacity_tokens, capacity)

    def _token_limit(self) -> int | None:
        if self.context_token_budget is None or self._kv_capacity_tokens is None:
            return self.context_token_budget
        return min(
            self.context_token_budget,
            self._kv_capacity_tokens * CONTEXT_KV_CAPACITY_PERCENT // 100,
        )

    def _effective_p0_reserve(self) -> int:
        return min(self.ceiling, self.p0_reserve) if self.borrowing_enabled else 0

    def _refuse(self, reason: AdmissionReason, detail: str) -> AdmissionResult:
        return AdmissionResult(None, reason, detail, self._snapshot())

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> BudgetSnapshot:
        names = self.shares.keys() | self._in_flight.keys()
        return BudgetSnapshot(
            ceiling=self.ceiling,
            in_flight=len(self._leases),
            by_class=MappingProxyType({name: self._in_flight.get(name, 0) for name in names}),
            class_limits=MappingProxyType({
                name: int(self.ceiling * self.shares.get(name, 0.0)) for name in names
            }),
            borrowed_in_flight=sum(self._borrowed.values()),
            borrowed_by_class=MappingProxyType({
                name: self._borrowed.get(name, 0) for name in names
            }),
            context_tokens_in_flight=self._context_tokens,
            context_token_budget=self.context_token_budget,
            effective_context_token_budget=self._token_limit(),
            kv_capacity_tokens=self._kv_capacity_tokens,
            borrowing_enabled=self.borrowing_enabled,
            p0_reserve=self._effective_p0_reserve(),
            p0_in_flight=self._p0_in_flight,
            encoder_tokens_in_flight=self._encoder_tokens,
            encoder_token_budget=self.encoder_token_budget,
        )

    @classmethod
    def from_classes(
        cls, ceiling: int, classes, offbox_configured: bool = False, *,
        borrowing_enabled: bool = False,
        p0_reserve: int = DEFAULT_P0_RESERVE,
        context_token_budget: int | None = None,
        encoder_token_budget: int | None = None,
    ) -> "ClassBudget":
        """Shares for every class that this engine actually serves.

        A class routed off-box takes no local slot; one merely *marked* for
        off-box routing, with no target configured, still does. Its share is
        borrowed from the interactive class it most resembles.
        """
        shares = {}
        for traffic_class in classes:
            if traffic_class.served_off_box and offbox_configured:
                continue
            share = traffic_class.kv_budget_share
            if traffic_class.served_off_box:
                share = LOCAL_FALLBACK_SHARE
            shares[traffic_class.name] = share
        return cls(
            ceiling=ceiling, shares=shares, borrowing_enabled=borrowing_enabled,
            p0_reserve=p0_reserve, context_token_budget=context_token_budget,
            encoder_token_budget=encoder_token_budget,
        )
