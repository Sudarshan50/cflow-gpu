"""Admission reservations for context memory and long-running generation."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from .models import EngineSnapshot


@dataclass(frozen=True)
class WorkloadLimits:
    large_context_tokens: int = 65_536
    large_context_requests: int = 4
    long_output_tokens: int = 2_048
    long_output_requests: int = 2
    reserved_tokens: int = 1_048_576
    projected_kv_limit: float = .85
    long_output_burst_requests: int | None = None
    long_output_burst_kv_limit: float = .55
    long_output_burst_running_limit: int = 8
    long_output_burst_prompt_limit: int = 32_768
    engine_queue_tolerance: int = 0
    adaptive_borrow_requests: int = 0
    adaptive_borrow_kv_limit: float = .70
    adaptive_borrow_itl_limit: float = .12

    def __post_init__(self):
        if min(self.large_context_tokens, self.large_context_requests, self.long_output_tokens,
               self.long_output_requests, self.reserved_tokens) <= 0:
            raise ValueError("Workload limits must be positive")
        if not 0 < self.projected_kv_limit < 1:
            raise ValueError("Projected KV limit must be between zero and one")
        if self.long_output_burst_requests is not None and self.long_output_burst_requests < self.long_output_requests:
            raise ValueError("Long-output burst limit cannot be below the base limit")
        if not 0 < self.long_output_burst_kv_limit < 1:
            raise ValueError("Burst KV limit must be between zero and one")
        if (self.long_output_burst_requests is not None
                and self.long_output_burst_requests > self.long_output_requests
                and self.long_output_burst_kv_limit > self.projected_kv_limit):
            raise ValueError("Enabled burst KV limit cannot exceed the heavy-work limit")
        if min(self.long_output_burst_running_limit, self.long_output_burst_prompt_limit) <= 0 or self.engine_queue_tolerance < 0:
            raise ValueError("Invalid burst running limit or engine queue tolerance")
        if self.adaptive_borrow_requests < 0:
            raise ValueError("Adaptive borrow request limit cannot be negative")
        if not 0 < self.adaptive_borrow_kv_limit < self.projected_kv_limit:
            raise ValueError("Adaptive borrow KV limit must be below the heavy-work limit")
        if not math.isfinite(self.adaptive_borrow_itl_limit) or self.adaptive_borrow_itl_limit <= 0:
            raise ValueError("Adaptive borrow ITL limit must be positive")


class WorkloadBudget:
    """Full input+granted-output reservations, never speculative cache discounts.

    Leases also make release idempotent: an error path cannot release another
    request's capacity. Engine occupancy covers work not owned by this process.
    """
    def __init__(self, limits: WorkloadLimits, health_source):
        self.limits = limits
        self.health_source = health_source
        self._lock = threading.Lock()
        self._next_id = 0
        self._leases: dict[int, tuple[int, bool, bool]] = {}
        self._tokens = self._large = self._long = 0
        self._burst_admissions = 0
        self._last_long_limit = limits.long_output_requests

    def requirements(self, prompt_tokens: int, output_tokens: int, has_images: bool = False):
        return (prompt_tokens + output_tokens,
                prompt_tokens >= self.limits.large_context_tokens or has_images,
                output_tokens >= self.limits.long_output_tokens)

    def _snapshot(self, heavy: bool):
        try:
            snapshot = self.health_source.snapshot()
        except Exception:
            return None
        with self._lock:
            owned = len(self._leases)
        # A just-completed request can still be present in the cached engine
        # gauge. Refresh before calling it unowned; the HTTP client's refresh
        # is single-flight and rate-bounded. Persistent discrepancies still
        # refuse heavy admission and can wait in the bounded admission queue.
        if heavy and snapshot.running + snapshot.waiting > owned:
            refresh = getattr(self.health_source, "refresh_snapshot", None)
            if refresh is not None:
                try:
                    snapshot = refresh()
                except Exception:
                    pass  # retain the last reading, not an optimistic empty engine
        return snapshot

    def _long_limit(self, snapshot, projected: float, large: bool, prompt_tokens: int) -> int:
        base = self.limits.long_output_requests
        burst = self.limits.long_output_burst_requests or base
        # Extra slots are for ordinary text requests, not additional large or
        # conservative/multimodal reservations. Unknown health never enables
        # bursts. Every admitted request still pays its full token reservation.
        if (burst > base and not large and snapshot is not None
                and prompt_tokens <= self.limits.long_output_burst_prompt_limit
                and snapshot.kv_capacity_tokens
                and math.isfinite(projected) and 0 <= projected <= self.limits.long_output_burst_kv_limit
                and snapshot.waiting == 0 and snapshot.preemptions_per_minute == 0
                and max(snapshot.running, len(self._leases)) < self.limits.long_output_burst_running_limit):
            return burst
        return base

    def _queue_tolerance(self, snapshot, projected: float) -> int:
        # Tolerating a transient waiter is a low-pressure optimization, not
        # permission to feed an already backed-up engine. Under pressure use
        # the original empty-engine-queue rule and wait outside the engine.
        if (not math.isfinite(projected) or projected > self.limits.long_output_burst_kv_limit
                or snapshot.preemptions_per_minute > 0
                or max(snapshot.running + snapshot.waiting, len(self._leases))
                >= self.limits.long_output_burst_running_limit):
            return 0
        return self.limits.engine_queue_tolerance

    def acquire(self, prompt_tokens: int, output_tokens: int, has_images: bool = False):
        if prompt_tokens < 0 or output_tokens <= 0:
            raise ValueError("Invalid workload token reservation")
        charge, large, long = self.requirements(prompt_tokens, output_tokens, has_images)
        snapshot = self._snapshot(large or long)
        with self._lock:
            capacity = getattr(snapshot, "kv_capacity_tokens", None)
            projected = snapshot.kv_usage if snapshot is not None else 0.0
            if capacity:
                projected = max(projected, self._tokens / capacity) + charge / capacity
            long_limit = self._long_limit(snapshot, projected, large, prompt_tokens)
            if long:
                self._last_long_limit = long_limit
            if large and self._large >= self.limits.large_context_requests:
                return None, "large_context_slots"
            if long and self._long >= long_limit:
                return None, "long_output_slots"
            if self._tokens + charge > self.limits.reserved_tokens:
                return None, "reserved_tokens"
            if snapshot is not None:
                if (large or long) and snapshot.waiting > self._queue_tolerance(snapshot, projected):
                    return None, "engine_queue"
                if (large or long) and snapshot.running + snapshot.waiting > len(self._leases):
                    return None, "untracked_engine_work"
                kv_limit = self.limits.projected_kv_limit if (large or long) else .97
                if projected > kv_limit:
                    return None, "kv_headroom"
            self._next_id += 1
            lease = self._next_id
            self._leases[lease] = (charge, large, long)
            if long and self._long >= self.limits.long_output_requests:
                self._burst_admissions += 1
            self._tokens += charge
            self._large += int(large)
            self._long += int(long)
            return lease, "admitted"

    def borrow_limit(
        self, class_name: str, prompt_tokens: int, output_tokens: int,
        has_images: bool = False,
    ) -> int:
        """Allow pooled interactive work to use idle class capacity.

        Borrowing is deliberately stricter than ordinary admission. KV
        headroom is necessary but not sufficient for K3: decode can saturate
        while the cache is mostly empty, so an engine queue, preemption, or
        unhealthy recent inter-token latency disables borrowing immediately.
        """
        if (
            self.limits.adaptive_borrow_requests <= 0
            or class_name not in ("P1-short-chat", "P2-agentic")
        ):
            return 0
        dynamic_limit = getattr(self.health_source, "borrow_limit", None)
        request_limit = (
            min(self.limits.adaptive_borrow_requests, dynamic_limit())
            if callable(dynamic_limit) else self.limits.adaptive_borrow_requests
        )
        if request_limit <= 0:
            return 0
        snapshot = self._snapshot(False)
        if (
            snapshot is None
            or not snapshot.kv_capacity_tokens
            or snapshot.waiting > 0
            or snapshot.preemptions_per_minute > 0
        ):
            return 0
        mean_itl = snapshot.mean_itl_seconds
        if mean_itl is not None and mean_itl > self.limits.adaptive_borrow_itl_limit:
            return 0
        charge, _, _ = self.requirements(prompt_tokens, output_tokens, has_images)
        with self._lock:
            projected = max(snapshot.kv_usage, self._tokens / snapshot.kv_capacity_tokens)
            projected += charge / snapshot.kv_capacity_tokens
        return request_limit if projected <= self.limits.adaptive_borrow_kv_limit else 0

    def can_borrow(
        self, class_name: str, prompt_tokens: int, output_tokens: int,
        has_images: bool = False,
    ) -> bool:
        """Compatibility predicate for callers that do not need the limit."""
        return self.borrow_limit(class_name, prompt_tokens, output_tokens, has_images) > 0

    def release(self, lease: int) -> bool:
        with self._lock:
            item = self._leases.pop(lease, None)
            if item is None:
                return False
            charge, large, long = item
            self._tokens -= charge
            self._large -= int(large)
            self._long -= int(long)
            return True

    def state(self) -> dict[str, int]:
        with self._lock:
            return {"active_requests": len(self._leases), "reserved_tokens": self._tokens,
                    "large_context_requests": self._large, "long_output_requests": self._long,
                    "long_output_last_admission_limit": self._last_long_limit,
                    "burst_admissions_total": self._burst_admissions}
