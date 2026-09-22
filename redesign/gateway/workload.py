"""Admission reservations for context memory and long-running generation."""
from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass

from .models import EngineSnapshot


@dataclass(frozen=True)
class WorkloadLimits:
    large_context_tokens: int = 65_536
    large_context_requests: int = 4
    long_output_tokens: int = 2_048
    # Deprecated compatibility inputs. Long-output admission is token weighted;
    # these request-count values are intentionally not enforced.
    long_output_requests: int = 2
    reserved_tokens: int = 1_048_576
    projected_kv_limit: float = .85
    long_output_burst_requests: int | None = None
    long_output_burst_kv_limit: float = .55
    long_output_burst_running_limit: int = 8
    long_output_burst_prompt_limit: int = 32_768
    long_output_base_token_budget: int = 8_192
    long_output_burst_token_budget: int = 32_768
    engine_queue_tolerance: int = 0
    adaptive_borrow_requests: int = 0
    adaptive_borrow_kv_limit: float = .70
    adaptive_borrow_itl_limit: float = .12
    output_history_size: int = 128
    output_min_samples: int = 8
    output_quantile: float = .95
    output_safety_factor: float = 1.25
    throughput_first: bool = False

    def __post_init__(self):
        if min(self.large_context_tokens, self.large_context_requests, self.long_output_tokens,
               self.long_output_requests, self.reserved_tokens) <= 0:
            raise ValueError("Workload limits must be positive")
        if not 0 < self.projected_kv_limit < 1:
            raise ValueError("Projected KV limit must be between zero and one")
        if not 0 < self.long_output_burst_kv_limit < 1:
            raise ValueError("Burst KV limit must be between zero and one")
        if self.long_output_burst_kv_limit > self.projected_kv_limit:
            raise ValueError("Burst KV limit cannot exceed the heavy-work limit")
        if min(
            self.long_output_burst_running_limit,
            self.long_output_burst_prompt_limit,
            self.long_output_base_token_budget,
            self.long_output_burst_token_budget,
        ) <= 0 or self.engine_queue_tolerance < 0:
            raise ValueError("Invalid burst running limit or engine queue tolerance")
        if self.long_output_burst_token_budget < self.long_output_base_token_budget:
            raise ValueError("Long-output burst token budget cannot be below base")
        if self.adaptive_borrow_requests < 0:
            raise ValueError("Adaptive borrow request limit cannot be negative")
        if self.throughput_first and self.adaptive_borrow_requests <= 0:
            raise ValueError("Throughput-first admission requires adaptive capacity")
        if not 0 < self.adaptive_borrow_kv_limit < self.projected_kv_limit:
            raise ValueError("Adaptive borrow KV limit must be below the heavy-work limit")
        if not math.isfinite(self.adaptive_borrow_itl_limit) or self.adaptive_borrow_itl_limit <= 0:
            raise ValueError("Adaptive borrow ITL limit must be positive")
        if self.output_history_size <= 0 or not 1 <= self.output_min_samples <= self.output_history_size:
            raise ValueError("Invalid output prediction history")
        if not 0 < self.output_quantile <= 1:
            raise ValueError("Output prediction quantile must be between zero and one")
        if not math.isfinite(self.output_safety_factor) or self.output_safety_factor < 1:
            raise ValueError("Output prediction safety factor must be at least one")


class WorkloadBudget:
    """Input plus conservative predicted-output reservations.

    The full output grant is still validated against the context window. The
    aggregate KV commitment uses observed completion lengths because vLLM
    allocates generated KV on demand. Leases make release idempotent.
    """
    def __init__(self, limits: WorkloadLimits, health_source):
        self.limits = limits
        self.health_source = health_source
        self._lock = threading.Lock()
        self._next_id = 0
        self._leases: dict[int, tuple[int, bool, bool, int, str, int]] = {}
        self._tokens = self._large = self._long = self._long_tokens = 0
        self._output_history: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=limits.output_history_size)
        )
        self._burst_admissions = 0
        self._last_long_token_budget = limits.long_output_base_token_budget
        self._last_output_prediction = limits.long_output_tokens

    def requirements(self, prompt_tokens: int, output_tokens: int, has_images: bool = False):
        return (prompt_tokens + output_tokens,
                prompt_tokens >= self.limits.large_context_tokens or has_images,
                 output_tokens >= self.limits.long_output_tokens)

    def reservation_limit(self, snapshot=None) -> int:
        """Use measured KV capacity in throughput mode, retaining a fallback."""
        capacity = getattr(snapshot, "kv_capacity_tokens", None)
        if self.limits.throughput_first and capacity:
            return max(1, int(capacity * self.limits.projected_kv_limit))
        return self.limits.reserved_tokens

    def can_fit_alone(self, prompt_tokens: int, output_tokens: int,
                      has_images: bool, class_name: str) -> bool:
        """Do not let a permanently impossible reservation age-block the queue."""
        try:
            snapshot = self.health_source.snapshot()
        except Exception:
            return True  # a bounded waiter can wait for health to recover
        charge, _, long = self.requirements(prompt_tokens, output_tokens, has_images)
        if long and not self.limits.throughput_first:
            with self._lock:
                charge = prompt_tokens + self._predicted_output(class_name, output_tokens)
        return charge <= self.reservation_limit(snapshot)

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

    def _long_token_budget(
        self, snapshot, projected: float, large: bool, prompt_tokens: int
    ) -> int:
        adaptive_burst = getattr(self.health_source, "allows_burst", None)
        controller_allows = not callable(adaptive_burst) or adaptive_burst()
        # The larger token budget is for ordinary text requests, not additional
        # large or conservative/multimodal reservations. Unknown health never
        # enables it. Every admitted request pays its predicted output cost;
        # there is deliberately no fixed long-request count cap.
        if (self.limits.long_output_burst_token_budget
                > self.limits.long_output_base_token_budget
                and controller_allows and not large and snapshot is not None
                and prompt_tokens <= self.limits.long_output_burst_prompt_limit
                and snapshot.kv_capacity_tokens
                and math.isfinite(projected) and 0 <= projected <= self.limits.long_output_burst_kv_limit
                and snapshot.waiting == 0 and snapshot.preemptions_per_minute == 0
                and max(snapshot.running, len(self._leases)) < self.limits.long_output_burst_running_limit):
            return self.limits.long_output_burst_token_budget
        return self.limits.long_output_base_token_budget

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

    def _predicted_output(self, class_name: str, granted: int) -> int:
        history = self._output_history[class_name]
        estimate = self.limits.long_output_tokens
        if len(history) >= self.limits.output_min_samples:
            ordered = sorted(history)
            index = max(0, math.ceil(len(ordered) * self.limits.output_quantile) - 1)
            estimate = ordered[index]
        prediction = max(
            1, math.ceil(estimate * self.limits.output_safety_factor)
        )
        return min(granted, prediction)

    def acquire(
        self, prompt_tokens: int, output_tokens: int, has_images: bool = False,
        class_name: str = "default",
    ):
        if prompt_tokens < 0 or output_tokens <= 0:
            raise ValueError("Invalid workload token reservation")
        charge, large, long = self.requirements(prompt_tokens, output_tokens, has_images)
        snapshot = self._snapshot(large or long)
        if self.limits.throughput_first and snapshot is None:
            return None, "engine_health_unavailable"
        with self._lock:
            long_tokens = self._predicted_output(class_name, output_tokens) if long else 0
            if long and not self.limits.throughput_first:
                # Context validity still uses the full grant in TokenClamp. KV
                # commitment is based on a conservative completion predictor
                # because vLLM allocates generated tokens on demand.
                charge = prompt_tokens + long_tokens
            capacity = getattr(snapshot, "kv_capacity_tokens", None)
            projected = snapshot.kv_usage if snapshot is not None else 0.0
            if capacity:
                projected = max(projected, self._tokens / capacity) + charge / capacity
            long_token_budget = self.limits.long_output_base_token_budget
            if long:
                long_token_budget = self._long_token_budget(
                    snapshot, projected, large, prompt_tokens
                )
                # A configured budget is an aggregate concurrency bound, not
                # a way to make one otherwise-valid completion impossible.
                if self._long == 0:
                    long_token_budget = max(long_token_budget, long_tokens)
                self._last_long_token_budget = long_token_budget
                self._last_output_prediction = long_tokens
            if self.limits.throughput_first:
                # The shared pool reserves full input + granted output for KV
                # safety. Predictions remain telemetry, not a second execution
                # cap or permission to overcommit future physical KV memory.
                self._last_long_token_budget = 0
            if not self.limits.throughput_first and large and self._large >= self.limits.large_context_requests:
                return None, "large_context_slots"
            if not self.limits.throughput_first and long and self._long_tokens + long_tokens > long_token_budget:
                return None, "long_output_budget"
            if self._tokens + charge > self.reservation_limit(snapshot):
                return None, "reserved_tokens"
            if snapshot is not None:
                if not self.limits.throughput_first and (large or long) and snapshot.waiting > self._queue_tolerance(snapshot, projected):
                    return None, "engine_queue"
                if (large or long) and snapshot.running + snapshot.waiting > len(self._leases):
                    return None, "untracked_engine_work"
                kv_limit = self.limits.projected_kv_limit if (self.limits.throughput_first or large or long) else .97
                if projected > kv_limit:
                    return None, "kv_headroom"
            self._next_id += 1
            lease = self._next_id
            self._leases[lease] = (
                charge, large, long, long_tokens, class_name, output_tokens
            )
            if (
                long
                and self._long_tokens + long_tokens
                > self.limits.long_output_base_token_budget
            ):
                self._burst_admissions += 1
            self._tokens += charge
            self._large += int(large)
            self._long += int(long)
            self._long_tokens += long_tokens
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

    def release(self, lease: int, actual_output_tokens: int | None = None) -> bool:
        with self._lock:
            item = self._leases.pop(lease, None)
            if item is None:
                return False
            charge, large, long, long_tokens, class_name, granted = item
            self._tokens -= charge
            self._large -= int(large)
            self._long -= int(long)
            self._long_tokens -= long_tokens
            if (
                long
                and type(actual_output_tokens) is int
                and 0 < actual_output_tokens <= granted
            ):
                self._output_history[class_name].append(actual_output_tokens)
            return True

    def state(self) -> dict[str, int | float]:
        with self._lock:
            samples = sum(len(history) for history in self._output_history.values())
            return {"active_requests": len(self._leases), "reserved_tokens": self._tokens,
                    "large_context_requests": self._large, "long_output_requests": self._long,
                    "long_output_reserved_tokens": self._long_tokens,
                    "long_output_slot_limit_enabled": 0,
                    "long_output_budget_enabled": int(not self.limits.throughput_first),
                    "large_context_slot_limit_enabled": int(not self.limits.throughput_first),
                    "throughput_first": int(self.limits.throughput_first),
                    "long_output_last_token_budget": self._last_long_token_budget,
                    "long_output_last_prediction": self._last_output_prediction,
                    "output_prediction_samples": samples,
                    "burst_admissions_total": self._burst_admissions}
