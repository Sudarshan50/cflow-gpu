"""Shared, pressure-driven execution capacity with a separate burst queue.

The controller adjusts starts, never queued deadlines or running work. It uses
the existing background-only health scraper; request threads read its snapshot.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .capacity import AdaptiveCapacityController, CapacityLimits


@dataclass(frozen=True)
class ThroughputLimits(CapacityLimits):
    max_borrow_requests: int = 64
    initial_requests: int = 8
    minimum_requests: int = 8
    green_itl: float = .04
    stop_itl: float = .08
    green_kv: float = .75
    stop_kv: float = .88
    critical_kv: float = .97
    pressure_samples: int = 3
    recovery_samples: int = 4
    adjustment_seconds: float = 2.0
    decrease_factor: float = .8
    tolerated_waiters: int = 2

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.minimum_requests <= self.initial_requests <= self.max_borrow_requests:
            raise ValueError("Minimum and initial capacity must fit the execution ceiling")
        if not self.stop_kv < self.critical_kv < 1:
            raise ValueError("Critical KV threshold must be above the pressure threshold")
        if min(self.pressure_samples, self.recovery_samples) < 2:
            raise ValueError("Pressure and recovery require multiple observations")
        if not math.isfinite(self.adjustment_seconds) or self.adjustment_seconds < self.poll_seconds:
            raise ValueError("Execution adjustment interval must cover a sampling interval")
        if not 0 < self.decrease_factor < 1 or self.tolerated_waiters < 0:
            raise ValueError("Invalid throughput decrease factor or queue tolerance")


class ThroughputCapacityController(AdaptiveCapacityController):
    """AIMD execution pool shared by all local traffic classes.

    An isolated slow sample freezes growth. Sustained congestion reduces the
    pool toward observed occupancy; sustained healthy demand increases it.
    Memory distress, preemption, and stale/missing measurements pause starts.
    The bounded admission queue remains available throughout recovery.
    """

    throughput_first = True

    def __init__(self, engine, limits: ThroughputLimits):
        super().__init__(engine, limits)
        self._borrow_limit = limits.initial_requests
        self._pressure_streak = self._healthy_streak = 0
        self._recovery_streak = 0
        self._last_adjusted = 0.0
        self._queued_demand = 0
        self._paused = True
        self._reasons: tuple[str, ...] = ()

    def set_queued_demand(self, count: int) -> None:
        with self._lock:
            self._queued_demand = max(0, count)

    def sample_once(self) -> None:
        try:
            snapshot = self.engine.refresh_snapshot()
            if (not snapshot.kv_capacity_tokens or snapshot.kv_capacity_tokens <= 0
                    or not math.isfinite(snapshot.kv_usage) or not 0 <= snapshot.kv_usage <= 1
                    or min(snapshot.running, snapshot.waiting, snapshot.preemptions_per_minute) < 0
                    or not math.isfinite(snapshot.preemptions_per_minute)
                    or any(value is not None and (not math.isfinite(value) or value < 0)
                           for value in (snapshot.mean_itl_seconds, snapshot.mean_ttft_seconds,
                                         snapshot.mean_prefill_seconds))):
                raise ValueError("Incomplete engine capacity observation")
        except Exception:
            with self._changed:
                self._errors += 1
                self._state = "stale"
                self._paused = True
                self._pressure_streak = self._healthy_streak = 0
                self._recovery_streak = 0
                self._reasons = ("engine_health_unavailable",)
                self._generation += 1
                self._changed.notify_all()
            return

        now = time.monotonic()
        active = snapshot.running + snapshot.waiting
        reasons = []
        if snapshot.kv_usage >= self.limits.stop_kv:
            reasons.append("kv_pressure")
        if snapshot.waiting > self.limits.tolerated_waiters:
            reasons.append("engine_queue")
        if active and self._above(snapshot.mean_itl_seconds, self.limits.stop_itl):
            reasons.append("decode_latency")
        latency_load = (snapshot.waiting > 0
                        or self._above(snapshot.mean_itl_seconds, self.limits.green_itl)
                        or active >= self.limits.initial_requests)
        # A single large cold prompt has intrinsic prefill/TTFT cost. Treat it
        # as a warning unless queueing, decode interference, or a busy pool also
        # establishes load. Otherwise one finished job can collapse idle capacity.
        for name, value, threshold in (
            ("first_token_latency", snapshot.mean_ttft_seconds, self.limits.stop_ttft),
            ("prefill_latency", snapshot.mean_prefill_seconds, self.limits.stop_prefill),
        ):
            if active and latency_load and self._above(value, threshold):
                reasons.append(name)
        critical = snapshot.kv_usage >= self.limits.critical_kv or snapshot.preemptions_per_minute > 0
        if snapshot.preemptions_per_minute > 0:
            reasons.append("preemption")
        green = (
            not reasons and snapshot.waiting == 0 and snapshot.kv_usage <= self.limits.green_kv
            and (not active or all(not self._above(value, threshold) for value, threshold in (
                (snapshot.mean_itl_seconds, self.limits.green_itl),
                (snapshot.mean_ttft_seconds, self.limits.green_ttft),
                (snapshot.mean_prefill_seconds, self.limits.green_prefill),
            )))
        )
        with self._changed:
            self._snapshot = snapshot
            self._sampled_at = now
            self._samples += 1
            self._pressure_streak = self._pressure_streak + 1 if reasons else 0
            self._healthy_streak = self._healthy_streak + 1 if green else 0
            self._recovery_streak = self._recovery_streak + 1 if not reasons else 0
            self._reasons = tuple(reasons)
            adjustment_due = now - self._last_adjusted >= self.limits.adjustment_seconds
            if active == 0 and green and self._queued_demand == 0:
                # Idle is not demand. Return to the latency-safe baseline;
                # queued work may grow the pool again after healthy samples.
                self._borrow_limit = self.limits.initial_requests
                self._paused = False
            if critical:
                self._state = "pressure"
                self._paused = True
                self._healthy_streak = 0
            elif self._pressure_streak >= self.limits.pressure_samples:
                self._state = "pressure"
                # A backed-up engine drains before more starts are released.
                self._paused = snapshot.waiting > self.limits.tolerated_waiters
                if adjustment_due:
                    # Occupancy is an observation, not the machine's capacity.
                    # Clamping the limit to a momentarily drained active count
                    # turned transient latency into a one-request replica and
                    # stranded a large external queue. Noncritical pressure
                    # decreases capacity gradually to a qualified floor;
                    # critical KV/preemption or an engine queue still pauses
                    # all new starts above.
                    self._borrow_limit = max(
                        self.limits.minimum_requests,
                        math.floor(
                            self._borrow_limit * self.limits.decrease_factor
                        ),
                    )
                    self._last_adjusted = now
            elif self._recovery_streak >= self.limits.recovery_samples:
                self._state = "green" if green else "warm"
                self._paused = False
                if (
                    self._healthy_streak >= self.limits.recovery_samples
                    and adjustment_due
                ):
                    # Recovering the configured baseline is not speculative
                    # growth and must not depend on a gateway queue. Requiring
                    # queued demand here can strand steady traffic at the
                    # pressure floor forever. Only growth above the baseline
                    # requires an observed queue.
                    if (
                        self._borrow_limit < self.limits.initial_requests
                        or self._queued_demand > 0
                    ):
                        self._borrow_limit = min(
                            self.limits.max_borrow_requests,
                            self.limits.initial_requests
                            if self._borrow_limit < self.limits.initial_requests
                            else self._borrow_limit + self.limits.increase_step,
                        )
                        self._last_adjusted = now
                elif (
                    self._queued_demand == 0
                    and self._borrow_limit > self.limits.initial_requests
                    and adjustment_due
                ):
                    # Do not preserve speculative headroom after a burst.
                    # Tracking current occupancy blocks replacements while
                    # allowing already-running requests to finish naturally.
                    self._borrow_limit = max(
                        self.limits.initial_requests, active
                    )
                    self._last_adjusted = now
            elif self._state == "cold" and not critical:
                # Startup capacity is bounded; a single bad startup observation
                # cannot open it until a healthy recovery has been observed.
                self._state = "warm"
                self._paused = not green
            else:
                # Freeze growth during transient/warning pressure. Preserve a
                # preceding pause until recovery, including after a scrape error.
                self._state = "warm" if not self._paused else self._state
            self._generation += 1
            self._changed.notify_all()

    def execution_limit(self) -> int:
        with self._lock:
            return self._borrow_limit if self._fresh() and not self._paused else 0

    def borrow_limit(self) -> int:
        return self.execution_limit()

    def allows_burst(self) -> bool:
        return self.execution_limit() > 0

    def queue_wait_seconds(self, configured: float) -> float:
        return configured

    def queue_limit(self, configured: int) -> int:
        return configured

    def state(self) -> dict:
        result = super().state()
        with self._lock:
            result.update(
                throughput_first=True,
                borrow_limit=self._borrow_limit if self._fresh() and not self._paused else 0,
                execution_limit=self._borrow_limit if self._fresh() and not self._paused else 0,
                target_execution_limit=self._borrow_limit,
                starts_paused=self._paused or not self._fresh(),
                queued_demand=self._queued_demand,
                pressure_streak=self._pressure_streak,
                healthy_streak=self._healthy_streak,
                recovery_streak=self._recovery_streak,
                pressure_reasons=self._reasons,
            )
        return result
