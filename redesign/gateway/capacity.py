"""Background adaptive capacity control for the pooled serving gateway.

Requests consume the latest immutable control snapshot; only this controller
scrapes engine health. Hard context, token-reservation, queue-memory, and global
concurrency limits remain enforced by their existing owners.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass

from .models import EngineSnapshot


@dataclass(frozen=True)
class CapacityLimits:
    max_borrow_requests: int = 32
    poll_seconds: float = .5
    freshness_seconds: float = 2.0
    increase_step: int = 4
    green_kv: float = .55
    stop_kv: float = .70
    green_itl: float = .08
    stop_itl: float = .12
    green_ttft: float = 2.0
    stop_ttft: float = 5.0
    green_prefill: float = 2.0
    stop_prefill: float = 10.0

    def __post_init__(self):
        if self.max_borrow_requests <= 0 or self.increase_step <= 0:
            raise ValueError("Adaptive request limits must be positive")
        if not 0 < self.poll_seconds <= 5 or not self.poll_seconds < self.freshness_seconds <= 30:
            raise ValueError("Invalid adaptive sampling intervals")
        for green, stop in (
            (self.green_kv, self.stop_kv),
            (self.green_itl, self.stop_itl),
            (self.green_ttft, self.stop_ttft),
            (self.green_prefill, self.stop_prefill),
        ):
            if not math.isfinite(green) or not math.isfinite(stop) or not 0 < green < stop:
                raise ValueError("Adaptive green thresholds must be below stop thresholds")


class AdaptiveCapacityController:
    """AIMD-style controller backed by non-blocking request-path reads."""

    def __init__(self, engine, limits: CapacityLimits):
        self.engine = engine
        self.limits = limits
        self._lock = threading.Lock()
        self._snapshot: EngineSnapshot | None = None
        self._sampled_at = 0.0
        self._borrow_limit = 0
        self._state = "cold"
        self._samples = 0
        self._errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        # Prime once during service startup. Requests never perform this I/O.
        self.sample_once()
        self._thread = threading.Thread(
            target=self._run, name="k3-capacity-controller", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.limits.poll_seconds * 3))

    def _run(self) -> None:
        while not self._stop.wait(self.limits.poll_seconds):
            self.sample_once()

    @staticmethod
    def _above(value: float | None, threshold: float) -> bool:
        return value is not None and value > threshold

    def _target(self, snapshot: EngineSnapshot) -> tuple[str, int]:
        pressure = (
            snapshot.waiting > 0
            or snapshot.preemptions_per_minute > 0
            or snapshot.kv_usage >= self.limits.stop_kv
            or self._above(snapshot.mean_itl_seconds, self.limits.stop_itl)
            or self._above(snapshot.mean_ttft_seconds, self.limits.stop_ttft)
            or self._above(snapshot.mean_prefill_seconds, self.limits.stop_prefill)
        )
        if pressure:
            return "pressure", 0
        warm = (
            snapshot.kv_usage > self.limits.green_kv
            or self._above(snapshot.mean_itl_seconds, self.limits.green_itl)
            or self._above(snapshot.mean_ttft_seconds, self.limits.green_ttft)
            or self._above(snapshot.mean_prefill_seconds, self.limits.green_prefill)
        )
        if warm:
            return "warm", min(24, self.limits.max_borrow_requests)
        return "green", self.limits.max_borrow_requests

    def sample_once(self) -> None:
        try:
            snapshot = self.engine.refresh_snapshot()
            state, target = self._target(snapshot)
        except Exception:
            with self._lock:
                self._errors += 1
                self._state = "stale"
                self._borrow_limit = 0
            return
        with self._lock:
            current = self._borrow_limit
            self._borrow_limit = (
                min(target, current + self.limits.increase_step)
                if target > current else target
            )
            self._snapshot = snapshot
            self._sampled_at = time.monotonic()
            self._state = state
            self._samples += 1

    def _fresh(self) -> bool:
        return (
            self._snapshot is not None
            and time.monotonic() - self._sampled_at <= self.limits.freshness_seconds
        )

    def snapshot(self) -> EngineSnapshot:
        with self._lock:
            if not self._fresh():
                raise OSError("adaptive capacity snapshot is stale")
            return self._snapshot

    def refresh_snapshot(self) -> EngineSnapshot:
        # Reconciliation on a request path remains non-blocking. The background
        # loop owns all engine metrics I/O.
        return self.snapshot()

    def borrow_limit(self) -> int:
        with self._lock:
            return self._borrow_limit if self._fresh() else 0

    def queue_wait_seconds(self, configured: float) -> float:
        with self._lock:
            if not self._fresh() or self._state == "pressure":
                return 0.0
            if self._state == "warm":
                return min(configured, 1.0)
            return configured

    def queue_limit(self, configured: int) -> int:
        with self._lock:
            if not self._fresh() or self._state == "pressure":
                return min(configured, 4)
            if self._state == "warm":
                return min(configured, 8)
            return configured

    def state(self) -> dict:
        with self._lock:
            fresh = self._fresh()
            snapshot = self._snapshot
            return {
                "state": self._state if fresh else "stale",
                "borrow_limit": self._borrow_limit if fresh else 0,
                "snapshot_age_seconds": (
                    max(0.0, time.monotonic() - self._sampled_at)
                    if self._sampled_at else None
                ),
                "samples": self._samples,
                "errors": self._errors,
                "kv_usage": snapshot.kv_usage if snapshot is not None else None,
                "running": snapshot.running if snapshot is not None else None,
                "waiting": snapshot.waiting if snapshot is not None else None,
                "mean_itl_seconds": snapshot.mean_itl_seconds if snapshot is not None else None,
                "mean_ttft_seconds": snapshot.mean_ttft_seconds if snapshot is not None else None,
                "mean_prefill_seconds": snapshot.mean_prefill_seconds if snapshot is not None else None,
                "limits": asdict(self.limits),
            }
