"""Value types for edge traffic observations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomerStats:
    customer: str
    requests: int
    failures: int
    distinct_ips: int
    in_bytes: int
    out_bytes: int
    p50_seconds: float | None
    p95_seconds: float | None

    @property
    def failure_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0


@dataclass(frozen=True)
class FailureCause:
    cause: str
    requests: int

    @property
    def status(self) -> int | None:
        head = self.cause.split(" ", 1)[0]
        return int(head) if head.isdigit() else None


@dataclass(frozen=True)
class PathStats:
    path: str
    requests: int
    failures: int

    @property
    def failure_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0


@dataclass(frozen=True)
class LatencyQuantiles:
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True)
class TrafficWindow:
    source: str
    coverage_seconds: float
    requests: int
    successful: int
    failures: int
    requests_per_minute: float
    status_spread: dict[str, int]
    failure_causes: tuple[FailureCause, ...]
    customers: tuple[CustomerStats, ...]
    paths: tuple[PathStats, ...]
    latency: LatencyQuantiles

    @property
    def failure_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0
