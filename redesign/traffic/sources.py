"""Traffic sources. Add a format by implementing TrafficSource."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from .records import (
    CustomerStats,
    FailureCause,
    LatencyQuantiles,
    PathStats,
    TrafficWindow,
)


class TrafficSource(Protocol):
    def load(self) -> TrafficWindow: ...


class ProdStatsSource:
    """Reads an eval-harness prod_stats.json produced by eval/harness."""

    def __init__(self, path: Path, window: str = "1h") -> None:
        self._path = path
        self._window = window

    def load(self) -> TrafficWindow:
        payload = json.loads(self._path.read_text())
        window = payload["windows"][self._window]

        return TrafficWindow(
            source=str(self._path),
            coverage_seconds=window.get("actual_coverage_seconds", 0.0),
            requests=window["requests"],
            successful=window["successful"],
            failures=window["failures"],
            requests_per_minute=window["requests_per_minute"],
            status_spread=dict(window["status_spread"]),
            failure_causes=tuple(
                FailureCause(cause=c["cause"], requests=c["requests"])
                for c in window["failure_causes"]
            ),
            customers=tuple(
                CustomerStats(
                    customer=c["customer"],
                    requests=c["requests"],
                    failures=c["failures"],
                    distinct_ips=c["distinct_ips"],
                    in_bytes=c["in_bytes"],
                    out_bytes=c["out_bytes"],
                    p50_seconds=c.get("p50_s"),
                    p95_seconds=c.get("p95_s"),
                )
                for c in window["per_customer"]
            ),
            paths=tuple(
                PathStats(path=p["path"], requests=p["requests"], failures=p["failures"])
                for p in window["per_path"]
            ),
            latency=LatencyQuantiles(
                p50=window["edge_latency_s"].get("p50"),
                p90=window["edge_latency_s"].get("p90"),
                p95=window["edge_latency_s"].get("p95"),
                p99=window["edge_latency_s"].get("p99"),
            ),
        )


def parse_duration(raw: str) -> float:
    token = raw.strip().lower()
    if token.isdigit():
        return float(token)
    unit = token[-1]
    if unit not in {"s", "m", "h", "d"} or not token[:-1]:
        raise ValueError(f"unsupported window: {raw}")
    return float(token[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


class GatewayTraceSource:
    """Reads the Z3/Z4 gateway capture (shapes only, no prompt text)."""

    def __init__(self, path: Path, window: str = "1h", now: float | None = None) -> None:
        self._path = path
        self._window = window
        self._now = now

    def load(self) -> TrafficWindow:
        horizon = parse_duration(self._window)
        now = self._now if self._now is not None else time.time()
        cutoff = now - horizon
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if float(row.get("timestamp", 0.0)) >= cutoff:
                records.append(row)

        successful = sum(
            1 for r in records if str(r.get("outcome", "")).lower() == "admit"
        )
        failures = len(records) - successful
        status_spread: dict[str, int] = defaultdict(int)
        causes: dict[str, int] = defaultdict(int)
        customers: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "failures": 0}
        )
        paths: dict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "failures": 0}
        )
        for row in records:
            outcome = str(row.get("outcome", "unknown"))
            status_spread[outcome] += 1
            failed = outcome.lower() != "admit"
            if failed:
                causes[outcome] += 1
            customer = str(row.get("customer") or "unknown")
            customers[customer]["requests"] += 1
            customers[customer]["failures"] += int(failed)
            path = str(row.get("path") or "/")
            paths[path]["requests"] += 1
            paths[path]["failures"] += int(failed)

        coverage = 0.0
        if records:
            stamps = [float(r.get("timestamp", 0.0)) for r in records]
            coverage = max(stamps) - min(stamps)

        return TrafficWindow(
            source=str(self._path),
            coverage_seconds=coverage,
            requests=len(records),
            successful=successful,
            failures=failures,
            requests_per_minute=(len(records) / coverage * 60.0) if coverage else 0.0,
            status_spread=dict(status_spread),
            failure_causes=tuple(
                FailureCause(cause=name, requests=count)
                for name, count in sorted(causes.items())
            ),
            customers=tuple(
                CustomerStats(
                    customer=name,
                    requests=stats["requests"],
                    failures=stats["failures"],
                    distinct_ips=0,
                    in_bytes=0,
                    out_bytes=0,
                    p50_seconds=None,
                    p95_seconds=None,
                )
                for name, stats in sorted(customers.items())
            ),
            paths=tuple(
                PathStats(
                    path=name,
                    requests=stats["requests"],
                    failures=stats["failures"],
                )
                for name, stats in sorted(paths.items())
            ),
            latency=LatencyQuantiles(None, None, None, None),
        )
