"""Traffic sources. Add a format by implementing TrafficSource."""

from __future__ import annotations

import json
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
