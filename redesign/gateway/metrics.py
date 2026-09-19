"""Per-class SLO instrumentation. Register item E2.

A global p95 is meaningless across a 1-second class and a 60-second class, so
every latency observation is bucketed by traffic class. Without this, session G2
has no pass/fail: its criterion is that P0 TTFT p95 does not regress, which is
not computable from an aggregate.

Exposes Prometheus text. Stdlib only; no client library.
"""

from __future__ import annotations

import threading
from bisect import insort
from dataclasses import dataclass, field

# Keeping every observation would grow without bound; keeping too few makes p95
# noisy. 2048 per class is a few hundred kB and stable at this request rate.
MAX_SAMPLES_PER_CLASS = 2048

QUANTILES = (0.5, 0.95, 0.99)


@dataclass
class LatencySeries:
    samples: list[float] = field(default_factory=list)

    def observe(self, seconds: float) -> None:
        if len(self.samples) >= MAX_SAMPLES_PER_CLASS:
            self.samples.pop(0)
        insort(self.samples, seconds)

    def quantile(self, q: float) -> float | None:
        if not self.samples:
            return None
        index = min(len(self.samples) - 1, int(q * len(self.samples)))
        return self.samples[index]


class Registry:
    """Thread-safe counters and per-class latency series."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._ttft: dict[str, LatencySeries] = {}
        self._total: dict[str, LatencySeries] = {}

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe_ttft(self, traffic_class: str, seconds: float) -> None:
        with self._lock:
            self._ttft.setdefault(traffic_class, LatencySeries()).observe(seconds)

    def observe_total(self, traffic_class: str, seconds: float) -> None:
        with self._lock:
            self._total.setdefault(traffic_class, LatencySeries()).observe(seconds)

    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            ttft = {k: list(v.samples) for k, v in self._ttft.items()}
            total = {k: list(v.samples) for k, v in self._total.items()}

        lines: list[str] = []
        for (name, labels), value in sorted(counters.items()):
            rendered = ",".join(f'{k}="{v}"' for k, v in labels)
            suffix = f"{{{rendered}}}" if rendered else ""
            lines.append(f"k3_gateway_{name}{suffix} {value:g}")

        lines += _render_quantiles("ttft_seconds", ttft)
        lines += _render_quantiles("request_seconds", total)
        return "\n".join(lines) + "\n"


def _render_quantiles(metric: str, series: dict[str, list[float]]) -> list[str]:
    lines = []
    for traffic_class, samples in sorted(series.items()):
        if not samples:
            continue
        for q in QUANTILES:
            index = min(len(samples) - 1, int(q * len(samples)))
            lines.append(
                f'k3_gateway_{metric}{{class="{traffic_class}",quantile="{q}"}} '
                f"{samples[index]:g}"
            )
        lines.append(
            f'k3_gateway_{metric}_count{{class="{traffic_class}"}} {len(samples)}'
        )
    return lines
