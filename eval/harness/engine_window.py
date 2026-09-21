#!/usr/bin/env python3
"""Windowed throughput sample of the live engine.

Counters in /metrics are cumulative since engine start, so a single scrape
cannot answer "what are we doing right now". This samples twice and reports
rates plus histogram-derived latency for the window only, which is what an
A/B across an engine restart needs.

Latency percentiles come from bucket deltas: the sum/count ratio would carry
every request served since boot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

METRICS_URL = "http://127.0.0.1:8001/metrics"

COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
)

LINE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+(\S+)$")


def scrape(url: str) -> dict:
    text = urllib.request.urlopen(url, timeout=10).read().decode("utf-8", "replace")
    counters: dict[str, float] = {}
    gauges: dict[str, float] = {}
    buckets: dict[str, dict[float, float]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            continue
        name, labels, raw = match.group(1), match.group(2) or "", match.group(3)
        try:
            value = float(raw)
        except ValueError:
            continue
        if value != value:
            continue
        if name.endswith("_bucket"):
            family = name[: -len("_bucket")]
            if family not in HISTOGRAMS:
                continue
            edge = re.search(r'le="([^"]+)"', labels)
            if not edge:
                continue
            bound = float("inf") if edge.group(1) in ("+Inf", "Inf") else float(edge.group(1))
            buckets.setdefault(family, {})
            buckets[family][bound] = buckets[family].get(bound, 0.0) + value
        elif name in COUNTERS:
            counters[name] = counters.get(name, 0.0) + value
        elif name in GAUGES:
            gauges[name] = gauges.get(name, 0.0) + value
    return {"counters": counters, "gauges": gauges, "buckets": buckets}


def bucket_percentiles(before: dict, after: dict, quantiles=(0.5, 0.9, 0.95, 0.99)) -> dict:
    """Percentiles over the requests that completed inside the window."""
    bounds = sorted(after)
    delta = {b: after[b] - before.get(b, 0.0) for b in bounds}
    total = delta.get(bounds[-1], 0.0) if bounds else 0.0
    if total <= 0:
        return {"n": 0}
    out: dict[str, float | int | None] = {"n": int(total)}
    for q in quantiles:
        want = q * total
        low_bound, low_count = 0.0, 0.0
        chosen = None
        for bound in bounds:
            count = delta[bound]
            if count >= want:
                if bound == float("inf"):
                    chosen = low_bound
                else:
                    span = count - low_count
                    frac = (want - low_count) / span if span > 0 else 1.0
                    chosen = low_bound + (bound - low_bound) * frac
                break
            low_bound, low_count = bound, count
        out[f"p{int(q * 100)}"] = round(chosen, 3) if chosen is not None else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(prog="engine_window")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--label", default="window")
    parser.add_argument("--samples", type=int, default=0,
                        help="gauge samples taken across the window (0 = seconds/5)")
    parser.add_argument("--url", default=METRICS_URL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    samples = args.samples or max(2, int(args.seconds // 5))
    interval = args.seconds / samples

    start = scrape(args.url)
    started_at = time.time()
    occupancy = []
    for _ in range(samples):
        time.sleep(interval)
        try:
            gauges = scrape(args.url)["gauges"]
        except Exception:  # noqa: BLE001 -- a missed sample must not lose the window
            continue
        occupancy.append(gauges)
    end = scrape(args.url)
    elapsed = time.time() - started_at

    def rate(name: str) -> float:
        delta = end["counters"].get(name, 0.0) - start["counters"].get(name, 0.0)
        return round(delta / elapsed * 60.0, 1)

    def mean(name: str) -> float | None:
        values = [s.get(name) for s in occupancy if s.get(name) is not None]
        return round(sum(values) / len(values), 2) if values else None

    prompt_delta = end["counters"].get("vllm:prompt_tokens_total", 0.0) - start["counters"].get("vllm:prompt_tokens_total", 0.0)
    gen_delta = end["counters"].get("vllm:generation_tokens_total", 0.0) - start["counters"].get("vllm:generation_tokens_total", 0.0)
    queries = end["counters"].get("vllm:prefix_cache_queries_total", 0.0) - start["counters"].get("vllm:prefix_cache_queries_total", 0.0)
    hits = end["counters"].get("vllm:prefix_cache_hits_total", 0.0) - start["counters"].get("vllm:prefix_cache_hits_total", 0.0)

    report = {
        "label": args.label,
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_seconds": round(elapsed, 1),
        "rates_per_minute": {
            "prompt_tokens": rate("vllm:prompt_tokens_total"),
            "generation_tokens": rate("vllm:generation_tokens_total"),
            "total_tokens": round((prompt_delta + gen_delta) / elapsed * 60.0, 1),
            "requests_finished": rate("vllm:request_success_total"),
            "preemptions": rate("vllm:num_preemptions_total"),
        },
        "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None,
        "occupancy_mean": {
            "running": mean("vllm:num_requests_running"),
            "waiting": mean("vllm:num_requests_waiting"),
            "kv_usage": mean("vllm:kv_cache_usage_perc"),
        },
        "latency_seconds": {},
        "request_shape": {},
    }
    for family in HISTOGRAMS:
        stats = bucket_percentiles(
            start["buckets"].get(family, {}), end["buckets"].get(family, {})
        )
        target = "request_shape" if family.endswith("_tokens") else "latency_seconds"
        report[target][family.replace("vllm:", "")] = stats

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
