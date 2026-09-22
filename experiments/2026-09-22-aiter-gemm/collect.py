#!/usr/bin/env python3
"""Capture reproducible live Kimi-K3 engine/gateway and AITER fallback evidence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ENGINE = "http://127.0.0.1:8001/metrics"
GATEWAY = "http://127.0.0.1:8002/metrics"
MISS = re.compile(
    r"shape is M:(\d+), N:(\d+), K:(\d+).*not found tuned config"
)


def scrape(url: str) -> dict[str, float]:
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        key, raw = line.rsplit(" ", 1)
        try:
            values[key] = float(raw)
        except ValueError:
            pass
    return values


def family(values: dict[str, float], name: str) -> float:
    return sum(v for k, v in values.items() if k.split("{", 1)[0] == name)


def digest_in_container(path: str) -> str | None:
    result = subprocess.run(
        ["docker", "exec", "k3", "sha256sum", path],
        text=True,
        capture_output=True,
    )
    return result.stdout.split()[0] if result.returncode == 0 else None


def fallback_inventory(since: str) -> dict:
    text = subprocess.check_output(
        ["journalctl", "-u", "k3", "--since", since, "--no-pager", "-o", "cat"],
        text=True,
        errors="replace",
    )
    rows = [
        tuple(map(int, match.groups()))
        for line in text.splitlines()
        if (match := MISS.search(line))
    ]
    exact = collections.Counter(rows)
    families = collections.Counter((n, k) for _, n, k in rows)
    return {
        "warning_lines": len(rows),
        "unique_shapes": len(exact),
        "tp8_unique_shape_estimate": len(rows) / 8,
        "m_buckets": dict(
            collections.Counter(
                "decode_le_128"
                if m <= 128
                else "prefill_le_4096"
                if m <= 4096
                else "large_gt_4096"
                for m, _, _ in rows
            )
        ),
        "top_shapes": [
            {"m": key[0], "n": key[1], "k": key[2], "warnings": count}
            for key, count in exact.most_common(100)
        ],
        "top_families": [
            {"n": key[0], "k": key[1], "warnings": count}
            for key, count in families.most_common(50)
        ],
    }


def snapshot() -> dict:
    inspect = json.loads(
        subprocess.check_output(["docker", "inspect", "k3"], text=True)
    )[0]
    started = inspect["State"]["StartedAt"]
    versions = subprocess.check_output(
        [
            "docker",
            "exec",
            "k3",
            "python3",
            "-c",
            "import torch,flydsl; print(torch.__version__); "
            "print(getattr(flydsl,'__version__','unknown'))",
        ],
        text=True,
    ).splitlines()
    return {
        "timestamp": time.time(),
        "engine_started": started,
        "container_id": inspect["Id"],
        "image": inspect["Config"]["Image"],
        "command": inspect["Config"]["Cmd"],
        "torch": versions[-2],
        "flydsl": versions[-1],
        "table_sha256": digest_in_container(
            "/tmp/aiter_configs/bf16_tuned_gemm.csv"
        ),
        "metrics": {"engine": scrape(ENGINE), "gateway": scrape(GATEWAY)},
        "fallbacks": fallback_inventory(started),
    }


def delta_report(start: dict, end: dict) -> dict:
    elapsed = end["timestamp"] - start["timestamp"]
    olde, newe = start["metrics"]["engine"], end["metrics"]["engine"]
    oldg, newg = start["metrics"]["gateway"], end["metrics"]["gateway"]

    def delta(old: dict[str, float], new: dict[str, float], name: str) -> float:
        return family(new, name) - family(old, name)

    report = {"elapsed_seconds": elapsed}
    counters = (
        "vllm:request_success_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:num_preemptions_total",
    )
    for name in counters:
        report[name] = delta(olde, newe, name)
    for root, label in (
        ("vllm:inter_token_latency_seconds", "mean_itl_seconds"),
        ("vllm:time_to_first_token_seconds", "mean_ttft_seconds"),
        ("vllm:request_prefill_time_seconds", "mean_prefill_seconds"),
        ("vllm:e2e_request_latency_seconds", "mean_e2e_seconds"),
    ):
        count = delta(olde, newe, root + "_count")
        total = delta(olde, newe, root + "_sum")
        report[label] = total / count if count else None
        report[label + "_samples"] = count
    report["generation_tokens_per_second"] = (
        report["vllm:generation_tokens_total"] / elapsed
    )
    report["prompt_tokens_per_second"] = (
        report["vllm:prompt_tokens_total"] / elapsed
    )
    queries = report["vllm:prefix_cache_queries_total"]
    hits = report["vllm:prefix_cache_hits_total"]
    report["prefix_hit_share"] = hits / queries if queries else None
    for name in (
        "k3_gateway_admission_queued_total",
        "k3_gateway_admission_queued_admitted_total",
        "k3_gateway_admission_queue_timeouts_total",
        "k3_gateway_admission_queue_full_total",
    ):
        report[name] = delta(oldg, newg, name)
    for name in (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    ):
        report["end_" + name] = family(newe, name)
    report["fallbacks"] = end["fallbacks"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "end"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.command == "start":
        value = snapshot()
        path = args.directory / "start.json"
    else:
        start_path = args.directory / "start.json"
        if not start_path.exists():
            parser.error(f"missing baseline snapshot: {start_path}")
        value = snapshot()
        path = args.directory / "end.json"
        report = delta_report(json.loads(start_path.read_text()), value)
        (args.directory / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    path.write_text(json.dumps(value, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
