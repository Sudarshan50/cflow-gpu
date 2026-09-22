#!/usr/bin/env python3
"""Compare complete, token-identical baseline/tail128 latency experiments."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from latency import CASES, REVISION, SAMPLES, WARMUP
from study import BASELINE, IMAGE


def compare(baseline, candidate):
    for report, variant in ((baseline, "baseline"), (candidate, "tail128")):
        if report.get("passed") is not True or report.get("failure"):
            raise ValueError("Incomplete or failed benchmark")
        if report.get("variant") != variant:
            raise ValueError("Mislabeled benchmark variant")
        if (report.get("fixture_revision"), report.get("warmups_per_case"), report.get("samples_per_case")) != (REVISION, WARMUP, SAMPLES):
            raise ValueError("Unrecognized benchmark fixture or sampling policy")
        if {row["case"] for row in report["requests"]} != {case for case, _ in CASES}:
            raise ValueError("Missing or unexpected benchmark case")
        identity = report["identity"]
        if identity["baseline_commit"] != BASELINE or IMAGE not in identity["image"]:
            raise ValueError("Unexpected baseline commit or model image")
        if report["cache_config"].get("prefix_match_unit") != ("128" if variant == "tail128" else "None"):
            raise ValueError("Effective cache setting does not match variant")
        control = report.get("cache_block_control")
        if control is not None and (
                identity["config"].get("num-gpu-blocks-override") != control
                or report["cache_config"].get("num_gpu_blocks_override") != str(control)
                or report["cache_config"].get("num_gpu_blocks") != str(control)):
            raise ValueError("Fixed cache control was not actually loaded")
        for row in report["requests"]:
            if (not row["isolated"] or row["finished_counter_delta"] != 1
                    or row["generation_counter_delta"] != row["output_tokens"]
                    or row.get("answer_correct") is False
                    or not row["finish_reasons"]):
                raise ValueError("Unattributed work or invalid completion")
            if not isinstance(row["ttft_seconds"], (int, float)) or not math.isfinite(row["ttft_seconds"]) or row["ttft_seconds"] <= 0:
                raise ValueError("Invalid first-token timing")

    if "prefix-match-unit" in baseline["identity"]["config"]:
        raise ValueError("Baseline contains a prefix-match-unit override")
    expected = {**baseline["identity"]["config"], "prefix-match-unit": 128}
    if candidate["identity"]["config"] != expected:
        raise ValueError("Configuration changed beyond prefix-match-unit")
    for key in ("fixture_revision", "warmups_per_case", "samples_per_case"):
        if baseline[key] != candidate[key]:
            raise ValueError("Benchmark fixture or warmup policy differs")
    if baseline.get("cache_block_control") != candidate.get("cache_block_control"):
        raise ValueError("Cache capacity control differs")
    for key in ("block_size", "mamba_cache_mode", "cache_dtype", "num_gpu_blocks", "kv_cache_size_tokens"):
        if baseline["cache_config"][key] != candidate["cache_config"][key]:
            raise ValueError("Hybrid cache geometry or capacity differs")

    def indexed(report):
        rows = {row["label"]: row for row in report["requests"]}
        if len(rows) != len(report["requests"]):
            raise ValueError("Duplicate request label")
        return rows

    left, right = indexed(baseline), indexed(candidate)
    if left.keys() != right.keys() or not left:
        raise ValueError("Benchmark requests differ")
    for label in left:
        for key in ("case", "phase", "prompt_tokens", "prompt_sha256", "output_tokens", "output_sha256", "finish_reasons"):
            if left[label][key] != right[label][key]:
                raise ValueError(f"Unmatched {key} for {label}")

    result = {"baseline_commit": BASELINE, "fixture_revision": baseline["fixture_revision"],
              "cache_block_control": baseline.get("cache_block_control"),
              "num_gpu_blocks": baseline["cache_config"]["num_gpu_blocks"],
              "kv_cache_size_tokens": baseline["cache_config"]["kv_cache_size_tokens"],
              "matched_requests": len(left), "cases": {},
              "note": "Sequential c=1 microbenchmarks; eight measured requests per case, not a throughput or production-tail claim."}
    for case in sorted({row["case"] for row in left.values()}):
        summary = {}
        for report, name in ((baseline, "baseline"), (candidate, "tail128")):
            rows = [row for row in report["requests"] if row["case"] == case]
            warmups = [row for row in rows if row["phase"] == "warmup"]
            measured = [row for row in rows if row["phase"] == "measure"]
            if len(warmups) != report["warmups_per_case"] or len(measured) != report["samples_per_case"]:
                raise ValueError("Incomplete or unbalanced case")
            times = [row["ttft_seconds"] for row in measured]
            summary[name] = {"samples": len(measured),
                             "cached_tokens": sorted({row["cached_tokens"] for row in measured}),
                             "ttft_median_seconds": statistics.median(times),
                             "ttft_min_seconds": min(times), "ttft_max_seconds": max(times)}
        summary["tail128_ttft_change_percent"] = 100 * (
            summary["tail128"]["ttft_median_seconds"] / summary["baseline"]["ttft_median_seconds"] - 1)
        result["cases"][case] = summary
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; preserve earlier evidence")
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
