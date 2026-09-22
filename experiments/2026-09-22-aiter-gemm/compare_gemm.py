#!/usr/bin/env python3
"""Fail-closed comparison for qualified baseline and AITER GEMM candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def checked(path: Path) -> tuple[dict, dict[int, dict]]:
    result = json.loads(path.read_text())
    if result.get("failure"):
        raise ValueError(f"failed qualification: {path}")
    smoke = result.get("smoke", [])
    if len(smoke) != 8 or not all(row.get("passed") for row in smoke):
        raise ValueError(f"unqualified correctness result: {path}")
    cases = {row["concurrency"]: row for row in result.get("benchmarks", [])}
    if set(cases) != {1, 16}:
        raise ValueError(f"incomplete workload cases: {path}")
    for row in cases.values():
        if row["errors"] or not row.get("isolated") or row["completed"] <= 0:
            raise ValueError(f"failed or contaminated workload: {path}")
        if row["metric_deltas"].get("vllm:num_preemptions_total", 0):
            raise ValueError(f"preemptions invalidate comparison: {path}")
    return result, cases


def change(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base_result, base = checked(args.baseline)
    cand_result, candidate = checked(args.candidate)
    a, b = base_result["provenance"], cand_result["provenance"]
    for key in ("target", "workload_sha256", "workload_version"):
        if a[key] != b[key]:
            raise ValueError(f"workload provenance differs: {key}")
    if a["config"] != b["config"]:
        raise ValueError("scheduler/model profiles differ")
    rows = []
    accepted = True
    for concurrency in (1, 16):
        old, new = base[concurrency], candidate[concurrency]
        throughput = change(
            new["output_tokens_per_second"], old["output_tokens_per_second"]
        )
        e2e = change(new["e2e_p95"], old["e2e_p95"])
        ttft = change(new["ttft_p95"], old["ttft_p95"])
        if throughput < -0.02:
            accepted = False
        rows.append(
            {
                "concurrency": concurrency,
                "baseline_output_tps": old["output_tokens_per_second"],
                "candidate_output_tps": new["output_tokens_per_second"],
                "throughput_change": throughput,
                "e2e_p95_change": e2e,
                "ttft_p95_change": ttft,
            }
        )
    report = {
        "accepted": accepted,
        "scope": "isolated fixed-output qualification; correctness is mandatory",
        "baseline_image": a["image"],
        "candidate_image": b["image"],
        "cases": rows,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit("candidate exceeds the 2% throughput regression gate")


if __name__ == "__main__":
    main()
