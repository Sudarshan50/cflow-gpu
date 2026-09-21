#!/usr/bin/env python3
"""Compare qualified matched workloads; never infer DSpark gain from stock live load."""
import argparse
import json
from pathlib import Path


def checked(path):
    result = json.loads(path.read_text())
    if result.get("failure") or len(result.get("smoke", [])) != 8 or not all(r["passed"] for r in result["smoke"]):
        raise ValueError(f"Unqualified correctness result: {path}")
    by_concurrency = {r["concurrency"]: r for r in result.get("benchmarks", [])}
    if set(by_concurrency) != {1, 16}:
        raise ValueError(f"Incomplete comparison cases: {path}")
    for row in by_concurrency.values():
        if row["errors"] or not row["completed"]:
            raise ValueError(f"Failed workload: {path}")
        if not row.get("isolated"):
            raise ValueError(f"Unaccounted engine output; comparison is not isolated: {path}")
        if len(set(row["task_counts"].values())) != 1:
            raise ValueError(f"Unbalanced task mix: {path}")
    return result, by_concurrency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("/scratch/deploy-state/amd-optimized"))
    args = parser.parse_args()
    plain, base = checked(args.directory / "optimized-base-qualification.json")
    draft, spec = checked(args.directory / "optimized-dspark-qualification.json")
    a_prov, b_prov = plain["provenance"], draft["provenance"]
    for key in ("image", "target", "tuning_env_sha256", "workload_sha256", "workload_version"):
        if a_prov[key] != b_prov[key]:
            raise ValueError(f"Runtime/workload provenance differs: {key}")
    spec_config = dict(b_prov["config"])
    speculation = spec_config.pop("speculative-config", None)
    if spec_config != a_prov["config"] or not speculation or speculation.get("method") != "dspark":
        raise ValueError("Profiles do not differ only by DSpark configuration")
    comparison = []
    for concurrency in (1, 16):
        a, b = base[concurrency], spec[concurrency]
        if a["offered_seconds"] != b["offered_seconds"] or abs(a["prompt_tokens_mean"] - b["prompt_tokens_mean"]) > 5:
            raise ValueError("Workload durations or prompt shape differ")
        counters = b["metric_deltas"]
        accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        proposed = counters.get("vllm:spec_decode_num_draft_tokens_total", 0)
        rounds = counters.get("vllm:spec_decode_num_drafts_total", 0)
        if rounds <= 0 or proposed <= 0:
            raise ValueError("No measured drafting: DSpark engagement is unproven")
        comparison.append({
            "concurrency": concurrency,
            "plain_output_tokens_per_second": a["output_tokens_per_second"],
            "dspark_output_tokens_per_second": b["output_tokens_per_second"],
            "dspark_over_plain": b["output_tokens_per_second"] / a["output_tokens_per_second"],
            "plain_e2e_p95": a["e2e_p95"], "dspark_e2e_p95": b["e2e_p95"],
            "accepted_draft_fraction": accepted / proposed,
            "mean_acceptance_length_with_bonus": 1 + accepted / rounds,
        })
    report = {"scope": "isolated warm-prefix fixed-256-token synthetic workload; not a universal production speedup",
              "cases": comparison,
              "smoke_output_hashes_equal": {a["name"]: a["output_sha256"] == b["output_sha256"] for a,b in zip(plain["smoke"], draft["smoke"])}}
    (args.directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
