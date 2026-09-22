#!/usr/bin/env python3
"""Require equivalent complete workloads before comparing prefill settings."""
import argparse
import json
from pathlib import Path

from bench import BASELINE, COLD_INPUT, COLD_OUTPUT, CONCURRENCIES, VERSION, WARM_INPUT, WARM_OUTPUT, percentile


def index(report, require_complete=True):
    if require_complete and (report.get("passed") is not True or report.get("failure")):
        raise ValueError("Failed or incomplete trial")
    if report["workload_version"] != VERSION or report["identity"]["baseline_commit"] != BASELINE:
        raise ValueError("Unrecognized experiment provenance")
    if len(report["quality"]) != 10 or not all(row["passed"] for row in report["quality"]):
        raise ValueError("Correctness qualification did not pass")
    phases = {(phase["concurrency"], phase["kind"], phase["round"]): phase for phase in report["phases"]}
    expected = {(c, kind, r) for c in CONCURRENCIES for kind in ("warm", "mixed") for r in range(report["rounds"])}
    coverage_ok = set(phases) == expected if require_complete else bool(phases) and set(phases) <= expected
    if not coverage_ok or len(phases) != len(report["phases"]):
        raise ValueError("Incomplete or duplicate phases")
    for (concurrency, kind, _), phase in phases.items():
        rows = phase["requests"]
        labels = {row["label"] for row in rows}
        expected_labels = {f"warm-{i}" for i in range(concurrency)}
        if kind == "mixed":
            expected_labels |= {"cold-0", "cold-1"}
        if labels != expected_labels or len(rows) != len(labels):
            raise ValueError("Incomplete phase request mix")
        for row in rows:
            warm = row["label"].startswith("warm-")
            expected_input, expected_output = (WARM_INPUT, WARM_OUTPUT) if warm else (COLD_INPUT, COLD_OUTPUT)
            if (row["usage"]["prompt_tokens"], row["usage"]["completion_tokens"], row["finishes"]) != (expected_input, expected_output, ["length"]):
                raise ValueError("Changed fixed-work budget")
            expected_cached = WARM_INPUT // 768 * 768 if warm else 0
            if row["usage"]["prompt_tokens_details"]["cached_tokens"] != expected_cached:
                raise ValueError("Changed cache warmth")
        counts = phase["accounting"]
        if (counts["vllm:generation_tokens_total"] != sum(row["usage"]["completion_tokens"] for row in rows)
                or counts["vllm:request_success_total"] != len(rows)
                or counts["vllm:num_preemptions_total"] != 0):
            raise ValueError("Engine work does not match the requests")
    return phases


def stats(phases):
    rows = [row for phase in phases for row in phase["requests"]]
    warm = [row for row in rows if row["label"].startswith("warm-")]
    cold = [row for row in rows if row["label"].startswith("cold-")]
    return {"rounds": len(phases), "warm_requests": len(warm), "cold_requests": len(cold),
            "aggregate_output_tps": sum(row["usage"]["completion_tokens"] for row in rows) / sum(phase["wall_seconds"] for phase in phases),
            "warm_e2e_p50": percentile([row["seconds"] for row in warm], .5),
            "warm_e2e_p95": percentile([row["seconds"] for row in warm], .95),
            "warm_after_first_tps_p50": percentile([row["after_first_tps"] for row in warm], .5),
            "warm_stream_p95_gap_median": percentile([row["visible_gap_p95"] for row in warm], .5),
            "warm_stream_max_gap": max(row["visible_gap_max"] for row in warm),
            "cold_ttft_p50": percentile([row["ttft"] for row in cold], .5),
            "cold_ttft_max": max((row["ttft"] for row in cold), default=None),
            "median_round_engine_itl_p95": percentile([phase["summary"]["engine_itl_p95_estimate"] for phase in phases], .5)}


def compare(baseline, candidate, completed_only=False):
    left, right = index(baseline, not completed_only), index(candidate, not completed_only)
    if baseline["variant"] != "baseline" or candidate["variant"] != "batch2048" or baseline["rounds"] != candidate["rounds"]:
        raise ValueError("Mismatched variants or repetitions")
    a, b = baseline["identity"], candidate["identity"]
    for field in ("image", "command", "tuning_sha256", "branch"):
        if a[field] != b[field]:
            raise ValueError("Runtime provenance differs: " + field)
    expected = {**a["config"], "max-num-batched-tokens": 2048,
                "num-gpu-blocks-override": int(a["cache_config"]["num_gpu_blocks"])}
    if b["config"] != expected:
        raise ValueError("Candidate changed additional numerical configuration")
    for field in ("block_size", "mamba_cache_mode", "prefix_match_unit", "num_gpu_blocks", "kv_cache_size_tokens", "cache_dtype"):
        if a["cache_config"][field] != b["cache_config"][field]:
            raise ValueError("Effective cache geometry/capacity differs")
    if baseline["fixtures"] != candidate["fixtures"]:
        raise ValueError("Input fixtures differ")
    common = set(left) & set(right)
    included = [c for c in CONCURRENCIES if all((c, kind, r) in common
                for kind in ("warm", "mixed") for r in range(baseline["rounds"]))]
    if not included:
        raise ValueError("No complete paired concurrency cohort")
    common = {key for key in common if key[0] in included}
    output_matches = total = 0
    for key in sorted(common):
        lrows = {row["label"]: row for row in left[key]["requests"]}
        rrows = {row["label"]: row for row in right[key]["requests"]}
        for label, row in lrows.items():
            other = rrows[label]
            for field in ("fixture", "prompt_sha256", "payload_sha256"):
                if row[field] != other[field]:
                    raise ValueError("A paired request has different input")
            output_matches += row["output_sha256"] == other["output_sha256"]
            total += 1
    result = {"baseline_commit": BASELINE, "workload_version": VERSION, "matched_request_pairs": total,
              "comparison_status": "completed-phases-only" if completed_only else "complete",
              "included_concurrencies": included,
              "excluded_concurrencies": [c for c in CONCURRENCIES if c not in included],
              "source_trial_failures": {"baseline": baseline.get("failure"), "candidate": candidate.get("failure")},
              "exact_benchmark_output_hash_matches": output_matches,
              "effective_cache_blocks": a["cache_config"]["num_gpu_blocks"],
              "correctness_checks_per_arm": len(baseline["quality"]), "cases": {}}
    for concurrency in included:
        for kind in ("warm", "mixed"):
            first = stats([left[(concurrency, kind, r)] for r in range(baseline["rounds"])])
            second = stats([right[(concurrency, kind, r)] for r in range(candidate["rounds"])])
            changes = {field: (second[field] / first[field] - 1) * 100 for field in (
                "aggregate_output_tps", "warm_e2e_p50", "warm_e2e_p95", "warm_after_first_tps_p50",
                "warm_stream_p95_gap_median", "warm_stream_max_gap", "cold_ttft_p50", "cold_ttft_max") if first[field]}
            result["cases"][f"c{concurrency}-{kind}"] = {"baseline": first, "batch2048": second,
                                                       "batch2048_change_percent": changes}
    result["scope"] = "Fixed-work synthetic tests; visible streaming gaps and cold-request latency are reported together."
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true",
                        help="Explicitly report only fully paired completed cohorts; retain trial failures")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output path")
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), args.completed_only)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
