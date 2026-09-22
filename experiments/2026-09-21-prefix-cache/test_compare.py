"""A completed flag alone must not qualify unmatched or contaminated trials."""
import copy
import unittest

from compare import compare
from latency import CASES, REVISION, SAMPLES, WARMUP
from study import BASELINE, IMAGE


def reports():
    baseline = {"passed": True, "variant": "baseline", "fixture_revision": REVISION,
                "warmups_per_case": WARMUP, "samples_per_case": SAMPLES,
                "identity": {"baseline_commit": BASELINE, "image": IMAGE,
                             "config": {"max-num-batched-tokens": 4096}},
                "cache_config": {"prefix_match_unit": "None", "block_size": "768",
                                 "mamba_cache_mode": "align", "cache_dtype": "auto",
                                 "num_gpu_blocks": "2177", "kv_cache_size_tokens": "1639906"},
                "requests": []}
    for case, length in CASES:
        for index in range(WARMUP + SAMPLES):
            baseline["requests"].append({
                "label": f"{case}-{index}", "case": case,
                "phase": "warmup" if index < WARMUP else "measure",
                "prompt_tokens": length, "prompt_sha256": "1" * 64,
                "output_tokens": 1, "output_sha256": "2" * 64,
                "finish_reasons": ["length"], "isolated": True,
                "finished_counter_delta": 1, "generation_counter_delta": 1,
                "cached_tokens": 0, "ttft_seconds": 1.0,
            })
    candidate = copy.deepcopy(baseline)
    candidate["variant"] = "tail128"
    candidate["cache_config"]["prefix_match_unit"] = "128"
    candidate["identity"]["config"]["prefix-match-unit"] = 128
    return baseline, candidate


class ComparisonContractTest(unittest.TestCase):
    def test_complete_balanced_trials_can_be_compared(self):
        result = compare(*reports())
        self.assertEqual(result["matched_requests"], len(CASES) * (WARMUP + SAMPLES))

    def test_equal_token_counts_do_not_establish_identical_prompts(self):
        baseline, candidate = reports()
        candidate["requests"][WARMUP]["prompt_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "Unmatched prompt_sha256"):
            compare(baseline, candidate)

    def test_shared_omission_cannot_hide_an_unmeasured_case(self):
        baseline, candidate = reports()
        for report in (baseline, candidate):
            report["requests"] = [row for row in report["requests"] if row["case"] != CASES[0][0]]
        with self.assertRaisesRegex(ValueError, "Missing or unexpected"):
            compare(baseline, candidate)

    def test_extra_engine_work_invalidates_a_report_even_when_marked_passed(self):
        baseline, candidate = reports()
        candidate["requests"][WARMUP]["generation_counter_delta"] = 2
        with self.assertRaisesRegex(ValueError, "Unattributed work"):
            compare(baseline, candidate)

    def test_other_runtime_tuning_cannot_be_attributed_to_cache_matching(self):
        baseline, candidate = reports()
        candidate["identity"]["config"]["max-num-batched-tokens"] = 8192
        with self.assertRaisesRegex(ValueError, "Configuration changed beyond"):
            compare(baseline, candidate)

    def test_even_one_block_of_capacity_drift_requires_a_rematch(self):
        baseline, candidate = reports()
        candidate["cache_config"]["num_gpu_blocks"] = "2176"
        with self.assertRaisesRegex(ValueError, "geometry or capacity"):
            compare(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
