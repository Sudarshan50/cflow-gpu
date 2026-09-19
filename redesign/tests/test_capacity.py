"""Guards the arithmetic docs/SYSTEM-DESIGN.md depends on.

    python3 -m unittest discover redesign/tests
"""

from __future__ import annotations

import unittest

from redesign.capacity import deployment, hypothesis, report, scenarios
from redesign.capacity.model import (
    GiB,
    KiB,
    MiB,
    Allocation,
    Architecture,
    CapacityModel,
    Parallelism,
    Precision,
    PromptDistribution,
)


class ReplicationFactorTest(unittest.TestCase):
    def test_single_kv_head_replicates_across_every_rank(self):
        self.assertEqual(Parallelism(tp_size=8, mla_kv_heads=1).replication_factor, 8)

    def test_deduplication_removes_the_factor(self):
        parallelism = Parallelism(tp_size=8, mla_kv_heads=1, deduplicated=True)
        self.assertEqual(parallelism.replication_factor, 1)

    def test_enough_kv_heads_shards_cleanly(self):
        self.assertEqual(Parallelism(tp_size=8, mla_kv_heads=8).replication_factor, 1)

    def test_more_heads_than_ranks_never_goes_below_one(self):
        self.assertEqual(Parallelism(tp_size=2, mla_kv_heads=8).replication_factor, 1)


class ArchitectureTest(unittest.TestCase):
    def test_layer_counts_must_sum(self):
        with self.assertRaises(ValueError):
            Architecture(
                total_layers=93,
                kda_layers=69,
                mla_layers=20,
                kda_state_bytes=54 * MiB,
                mla_bytes_per_token_per_rank=27 * KiB,
            )


class CostModelTest(unittest.TestCase):
    def setUp(self):
        self.model = deployment.production_model()

    def test_replicated_cost_is_eight_ranks_worth(self):
        self.assertEqual(self.model.bytes_per_token, 216 * KiB)

    def test_deduplicated_cost_is_one_rank_worth(self):
        self.assertEqual(self.model.with_variant(deduplicated=True).bytes_per_token, 27 * KiB)

    def test_fp8_halves_the_cost(self):
        self.assertEqual(self.model.with_variant(fp8_kv=True).bytes_per_token, 108 * KiB)

    def test_breakeven_is_where_kda_equals_mla(self):
        self.assertAlmostEqual(self.model.breakeven_tokens, 256.0)

    def test_sequence_cost_has_a_fixed_and_a_linear_term(self):
        zero_length = self.model.sequence_cost(0)
        self.assertEqual(zero_length, 54 * MiB)
        self.assertEqual(self.model.sequence_cost(1_000) - zero_length, 1_000 * 216 * KiB)

    def test_with_variant_leaves_the_base_untouched(self):
        self.model.with_variant(deduplicated=True, fp8_kv=True)
        self.assertEqual(self.model.bytes_per_token, 216 * KiB)

    def test_a_maximum_length_request_holds_a_tenth_of_the_pool(self):
        self.assertAlmostEqual(self.model.pool_share(262_144), 0.111, places=3)


class HypothesisTest(unittest.TestCase):
    def setUp(self):
        self.result = hypothesis.evaluate(
            deployment.production_model(), deployment.REPORTED_POOL_TOKENS
        )

    def test_replication_is_the_better_explanation(self):
        self.assertTrue(self.result.confirmed)

    def test_replicated_prediction_lands_within_five_percent(self):
        self.assertLess(self.result.error_replicated, 0.05)

    def test_deduplicated_prediction_is_off_by_roughly_the_tp_size(self):
        self.assertGreater(self.result.error_deduplicated, 5.0)

    def test_residual_is_a_plausible_kda_reservation(self):
        self.assertGreater(self.result.residual_as_kda_sequences, 0)
        self.assertLess(
            self.result.residual_as_kda_sequences, deployment.CONFIGURED_MAX_NUM_SEQS
        )


class ScenarioTest(unittest.TestCase):
    def setUp(self):
        base = deployment.production_model()
        self.by_ids = {
            r.scenario.register_ids: r
            for r in scenarios.evaluate(base, int(deployment.OBSERVED_PROMPTS.mean_tokens))
        }

    def test_baseline_multiplier_is_one(self):
        self.assertAlmostEqual(self.by_ids[()].multiplier, 1.0)

    def test_deduplication_is_worth_the_tp_size(self):
        self.assertAlmostEqual(self.by_ids[("A1",)].multiplier, 8.0)

    def test_fp8_is_worth_two(self):
        self.assertAlmostEqual(self.by_ids[("A2",)].multiplier, 2.0)

    def test_deduplication_and_fp8_multiply(self):
        self.assertAlmostEqual(self.by_ids[("A1", "A2")].multiplier, 16.0)

    def test_host_tier_adds_addressable_not_resident(self):
        tiered = self.by_ids[("A3",)]
        self.assertEqual(tiered.resident_tokens, self.by_ids[()].resident_tokens)
        self.assertGreater(tiered.tier_tokens, 0)


class ProductionCrossCheckTest(unittest.TestCase):
    """The model must reproduce what the box actually does."""

    def setUp(self):
        self.report = report.build()

    def test_modelled_concurrency_matches_observed_steady_state(self):
        low = deployment.OBSERVED_RUNNING[0] + deployment.OBSERVED_QUEUED[0]
        high = deployment.OBSERVED_RUNNING[1] + deployment.OBSERVED_QUEUED[1]
        self.assertGreaterEqual(self.report.modelled_concurrency, low - 10)
        self.assertLessEqual(self.report.modelled_concurrency, high + 10)

    def test_configured_ceiling_is_a_large_overcommit(self):
        self.assertGreater(self.report.overcommit_factor, 5.0)


class PromptDistributionTest(unittest.TestCase):
    def test_mean_of_a_single_bucket_is_its_midpoint(self):
        self.assertAlmostEqual(PromptDistribution(((1_000, 1.0),)).mean_tokens, 500.0)

    def test_observed_mean_sits_in_the_long_tail(self):
        mean = deployment.OBSERVED_PROMPTS.mean_tokens
        self.assertGreater(mean, 20_000)
        self.assertLess(mean, 50_000)


class AllocationTest(unittest.TestCase):
    def test_reserving_kda_state_reduces_token_capacity(self):
        model = deployment.production_model()
        self.assertLess(model.resident_tokens(512), model.resident_tokens(0))

    def test_an_over_reserved_pool_reports_zero_rather_than_negative(self):
        model = CapacityModel(
            deployment.KIMI_K3,
            deployment.TP8,
            Precision(),
            Allocation(hbm_bytes=1 * GiB),
        )
        self.assertEqual(model.resident_tokens(1_000), 0)


if __name__ == "__main__":
    unittest.main()
