"""Regressions for the remaining audit items and the unfinished register.

Each test is written to fail against the pre-this-commit code.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redesign.capacity import deployment, hypothesis
from redesign.distill.runner import Checkpoint, DistillResult, DistillSample, load_prompts
from redesign.edge.keys import KeyTableError, load_customers, render_map
from redesign.probe.cache_salt import compare_counters
from redesign.sessions import REGISTRY


class KeyMapTest(unittest.TestCase):
    def test_a_quote_in_a_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "customers.tsv"
            table.write_text('evil\tfoo" default "everyone;\n')
            with self.assertRaises(KeyTableError):
                load_customers(table)

    def test_an_empty_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "customers.tsv"
            table.write_text("# no customers\n")
            with self.assertRaises(KeyTableError) as ctx:
                load_customers(table)
            self.assertIn("no customers", str(ctx.exception))

    def test_the_map_is_passthrough_not_an_allowlist(self):
        rendered = render_map()
        self.assertIn('default "bearer"', rendered)
        self.assertIn('""      ""', rendered)
        self.assertNotIn("sk-abc", rendered)
        self.assertIn("map_hash_bucket_size 256", rendered)


class DistillCheckpointTest(unittest.TestCase):
    def test_a_completed_sample_is_skipped_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Checkpoint(Path(tmp) / "c.jsonl")
            ckpt.record(DistillResult("a", True, text="ok"))
            restarted = Checkpoint(Path(tmp) / "c.jsonl")
            self.assertIn("a", restarted.completed)

    def test_a_failed_sample_is_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Checkpoint(Path(tmp) / "c.jsonl")
            ckpt.record(DistillResult("a", False, error="HTTP 404"))
            restarted = Checkpoint(Path(tmp) / "c.jsonl")
            self.assertNotIn("a", restarted.completed)

    def test_prompts_load_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            path.write_text(json.dumps({"id": "1", "prompt": "hi"}) + "\n")
            samples = load_prompts(path)
        self.assertEqual(samples, [DistillSample("1", "hi")])


class CacheSaltTest(unittest.TestCase):
    def test_a_rising_hit_counter_means_sharing(self):
        self.assertTrue(compare_counters(0, 10).passed)

    def test_a_zero_second_hit_means_salt_or_empty(self):
        self.assertFalse(compare_counters(0, 0).passed)
        self.assertFalse(compare_counters(5, 0).passed)

    def test_a_warm_cache_still_counts_as_sharing(self):
        self.assertTrue(compare_counters(10, 10).passed)


class SessionRegistryTest(unittest.TestCase):
    def test_every_gpu_session_has_pass_fail_and_rollback(self):
        self.assertEqual({s.id for s in REGISTRY}, {"G-build", "G2", "G3", "G4"})
        for session in REGISTRY:
            self.assertTrue(session.pass_fail)
            self.assertTrue(session.rollback)
            self.assertTrue(session.profile)

    def test_g2_pass_mentions_aggregate_not_per_rank_growth(self):
        g2 = next(s for s in REGISTRY if s.id == "G2")
        self.assertIn("AGGREGATE", g2.pass_fail)
        self.assertIn("per-rank", g2.pass_fail)


class AdmissionWiringTest(unittest.TestCase):
    def test_deploy_json_exposes_recommended_admission(self):
        from redesign.capacity import report
        from redesign.capacity.renderers import JsonRenderer

        payload = json.loads(JsonRenderer().render(report.build()))
        self.assertGreaterEqual(
            payload["recommended_admission"], deployment.OBSERVED_PEAK_CONCURRENCY
        )
        baseline = next(s for s in payload["scenarios"] if s["register_ids"] == [])
        self.assertGreaterEqual(
            baseline["recommended_admission"], deployment.OBSERVED_PEAK_CONCURRENCY
        )
        self.assertLess(baseline["max_concurrency"], 100)


if __name__ == "__main__":
    unittest.main()
