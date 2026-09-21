"""Covers the traffic checks against the surviving production window."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from redesign.traffic import analysis
from redesign.traffic.records import (
    CustomerStats,
    FailureCause,
    LatencyQuantiles,
    PathStats,
    TrafficWindow,
)
from redesign.traffic.sources import GatewayTraceSource, ProdStatsSource, parse_duration

BASELINE = Path("eval/runs/20260919T145955Z-baseline/prod_stats.json")


def _window(**overrides) -> TrafficWindow:
    defaults = dict(
        source="test",
        coverage_seconds=3600.0,
        requests=100,
        successful=100,
        failures=0,
        requests_per_minute=1.0,
        status_spread={"200": 100},
        failure_causes=(),
        customers=(),
        paths=(),
        latency=LatencyQuantiles(1.0, 1.0, 1.0, 1.0),
    )
    return TrafficWindow(**{**defaults, **overrides})


class FailureCauseTest(unittest.TestCase):
    def test_status_is_parsed_from_the_cause_string(self):
        self.assertEqual(FailureCause("400 request body rejected", 5).status, 400)

    def test_a_cause_without_a_status_yields_none(self):
        self.assertIsNone(FailureCause("engine wedged", 5).status)


class RejectedRequestsTest(unittest.TestCase):
    def setUp(self):
        self.check = analysis.RejectedRequests()

    def test_heavy_rejection_is_critical(self):
        window = _window(
            requests=651,
            failures=108,
            failure_causes=(FailureCause("400 request body rejected", 75),),
        )
        self.assertEqual(self.check.run(window).severity, analysis.CRITICAL)

    def test_no_rejections_is_ok(self):
        self.assertEqual(self.check.run(_window()).severity, analysis.OK)


class ClientConcentrationTest(unittest.TestCase):
    def setUp(self):
        self.check = analysis.ClientConcentration()

    def _customer(self, name, requests):
        return CustomerStats(name, requests, 0, 1, 0, 0, None, None)

    def test_a_dominant_client_warns(self):
        window = _window(customers=(self._customer("a", 571), self._customer("b", 53)))
        self.assertEqual(self.check.run(window).severity, analysis.WARN)

    def test_spread_traffic_is_ok(self):
        window = _window(customers=(self._customer("a", 50), self._customer("b", 50)))
        self.assertEqual(self.check.run(window).severity, analysis.OK)

    def test_unauthenticated_probes_are_excluded_from_the_denominator(self):
        window = _window(
            customers=(
                self._customer("a", 50),
                self._customer("b", 50),
                self._customer("(unauthenticated)", 500),
            )
        )
        self.assertEqual(self.check.run(window).severity, analysis.OK)


class LatencyBudgetTest(unittest.TestCase):
    def setUp(self):
        self.check = analysis.LatencyBudget()

    def test_a_median_over_the_target_is_critical(self):
        window = _window(latency=LatencyQuantiles(57.9, 300.7, 372.8, 1043.9))
        self.assertEqual(self.check.run(window).severity, analysis.CRITICAL)

    def test_missing_latency_is_unknown_not_ok(self):
        window = _window(latency=LatencyQuantiles(None, None, None, None))
        self.assertEqual(self.check.run(window).severity, analysis.UNKNOWN)


class UnservedEndpointsTest(unittest.TestCase):
    def test_a_wholly_failing_route_warns(self):
        window = _window(paths=(PathStats("/v1/embeddings", 2, 2),))
        self.assertEqual(analysis.UnservedEndpoints().run(window).severity, analysis.WARN)

    def test_a_healthy_route_is_ok(self):
        window = _window(paths=(PathStats("/v1/chat/completions", 606, 83),))
        self.assertEqual(analysis.UnservedEndpoints().run(window).severity, analysis.OK)


class SampleAdequacyTest(unittest.TestCase):
    def test_a_short_window_warns(self):
        window = _window(requests=651, coverage_seconds=3487.0)
        self.assertEqual(analysis.SampleAdequacy().run(window).severity, analysis.WARN)

    def test_a_full_day_is_ok(self):
        window = _window(requests=50_000, coverage_seconds=90_000)
        self.assertEqual(analysis.SampleAdequacy().run(window).severity, analysis.OK)


@unittest.skipUnless(BASELINE.exists(), "baseline evidence not present")
class BaselineWindowTest(unittest.TestCase):
    """Regression guard on the real surviving window."""

    def setUp(self):
        self.window = ProdStatsSource(BASELINE, "1h").load()
        self.findings = {f.check: f for f in analysis.analyse(self.window)}

    def test_the_window_parses(self):
        self.assertEqual(self.window.requests, 651)
        self.assertAlmostEqual(self.window.failure_rate, 0.1659, places=3)

    def test_rejections_are_flagged_critical(self):
        self.assertEqual(self.findings["rejected requests"].severity, analysis.CRITICAL)

    def test_concentration_is_flagged(self):
        self.assertEqual(self.findings["client concentration"].severity, analysis.WARN)

    def test_findings_are_ordered_worst_first(self):
        severities = [f.severity for f in analysis.analyse(self.window)]
        ranks = [analysis.SEVERITY_ORDER[s] for s in severities]
        self.assertEqual(ranks, sorted(ranks))


class GatewayTraceSourceTest(unittest.TestCase):
    def test_window_units(self):
        self.assertEqual(parse_duration("24h"), 86400.0)
        self.assertEqual(parse_duration("15m"), 900.0)

    def test_admit_and_shed_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            now = 1_000_000.0
            rows = [
                {"timestamp": now - 10, "customer": "acme", "path": "/v1/chat/completions",
                 "outcome": "admit"},
                {"timestamp": now - 5, "customer": "acme", "path": "/v1/chat/completions",
                 "outcome": "reject_shed"},
                {"timestamp": now - 90_000, "customer": "old", "path": "/v1/chat/completions",
                 "outcome": "admit"},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            window = GatewayTraceSource(path, "1h", now=now).load()
        self.assertEqual(window.requests, 2)
        self.assertEqual(window.successful, 1)
        self.assertEqual(window.failures, 1)
        self.assertEqual(window.customers[0].customer, "acme")


if __name__ == "__main__":
    unittest.main()
