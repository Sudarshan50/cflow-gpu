"""E2 alert evaluation against the shipped rules file."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from redesign.alerts.evaluate import evaluate, load_rules
from redesign.alerts.parser import parse_expression, parse_exposition, parse_window


RULES = Path(__file__).resolve().parents[1] / "deploy" / "alerts" / "k3-gateway.rules.yml"

QUIET = """
k3_gateway_ttft_seconds{class="P0-interactive",quantile="0.95"} 0.4
k3_gateway_requests_total{outcome="reject_shed"} 0
k3_gateway_engine_errors_total 0
k3_gateway_offbox_fallback_total 0
"""

LOUD = """
k3_gateway_ttft_seconds{class="P0-interactive",quantile="0.95"} 4.2
k3_gateway_requests_total{outcome="reject_shed"} 3
k3_gateway_engine_errors_total 9
k3_gateway_offbox_fallback_total 1
"""


class ParserTest(unittest.TestCase):
    def test_labeled_samples_keep_their_class(self):
        samples = parse_exposition(QUIET)
        p0 = next(
            s for s in samples
            if s.name == "k3_gateway_ttft_seconds" and dict(s.labels)["class"] == "P0-interactive"
        )
        self.assertEqual(p0.value, 0.4)

    def test_increase_expression_round_trips(self):
        expr = parse_expression(
            'increase(k3_gateway_requests_total{outcome="reject_shed"}[5m]) > 0'
        )
        self.assertEqual(expr.name, "k3_gateway_requests_total")
        self.assertEqual(expr.labels, {"outcome": "reject_shed"})
        self.assertEqual(expr.increase_window_seconds, 300.0)

    def test_window_units(self):
        self.assertEqual(parse_window("10m"), 600.0)
        self.assertEqual(parse_window("5m"), 300.0)


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.rules = load_rules(RULES)

    def test_the_shipped_rules_file_loads(self):
        self.assertEqual(
            {r.name for r in self.rules},
            {
                "K3GatewayP0TTFTRegression",
                "K3GatewayShedding",
                "K3GatewayEngineErrors",
                "K3GatewayOffboxFallback",
            },
        )

    def test_quiet_scrape_is_ok(self):
        verdicts = evaluate(self.rules, QUIET)
        self.assertTrue(all(not v.firing and not v.pending for v in verdicts))

    def test_p0_ttft_is_pending_until_the_for_window(self):
        rule = next(r for r in self.rules if r.name == "K3GatewayP0TTFTRegression")
        pending = {}
        first = evaluate(self.rules, LOUD, pending_since=pending, now=100.0)
        ttft = next(v for v in first if v.name == "K3GatewayP0TTFTRegression")
        self.assertTrue(ttft.pending)
        self.assertFalse(ttft.firing)
        later = evaluate(
            self.rules, LOUD, pending_since=pending,
            now=100.0 + rule.pending_for_seconds + 1,
        )
        ttft = next(v for v in later if v.name == "K3GatewayP0TTFTRegression")
        self.assertTrue(ttft.firing)

    def test_increase_rules_need_a_previous_scrape(self):
        first = evaluate(self.rules, LOUD)
        shed = next(v for v in first if v.name == "K3GatewayShedding")
        self.assertFalse(shed.firing)
        second = evaluate(self.rules, LOUD, QUIET, elapsed_seconds=300.0, now=300.0)
        errors = next(v for v in second if v.name == "K3GatewayEngineErrors")
        offbox = next(v for v in second if v.name == "K3GatewayOffboxFallback")
        self.assertTrue(errors.firing)
        self.assertTrue(offbox.firing)


class RulesFallbackTest(unittest.TestCase):
    def test_rules_load_without_pyyaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.yml"
            path.write_text(RULES.read_text(encoding="utf-8"), encoding="utf-8")
            rules = load_rules(path)
        self.assertGreaterEqual(len(rules), 4)


if __name__ == "__main__":
    unittest.main()
