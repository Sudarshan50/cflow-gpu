"""Regression coverage for the production dashboard data contract."""

from __future__ import annotations

import importlib.util
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "dashboard" / "server.py"
HTML = ROOT / "dashboard" / "index.html"
SPEC = importlib.util.spec_from_file_location("k3dash_monitor", SERVER)
dash = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dash)


class PrometheusParsingTest(unittest.TestCase):
    def test_labelled_samples_preserve_gateway_dimensions(self):
        rows = dash.parse_prometheus_samples(
            'k3_gateway_requests_total{outcome="reject_shed",'
            'traffic_class="P2-agentic"} 7\n'
            'k3_gateway_admission_queued_requests 2\n'
        )
        self.assertEqual(rows[0]["labels"]["outcome"], "reject_shed")
        self.assertEqual(rows[0]["labels"]["traffic_class"], "P2-agentic")
        self.assertEqual(rows[0]["value"], 7)
        self.assertEqual(rows[1]["labels"], {})

    def test_live_access_rows_do_not_invent_request_bytes(self):
        row = dash.parse_access_line(
            '45.78.68.8 bearer [22/Sep/2026:08:00:00 +0000] '
            '"POST /v1/chat/completions HTTP/2.0" 200 4504 '
            "rt=1.106 ttfb=0.589 cls=P2-agentic"
        )
        self.assertIsNone(row["in"])
        self.assertEqual(row["out"], 4504)
        self.assertEqual(row["cls"], "P2-agentic")


class GatewayViewTest(unittest.TestCase):
    @staticmethod
    def sample(ts, admitted, rejected, timeouts, queued):
        counters = {
            dash._metric_key(
                "k3_gateway_requests_total",
                {"outcome": "admit", "traffic_class": "P2-agentic"},
            ): admitted,
            dash._metric_key(
                "k3_gateway_workload_rejections_total",
                {"reason": "long_output_slots"},
            ): rejected,
            dash._metric_key(
                "k3_gateway_admission_queue_timeouts_total", {}
            ): timeouts,
        }
        gauges = {
            dash._metric_key(
                "k3_gateway_admission_queued_requests", {}
            ): queued,
        }
        latency = {
            dash._metric_key(
                "k3_gateway_ttft_seconds",
                {"class": "P2-agentic", "quantile": "0.95"},
            ): 2.5,
            dash._metric_key(
                "k3_gateway_ttft_seconds_count",
                {"class": "P2-agentic"},
            ): admitted,
        }
        return {"ts": ts, "counters": counters, "gauges": gauges, "latency": latency}

    def test_view_exposes_current_and_counter_deltas(self):
        now = time.time()
        monitor = dash.GatewayMonitor("http://example.invalid/metrics")
        monitor.samples.extend([
            self.sample(now - 900, 100, 4, 3, 0),
            self.sample(now, 140, 7, 5, 2),
        ])
        monitor.state = "up"
        monitor.last_success = now
        view = monitor.view()
        self.assertEqual(view["current"]["queued_requests"], 2)
        self.assertEqual(view["recent"]["admission"]["timeouts"], 2)
        self.assertEqual(view["recent"]["rejection_reasons"][0]["value"], 3)
        self.assertEqual(view["current"]["latency"][0]["ttft_p95"], 2.5)
        self.assertFalse(view["recent"]["partial"])

    def test_stale_gateway_values_are_not_presented_as_current(self):
        now = time.time()
        monitor = dash.GatewayMonitor("http://example.invalid/metrics")
        monitor.samples.append(self.sample(now - 30, 10, 1, 1, 3))
        monitor.state = "unreachable"
        monitor.last_success = now - 30
        view = monitor.view()
        self.assertTrue(view["source"]["stale"])
        self.assertIsNone(view["current"])
        self.assertEqual(view["last_current"]["queued_requests"], 3)


class HtmlContractTest(unittest.TestCase):
    def test_new_portal_keeps_runtime_hooks(self):
        html = HTML.read_text()
        for hook in (
            'id="sourceBar"',
            'id="admissionKpis"',
            'id="rollupBody"',
            'id="chartTokens"',
            'id="chartCache"',
            'id="mPrefix"',
            'id="engineKpis"',
            "fetch('/api/state'",
            "prefers-reduced-motion",
        ):
            self.assertIn(hook, html)
        self.assertNotIn("Peak TPM", html)
        self.assertNotIn(">sent<", html.lower())


if __name__ == "__main__":
    unittest.main()
