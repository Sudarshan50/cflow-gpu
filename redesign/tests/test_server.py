"""End-to-end HTTP tests: a real gateway in front of a mock engine.

Proves the wiring the deployer depends on before any of it reaches a GPU.
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from redesign.gateway import metrics, tokens
from redesign.gateway.engine import EngineClient, parse_prometheus
from redesign.gateway.server import Handler, build_service

WINDOW = 262_144


class MockEngine(BaseHTTPRequestHandler):
    """Records what the gateway forwarded, and refuses over-long requests
    exactly as the real engine does."""

    protocol_version = "HTTP/1.1"
    received: list[dict] = []
    kv_usage = 0.5
    preemptions = 0.0

    def log_message(self, *args) -> None:
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b'{"status":"ok"}', "application/json")
        elif self.path == "/metrics":
            body = (
                f"sglang:token_usage {self.kv_usage}\n"
                f"sglang:num_running_reqs 12\n"
                f"sglang:num_queue_reqs 3\n"
                f"sglang:num_preemptions_total {self.preemptions}\n"
            ).encode()
            self._send(200, body, "text/plain")
        else:
            self._send(404, b"{}", "application/json")

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockEngine.received.append(payload)

        # The defect the clamp exists to prevent.
        if payload.get("max_tokens", 0) + 1_000 > WINDOW:
            self._send(400, b'{"error":"context window exceeded"}', "application/json")
            return
        self._send(200, json.dumps({"choices": [{"text": "ok"}]}).encode(),
                   "application/json")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(url, payload, headers=None):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, exc.read(), dict(exc.headers)


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = _serve(MockEngine)
        engine_url = f"http://127.0.0.1:{cls.engine.server_address[1]}"

        Handler.service = build_service(
            engine_url=engine_url,
            max_model_len=WINDOW,
            concurrency_ceiling=96,
            trace_path=None,
            model_path=None,
            send_priority=True,
        )
        cls.gateway = _serve(Handler)
        cls.base = f"http://127.0.0.1:{cls.gateway.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.gateway.shutdown()
        cls.engine.shutdown()

    def setUp(self):
        MockEngine.received.clear()
        MockEngine.kv_usage = 0.5
        MockEngine.preemptions = 0.0

    def _chat(self, content, **extra):
        return {"messages": [{"role": "user", "content": content}], **extra}


class HealthTest(ServerTestCase):
    def test_health_reports_the_engine_state(self):
        with urllib.request.urlopen(f"{self.base}/health", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["engine"], "up")

    def test_unknown_routes_are_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self.base}/v1/embeddings", timeout=5)
        with ctx.exception as error:
            error.read()
        self.assertEqual(ctx.exception.code, 404)


class ClampTest(ServerTestCase):
    def test_a_fixed_max_tokens_is_clamped_before_the_engine_sees_it(self):
        status, _, headers = _post(
            f"{self.base}/v1/chat/completions",
            self._chat("hello", max_tokens=128_000),
        )
        self.assertEqual(status, 200)
        forwarded = MockEngine.received[-1]["max_tokens"]
        self.assertLess(forwarded, 128_000)
        self.assertEqual(headers["x-k3-max-tokens-granted"], str(forwarded))

    def test_without_the_clamp_this_request_would_have_400ed(self):
        """The mock engine refuses max_tokens + 1000 > window, as the real one does."""
        status, _, _ = _post(
            f"{self.base}/v1/chat/completions",
            self._chat("hello", max_tokens=WINDOW),
        )
        self.assertEqual(status, 200)

    def test_a_prompt_filling_the_window_is_refused_as_400(self):
        status, body, _ = _post(
            f"{self.base}/v1/chat/completions",
            self._chat("x" * (WINDOW * 4)),
        )
        self.assertEqual(status, 400)
        self.assertIn("no room for output", json.loads(body)["error"]["message"])


class ClassificationTest(ServerTestCase):
    def test_class_is_reported_back_on_the_response(self):
        _, _, headers = _post(f"{self.base}/v1/chat/completions", self._chat("hi"))
        self.assertEqual(headers["x-k3-class"], "P1-short-chat")

    def test_tools_lift_a_short_prompt_to_interactive(self):
        payload = self._chat("hi", tools=[{"type": "function", "function": {"name": "f"}}])
        _, _, headers = _post(f"{self.base}/v1/chat/completions", payload)
        self.assertEqual(headers["x-k3-class"], "P0-interactive")

    def test_a_batch_header_wins_over_length(self):
        _, _, headers = _post(
            f"{self.base}/v1/chat/completions", self._chat("hi"),
            headers={"x-k3-batch": "1"},
        )
        self.assertEqual(headers["x-k3-class"], "P3-batch")


class PriorityTest(ServerTestCase):
    def test_priority_is_attached_for_the_engine_scheduler(self):
        payload = self._chat("hi", tools=[{"type": "function", "function": {"name": "f"}}])
        _post(f"{self.base}/v1/chat/completions", payload)
        self.assertEqual(MockEngine.received[-1]["priority"], 0)

    def test_batch_traffic_carries_the_lowest_priority(self):
        _post(f"{self.base}/v1/chat/completions", self._chat("hi"),
              headers={"x-k3-batch": "1"})
        self.assertEqual(MockEngine.received[-1]["priority"], 3)


class BackpressureTest(ServerTestCase):
    def test_a_distressed_engine_sheds_long_context_with_retry_after(self):
        MockEngine.kv_usage = 0.99
        status, _, headers = _post(
            f"{self.base}/v1/chat/completions",
            self._chat("x" * 200_000),
        )
        self.assertEqual(status, 503)
        self.assertIn("Retry-After", headers)

    def test_a_distressed_engine_still_serves_interactive(self):
        MockEngine.kv_usage = 0.99
        payload = self._chat("hi", tools=[{"type": "function", "function": {"name": "f"}}])
        status, _, _ = _post(f"{self.base}/v1/chat/completions", payload)
        self.assertEqual(status, 200)


class MetricsTest(ServerTestCase):
    def test_per_class_latency_is_exposed(self):
        _post(f"{self.base}/v1/chat/completions", self._chat("hi"))
        with urllib.request.urlopen(f"{self.base}/metrics", timeout=5) as response:
            body = response.read().decode()
        self.assertIn("k3_gateway_requests_total", body)
        self.assertIn('k3_gateway_request_seconds{class="P1-short-chat"', body)


class EngineClientTest(unittest.TestCase):
    def test_prometheus_parsing_ignores_comments_and_sums_labels(self):
        parsed = parse_prometheus(
            "# HELP x\nfoo 1.5\nbar{a=\"1\"} 2\nbar{a=\"2\"} 3\nbroken\n"
        )
        self.assertEqual(parsed["foo"], 1.5)
        self.assertEqual(parsed["bar"], 5.0)
        self.assertNotIn("broken", parsed)

    def test_an_unreachable_engine_reports_unhealthy_rather_than_raising(self):
        self.assertFalse(EngineClient("http://127.0.0.1:1").healthy())


class EstimatorTest(unittest.TestCase):
    def setUp(self):
        self.estimator = tokens.HeuristicEstimator()

    def test_the_estimate_is_conservative(self):
        text = "word " * 1_000
        payload = {"messages": [{"role": "user", "content": text}]}
        # ~5000 chars; a real tokenizer yields ~1000 tokens. Over-counting is
        # the safe direction for a clamp.
        self.assertGreater(self.estimator.estimate(payload), 1_000)

    def test_tool_schemas_count_toward_the_prompt(self):
        base = {"messages": [{"role": "user", "content": "hi"}]}
        with_tools = {**base, "tools": [{"function": {"name": "x" * 500}}]}
        self.assertGreater(
            self.estimator.estimate(with_tools), self.estimator.estimate(base)
        )

    def test_multimodal_content_parts_are_counted(self):
        payload = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "y" * 400},
        ]}]}
        self.assertGreater(self.estimator.estimate(payload), 100)

    def test_a_zero_divisor_is_rejected(self):
        with self.assertRaises(ValueError):
            tokens.HeuristicEstimator(chars_per_token=0)


class RegistryTest(unittest.TestCase):
    def test_quantiles_need_no_samples_to_render(self):
        self.assertEqual(metrics.Registry().render().strip(), "")

    def test_samples_are_bounded(self):
        series = metrics.LatencySeries()
        for i in range(metrics.MAX_SAMPLES_PER_CLASS + 500):
            series.observe(float(i))
        self.assertEqual(len(series.samples), metrics.MAX_SAMPLES_PER_CLASS)


if __name__ == "__main__":
    unittest.main()
