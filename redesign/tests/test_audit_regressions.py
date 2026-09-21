"""Regression tests for the 2026-09-20 independent audit.

Each test is written to FAIL against the pre-audit code. The original suite
passed 117 tests while every defect below was live, because it tested the
decision logic and never the concurrency, the streaming path, or the malformed
input. See redesign/AUDIT-2026-09-20.md.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from redesign.gate.__main__ import main as gate_main
from redesign.gateway.backpressure import ClassBudget
from redesign.gateway.classification import ALL_CLASSES, LONG_CONTEXT
from redesign.gateway.metrics import MAX_SAMPLES_PER_CLASS, LatencySeries
from redesign.gateway.server import Handler, build_service

WINDOW = 262_144


class ClassBudgetConcurrencyTest(unittest.TestCase):
    """G-3: unguarded read-modify-write leaked up to 19 slots per trial."""

    def test_balanced_churn_leaks_no_slots(self):
        for _ in range(10):
            budget = ClassBudget.from_classes(1_000, ALL_CLASSES)

            def churn():
                for _ in range(400):
                    if budget.try_acquire(LONG_CONTEXT):
                        budget.release(LONG_CONTEXT)

            threads = [threading.Thread(target=churn) for _ in range(16)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(budget.in_flight(LONG_CONTEXT), 0)

    def test_the_ceiling_is_never_exceeded_under_contention(self):
        budget = ClassBudget.from_classes(40, ALL_CLASSES)  # LONG_CONTEXT -> 10
        limit = budget.limit_for(LONG_CONTEXT)
        peak = [0]
        lock = threading.Lock()

        def acquire_and_hold():
            if budget.try_acquire(LONG_CONTEXT):
                with lock:
                    peak[0] = max(peak[0], budget.in_flight(LONG_CONTEXT))
                time.sleep(0.005)
                budget.release(LONG_CONTEXT)

        threads = [threading.Thread(target=acquire_and_hold) for _ in range(80)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertLessEqual(peak[0], limit)


class LatencySeriesEvictionTest(unittest.TestCase):
    """G-4: the series was kept sorted and pop(0) evicted the SMALLEST sample,
    so one incident pinned p50/p95 at the incident value permanently."""

    def test_an_incident_does_not_pin_the_quantiles_forever(self):
        series = LatencySeries()
        for _ in range(MAX_SAMPLES_PER_CLASS):
            series.observe(9.0)
        for _ in range(MAX_SAMPLES_PER_CLASS * 2):
            series.observe(0.1)
        self.assertAlmostEqual(series.quantile(0.5), 0.1)
        self.assertAlmostEqual(series.quantile(0.95), 0.1)

    def test_eviction_is_by_age_not_by_value(self):
        series = LatencySeries()
        for value in range(MAX_SAMPLES_PER_CLASS):
            series.observe(float(value))
        series.observe(-1.0)  # newest, and the smallest
        self.assertIn(-1.0, series.samples)
        self.assertNotIn(0.0, series.samples)  # oldest evicted, not smallest

    def test_the_window_stays_bounded(self):
        series = LatencySeries()
        for i in range(MAX_SAMPLES_PER_CLASS * 3):
            series.observe(float(i))
        self.assertEqual(len(series.samples), MAX_SAMPLES_PER_CLASS)


class GateEmptySelectionTest(unittest.TestCase):
    """Q-1: `all([])` is True, so a run with zero checks reported PASS and
    exited 0 -- against a dead engine."""

    def test_zero_checks_never_reports_success(self):
        # Tier 3 with a context list that yields no checks would previously
        # produce "GATE PASS 0 checks" and exit 0.
        code = gate_main(["--tier", "1", "--url", "http://127.0.0.1:1"])
        self.assertEqual(code, 1)

    def test_long_context_requires_a_value(self):
        with self.assertRaises(SystemExit):
            gate_main(["--tier", "3", "--long-context", "--url", "http://127.0.0.1:1"])


class StreamingMockEngine(BaseHTTPRequestHandler):
    """Emits SSE events slowly, the way a real token stream does."""

    protocol_version = "HTTP/1.1"
    events = 40
    interval = 0.01
    mode = "stream"

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b"sglang:token_usage 0.5\n" if self.path == "/metrics" else b"{}"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            for i in range(self.events):
                payload = json.dumps({"choices": [{"delta": {"content": f"tok{i} "}}]})
                chunk = f"data: {payload}\n\n".encode()
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                time.sleep(self.interval)
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


class StreamingTest(unittest.TestCase):
    """G-1: read(8192) blocked until 8 KB accumulated, so the first byte
    reached the client seconds after the engine produced it."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ThreadingHTTPServer(("127.0.0.1", 0), StreamingMockEngine)
        cls.engine.daemon_threads = True
        threading.Thread(target=cls.engine.serve_forever, daemon=True).start()

        Handler.service = build_service(
            engine_url=f"http://127.0.0.1:{cls.engine.server_address[1]}",
            max_model_len=WINDOW, concurrency_ceiling=96,
            trace_path=None, model_path=None, send_priority=False,
            snapshot_ttl=0.0,
        )
        cls.gateway = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.gateway.daemon_threads = True
        threading.Thread(target=cls.gateway.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.gateway.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.gateway.shutdown()
        cls.engine.shutdown()

    def test_first_byte_arrives_before_the_stream_completes(self):
        request = urllib.request.Request(
            f"{self.base}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "user", "content": "hi"}], "stream": True,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response:
            first = response.read(1)
            first_byte_at = time.monotonic() - started
            response.read()
        total = time.monotonic() - started

        self.assertTrue(first)
        # The engine emits its first event after ~one interval. Buffering to
        # 8 KB would push this to the end of the stream.
        self.assertLess(first_byte_at, total * 0.5)


class MalformedBodyTest(unittest.TestCase):
    """G-5: non-dict JSON reached payload.get() and killed the handler thread
    with no HTTP response at all; `null` pinned the client until its timeout."""

    @classmethod
    def setUpClass(cls):
        StreamingTest.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        StreamingTest.tearDownClass.__func__(cls)

    def _post_raw(self, body: bytes):
        request = urllib.request.Request(
            f"{self.base}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            with exc:
                exc.read()
                return exc.code

    def test_a_json_array_body_gets_a_400_not_a_dropped_connection(self):
        self.assertEqual(self._post_raw(b'[1,2,3]'), 400)

    def test_a_json_null_body_gets_a_400_not_a_hang(self):
        self.assertEqual(self._post_raw(b'null'), 400)

    def test_a_json_string_body_gets_a_400(self):
        self.assertEqual(self._post_raw(b'"hello"'), 400)

    def test_a_json_number_body_gets_a_400(self):
        self.assertEqual(self._post_raw(b'42'), 400)


if __name__ == "__main__":
    unittest.main()
