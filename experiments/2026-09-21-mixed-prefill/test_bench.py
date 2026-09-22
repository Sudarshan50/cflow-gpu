"""Exercise real SSE parsing and reject shortened or contaminated workloads."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import bench


class StreamContractTest(unittest.TestCase):
    def setUp(self):
        self.mode = "complete"
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                events = [
                    {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "one"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " two"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": " three"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                    {"choices": [], "usage": {"prompt_tokens": 6,
                     "completion_tokens": 2 if owner.mode == "short-output" else 3,
                     "prompt_tokens_details": {"cached_tokens": 0}}},
                ]
                body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                if owner.mode != "missing-done":
                    body += "data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body.encode())))
                self.end_headers()
                self.wfile.write(body.encode())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def probe(self):
        self.first_events = 0
        def first():
            self.first_events += 1
        spec = {"name": "test", "payload": {"messages": []}, "prompt_tokens": 6,
                "prompt_sha256": "a", "payload_sha256": "b"}
        return bench.stream(f"http://127.0.0.1:{self.server.server_port}", spec, "synthetic", 3, "test", first)

    def test_complete_stream_records_actual_content_and_one_first_event(self):
        row = self.probe()
        self.assertEqual(self.first_events, 1)
        self.assertEqual(row["usage"]["completion_tokens"], 3)
        self.assertEqual(row["visible_gaps"], 2)

    def test_partial_sse_cannot_count_as_success(self):
        self.mode = "missing-done"
        with self.assertRaisesRegex(RuntimeError, "Incomplete fixed-work"):
            self.probe()

    def test_reduced_output_budget_cannot_look_like_a_speedup(self):
        self.mode = "short-output"
        with self.assertRaisesRegex(RuntimeError, "workload differs"):
            self.probe()


class PhaseAttributionTest(unittest.TestCase):
    def test_extra_engine_request_invalidates_whole_concurrent_phase(self):
        before = {"values": {"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 0,
                             "vllm:generation_tokens_total": 10, "vllm:request_success_total": 2}}
        after = {"values": {**before["values"], "vllm:generation_tokens_total": 14,
                            "vllm:request_success_total": 4}}
        with self.assertRaisesRegex(RuntimeError, "Non-isolated phase"):
            bench.accounting(before, after, [{"usage": {"completion_tokens": 3}}])

    def test_busy_boundary_invalidates_phase_even_when_finished_counts_match(self):
        before = {"values": {"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 0}}
        after = {"values": {"vllm:num_requests_running": 1, "vllm:num_requests_waiting": 0,
                            "vllm:generation_tokens_total": 3, "vllm:request_success_total": 1}}
        with self.assertRaisesRegex(RuntimeError, "Non-isolated phase"):
            bench.accounting(before, after, [{"usage": {"completion_tokens": 3}}])


if __name__ == "__main__":
    unittest.main()
