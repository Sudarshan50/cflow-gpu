"""Reject misleading measurements from partial SSE or unaccounted engine work."""
import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location("prefix_study", Path(__file__).with_name("study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class MeasurementContractTest(unittest.TestCase):
    def setUp(self):
        self.mode = "complete"
        self.finished = 0
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                output = owner.finished * (2 if owner.mode == "other-work" else 1)
                body = ("vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
                        'vllm:cache_config_info{block_size="768",mamba_cache_mode="align",'
                        'enable_prefix_caching="True",prefix_match_unit="None",'
                        'num_gpu_blocks_override="None",num_gpu_blocks="2177"} 1\n'
                        f"vllm:generation_tokens_total {output}\n"
                        f"vllm:request_success_total {owner.finished}\n").encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                events = [{"choices": [{"text": "x", "finish_reason": None}]},
                          {"choices": [{"text": "", "finish_reason": "length"}]},
                          {"choices": [], "usage": {"prompt_tokens": 4 if owner.mode == "wrong-count" else 3,
                           "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}}]
                body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                if owner.mode != "missing-done":
                    body += "data: [DONE]\n\n"
                owner.finished += 1
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

    def run_probe(self):
        return study.generate(f"http://127.0.0.1:{self.server.server_port}", "test",
                              {"prompt": [1, 2, 3]}, [1, 2, 3], "test", 768, completion=True)

    def test_complete_response_has_attributable_counts(self):
        row = self.run_probe()
        self.assertTrue(row["isolated"])
        self.assertEqual(row["prompt_tokens"], 3)
        self.assertEqual(row["cached_tokens"], 0)

    def test_missing_done_fails_even_with_usage_and_finish(self):
        self.mode = "missing-done"
        with self.assertRaisesRegex(RuntimeError, "Incomplete response"):
            self.run_probe()

    def test_count_mismatch_invalidates_token_boundary_assumption(self):
        self.mode = "wrong-count"
        with self.assertRaisesRegex(RuntimeError, "prompt counts differ"):
            self.run_probe()

    def test_other_engine_work_invalidates_attribution(self):
        self.mode = "other-work"
        with self.assertRaisesRegex(RuntimeError, "Unexpected engine work"):
            self.run_probe()

    def test_unrestarted_baseline_cannot_be_measured_as_tail128(self):
        base = f"http://127.0.0.1:{self.server.server_port}"
        self.assertEqual(study.checked_cache_config(base, "baseline")["prefix_match_unit"], "None")
        with self.assertRaisesRegex(RuntimeError, "Effective engine cache configuration"):
            study.checked_cache_config(base, "tail128")
        self.assertEqual(self.finished, 0)


if __name__ == "__main__":
    unittest.main()
