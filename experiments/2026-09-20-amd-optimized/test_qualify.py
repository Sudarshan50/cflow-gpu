"""Real loopback HTTP tests: partial SSE must never count as benchmark success."""
import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location("k3_qualify", Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class StreamQualificationTest(unittest.TestCase):
    def setUp(self):
        self.mode = "complete"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                events = [{"choices": [{"delta": {"content": "one token"}, "finish_reason": None}]}]
                if owner.mode == "error":
                    events.append({"error": {"message": "synthetic failure"}})
                else:
                    if owner.mode != "missing-finish":
                        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                    if owner.mode != "missing-usage":
                        events.append({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})
                payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                if owner.mode != "missing-done":
                    payload += "data: [DONE]\n\n"
                body = payload.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_complete_stream_preserves_usage_and_meaningful_ttft(self):
        row, text, _ = qualify.submit(self.base, {"stream": True})
        self.assertEqual(text, "one token")
        self.assertEqual(row["usage"]["completion_tokens"], 2)
        self.assertEqual(row["finishes"], ["stop"])
        self.assertIsNotNone(row["ttft"])

    def test_truncated_or_error_stream_is_never_success(self):
        for mode in ("missing-finish", "missing-usage", "missing-done", "error"):
            with self.subTest(mode=mode):
                self.mode = mode
                with self.assertRaises(RuntimeError):
                    qualify.submit(self.base, {"stream": True})


if __name__ == "__main__":
    unittest.main()
