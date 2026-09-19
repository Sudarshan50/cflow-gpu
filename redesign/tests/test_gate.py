"""Gate checks against a scriptable mock model.

The important tests are the negative ones: a gate that cannot fail a broken
build is worse than no gate, because it licenses a bad deploy.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from redesign.gate.checks import (
    CleanText,
    ExactAnswer,
    GreedyDeterminism,
    NeedleRetrieval,
    NoDegenerateRepetition,
    StructuredOutput,
    build_registry,
)
from redesign.gate.client import CompletionError, GateClient


class ScriptedModel(BaseHTTPRequestHandler):
    """Replies from a queue, or echoes a canned good answer."""

    protocol_version = "HTTP/1.1"
    script: list[str] = []
    behaviour = "good"

    def log_message(self, *args) -> None:
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = payload["messages"][0]["content"]
        text = ScriptedModel.script.pop(0) if ScriptedModel.script else self._answer(prompt)
        body = json.dumps({
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 8},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _answer(self, prompt: str) -> str:
        if "maintenance code" in prompt:
            return "PHOENIX-4471" if self.behaviour == "good" else "I could not find it."
        if "prime numbers" in prompt:
            return "2, 3, 5, 7, 11, 13, 17, 19"
        if "kilometres in total" in prompt:
            return "60 + 80 + 100 = 240"
        if "Grons" in prompt:
            return "There are at least 12"
        if "JSON object" in prompt:
            return '{"city": "Delhi", "count": 3}'
        if "printing press" in prompt:
            return "Gutenberg introduced movable type. It spread quickly. Literacy rose."
        return "ready"


class GateTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedModel)
        cls.server.daemon_threads = True
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = GateClient(f"http://127.0.0.1:{cls.server.server_address[1]}")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        ScriptedModel.script = []
        ScriptedModel.behaviour = "good"


class HealthyModelTest(GateTestCase):
    def test_determinism_passes_when_output_is_stable(self):
        self.assertTrue(GreedyDeterminism().run(self.client).passed)

    def test_arithmetic_is_extracted_from_worked_reasoning(self):
        check = ExactAnswer(prompt="kilometres in total", expected="240", label="a")
        self.assertTrue(check.run(self.client).passed)

    def test_structured_output_passes(self):
        self.assertTrue(StructuredOutput().run(self.client).passed)

    def test_clean_text_passes(self):
        self.assertTrue(CleanText().run(self.client).passed)

    def test_needle_is_retrieved(self):
        self.assertTrue(NeedleRetrieval(8_000, 0.5).run(self.client).passed)


class BrokenModelTest(GateTestCase):
    """Each of these is a failure mode quantised KV actually produces."""

    def test_nondeterminism_is_caught(self):
        ScriptedModel.script = ["2, 3, 5", "2, 3, 7", "2, 3, 5"]
        result = GreedyDeterminism().run(self.client)
        self.assertFalse(result.passed)
        self.assertIn("distinct outputs", result.detail)

    def test_degenerate_repetition_is_caught(self):
        ScriptedModel.script = ["the " * 60]
        result = NoDegenerateRepetition().run(self.client)
        self.assertFalse(result.passed)
        self.assertIn("repeats", result.detail)

    def test_lost_needle_is_caught(self):
        ScriptedModel.behaviour = "broken"
        result = NeedleRetrieval(8_000, 0.5).run(self.client)
        self.assertFalse(result.passed)
        self.assertIn("not retrieved", result.detail)

    def test_a_confidently_wrong_answer_is_caught(self):
        """The silent-garbage signature: fluent, plausible, wrong."""
        ScriptedModel.script = ["Adding them gives 260 kilometres in total."]
        check = ExactAnswer(prompt="kilometres in total", expected="240", label="a")
        result = check.run(self.client)
        self.assertFalse(result.passed)

    def test_empty_output_is_caught(self):
        ScriptedModel.script = ["   "]
        self.assertFalse(CleanText().run(self.client).passed)

    def test_malformed_json_is_caught(self):
        ScriptedModel.script = ['{"city": "Delhi", "count": }']
        self.assertFalse(StructuredOutput().run(self.client).passed)

    def test_right_shape_wrong_values_is_caught(self):
        ScriptedModel.script = ['{"city": "Mumbai", "count": 3}']
        result = StructuredOutput().run(self.client)
        self.assertFalse(result.passed)
        self.assertIn("wrong values", result.detail)


class RegistryTest(unittest.TestCase):
    def test_all_three_tiers_are_represented(self):
        self.assertEqual({c.tier for c in build_registry()}, {1, 2, 3})

    def test_long_context_sizes_expand_across_depths(self):
        registry = build_registry(long_context_tokens=(32_000, 128_000))
        needles = [c for c in registry if isinstance(c, NeedleRetrieval)]
        self.assertEqual(len(needles), 6)

    def test_check_names_are_unique(self):
        names = [c.name for c in build_registry()]
        self.assertEqual(len(names), len(set(names)))

    def test_the_needle_sits_at_the_requested_depth(self):
        check = NeedleRetrieval(8_000, 0.9)
        prompt = check._build_prompt()
        position = prompt.index(check.CODE) / len(prompt)
        self.assertGreater(position, 0.7)


class ClientTest(unittest.TestCase):
    def test_an_unreachable_engine_raises_rather_than_passing(self):
        with self.assertRaises(CompletionError):
            GateClient("http://127.0.0.1:1", timeout=2).complete("hi")


if __name__ == "__main__":
    unittest.main()
