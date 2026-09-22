"""LiteLLM tenancy policy: class, clamp, P1 off-box only when configured."""

from __future__ import annotations

import unittest

from redesign.tenancy.policy import OFFBOX_MODEL, TenancyPolicy
from redesign.tenancy.render_config import render


def _chat(text: str, max_tokens: int = 128_000, tools=None) -> dict:
    return {
        "model": "FW-Kimi-K3",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "tools": tools,
    }


class TenancyPolicyTest(unittest.TestCase):
    def test_short_chat_stays_local_without_an_offbox_url(self):
        policy = TenancyPolicy(offbox_configured=False)
        payload = _chat("hi")
        decision = policy.apply(payload, "FW-Kimi-K3")
        self.assertEqual(decision.traffic_class, "P1-short-chat")
        self.assertEqual(decision.priority, 1)
        self.assertEqual(payload["model"], "FW-Kimi-K3")
        self.assertLessEqual(payload["max_tokens"], 4096)

    def test_short_chat_routes_off_box_when_configured(self):
        policy = TenancyPolicy(offbox_configured=True)
        payload = _chat("hi")
        decision = policy.apply(payload, "FW-Kimi-K3")
        self.assertTrue(decision.routed_off_box)
        self.assertEqual(payload["model"], OFFBOX_MODEL)

    def test_tools_are_agentic_not_short_chat(self):
        policy = TenancyPolicy(offbox_configured=True)
        payload = _chat("hi", tools=[{"type": "function"}])
        decision = policy.apply(payload, "FW-Kimi-K3")
        self.assertEqual(decision.traffic_class, "P2-agentic")
        self.assertFalse(decision.routed_off_box)
        self.assertLessEqual(payload["max_tokens"], 1536)

    def test_an_image_turn_is_agentic_and_billed_above_one_image_token(self):
        policy = TenancyPolicy()
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]}],
        }
        decision = policy.decide(payload, "FW-Kimi-K3")
        self.assertEqual(decision.traffic_class, "P2-agentic")

    def test_prompt_past_the_window_is_rejected(self):
        policy = TenancyPolicy(max_model_len=1000)
        decision = policy.decide(_chat("x" * 8000, max_tokens=16), "FW-Kimi-K3")
        self.assertTrue(decision.reject)


class RenderConfigTest(unittest.TestCase):
    def test_offbox_is_absent_until_a_url_exists(self):
        self.assertNotIn("p1-offbox", render(""))

    def test_offbox_and_fallback_appear_when_a_url_exists(self):
        text = render("https://example.invalid/v1", "k", "other")
        self.assertIn("p1-offbox", text)
        self.assertIn("https://example.invalid/v1", text)
        self.assertIn("fallbacks:", text)

    def test_redis_is_absent_until_a_host_exists(self):
        text = render("")
        self.assertNotIn("coordination_redis", text)
        self.assertNotIn("type: redis", text)

    def test_vision_and_tools_are_declared(self):
        text = render("")
        self.assertIn("supports_vision: true", text)
        self.assertIn("supports_function_calling: true", text)
        self.assertIn("supports_tool_choice: true", text)

    def test_redis_is_wired_when_a_host_exists(self):
        text = render(redis_host="127.0.0.1")
        self.assertIn("type: redis", text)
        self.assertIn("coordination_redis", text)
        self.assertIn("enable_redis_auth_cache: true", text)
        self.assertIn("use_redis_transaction_buffer: true", text)
        self.assertIn("redis_host: os.environ/REDIS_HOST", text)


if __name__ == "__main__":
    unittest.main()
