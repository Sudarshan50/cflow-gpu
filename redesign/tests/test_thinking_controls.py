"""Client reasoning intent must reach K3's native tokenizer controls."""

import copy
import unittest

from redesign.gateway.media import normalize_payload


class ThinkingControlsTest(unittest.TestCase):
    def test_explicit_effort_reaches_native_template(self):
        for field in ("reasoning_effort", "thinking_effort"):
            for value, expected in (("low", "low"), ("medium", "high"), ("max", "max")):
                with self.subTest(field=field, value=value):
                    payload = normalize_payload({field: value})
                    self.assertEqual(payload["chat_template_kwargs"]["thinking_effort"], expected)

    def test_none_and_off_disable_thinking_instead_of_only_renaming_effort(self):
        for value in ("none", "off"):
            payload = normalize_payload({"reasoning_effort": value})
            self.assertIs(payload["chat_template_kwargs"]["thinking"], False)

    def test_explicit_native_settings_win(self):
        payload = normalize_payload({
            "reasoning_effort": "low",
            "chat_template_kwargs": {"thinking": True, "thinking_effort": "max"},
        })
        self.assertEqual(payload["chat_template_kwargs"], {"thinking": True, "thinking_effort": "max"})
        payload = normalize_payload({"reasoning_effort": "none", "chat_template_kwargs": {"thinking": True}})
        self.assertIs(payload["chat_template_kwargs"]["thinking"], True)

    def test_enable_thinking_alias_reaches_native_boolean(self):
        payload = normalize_payload({"chat_template_kwargs": {"enable_thinking": False}})
        self.assertIs(payload["chat_template_kwargs"]["thinking"], False)

    def test_extra_body_effort_and_template_precedence_survive_two_hops(self):
        original = {
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking": False, "unrelated": 7},
            "extra_body": {"reasoning_effort": "low", "chat_template_kwargs": {"another": 8}},
        }
        once = normalize_payload(copy.deepcopy(original))
        self.assertEqual(once["extra_body"]["chat_template_kwargs"], {
            "thinking": False, "thinking_effort": "low", "unrelated": 7, "another": 8,
        })
        self.assertEqual(normalize_payload(copy.deepcopy(once)), once)

    def test_missing_controls_do_not_change_the_model_default(self):
        original = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(normalize_payload(copy.deepcopy(original)), original)

    def test_invalid_template_is_left_for_upstream_validation(self):
        self.assertEqual(normalize_payload({"reasoning_effort": "low", "chat_template_kwargs": "invalid"})[
            "chat_template_kwargs"], "invalid")


if __name__ == "__main__":
    unittest.main()
