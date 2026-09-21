"""Client reasoning intent must reach K3's native tokenizer controls."""

import copy
import unittest
from unittest.mock import patch

from redesign.gateway.media import normalize_controls, normalize_payload


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

    def test_controls_only_does_not_download_validate_or_rewrite_messages(self):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://media.invalid/image?token=secret"}},
            {"type": "input_audio", "input_audio": {"data": "invalid"}},
        ]}]
        payload = {
            "reasoning_effort": "off", "messages": messages,
            "tools": [{"type": "function", "function": {"name": name}} for name in ("z", "a")],
        }
        with patch("redesign.gateway.media._download", side_effect=AssertionError("download")), \
             patch("redesign.gateway.media._validate_image", side_effect=AssertionError("decode")), \
             patch("redesign.gateway.media._media_worker", side_effect=AssertionError("media worker")):
            result = normalize_controls(payload)
            self.assertIs(result, payload)
            self.assertIs(result["messages"], messages)
            self.assertIs(result["chat_template_kwargs"]["thinking"], False)
            self.assertEqual([tool["function"]["name"] for tool in result["tools"]], ["a", "z"])
            self.assertEqual(normalize_controls(copy.deepcopy(result)), result)

    def test_controls_are_idempotent_across_effort_and_template_precedence(self):
        for effort in (None, "none", "off", "low", "medium", "max", "invalid"):
            for template in (
                {}, {"thinking": True}, {"thinking": False}, {"enable_thinking": False},
                {"thinking_effort": "max"}, {"reasoning_effort": "none"},
                {"thinking_effort": "invalid"},
            ):
                for extra in (
                    {}, {"reasoning_effort": "low", "chat_template_kwargs": {"other": 1}},
                    {"reasoning_effort": "invalid"},
                    {"chat_template_kwargs": {"thinking_effort": "invalid"}},
                    {"chat_template_kwargs": {"reasoning_effort": "invalid"}},
                ):
                    with self.subTest(effort=effort, template=template, extra=extra):
                        payload = {"chat_template_kwargs": copy.deepcopy(template), "extra_body": copy.deepcopy(extra)}
                        if effort is not None:
                            payload["reasoning_effort"] = effort
                        once = normalize_controls(payload)
                        self.assertEqual(normalize_controls(copy.deepcopy(once)), once)

    def test_controls_only_preserves_unspecified_reasoning_default(self):
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        self.assertEqual(normalize_controls(copy.deepcopy(payload)), payload)


if __name__ == "__main__":
    unittest.main()
