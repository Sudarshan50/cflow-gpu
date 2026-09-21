"""Required image parts are validated and preserved; invalid media is an error."""

from __future__ import annotations

import base64
import unittest

from redesign.gateway.media import MediaValidationError, PNG_MAGIC, normalize_payload

PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


def _payload(part: dict) -> dict:
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, part]}]}


class MediaNormalizeTest(unittest.TestCase):
    def test_valid_png_stays_an_image(self):
        out = normalize_payload(_payload({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
        }))
        part = out["messages"][0]["content"][1]
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))
        raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        self.assertTrue(raw.startswith(PNG_MAGIC))

    def test_garbage_image_is_an_explicit_part_indexed_error(self):
        junk = base64.b64encode(b"not-an-image").decode()
        payload = _payload({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{junk}"},
        })
        with self.assertRaises(MediaValidationError) as caught:
            normalize_payload(payload)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(caught.exception.part_index, 1)
        self.assertIn("messages[0].content[1]", str(caught.exception))
        self.assertNotIn(junk, str(caught.exception))
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0]["text"], "hi")
        self.assertEqual(parts[1]["type"], "image_url")

    def test_anthropic_image_block_is_accepted(self):
        out = normalize_payload(_payload({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64},
        }))
        self.assertEqual(out["messages"][0]["content"][1]["type"], "image_url")

    def test_string_image_url_is_accepted(self):
        out = normalize_payload(_payload({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{PNG_B64}",
        }))
        self.assertEqual(out["messages"][0]["content"][1]["type"], "image_url")

    def test_cache_salt_is_stripped_so_a_portal_pool_can_share_prefixes(self):
        payload = {
            "cache_salt": "per-user-1",
            "priority": 0,
            "extra_body": {"prompt_cache_key": "u2", "kv_cache_salt": "u3", "priority": 1},
            "messages": [{"role": "user", "content": "hi"}],
        }
        out = normalize_payload(payload)
        self.assertNotIn("cache_salt", out)
        self.assertNotIn("priority", out)
        self.assertNotIn("prompt_cache_key", out["extra_body"])
        self.assertNotIn("kv_cache_salt", out["extra_body"])
        self.assertNotIn("priority", out["extra_body"])

    def test_tool_array_order_is_canonicalized(self):
        """K3 renders tools before all content and preserves array order, so a
        reshuffle costs the whole prompt: 1142 shared tokens drop to 35."""
        def tools(*names):
            return [{"type": "function", "function": {"name": n}} for n in names]

        forward = normalize_payload({"tools": tools("read", "run", "write")})
        shuffled = normalize_payload({"tools": tools("write", "read", "run")})
        self.assertEqual(forward["tools"], shuffled["tools"])
        self.assertEqual(
            [t["function"]["name"] for t in forward["tools"]],
            ["read", "run", "write"],
        )

    def test_legacy_functions_are_canonicalized_too(self):
        out = normalize_payload({"functions": [{"name": "b"}, {"name": "a"}]})
        self.assertEqual([f["name"] for f in out["functions"]], ["a", "b"])

    def test_duplicate_tool_names_keep_the_caller_order(self):
        """Names no longer identify the entries, so any sort is a guess."""
        original = [
            {"type": "function", "function": {"name": "dup", "description": "first"}},
            {"type": "function", "function": {"name": "dup", "description": "second"}},
        ]
        out = normalize_payload({"tools": list(original)})
        self.assertEqual(out["tools"], original)

    def test_an_unnamed_tool_leaves_the_array_untouched(self):
        original = [{"type": "custom"}, {"type": "function", "function": {"name": "a"}}]
        out = normalize_payload({"tools": list(original)})
        self.assertEqual(out["tools"], original)

    def test_openai_medium_thinking_is_mapped_to_k3_high(self):
        out = normalize_payload({
            "reasoning_effort": "medium",
            "extra_body": {"thinking_effort": "medium"},
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(out["reasoning_effort"], "high")
        self.assertEqual(out["extra_body"]["thinking_effort"], "high")

    def test_a_valid_in_budget_png_is_not_reencoded(self):
        """Re-encoding changes bytes and busts the vision-token prefix."""
        url = f"data:image/png;base64,{PNG_B64}"
        out = normalize_payload(_payload({
            "type": "image_url",
            "image_url": {"url": url},
        }))
        self.assertEqual(out["messages"][0]["content"][1]["image_url"]["url"], url)
