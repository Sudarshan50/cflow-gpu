"""Exercise installed LiteLLM against a Chat-only stub, without GPU requests."""

from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from redesign.tenancy.render_config import render


@unittest.skipUnless(importlib.util.find_spec("litellm"), "requires installed proxy LiteLLM")
class ResponsesBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import litellm
        import yaml
        from redesign.tenancy.callback import K3TenancyCallback

        self.requests = []
        requests = self.requests

        class ChatOnlyHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, body))
                usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "total_tokens": 102,
                    "prompt_tokens_details": {"cached_tokens": 64},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                }
                base = {"id": "chatcmpl-offline", "created": 1, "model": "FW-Kimi-K3"}
                if self.path != "/v1/chat/completions":
                    status, mime = 404, "application/json"
                    raw = json.dumps({"error": {"message": "not found", "type": "invalid_request"}}).encode()
                elif body.get("stream"):
                    status, mime = 200, "text/event-stream"
                    chunks = [
                        {**base, "object": "chat.completion.chunk", "choices": [
                            {"index": 0, "delta": {"role": "assistant", "reasoning": "why"}, "finish_reason": None}]},
                        {**base, "object": "chat.completion.chunk", "choices": [
                            {"index": 0, "delta": {"content": "OK"}, "finish_reason": None}]},
                        {**base, "object": "chat.completion.chunk", "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    ]
                    if body.get("stream_options", {}).get("include_usage"):
                        chunks.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
                    raw = ("".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
                else:
                    status, mime = 200, "application/json"
                    raw = json.dumps({**base, "object": "chat.completion", "usage": usage, "choices": [
                        {"index": 0, "message": {
                            "role": "assistant", "content": "OK", "reasoning": "why",
                        }, "finish_reason": "stop"}]}).encode()
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatOnlyHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.litellm = litellm
        self.callback = K3TenancyCallback(cache_mode="off")
        for context in (patch.object(litellm, "callbacks", [self.callback]), patch.object(litellm, "cache", None)):
            context.start()
            self.addCleanup(context.stop)
        config = yaml.safe_load(render())
        self.models = config["model_list"]
        for model in self.models:
            model["litellm_params"]["api_base"] = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.router = litellm.Router(model_list=self.models, num_retries=0)

    async def test_responses_aliases_use_chat_and_preserve_limits_and_usage(self):
        from litellm.proxy._types import UserAPIKeyAuth

        for model in self.models:
            payload = {"model": model["model_name"], "input": "Reply OK", "max_output_tokens": 16}
            await self.callback.async_pre_call_hook(UserAPIKeyAuth(api_key="offline"), None, payload, "aresponses")
            result = await self.router.aresponses(**payload)
            path, body = self.requests[-1]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["max_tokens"], 16)
            self.assertEqual(body["messages"], [{"role": "user", "content": "Reply OK"}])
            self.assertNotIn("use_chat_completions_api", body)
            self.assertEqual(result.usage.input_tokens_details.cached_tokens, 64)
            self.assertEqual(result.usage.output_tokens_details.reasoning_tokens, 1)

    async def test_stream_has_terminal_response_and_cached_usage(self):
        result = await self.router.aresponses(
            model="FW-Kimi-K3", input="Reply OK", max_output_tokens=16, stream=True,
        )
        events = [event async for event in result]
        terminal = [event for event in events if event.type == "response.completed"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].response.usage.input_tokens_details.cached_tokens, 64)
        self.assertEqual(
            terminal[0].response.usage.output_tokens_details.reasoning_tokens, 1
        )
        self.assertEqual(self.requests[-1][0], "/v1/chat/completions")

    async def test_chat_still_uses_chat_without_leaking_bridge_control(self):
        result = await self.router.acompletion(
            model="FW-Kimi-K3", messages=[{"role": "user", "content": "Reply OK"}], max_tokens=16,
        )
        await self.callback.async_post_call_success_hook({}, None, result)
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(result.choices[0].message.reasoning_content, "why")
        self.assertEqual(result.choices[0].message.reasoning, "why")
        self.assertEqual(result.usage.completion_tokens_details.reasoning_tokens, 1)
        self.assertEqual(self.requests[-1][0], "/v1/chat/completions")
        self.assertNotIn("use_chat_completions_api", self.requests[-1][1])


if __name__ == "__main__":
    unittest.main()
