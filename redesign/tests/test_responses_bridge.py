"""Installed LiteLLM's real proxy/Router/bridge lifecycle, with mock HTTP only."""

from __future__ import annotations

import asyncio
import base64
import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from redesign.tenancy.render_config import render

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="


def responses(**kwargs):
    return {"model": "FW-Kimi-K3", "input": "Reply OK", **kwargs}


def function(name="lookup", **kwargs):
    return {"type": "function", "name": name, "parameters": {"type": "object"}, **kwargs}


@unittest.skipUnless(importlib.util.find_spec("litellm"), "requires installed proxy LiteLLM")
class ResponsesBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import httpx
        import litellm
        import yaml
        from openai import AsyncOpenAI
        from litellm.caching.caching import Cache
        from litellm.proxy import proxy_server
        from redesign.tenancy.callback import K3TenancyCallback
        from redesign.tenancy.policy import TenancyPolicy

        # Each test creates real Routers; their bound logging callbacks must
        # not accumulate in LiteLLM's secondary process-global callback lists.
        for name in ("input_callback", "success_callback", "failure_callback", "service_callback",
                     "_async_input_callback", "_async_success_callback", "_async_failure_callback"):
            context = patch.object(litellm, name, [])
            context.start()
            self.addCleanup(context.stop)
        self.litellm = litellm
        self.callback = K3TenancyCallback(policy=TenancyPolicy(), cache_mode="static", team_pool_ids=frozenset())
        self.cache = Cache(type="local", mode="default_off", supported_call_types=["acompletion"], ttl=300)
        self.requests, self.prepared, self.stages = [], [], []
        self.through_gateway = False
        self.gateway_results = []
        self.tool_response = False
        self.replace_inner_logging = False
        self.inner_logging = None
        self.models = yaml.safe_load(render())["model_list"]
        for model in self.models:
            model["litellm_params"].update(api_base="http://provider.invalid/v1", api_key="offline-provider")
        self.shared_session = await proxy_server._initialize_shared_aiohttp_session()
        self.addAsyncCleanup(self.shared_session.close)

        def transport(request):
            body = json.loads(request.content)
            self.requests.append((request.url.path, body))
            self.assertEqual(request.url.path, "/v1/chat/completions", "native Responses must never reach the gateway")
            if self.through_gateway:
                from redesign.tests.test_output_policy import relay_through_gateway
                result = relay_through_gateway(body)
                self.gateway_results.append(result)
                self.assertEqual(result.status, 200, result.error)
                body = result.forwarded
            usage = {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101,
                     "prompt_tokens_details": {"cached_tokens": 64}}
            base = {"id": f"chatcmpl-offline-{len(self.requests)}", "created": 1, "model": body["model"]}
            message = {"role": "assistant", "content": "OK"}
            finish = "stop"
            if self.tool_response:
                message = {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-offline", "type": "function",
                    "function": {"name": "lookup", "arguments": '{"x":"value"}'},
                }]}
                finish = "tool_calls"
            if not body.get("stream"):
                return httpx.Response(200, json={**base, "object": "chat.completion", "usage": usage, "choices": [
                    {"index": 0, "message": message, "finish_reason": finish},
                ]})
            delta = copy.deepcopy(message)
            if self.tool_response:
                delta["tool_calls"][0]["index"] = 0
            chunks = [
                {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": delta, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": {}, "finish_reason": finish}]},
            ]
            if body.get("stream_options", {}).get("include_usage"):
                chunks.append({**base, "object": "chat.completion.chunk", "choices": [], "usage": usage})
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n")

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.client = AsyncOpenAI(api_key="offline-provider", base_url="http://provider.invalid/v1", http_client=self.http)
        self.addAsyncCleanup(self.client.close)
        original_hook = self.callback.async_pre_call_deployment_hook

        async def deployment(kwargs, call_type):
            if call_type == "acompletion" and self.replace_inner_logging:
                kwargs["litellm_logging_obj"] = self.inner_logging
            before = self.snapshot(kwargs)
            stage = {"call_type": call_type, "before": before}
            self.stages.append(stage)
            result = await original_hook(kwargs, call_type)
            stage["after"] = self.snapshot(result)
            if call_type == "acompletion":
                # Router.aresponses does not select an AsyncOpenAI client for
                # the inner SDK call. Inject only the mock transport here.
                result["client"] = self.client
            return result

        for context in (
            patch.object(litellm, "callbacks", [self.callback]), patch.object(litellm, "cache", self.cache),
            patch.object(litellm, "drop_params", True),
            patch.object(proxy_server, "shared_aiohttp_session", self.shared_session),
            patch.object(self.callback, "async_pre_call_deployment_hook", side_effect=deployment),
        ):
            context.start()
            self.addCleanup(context.stop)

    @staticmethod
    def snapshot(data):
        # Copy only synthetic request fields, never the logging/client/session.
        snapshot = copy.deepcopy({k: data[k] for k in (
            "model", "api_base", "api_key", "messages", "input", "instructions", "tools",
            "max_output_tokens", "max_tokens", "max_completion_tokens", "response_format",
            "reasoning_effort", "cache", "caching", "cache_key", "metadata", "litellm_metadata",
        ) if k in data})
        snapshot["logging"] = data.get("litellm_logging_obj")
        snapshot["client"] = data.get("client")
        return snapshot

    async def prepare(self, payload, *, route="aresponses", key="offline-auth"):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
        from litellm.proxy.utils import ProxyLogging
        from starlette.requests import Request

        proxy = ProxyLogging(user_api_key_cache=self.litellm.DualCache())
        request = Request({"type": "http", "method": "POST", "path": "/v1/responses" if route == "aresponses" else "/v1/chat/completions",
                           "headers": [], "scheme": "http", "server": ("test.invalid", 80), "query_string": b""})
        with patch.object(proxy, "_init_response_taking_too_long_task"):
            data, _ = await ProxyBaseLLMRequestProcessing(copy.deepcopy(payload)).common_processing_pre_call_logic(
                request=request, general_settings={"always_include_stream_usage": True}, proxy_logging_obj=proxy,
                user_api_key_dict=UserAPIKeyAuth(api_key=key), proxy_config=None, route_type=route,
            )
        self.prepared.append(data)
        return data

    async def complete(self, payload=None, *, defaults=None, route="aresponses", key="offline-auth"):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.route_llm_request import route_request
        from litellm.proxy.utils import ProxyLogging

        data = await self.prepare(payload if payload is not None else responses(), route=route, key=key)
        models = copy.deepcopy(self.models)
        for model in models:
            model["litellm_params"].update(defaults or {})
        router = self.litellm.Router(model_list=models, num_retries=0)
        call = await route_request(data=data, llm_router=router, user_model=None, route_type=route,
                                   user_api_key_dict=UserAPIKeyAuth(api_key=key))
        result = await call
        if data.get("stream"):
            proxy = ProxyLogging(user_api_key_cache=self.litellm.DualCache())
            result = [chunk async for chunk in proxy.async_post_call_streaming_iterator_hook(
                response=result, user_api_key_dict=UserAPIKeyAuth(api_key=key), request_data=data,
            )]
        for _ in range(10):
            await asyncio.sleep(0)
        return result

    async def assert_bad_request(self, payload, *, defaults=None, early=True, route="aresponses"):
        before = len(self.requests), len(self.stages)
        with self.assertRaises(Exception) as error:
            await self.complete(payload, defaults=defaults, route=route)
        self.assertEqual(getattr(error.exception, "status_code", None), 400, repr(error.exception))
        self.assertEqual(len(self.requests), before[0])
        if early:
            self.assertEqual(len(self.stages), before[1], "reject before Router and native/session handlers")

    async def test_proxy_hook_only_captures_context_and_validates(self):
        from redesign.tenancy.cache_policy import CONTEXT_ATTR
        payload = responses(input="x" * 35000, instructions="Follow these instructions", tools=[function()], max_output_tokens=9000)
        with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
             patch("redesign.tenancy.callback.normalize_payload", side_effect=AssertionError("early normalization")) as normalize:
            data = await self.prepare(payload)
        apply.assert_not_called()
        normalize.assert_not_called()
        for key, value in payload.items():
            self.assertEqual(data[key], value)
        self.assertNotIn("max_tokens", data)
        self.assertIsNone(getattr(data["litellm_logging_obj"], CONTEXT_ATTR))
        self.assertTrue(data["cache"]["no-store"])

    async def test_responses_aliases_use_chat_and_preserve_limits_and_usage(self):
        for model in self.models:
            self.assertIs(model["litellm_params"]["use_chat_completions_api"], True)
            result = await self.complete(responses(model=model["model_name"], max_output_tokens=16))
            path, body = self.requests[-1]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(body["max_tokens"], 16)
            self.assertEqual(body["messages"], [{"role": "user", "content": "Reply OK"}])
            for field in ("max_output_tokens", "use_chat_completions_api", "_skip_responses_api_bridge", "cache_key", "litellm_params"):
                self.assertNotIn(field, body)
            self.assertEqual(result.usage.input_tokens_details.cached_tokens, 64)

    async def test_policy_runs_once_after_conversion_and_defaults(self):
        from redesign.gateway.media import normalize_payload
        with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
             patch.object(self.callback.policy.estimator, "estimate", wraps=self.callback.policy.estimator.estimate) as estimate, \
             patch("redesign.tenancy.callback.normalize_payload", wraps=normalize_payload) as normalize:
            await self.complete(responses(max_output_tokens=9000), defaults={"instructions": "x" * 35000})
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(estimate.call_count, 1)
        self.assertEqual(normalize.call_count, 1)
        self.assertEqual([stage["call_type"] for stage in self.stages], ["aresponses", "acompletion"])
        outer, inner = self.stages
        self.assertNotIn("messages", outer["after"])
        self.assertIs(inner["before"]["logging"], self.prepared[0]["litellm_logging_obj"])
        self.assertEqual(inner["before"]["logging"].call_type, "aresponses")
        self.assertEqual(inner["before"]["max_tokens"], 9000)
        self.assertEqual(inner["after"]["max_tokens"], 8192)
        self.assertEqual(inner["after"]["litellm_metadata"]["k3_class"], "P0-interactive")
        self.assertEqual(self.requests[-1][1]["messages"][0], {"role": "system", "content": "x" * 35000})

    async def test_responses_limits_are_authoritative_and_defaults_survive(self):
        for payload, defaults, expected in (
            ({"max_output_tokens": 1}, {}, 1),
            ({"max_output_tokens": 16, "max_completion_tokens": 2000, "max_tokens": 9000}, {}, 16),
            ({"max_output_tokens": 16}, {"max_completion_tokens": 2000, "max_tokens": 9000}, 16),
            ({"max_output_tokens": 16}, {"max_output_tokens": 9000}, 16),
            ({"max_output_tokens": 1000, "max_completion_tokens": 16}, {}, 1000),
            ({"max_output_tokens": 9000}, {}, 4096),
            ({}, {"max_output_tokens": 19}, 19),
            ({}, {"max_tokens": 20}, 20),
            ({}, {"max_completion_tokens": 21, "max_tokens": 1000}, 21),
            ({"max_output_tokens": None}, {"max_tokens": 22}, 22),
        ):
            with self.subTest(payload=payload, defaults=defaults):
                with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply:
                    await self.complete(responses(**payload), defaults=defaults)
                self.assertEqual(apply.call_count, 1)
                body = self.requests[-1][1]
                self.assertEqual(body["max_tokens"], expected)
                if "max_completion_tokens" in body:
                    self.assertEqual(body["max_completion_tokens"], expected)
                if payload.get("max_output_tokens"):
                    self.assertLessEqual(expected, payload["max_output_tokens"])
                self.assertNotIn("max_output_tokens", body)

    async def test_8192_tool_and_forced_file_budgets_survive_the_entire_pipeline(self):
        from redesign.tests.test_output_policy import file_tool
        self.through_gateway = True
        for route in ("acompletion", "aresponses"):
            for forced in (False, True):
                with self.subTest(route=route, forced=forced):
                    tool = file_tool()
                    if route == "aresponses":
                        payload = responses(tools=[{"type": "function", **tool["function"]}], max_output_tokens=8192)
                        if forced:
                            payload["tool_choice"] = {"type": "function", "name": "write_file"}
                    else:
                        payload = {"model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "Write the entire file"}],
                                   "tools": [tool], "max_completion_tokens": 8192}
                        if forced:
                            payload["tool_choice"] = {"type": "function", "function": {"name": "write_file"}}
                    await self.complete(payload, route=route)
                    metadata = self.stages[-1]["after"]["litellm_metadata" if route == "aresponses" else "metadata"]
                    self.assertEqual(metadata["k3_output_policy"], {
                        "stage": "tenancy", "requested_max_tokens": 8192, "granted_max_tokens": 8192,
                        "default_output_tokens": 2048, "default_applied": False,
                        "clamp_reason": "unchanged", "traffic_class": "P2-agentic",
                    })
                    gateway = self.gateway_results[-1]
                    self.assertEqual(gateway.forwarded["max_tokens"], 8192)
                    self.assertEqual(gateway.record.traffic_class, "P2-agentic")
                    self.assertEqual(gateway.during.context_tokens_in_flight, gateway.record.prompt_tokens + 8192)
                    self.assertEqual(gateway.after.context_tokens_in_flight, 0)
                    if forced:
                        self.assertEqual(gateway.forwarded["tool_choice"], {"type": "function", "function": {"name": "write_file"}})

    async def test_policy_defaults_and_deployment_limits_have_truthful_hop_diagnostics(self):
        self.through_gateway = True
        for route in ("acompletion", "aresponses"):
            for caller, defaults, requested, granted, reason in (
                ({}, {}, None, 2048, "default"),
                ({}, {"max_tokens": 8192}, 8192, 8192, "unchanged"),
                ({}, {"max_output_tokens": 19}, 19, 19, "unchanged"),
                ({}, {"max_completion_tokens": 32768}, 32768, 16384, "class_ceiling"),
                ({"max_output_tokens": 1}, {"max_tokens": 8192}, 1, 1, "unchanged"),
                ({"max_completion_tokens": 16}, {"max_tokens": 8192}, 16, 16, "unchanged"),
            ):
                with self.subTest(route=route, caller=caller, defaults=defaults):
                    tool = function()
                    payload = responses(tools=[tool], **caller) if route == "aresponses" else {
                        "model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "private prompt sentinel"}],
                        "tools": [{"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}}],
                        "metadata": {"k3_output_policy": {"requested_max_tokens": 999999}}, **caller,
                    }
                    await self.complete(payload, defaults=defaults, route=route)
                    metadata = self.stages[-1]["after"]["litellm_metadata" if route == "aresponses" else "metadata"]
                    diagnostic = metadata["k3_output_policy"]
                    self.assertEqual(diagnostic["requested_max_tokens"], requested)
                    self.assertEqual(diagnostic["granted_max_tokens"], granted)
                    self.assertEqual(diagnostic["default_output_tokens"], 2048)
                    self.assertEqual(diagnostic["default_applied"], requested is None)
                    self.assertEqual(diagnostic["clamp_reason"], reason)
                    self.assertNotIn("private prompt sentinel", json.dumps(diagnostic))
                    gateway = self.gateway_results[-1]
                    self.assertEqual(gateway.record.requested_max_tokens, granted)
                    self.assertEqual(gateway.forwarded["max_tokens"], granted)
                    self.assertEqual(gateway.headers["x-k3-max-tokens-requested"], str(granted))
                    self.assertEqual(gateway.headers["x-k3-output-default-applied"], "false")
                    self.assertEqual(gateway.record.clamp_reason, "unchanged")

    async def test_none_disables_tools_but_not_images_after_responses_conversion(self):
        self.through_gateway = True
        for payload, expected_class, expected_limit in (
            (responses(tools=[function()], tool_choice="none"), "P1-short-chat", 1024),
            (responses(input="x" * 35000, tools=[function()], tool_choice="none"), "P0-interactive", 2048),
            (responses(tools=[function()], tool_choice=None), "P2-agentic", 2048),
            (responses(input=[{"role": "user", "content": [{"type": "input_image", "image_url": PNG}]}],
                       tool_choice="none"), "P2-agentic", 2048),
            (responses(input=[{"role": "user", "content": [{"type": "input_image", "image_url": PNG}]}],
                       tool_choice="none", max_output_tokens=32768), "P2-agentic", 16384),
        ):
            with self.subTest(expected_class=expected_class, expected_limit=expected_limit):
                await self.complete(payload)
                metadata = self.stages[-1]["after"]["litellm_metadata"]
                self.assertEqual(metadata["k3_class"], expected_class)
                gateway = self.gateway_results[-1]
                self.assertEqual(gateway.record.traffic_class, expected_class)
                self.assertEqual(gateway.forwarded["max_tokens"], expected_limit)
                if payload.get("tools"):
                    self.assertTrue(gateway.record.has_tools)
                    self.assertTrue(gateway.forwarded["tools"])

    async def test_invalid_chat_and_responses_limits_are_400_before_router_or_normalization(self):
        with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
             patch("redesign.tenancy.callback.normalize_payload", side_effect=AssertionError("invalid limit reached media")) as normalize:
            for route in ("acompletion", "aresponses"):
                for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                    for value in (True, False, -1, 0, "16", 1.5):
                        with self.subTest(route=route, field=field, value=value):
                            payload = responses(**{field: value}) if route == "aresponses" else {
                                "model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "hi"}], field: value,
                            }
                            await self.assert_bad_request(payload, route=route)
            for route in ("acompletion", "aresponses"):
                for field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
                    with self.subTest(route=route, deployment_field=field):
                        payload = responses() if route == "aresponses" else {
                            "model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "hi"}],
                        }
                        await self.assert_bad_request(payload, route=route, defaults={field: -1}, early=False)
            apply.assert_not_called()
            normalize.assert_not_called()

    async def test_instructions_tools_images_and_json_schema_are_preserved(self):
        schema = {"type": "object", "properties": {
            "previous_response_id": {"type": "string"}, "cache_key": {"type": "string"},
            "store": {"type": "boolean"}, "max_output_tokens": {"type": "integer", "enum": [16, 1]},
        }, "required": ["store", "cache_key"], "additionalProperties": False}
        payload = responses(instructions="Keep these instructions", input=[
            {"role": "developer", "content": "Keep this message"},
            {"role": "user", "content": [{"type": "input_text", "text": "Describe"},
                                            {"type": "input_image", "image_url": PNG}]},
        ], tools=[function("z_tool", parameters=schema, strict=True)],
            tool_choice={"type": "function", "name": "lookup"}, parallel_tool_calls=False,
            text={"format": {"type": "json_schema", "name": "answer", "schema": schema, "strict": True}},
            max_output_tokens=9000)
        original = copy.deepcopy(payload)
        await self.complete(payload, defaults={"tools": [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]})
        self.assertEqual(payload, original)
        body = self.requests[-1][1]
        self.assertEqual(self.stages[-1]["after"]["messages"][:2], [
            {"role": "system", "content": payload["instructions"]}, payload["input"][0],
        ])
        # The installed OpenAI Chat adapter maps developer to system for K3.
        # Both instruction texts and their order survive the provider adapter.
        self.assertEqual(body["messages"][:2], [
            {"role": "system", "content": payload["instructions"]},
            {"role": "system", "content": "Keep this message"},
        ])
        self.assertEqual(body["messages"][2]["content"][0], {"type": "text", "text": "Describe"})
        self.assertEqual(body["messages"][2]["content"][1]["type"], "image_url")
        self.assertTrue(body["messages"][2]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual([tool["function"]["name"] for tool in body["tools"]], ["lookup", "z_tool"])
        self.assertEqual(body["tools"][1]["function"]["parameters"], schema)
        self.assertIs(body["tools"][1]["function"]["strict"], True)
        self.assertEqual(body["tool_choice"], {"type": "function", "function": {"name": "lookup"}})
        self.assertIs(body["parallel_tool_calls"], False)
        self.assertEqual(body["response_format"], {"type": "json_schema", "json_schema": {
            "name": "answer", "schema": schema, "strict": True,
        }})
        self.assertEqual(body["max_tokens"], 9000)

    async def test_image_only_classification_uses_converted_images(self):
        await self.complete(responses(input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": PNG},
        ]}], max_output_tokens=9000))
        stage = self.stages[-1]["after"]
        self.assertEqual(stage["litellm_metadata"]["k3_class"], "P2-agentic")
        self.assertEqual(self.requests[-1][1]["max_tokens"], 9000)
        self.assertNotIn("tools", self.requests[-1][1])

    async def test_full_inline_function_history_does_not_recover_foreign_cached_arguments(self):
        from litellm.responses.litellm_completion_transformation.transformation import TOOL_CALLS_CACHE
        arguments = '{"previous_response_id":"argument-data","cache_key":"argument-data"}'
        output = {"store": True, "extra_body": {"max_tokens": 99999}, "previous_response_id": "result-data"}
        history = [
            {"role": "user", "content": "Lookup"},
            {"type": "function_call", "id": "fc-history", "call_id": "call-history", "name": "lookup", "arguments": arguments},
            {"type": "function_call_output", "call_id": "call-history", "output": output},
        ]
        foreign = {"id": "call-history", "type": "function", "function": {"name": "foreign", "arguments": "foreign"}}
        with patch.object(TOOL_CALLS_CACHE, "get_cache", return_value=foreign):
            await self.complete(responses(input=history, tools=[function()], max_output_tokens=16))
        messages = self.requests[-1][1]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-history")
        self.assertEqual(messages[1]["tool_calls"][0]["function"], {"name": "lookup", "arguments": arguments})
        self.assertEqual(messages[2]["tool_call_id"], "call-history")
        self.assertEqual(json.loads(messages[2]["content"]), output)
        self.assertNotIn("foreign", json.dumps(messages))
        with patch.object(TOOL_CALLS_CACHE, "get_cache", side_effect=AssertionError("cross-request history")) as lookup:
            await self.assert_bad_request(responses(input=[history[-1]], tools=[function()]))
            await self.assert_bad_request(responses(input=[history[-1], history[-2]], tools=[function()]))
            lookup.assert_not_called()

    async def test_stream_has_terminal_response_and_cached_usage(self):
        events = await self.complete(responses(max_output_tokens=16, stream=True))
        terminal = [event for event in events if event.type == "response.completed"]
        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0].response.usage.input_tokens_details.cached_tokens, 64)
        self.assertEqual(self.requests[-1][1]["stream_options"], {"include_usage": True})

    async def test_function_call_response_is_preserved_for_stream_and_nonstream(self):
        self.tool_response = True
        for stream in (False, True):
            result = await self.complete(responses(tools=[function()], max_output_tokens=16, stream=stream))
            if stream:
                completed = [event for event in result if event.type == "response.completed"]
                self.assertEqual(len(completed), 1)
                result = completed[0].response
            calls = [item for item in result.output if item.type == "function_call"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].call_id, "call-offline")
            self.assertEqual(calls[0].name, "lookup")
            self.assertEqual(json.loads(calls[0].arguments), {"x": "value"})
            self.assertEqual(result.usage.input_tokens_details.cached_tokens, 64)

    async def test_chat_still_uses_chat_without_leaking_bridge_control(self):
        result = await self.complete({"model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "Reply OK"}],
                                      "max_output_tokens": 16}, route="acompletion")
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(self.requests[-1][1]["max_tokens"], 16)
        self.assertNotIn("use_chat_completions_api", self.requests[-1][1])
        self.assertNotIn("max_output_tokens", self.requests[-1][1])
        self.assertEqual([stage["call_type"] for stage in self.stages], ["acompletion"])

    async def test_postroute_policy_retains_target_and_truthful_offbox_metadata(self):
        self.callback.policy.offbox_configured = True
        for route in ("aresponses", "acompletion"):
            payload = responses(max_output_tokens=16) if route == "aresponses" else {
                "model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16,
            }
            await self.complete(payload, route=route)
            stage = self.stages[-1]
            for key in ("model", "api_base", "api_key"):
                self.assertEqual(stage["before"][key], stage["after"][key])
            self.assertIs(stage["before"]["client"], stage["after"]["client"])
            # Responses resolves the provider prefix before the inner SDK call.
            self.assertEqual(stage["after"]["model"].removeprefix("openai/"), "FW-Kimi-K3")
            metadata = stage["after"]["litellm_metadata" if route == "aresponses" else "metadata"]
            self.assertFalse(metadata["k3_offbox"])
            self.assertEqual(self.requests[-1][1]["model"], "FW-Kimi-K3")

    async def test_responses_never_read_or_write_full_response_cache(self):
        self.cache.mode = "default_on"
        self.cache.supported_call_types = ["acompletion", "aresponses", "responses"]
        with patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
             patch.object(self.cache, "get_cache", side_effect=AssertionError("sync cache read")) as sync_read, \
             patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write, \
             patch.object(self.cache, "add_cache", side_effect=AssertionError("sync cache write")) as sync_write:
            for mode in ("static", "opt_in", "off"):
                self.callback.cache_mode = mode
                for stream in (False, True):
                    payload = responses(max_output_tokens=16, stream=stream, store=False, background=False,
                                        caching=True, cache={"use-cache": True}, cache_key="forged",
                                        litellm_params={"preset_cache_key": "forged"})
                    await self.complete(payload)
                    await self.complete(payload)
            for mock in (read, sync_read, write, sync_write):
                mock.assert_not_called()
        self.assertEqual(len(self.requests), 12)
        for stage in self.stages:
            self.assertFalse(stage["after"]["caching"])
            self.assertTrue(stage["after"]["cache"]["no-cache"])
            self.assertTrue(stage["after"]["cache"]["no-store"])

    async def test_inner_logging_replacement_cannot_enable_responses_cache_or_skip_policy(self):
        from redesign.tenancy.cache_policy import CONTEXT_ATTR
        chat = await self.prepare({"model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "hi"}]}, route="acompletion")
        authenticated_chat_logging = chat["litellm_logging_obj"]
        self.assertIsNotNone(getattr(authenticated_chat_logging, CONTEXT_ATTR))
        self.replace_inner_logging = True
        self.cache.mode = "default_on"
        for replacement in (None, authenticated_chat_logging):
            self.inner_logging = replacement
            with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
                 patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
                 patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write:
                await self.complete(responses(max_output_tokens=9000))
                self.assertEqual(apply.call_count, 1)
                read.assert_not_called()
                write.assert_not_called()
            self.assertEqual(self.requests[-1][1]["max_tokens"], 4096)

    async def test_stateful_and_native_features_are_rejected_before_handlers(self):
        from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler
        with patch.object(ResponsesSessionHandler, "get_chat_completion_message_history_for_previous_response_id",
                          side_effect=AssertionError("session lookup")) as history, \
             patch("litellm.responses.main.aresponses_api_with_mcp", side_effect=AssertionError("MCP handler")) as mcp, \
             patch("litellm.responses.file_search.emulated_handler.aresponses_with_emulated_file_search",
                   side_effect=AssertionError("file search handler")) as search:
            for fields in (
                {"previous_response_id": "resp-missing"}, {"conversation": "conv-missing"},
                {"store": True}, {"background": True}, {"prompt": {"id": "pmpt-missing"}},
                {"prompt_id": "pmpt-missing"}, {"include": ["reasoning.encrypted_content"]},
                {"truncation": "auto"}, {"context_management": [{"type": "compaction"}]},
                {"input": [{"type": "item_reference", "id": "item-missing"}]},
                {"input": [{"type": "reasoning", "encrypted_content": "opaque", "summary": []}]},
                {"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file-missing"}]}]},
                {"input": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}]},
                {"input": [{"role": "user", "content": [{"type": "input_file", "file_data": "AA=="}]}]},
                {"use_chat_completions_api": False}, {"reasoning": {"summary": "detailed"}},
            ):
                with self.subTest(fields=fields):
                    await self.assert_bad_request(responses(**fields))
            for kind in ("web_search", "web_search_preview", "file_search", "mcp", "code_interpreter",
                         "computer_use", "computer_use_preview", "image_generation", "shell", "local_shell",
                         "apply_patch", "custom", "namespace", "future_native_tool"):
                with self.subTest(tool=kind):
                    await self.assert_bad_request(responses(tools=[{"type": kind}]))
            history.assert_not_called()
            mcp.assert_not_called()
            search.assert_not_called()

    async def test_deployment_defaults_and_extra_body_are_revalidated(self):
        for defaults in (
            {"tools": [{"type": "shell"}]}, {"previous_response_id": "resp-default"},
            {"store": True}, {"background": True}, {"tool_choice": "required"},
            {"extra_body": {"max_completion_tokens": 128000}},
        ):
            with self.subTest(defaults=defaults):
                with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply:
                    await self.assert_bad_request(responses(), defaults=defaults, early=False)
                apply.assert_not_called()
                self.assertEqual(self.stages[-1]["call_type"], "aresponses")
        for field, value in (
            ("messages", [{"role": "user", "content": "hidden"}]), ("tools", [{"type": "shell"}]),
            ("max_tokens", 128000), ("max_completion_tokens", 128000), ("max_output_tokens", 128000),
            ("previous_response_id", "hidden"), ("text", {"format": {"type": "json_object"}}),
            ("extra_body", {"max_tokens": 128000}),
        ):
            await self.assert_bad_request(responses(max_output_tokens=16, extra_body={field: value}))

    async def test_malformed_inputs_return_400_instead_of_being_dropped(self):
        for fields in (
            {"max_output_tokens": 0}, {"max_output_tokens": -1}, {"max_output_tokens": True},
            {"max_output_tokens": "16"}, {"input": None}, {"input": {}}, {"input": [None]},
            {"input": [{"role": [], "content": "hi"}]},
            {"input": [{"role": "user", "content": [{"type": [], "text": "hi"}]}]},
            {"input": [{"type": "function_call", "call_id": "call-x", "name": "lookup"}]},
            {"input": [{"type": "function_call_output", "call_id": "", "output": "result"}]},
            {"tools": {}}, {"tools": [{"type": "function"}]}, {"tools": [function(parameters=[])]},
            {"tool_choice": {"type": "function"}}, {"tool_choice": {"type": "web_search"}},
            {"tool_choice": {"type": "function", "function": {"name": "lookup"}, "name": "other"}},
            {"text": {"format": {"type": "json_schema", "name": "bad", "schema": []}}},
            {"text": {"format": {"type": []}}}, {"reasoning": {"effort": []}},
            {"stream_options": {"include_obfuscation": True}}, {"extra_body": []},
        ):
            with self.subTest(fields=fields):
                await self.assert_bad_request(responses(**fields))

    async def test_converted_prompt_past_window_is_rejected_before_provider(self):
        from redesign.gateway.clamping import TokenClamp
        self.callback.policy.clamp = TokenClamp(max_model_len=1000)
        with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply:
            await self.assert_bad_request(responses(instructions="x" * 8000, max_output_tokens=16), early=False)
        self.assertEqual(apply.call_count, 1)
        self.assertEqual([stage["call_type"] for stage in self.stages], ["aresponses", "acompletion"])

    async def test_offbox_rendered_deployment_also_uses_chat_bridge(self):
        import yaml
        self.models = yaml.safe_load(render("http://provider.invalid/v1", "offline-provider", "remote-model"))["model_list"]
        for model in self.models:
            self.assertIs(model["litellm_params"]["use_chat_completions_api"], True)
        await self.complete(responses(model="p1-offbox", max_output_tokens=16))
        self.assertEqual(self.requests[-1][1]["model"], "remote-model")

    async def test_deployed_normalizer_interface_is_sufficient(self):
        path = Path("/usr/local/lib/k3/redesign/gateway/media.py")
        if not path.is_file():
            self.skipTest("requires the installed gateway normalizer")
        spec = importlib.util.spec_from_file_location("deployed_k3_media", path)
        deployed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployed)
        with patch("redesign.tenancy.callback.normalize_payload", wraps=deployed.normalize_payload) as normalize:
            await self.complete(responses(input=[{"role": "user", "content": [
                {"type": "input_text", "text": "Describe"}, {"type": "input_image", "image_url": PNG},
            ]}], tools=[function()], reasoning={"effort": "medium"}, max_output_tokens=9000))
        self.assertEqual(normalize.call_count, 1)
        body = self.requests[-1][1]
        part = body["messages"][0]["content"][1]
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG"))
        self.assertEqual(body["max_tokens"], 9000)
        self.assertEqual(self.stages[-1]["after"]["reasoning_effort"], "high")


if __name__ == "__main__":
    unittest.main()
