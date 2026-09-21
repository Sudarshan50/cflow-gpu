"""Response-cache regressions; real LiteLLM with local memory and mock HTTP only.

Run with /usr/local/lib/k3/venv/bin/python -m unittest
redesign.tests.test_cache_policy -v on the proxy's installed LiteLLM build.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from redesign.tenancy.cache_policy import (
    authenticated_scope, cached_usage_details, capture_context, finalize_response_cache,
)
from redesign.tenancy.render_config import render


def chat(**kwargs):
    return {
        "model": "FW-Kimi-K3", "messages": [{"role": "user", "content": "hello"}],
        **kwargs,
    }


class CachePolicyTest(unittest.TestCase):
    def key(self, data=None, auth=None, **settings):
        data = copy.deepcopy(data if data is not None else chat())
        context = capture_context(data, auth or {"api_key": "test-key-a"}, "acompletion", **settings)
        finalize_response_cache(data, context, "acompletion")
        return data

    def test_key_scope_is_stable_and_never_falls_back_to_shared_user_or_team(self):
        a = {"api_key": "test-key-a", "team_id": "shared", "user_id": "owner"}
        b = {**a, "api_key": "test-key-b"}
        self.assertNotEqual(authenticated_scope(a), authenticated_scope(b))
        self.assertIsNone(authenticated_scope({"team_id": "shared", "user_id": "owner"}))
        self.assertEqual(authenticated_scope(a), authenticated_scope({"token": hashlib.sha256(b"test-key-a").hexdigest()}))
        self.assertNotIn("test-key-a", self.key(auth=a)["cache_key"])

    def test_only_operator_allowlisted_teams_share(self):
        a, b = {"api_key": "a", "team_id": "t"}, {"api_key": "b", "team_id": "t"}
        pools = frozenset({"t"})
        self.assertEqual(authenticated_scope(a, pools), authenticated_scope(b, pools))
        self.assertNotEqual(authenticated_scope(a, pools), authenticated_scope({**a, "team_id": "other"}, pools))
        self.assertNotEqual(authenticated_scope(a), authenticated_scope(b))

    def test_every_generation_setting_and_target_changes_key(self):
        baseline = self.key()["cache_key"]
        for variation in (
            {"temperature": 0.3}, {"top_p": 0.9}, {"max_tokens": 12}, {"seed": 9},
            {"stop": ["END"]}, {"reasoning_effort": "max"}, {"stream": True},
            {"chat_template_kwargs": {"thinking": True}},
            {"extra_body": {"chat_template_kwargs": {"thinking_effort": "max"}}},
            {"extra_body": {"top_k": 7}}, {"response_format": {"type": "json_object"}},
            {"api_base": "http://other.invalid/v1"}, {"api_key": "provider-test"},
            {"model": "other"}, {"future_generation_option": {"mode": "new"}},
        ):
            with self.subTest(variation=variation):
                self.assertNotEqual(baseline, self.key(chat(**variation))["cache_key"])
        self.assertNotEqual(baseline, self.key(revision="new-weights")["cache_key"])

    def test_order_and_request_bookkeeping_do_not_change_keys(self):
        a = chat(extra_body={"top_k": 5, "chat_template_kwargs": {"thinking": True, "x": 2}})
        b = dict(reversed(list(a.items())))
        b["extra_body"] = {"chat_template_kwargs": {"x": 2, "thinking": True}, "top_k": 5}
        b.update(litellm_call_id="fresh", metadata={"user_api_key": "spoof", "trace_id": "random"})
        self.assertEqual(self.key(a)["cache_key"], self.key(b)["cache_key"])
        a["stop"], b["stop"] = ["one", "two"], ["two", "one"]
        self.assertNotEqual(self.key(a)["cache_key"], self.key(b)["cache_key"])

    def test_static_default_operator_modes_and_explicit_opt_outs(self):
        for payload in (chat(), chat(cache={}), chat(cache=None)):
            self.assertTrue(self.key(payload)["caching"])
            self.assertFalse(self.key(payload, mode="opt_in")["caching"])
        opted_in = chat(cache={"use-cache": True})
        self.assertTrue(self.key(opted_in, mode="opt_in")["caching"])
        self.assertFalse(self.key(opted_in, mode="off")["caching"])
        self.assertFalse(self.key(opted_in, mode="invalid")["caching"])
        for mode in ("static", "opt_in", "off"):
            for payload in (chat(caching=False, cache={"use-cache": True}),
                            chat(cache={"use-cache": False}),
                            chat(cache={"use-cache": True, "no-cache": True}),
                            chat(cache={"use-cache": True, "no-store": True})):
                self.assertFalse(self.key(payload, mode=mode)["caching"])

    def test_tool_mutable_media_always_bypass(self):
        cases = [chat(cache={"use-cache": False}), chat(cache={"use-cache": "true"}), chat(caching=False)]
        for field in ("tools", "functions", "previous_response_id", "session_id", "web_search_options", "prompt_id"):
            cases.append(chat(**{field: ["present"]}))
        cases += [
            chat(messages=[{"role": "tool", "content": "result"}]),
            chat(messages=[{"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]}]),
            chat(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://mutable.invalid/a"}}]}]),
            chat(extra_body={"tools": [{"type": "function"}]}),
            chat(extra_body={"messages": [{"role": "tool", "content": "result"}]}),
            chat(metadata={"previous_response_id": "state-handle"}),
            chat(cache={"use-cache": True, "no-store": True}),
            chat(cache={"use-cache": True, "no-cache": True}),
        ]
        for data in cases:
            with self.subTest(data=data):
                result = self.key(data)
                self.assertFalse(result["caching"])
                self.assertTrue(result["cache"]["no-cache"])
                self.assertTrue(result["cache"]["no-store"])

    def test_logging_session_ids_are_not_conversation_state_or_cache_identity(self):
        baseline = self.key()["cache_key"]
        for index in range(2):
            payload = chat(litellm_session_id=f"logging-{index}", litellm_call_id=f"call-{index}",
                           metadata={"session_id": f"log-{index}", "trace_id": f"trace-{index}"},
                           litellm_metadata={"session_id": f"other-log-{index}"})
            self.assertEqual(self.key(payload)["cache_key"], baseline)
        self.assertFalse(self.key(chat(session_id="actual-state"))["caching"])
        self.assertFalse(self.key(chat(conversation_id="actual-state"))["caching"])

    def test_namespace_and_preset_spoofs_are_overridden_in_all_envelopes(self):
        payload = chat(cache={"use-cache": True, "namespace": "victim"})
        for name in (None, "metadata", "litellm_metadata", "litellm_params", "extra_body"):
            body = payload if name is None else payload.setdefault(name, {})
            body.update(cache_key="victim", preset_cache_key="victim", redis_namespace="victim", caching_groups=[["victim"]])
        result = self.key(payload)
        self.assertNotIn("victim", json.dumps(result))
        self.assertNotIn("litellm_params", result)
        self.assertTrue(result["cache_key"].startswith("k3-response-v1:"))

    def test_unserializable_and_nonfinite_settings_fail_closed(self):
        for value in (object(), float("nan"), float("inf")):
            self.assertFalse(self.key(chat(extra_body={"setting": value}))["caching"])
        self.assertFalse(self.key(chat(cache={"use-cache": True, "ttl": "300"}))["caching"])
        self.assertEqual(self.key(chat(cache={"use-cache": True, "ttl": 900}))["cache"]["ttl"], 300)

    def test_cached_usage_details_accept_saved_snapshots_without_executing_text(self):
        snapshot = {"usage": {"prompt_tokens": 12, "prompt_tokens_details": {"cached_tokens": 8}}}
        expected = {"prompt_tokens_details": {"cached_tokens": 8}}
        for saved in (snapshot, str(snapshot), json.dumps(snapshot)):
            self.assertEqual(cached_usage_details(saved), expected)
        for saved in (None, "redacted", "__import__('os').getcwd()", {"usage": None}):
            self.assertEqual(cached_usage_details(saved), {})


class TokenLimitPrecedenceTest(unittest.TestCase):
    def test_effective_limit_survives_tenancy_and_gateway_without_expansion(self):
        from redesign.gateway.server import GatewayService
        from redesign.tenancy.policy import TenancyPolicy
        estimator = SimpleNamespace(estimate=lambda payload: 32)
        policy = TenancyPolicy(estimator=estimator)
        service = SimpleNamespace(estimator=estimator, batch_customers=frozenset())
        for aliases, expected in (
            ({"max_tokens": 16}, 16),
            ({"max_completion_tokens": 16}, 16),
            ({"max_tokens": 1000, "max_completion_tokens": 16}, 16),
            ({"max_tokens": 16, "max_completion_tokens": 1000}, 1000),
            ({"max_tokens": 16, "max_completion_tokens": None}, 16),
            ({"max_tokens": 128_000, "max_completion_tokens": 128_000}, 4096),
            ({}, 1024),
        ):
            with self.subTest(aliases=aliases):
                data = chat(**aliases)
                before = GatewayService.envelope(service, data, {}, "/v1/chat/completions")
                decision = policy.apply(data, data["model"])
                after = GatewayService.envelope(service, data, {}, "/v1/chat/completions")
                self.assertFalse(decision.reject)
                self.assertEqual(decision.granted_max_tokens, expected)
                self.assertEqual(after.requested_max_tokens, expected)
                self.assertEqual(data["max_tokens"], expected)
                if "max_completion_tokens" in aliases:
                    self.assertEqual(data["max_completion_tokens"], expected)
                if before.requested_max_tokens is not None:
                    self.assertLessEqual(after.requested_max_tokens, before.requested_max_tokens)
                repeated = policy.apply(data, data["model"])
                self.assertEqual(repeated.granted_max_tokens, decision.granted_max_tokens)
                self.assertEqual(repeated.traffic_class, decision.traffic_class)
                self.assertEqual(repeated.model, decision.model)
                # Grants are idempotent; diagnostics describe each invocation's
                # actual input, not a reconstructed original caller budget.
                self.assertEqual(repeated.requested_max_tokens, expected)
                self.assertEqual(repeated.clamp_reason, "unchanged")
                self.assertFalse(repeated.default_applied)

    def test_invalid_aliases_are_rejected_instead_of_falling_back(self):
        from redesign.gateway.clamping import InvalidTokenLimit
        from redesign.gateway.server import GatewayService
        from redesign.tenancy.policy import TenancyPolicy
        estimator = SimpleNamespace(estimate=lambda payload: 32)
        policy = TenancyPolicy(estimator=estimator)
        service = SimpleNamespace(estimator=estimator, batch_customers=frozenset())
        for value in (0, -1, True, False, "1000", 1.5):
            for field, other in (("max_completion_tokens", "max_tokens"), ("max_tokens", "max_completion_tokens")):
                with self.subTest(field=field, value=value):
                    data = chat(**{field: value, other: 16})
                    with self.assertRaisesRegex(InvalidTokenLimit, field):
                        policy.apply(data, data["model"])
                    with self.assertRaisesRegex(InvalidTokenLimit, field):
                        GatewayService.envelope(service, data, {}, "/v1/chat/completions")
                    self.assertEqual(data[field], value)


@unittest.skipUnless(importlib.util.find_spec("litellm"), "requires installed proxy LiteLLM")
class InstalledLiteLLMTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import litellm
        import httpx
        from openai import AsyncOpenAI
        from litellm.caching.caching import Cache
        from litellm.proxy import proxy_server
        from redesign.tenancy.callback import K3TenancyCallback
        # Router instances register bound methods in these process-global lists.
        # Isolate and restore all of them, not just litellm.callbacks, so the
        # full suite cannot retain old Routers/callbacks or hit MAX_CALLBACKS.
        for name in ("input_callback", "success_callback", "failure_callback", "service_callback",
                     "_async_input_callback", "_async_success_callback", "_async_failure_callback"):
            context = patch.object(litellm, name, [])
            context.start()
            self.addCleanup(context.stop)
        self.litellm = litellm
        self.callback = K3TenancyCallback(team_pool_ids=frozenset(), cache_mode="static")
        self.general_settings = {"always_include_stream_usage": True}
        self.prepared = []
        self.cache = Cache(type="local", mode="default_off", supported_call_types=["acompletion"], ttl=300)
        # Match proxy startup without starting a server or connecting to Redis.
        # route_request() attaches this pool AFTER the authenticated proxy hook.
        self.shared_session = await proxy_server._initialize_shared_aiohttp_session()
        self.assertIsNotNone(self.shared_session)
        self.addAsyncCleanup(self.shared_session.close)
        self.patches = [patch.object(litellm, "cache", self.cache), patch.object(litellm, "callbacks", [self.callback]),
                        patch.object(litellm, "enable_caching_on_provider_specific_optional_params", True),
                        patch.object(proxy_server, "shared_aiohttp_session", self.shared_session)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.bodies = []

        def transport(request):
            self.bodies.append(json.loads(request.content))
            if self.bodies[-1].get("stream"):
                chunks = [
                    {"id": "test-stream", "object": "chat.completion.chunk", "created": 1, "model": "FW-Kimi-K3",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                    {"id": "test-stream", "object": "chat.completion.chunk", "created": 1, "model": "FW-Kimi-K3",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                if self.bodies[-1].get("stream_options", {}).get("include_usage"):
                    chunks.append({
                        "id": "test-stream", "object": "chat.completion.chunk", "created": 1,
                        "model": "FW-Kimi-K3", "choices": [],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13,
                                  "prompt_tokens_details": {"cached_tokens": 8}},
                    })
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n")
            return httpx.Response(200, json={
                "id": "test-completion", "object": "chat.completion", "created": 1,
                "model": "FW-Kimi-K3", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            })

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        self.client = AsyncOpenAI(api_key="test-provider-key", base_url="http://provider.invalid/v1", http_client=self.http)
        self.addAsyncCleanup(self.client.close)

    async def prepare(self, payload=None, key="test-key-a", *, key_metadata=None, headers=()):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
        from litellm.proxy.utils import ProxyLogging
        from starlette.requests import Request
        data = copy.deepcopy(payload if payload is not None else chat())
        proxy = ProxyLogging(user_api_key_cache=self.litellm.DualCache())
        request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                           "headers": list(headers), "scheme": "http", "server": ("test.invalid", 80), "query_string": b""})
        # Actual proxy setup and callback dispatch (no HTTP server or auth DB).
        with patch.object(proxy, "_init_response_taking_too_long_task"):
            data, _ = await ProxyBaseLLMRequestProcessing(data).common_processing_pre_call_logic(
                request=request, general_settings=self.general_settings, proxy_logging_obj=proxy,
                user_api_key_dict=UserAPIKeyAuth(api_key=key, metadata=key_metadata or {}), proxy_config=None,
                route_type="acompletion",
            )
            self.prepared.append(data)
            return data

    async def complete(self, payload=None, key="test-key-a", **deployment):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.route_llm_request import route_request
        data = await self.prepare(payload, key)
        # Include the proxy dispatch step, not just Router.acompletion: this
        # is where production injects the non-JSON shared_session object.
        router = self.litellm.Router(model_list=[{
            "model_name": "FW-Kimi-K3",
            "litellm_params": {"model": "openai/FW-Kimi-K3", "api_base": "http://provider.invalid/v1", "api_key": "test-provider-key", **deployment},
        }], num_retries=0)
        with patch.object(router, "_get_async_openai_model_client", return_value=self.client):
            call = await route_request(
                data=data, llm_router=router, user_model=None, route_type="acompletion",
                user_api_key_dict=UserAPIKeyAuth(api_key=key),
            )
            response = await call
            if data.get("stream"):
                from litellm.proxy.utils import ProxyLogging
                proxy = ProxyLogging(user_api_key_cache=self.litellm.DualCache())
                response = [chunk async for chunk in proxy.async_post_call_streaming_iterator_hook(
                    response=response, user_api_key_dict=UserAPIKeyAuth(api_key=key), request_data=data,
                )]
        # Cache writes are background tasks; local memory only, never Redis.
        for _ in range(10):
            await asyncio.sleep(0)
        return response

    async def test_real_proxy_dispatch_keeps_context_and_reuses_across_worker_sessions(self):
        from aiohttp import ClientSession
        from litellm.proxy import proxy_server
        from redesign.tenancy.cache_policy import CONTEXT_ATTR, _BOOKKEEPING
        snapshots = []
        original_hook = self.callback.async_pre_call_deployment_hook

        async def inspect_shape(kwargs, call_type):
            # Record only shape/booleans here; never request/identity values.
            context_present = getattr(kwargs.get("litellm_logging_obj"), CONTEXT_ATTR, None) is not None
            session_present = isinstance(kwargs.get("shared_session"), ClientSession)
            non_json_fields = []
            for name, value in kwargs.items():
                if name not in _BOOKKEEPING:
                    try:
                        json.dumps(value, allow_nan=False)
                    except (TypeError, ValueError):
                        non_json_fields.append(name)
            result = await original_hook(kwargs, call_type)
            snapshots.append((context_present, session_present, non_json_fields,
                              result.get("cache_key", "").startswith("k3-response-v1:")))
            return result

        with patch.object(self.callback, "async_pre_call_deployment_hook", side_effect=inspect_shape):
            await self.complete(chat())
            other_session = await proxy_server._initialize_shared_aiohttp_session()
            self.assertIsNotNone(other_session)
            self.addAsyncCleanup(other_session.close)
            self.assertIsNot(other_session, self.shared_session)
            with patch.object(proxy_server, "shared_aiohttp_session", other_session):
                await self.complete(chat())
        self.assertEqual(snapshots, [(True, True, [], True), (True, True, [], True)])
        self.assertEqual(len(self.bodies), 1, "connection pool identity must not disable or partition response caching")
        self.assertNotIn("shared_session", self.bodies[0])

    async def test_real_cache_lifecycle_stable_hit_tenant_and_provider_variation(self):
        payload = chat(extra_body={"chat_template_kwargs": {"thinking": True, "thinking_effort": "high"}})
        self.assertNotIn("cache", payload)
        first = await self.complete(payload, temperature=0.2)
        second = await self.complete(dict(reversed(list(payload.items()))), temperature=0.2)
        self.assertEqual(len(self.bodies), 1, "second identical authenticated request must hit")
        self.assertEqual(first.choices[0].message.content, second.choices[0].message.content)
        await self.complete(payload, key="test-key-b", temperature=0.2)
        await self.complete(payload, temperature=0.3)
        changed = copy.deepcopy(payload)
        changed["extra_body"]["chat_template_kwargs"]["thinking"] = False
        await self.complete(changed, temperature=0.2)
        self.assertEqual(len(self.bodies), 4)
        for body in self.bodies:
            self.assertNotIn("cache_key", body)
            self.assertNotIn("litellm_params", body)
            self.assertNotIn("cache_salt", body)

    async def test_explicit_opt_outs_and_tools_neither_read_nor_write(self):
        for mode in ("default_off", "default_on"):
            self.cache.mode = mode
            for payload in (chat(cache={"use-cache": False}, stream=True), chat(caching=False),
                            chat(cache={"no-cache": True}), chat(cache={"no-store": True}),
                            chat(tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}])):
                with patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
                     patch.object(self.cache, "get_cache", side_effect=AssertionError("cache read")) as sync_read, \
                     patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write:
                    await self.complete(payload)
                    read.assert_not_called()
                    sync_read.assert_not_called()
                    write.assert_not_called()

    async def test_standard_chat_hits_with_real_generated_session_bookkeeping(self):
        self.general_settings["missing_session_id"] = "generate"
        await self.complete(chat())
        await self.complete(chat())
        self.assertEqual(len(self.bodies), 1)
        first, second = self.prepared
        self.assertTrue(first["metadata"]["litellm_session_id_generated"])
        self.assertTrue(second["metadata"]["litellm_session_id_generated"])
        self.assertNotEqual(first["litellm_session_id"], second["litellm_session_id"])
        self.assertNotEqual(first["litellm_call_id"], second["litellm_call_id"])

    async def test_opt_out_survives_key_level_defaults_and_http_cache_control(self):
        for control in ({"use-cache": False}, {"no-cache": True}, {"no-store": True}):
            data = await self.prepare(chat(cache=control), key_metadata={"cache": {"ttl": 60}})
            await self.callback.async_pre_call_deployment_hook(data, "acompletion")
            self.assertTrue(data["cache"]["no-store"])
        self.callback.cache_mode = "opt_in"
        data = await self.prepare(chat(cache={"use-cache": True}), key_metadata={"cache": {"ttl": 60}})
        await self.callback.async_pre_call_deployment_hook(data, "acompletion")
        self.assertTrue(data["caching"])
        self.assertEqual(data["cache"]["ttl"], 60)
        self.callback.cache_mode = "static"
        for value in (b"no-cache", b"public, no-store", b"max-age=300, No-Cache"):
            data = await self.prepare(chat(), headers=((b"cache-control", value),))
            await self.callback.async_pre_call_deployment_hook(data, "acompletion")
            self.assertTrue(data["cache"]["no-store"])

    async def test_operator_modes_from_environment_on_real_pipeline(self):
        from redesign.tenancy.callback import K3TenancyCallback
        with patch.dict(os.environ):
            os.environ.pop("K3_RESPONSE_CACHE_MODE", None)
            self.assertEqual(K3TenancyCallback(team_pool_ids=frozenset()).cache_mode, "static")
        for mode in ("static", "opt_in", "off"):
            with patch.dict(os.environ, {"K3_RESPONSE_CACHE_MODE": mode}):
                callback = K3TenancyCallback(team_pool_ids=frozenset())
            self.assertEqual(callback.cache_mode, mode)
            with patch.object(self.litellm, "callbacks", [callback]):
                start = len(self.bodies)
                payload = chat(messages=[{"role": "user", "content": mode}])
                await self.complete(payload)
                await self.complete(payload)
                self.assertEqual(len(self.bodies) - start, 1 if mode == "static" else 2)
                start = len(self.bodies)
                payload = chat(messages=[{"role": "user", "content": mode + " opted in"}], cache={"use-cache": True})
                await self.complete(payload)
                await self.complete(payload)
                self.assertEqual(len(self.bodies) - start, 2 if mode == "off" else 1)
        with patch.dict(os.environ, {"K3_RESPONSE_CACHE_MODE": "typo"}):
            with self.assertRaisesRegex(ValueError, "static, opt_in, or off"):
                K3TenancyCallback()

    async def test_effective_token_limits_are_cached_and_sent_without_expansion(self):
        await self.complete(chat(max_tokens=1000, max_completion_tokens=16))
        await self.complete(chat(max_tokens=2000, max_completion_tokens=16))
        self.assertEqual(len(self.bodies), 1, "ignored legacy alias must not change the effective key")
        await self.complete(chat(max_tokens=16, max_completion_tokens=1000))
        self.assertEqual(len(self.bodies), 2)
        for body, expected in zip(self.bodies, (16, 1000)):
            for name in ("max_tokens", "max_completion_tokens"):
                if name in body:
                    self.assertEqual(body[name], expected)

    async def test_chat_portal_alias_is_canonical_before_policy_and_cache(self):
        for payload in (chat(max_output_tokens=16), chat(max_tokens=16),
                        chat(max_tokens=2000, max_output_tokens=16)):
            await self.complete(payload)
        self.assertEqual(len(self.bodies), 1, "equivalent effective Chat limits must share a key")
        await self.complete(chat(max_output_tokens=16, max_completion_tokens=1000))
        self.assertEqual(len(self.bodies), 2)
        self.assertEqual(self.bodies[0]["max_tokens"], 16)
        self.assertEqual(self.bodies[1]["max_completion_tokens"], 1000)
        self.assertEqual(self.bodies[1]["max_tokens"], 1000)
        for body in self.bodies:
            self.assertNotIn("max_output_tokens", body)

    async def test_deployment_defaults_are_clamped_once_before_cache_lookup(self):
        from redesign.gateway.media import normalize_payload
        with patch.object(self.callback.policy, "apply", wraps=self.callback.policy.apply) as apply, \
             patch("redesign.tenancy.callback.normalize_payload", wraps=normalize_payload) as normalize:
            await self.complete(chat(), max_tokens=19)
            await self.complete(chat(), max_tokens=19)
            self.assertEqual(apply.call_count, 2, "one application per request, including a cache hit")
            self.assertEqual(normalize.call_count, 2)
        self.assertEqual(len(self.bodies), 1)
        self.assertEqual(self.bodies[0]["max_tokens"], 19)
        await self.complete(chat(), max_tokens=20)
        self.assertEqual(len(self.bodies), 2, "the effective deployment limit is part of cache identity")

    async def test_reserved_extra_body_overrides_cannot_bypass_chat_policy(self):
        from fastapi import HTTPException
        fields = {
            "messages": [{"role": "user", "content": "x" * 1000}],
            "tools": [{"type": "function", "function": {"name": "hidden"}}],
            "max_tokens": 128000, "max_completion_tokens": 128000, "max_output_tokens": 128000,
            "model": "other", "stream": True, "extra_body": {"max_tokens": 128000},
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                data = await self.prepare(chat(max_tokens=16, extra_body={field: value}))
                with self.assertRaises(HTTPException) as error:
                    await self.callback.async_pre_call_deployment_hook(data, "acompletion")
                self.assertEqual(error.exception.status_code, 400)
        self.assertFalse(self.bodies)

    async def test_deployment_media_failure_cannot_turn_static_context_cacheable(self):
        from fastapi import HTTPException
        from redesign.gateway.media import MediaValidationError
        # An authenticated text request captures eligibility before Router adds
        # media. A failed decode must reject before cache lookup or inference.
        data = await self.prepare(chat())
        data["messages"] = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "bad-image"}},
        ]}]
        with patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
             patch.object(self.cache, "get_cache", side_effect=AssertionError("cache read")) as sync_read, \
             patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write:
            with self.assertRaises(HTTPException) as error:
                await self.callback.async_pre_call_deployment_hook(data, "acompletion")
            read.assert_not_called()
            sync_read.assert_not_called()
            write.assert_not_called()
        self.assertEqual(error.exception.status_code, 400)
        self.assertIsInstance(error.exception.__cause__, MediaValidationError)
        self.assertIn("messages[0].content[0]", error.exception.detail)
        self.assertNotIn("bad-image", error.exception.detail)
        self.assertEqual(data["messages"][0]["content"][0]["type"], "image_url")
        self.assertTrue(data["cache"]["no-store"])
        self.assertTrue(data["cache"]["no-cache"])
        self.assertFalse(data["caching"])
        self.assertFalse(self.bodies)

    async def test_missing_logging_context_at_real_sdk_barrier_fails_closed(self):
        original = self.callback.async_pre_call_deployment_hook

        async def without_logging(kwargs, call_type):
            kwargs.pop("litellm_logging_obj", None)
            return await original(kwargs, call_type)

        with patch.object(self.callback, "async_pre_call_deployment_hook", side_effect=without_logging), \
             patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
             patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write:
            await self.complete(chat(max_output_tokens=16))
            await self.complete(chat(max_output_tokens=16))
            read.assert_not_called()
            write.assert_not_called()
        self.assertEqual(len(self.bodies), 2)
        self.assertEqual(self.bodies[0]["max_tokens"], 16, "policy still runs without authenticated logging")

    async def test_stream_cache_and_inbound_preset_cannot_cross_tenants(self):
        first = await self.complete(chat(stream=True))
        second = await self.complete(chat(stream=True))
        self.assertEqual(len(self.bodies), 1)
        self.assertTrue(first and second)
        keys = list(self.cache.cache.cache_dict)
        self.assertTrue(keys)
        spoof = chat(stream=True, cache_key=keys[0], litellm_params={"preset_cache_key": keys[0]},
                     cache={"use-cache": True, "namespace": keys[0].rsplit(":", 1)[0]},
                     metadata={"user_api_key": "test-key-a", "redis_namespace": "victim"})
        await self.complete(spoof, key="test-key-b")
        self.assertEqual(len(self.bodies), 2)

    async def test_stream_usage_and_cached_token_details_survive_response_cache(self):
        for options in (None, {}, {"include_usage": False}):
            payload = chat(stream=True)
            if options is not None:
                payload["stream_options"] = options
            for _ in range(2):
                chunks = await self.complete(payload)
                usage = [chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None]
                if options == {"include_usage": False}:
                    self.assertFalse(usage)
                else:
                    self.assertTrue(usage)
                    self.assertEqual(usage[-1].prompt_tokens, 12)
                    self.assertEqual(usage[-1].prompt_tokens_details.cached_tokens, 8)
        # Missing/empty options have identical effective settings; false is separate.
        self.assertEqual(len(self.bodies), 2)
        self.assertEqual(self.bodies[0]["stream_options"], {"include_usage": True})
        self.assertEqual(self.bodies[1]["stream_options"], {"include_usage": False})

    async def test_media_failure_returns_400_before_cache_or_provider(self):
        payload = chat(cache_salt="random", extra_body={"kv_cache_salt": "random", "priority": 99},
                       messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "bad-image"}}]}])
        with patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
             patch.object(self.cache, "get_cache", side_effect=AssertionError("cache read")) as sync_read, \
             patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write:
            with self.assertRaises(Exception) as error:
                await self.complete(payload)
            self.assertEqual(getattr(error.exception, "status_code", None), 400, repr(error.exception))
            read.assert_not_called()
            sync_read.assert_not_called()
            write.assert_not_called()
        self.assertTrue(self.prepared[-1]["cache"]["no-store"])
        self.assertFalse(self.bodies)

    async def test_valid_media_stays_uncacheable_and_gpu_salts_are_removed(self):
        import base64
        import io
        from PIL import Image
        with Image.new("RGB", (2, 2), "blue") as image, io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = chat(cache_salt="random", extra_body={"kv_cache_salt": "random", "priority": 99},
                       messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}])
        data = await self.prepare(payload)
        await self.callback.async_pre_call_deployment_hook(data, "acompletion")
        self.assertTrue(data["cache"]["no-store"])
        self.assertNotIn("cache_salt", data)
        self.assertNotIn("kv_cache_salt", data["extra_body"])
        self.assertNotIn("priority", data["extra_body"])
        self.assertEqual(data["messages"], payload["messages"])
        with patch.object(self.cache, "async_get_cache", side_effect=AssertionError("cache read")) as read, \
             patch.object(self.cache, "get_cache", side_effect=AssertionError("cache read")) as sync_read, \
             patch.object(self.cache, "async_add_cache", side_effect=AssertionError("cache write")) as write:
            await self.complete(payload)
            await self.complete(payload)
            read.assert_not_called()
            sync_read.assert_not_called()
            write.assert_not_called()
        self.assertEqual(len(self.bodies), 2)
        for body in self.bodies:
            self.assertEqual(body["messages"], payload["messages"])
            for name in ("cache_salt", "kv_cache_salt", "priority"):
                self.assertNotIn(name, body)

    async def test_normalization_runs_off_loop_and_cancellation_does_not_mutate_request(self):
        started, release = threading.Event(), threading.Event()
        original = chat()

        def blocked(data):
            started.set()
            release.wait(2)
            data["messages"][0]["content"] = "worker mutation"

        with patch("redesign.tenancy.callback.normalize_payload", side_effect=blocked):
            task = asyncio.create_task(self.callback.async_pre_call_deployment_hook(original, "acompletion"))
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(started.is_set())
                self.assertFalse(task.done(), "blocking normalizer ran on event loop")
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
        self.assertEqual(original["messages"][0]["content"], "hello")

    async def test_missing_proxy_context_cannot_enable_sdk_cache(self):
        data = chat(cache_key="victim", litellm_params={"preset_cache_key": "victim"})
        await self.callback.async_pre_call_deployment_hook(data, "acompletion")
        self.assertTrue(data["cache"]["no-store"])
        self.assertNotIn("cache_key", data)

    async def test_config_and_provider_flag_are_supported_by_installed_cache(self):
        import yaml
        from litellm.caching.caching import Cache
        from litellm.proxy._types import ConfigYAML
        config = yaml.safe_load(render(redis_host="unused.invalid"))
        # Config loader resolves env references before schema validation.
        validated = copy.deepcopy(config)
        validated["general_settings"]["coordination_redis"]["port"] = 6379
        ConfigYAML.model_validate(validated)
        params = config["litellm_settings"]["cache_params"]
        for name in params:
            self.assertIn(name, inspect.signature(Cache).parameters)
        self.assertEqual(params["mode"], "default_off")
        self.assertEqual(params["supported_call_types"], ["acompletion"])
        self.assertTrue(config["litellm_settings"]["enable_redis_auth_cache"])
        self.assertTrue(config["general_settings"]["always_include_stream_usage"])
        flag = config["litellm_settings"]["enable_caching_on_provider_specific_optional_params"]
        self.assertIs(flag, True)
        plain = {"model": "openai/FW-Kimi-K3", "messages": chat()["messages"]}
        with patch.object(self.litellm, "enable_caching_on_provider_specific_optional_params", flag):
            self.assertNotEqual(self.cache.get_cache_key(**plain, chat_template_kwargs={"thinking": True}),
                                self.cache.get_cache_key(**plain, chat_template_kwargs={"thinking": False}))
            self.assertNotEqual(self.cache.get_cache_key(**plain, extra_body={"top_k": 1}),
                                self.cache.get_cache_key(**plain, extra_body={"top_k": 2}))
        self.assertFalse(self.cache.should_use_cache(caching=True))
        self.assertTrue(self.cache.should_use_cache(cache={"use-cache": True}))


if __name__ == "__main__":
    unittest.main()
