"""Output budgets across tenancy, the real gateway handler and admission leases.

Prompt estimation, classification, clamping, relay ownership and context-work
accounting use the source implementation; HTTP I/O is mocked.
"""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

from redesign.gateway.backpressure import CircuitBreaker, ClassBudget, StaticHealthSource
from redesign.gateway.capture import MemorySink, TraceRecorder
from redesign.gateway.classification import (
    AGENTIC, ALL_CLASSES, BATCH, INTERACTIVE, LONG_CONTEXT, SHORT_CHAT, Classifier, TrafficClass,
)
from redesign.gateway.clamping import (
    APPLIED_DEFAULT, CLASS_CEILING, CONTEXT_WINDOW, PROMPT_TOO_LONG, RESERVE_TOKENS,
    UNCHANGED, InvalidTokenLimit, TokenClamp,
)
from redesign.gateway.metrics import Registry
from redesign.gateway.models import EngineSnapshot, Priority, RequestEnvelope
from redesign.gateway.offbox import OffBoxClient
from redesign.gateway.policy import GatewayPolicy, build_default
from redesign.gateway.server import GatewayService
from redesign.gateway.tokens import HeuristicEstimator
from redesign.tenancy.policy import TenancyPolicy
from redesign.tests.test_serving_invariants import fake_engine, fake_handler

WINDOW = 262_144
MODEL = "selected-k3"
PATH = "/v1/chat/completions"
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="


def chat(text="hi", **kwargs):
    return {"model": MODEL, "messages": [{"role": "user", "content": text}], **kwargs}


def file_tool(**kwargs):
    return {"type": "function", "function": {
        "name": "write_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }, **kwargs,
    }}


def relay_through_gateway(payload, *, policy=None, estimator=None, headers=None):
    """Also used by installed-LiteLLM tests at their mock HTTP transport boundary."""
    handler, _, connection, _ = fake_handler(copy.deepcopy(payload))
    policy = policy if policy is not None else build_default(WINDOW, 64)
    handler.service.policy = policy
    handler.service.estimator = estimator if estimator is not None else HeuristicEstimator()
    handler.headers = headers or {}
    sink, during = MemorySink(), []
    handler.service.recorder = TraceRecorder(sink)
    connection.request.side_effect = lambda *args, **kwargs: during.append(policy.budget_snapshot())
    handler.do_POST()
    error = None
    if handler._json.called:
        status, error = handler._json.call_args.args[:2]
        response_headers = handler._json.call_args.kwargs.get("extra_headers", {})
    else:
        status = handler.send_response.call_args.args[0]
        response_headers = dict(call.args for call in handler.send_header.call_args_list)
    return SimpleNamespace(
        status=status, error=error, headers=response_headers,
        forwarded=json.loads(connection.request.call_args.kwargs["body"]) if connection.request.called else None,
        record=sink.records[-1] if sink.records else None,
        during=during[0] if during else None, after=policy.budget_snapshot(),
    )


class OutputClampTest(unittest.TestCase):
    def test_class_defaults_are_separate_from_hard_ceilings(self):
        for traffic_class, default, ceiling in (
            (INTERACTIVE, 2_048, 8_192), (SHORT_CHAT, 1_024, 4_096),
            (LONG_CONTEXT, 4_096, 16_384), (BATCH, 8_192, 32_768), (AGENTIC, 2_048, 16_384),
        ):
            with self.subTest(traffic_class=traffic_class.name):
                self.assertEqual(traffic_class.default_output_tokens, default)
                self.assertEqual(traffic_class.max_output_tokens, ceiling)
                clamp = TokenClamp(WINDOW)
                implicit = clamp.apply(1_000, None, traffic_class)
                self.assertEqual((implicit.granted, implicit.reason), (default, APPLIED_DEFAULT))
                self.assertTrue(implicit.default_applied)
                for requested in (1, 16, default + 1, ceiling):
                    explicit = clamp.apply(1_000, requested, traffic_class)
                    self.assertEqual((explicit.granted, explicit.reason), (requested, UNCHANGED))
                    self.assertFalse(explicit.default_applied)
                bounded = clamp.apply(1_000, ceiling * 2, traffic_class)
                self.assertEqual((bounded.granted, bounded.reason), (ceiling, CLASS_CEILING))

    def test_positional_api_and_custom_max_as_default_remain_compatible(self):
        legacy = TrafficClass("custom", Priority.BATCH, 73, None, 0.25, True)
        self.assertTrue(legacy.served_off_box)
        self.assertIsNone(legacy.default_output_tokens)
        clamp = TokenClamp(WINDOW)
        self.assertEqual(clamp.apply(1_000, None, legacy).granted, 73)
        # Even an operator default above a custom class ceiling cannot expand it.
        result = clamp.apply(1_000, None, replace(legacy, default_output_tokens=100))
        self.assertEqual((result.granted, result.reason), (73, CLASS_CEILING))
        self.assertTrue(result.default_applied)
        envelope = RequestEnvelope("legacy", 10, 16, False, True, False, False, PATH)
        self.assertEqual(envelope.path, PATH)
        self.assertFalse(envelope.tools_disabled)

    def test_default_reports_a_binding_window_cap_without_losing_default_provenance(self):
        for traffic_class in ALL_CLASSES:
            with self.subTest(traffic_class=traffic_class.name):
                result = TokenClamp(WINDOW).apply(WINDOW - RESERVE_TOKENS - 512, None, traffic_class)
                self.assertEqual((result.granted, result.reason), (512, CONTEXT_WINDOW))
                self.assertEqual(result.default_output_tokens, traffic_class.default_output_tokens)
                self.assertIsNone(result.requested)
                self.assertTrue(result.default_applied)

    def test_small_valid_output_needs_only_its_requested_space(self):
        self.assertEqual(RESERVE_TOKENS, 256)
        clamp = TokenClamp(WINDOW)
        for requested in (1, 16, 63, 64):
            with self.subTest(requested=requested):
                prompt = WINDOW - RESERVE_TOKENS - requested
                exact = clamp.apply(prompt, requested, AGENTIC)
                self.assertEqual((exact.granted, exact.reason), (requested, UNCHANGED))
                self.assertFalse(exact.clamped)
                short = clamp.apply(prompt + 1, requested, AGENTIC)
                self.assertEqual((short.granted, short.reason), (0, PROMPT_TOO_LONG))
        for requested in (None, 65, 8_192):
            result = clamp.apply(WINDOW - RESERVE_TOKENS - 63, requested, AGENTIC)
            self.assertEqual(result.reason, PROMPT_TOO_LONG)
        result = clamp.apply(WINDOW - RESERVE_TOKENS - 64, 8_192, AGENTIC)
        self.assertEqual((result.granted, result.reason), (64, CONTEXT_WINDOW))

    def test_invalid_direct_clamp_inputs_never_become_defaults(self):
        for requested in (True, False, 0, -1, "16", 1.5):
            with self.subTest(requested=requested), self.assertRaises(InvalidTokenLimit):
                TokenClamp(WINDOW).apply(1_000, requested, AGENTIC)


class ComposedOutputPolicyTest(unittest.TestCase):
    def compose(self, payload, *, tenancy=None, gateway=None, headers=None):
        data = copy.deepcopy(payload)
        tenancy = tenancy if tenancy is not None else TenancyPolicy()
        decision = tenancy.apply(data, data["model"], route_model=False)
        self.assertFalse(decision.reject)
        result = relay_through_gateway(data, policy=gateway, headers=headers)
        self.assertEqual(result.status, 200, result.error)
        self.assertEqual(result.after.context_tokens_in_flight, 0)
        self.assertEqual(result.after.in_flight, 0)
        self.assertEqual(result.forwarded["model"], payload["model"])
        return decision, result

    def test_enabled_and_forced_file_tools_preserve_8192_across_both_hops(self):
        for choices in (
            {}, {"tool_choice": None}, {"tool_choice": "auto"}, {"tool_choice": "required"},
            {"tool_choice": {"type": "function", "function": {"name": "write_file"}}},
            {"function_call": {"name": "write_file"}},
        ):
            with self.subTest(choices=choices):
                payload = chat("Write the complete file", tools=[file_tool()], max_completion_tokens=8_192, **choices)
                decision, result = self.compose(payload)
                self.assertEqual(decision.traffic_class, AGENTIC.name)
                self.assertEqual(decision.requested_max_tokens, 8_192)
                self.assertEqual(decision.granted_max_tokens, 8_192)
                self.assertEqual(decision.clamp_reason, UNCHANGED)
                self.assertEqual(result.forwarded["max_tokens"], 8_192)
                self.assertEqual(result.forwarded["max_completion_tokens"], 8_192)
                self.assertEqual(result.record.traffic_class, AGENTIC.name)
                self.assertEqual(result.record.clamp_reason, UNCHANGED)
                for field, value in choices.items():
                    self.assertEqual(result.forwarded[field], value)

    def test_only_literal_none_disables_tools_and_text_then_uses_length(self):
        for field in ("tool_choice", "function_call"):
            for text, traffic_class in (("hi", SHORT_CHAT), ("x" * 35_000, INTERACTIVE), ("x" * 140_000, LONG_CONTEXT)):
                with self.subTest(field=field, traffic_class=traffic_class.name):
                    payload = chat(text, tools=[file_tool()], max_tokens=8_192, **{field: "none"})
                    decision, result = self.compose(payload)
                    expected = min(8_192, traffic_class.max_output_tokens)
                    self.assertEqual(decision.traffic_class, traffic_class.name)
                    self.assertEqual(result.record.traffic_class, traffic_class.name)
                    self.assertEqual(result.forwarded["max_tokens"], expected)
                    self.assertTrue(result.record.has_tools)
                    self.assertEqual(result.forwarded["tools"], payload["tools"])
        for value in (None, False, "None", "NONE", "auto", {"name": "none"}):
            for field in ("tool_choice", "function_call"):
                with self.subTest(field=field, value=value):
                    decision, result = self.compose(chat(tools=[file_tool()], **{field: value}))
                    self.assertEqual(decision.traffic_class, AGENTIC.name)
                    self.assertEqual(result.forwarded["max_tokens"], 2_048)

    def test_disabled_legacy_functions_and_their_schemas_still_count_in_prompt(self):
        for field in ("tools", "functions"):
            tool = file_tool(description="z" * 30_000)
            definitions = [tool] if field == "tools" else [tool["function"]]
            payload = chat(**{field: definitions}, function_call="none", max_tokens=8_192)
            with self.subTest(field=field):
                decision, result = self.compose(payload)
                self.assertGreater(result.record.prompt_tokens, 8_192)
                self.assertLess(HeuristicEstimator().estimate(chat()), 8_192)
                self.assertEqual(decision.traffic_class, INTERACTIVE.name)
                self.assertEqual(result.record.traffic_class, INTERACTIVE.name)
                self.assertEqual(result.forwarded["max_tokens"], 8_192)
                self.assertEqual(result.forwarded[field], definitions)

    def test_request_priority_and_diagnostic_flags_cannot_override_classification(self):
        payload = chat(tools=[file_tool()], tools_disabled=True, priority=0,
                       metadata={"k3_class": INTERACTIVE.name, "k3_output_policy": {"requested_max_tokens": 99_999}})
        decision, result = self.compose(payload)
        self.assertEqual(decision.traffic_class, AGENTIC.name)
        self.assertIsNone(decision.requested_max_tokens)
        self.assertEqual(result.forwarded["priority"], int(Priority.LONG_CONTEXT))
        self.assertEqual(result.forwarded["max_tokens"], 2_048)

    def test_images_keep_agentic_default_and_ceiling_even_with_tools_disabled(self):
        for requested, expected in ((None, 2_048), (16, 16), (8_192, 8_192), (32_768, 16_384)):
            with self.subTest(requested=requested):
                payload = chat([{"type": "image_url", "image_url": {"url": PNG}}],
                               tool_choice="none", max_tokens=requested)
                decision, result = self.compose(payload)
                self.assertEqual(decision.traffic_class, AGENTIC.name)
                self.assertEqual(result.record.traffic_class, AGENTIC.name)
                self.assertEqual(decision.granted_max_tokens, expected)
                self.assertEqual(result.forwarded["max_tokens"], expected)
                self.assertEqual(decision.default_output_tokens, 2_048)
                self.assertEqual(result.forwarded["messages"][0]["content"][0]["type"], "image_url")

    def test_every_class_uses_its_default_only_when_the_input_budget_is_missing(self):
        for payload, traffic_class, headers in (
            (chat(), SHORT_CHAT, {}), (chat("x" * 35_000), INTERACTIVE, {}),
            (chat("x" * 140_000), LONG_CONTEXT, {}), (chat(tools=[file_tool()]), AGENTIC, {}),
            (chat(k3_batch=True), BATCH, {"x-k3-batch": "1"}),
        ):
            with self.subTest(traffic_class=traffic_class.name):
                decision, result = self.compose(payload, headers=headers)
                expected = traffic_class.default_output_tokens
                self.assertEqual(decision.traffic_class, traffic_class.name)
                self.assertEqual((decision.granted_max_tokens, decision.clamp_reason), (expected, APPLIED_DEFAULT))
                self.assertTrue(decision.default_applied)
                self.assertIsNone(decision.requested_max_tokens)
                self.assertEqual(result.forwarded["max_tokens"], expected)
                self.assertEqual(result.record.clamp_reason, UNCHANGED)
                self.assertEqual(result.headers["x-k3-output-default-applied"], "false")

    def test_reapplication_is_idempotent_but_diagnostics_are_hop_local(self):
        for requested, expected in ((None, 2_048), (1, 1), (16, 16), (8_192, 8_192), (99_999, 16_384)):
            with self.subTest(requested=requested):
                payload = chat("private prompt sentinel", tools=[file_tool()], max_tokens=requested)
                direct = relay_through_gateway(payload)
                self.assertEqual(direct.status, 200)
                self.assertEqual(direct.record.requested_max_tokens, requested)
                self.assertEqual(direct.forwarded["max_tokens"], expected)
                self.assertEqual(direct.headers["x-k3-max-tokens-requested"], str(requested) if requested is not None else "none")
                self.assertEqual(direct.headers["x-k3-output-default-applied"], str(requested is None).lower())
                decision, result = self.compose(payload)
                self.assertEqual(decision.requested_max_tokens, requested)
                self.assertEqual(result.record.requested_max_tokens, expected)
                self.assertEqual(result.headers["x-k3-max-tokens-requested"], str(expected))
                self.assertEqual(result.headers["x-k3-output-policy-stage"], "gateway")
                self.assertEqual(result.headers["x-k3-output-tokens-default"], "2048")
                self.assertEqual(result.headers["x-k3-clamp-reason"], UNCHANGED)
                repeated, again = self.compose(result.forwarded)
                self.assertEqual(repeated.requested_max_tokens, expected)
                self.assertFalse(repeated.default_applied)
                self.assertEqual(repeated.clamp_reason, UNCHANGED)
                self.assertEqual(again.forwarded["max_tokens"], expected)
                self.assertEqual(again.forwarded, result.forwarded)
                diagnostics = json.dumps({"tenancy": asdict(decision), "trace": asdict(result.record), "headers": result.headers})
                self.assertNotIn("private prompt sentinel", diagnostics)
                self.assertNotIn("original_max_tokens", diagnostics)

    def test_disabled_tools_keep_selected_model_until_gateway_offbox_dispatch(self):
        data = chat(tools=[file_tool()], tool_choice="none", max_tokens=8192)
        decision = TenancyPolicy(offbox_configured=True).apply(data, MODEL, route_model=False)
        self.assertEqual(decision.traffic_class, SHORT_CHAT.name)
        self.assertFalse(decision.routed_off_box)
        self.assertEqual(data["model"], MODEL)
        handler, _, local_connection, _ = fake_handler(data)
        handler.service.policy = build_default(WINDOW, 64, offbox_configured=True)
        handler.service.estimator = HeuristicEstimator()
        remote_engine, remote_connection, _ = fake_engine()
        offbox = OffBoxClient("http://offbox.invalid", model="remote-k3")
        offbox.engine = remote_engine
        handler.service.offbox = offbox
        handler.do_POST()
        local_connection.request.assert_not_called()
        forwarded = json.loads(remote_connection.request.call_args.kwargs["body"])
        self.assertEqual(forwarded["model"], "remote-k3")
        self.assertEqual(forwarded["max_tokens"], 4096)
        self.assertEqual(forwarded["tool_choice"], "none")
        self.assertEqual(forwarded["tools"], data["tools"])
        self.assertEqual(handler.service.policy.budget_snapshot().context_tokens_in_flight, 0)
        self.assertEqual(data["model"], MODEL)

    def test_different_window_caps_only_reduce_limits_at_each_hop(self):
        payload = chat(tools=[file_tool()], max_tokens=8_192)
        prompt = HeuristicEstimator().estimate(payload)
        tenancy = TenancyPolicy(max_model_len=prompt + RESERVE_TOKENS + 4_096)
        gateway = build_default(prompt + RESERVE_TOKENS + 1_024, 64)
        decision, result = self.compose(payload, tenancy=tenancy, gateway=gateway)
        self.assertEqual((decision.granted_max_tokens, decision.clamp_reason), (4_096, CONTEXT_WINDOW))
        self.assertEqual(result.record.requested_max_tokens, 4_096)
        self.assertEqual((result.forwarded["max_tokens"], result.record.clamp_reason), (1_024, CONTEXT_WINDOW))
        self.assertEqual(result.during.context_tokens_in_flight, prompt + 1_024)
        repeated, again = self.compose(result.forwarded)
        self.assertEqual(repeated.granted_max_tokens, 1_024)
        self.assertEqual(again.forwarded["max_tokens"], 1_024)

    def test_small_output_fits_an_exact_window_at_both_hops(self):
        for requested in (1, 16):
            with self.subTest(requested=requested):
                payload = chat(tools=[file_tool()], max_tokens=requested)
                prompt = HeuristicEstimator().estimate(payload)
                window = prompt + RESERVE_TOKENS + requested
                decision, result = self.compose(
                    payload, tenancy=TenancyPolicy(max_model_len=window), gateway=build_default(window, 64),
                )
                self.assertEqual(decision.granted_max_tokens, requested)
                self.assertEqual(result.forwarded["max_tokens"], requested)
                self.assertEqual(result.during.context_tokens_in_flight, prompt + requested)


class OutputAdmissionIntegrationTest(unittest.TestCase):
    def test_step1_leases_charge_exact_new_grants_reject_and_release_once(self):
        for requested, granted in ((None, 2_048), (8_192, 8_192), (99_999, 16_384)):
            with self.subTest(requested=requested):
                data = chat(tools=[file_tool()], max_tokens=requested)
                estimator = HeuristicEstimator()
                tenancy = TenancyPolicy(estimator=estimator).apply(data, MODEL, route_model=False)
                self.assertEqual(tenancy.granted_max_tokens, granted)
                cost = estimator.estimate(data) + granted
                budget = ClassBudget.from_classes(
                    64, ALL_CLASSES, borrowing_enabled=True, context_token_budget=2 * cost,
                )
                policy = GatewayPolicy(
                    Classifier(), TokenClamp(WINDOW), budget,
                    CircuitBreaker(StaticHealthSource(EngineSnapshot.healthy())),
                )
                service = GatewayService(policy, Mock(), estimator, TraceRecorder(MemorySink()), Registry(), True)
                envelope = service.envelope(data, {}, PATH)
                first, second = policy.decide(envelope), policy.decide(envelope)
                self.assertTrue(first.admitted and second.admitted)
                self.assertEqual(first.metadata["context_tokens"], cost)
                self.assertEqual(second.metadata["context_tokens_in_flight"], 2 * cost)

                refused = relay_through_gateway(data, policy=policy, estimator=estimator)
                self.assertEqual(refused.status, 429)
                self.assertEqual(refused.error["error"]["k3_admission_reason"], "token")
                self.assertIsNone(refused.forwarded)
                self.assertEqual(refused.after.context_tokens_in_flight, 2 * cost)
                self.assertEqual(refused.headers["x-k3-max-tokens-granted"], str(granted))

                policy.release(first)
                policy.release(replace(first))
                self.assertEqual(policy.budget_snapshot().context_tokens_in_flight, cost)
                admitted = relay_through_gateway(data, policy=policy, estimator=estimator)
                self.assertEqual(admitted.status, 200)
                self.assertEqual(admitted.forwarded["max_tokens"], granted)
                self.assertEqual(admitted.during.context_tokens_in_flight, 2 * cost)
                self.assertEqual(admitted.after.context_tokens_in_flight, cost)
                self.assertEqual(admitted.after.in_flight, 1)
                policy.release(second)
                self.assertEqual(policy.budget_snapshot().context_tokens_in_flight, 0)
                self.assertEqual(policy.budget_snapshot().in_flight, 0)


if __name__ == "__main__":
    unittest.main()
