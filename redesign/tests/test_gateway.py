"""Covers classification, clamping, backpressure and the composed policy."""

from __future__ import annotations

import unittest

from redesign.gateway import clamping
from redesign.gateway.backpressure import (
    CircuitBreaker,
    ClassBudget,
    StaticHealthSource,
)
from redesign.gateway.capture import MemorySink, TraceRecorder
from redesign.gateway.classification import (
    AGENTIC,
    ALL_CLASSES,
    BATCH,
    INTERACTIVE,
    LONG_CONTEXT,
    SHORT_CHAT,
    Classifier,
)
from redesign.gateway.clamping import TokenClamp
from redesign.gateway.models import EngineSnapshot, Outcome, Priority, RequestEnvelope
from redesign.gateway.policy import GatewayPolicy, build_default

WINDOW = 262_144


def _envelope(**overrides) -> RequestEnvelope:
    defaults = dict(customer="acme", prompt_tokens=1_000)
    return RequestEnvelope(**{**defaults, **overrides})


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier()

    def test_an_explicit_batch_hint_beats_every_length_rule(self):
        envelope = _envelope(prompt_tokens=500, batch_hint=True)
        self.assertIs(self.classifier.classify(envelope), BATCH)

    def test_a_short_toolless_prompt_is_short_chat(self):
        self.assertIs(self.classifier.classify(_envelope(prompt_tokens=4_000)), SHORT_CHAT)

    def test_tools_make_a_short_prompt_agentic_not_interactive(self):
        envelope = _envelope(prompt_tokens=4_000, has_tools=True)
        self.assertIs(self.classifier.classify(envelope), AGENTIC)

    def test_an_image_turn_is_agentic_at_any_length(self):
        envelope = _envelope(prompt_tokens=4_000, has_images=True)
        self.assertIs(self.classifier.classify(envelope), AGENTIC)

    def test_an_explicit_batch_hint_still_beats_the_agentic_rule(self):
        envelope = _envelope(prompt_tokens=4_000, has_tools=True, batch_hint=True)
        self.assertIs(self.classifier.classify(envelope), BATCH)

    def test_a_medium_prompt_is_interactive(self):
        self.assertIs(self.classifier.classify(_envelope(prompt_tokens=20_000)), INTERACTIVE)

    def test_anything_past_the_interactive_ceiling_is_long_context(self):
        self.assertIs(self.classifier.classify(_envelope(prompt_tokens=200_000)), LONG_CONTEXT)

    def test_rules_must_not_be_empty(self):
        with self.assertRaises(ValueError):
            Classifier(rules=())


class TokenClampTest(unittest.TestCase):
    def setUp(self):
        self.clamp = TokenClamp(max_model_len=WINDOW)

    def test_a_modest_request_passes_through(self):
        result = self.clamp.apply(1_000, 2_000, INTERACTIVE)
        self.assertEqual(result.granted, 2_000)
        self.assertEqual(result.reason, clamping.UNCHANGED)
        self.assertFalse(result.clamped)

    def test_an_oversized_request_is_cut_to_the_class_ceiling(self):
        result = self.clamp.apply(1_000, 128_000, INTERACTIVE)
        self.assertEqual(result.granted, INTERACTIVE.max_output_tokens)
        self.assertEqual(result.reason, clamping.CLASS_CEILING)
        self.assertTrue(result.clamped)

    def test_a_long_prompt_is_cut_to_the_remaining_window(self):
        result = self.clamp.apply(250_000, 128_000, LONG_CONTEXT)
        self.assertEqual(result.reason, clamping.CONTEXT_WINDOW)
        self.assertEqual(result.granted, self.clamp.available_for_output(250_000))

    def test_the_grant_always_fits_the_window(self):
        for prompt in (1_000, 100_000, 200_000, 245_000):
            result = self.clamp.apply(prompt, 128_000, LONG_CONTEXT)
            self.assertLessEqual(prompt + result.granted, WINDOW)

    def test_a_missing_max_tokens_gets_the_class_default(self):
        result = self.clamp.apply(1_000, None, SHORT_CHAT)
        self.assertEqual(result.granted, SHORT_CHAT.default_output_tokens)
        self.assertEqual(result.reason, clamping.APPLIED_DEFAULT)

    def test_a_prompt_filling_the_window_is_unservable(self):
        result = self.clamp.apply(WINDOW - 10, 1_000, LONG_CONTEXT)
        self.assertEqual(result.granted, 0)
        self.assertEqual(result.reason, clamping.PROMPT_TOO_LONG)


class CircuitBreakerTest(unittest.TestCase):
    def _breaker(self, **snapshot) -> CircuitBreaker:
        defaults = dict(kv_usage=0.5, running=10, waiting=0, preemptions_per_minute=0.0)
        source = StaticHealthSource(EngineSnapshot(**{**defaults, **snapshot}))
        return CircuitBreaker(source)

    def test_a_healthy_engine_sheds_nothing(self):
        self.assertFalse(self._breaker().should_shed(BATCH).distressed)

    def test_kv_pressure_sheds_batch(self):
        self.assertTrue(self._breaker(kv_usage=0.98).should_shed(BATCH).distressed)

    def test_a_full_pool_with_a_queue_is_distress_before_preemption_starts(self):
        breaker = self._breaker(kv_usage=0.96, running=74, waiting=21)
        self.assertTrue(breaker.should_shed(LONG_CONTEXT).distressed)

    def test_a_full_pool_with_no_queue_is_not_yet_distress(self):
        breaker = self._breaker(kv_usage=0.93, running=74, waiting=0)
        self.assertFalse(breaker.should_shed(LONG_CONTEXT).distressed)

    def test_preemptions_alone_are_distress(self):
        breaker = self._breaker(preemptions_per_minute=4.0)
        self.assertTrue(breaker.should_shed(LONG_CONTEXT).distressed)

    def test_interactive_is_never_shed(self):
        breaker = self._breaker(kv_usage=0.99, preemptions_per_minute=9.0)
        self.assertFalse(breaker.should_shed(INTERACTIVE).distressed)

    def test_a_broken_health_source_fails_open(self):
        class Broken(StaticHealthSource):
            def snapshot(self):
                raise RuntimeError("scrape failed")

        breaker = CircuitBreaker(Broken(EngineSnapshot.healthy()))
        self.assertFalse(breaker.evaluate().distressed)


class ClassBudgetTest(unittest.TestCase):
    def setUp(self):
        self.budget = ClassBudget.from_classes(100, ALL_CLASSES)

    def test_shares_follow_the_class_definitions(self):
        self.assertEqual(self.budget.limit_for(INTERACTIVE), 50)
        self.assertEqual(self.budget.limit_for(LONG_CONTEXT), 25)

    def test_an_unrouted_off_box_class_is_still_bounded_locally(self):
        self.assertGreater(self.budget.limit_for(SHORT_CHAT), 0)

    def test_a_routed_off_box_class_takes_no_local_slots(self):
        budget = ClassBudget.from_classes(100, ALL_CLASSES, offbox_configured=True)
        self.assertEqual(budget.limit_for(SHORT_CHAT), 0)

    def test_slots_are_exhausted_then_refused(self):
        for _ in range(25):
            self.assertTrue(self.budget.try_acquire(LONG_CONTEXT))
        self.assertFalse(self.budget.try_acquire(LONG_CONTEXT))

    def test_releasing_returns_a_slot(self):
        for _ in range(25):
            self.budget.try_acquire(LONG_CONTEXT)
        self.budget.release(LONG_CONTEXT)
        self.assertTrue(self.budget.try_acquire(LONG_CONTEXT))

    def test_release_never_goes_negative(self):
        self.budget.release(LONG_CONTEXT)
        self.assertEqual(self.budget.in_flight(LONG_CONTEXT), 0)


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = build_default(max_model_len=WINDOW, concurrency_ceiling=96)

    def test_a_normal_request_is_admitted_with_a_priority(self):
        decision = self.policy.decide(_envelope(prompt_tokens=20_000))
        self.assertTrue(decision.admitted)
        self.assertIs(decision.priority, Priority.INTERACTIVE)

    def test_an_agentic_turn_preserves_an_explicit_budget_below_its_ceiling(self):
        decision = self.policy.decide(
            _envelope(prompt_tokens=20_000, has_tools=True, requested_max_tokens=8_192)
        )
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.traffic_class, AGENTIC.name)
        self.assertEqual(decision.clamp.granted, 8_192)
        self.assertEqual(decision.clamp.reason, clamping.UNCHANGED)

    def test_a_long_prompt_with_a_fixed_max_tokens_is_rescued(self):
        decision = self.policy.decide(
            _envelope(prompt_tokens=200_000, requested_max_tokens=128_000)
        )
        self.assertTrue(decision.admitted)
        self.assertTrue(decision.rescued_from_rejection)
        self.assertLessEqual(200_000 + decision.clamp.granted, WINDOW)

    def test_an_unservable_prompt_is_refused_with_a_reason(self):
        decision = self.policy.decide(_envelope(prompt_tokens=WINDOW))
        self.assertFalse(decision.admitted)
        self.assertIn("no room for output", decision.reason)

    def test_short_chat_takes_a_local_slot_while_no_off_box_target_exists(self):
        """Honouring served_off_box with nowhere to route would admit this
        traffic to the local engine while skipping its concurrency slot."""
        decision = self.policy.decide(_envelope(prompt_tokens=2_000))
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.notes, ())

    def test_short_chat_skips_the_local_budget_once_off_box_is_configured(self):
        policy = build_default(
            max_model_len=WINDOW, concurrency_ceiling=96, offbox_configured=True
        )
        decision = policy.decide(_envelope(prompt_tokens=2_000))
        self.assertTrue(decision.admitted)
        self.assertIn("routed off-box", decision.notes)

    def test_a_distressed_engine_sheds_long_context(self):
        policy = GatewayPolicy(
            classifier=Classifier(),
            clamp=TokenClamp(max_model_len=WINDOW),
            budget=ClassBudget.from_classes(96, ALL_CLASSES),
            breaker=CircuitBreaker(
                StaticHealthSource(EngineSnapshot(0.99, 60, 30, 5.0))
            ),
        )
        decision = policy.decide(_envelope(prompt_tokens=100_000))
        self.assertIs(decision.outcome, Outcome.REJECT_SHED)
        self.assertIsNotNone(decision.retry_after_seconds)

    def test_a_distressed_engine_still_serves_interactive(self):
        policy = GatewayPolicy(
            classifier=Classifier(),
            clamp=TokenClamp(max_model_len=WINDOW),
            budget=ClassBudget.from_classes(96, ALL_CLASSES),
            breaker=CircuitBreaker(
                StaticHealthSource(EngineSnapshot(0.99, 60, 30, 5.0))
            ),
        )
        decision = policy.decide(_envelope(prompt_tokens=20_000))
        self.assertTrue(decision.admitted)

    def test_a_distressed_engine_sheds_agentic_traffic(self):
        """The live regression: `tools` made these P0, which is never shed."""
        policy = GatewayPolicy(
            classifier=Classifier(),
            clamp=TokenClamp(max_model_len=WINDOW),
            budget=ClassBudget.from_classes(96, ALL_CLASSES),
            breaker=CircuitBreaker(
                StaticHealthSource(EngineSnapshot(0.99, 60, 30, 5.0))
            ),
        )
        decision = policy.decide(_envelope(prompt_tokens=20_000, has_tools=True))
        self.assertIs(decision.outcome, Outcome.REJECT_SHED)

    def test_budget_exhaustion_returns_retry_after(self):
        envelope = _envelope(prompt_tokens=100_000)
        limit = ClassBudget.from_classes(96, ALL_CLASSES).limit_for(LONG_CONTEXT)
        for _ in range(limit):
            self.assertTrue(self.policy.decide(envelope).admitted)
        decision = self.policy.decide(envelope)
        self.assertIs(decision.outcome, Outcome.REJECT_BUDGET)
        self.assertEqual(decision.retry_after_seconds, 30)


class TraceRecorderTest(unittest.TestCase):
    def setUp(self):
        self.sink = MemorySink()
        self.recorder = TraceRecorder(self.sink)
        self.policy = build_default(max_model_len=WINDOW, concurrency_ceiling=96)

    def test_a_decision_is_recorded_as_shape_only(self):
        decision = self.policy.decide(_envelope(prompt_tokens=20_000, has_tools=True))
        record = self.recorder.record(decision)
        self.assertEqual(record.prompt_tokens, 20_000)
        self.assertEqual(record.outcome, "ADMIT")
        self.assertNotIn("acme", str(record))

    def test_customer_labels_are_hashed_by_default(self):
        decision = self.policy.decide(_envelope())
        self.assertNotEqual(self.recorder.record(decision).customer, "acme")

    def test_hashing_can_be_disabled_for_local_analysis(self):
        recorder = TraceRecorder(MemorySink(), hash_customers=False)
        decision = self.policy.decide(_envelope())
        self.assertEqual(recorder.record(decision).customer, "acme")

    def test_every_decision_reaches_the_sink(self):
        for tokens in (1_000, 20_000, 200_000):
            self.recorder.record(self.policy.decide(_envelope(prompt_tokens=tokens)))
        self.assertEqual(len(self.sink.records), 3)


if __name__ == "__main__":
    unittest.main()
