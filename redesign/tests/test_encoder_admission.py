"""Offline encoder admission: rendered-count bounds, mixed traffic and lease races.

Rendered-count fixtures are engine response inputs to the policy, not heuristic
image estimates. The 21,050 / 17,847 prompt layouts and embedding counts match
the constructed metadata cases in experiments/2026-09-21-encoder-deadlock.
No engine, tokenizer HTTP, model, media processing or deployment is invoked.
"""

from __future__ import annotations

import json
import threading
import unittest
from collections import Counter
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

from redesign.gateway.backpressure import CircuitBreaker, ClassBudget, StaticHealthSource
from redesign.gateway.classification import (
    AGENTIC, ALL_CLASSES, BATCH, INTERACTIVE, LONG_CONTEXT, SHORT_CHAT, Classifier,
)
from redesign.gateway.clamping import TokenClamp
from redesign.gateway.engine_tokens import TokenCountResult
from redesign.gateway.models import (
    AdmissionReason, CountSource, EngineSnapshot, Outcome, Priority, RequestEnvelope,
)
from redesign.gateway.policy import GatewayPolicy, build_default


ENCODER_CAPACITY = 262_144
CONTEXT_BUDGET = 1_310_000


def _health(**overrides):
    return replace(EngineSnapshot(0.5, 12, 0, 0.0), **overrides)


def _budget(**overrides):
    return ClassBudget.from_classes(64, ALL_CLASSES, **{
        "borrowing_enabled": True,
        "context_token_budget": CONTEXT_BUDGET,
        "encoder_token_budget": ENCODER_CAPACITY,
        **overrides,
    })


def _policy(budget, *, source=None, classifier=None, offbox=False):
    return GatewayPolicy(
        classifier if classifier is not None else Classifier(),
        TokenClamp(262_144), budget,
        CircuitBreaker(source if source is not None else StaticHealthSource(_health())),
        offbox_configured=offbox,
    )


def _image(prompt_tokens=21_050, **overrides):
    return replace(RequestEnvelope(
        "encoder-test", prompt_tokens, requested_max_tokens=64,
        streaming=True, has_images=True,
        prompt_count_source=CountSource.ENGINE_RENDERED,
    ), **overrides)


def _acquire_image(budget, prompt_tokens=21_050, *, traffic_class=AGENTIC, health=None):
    return budget.try_acquire_lease(
        traffic_class, prompt_tokens + 64, health if health is not None else _health(),
        has_images=True, prompt_tokens=prompt_tokens,
        prompt_count_source=CountSource.ENGINE_RENDERED,
    )


def _race(count, action):
    start = threading.Barrier(count)

    def run(index):
        start.wait(timeout=15)
        return action(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(run, range(count)))


class EncoderAdmissionTest(unittest.TestCase):
    def assert_empty(self, budget):
        state = budget.snapshot()
        self.assertEqual(state.in_flight, 0)
        self.assertEqual(sum(state.by_class.values()), 0)
        self.assertEqual(state.context_tokens_in_flight, 0)
        self.assertEqual(state.encoder_tokens_in_flight, 0)
        self.assertEqual(state.borrowed_in_flight, 0)
        self.assertEqual(state.p0_in_flight, 0)

    def assert_bounded(self, state):
        self.assertEqual(state.in_flight, sum(state.by_class.values()))
        self.assertLessEqual(state.in_flight, 64)
        self.assertLessEqual(state.in_flight - state.p0_in_flight, 60)
        self.assertGreaterEqual(state.encoder_tokens_in_flight, 0)
        self.assertLessEqual(state.encoder_tokens_in_flight, ENCODER_CAPACITY)
        self.assertGreaterEqual(state.context_tokens_in_flight, 0)
        self.assertLessEqual(state.context_tokens_in_flight, CONTEXT_BUDGET)

    def test_balanced_image_and_text_traffic_fills_encoder_budget_exactly(self):
        budget = _budget()
        policy = _policy(budget)
        held = []
        # Variable per-request FULL prompt costs, deliberately totaling the cap.
        image_counts = (21_050, 17_847, 70_001, 153_246)
        for count in image_counts:
            held.append(policy.decide(_image(count)))
            held.append(policy.decide(_image(
                10_000, has_images=False, has_tools=True,
                prompt_count_source=CountSource.HEURISTIC,
            )))
        self.assertTrue(all(decision.admitted for decision in held))
        self.assertEqual(len({d.admission_id for d in held}), 8)
        state = policy.budget_snapshot()
        self.assertEqual(state.encoder_tokens_in_flight, sum(image_counts))
        self.assertEqual(state.encoder_tokens_in_flight, state.encoder_token_budget)
        self.assertEqual(state.context_tokens_in_flight, sum(image_counts) + 40_000 + 8 * 64)
        refused = policy.decide(_image(17_847))
        self.assertIs(refused.outcome, Outcome.REJECT_BUDGET)
        self.assertIs(refused.metadata["admission_reason"], AdmissionReason.ENCODER)
        self.assertEqual(refused.metadata["encoder_tokens_in_flight"], ENCODER_CAPACITY)
        self.assertEqual(refused.metadata["encoder_token_budget"], ENCODER_CAPACITY)
        self.assertEqual(refused.retry_after_seconds, 2)
        self.assertIsNone(refused.admission_id)
        self.assertFalse(policy.prefill_complete(refused))
        policy.release(refused)
        self.assertEqual(policy.budget_snapshot(), state)
        for decision in held:
            self.assertEqual(
                decision.metadata["encoder_tokens"],
                decision.envelope.prompt_tokens if decision.envelope.has_images else 0,
            )
            policy.release(decision)
        self.assert_empty(budget)

    def test_full_rendered_prompt_is_an_exact_reservation_and_embedding_upper_bound(self):
        # The two metadata-reproducer layouts include wrappers and trailing text.
        # The last fixture exercises nearly the whole window, far above the
        # unchanged 16,817-token per-step encoder compute budget.
        fixtures = (
            (TokenCountResult(21_050, 262_144), (10_000, 10_000)),
            (TokenCountResult(17_847, 262_144), (16_817,)),
            (TokenCountResult(262_143, 262_144), (16_817,) * 15),
        )
        for count, embeddings in fixtures:
            for cached in (False, True):
                with self.subTest(prompt=count.count, cached=cached):
                    # Many requests have identical prompts/count-cache hits;
                    # each must still reserve its own full prompt count.
                    budget = _budget()
                    policy = _policy(budget)
                    envelope = _image(
                        count.count, requested_max_tokens=1,
                        prompt_count_source=count.source, prompt_count_cached=cached,
                        model_context_limit=count.max_model_len,
                    )
                    held = []
                    for _ in range(ENCODER_CAPACITY // count.count):
                        decision = policy.decide(envelope)
                        self.assertTrue(decision.admitted)
                        self.assertEqual(decision.metadata["encoder_tokens"], count.count)
                        self.assertEqual(decision.metadata["context_tokens"], count.count + 1)
                        held.append(decision)
                        reserved = policy.budget_snapshot().encoder_tokens_in_flight
                        self.assertEqual(reserved, len(held) * count.count)
                        self.assertLessEqual(len(held) * sum(embeddings), reserved)
                        self.assertLessEqual(reserved, ENCODER_CAPACITY)
                    refused = policy.decide(envelope)
                    self.assertEqual(refused.metadata["admission_reason"], "encoder")
                    self.assertGreater((len(held) + 1) * count.count, ENCODER_CAPACITY)
                    for decision in held:
                        policy.release(decision)
                    self.assert_empty(budget)

    def test_exact_capacity_admits_but_one_more_token_does_not(self):
        budget = _budget()
        before = budget.snapshot()
        too_large = _acquire_image(budget, ENCODER_CAPACITY + 1)
        self.assertIs(too_large.reason, AdmissionReason.ENCODER)
        self.assertEqual(budget.snapshot(), before)
        exact = _acquire_image(budget, ENCODER_CAPACITY)
        self.assertTrue(exact.admitted)
        self.assertEqual(exact.lease.encoder_tokens, ENCODER_CAPACITY)
        self.assertEqual(_acquire_image(budget, 1).reason, "encoder")
        budget.release_lease(exact.lease.admission_id)
        self.assert_empty(budget)

    def test_missing_untrusted_or_invalid_image_count_has_typed_rejection(self):
        budget = _budget()
        policy = _policy(budget)
        before = budget.snapshot()
        for source in (CountSource.HEURISTIC, CountSource.UNKNOWN, "", "estimated", None):
            with self.subTest(source=source):
                refused = policy.decide(_image(prompt_count_source=source))
                self.assertIs(refused.metadata["admission_reason"], AdmissionReason.ENCODER_COUNT)
                self.assertIs(refused.outcome, Outcome.REJECT_BUDGET)
                self.assertIsNone(refused.admission_id)
                self.assertIsNone(refused.retry_after_seconds)
                self.assertEqual(refused.metadata["prompt_count_source"], source)
                self.assertEqual(refused.metadata["required_prompt_count_source"], "engine_rendered")
                self.assertFalse(policy.prefill_complete(refused))
                policy.release(refused)
                self.assertEqual(budget.snapshot(), before)
        for count in (None, -1, True, 1.5, "21050"):
            with self.subTest(count=count):
                refused = budget.try_acquire_lease(
                    AGENTIC, 21_114, _health(), has_images=True,
                    prompt_tokens=count, prompt_count_source=CountSource.ENGINE_RENDERED,
                )
                self.assertIs(refused.reason, AdmissionReason.ENCODER_COUNT)
                self.assertEqual(budget.snapshot(), before)

    def test_text_is_never_encoder_gated_even_in_image_class_or_above_encoder_limit(self):
        budget = _budget(encoder_token_budget=17_847)
        policy = _policy(budget)
        image = policy.decide(_image(17_847))
        self.assertTrue(image.admitted)
        for overrides, expected_class in (
            ({"prompt_tokens": 200_000}, LONG_CONTEXT),
            ({"prompt_tokens": 20_000}, INTERACTIVE),
            ({"prompt_tokens": 1_000, "path": "/v1/completions"}, SHORT_CHAT),
            ({"prompt_tokens": 20_000, "has_tools": True}, AGENTIC),
            ({"prompt_tokens": 20_000, "batch_hint": True}, BATCH),
            ({"prompt_tokens": 1_000, "has_tools": True, "tools_disabled": True}, SHORT_CHAT),
        ):
            for source in (CountSource.HEURISTIC, CountSource.UNKNOWN, CountSource.ENGINE_RENDERED):
                with self.subTest(overrides=overrides, source=source):
                    text = policy.decide(_image(
                        has_images=False, prompt_count_source=source, **overrides,
                    ))
                    self.assertTrue(text.admitted)
                    self.assertEqual(text.traffic_class, expected_class.name)
                    self.assertEqual(text.metadata["encoder_tokens"], 0)
                    self.assertFalse(policy.prefill_complete(text))
                    policy.release(text)
                    self.assertEqual(budget.snapshot().encoder_tokens_in_flight, 17_847)
        policy.release(image)
        self.assert_empty(budget)

    def test_encoder_budget_is_opt_in_and_string_provenance_remains_compatible(self):
        for value in (0, -1, True, 1.5, "262144"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _budget(encoder_token_budget=value)
        policy = build_default(262_144, 64)
        heuristic = policy.decide(_image(prompt_count_source="heuristic"))
        self.assertTrue(heuristic.admitted)
        self.assertIsNone(policy.budget_snapshot().encoder_token_budget)
        self.assertEqual(heuristic.metadata["encoder_tokens"], 0)
        policy.release(heuristic)
        # New named callers can still omit all encoder arguments for text.
        budget = _budget(context_token_budget=None)
        self.assertFalse(budget.try_acquire(AGENTIC))  # Anonymous has no image flag.
        text = budget.try_acquire_lease(AGENTIC, 500, _health())
        self.assertTrue(text.admitted)
        budget.release_lease(text.lease.admission_id)
        policy = _policy(budget)
        image = policy.decide(_image(prompt_count_source="engine_rendered"))
        self.assertTrue(image.admitted)
        self.assertEqual(image.metadata["encoder_tokens"], 21_050)
        policy.release(image)
        self.assertEqual(str(CountSource.HEURISTIC), "heuristic")
        self.assertEqual(json.loads(json.dumps({
            "source": CountSource.ENGINE_RENDERED, "reason": AdmissionReason.ENCODER_COUNT,
        })), {"source": "engine_rendered", "reason": "encoder_count"})
        self.assert_empty(budget)

    def test_prefill_release_retains_every_main_lease_gauge_and_uses_acquired_cost(self):
        budget = ClassBudget(
            64, {INTERACTIVE.name: 1 / 64}, borrowing_enabled=True,
            context_token_budget=CONTEXT_BUDGET, encoder_token_budget=2 * 21_050,
        )
        classifier = Mock(classify=Mock(return_value=INTERACTIVE))
        policy = _policy(budget, classifier=classifier)
        first, second = policy.decide(_image()), policy.decide(_image())
        self.assertTrue(first.admitted and second.admitted)
        self.assertTrue(second.metadata["borrowed"])
        before = policy.budget_snapshot()
        altered = replace(
            second, traffic_class=AGENTIC.name, priority=Priority.LONG_CONTEXT,
            envelope=_image(1, has_images=False), metadata={"encoder_tokens": 1},
        )
        self.assertTrue(policy.prefill_complete(altered))
        after = policy.budget_snapshot()
        self.assertEqual(after, replace(before, encoder_tokens_in_flight=21_050))
        self.assertEqual(before.encoder_tokens_in_flight, 2 * 21_050)
        self.assertEqual(after.p0_in_flight, 2)
        self.assertEqual(after.borrowed_in_flight, 1)
        self.assertFalse(policy.prefill_complete(second))
        policy.release(altered)
        policy.release(second)
        self.assertEqual(policy.budget_snapshot().encoder_tokens_in_flight, 21_050)
        self.assertEqual(policy.budget_snapshot().context_tokens_in_flight, 21_114)
        self.assertEqual(classifier.classify.call_count, 2)
        policy.release(first)
        self.assert_empty(budget)

    def test_early_release_allows_next_prefill_but_not_extra_context_work(self):
        budget = _budget(encoder_token_budget=21_050, context_token_budget=2 * 21_114)
        policy = _policy(budget)
        first = policy.decide(_image())
        self.assertEqual(policy.decide(_image()).metadata["admission_reason"], "encoder")
        self.assertTrue(policy.prefill_complete(first))
        second = policy.decide(_image())
        self.assertTrue(second.admitted)
        self.assertNotEqual(first.admission_id, second.admission_id)
        self.assertFalse(policy.prefill_complete(first))
        self.assertTrue(policy.prefill_complete(second))
        before = policy.budget_snapshot()
        self.assertEqual(before.encoder_tokens_in_flight, 0)
        self.assertEqual(before.context_tokens_in_flight, 2 * 21_114)
        refused = policy.decide(_image())
        self.assertEqual(refused.metadata["admission_reason"], "token")
        self.assertEqual(refused.retry_after_seconds, 30)
        self.assertEqual(policy.budget_snapshot(), before)
        policy.release(first)
        replacement = policy.decide(_image())
        self.assertTrue(replacement.admitted)
        policy.release(first)
        self.assertFalse(policy.prefill_complete(first))
        self.assertEqual(policy.budget_snapshot().encoder_tokens_in_flight, 21_050)
        policy.release(second)
        policy.release(replacement)
        self.assert_empty(budget)

    def test_nonstreaming_holds_encoder_until_terminal_release(self):
        budget = _budget(encoder_token_budget=21_050)
        policy = _policy(budget)
        first = policy.decide(_image(streaming=False))
        before = policy.budget_snapshot()
        self.assertFalse(policy.prefill_complete(first))
        self.assertEqual(policy.budget_snapshot(), before)
        self.assertEqual(policy.decide(_image()).metadata["admission_reason"], "encoder")
        policy.release(first)
        policy.release(first)
        self.assert_empty(budget)
        next_request = policy.decide(_image())
        self.assertTrue(next_request.admitted)
        policy.release(next_request)
        self.assert_empty(budget)

    def test_failed_cancelled_or_terminal_requests_release_remaining_cost_once(self):
        for streaming in (False, True):
            for prefill_finished in (False, True):
                for failure in (CancelledError, RuntimeError, None):
                    with self.subTest(streaming=streaming, prefill=prefill_finished, failure=failure):
                        budget = _budget()
                        policy = _policy(budget)
                        decision = policy.decide(_image(streaming=streaming))
                        self.assertTrue(decision.admitted)
                        try:
                            if prefill_finished:
                                self.assertEqual(policy.prefill_complete(decision), streaming)
                            if failure is not None:
                                raise failure("synthetic request failure")
                        except (CancelledError, RuntimeError):
                            pass
                        finally:
                            policy.release(decision)
                        policy.release(replace(decision))
                        self.assertFalse(policy.prefill_complete(decision))
                        self.assertFalse(budget.release_encoder_lease(decision.admission_id))
                        self.assertFalse(budget.release_lease(decision.admission_id))
                        self.assert_empty(budget)

    def test_concurrent_encoder_release_only_succeeds_once_and_old_id_cannot_free_new_work(self):
        budget = _budget(encoder_token_budget=21_050)
        first = _acquire_image(budget).lease
        before = budget.snapshot()
        releases = _race(32, lambda _: budget.release_encoder_lease(first.admission_id))
        self.assertEqual(sum(releases), 1)
        self.assertEqual(budget.snapshot(), replace(before, encoder_tokens_in_flight=0))
        self.assertEqual(first.encoder_tokens, 21_050)  # Immutable acquisition fact.
        replacement = _acquire_image(budget).lease
        self.assertIsNotNone(replacement)
        self.assertNotEqual(first.admission_id, replacement.admission_id)
        self.assertFalse(budget.release_encoder_lease(first.admission_id))
        self.assertFalse(budget.release_encoder_lease("unknown"))
        self.assertTrue(budget.release_lease(first.admission_id))
        self.assertEqual(budget.snapshot().encoder_tokens_in_flight, 21_050)
        budget.release_lease(replacement.admission_id)
        self.assert_empty(budget)

    def test_prefill_and_full_release_race_does_not_free_another_requests_encoder(self):
        budget = _budget(encoder_token_budget=2 * 21_050)
        first, other = _acquire_image(budget).lease, _acquire_image(budget).lease

        def release(index):
            if index % 2:
                return "full", budget.release_lease(first.admission_id)
            return "encoder", budget.release_encoder_lease(first.admission_id)

        results = _race(32, release)
        self.assertEqual(sum(ok for kind, ok in results if kind == "full"), 1)
        self.assertLessEqual(sum(ok for kind, ok in results if kind == "encoder"), 1)
        state = budget.snapshot()
        self.assertEqual(state.in_flight, 1)
        self.assertEqual(state.encoder_tokens_in_flight, other.encoder_tokens)
        self.assertEqual(state.context_tokens_in_flight, other.context_tokens)
        budget.release_lease(other.admission_id)
        self.assert_empty(budget)

    def test_concurrent_mixed_admission_never_exceeds_encoder_context_or_slot_caps(self):
        budget = _budget()
        classes = (INTERACTIVE, AGENTIC, BATCH, LONG_CONTEXT)
        counts = (17_847, 21_050, 45_003, 65_536)

        def acquire(index):
            prompt = counts[index % 4] if index % 2 else 2_000
            return budget.try_acquire_lease(
                classes[index % 4], prompt + 64, _health(),
                has_images=bool(index % 2), prompt_tokens=prompt,
                prompt_count_source=CountSource.ENGINE_RENDERED,
            )

        results = _race(96, acquire)
        held = [result.lease for result in results if result.admitted]
        images = [lease for lease in held if lease.encoder_tokens]
        self.assertTrue(images)
        self.assertLess(len(images), len(held))
        self.assertIn(AdmissionReason.ENCODER, {r.reason for r in results if not r.admitted})
        state = budget.snapshot()
        self.assertEqual(state.encoder_tokens_in_flight, sum(l.encoder_tokens for l in held))
        self.assertEqual(state.context_tokens_in_flight, sum(l.context_tokens for l in held))
        self.assertEqual(state.in_flight, len(held))
        self.assertEqual(len({l.admission_id for l in held}), len(held))
        counts_by_class = Counter(l.traffic_class for l in held)
        for name, count in state.by_class.items():
            self.assertEqual(count, counts_by_class[name])
        for result in results:
            self.assert_bounded(result.snapshot)
        with ThreadPoolExecutor(max_workers=16) as pool:
            releases = list(pool.map(budget.release_lease, [l.admission_id for l in held] * 2))
        self.assertEqual(sum(releases), len(held))
        self.assert_empty(budget)

    def test_admission_overlaps_early_and_full_release_without_overbooking(self):
        budget = _budget()
        held = [_acquire_image(budget, 65_536).lease for _ in range(4)]
        self.assertTrue(all(held))

        def act(index):
            if index < 16:
                lease = held[index % 4]
                if index % 2:
                    budget.release_lease(lease.admission_id)
                else:
                    budget.release_encoder_lease(lease.admission_id)
                return None, budget.snapshot()
            result = _acquire_image(budget, (17_847, 21_050, 65_536)[index % 3])
            return result.lease, result.snapshot

        results = _race(64, act)
        for lease, state in results:
            self.assert_bounded(state)
            if lease is not None:
                held.append(lease)
        for lease in held:
            budget.release_lease(lease.admission_id)
        self.assert_empty(budget)

    def test_encoder_release_preserves_64_global_and_four_protected_slots(self):
        budget = _budget()
        held = []
        for _ in range(60):
            lease = _acquire_image(budget, 1_000).lease
            self.assertIsNotNone(lease)
            held.append(lease)
            self.assertTrue(budget.release_encoder_lease(lease.admission_id))
        self.assertEqual(_acquire_image(budget, 1_000).reason, "protected_reserve")
        self.assertEqual(budget.snapshot().context_token_budget, CONTEXT_BUDGET)
        self.assertEqual(budget.snapshot().p0_reserve, 4)
        for _ in range(4):
            lease = _acquire_image(budget, 1_000, traffic_class=INTERACTIVE).lease
            self.assertIsNotNone(lease)
            held.append(lease)
        self.assertEqual(_acquire_image(budget, 1_000, traffic_class=INTERACTIVE).reason, "global")
        self.assertEqual(budget.snapshot().in_flight, 64)
        for lease in held:
            budget.release_lease(lease.admission_id)
        self.assert_empty(budget)

    def test_encoder_room_cannot_bypass_borrowing_health_guard(self):
        for health in (
            _health(known=False), _health(kv_usage=0.80), _health(waiting=1),
            _health(preemptions_per_minute=0.01),
        ):
            with self.subTest(health=health):
                budget = ClassBudget(
                    64, {AGENTIC.name: 2 / 64}, borrowing_enabled=True,
                    context_token_budget=CONTEXT_BUDGET, encoder_token_budget=ENCODER_CAPACITY,
                )
                held = [_acquire_image(budget).lease for _ in range(2)]
                before = budget.snapshot()
                refused = _acquire_image(budget, health=health)
                self.assertEqual(refused.reason, "pressure")
                self.assertEqual(budget.snapshot(), before)
                borrowed = _acquire_image(budget).lease
                self.assertTrue(borrowed.borrowed)
                for lease in [*held, borrowed]:
                    budget.release_lease(lease.admission_id)
                self.assert_empty(budget)

    def test_encoder_limit_and_detached_gauges_are_independent_of_kv_capacity(self):
        source = Mock(snapshot=Mock(return_value=_health(kv_capacity_tokens=1_000_000)))
        budget = _budget()
        policy = _policy(budget, source=source)
        before = policy.budget_snapshot()
        decision = policy.decide(_image())
        during = policy.budget_snapshot()
        self.assertEqual(before.encoder_tokens_in_flight, 0)
        self.assertEqual(during.encoder_tokens_in_flight, 21_050)
        self.assertEqual(during.encoder_token_budget, ENCODER_CAPACITY)
        self.assertEqual(during.effective_context_token_budget, 800_000)
        self.assertNotIn(decision.admission_id, repr(during))
        self.assertNotIn("encoder-test", repr(during))
        self.assertTrue(policy.prefill_complete(decision))
        policy.release(decision)
        self.assertEqual(during.encoder_tokens_in_flight, 21_050)
        source.snapshot.assert_called_once()
        self.assert_empty(budget)

    def test_offbox_vision_and_unservable_prompts_never_take_local_encoder_work(self):
        remote = replace(AGENTIC, served_off_box=True)
        classifier = Mock(classify=Mock(return_value=remote))
        source = Mock(snapshot=Mock(side_effect=AssertionError("unexpected collection")))
        budget = _budget()
        policy = _policy(budget, classifier=classifier, source=source, offbox=True)
        offbox = policy.decide(_image(prompt_count_source=CountSource.UNKNOWN))
        self.assertTrue(offbox.admitted)
        self.assertIsNone(offbox.admission_id)
        self.assertFalse(policy.prefill_complete(offbox))
        policy.release(offbox)
        refused = policy.decide(_image(262_144))
        self.assertFalse(refused.admitted)
        self.assertFalse(policy.prefill_complete(refused))
        policy.release(refused)
        source.snapshot.assert_not_called()
        self.assert_empty(budget)


if __name__ == "__main__":
    unittest.main()
