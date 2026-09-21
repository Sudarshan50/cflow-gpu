"""Offline admission races, health gating and exactly-once context accounting."""

from __future__ import annotations

import threading
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

from redesign.gateway.backpressure import CircuitBreaker, ClassBudget, StaticHealthSource
from redesign.gateway.classification import (
    AGENTIC, ALL_CLASSES, BATCH, INTERACTIVE, SHORT_CHAT, Classifier,
)
from redesign.gateway.clamping import TokenClamp
from redesign.gateway.models import EngineSnapshot, Outcome, Priority, RequestEnvelope
from redesign.gateway.policy import GatewayPolicy, build_default


def _health(**overrides):
    return replace(EngineSnapshot(0.5, 12, 0, 0.0), **overrides)


def _envelope(**overrides):
    return replace(
        RequestEnvelope("test", 1_000, requested_max_tokens=8_192, has_tools=True),
        **overrides,
    )


def _policy(budget, *, source=None, health=None, classifier=None, offbox=False):
    return GatewayPolicy(
        classifier if classifier is not None else Classifier(),
        TokenClamp(262_144), budget,
        CircuitBreaker(source if source is not None else StaticHealthSource(
            health if health is not None else _health(),
        )),
        offbox_configured=offbox,
    )


def _race(count, action):
    start = threading.Barrier(count)

    def run(index):
        start.wait(timeout=15)
        return action(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(run, range(count)))


class ConcurrentAdmissionTest(unittest.TestCase):
    def assert_empty(self, budget):
        state = budget.snapshot()
        self.assertEqual(state.in_flight, 0)
        self.assertEqual(sum(state.by_class.values()), 0)
        self.assertEqual(state.context_tokens_in_flight, 0)
        self.assertEqual(state.borrowed_in_flight, 0)
        self.assertEqual(state.p0_in_flight, 0)

    def test_agentic_borrowing_preserves_p0_reserve_and_global_cap(self):
        budget = ClassBudget.from_classes(64, ALL_CLASSES, borrowing_enabled=True)
        agentic = _race(80, lambda _: budget.try_acquire_lease(AGENTIC, 100, _health()))
        held = [result.lease for result in agentic if result.admitted]
        self.assertEqual(len(held), 60)
        self.assertEqual(sum(lease.borrowed for lease in held), 60 - 28)
        self.assertEqual(
            {result.reason for result in agentic if not result.admitted},
            {"protected_reserve"},
        )
        self.assertEqual(budget.snapshot().p0_reserve, 4)

        p0 = _race(16, lambda _: budget.try_acquire_lease(INTERACTIVE, 200, _health()))
        held += [result.lease for result in p0 if result.admitted]
        state = budget.snapshot()
        self.assertEqual(len(held), 64)
        self.assertEqual(state.in_flight, 64)
        self.assertEqual(state.by_class[AGENTIC.name], 60)
        self.assertEqual(state.p0_in_flight, 4)
        for traffic_class in ALL_CLASSES:
            self.assertEqual(
                budget.try_acquire_lease(traffic_class, 1, _health()).reason, "global",
            )
        with ThreadPoolExecutor(max_workers=16) as pool:
            released = list(pool.map(budget.release_lease, [l.admission_id for l in held] * 2))
        self.assertEqual(sum(released), 64)
        self.assert_empty(budget)

    def test_p0_can_borrow_up_to_the_entire_global_ceiling(self):
        budget = ClassBudget.from_classes(64, ALL_CLASSES, borrowing_enabled=True)
        results = _race(80, lambda _: budget.try_acquire_lease(INTERACTIVE, 1, _health()))
        held = [result.lease for result in results if result.admitted]
        self.assertEqual(len(held), 64)
        self.assertEqual(sum(lease.borrowed for lease in held), 32)
        self.assertEqual(budget.snapshot().p0_in_flight, 64)
        for lease in held:
            budget.release_lease(lease.admission_id)
        self.assert_empty(budget)

    def test_reserve_also_covers_mixed_ordinary_non_p0_admissions(self):
        classes = (AGENTIC, BATCH, SHORT_CHAT)
        budget = ClassBudget(
            64, {c.name: 1.0 for c in (*classes, INTERACTIVE)}, borrowing_enabled=True,
        )
        # Active P0 requests consume the reserve rather than reducing the
        # non-P0 allowance: 60 non-P0 + 2 P0 leaves 2 protected slots.
        held = [budget.try_acquire_lease(INTERACTIVE, 1, _health()).lease for _ in range(2)]
        results = _race(80, lambda i: budget.try_acquire_lease(classes[i % 3], 1, _health()))
        ordinary = [result.lease for result in results if result.admitted]
        self.assertEqual(len(ordinary), 60)
        self.assertFalse(any(lease.borrowed for lease in ordinary))
        self.assertEqual(budget.snapshot().in_flight, 62)
        self.assertEqual(
            {result.reason for result in results if not result.admitted},
            {"protected_reserve"},
        )
        held += ordinary
        held += [budget.try_acquire_lease(INTERACTIVE, 1, _health()).lease for _ in range(2)]
        self.assertEqual(budget.snapshot().in_flight, 64)
        for lease in held:
            budget.release_lease(lease.admission_id)
        self.assert_empty(budget)

    def test_mixed_cost_class_and_slot_allocations_are_atomic(self):
        classes = (INTERACTIVE, AGENTIC, BATCH, SHORT_CHAT)
        costs = (5, 11, 17, 23)
        budget = ClassBudget(
            13, {c.name: 0.25 for c in classes}, borrowing_enabled=True,
            p0_reserve=3, context_token_budget=101,
        )
        results = _race(80, lambda i: budget.try_acquire_lease(
            classes[i % 4], costs[i % 4], _health(),
        ))
        held = [result.lease for result in results if result.admitted]
        state = budget.snapshot()
        self.assertGreater(len(held), 0)
        self.assertLessEqual(len(held), 13)
        self.assertLessEqual(sum(l.priority != Priority.INTERACTIVE for l in held), 10)
        self.assertEqual(len({lease.admission_id for lease in held}), len(held))
        self.assertEqual(state.in_flight, len(held))
        self.assertEqual(state.context_tokens_in_flight, sum(l.context_tokens for l in held))
        self.assertLessEqual(state.context_tokens_in_flight, 101)
        counts = Counter(l.traffic_class for l in held)
        for traffic_class in classes:
            self.assertEqual(state.by_class[traffic_class.name], counts[traffic_class.name])
        for result in results:
            # Every returned observation must be internally consistent even
            # while the other threads are changing all three budgets.
            self.assertEqual(result.snapshot.in_flight, sum(result.snapshot.by_class.values()))
            self.assertLessEqual(result.snapshot.context_tokens_in_flight, 101)
            self.assertLessEqual(result.snapshot.in_flight, 13)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(budget.release_lease, [l.admission_id for l in held] * 2))
        self.assert_empty(budget)
        # A rejected high-cost attempt did not leave a partial reservation.
        self.assertTrue(budget.try_acquire_lease(INTERACTIVE, 101, _health()).admitted)

    def test_legacy_and_named_acquisitions_share_caps_but_not_release_ownership(self):
        budget = ClassBudget(7, {AGENTIC.name: 1.0})
        self.assertTrue(budget.try_acquire(AGENTIC))
        initial = budget.try_acquire_lease(AGENTIC, 17).lease

        def acquire(index):
            if index % 2:
                return budget.try_acquire(AGENTIC), None
            result = budget.try_acquire_lease(AGENTIC, 17)
            return result.admitted, result.lease

        results = [(True, None), (True, initial)] + _race(32, acquire)
        self.assertEqual(sum(admitted for admitted, _ in results), 7)
        named = [lease for _, lease in results if lease is not None]
        for admitted, lease in results:
            if admitted and lease is None:
                budget.release(AGENTIC)
        # Extra anonymous releases must not steal ticketed costs/slots.
        budget.release(AGENTIC)
        budget.release(AGENTIC)
        self.assertEqual(budget.snapshot().in_flight, len(named))
        self.assertEqual(budget.snapshot().context_tokens_in_flight, len(named) * 17)
        for lease in named:
            budget.release_lease(lease.admission_id)
        self.assert_empty(budget)


class BorrowingHealthTest(unittest.TestCase):
    def test_only_known_healthy_metrics_allow_borrowing(self):
        unhealthy = (
            None,
            _health(known=False),  # Sources also use this for stale scrapes.
            _health(kv_usage=0.80),
            _health(kv_usage=0.90, running=27),
            _health(waiting=1),
            _health(preemptions_per_minute=0.0001),
            _health(kv_usage=float("nan")),
            _health(preemptions_per_minute=float("inf")),
            _health(waiting=-1),
        )
        for health in unhealthy:
            with self.subTest(health=health):
                budget = ClassBudget(8, {AGENTIC.name: 0.25}, borrowing_enabled=True, p0_reserve=1)
                for _ in range(2):
                    self.assertTrue(budget.try_acquire_lease(AGENTIC, 10, health).admitted)
                before = budget.snapshot()
                refused = budget.try_acquire_lease(AGENTIC, 10, health)
                self.assertEqual(refused.reason, "pressure")
                self.assertEqual(budget.snapshot(), before)
                allowed = budget.try_acquire_lease(AGENTIC, 10, _health(kv_usage=0.7999))
                self.assertTrue(allowed.lease.borrowed)

    def test_unknown_or_broken_health_fails_open_only_for_ordinary_policy_admission(self):
        sources = (
            Mock(snapshot=Mock(return_value=_health(
                kv_usage=0.99, waiting=50, preemptions_per_minute=5.0, known=False,
            ))),
            Mock(snapshot=Mock(side_effect=RuntimeError("scrape unavailable"))),
        )
        for source in sources:
            with self.subTest(source=source):
                budget = ClassBudget(4, {AGENTIC.name: 0.25}, borrowing_enabled=True, p0_reserve=0)
                policy = _policy(budget, source=source)
                ordinary = policy.decide(_envelope())
                self.assertTrue(ordinary.admitted)
                refused = policy.decide(_envelope())
                self.assertIs(refused.outcome, Outcome.REJECT_BUDGET)
                self.assertEqual(refused.metadata["admission_reason"], "pressure")
                self.assertEqual(source.snapshot.call_count, 2)
                policy.release(ordinary)
                self.assertEqual(budget.snapshot().context_tokens_in_flight, 0)

    def test_one_snapshot_per_decision_and_pressure_stops_further_borrowing(self):
        source = Mock(snapshot=Mock(side_effect=[_health(), _health(), _health(kv_usage=0.80)]))
        budget = ClassBudget(8, {AGENTIC.name: 0.125}, borrowing_enabled=True, p0_reserve=1)
        policy = _policy(budget, source=source)
        ordinary = policy.decide(_envelope())
        borrowed = policy.decide(_envelope())
        refused = policy.decide(_envelope())
        self.assertTrue(ordinary.admitted)
        self.assertTrue(borrowed.metadata["borrowed"])
        self.assertEqual(borrowed.metadata["class_borrowed_in_flight"], 1)
        self.assertEqual(borrowed.metadata["borrowed_in_flight"], 1)
        self.assertIn("1 class / 1 total", borrowed.notes[0])
        self.assertEqual(refused.metadata["admission_reason"], "pressure")
        self.assertEqual(policy.budget_snapshot().in_flight, 2)
        self.assertEqual(source.snapshot.call_count, 3)
        policy.release(ordinary)
        # Borrowing is an acquisition fact, not recomputed from the new total.
        self.assertEqual(policy.budget_snapshot().borrowed_in_flight, 1)
        policy.release(borrowed)
        self.assertEqual(policy.budget_snapshot().borrowed_in_flight, 0)

    def test_explicit_breaker_snapshot_never_fetches_again(self):
        source = Mock(snapshot=Mock(side_effect=RuntimeError("must not collect")))
        breaker = CircuitBreaker(source)
        health = _health(kv_usage=0.99)
        self.assertTrue(breaker.evaluate(health).distressed)
        self.assertTrue(breaker.should_shed(BATCH, health).distressed)
        self.assertFalse(breaker.should_shed(INTERACTIVE, health).distressed)
        self.assertFalse(breaker.evaluate(replace(health, known=False)).distressed)
        source.snapshot.assert_not_called()
        self.assertFalse(breaker.evaluate().distressed)
        source.snapshot.assert_called_once()

    def test_zero_share_or_routed_offbox_class_cannot_borrow(self):
        for budget, traffic_class in (
            (ClassBudget(64, {AGENTIC.name: 0.0}, borrowing_enabled=True), AGENTIC),
            (ClassBudget.from_classes(64, ALL_CLASSES, True, borrowing_enabled=True), SHORT_CHAT),
        ):
            result = budget.try_acquire_lease(traffic_class, 1, _health())
            self.assertEqual(result.reason, "class")
            self.assertEqual(result.snapshot.in_flight, 0)


class ContextLeaseTest(unittest.TestCase):
    def test_counts_prompt_plus_granted_output_for_every_request_without_sharing(self):
        cost = 1_000 + 8_192
        budget = ClassBudget.from_classes(64, ALL_CLASSES, context_token_budget=2 * cost + 10)
        policy = _policy(budget)
        first = policy.decide(_envelope())
        second = policy.decide(_envelope())  # Identical prompts still count twice.
        refused = policy.decide(_envelope())
        self.assertTrue(first.admitted and second.admitted)
        self.assertEqual(first.clamp.granted, 8_192)
        self.assertNotEqual(first.admission_id, second.admission_id)
        self.assertEqual(first.metadata["context_tokens"], cost)
        self.assertEqual(second.metadata["context_tokens_in_flight"], 2 * cost)
        self.assertEqual(refused.metadata["admission_reason"], "token")
        policy.release(refused)
        self.assertEqual(budget.snapshot().context_tokens_in_flight, 2 * cost)
        policy.release(first)
        self.assertTrue(policy.decide(_envelope()).admitted)
        self.assertEqual(budget.snapshot().context_tokens_in_flight, 2 * cost)

    def test_token_budget_is_capped_by_80_percent_of_reported_capacity(self):
        for capacity, expected in (
            (None, 1_310_000), (2_000_000, 1_310_000), (1_000_000, 800_000), (0, 0),
        ):
            with self.subTest(capacity=capacity):
                budget = ClassBudget(64, {AGENTIC.name: 1.0}, context_token_budget=1_310_000)
                health = _health(kv_capacity_tokens=capacity)
                refused = budget.try_acquire_lease(AGENTIC, expected + 1, health)
                self.assertEqual(refused.reason, "token")
                self.assertEqual(refused.snapshot.effective_context_token_budget, expected)
                self.assertEqual(refused.snapshot.context_tokens_in_flight, 0)
                if expected:
                    exact = budget.try_acquire_lease(AGENTIC, expected, health)
                    self.assertTrue(exact.admitted)
                    self.assertEqual(budget.try_acquire_lease(AGENTIC, 1, health).reason, "token")
                    budget.release_lease(exact.lease.admission_id)
                    self.assertEqual(budget.snapshot().context_tokens_in_flight, 0)

    def test_missing_or_stale_capacity_does_not_relax_the_last_observed_cap(self):
        budget = ClassBudget(64, {AGENTIC.name: 1.0}, context_token_budget=1_310_000)
        held = budget.try_acquire_lease(AGENTIC, 800_000, _health(kv_capacity_tokens=1_000_000))
        for health in (
            _health(), _health(known=False), _health(known=False, kv_capacity_tokens=2_000_000),
        ):
            with self.subTest(health=health):
                refused = budget.try_acquire_lease(AGENTIC, 1, health)
                self.assertEqual(refused.reason, "token")
                self.assertEqual(refused.snapshot.effective_context_token_budget, 800_000)
        budget.release_lease(held.lease.admission_id)
        # A new known report may legitimately reflect a resized engine pool.
        allowed = budget.try_acquire_lease(AGENTIC, 1_310_000, _health(kv_capacity_tokens=2_000_000))
        self.assertTrue(allowed.admitted)

    def test_capacity_shrink_blocks_new_work_until_existing_leases_drain(self):
        budget = ClassBudget(64, {AGENTIC.name: 1.0}, context_token_budget=1_310_000)
        held = budget.try_acquire_lease(AGENTIC, 600_000, _health(kv_capacity_tokens=1_000_000))
        refused = budget.try_acquire_lease(AGENTIC, 1, _health(kv_capacity_tokens=500_000))
        self.assertEqual(refused.reason, "token")
        self.assertEqual(refused.snapshot.effective_context_token_budget, 400_000)
        self.assertEqual(refused.snapshot.context_tokens_in_flight, 600_000)
        budget.release_lease(held.lease.admission_id)
        self.assertTrue(budget.try_acquire_lease(AGENTIC, 400_000).admitted)

    def test_concurrent_duplicate_release_returns_exact_cost_once(self):
        budget = ClassBudget(64, {AGENTIC.name: 1.0}, context_token_budget=1_000)
        first = budget.try_acquire_lease(AGENTIC, 300).lease
        second = budget.try_acquire_lease(AGENTIC, 700).lease
        releases = _race(32, lambda _: budget.release_lease(first.admission_id))
        self.assertEqual(sum(releases), 1)
        self.assertEqual(budget.snapshot().in_flight, 1)
        self.assertEqual(budget.snapshot().context_tokens_in_flight, 700)
        self.assertFalse(budget.release_lease("not-an-admission"))
        replacement = budget.try_acquire_lease(AGENTIC, 300).lease
        self.assertNotEqual(replacement.admission_id, first.admission_id)
        self.assertFalse(budget.release_lease(first.admission_id))
        self.assertEqual(budget.snapshot().context_tokens_in_flight, 1_000)
        budget.release_lease(second.admission_id)
        budget.release_lease(replacement.admission_id)
        self.assertEqual(budget.snapshot().context_tokens_in_flight, 0)

    def test_release_uses_acquired_class_priority_and_cost_despite_reclassification(self):
        classifier = Mock(classify=Mock(side_effect=[AGENTIC, INTERACTIVE]))
        budget = ClassBudget.from_classes(64, ALL_CLASSES, context_token_budget=100_000)
        policy = _policy(budget, classifier=classifier)
        agentic = policy.decide(_envelope())
        interactive = policy.decide(_envelope())
        altered = replace(
            agentic, traffic_class=SHORT_CHAT.name, priority=Priority.SHORT_CHAT,
            envelope=_envelope(prompt_tokens=0, has_tools=False),
            clamp=replace(agentic.clamp, granted=0),
            metadata={"context_tokens": 0, "borrowed": True},
        )
        policy.release(altered)
        policy.release(agentic)
        state = budget.snapshot()
        self.assertEqual(state.by_class[AGENTIC.name], 0)
        self.assertEqual(state.by_class[INTERACTIVE.name], 1)
        self.assertEqual(state.p0_in_flight, 1)
        self.assertEqual(state.context_tokens_in_flight, 1_000 + interactive.clamp.granted)
        policy.release(interactive)
        self.assertEqual(classifier.classify.call_count, 2)
        self.assertEqual(budget.snapshot().context_tokens_in_flight, 0)
        self.assertEqual(budget.snapshot().p0_in_flight, 0)

    def test_token_budget_cannot_be_bypassed_by_anonymous_or_negative_costs(self):
        budget = ClassBudget(64, {AGENTIC.name: 1.0}, context_token_budget=100)
        self.assertFalse(budget.try_acquire(AGENTIC))
        for cost in (-1, 1.5, True, None):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                budget.try_acquire_lease(AGENTIC, cost)
        self.assertEqual(budget.snapshot().in_flight, 0)
        self.assertTrue(budget.try_acquire_lease(AGENTIC, 100).admitted)


class PolicyCompatibilityTest(unittest.TestCase):
    def test_default_policy_retains_static_28_agentic_slots_and_grants_requested_output(self):
        policy = build_default(262_144, 64)
        held = [policy.decide(_envelope()) for _ in range(28)]
        self.assertTrue(all(decision.admitted for decision in held))
        self.assertTrue(all(decision.clamp.granted == 8_192 for decision in held))
        self.assertTrue(all(decision.notes == () for decision in held))
        refused = policy.decide(_envelope())
        self.assertEqual(refused.metadata["admission_reason"], "class")
        state = policy.budget_snapshot()
        self.assertFalse(state.borrowing_enabled)
        self.assertEqual(state.p0_reserve, 0)
        self.assertIsNone(state.effective_context_token_budget)
        for decision in held:
            policy.release(decision)

    def test_disabled_features_allow_legacy_non_p0_to_fill_global_ceiling(self):
        budget = ClassBudget.from_classes(64, ALL_CLASSES)
        held = []
        for traffic_class in (AGENTIC, BATCH, SHORT_CHAT, *ALL_CLASSES):
            if traffic_class.priority == Priority.INTERACTIVE:
                continue
            while budget.try_acquire(traffic_class):
                held.append(traffic_class)
        self.assertEqual(len(held), 64)
        for traffic_class in held:
            budget.release(traffic_class)
        # Reported capacity does not silently enable token limiting either.
        self.assertTrue(budget.try_acquire_lease(
            AGENTIC, 2_000_000, _health(kv_capacity_tokens=0),
        ).admitted)
        self.assertIsNone(budget.snapshot().effective_context_token_budget)

    def test_policy_rejection_codes_are_distinct_and_never_allocate_partial_work(self):
        cases = (
            ("global", ClassBudget(1, {AGENTIC.name: 1.0}), _health()),
            ("class", ClassBudget(4, {AGENTIC.name: 0.25}), _health()),
            ("token", ClassBudget(4, {AGENTIC.name: 1.0}, context_token_budget=9_192), _health()),
            ("pressure", ClassBudget(
                4, {AGENTIC.name: 0.25}, borrowing_enabled=True, p0_reserve=0,
            ), _health(kv_usage=0.8)),
            ("protected_reserve", ClassBudget(
                2, {AGENTIC.name: 1.0}, borrowing_enabled=True, p0_reserve=1,
            ), _health()),
        )
        reasons = set()
        for code, budget, health in cases:
            with self.subTest(code=code):
                policy = _policy(budget, health=health)
                first = policy.decide(_envelope())
                self.assertTrue(first.admitted)
                before = budget.snapshot()
                refused = policy.decide(_envelope())
                self.assertIs(refused.outcome, Outcome.REJECT_BUDGET)
                self.assertIsNone(refused.admission_id)
                self.assertEqual(refused.metadata["admission_reason"], code)
                self.assertEqual(refused.retry_after_seconds, 30)
                reasons.add(refused.reason)
                policy.release(refused)
                self.assertEqual(budget.snapshot(), before)
                policy.release(first)
        self.assertEqual(len(reasons), 5)
        policy = _policy(ClassBudget.from_classes(64, ALL_CLASSES), health=_health(kv_usage=0.99))
        shed = policy.decide(_envelope())
        self.assertIs(shed.outcome, Outcome.REJECT_SHED)
        self.assertEqual(shed.metadata["admission_reason"], "pressure")

    def test_offbox_and_unservable_requests_do_not_acquire_or_collect_health(self):
        source = Mock(snapshot=Mock(side_effect=AssertionError("unexpected collection")))
        budget = ClassBudget.from_classes(
            64, ALL_CLASSES, True, borrowing_enabled=True, context_token_budget=1,
        )
        policy = _policy(budget, source=source, offbox=True)
        offbox = policy.decide(_envelope(has_tools=False))
        unservable = policy.decide(_envelope(prompt_tokens=262_144))
        self.assertTrue(offbox.admitted)
        self.assertIn("routed off-box", offbox.notes)
        self.assertIsNone(offbox.admission_id)
        self.assertFalse(unservable.admitted)
        policy.release(offbox)
        policy.release(unservable)
        self.assertEqual(policy.budget_snapshot().in_flight, 0)
        source.snapshot.assert_not_called()

    def test_budget_gauges_are_atomic_detached_and_do_not_scrape(self):
        source = Mock(snapshot=Mock(return_value=_health()))
        policy = _policy(ClassBudget.from_classes(64, ALL_CLASSES), source=source)
        before = policy.budget_snapshot()
        decision = policy.decide(_envelope())
        during = policy.budget_snapshot()
        self.assertEqual(before.in_flight, 0)
        self.assertEqual(before.by_class[AGENTIC.name], 0)
        self.assertEqual(during.in_flight, sum(during.by_class.values()))
        self.assertEqual(during.context_tokens_in_flight, 9_192)
        with self.assertRaises(TypeError):
            during.by_class[AGENTIC.name] = 100
        with self.assertRaises(TypeError):
            during.borrowed_by_class[AGENTIC.name] = 100
        self.assertNotIn(decision.admission_id, repr(during))
        policy.release(decision)
        self.assertEqual(during.in_flight, 1)
        self.assertEqual(policy.budget_snapshot().in_flight, 0)
        source.snapshot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
