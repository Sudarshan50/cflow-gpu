"""Admission contention, bounded waiting, cancellation, and HTTP behavior."""
import concurrent.futures
import json
import socket
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from redesign.gateway.admission import AdmissionController, AdmissionLimits
from redesign.gateway.backpressure import StaticHealthSource
from redesign.gateway.engine import EngineClient
from redesign.gateway.models import EngineSnapshot, RequestEnvelope
from redesign.gateway.policy import build_default
from redesign.gateway.workload import WorkloadBudget, WorkloadLimits
from redesign.tests import test_workload_guard as guard_tests


def wait_for(predicate, timeout=3):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError("condition did not become true")


class ElasticBudgetTests(unittest.TestCase):
    def budget(self, **overrides):
        values = dict(large_context_tokens=1000, large_context_requests=4,
                      long_output_tokens=100, long_output_requests=2,
                      reserved_tokens=5000, long_output_burst_requests=4,
                      long_output_base_token_budget=250,
                      long_output_burst_token_budget=500)
        values.update(overrides)
        return WorkloadBudget(WorkloadLimits(**values),
                              StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 10000)))

    def test_burst_is_bounded_under_concurrent_arrival_and_fully_charged(self):
        budget = self.budget()
        barrier = threading.Barrier(24)
        def acquire(_):
            barrier.wait()
            return budget.acquire(100, 200)[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            leases = list(pool.map(acquire, range(24)))
        self.assertEqual(sum(lease is not None for lease in leases), 4)
        self.assertEqual(budget.state()["reserved_tokens"], 900)
        self.assertEqual(budget.state()["burst_admissions_total"], 2)
        for lease in leases:
            if lease is not None:
                budget.release(lease)
        self.assertEqual(budget.state()["active_requests"], 0)

    def test_pressure_removes_burst_capacity_without_evicting_existing_work(self):
        budget = self.budget()
        leases = [budget.acquire(100, 200)[0] for _ in range(3)]
        budget.health_source = StaticHealthSource(EngineSnapshot(.60, 0, 0, 0, 10000))
        self.assertEqual(budget.acquire(100, 200)[1], "long_output_budget")
        self.assertEqual(budget.state()["active_requests"], 3)
        budget.release(leases.pop())
        self.assertIsNone(budget.acquire(100, 200)[0])
        budget.release(leases.pop())
        self.assertIsNotNone(budget.acquire(100, 200)[0])

    def test_large_and_multimodal_requests_cannot_use_extra_output_slots(self):
        for prompt, images in ((1000, False), (10, True)):
            with self.subTest(prompt=prompt, images=images):
                budget = self.budget()
                self.assertIsNotNone(budget.acquire(prompt, 200, images)[0])
                self.assertIsNotNone(budget.acquire(prompt, 200, images)[0])
                self.assertEqual(budget.acquire(prompt, 200, images)[1], "long_output_budget")

    def test_burst_needs_known_capacity_no_queue_and_no_preemptions(self):
        for snapshot in (
            EngineSnapshot(0, 0, 0, 0),
            EngineSnapshot(.53, 0, 0, 0, 10000),
            EngineSnapshot(.1, 0, 1, 0, 10000),
            EngineSnapshot(.1, 0, 0, .5, 10000),
            EngineSnapshot(.1, 8, 0, 0, 10000),
        ):
            with self.subTest(snapshot=snapshot):
                budget = self.budget()
                budget.acquire(100, 200)
                budget.acquire(100, 200)
                budget.health_source = StaticHealthSource(snapshot)
                self.assertIsNone(budget.acquire(100, 200)[0])

    def test_medium_long_context_does_not_borrow_short_chat_burst_capacity(self):
        budget = self.budget(large_context_tokens=65536, reserved_tokens=300000)
        budget.health_source = StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 1000000))
        self.assertIsNotNone(budget.acquire(50000, 200)[0])
        self.assertIsNotNone(budget.acquire(50000, 200)[0])
        self.assertEqual(budget.acquire(50000, 200)[1], "long_output_budget")

    def test_no_extra_capacity_when_health_is_unavailable(self):
        class Broken:
            def snapshot(self):
                raise OSError("metrics unavailable")
        budget = self.budget()
        budget.health_source = Broken()
        self.assertIsNotNone(budget.acquire(100, 200)[0])
        self.assertIsNotNone(budget.acquire(100, 200)[0])
        self.assertEqual(budget.acquire(100, 200)[1], "long_output_budget")

    def test_token_budget_and_kv_limit_remain_hard_bounds(self):
        budget = self.budget(reserved_tokens=650)
        budget.acquire(100, 200)
        budget.acquire(100, 200)
        self.assertEqual(budget.acquire(100, 200)[1], "reserved_tokens")
        budget = self.budget()
        budget.health_source = StaticHealthSource(EngineSnapshot(.84, 0, 0, 0, 10000))
        self.assertEqual(budget.acquire(100, 200)[1], "kv_headroom")

    def test_long_output_budget_is_token_weighted_and_conserved(self):
        budget = self.budget(
            long_output_tokens=2048,
            long_output_requests=2,
            long_output_burst_requests=8,
            long_output_base_token_budget=8192,
            long_output_burst_token_budget=32768,
            reserved_tokens=100000,
            output_safety_factor=4,
        )
        budget.health_source = StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 1000000))
        leases = [budget.acquire(100, 8192)[0] for _ in range(4)]
        self.assertTrue(all(lease is not None for lease in leases))
        self.assertEqual(budget.acquire(100, 8192)[1], "long_output_budget")
        self.assertEqual(budget.state()["long_output_reserved_tokens"], 32768)
        self.assertTrue(budget.release(leases.pop()))
        self.assertEqual(budget.state()["long_output_reserved_tokens"], 24576)
        self.assertIsNotNone(budget.acquire(100, 8192)[0])

    def test_one_valid_output_larger_than_base_budget_can_still_run(self):
        budget = self.budget(
            long_output_tokens=2048,
            long_output_requests=2,
            long_output_burst_requests=2,
            long_output_base_token_budget=8192,
            long_output_burst_token_budget=8192,
            reserved_tokens=100000,
            output_safety_factor=4,
        )
        budget.health_source = StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 1000000))
        self.assertIsNotNone(budget.acquire(100, 16000)[0])
        self.assertEqual(budget.acquire(100, 16000)[1], "long_output_budget")

    def test_3000_token_requests_are_bounded_by_tokens_not_fixed_slots(self):
        budget = self.budget(
            long_output_tokens=2048,
            long_output_requests=2,
            long_output_burst_requests=8,
            long_output_burst_running_limit=32,
            long_output_base_token_budget=8192,
            long_output_burst_token_budget=32768,
            reserved_tokens=100000,
        )
        budget.health_source = StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 1000000))
        # The deprecated request-count value is eight, but the 32K weighted
        # budget safely admits twelve cold-start 2,560-token reservations.
        leases = [budget.acquire(100, 3000)[0] for _ in range(12)]
        self.assertTrue(all(lease is not None for lease in leases))
        self.assertEqual(budget.acquire(100, 3000)[1], "long_output_budget")
        for lease in leases:
            budget.release(lease)
        budget.health_source = StaticHealthSource(EngineSnapshot(.60, 0, 0, 0, 1000000))
        self.assertIsNotNone(budget.acquire(100, 3000)[0])
        self.assertIsNotNone(budget.acquire(100, 3000)[0])
        self.assertIsNotNone(budget.acquire(100, 3000)[0])
        self.assertEqual(budget.acquire(100, 3000)[1], "long_output_budget")

    def test_completed_outputs_adapt_future_reservations_by_class(self):
        budget = self.budget(
            long_output_tokens=2048,
            long_output_requests=16,
            long_output_burst_requests=16,
            long_output_base_token_budget=65536,
            long_output_burst_token_budget=65536,
            reserved_tokens=100000,
            output_min_samples=3,
        )
        budget.health_source = StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 1000000))
        for actual in (400, 500, 600):
            lease, _ = budget.acquire(100, 4096, class_name="P1-short-chat")
            self.assertTrue(budget.release(lease, actual))
        lease, _ = budget.acquire(100, 4096, class_name="P1-short-chat")
        self.assertEqual(budget.state()["long_output_last_prediction"], 750)
        budget.release(lease)

        # A different class cold-starts from the configured long-output
        # threshold rather than borrowing another class's completion shape.
        lease, _ = budget.acquire(100, 4096, class_name="P0-interactive")
        self.assertEqual(budget.state()["long_output_last_prediction"], 2560)
        budget.release(lease)

    def test_staged_profile_admits_sixteen_4096_grants_only_when_healthy(self):
        budget = self.budget(
            long_output_tokens=2048,
            long_output_requests=4,
            long_output_burst_requests=16,
            long_output_burst_running_limit=16,
            long_output_base_token_budget=12288,
            long_output_burst_token_budget=49152,
            reserved_tokens=1000000,
        )
        budget.health_source = StaticHealthSource(
            EngineSnapshot(0, 0, 0, 0, 1000000)
        )
        leases = [
            budget.acquire(100, 4096, class_name="P1-short-chat")[0]
            for _ in range(16)
        ]
        self.assertTrue(all(lease is not None for lease in leases))
        self.assertEqual(
            budget.acquire(100, 4096, class_name="P1-short-chat")[1],
            "long_output_budget",
        )
        for lease in leases:
            budget.release(lease, 800)

        budget.health_source = StaticHealthSource(
            EngineSnapshot(.60, 0, 0, 0, 1000000)
        )
        pressured = [
            budget.acquire(100, 4096, class_name="P1-short-chat")[0]
            for _ in range(12)
        ]
        self.assertTrue(all(lease is not None for lease in pressured))
        self.assertEqual(
            budget.acquire(100, 4096, class_name="P1-short-chat")[1],
            "long_output_budget",
        )

    def test_queue_tolerance_is_bounded_and_does_not_grant_burst_slots(self):
        budget = self.budget(engine_queue_tolerance=2)
        budget.acquire(10, 10)
        budget.acquire(10, 10)
        budget.health_source = StaticHealthSource(EngineSnapshot(.1, 0, 2, 0, 10000))
        self.assertIsNotNone(budget.acquire(100, 200)[0])
        self.assertIsNotNone(budget.acquire(100, 200)[0])
        self.assertEqual(budget.acquire(100, 200)[1], "long_output_budget")
        budget = self.budget(engine_queue_tolerance=2)
        budget.health_source = StaticHealthSource(EngineSnapshot(.1, 0, 3, 0, 10000))
        self.assertEqual(budget.acquire(100, 200)[1], "engine_queue")

    def test_queue_tolerance_disappears_under_real_pressure(self):
        budget = self.budget(engine_queue_tolerance=2)
        for _ in range(8):
            budget.acquire(10, 10)
        budget.health_source = StaticHealthSource(EngineSnapshot(.1, 6, 2, 0, 10000))
        self.assertEqual(budget.acquire(100, 200)[1], "engine_queue")
        budget = self.budget(engine_queue_tolerance=2)
        budget.acquire(10, 10)
        budget.health_source = StaticHealthSource(EngineSnapshot(.6, 0, 1, 0, 10000))
        self.assertEqual(budget.acquire(100, 200)[1], "engine_queue")

    def test_unowned_queued_work_is_not_treated_as_free_capacity(self):
        budget = self.budget(engine_queue_tolerance=2)
        budget.health_source = StaticHealthSource(EngineSnapshot(.1, 0, 1, 0, 10000))
        self.assertEqual(budget.acquire(100, 200)[1], "untracked_engine_work")

    def test_unowned_work_is_refreshed_before_rejection(self):
        class Stale:
            def __init__(self):
                self.refreshes = 0
            def snapshot(self):
                return EngineSnapshot(.1, 1, 0, 0, 10000)
            def refresh_snapshot(self):
                self.refreshes += 1
                return EngineSnapshot(.1, 0, 0, 0, 10000)
        budget = self.budget()
        source = Stale()
        budget.health_source = source
        self.assertIsNotNone(budget.acquire(100, 200)[0])
        self.assertEqual(source.refreshes, 1)

    def test_failed_refresh_retains_the_conservative_snapshot(self):
        class Stale:
            def snapshot(self):
                return EngineSnapshot(.1, 1, 0, 0, 10000)
            def refresh_snapshot(self):
                raise OSError("refresh failed")
        budget = self.budget()
        budget.health_source = Stale()
        self.assertEqual(budget.acquire(100, 200)[1], "untracked_engine_work")

    def test_adaptive_borrow_requires_kv_and_decode_headroom(self):
        limits = WorkloadLimits(
            large_context_tokens=1000, large_context_requests=4,
            long_output_tokens=1000, long_output_requests=2,
            reserved_tokens=5000, adaptive_borrow_requests=32,
            adaptive_borrow_kv_limit=.70, adaptive_borrow_itl_limit=.12,
        )
        healthy = WorkloadBudget(
            limits, StaticHealthSource(EngineSnapshot(.20, 8, 0, 0, 10000, mean_itl_seconds=.05))
        )
        self.assertTrue(healthy.can_borrow("P1-short-chat", 100, 20))
        self.assertTrue(healthy.can_borrow("P2-agentic", 100, 20))
        self.assertFalse(healthy.can_borrow("P2-long-context", 100, 20))
        for snapshot in (
            EngineSnapshot(.70, 8, 0, 0, 10000, mean_itl_seconds=.05),
            EngineSnapshot(.20, 8, 1, 0, 10000, mean_itl_seconds=.05),
            EngineSnapshot(.20, 8, 0, 1, 10000, mean_itl_seconds=.05),
            EngineSnapshot(.20, 8, 0, 0, 10000, mean_itl_seconds=.13),
        ):
            with self.subTest(snapshot=snapshot):
                guarded = WorkloadBudget(limits, StaticHealthSource(snapshot))
                self.assertFalse(guarded.can_borrow("P1-short-chat", 100, 20))

    def test_policy_borrows_idle_short_chat_capacity_but_keeps_global_bound(self):
        policy = build_default(262144, 64)
        policy.workload_budget = WorkloadBudget(
            WorkloadLimits(
                large_context_tokens=1000, long_output_tokens=2000,
                reserved_tokens=100000, adaptive_borrow_requests=32,
            ),
            StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 100000)),
        )
        decisions = [
            policy.decide(RequestEnvelope("pool", 50, 1_000))
            for _ in range(33)
        ]
        self.assertEqual(sum(d.admitted for d in decisions), 32)
        self.assertIn("adaptive class borrow", decisions[16].notes)
        self.assertFalse(decisions[-1].admitted)
        self.assertIn("limit of 32", decisions[-1].reason)
        for decision in decisions:
            policy.release(decision)


class AdmissionTests(unittest.TestCase):
    def controller(self, **limits):
        policy = build_default(262144, 64)
        policy.workload_budget = WorkloadBudget(
            WorkloadLimits(large_context_tokens=1000, long_output_tokens=100,
                           long_output_requests=1, reserved_tokens=5000,
                           long_output_base_token_budget=125,
                           long_output_burst_token_budget=125),
            StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 10000)),
        )
        values = dict(wait_seconds=2, poll_seconds=.01)
        values.update(limits)
        return AdmissionController(policy, AdmissionLimits(**values))

    def request(self, customer="a", output=200):
        return RequestEnvelope(customer, 50, output)

    def test_waiting_does_not_hold_execution_reservations(self):
        controller = self.controller()
        held = controller.acquire(self.request())
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(controller.acquire, self.request())
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            self.assertEqual(controller.policy.workload_budget.state()["reserved_tokens"], 175)
            self.assertEqual(controller.state()["queued_tokens"], 250)
            controller.release(held)
            admitted = pending.result(timeout=2)
        self.assertTrue(admitted.admitted)
        self.assertTrue(admitted.admission_queued)
        self.assertEqual(admitted.clamp.granted, 200)
        controller.release(admitted)
        self.assertEqual(controller.state()["queued_tokens"], 0)
        self.assertEqual(controller.policy.workload_budget.state()["reserved_tokens"], 0)
        self.assertEqual(controller.state()["queued_admitted_total"], 1)

    def test_concurrent_probe_uses_available_capacity_before_queueing(self):
        controller = self.controller(max_waiters=1)
        controller.policy.workload_budget = WorkloadBudget(
            WorkloadLimits(
                large_context_tokens=1000,
                long_output_tokens=100,
                long_output_requests=2,
                reserved_tokens=5000,
                long_output_base_token_budget=250,
                long_output_burst_token_budget=250,
            ),
            StaticHealthSource(EngineSnapshot(0, 0, 0, 0, 10000)),
        )
        original = controller.policy.decide
        entered = threading.Event()
        proceed = threading.Event()
        calls = 0
        call_lock = threading.Lock()

        def delayed(*args, **kwargs):
            nonlocal calls
            with call_lock:
                calls += 1
                first = calls == 1
            if first:
                entered.set()
                proceed.wait(1)
            return original(*args, **kwargs)

        with patch.object(controller.policy, "decide", side_effect=delayed):
            with concurrent.futures.ThreadPoolExecutor() as pool:
                first = pool.submit(controller.acquire, self.request("first"))
                self.assertTrue(entered.wait(1))
                second = controller.acquire(self.request("second"))
                self.assertTrue(second.admitted)
                self.assertFalse(second.admission_queued)
                controller.release(second)
                proceed.set()
                first_decision = first.result(timeout=1)
        self.assertTrue(first_decision.admitted)
        controller.release(first_decision)

    def test_pressure_does_not_shorten_an_existing_waiters_deadline(self):
        controller = self.controller(wait_seconds=.3)
        held = controller.acquire(self.request())
        pressured = threading.Event()
        controller.policy.admission_wait_seconds = (
            lambda configured: 0.0 if pressured.is_set() else configured
        )
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(controller.acquire, self.request("waiting"))
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            pressured.set()
            time.sleep(.03)
            controller.release(held)
            admitted = pending.result(timeout=1)
        self.assertTrue(admitted.admitted)
        self.assertTrue(admitted.admission_queued)
        controller.release(admitted)

    def test_fifo_prevents_new_arrivals_stealing_a_waiting_lane(self):
        controller = self.controller()
        held = controller.acquire(self.request())
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(controller.acquire, self.request("first"))
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            second = pool.submit(controller.acquire, self.request("second"))
            wait_for(lambda: controller.state()["queued_requests"] == 2)
            controller.release(held)
            a = first.result(timeout=1)
            self.assertFalse(second.done())
            controller.release(a)
            b = second.result(timeout=1)
            controller.release(b)
        self.assertEqual(controller.state()["queued_requests"], 0)

    def test_full_heavy_queue_does_not_block_small_work(self):
        controller = self.controller(max_waiters=1)
        held = controller.acquire(self.request())
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(controller.acquire, self.request())
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            refused = controller.acquire(self.request("b"))
            self.assertFalse(refused.admitted)
            self.assertEqual(refused.admission_reason, "queue_full")
            small = controller.acquire(self.request("b", output=20))
            self.assertTrue(small.admitted)
            self.assertFalse(small.admission_queued)
            controller.release(small)
            controller.release(held)
            controller.release(pending.result(timeout=1))

    def test_customer_token_and_byte_queue_bounds(self):
        cases = (({"max_per_customer": 1}, "customer_queue_full"),
                 ({"max_queued_tokens": 300}, "queued_token_limit"),
                 ({"max_queued_bytes": 150}, "queued_byte_limit"))
        for limits, reason in cases:
            with self.subTest(reason=reason):
                controller = self.controller(**limits)
                held = controller.acquire(self.request())
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pending = pool.submit(controller.acquire, self.request(), body_bytes=100)
                    wait_for(lambda: controller.state()["queued_requests"] == 1)
                    refused = controller.acquire(self.request(), body_bytes=100)
                    self.assertEqual(refused.admission_reason, reason)
                    controller.release(held)
                    controller.release(pending.result(timeout=1))
                self.assertEqual(controller.state()["queued_bytes"], 0)

    def test_wait_expires_without_leaking_a_reservation(self):
        controller = self.controller(wait_seconds=.08)
        held = controller.acquire(self.request())
        refused = controller.acquire(self.request())
        self.assertFalse(refused.admitted)
        self.assertEqual(refused.admission_reason, "queue_timeout")
        self.assertGreaterEqual(refused.admission_wait_seconds, .07)
        self.assertEqual(controller.state()["queued_requests"], 0)
        self.assertEqual(controller.policy.workload_budget.state()["active_requests"], 1)
        controller.release(held)

    def test_cancelled_waiter_is_removed_without_dispatch(self):
        controller = self.controller()
        held = controller.acquire(self.request())
        cancelled = threading.Event()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(controller.acquire, self.request(), cancelled=cancelled.is_set)
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            cancelled.set()
            refused = pending.result(timeout=1)
        self.assertEqual(refused.admission_reason, "client_disconnected")
        self.assertEqual(controller.policy.workload_budget.state()["active_requests"], 1)
        self.assertEqual(controller.state()["queued_requests"], 0)
        controller.release(held)

    def test_socket_check_failure_after_acquire_does_not_leak_a_lease(self):
        controller = self.controller()
        calls = []
        def cancelled():
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("closed file descriptor")
            return False
        result = controller.acquire(self.request(), cancelled=cancelled)
        self.assertEqual(result.admission_reason, "client_disconnected")
        self.assertEqual(controller.policy.workload_budget.state()["active_requests"], 0)
        self.assertEqual(controller._probing, set())

    def test_input_and_individually_impossible_reservations_do_not_wait(self):
        controller = self.controller()
        held = controller.acquire(self.request())
        invalid = controller.acquire(RequestEnvelope("bad", 262144, 200))
        impossible = controller.acquire(self.request(), reserved_prompt_tokens=10000)
        self.assertEqual(invalid.clamp.granted, 0)
        self.assertFalse(impossible.admitted)
        self.assertFalse(invalid.admission_queued)
        self.assertFalse(impossible.admission_queued)
        self.assertEqual(controller.state()["queued_total"], 0)
        controller.release(held)

    def test_policy_exception_cleans_queue_and_probe_ownership(self):
        controller = self.controller()
        held = controller.acquire(self.request())
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(controller.acquire, self.request())
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            with patch.object(controller.policy, "decide", side_effect=RuntimeError("test failure")):
                with self.assertRaises(RuntimeError):
                    pending.result(timeout=1)
        self.assertEqual(controller.state()["queued_requests"], 0)
        self.assertEqual(controller._probing, set())
        controller.release(held)
        next_request = controller.acquire(self.request())
        self.assertTrue(next_request.admitted)
        controller.release(next_request)

    def test_lease_is_returned_if_admission_completes_after_deadline(self):
        controller = self.controller(wait_seconds=.12)
        held = controller.acquire(self.request())
        original = controller.policy.decide
        def slow(*args, **kwargs):
            decision = original(*args, **kwargs)
            if decision.admitted:
                time.sleep(.16)
            return decision
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(controller.acquire, self.request())
            wait_for(lambda: controller.state()["queued_requests"] == 1)
            with patch.object(controller.policy, "decide", side_effect=slow):
                controller.release(held)
                refused = pending.result(timeout=1)
        self.assertEqual(refused.admission_reason, "queue_timeout")
        self.assertEqual(controller.policy.workload_budget.state()["active_requests"], 0)
        self.assertEqual(controller.state()["queued_requests"], 0)

    def test_disabled_controller_retains_fail_fast_behavior(self):
        controller = self.controller(wait_seconds=0)
        held = controller.acquire(self.request())
        refused = controller.acquire(self.request())
        self.assertFalse(refused.admitted)
        self.assertFalse(refused.admission_queued)
        self.assertEqual(controller.state()["queued_total"], 0)
        controller.release(held)

    def test_invalid_queue_configuration_is_rejected(self):
        for values in ({"wait_seconds": -1}, {"wait_seconds": float("nan")},
                       {"wait_seconds": 61}, {"max_waiters": 0}, {"poll_seconds": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AdmissionLimits(**values)


class HttpAdmissionTests(unittest.TestCase):
    post = guard_tests.HttpGuardTests.post
    tearDown = guard_tests.HttpGuardTests.tearDown

    def setUp(self):
        guard_tests.HttpGuardTests.setUp(self)
        self.service.admission = AdmissionController(
            self.service.policy, AdmissionLimits(wait_seconds=2, poll_seconds=.01))

    def test_queued_requests_relay_exactly_once_and_small_work_progresses(self):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(self.post, "large")
            self.assertTrue(self.started.wait(2))
            second = pool.submit(self.post, "large")
            wait_for(lambda: self.service.admission.state()["queued_requests"] == 1)
            self.assertEqual(self.post("small")[0], 200)
            self.release.set()
            self.assertEqual(first.result(timeout=2)[0], 200)
            code, body, headers = second.result(timeout=2)
        self.assertEqual(code, 200)
        self.assertGreater(float(headers["x-k3-admission-wait-ms"]), 0)
        self.assertEqual(len(self.inferences), 3)
        self.assertTrue(all(row["max_tokens"] == 16 for row in self.inferences))
        self.assertTrue(all(row["chat_template_kwargs"] == {"thinking": False} for row in self.inferences))
        self.assertEqual(self.service.policy.workload_budget.state()["reserved_tokens"], 0)

    def test_timeout_is_overload_with_reason_and_no_duplicate_inference(self):
        self.service.admission.limits = replace(self.service.admission.limits, wait_seconds=.1)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(self.post, "large")
            self.assertTrue(self.started.wait(2))
            code, body, headers = self.post("large")
            self.assertEqual(code, 429)
            self.assertEqual(body["error"]["type"], "rate_limit_error")
            self.assertEqual(body["error"]["code"], "queue_timeout")
            self.assertIn("large_context_slots", body["error"]["message"])
            self.assertIn("Retry-After", headers)
            self.assertEqual(len(self.inferences), 1)
            self.release.set()
            self.assertEqual(first.result(timeout=2)[0], 200)

    def test_disconnected_http_waiter_never_reaches_engine(self):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(self.post, "large")
            self.assertTrue(self.started.wait(2))
            payload = json.dumps({"model": "FW-Kimi-K3", "messages": [
                {"role": "user", "content": "large"}], "max_tokens": 16}).encode()
            with socket.create_connection(("127.0.0.1", self.gateway.server_port)) as client:
                client.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                f"Content-Length: {len(payload)}\r\nContent-Type: application/json\r\n\r\n").encode() + payload)
                wait_for(lambda: self.service.admission.state()["queued_requests"] == 1)
            wait_for(lambda: self.service.admission.state()["cancelled_total"] == 1)
            self.assertEqual(len(self.inferences), 1)
            self.assertEqual(self.service.admission.state()["queued_requests"], 0)
            self.release.set()
            self.assertEqual(first.result(timeout=2)[0], 200)

    def test_engine_refresh_is_rate_bounded_and_timestamped(self):
        engine = EngineClient(f"http://127.0.0.1:{self.engine.server_port}", snapshot_ttl=60)
        first = engine.snapshot()
        time.sleep(.26)
        refreshed = engine.refresh_snapshot()
        self.assertGreater(refreshed.sampled_at, first.sampled_at)
        self.assertIs(engine.refresh_snapshot(), refreshed)
        self.assertIs(engine.snapshot(), refreshed)


if __name__ == "__main__":
    unittest.main()
