"""Pressure-based admission: healthy capacity, burst recovery, and real limits."""
import concurrent.futures
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from redesign.gateway.admission import AdmissionController, AdmissionLimits
from redesign.gateway.models import EngineSnapshot, RequestEnvelope
from redesign.gateway.inspection import InspectionError, PromptInspector
from redesign.gateway.policy import build_default
from redesign.gateway.throughput import ThroughputCapacityController, ThroughputLimits
from redesign.gateway.workload import WorkloadBudget, WorkloadLimits
from redesign.tests.test_admission import wait_for
from redesign.tests import test_workload_guard as guard_tests


class Engine:
    def __init__(self):
        self.value = EngineSnapshot(
            .03, 0, 0, 0, 1_641_413,
            mean_itl_seconds=.04, mean_ttft_seconds=.5, mean_prefill_seconds=.5,
        )

    def refresh_snapshot(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def system(initial=32, maximum=64, **queue_options):
    engine = Engine()
    capacity = ThroughputCapacityController(engine, ThroughputLimits(
        initial_requests=initial, max_borrow_requests=maximum,
        minimum_requests=min(8, initial, maximum),
        poll_seconds=.01, freshness_seconds=5, adjustment_seconds=.01,
    ))
    capacity.sample_once()
    policy = build_default(262144, 64)
    policy.capacity_controller = capacity
    policy.workload_budget = WorkloadBudget(WorkloadLimits(
        throughput_first=True, adaptive_borrow_requests=maximum,
        long_output_base_token_budget=12288, long_output_burst_token_budget=49152,
        projected_kv_limit=.92, adaptive_borrow_kv_limit=.88,
    ), capacity)
    values = dict(wait_seconds=.5, max_waiters=128, max_per_customer=128,
                  max_queued_tokens=16_777_216, poll_seconds=.005,
                  fairness_age_seconds=.05)
    values.update(queue_options)
    return engine, capacity, policy, AdmissionController(policy, AdmissionLimits(**values))


class ThroughputControllerTests(unittest.TestCase):
    def test_transient_pressure_keeps_queue_and_execution_capacity(self):
        engine, capacity, _, _ = system()
        engine.value = replace(engine.value, running=12, mean_ttft_seconds=2.1)
        capacity.sample_once()
        self.assertEqual(capacity.execution_limit(), 32)
        self.assertEqual(capacity.queue_wait_seconds(60), 60)
        self.assertEqual(capacity.queue_limit(128), 128)
        engine.value = replace(engine.value, waiting=1, mean_ttft_seconds=.5)
        capacity.sample_once()
        self.assertEqual(capacity.execution_limit(), 32)
        self.assertEqual(capacity.queue_wait_seconds(60), 60)

    def test_sustained_decode_pressure_reduces_starts_with_low_kv(self):
        engine, capacity, _, _ = system()
        engine.value = replace(engine.value, running=16, mean_itl_seconds=.2)
        capacity.sample_once()
        capacity.sample_once()
        self.assertEqual(capacity.execution_limit(), 32)
        capacity.sample_once()
        self.assertGreater(capacity.execution_limit(), 0)
        self.assertEqual(capacity.execution_limit(), 25)
        self.assertEqual(capacity.state()["state"], "pressure")
        self.assertIn("decode_latency", capacity.state()["pressure_reasons"])
        self.assertEqual(capacity.queue_limit(128), 128)

    def test_sustained_engine_queue_pauses_starts_until_healthy_recovery(self):
        engine, capacity, _, _ = system()
        engine.value = replace(engine.value, running=10, waiting=4)
        for _ in range(3):
            capacity.sample_once()
        self.assertEqual(capacity.execution_limit(), 0)
        engine.value = replace(engine.value, waiting=0)
        for _ in range(3):
            capacity.sample_once()
            self.assertEqual(capacity.execution_limit(), 0)
        capacity.sample_once()
        self.assertGreater(capacity.execution_limit(), 0)
        self.assertEqual(capacity.queue_wait_seconds(60), 60)

    def test_memory_distress_and_preemption_pause_every_class_immediately(self):
        for changes in ({"kv_usage": .98}, {"preemptions_per_minute": 1}):
            with self.subTest(changes=changes):
                engine, capacity, policy, _ = system()
                engine.value = replace(engine.value, **changes)
                capacity.sample_once()
                self.assertEqual(capacity.execution_limit(), 0)
                for request in (RequestEnvelope("a", 10, 1), RequestEnvelope("a", 20000, 32)):
                    self.assertFalse(policy.decide(request).admitted)
                self.assertEqual(capacity.queue_wait_seconds(60), 60)

    def test_unknown_or_stale_metrics_cannot_expand_execution(self):
        for value in (OSError("unavailable"), EngineSnapshot(.1, 1, 0, 0),
                      EngineSnapshot(float("nan"), 1, 0, 0, 1000000)):
            with self.subTest(value=value):
                engine, capacity, _, _ = system()
                engine.value = value
                capacity.sample_once()
                self.assertEqual(capacity.execution_limit(), 0)
                self.assertEqual(capacity.queue_limit(128), 128)
        _, capacity, _, _ = system()
        capacity._sampled_at -= 10
        self.assertEqual(capacity.execution_limit(), 0)

    def test_capacity_grows_with_healthy_demand_and_stops_at_engine_ceiling(self):
        engine = Engine()
        capacity = ThroughputCapacityController(engine, ThroughputLimits())
        with patch("redesign.gateway.throughput.time.monotonic") as clock:
            for i in range(10):
                clock.return_value = 100 + i * 2
                capacity.sample_once()
            self.assertEqual(capacity.execution_limit(), 8)
            capacity.set_queued_demand(20)
            for i in range(20):
                clock.return_value = 120 + i * 2
                capacity.sample_once()
            self.assertEqual(capacity.execution_limit(), 64)
            self.assertEqual(capacity.state()["queued_demand"], 20)

    def test_idle_returns_to_latency_safe_baseline_after_demand_burst(self):
        engine = Engine()
        capacity = ThroughputCapacityController(engine, ThroughputLimits())
        with patch("redesign.gateway.throughput.time.monotonic") as clock:
            capacity.set_queued_demand(10)
            for i in range(8):
                clock.return_value = 100 + i * 2
                capacity.sample_once()
            self.assertGreater(capacity.execution_limit(), 8)
            capacity.set_queued_demand(0)
            engine.value = replace(engine.value, running=0)
            clock.return_value = 120
            capacity.sample_once()
            self.assertEqual(capacity.execution_limit(), 8)
            self.assertEqual(capacity.queue_wait_seconds(60), 60)

    def test_active_occupancy_without_a_queue_does_not_expand_capacity(self):
        engine = Engine()
        capacity = ThroughputCapacityController(engine, ThroughputLimits())
        engine.value = replace(engine.value, running=8)
        with patch("redesign.gateway.throughput.time.monotonic") as clock:
            for i in range(12):
                clock.return_value = 100 + i * 2
                capacity.sample_once()
            self.assertEqual(capacity.execution_limit(), 8)
            self.assertEqual(capacity.state()["queued_demand"], 0)

    def test_healthy_traffic_recovers_to_baseline_without_a_queue(self):
        engine, capacity, _, _ = system(initial=16, maximum=24)
        engine.value = replace(engine.value, running=16, mean_itl_seconds=.2)
        for _ in range(3):
            capacity.sample_once()
            time.sleep(.011)
        self.assertEqual(capacity.execution_limit(), 12)

        engine.value = replace(engine.value, running=8, mean_itl_seconds=.02)
        for _ in range(4):
            capacity.sample_once()
            time.sleep(.011)
        self.assertEqual(capacity.execution_limit(), 16)
        self.assertEqual(capacity.state()["queued_demand"], 0)

    def test_intrinsic_cold_prefill_is_a_warning_without_load_evidence(self):
        engine, capacity, _, _ = system()
        engine.value = replace(engine.value, running=4, mean_ttft_seconds=12,
                               mean_prefill_seconds=12, mean_itl_seconds=.04)
        for _ in range(10):
            capacity.sample_once()
        self.assertEqual(capacity.execution_limit(), 32)
        self.assertEqual(capacity.state()["state"], "warm")
        self.assertEqual(capacity.queue_wait_seconds(60), 60)

    def test_drained_occupancy_cannot_redefine_capacity_as_one(self):
        engine, capacity, _, _ = system()
        engine.value = replace(engine.value, running=1, mean_itl_seconds=.2)
        for _ in range(30):
            capacity.sample_once()
            time.sleep(.011)
        self.assertEqual(capacity.execution_limit(), 8)
        self.assertEqual(capacity.state()["state"], "pressure")


class SharedResourceTests(unittest.TestCase):
    def test_idle_class_capacity_and_long_output_capacity_are_shared(self):
        engine, capacity, policy, _ = system(initial=64)
        decisions = [policy.decide(RequestEnvelope("pool", 100, 4096)) for _ in range(40)]
        self.assertTrue(all(d.admitted for d in decisions))
        engine.value = replace(engine.value, running=40)
        capacity.sample_once()
        large = policy.decide(RequestEnvelope("other", 40000, 8192))
        self.assertTrue(large.admitted, large.reason)
        self.assertEqual(large.clamp.granted, 1536)
        self.assertEqual(policy.workload_budget.state()["long_output_budget_enabled"], 0)
        for decision in decisions + [large]:
            policy.release(decision)
        self.assertEqual(policy.workload_budget.state()["reserved_tokens"], 0)

    def test_healthy_sixteenth_request_does_not_shrink_output_capacity(self):
        engine, capacity, policy, _ = system()
        held = [policy.decide(RequestEnvelope("pool", 100, 4096)) for _ in range(16)]
        self.assertTrue(all(d.admitted for d in held))
        engine.value = replace(engine.value, running=16)
        capacity.sample_once()
        next_request = policy.decide(RequestEnvelope("pool", 100, 4096))
        self.assertTrue(next_request.admitted, next_request.reason)
        for decision in held + [next_request]:
            policy.release(decision)

    def test_parallel_arrivals_cannot_exceed_global_execution_limit(self):
        _, _, policy, _ = system(initial=64)
        barrier = threading.Barrier(100)
        def arrive(i):
            barrier.wait()
            return policy.decide(RequestEnvelope(str(i), 100, 1, has_tools=i % 2 == 0))
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as pool:
            decisions = list(pool.map(arrive, range(100)))
        self.assertEqual(sum(d.admitted for d in decisions), 64)
        for decision in decisions:
            policy.release(decision)
            policy.release(decision)
        self.assertEqual(policy.workload_budget.state()["active_requests"], 0)
        self.assertEqual(sum(policy._budget._in_flight.values()), 0)

    def test_conservative_formats_are_bounded_by_measured_memory(self):
        _, _, policy, _ = system()
        request = RequestEnvelope("pool", 100, 1536, has_images=True)
        decisions = [policy.decide(request, reserved_prompt_tokens=262144) for _ in range(6)]
        self.assertEqual(sum(d.admitted for d in decisions), 5)
        self.assertIn("reserved_tokens", decisions[-1].reason)
        self.assertLessEqual(policy.workload_budget.state()["reserved_tokens"], int(1_641_413*.92))
        for decision in decisions:
            policy.release(decision)

    def test_actual_kv_pressure_cannot_be_hidden_by_small_reservations(self):
        engine, capacity, policy, _ = system()
        engine.value = replace(engine.value, kv_usage=.93)
        capacity.sample_once()
        result = policy.decide(RequestEnvelope("a", 100, 1))
        self.assertFalse(result.admitted)
        self.assertIn("kv_headroom", result.reason)

    def test_memory_reservations_cover_full_grants_even_after_short_outputs(self):
        _, _, policy, _ = system()
        budget = policy.workload_budget
        for _ in range(10):
            lease, _ = budget.acquire(100, 8192, class_name="P0-interactive")
            budget.release(lease, 1)
        request = policy.decide(RequestEnvelope("a", 20000, 8192))
        self.assertTrue(request.admitted)
        self.assertEqual(budget.state()["reserved_tokens"], 21536)
        policy.release(request)


class ThroughputQueueTests(unittest.TestCase):
    def test_burst_waits_through_pressure_then_runs_once_capacity_recovers(self):
        engine, capacity, policy, admission = system()
        engine.value = replace(engine.value, running=2, waiting=3)
        for _ in range(3):
            capacity.sample_once()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(admission.acquire, RequestEnvelope("pool", 100, 4096))
            wait_for(lambda: admission.state()["queued_requests"] == 1)
            self.assertEqual(policy.workload_budget.state()["active_requests"], 0)
            self.assertEqual(capacity.state()["queued_demand"], 1)
            engine.value = replace(engine.value, running=0, waiting=0)
            for _ in range(4):
                capacity.sample_once()
            admitted = pending.result(timeout=1)
        self.assertTrue(admitted.admitted)
        self.assertTrue(admitted.admission_queued)
        self.assertEqual(admitted.clamp.granted, 2048)
        self.assertEqual(capacity.state()["queued_demand"], 0)
        admission.release(admitted)

    def test_aged_large_lane_gets_capacity_before_new_small_arrivals(self):
        _, _, _, admission = system(initial=1, maximum=1, fairness_age_seconds=.02)
        held = admission.acquire(RequestEnvelope("busy", 100, 100))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            large = pool.submit(admission.acquire, RequestEnvelope("large", 40000, 4096))
            wait_for(lambda: admission.state()["queued_requests"] == 1)
            time.sleep(.03)
            small = pool.submit(admission.acquire, RequestEnvelope("small", 100, 1))
            wait_for(lambda: admission.state()["queued_requests"] == 2)
            admission.release(held)
            first = large.result(timeout=1)
            self.assertTrue(first.admitted)
            self.assertFalse(small.done())
            admission.release(first)
            second = small.result(timeout=1)
            self.assertTrue(second.admitted)
            admission.release(second)

    def test_impossible_reservation_does_not_age_block_other_lanes(self):
        _, _, policy, admission = system(initial=1, maximum=1)
        impossible = admission.acquire(RequestEnvelope("batch", 10, 1), reserved_prompt_tokens=3_000_000)
        self.assertFalse(impossible.admitted)
        self.assertFalse(impossible.admission_queued)
        small = admission.acquire(RequestEnvelope("small", 100, 1))
        self.assertTrue(small.admitted)
        admission.release(small)
        self.assertEqual(policy.workload_budget.state()["reserved_tokens"], 0)

    def test_expiry_and_cancellation_return_queue_capacity(self):
        _, _, policy, admission = system(initial=1, maximum=1, wait_seconds=.08)
        held = admission.acquire(RequestEnvelope("busy", 100, 4096))
        expired = admission.acquire(RequestEnvelope("expired", 100, 4096))
        self.assertEqual(expired.admission_reason, "queue_timeout")
        cancelled = threading.Event()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(admission.acquire, RequestEnvelope("gone", 100, 4096), cancelled=cancelled.is_set)
            wait_for(lambda: admission.state()["queued_requests"] == 1)
            cancelled.set()
            result = pending.result(timeout=1)
        self.assertEqual(result.admission_reason, "client_disconnected")
        self.assertEqual(admission.state()["queued_requests"], 0)
        self.assertEqual(policy.workload_budget.state()["active_requests"], 1)
        admission.release(held)

    def test_expired_preparation_deadline_never_dispatches_inference(self):
        _, _, policy, admission = system()
        result = admission.acquire(RequestEnvelope("late", 100, 1), deadline=time.monotonic() - .01)
        self.assertFalse(result.admitted)
        self.assertEqual(result.admission_reason, "queue_timeout")
        self.assertEqual(policy.workload_budget.state()["active_requests"], 0)

    def test_prompt_inspection_waits_for_burst_slot_and_honors_cancellation(self):
        inspector = PromptInspector("http://127.0.0.1:1", max_inflight=1, slot_wait_seconds=1)
        inspector._slots.acquire()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(inspector.inspect, {"prompt": [1, 2, 3]}, deadline=time.monotonic() + .5)
            time.sleep(.02)
            self.assertFalse(pending.done())
            inspector._slots.release()
            self.assertEqual(pending.result(timeout=1).tokens, 3)
        with self.assertRaises(InspectionError) as error:
            inspector.inspect({"prompt": [1]}, cancelled=lambda: True)
        self.assertEqual(error.exception.reason, "client_disconnected")


class ThroughputHttpTests(unittest.TestCase):
    post = guard_tests.HttpGuardTests.post
    tearDown = guard_tests.HttpGuardTests.tearDown

    def setUp(self):
        guard_tests.HttpGuardTests.setUp(self)
        capacity = ThroughputCapacityController(self.service.engine, ThroughputLimits(
            max_borrow_requests=1, initial_requests=1, minimum_requests=1,
        ))
        capacity.sample_once()
        self.service.capacity_controller = capacity
        self.service.policy.capacity_controller = capacity
        self.service.policy.workload_budget = WorkloadBudget(WorkloadLimits(
            throughput_first=True, adaptive_borrow_requests=1,
        ), capacity)
        self.service.admission = AdmissionController(self.service.policy, AdmissionLimits(
            wait_seconds=.08, poll_seconds=.005,
        ))

    def test_exhausted_wait_is_429_and_recovery_preserves_payload(self):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(self.post, "large")
            self.assertTrue(self.started.wait(1))
            code, body, headers = self.post("small")
            self.assertEqual(code, 429)
            self.assertEqual(body["error"]["type"], "rate_limit_error")
            self.assertEqual(body["error"]["code"], "queue_timeout")
            self.assertGreater(float(headers["x-k3-admission-wait-ms"]), 50)
            self.assertEqual(len(self.inferences), 1)
            self.release.set()
            self.assertEqual(first.result(timeout=1)[0], 200)
        self.assertEqual(self.post("small")[0], 200)
        self.assertTrue(all(r["max_tokens"] == 16 for r in self.inferences))
        self.assertEqual(self.service.policy.workload_budget.state()["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
