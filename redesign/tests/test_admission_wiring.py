"""Production builder connects lease accounting to observable admission gauges."""
import unittest

from redesign.gateway.backpressure import CircuitBreaker, StaticHealthSource
from redesign.gateway.models import EngineSnapshot, RequestEnvelope
from redesign.gateway.server import build_service


class AdmissionWiringTest(unittest.TestCase):
    def test_builder_and_gauges_follow_borrowed_leases_and_release(self):
        service = build_service("http://127.0.0.1:1", 262144, 64, None, None, True,
                                borrowing_enabled=True, context_token_budget=1_310_000)
        service.policy._breaker = CircuitBreaker(StaticHealthSource(EngineSnapshot.healthy()))
        request = RequestEnvelope("test", 1000, 16, has_tools=True)
        decisions = [service.policy.decide(request) for _ in range(40)]
        self.assertTrue(all(d.admitted for d in decisions))
        self.assertEqual(sum(bool(d.metadata["borrowed"]) for d in decisions), 12)
        service.publish_admission_metrics()
        metrics = service.registry.render()
        self.assertIn("k3_gateway_admission_in_flight 40\n", metrics)
        self.assertIn("k3_gateway_admission_borrowed_in_flight 12\n", metrics)
        self.assertIn("k3_gateway_admission_context_work_tokens 40640\n", metrics)
        for decision in decisions:
            service.policy.release(decision)
            service.policy.release(decision)
        service.publish_admission_metrics()
        self.assertIn("k3_gateway_admission_in_flight 0\n", service.registry.render())
        self.assertIn("k3_gateway_admission_context_work_tokens 0\n", service.registry.render())


if __name__ == "__main__":
    unittest.main()
