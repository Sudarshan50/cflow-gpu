import unittest

from redesign.gateway.capacity import AdaptiveCapacityController, CapacityLimits
from redesign.gateway.models import EngineSnapshot


class _Engine:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)

    def refresh_snapshot(self):
        item = next(self.snapshots)
        if isinstance(item, Exception):
            raise item
        return item


def snapshot(**changes):
    values = dict(
        kv_usage=.2,
        running=4,
        waiting=0,
        preemptions_per_minute=0,
        kv_capacity_tokens=1_000_000,
        mean_itl_seconds=.04,
        mean_ttft_seconds=.5,
        mean_prefill_seconds=.5,
    )
    values.update(changes)
    return EngineSnapshot(**values)


class AdaptiveCapacityControllerTest(unittest.TestCase):
    def test_green_capacity_ramps_and_pressure_drops_immediately(self):
        controller = AdaptiveCapacityController(
            _Engine([snapshot(), snapshot(), snapshot(waiting=1)]),
            CapacityLimits(max_borrow_requests=32, increase_step=4),
        )
        controller.sample_once()
        self.assertEqual(controller.borrow_limit(), 4)
        self.assertTrue(controller.allows_burst())
        controller.sample_once()
        self.assertEqual(controller.borrow_limit(), 8)
        controller.sample_once()
        self.assertEqual(controller.borrow_limit(), 0)
        self.assertFalse(controller.allows_burst())
        self.assertEqual(controller.queue_wait_seconds(3), 0)
        self.assertEqual(controller.queue_limit(16), 4)

    def test_warm_state_caps_borrowing_and_shortens_queue(self):
        controller = AdaptiveCapacityController(
            _Engine([snapshot(kv_usage=.6)]),
            CapacityLimits(max_borrow_requests=32, increase_step=32),
        )
        controller.sample_once()
        self.assertEqual(controller.borrow_limit(), 24)
        self.assertFalse(controller.allows_burst())
        self.assertEqual(controller.queue_wait_seconds(3), 1)
        self.assertEqual(controller.queue_limit(16), 8)

    def test_scrape_failure_and_stale_snapshot_fail_closed_for_borrowing(self):
        controller = AdaptiveCapacityController(
            _Engine([snapshot(), OSError("metrics unavailable")]),
            CapacityLimits(max_borrow_requests=32, increase_step=32, freshness_seconds=2),
        )
        controller.sample_once()
        self.assertEqual(controller.borrow_limit(), 32)
        controller.sample_once()
        self.assertEqual(controller.borrow_limit(), 0)
        self.assertEqual(controller.state()["state"], "stale")

        controller._sampled_at -= 3
        self.assertEqual(controller.borrow_limit(), 0)
        with self.assertRaises(OSError):
            controller.snapshot()

    def test_decode_and_prefill_latency_are_independent_stop_signals(self):
        for changes in (
            {"mean_itl_seconds": .13},
            {"mean_ttft_seconds": 5.1},
            {"mean_prefill_seconds": 10.1},
            {"preemptions_per_minute": .1},
        ):
            with self.subTest(changes=changes):
                controller = AdaptiveCapacityController(
                    _Engine([snapshot(**changes)]),
                    CapacityLimits(increase_step=32),
                )
                controller.sample_once()
                self.assertEqual(controller.borrow_limit(), 0)
                self.assertEqual(controller.state()["state"], "pressure")

    def test_completed_latency_does_not_keep_an_idle_engine_pressured(self):
        controller = AdaptiveCapacityController(
            _Engine([snapshot(
                running=0,
                mean_itl_seconds=.5,
                mean_ttft_seconds=20,
                mean_prefill_seconds=30,
            )]),
            CapacityLimits(increase_step=32),
        )
        controller.sample_once()
        self.assertEqual(controller.state()["state"], "green")
        self.assertEqual(controller.borrow_limit(), 32)
        self.assertTrue(controller.allows_burst())

    def test_reconciliation_waits_for_a_new_background_sample(self):
        controller = AdaptiveCapacityController(
            _Engine([snapshot(running=1), snapshot(running=0)]),
            CapacityLimits(
                poll_seconds=.05, freshness_seconds=1, increase_step=32
            ),
        )
        try:
            controller.start()
            self.assertEqual(controller.snapshot().running, 1)
            self.assertEqual(controller.refresh_snapshot().running, 0)
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
