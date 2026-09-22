"""Offline admission and upstream ownership regressions; no inference required."""

from __future__ import annotations

import http.client
import io
import json
import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from redesign.gateway.backpressure import CircuitBreaker, ClassBudget, StaticHealthSource
from redesign.gateway.capture import MemorySink, TraceRecorder
from redesign.gateway.classification import ALL_CLASSES, INTERACTIVE, SHORT_CHAT, Classifier
from redesign.gateway.clamping import TokenClamp
from redesign.gateway.engine import EngineClient
from redesign.gateway.metrics import Registry
from redesign.gateway.models import EngineSnapshot
from redesign.gateway.policy import GatewayPolicy
from redesign.gateway.server import GatewayService, Handler


def fake_engine(status=200):
    response = Mock(spec=http.client.HTTPResponse)
    response.status = status
    response.getheaders.return_value = [("Content-Type", "application/json")]
    response.read.return_value = b'{"choices":[]}'
    response.read1.side_effect = [b"data: test\n\n", b""]
    connection = Mock(spec=http.client.HTTPConnection)
    connection.getresponse.return_value = response
    engine = EngineClient("http://offline.invalid")
    engine._connect = Mock(return_value=connection)
    return engine, connection, response


def fake_handler(payload=None, status=200):
    engine, connection, response = fake_engine(status)
    budget = ClassBudget(2, {SHORT_CHAT.name: 0.5, INTERACTIVE.name: 0.5})
    policy = GatewayPolicy(
        Classifier(), TokenClamp(262_144), budget,
        CircuitBreaker(StaticHealthSource(EngineSnapshot.healthy())),
    )
    service = GatewayService(
        policy, engine, Mock(estimate=Mock(return_value=1_000)),
        TraceRecorder(MemorySink()), Registry(), send_priority=True,
    )
    handler = Handler.__new__(Handler)
    handler.service = service
    handler.path = "/v1/chat/completions"
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)
    handler.close_connection = False
    handler._read_payload = Mock(
        return_value={"prompt": "hi", "max_tokens": 1_000}
        if payload is None else payload
    )
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock()
    handler._json = Mock()
    handler.log_error = Mock()
    handler.wfile = io.BytesIO()
    return handler, budget, connection, response


def counter(handler, name):
    prefix = f"k3_gateway_{name}{{"
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in handler.service.registry.render().splitlines()
        if line.startswith(prefix)
    )


class MixedAdmissionTest(unittest.TestCase):
    def test_class_limits_and_total_limit_both_apply_and_release(self):
        budget = ClassBudget.from_classes(96, ALL_CLASSES)
        self.assertGreater(sum(budget.limit_for(c) for c in ALL_CLASSES), 96)
        held = []
        for traffic_class in ALL_CLASSES:
            for _ in range(budget.limit_for(traffic_class)):
                if budget.try_acquire(traffic_class):
                    held.append(traffic_class)
            self.assertLessEqual(budget.in_flight(traffic_class), budget.limit_for(traffic_class))
        self.assertEqual(len(held), 96)
        for traffic_class in ALL_CLASSES:
            self.assertFalse(budget.try_acquire(traffic_class))

        released = held.pop()
        budget.release(released)
        self.assertTrue(budget.try_acquire(released))
        held.append(released)
        for traffic_class in held:
            budget.release(traffic_class)
        for traffic_class in ALL_CLASSES:
            budget.release(traffic_class)  # An empty class cannot free another slot.
            self.assertEqual(budget.in_flight(traffic_class), 0)
        self.assertTrue(budget.try_acquire(released))
        budget.release(released)

    def test_class_exhaustion_does_not_consume_total_capacity(self):
        budget = ClassBudget(4, {SHORT_CHAT.name: 0.25, INTERACTIVE.name: 1.0})
        self.assertTrue(budget.try_acquire(SHORT_CHAT))
        self.assertFalse(budget.try_acquire(SHORT_CHAT))
        for _ in range(3):
            self.assertTrue(budget.try_acquire(INTERACTIVE))
        self.assertFalse(budget.try_acquire(INTERACTIVE))
        budget.release(SHORT_CHAT)
        budget.release(SHORT_CHAT)
        self.assertTrue(budget.try_acquire(INTERACTIVE))
        self.assertFalse(budget.try_acquire(INTERACTIVE))
        for _ in range(4):
            budget.release(INTERACTIVE)
        self.assertEqual(budget.in_flight(INTERACTIVE), 0)

    def test_mixed_class_admissions_are_atomic_under_contention(self):
        classes = (SHORT_CHAT, INTERACTIVE)
        budget = ClassBudget(7, {c.name: 1.0 for c in classes})
        start = threading.Barrier(32)

        def acquire(index):
            traffic_class = classes[index % len(classes)]
            start.wait(timeout=10)
            return traffic_class if budget.try_acquire(traffic_class) else None

        with ThreadPoolExecutor(max_workers=32) as pool:
            # Hold successful acquisitions until every contender has finished.
            admitted = [c for c in pool.map(acquire, range(32)) if c is not None]
        self.assertEqual(len(admitted), budget.ceiling)
        self.assertEqual(sum(budget.in_flight(c) for c in classes), budget.ceiling)
        for traffic_class in admitted:
            budget.release(traffic_class)
        self.assertEqual(sum(budget.in_flight(c) for c in classes), 0)


class PostAdmissionTest(unittest.TestCase):
    def test_throwing_recorder_releases_only_the_acquired_slot(self):
        handler, budget, connection, _ = fake_handler()
        self.assertTrue(budget.try_acquire(INTERACTIVE))
        handler.service.recorder.record = Mock(side_effect=RuntimeError("sink failed"))
        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            handler.do_POST()
        self.assertEqual(budget.in_flight(SHORT_CHAT), 0)
        self.assertEqual(budget.in_flight(INTERACTIVE), 1)
        self.assertTrue(budget.try_acquire(SHORT_CHAT))
        connection.request.assert_not_called()

    def test_other_postadmission_exceptions_release_slots(self):
        for stage in ("registry", "clamp", "proxy", "header", "ttft", "total"):
            with self.subTest(stage=stage):
                handler, budget, connection, response = fake_handler()
                error = RuntimeError(stage)
                if stage == "registry":
                    target, attribute = handler.service.registry, "increment"
                elif stage == "clamp":
                    target, attribute = "redesign.gateway.server.apply_granted_tokens", None
                elif stage == "proxy":
                    target, attribute = connection, "request"
                elif stage == "header":
                    target, attribute = handler, "send_header"
                else:
                    target, attribute = handler.service.registry, f"observe_{stage}"
                context = (
                    patch(target, side_effect=error) if attribute is None
                    else patch.object(target, attribute, side_effect=error)
                )
                with context, self.assertRaisesRegex(RuntimeError, stage):
                    handler.do_POST()
                self.assertEqual(budget.in_flight(SHORT_CHAT), 0)
                self.assertTrue(budget.try_acquire(SHORT_CHAT))
                if stage not in ("registry", "clamp"):
                    connection.close.assert_called_once()
                if stage in ("header", "ttft", "total"):
                    response.close.assert_called_once()

    def test_rejection_does_not_release_another_requests_slot(self):
        handler, budget, connection, _ = fake_handler()
        self.assertTrue(budget.try_acquire(SHORT_CHAT))
        handler.service.recorder.record = Mock(side_effect=RuntimeError("sink failed"))
        with self.assertRaises(RuntimeError):
            handler.do_POST()
        self.assertEqual(budget.in_flight(SHORT_CHAT), 1)
        connection.request.assert_not_called()

    def test_header_disconnect_closes_unstarted_upstream_and_releases(self):
        for stage in ("send_response", "send_header", "end_headers"):
            for error in (BrokenPipeError, ConnectionResetError):
                with self.subTest(stage=stage, error=error):
                    handler, budget, connection, response = fake_handler()
                    getattr(handler, stage).side_effect = error("client gone")
                    handler.do_POST()
                    response.read.assert_not_called()
                    response.close.assert_called_once()
                    connection.close.assert_called_once()
                    self.assertEqual(budget.in_flight(SHORT_CHAT), 0)
                    self.assertEqual(counter(handler, "client_disconnects_total"), 1)
                    self.assertTrue(handler.close_connection)

    def test_body_disconnect_closes_upstream_and_releases(self):
        handler, budget, connection, response = fake_handler({"prompt": "hi", "stream": True})
        handler.wfile = Mock(write=Mock(side_effect=BrokenPipeError("client gone")))
        handler.do_POST()
        response.read1.assert_called_once()
        response.close.assert_called_once()
        connection.close.assert_called_once()
        self.assertEqual(budget.in_flight(SHORT_CHAT), 0)
        self.assertEqual(counter(handler, "client_disconnects_total"), 1)

    def test_fallback_instrumentation_failure_closes_owned_response(self):
        handler, budget, local_connection, _ = fake_handler()
        local_connection.request.side_effect = OSError("local unavailable")
        offbox, connection, response = fake_engine()
        handler.service.offbox = offbox
        increment = handler.service.registry.increment

        def fail_fallback(name, **labels):
            if name == "offbox_fallback_total":
                raise RuntimeError("counter failed")
            increment(name, **labels)

        handler.service.registry.increment = fail_fallback
        with self.assertRaisesRegex(RuntimeError, "counter failed"):
            handler.do_POST()
        local_connection.close.assert_called_once()
        response.close.assert_called_once()
        connection.close.assert_called_once()
        self.assertEqual(budget.in_flight(SHORT_CHAT), 0)


class EngineOwnershipTest(unittest.TestCase):
    def test_proxy_setup_failures_close_every_resource_already_acquired(self):
        for stage in ("request", "getresponse", "getheaders"):
            with self.subTest(stage=stage):
                engine, connection, response = fake_engine()
                target = response if stage == "getheaders" else connection
                getattr(target, stage).side_effect = http.client.HTTPException(stage)
                with self.assertRaises(http.client.HTTPException):
                    engine.proxy("/v1/chat/completions", {}, stream=True)
                connection.close.assert_called_once()
                if stage == "getheaders":
                    response.close.assert_called_once()
                else:
                    response.close.assert_not_called()

    def test_close_before_iteration_is_explicit_and_idempotent(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                engine, connection, response = fake_engine()
                proxied = engine.proxy("/v1/chat/completions", {}, stream=stream)
                proxied.close()
                proxied.close()
                self.assertEqual(list(proxied.body), [])
                response.read.assert_not_called()
                response.read1.assert_not_called()
                response.close.assert_called_once()
                connection.close.assert_called_once()

    def test_exhaustion_and_read_failure_close_without_caller_cleanup(self):
        for stream in (False, True):
            for fail in (False, True):
                with self.subTest(stream=stream, fail=fail):
                    engine, connection, response = fake_engine()
                    if fail:
                        read = response.read1 if stream else response.read
                        read.side_effect = http.client.IncompleteRead(b"partial")
                    proxied = engine.proxy("/v1/chat/completions", {}, stream=stream)
                    if fail:
                        with self.assertRaises(http.client.IncompleteRead):
                            list(proxied.body)
                    else:
                        self.assertTrue(list(proxied.body))
                    proxied.close()
                    response.close.assert_called_once()
                    connection.close.assert_called_once()

    def test_connection_is_closed_even_if_response_close_raises(self):
        engine, connection, response = fake_engine()
        proxied = engine.proxy("/v1/chat/completions", {}, stream=True)
        response.close.side_effect = OSError("close failed")
        with self.assertRaisesRegex(OSError, "close failed"):
            proxied.close()
        connection.close.assert_called_once()

    def test_real_http_response_owns_socket_after_connection_relinquishes_it(self):
        # A socketpair exercises real HTTPConnection/HTTPResponse ownership,
        # including Connection: close, without listening or contacting a service.
        for mode in ("unstarted", "partial", "read_failure", "headers_failure"):
            with self.subTest(mode=mode):
                client_socket, peer = socket.socketpair()
                self.addCleanup(client_socket.close)
                self.addCleanup(peer.close)
                client_socket.settimeout(2)
                peer.settimeout(2)
                peer.sendall(
                    b"HTTP/1.1 200 OK\r\nConnection: close\r\n"
                    b"Content-Length: 100\r\n\r\npartial"
                )
                peer.shutdown(socket.SHUT_WR)
                connection = http.client.HTTPConnection("offline.invalid")
                connection.sock = client_socket
                engine = EngineClient("http://offline.invalid")
                engine._connect = Mock(return_value=connection)
                acquired = []
                getresponse = connection.getresponse

                def capture_response():
                    response = getresponse()
                    acquired.append((response, response.fp))
                    if mode == "headers_failure":
                        response.getheaders = Mock(side_effect=RuntimeError("headers"))
                    return response

                with patch.object(connection, "getresponse", side_effect=capture_response):
                    if mode == "headers_failure":
                        with self.assertRaisesRegex(RuntimeError, "headers"):
                            engine.proxy("/v1/completions", {}, stream=True)
                    else:
                        proxied = engine.proxy(
                            "/v1/completions", {}, stream=mode != "read_failure"
                        )
                        self.assertIsNone(connection.sock)
                        self.assertFalse(acquired[0][1].closed)
                        if mode == "partial":
                            self.assertEqual(next(proxied.body), b"partial")
                        elif mode == "read_failure":
                            with self.assertRaises(http.client.IncompleteRead):
                                list(proxied.body)
                        proxied.close()
                response, file = acquired[0]
                self.assertIsNone(response.fp)
                self.assertTrue(file.closed)
                self.assertEqual(client_socket.fileno(), -1)
                # Drain the request; EOF proves the actual peer sees closure.
                while peer.recv(8192):
                    pass


class MetricsAndStatusTest(unittest.TestCase):
    def test_current_and_legacy_vllm_kv_metrics_drive_distress(self):
        for name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            with self.subTest(name=name):
                engine, connection, response = fake_engine()
                response.read.return_value = (
                    f'{name}{{model_name="test"}} 0.96\n'
                    "vllm:num_requests_running 74\n"
                    "vllm:num_requests_waiting 21\n"
                ).encode()
                snapshot = engine.snapshot()
                self.assertEqual(snapshot.kv_usage, 0.96)
                self.assertEqual(snapshot.running, 74)
                self.assertEqual(snapshot.waiting, 21)
                self.assertTrue(CircuitBreaker(engine).evaluate().distressed)
                connection.close.assert_called_once()

    def test_current_metric_has_precedence_over_legacy_alias(self):
        engine, _, response = fake_engine()
        response.read.return_value = (
            b"vllm:kv_cache_usage_perc 0.98\nvllm:gpu_cache_usage_perc 0.1\n"
        )
        self.assertEqual(engine.snapshot().kv_usage, 0.98)

    def test_recent_inter_token_latency_is_derived_from_counter_deltas(self):
        engine, _, response = fake_engine()
        engine._snapshot_ttl = 0
        response.read.side_effect = [
            b"vllm:inter_token_latency_seconds_sum 10\n"
            b"vllm:inter_token_latency_seconds_count 100\n"
            b"vllm:time_to_first_token_seconds_sum 20\n"
            b"vllm:time_to_first_token_seconds_count 10\n"
            b"vllm:request_prefill_time_seconds_sum 30\n"
            b"vllm:request_prefill_time_seconds_count 10\n",
            b"vllm:inter_token_latency_seconds_sum 16\n"
            b"vllm:inter_token_latency_seconds_count 150\n"
            b"vllm:time_to_first_token_seconds_sum 25\n"
            b"vllm:time_to_first_token_seconds_count 12\n"
            b"vllm:request_prefill_time_seconds_sum 34\n"
            b"vllm:request_prefill_time_seconds_count 12\n"
            b'vllm:cache_config_info{block_size="384",mamba_block_size="16",'
            b'kv_cache_size_tokens="1000000"} 1\n',
        ]
        self.assertIsNone(engine.snapshot().mean_itl_seconds)
        current = engine.snapshot()
        self.assertAlmostEqual(current.mean_itl_seconds, .12)
        self.assertAlmostEqual(current.mean_ttft_seconds, 2.5)
        self.assertAlmostEqual(current.mean_prefill_seconds, 2)
        self.assertEqual(current.cache_block_size_tokens, 384)

    def test_http_5xx_is_counted_once_and_status_and_body_are_relayed(self):
        for status in (200, 400, 429, 500, 502, 503, 599):
            with self.subTest(status=status):
                handler, budget, connection, response = fake_handler(status=status)
                handler.do_POST()
                handler.send_response.assert_called_once_with(status)
                self.assertIn(response.read.return_value, handler.wfile.getvalue())
                self.assertTrue(handler.wfile.getvalue().endswith(b"0\r\n\r\n"))
                self.assertEqual(counter(handler, "engine_errors_total"), int(status >= 500))
                connection.close.assert_called_once()
                self.assertEqual(budget.in_flight(SHORT_CHAT), 0)

    def test_body_failure_is_counted_once_even_after_5xx_headers(self):
        for status in (200, 503):
            with self.subTest(status=status):
                handler, budget, connection, response = fake_handler(status=status)
                response.read.side_effect = http.client.IncompleteRead(b"partial")
                handler.do_POST()
                self.assertEqual(counter(handler, "engine_errors_total"), 1)
                self.assertTrue(handler.close_connection)
                self.assertFalse(handler.wfile.getvalue().endswith(b"0\r\n\r\n"))
                response.close.assert_called_once()
                connection.close.assert_called_once()
                self.assertEqual(budget.in_flight(SHORT_CHAT), 0)

    def test_connection_failure_counts_once_and_releases(self):
        handler, budget, connection, _ = fake_handler()
        connection.getresponse.side_effect = http.client.RemoteDisconnected("no headers")
        handler.do_POST()
        self.assertEqual(handler._json.call_args.args[0], 502)
        self.assertEqual(counter(handler, "engine_errors_total"), 1)
        connection.close.assert_called_once()
        self.assertEqual(budget.in_flight(SHORT_CHAT), 0)

    def test_5xx_is_counted_even_when_client_disconnects_during_headers(self):
        handler, budget, connection, _ = fake_handler(status=503)
        handler.end_headers.side_effect = BrokenPipeError("gone")
        handler.do_POST()
        self.assertEqual(counter(handler, "engine_errors_total"), 1)
        self.assertEqual(counter(handler, "client_disconnects_total"), 1)
        connection.close.assert_called_once()
        self.assertEqual(budget.in_flight(SHORT_CHAT), 0)


class TokenAliasesAndReplicasTest(unittest.TestCase):
    def test_effective_completion_limit_is_never_expanded(self):
        for aliases, expected in (
            ({"max_tokens": 16}, 16),
            ({"max_completion_tokens": 16}, 16),
            ({"max_tokens": 1_000, "max_completion_tokens": 16}, 16),
            ({"max_tokens": 16, "max_completion_tokens": 1_000}, 1_000),
            ({"max_tokens": 16, "max_completion_tokens": None}, 16),
            ({"max_tokens": 128_000, "max_completion_tokens": 128_000}, SHORT_CHAT.max_output_tokens),
            ({}, SHORT_CHAT.max_output_tokens),
        ):
            with self.subTest(aliases=aliases):
                payload = {"prompt": "hi", **aliases}
                handler, budget, connection, _ = fake_handler(payload)
                sink = MemorySink()
                handler.service.recorder = TraceRecorder(sink)
                handler.do_POST()
                forwarded = json.loads(connection.request.call_args.kwargs["body"])
                self.assertEqual(forwarded["max_tokens"], expected)
                if "max_completion_tokens" in aliases:
                    self.assertEqual(forwarded["max_completion_tokens"], expected)
                record = sink.records[-1]
                self.assertEqual(record.granted_max_tokens, expected)
                self.assertEqual(budget.in_flight(SHORT_CHAT), 0)
                self.assertEqual(budget.in_flight(INTERACTIVE), 0)

    def test_either_replica_alias_over_one_is_rejected_before_admission(self):
        for replicas in ({"n": 1, "best_of": 2}, {"n": 2, "best_of": 1}, {"best_of": 2}):
            with self.subTest(replicas=replicas):
                handler, budget, connection, _ = fake_handler({"prompt": "hi", **replicas})
                handler.do_POST()
                self.assertEqual(handler._json.call_args.args[0], 400)
                connection.request.assert_not_called()
                self.assertEqual(counter(handler, "requests_total"), 0)
                self.assertEqual(budget.in_flight(SHORT_CHAT), 0)

    def test_single_replica_aliases_still_pass(self):
        handler, _, connection, _ = fake_handler({"prompt": "hi", "n": 1, "best_of": 1})
        handler.do_POST()
        connection.request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
