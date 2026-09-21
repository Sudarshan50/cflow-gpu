"""Cancellation over actual ephemeral loopback sockets; no inference/services.

Peers observe EOF/reset, admission is released, and readers/guards are checked
for cleanup. A successful write or a mock close call alone is not that proof.
"""

from __future__ import annotations

import http.client
import json
import queue
import select
import socket
import struct
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

from redesign.gateway.backpressure import CircuitBreaker, ClassBudget, StaticHealthSource
from redesign.gateway.cancellation import (
    DEFAULT_TIMEOUT_SECONDS,
    CancelReason,
    RequestCancellation,
    RequestCancelled,
    UpstreamResources,
    current_cancellation,
)
from redesign.gateway.capture import MemorySink, TraceRecorder
from redesign.gateway.classification import INTERACTIVE, SHORT_CHAT, Classifier
from redesign.gateway.clamping import TokenClamp
from redesign.gateway.engine import EngineClient, ProxyResponse, parse_prometheus
from redesign.gateway.metrics import Registry
from redesign.gateway.models import EngineSnapshot
from redesign.gateway.offbox import OffBoxClient
from redesign.gateway.policy import GatewayPolicy
from redesign.gateway.server import GatewayService, Handler, build_service
from redesign.gateway.tokens import HeuristicEstimator


PATH = "/v1/chat/completions"
SSE = b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'


def request_bytes(stream=False):
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": stream}).encode()
    return (
        f"POST {PATH} HTTP/1.1\r\nHost: localhost\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body


def metric(registry, name, **labels):
    prefix = f"k3_gateway_{name}"
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in registry.render().splitlines()
        if (line.startswith(prefix + "{") or line.startswith(prefix + " "))
        and all(f'{key}="{value}"' in line for key, value in labels.items())
    )


def receive_to_eof(sock):
    chunks = []
    while True:
        try:
            chunk = sock.recv(8192)
        except ConnectionResetError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def read_response(file):
    """One shared reader preserves buffered bytes between pipelined responses."""
    status = int(file.readline().split()[1])
    headers = {}
    while (line := file.readline()) != b"\r\n":
        if not line:
            raise AssertionError("EOF in response headers")
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    body = bytearray()
    if headers.get("transfer-encoding") == "chunked":
        while True:
            size = int(file.readline().strip(), 16)
            if not size:
                assert file.readline() == b"\r\n"
                break
            body.extend(file.read(size))
            assert file.read(2) == b"\r\n"
    else:
        body.extend(file.read(int(headers["content-length"])))
    return status, bytes(body)


class _Control:
    def __init__(self, mode, status=200, connection_close=False):
        self.mode = mode
        self.status = status
        self.connection_close = connection_close
        self.received = threading.Event()
        self.parked = threading.Event()
        self.peer_closed = threading.Event()
        self.allow_body = threading.Event()
        self.stop = threading.Event()
        self.requests = []
        self.sockets = []
        self.errors = []
        self.chunks = 0
        self.metrics = b"vllm:kv_cache_usage_perc 0.5\nvllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
        self.metrics_status = 200


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 3

    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.server.control.sockets.append(self.connection)

    def do_GET(self):
        control = self.server.control
        if control.mode == "health_headers":
            control.parked.set()
            self._wait_for_close()
            return
        self.send_response(control.metrics_status)
        self.send_header("Content-Length", str(len(control.metrics)))
        self.end_headers()
        self.wfile.write(control.metrics)

    def do_POST(self):
        control = self.server.control
        control.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        control.received.set()
        mode = control.mode
        if mode == "drop_headers":
            self.close_connection = True
            return
        if mode == "headers":
            control.parked.set()
            self._wait_for_close()
            return

        streaming = mode in ("quiet", "drip_stream", "sse_eof", "flood")
        self.send_response(control.status)
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        if control.connection_close:
            self.send_header("Connection", "close")
        if streaming:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", "1000000" if mode in ("body", "drip_body", "truncated", "reset_body") else "2")
        self.end_headers()

        try:
            if mode == "sse_eof":
                self._chunk(SSE)
                self.wfile.write(b"0\r\n\r\n")
            elif mode == "success":
                self.wfile.write(b"{}")
            elif mode == "gated":
                control.parked.set()
                if control.allow_body.wait(2):
                    self.wfile.write(b"{}")
            elif mode == "truncated":
                self.wfile.write(b"partial")
                self.close_connection = True
            elif mode == "reset_body":
                control.parked.set()
                if control.allow_body.wait(2):
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    self.rfile.close()
                    self.connection.close()
                self.close_connection = True
            elif mode == "flood":
                control.parked.set()
                while not control.stop.is_set():
                    self._chunk(b"x" * 65_536)
                    control.chunks += 1
                self.close_connection = True
            elif mode in ("drip_stream", "drip_body"):
                control.parked.set()
                while not control.stop.is_set():
                    if mode == "drip_stream":
                        self._chunk(SSE)
                    else:
                        self.wfile.write(b"x")
                    control.chunks += 1
                    readable, _, _ = select.select([self.connection], [], [], 0.02)
                    if readable and not self.connection.recv(1):
                        control.peer_closed.set()
                        break
                self.close_connection = True
            else:
                if mode == "body":
                    self.wfile.write(b'{"partial":')
                control.parked.set()
                self._wait_for_close()
        except (BrokenPipeError, ConnectionResetError):
            control.peer_closed.set()
            self.close_connection = True

    def _chunk(self, body):
        self.wfile.write(f"{len(body):X}\r\n".encode() + body + b"\r\n")

    def _wait_for_close(self):
        self.close_connection = True
        try:
            if not self.connection.recv(1):
                self.server.control.peer_closed.set()
        except ConnectionResetError:
            self.server.control.peer_closed.set()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        self.control.errors.append(sys.exc_info()[1])


def serve(handler, control):
    server = _Server(("127.0.0.1", 0), handler)
    server.control = control
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return server, thread


class _RecordingEngine(EngineClient):
    def __init__(self, url, **kwargs):
        super().__init__(url, **kwargs)
        self.connections = []
        self.responses = []
        self.reading = threading.Event()

    def _connect(self, timeout):
        conn = super()._connect(timeout)
        self.connections.append(conn)
        getresponse = conn.getresponse

        def capture():
            sock = conn.sock
            response = getresponse()
            self.responses.append((response, response.fp, sock))
            for name in ("read", "read1"):
                original = getattr(response, name)

                def read(*args, _original=original):
                    self.reading.set()
                    return _original(*args)

                setattr(response, name, read)
            return response

        conn.getresponse = capture
        return conn


class _Policy(GatewayPolicy):
    def __init__(self, budget):
        super().__init__(
            Classifier(), TokenClamp(262_144), budget,
            CircuitBreaker(StaticHealthSource(EngineSnapshot.healthy())),
        )
        self.released = queue.Queue()

    def release(self, decision):
        super().release(decision)
        self.released.put(decision)


class _Gateway:
    def __init__(self, mode="success", timeout=5, status=200, connection_close=False):
        self.control = _Control(mode, status, connection_close)
        self.upstream, self.upstream_thread = serve(_Upstream, self.control)
        self.url = f"http://127.0.0.1:{self.upstream.server_port}"
        self.engine = _RecordingEngine(self.url)
        self.budget = ClassBudget(2, {SHORT_CHAT.name: 0.5, INTERACTIVE.name: 0.5})
        self.policy = _Policy(self.budget)
        self.registry = Registry()
        self.service = GatewayService(
            self.policy, self.engine, HeuristicEstimator(), TraceRecorder(MemorySink()),
            self.registry, send_priority=False, request_timeout=timeout,
        )
        service = self.service

        class GatewayHandler(Handler):
            pass

        GatewayHandler.service = service
        self.server, self.thread = serve(GatewayHandler, self.control)
        self.clients = []

    def __enter__(self):
        return self

    def client(self, stream=False):
        client = socket.create_connection(self.server.server_address, timeout=2)
        self.clients.append(client)
        client.sendall(request_bytes(stream))
        return client

    def connection(self):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=2)
        self.clients.append(conn)
        return conn

    def assert_released(self, test, *, admitted=True, single=True):
        decision = self.policy.released.get(timeout=1)
        test.assertEqual(decision.admitted, admitted)
        if single:
            test.assertTrue(self.policy.released.empty())
        if admitted and single:
            test.assertEqual(self.budget.in_flight(SHORT_CHAT), 0)
        return decision

    def assert_transport_closed(self, test):
        test.assertTrue(self.control.peer_closed.wait(1), "upstream did not observe EOF/reset")
        for conn in self.engine.connections:
            test.assertIsNone(conn.sock)
        for response, file, sock in self.engine.responses:
            test.assertTrue(file.closed, "HTTPResponse buffered reader leaked")
            test.assertIsNone(response.fp)
            test.assertEqual(sock.fileno(), -1, "raw upstream descriptor leaked")

    def __exit__(self, *exc):
        self.control.stop.set()
        self.control.allow_body.set()
        for client in self.clients:
            client.close()
        for sock in self.control.sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for server, thread in ((self.server, self.thread), (self.upstream, self.upstream_thread)):
            server.shutdown()
            server.server_close()
            thread.join(1)


class SocketCancellationTest(unittest.TestCase):
    def assert_cancel_metrics(self, gateway, outcome):
        registry = gateway.registry
        self.assertEqual(metric(registry, "relay_terminal_total"), 1)
        self.assertEqual(metric(registry, "relay_terminal_total", outcome=outcome), 1)
        self.assertEqual(metric(registry, "engine_errors_total"), 0)
        self.assertEqual(metric(registry, "upstream_transport_errors_total"), 0)
        self.assertEqual(metric(registry, "upstream_http_5xx_total"), 0)
        self.assertEqual(metric(registry, "client_disconnects_total"), int(outcome == "client_cancel"))
        self.assertEqual(metric(registry, "request_deadlines_total"), int(outcome == "deadline"))
        self.assertEqual(metric(registry, "cancellation_unsupported_total"), 0)
        self.assertEqual(gateway.control.errors, [])

    def test_disconnect_before_upstream_headers_closes_and_releases_without_fallback(self):
        with _Gateway("headers") as gateway:
            fallback = Mock()
            gateway.service.offbox = fallback
            client = gateway.client()
            self.assertTrue(gateway.control.parked.wait(1))
            started = time.monotonic()
            client.close()
            gateway.assert_released(self)
            gateway.assert_transport_closed(self)
            self.assertLess(time.monotonic() - started, 1)
            fallback.proxy.assert_not_called()
            self.assert_cancel_metrics(gateway, "client_cancel")

    def test_disconnect_during_quiet_sse_and_nonstream_body(self):
        for mode in ("quiet", "body"):
            for connection_close in (False, True):
                with self.subTest(mode=mode, connection_close=connection_close):
                    with _Gateway(mode, connection_close=connection_close) as gateway:
                        client = gateway.client(stream=mode == "quiet")
                        self.assertTrue(gateway.engine.reading.wait(1))
                        client.close()
                        gateway.assert_released(self)
                        gateway.assert_transport_closed(self)
                        self.assert_cancel_metrics(gateway, "client_cancel")

    def test_deadline_interrupts_headers_quiet_sse_and_nonstream_body(self):
        for mode in ("headers", "quiet", "body"):
            with self.subTest(mode=mode), _Gateway(mode, timeout=0.2) as gateway:
                started = time.monotonic()
                client = gateway.client(stream=mode == "quiet")
                self.assertTrue(gateway.control.parked.wait(1))
                data = receive_to_eof(client)
                gateway.assert_released(self)
                gateway.assert_transport_closed(self)
                self.assertLess(time.monotonic() - started, 1)
                self.assertFalse(data.endswith(b"0\r\n\r\n"))
                self.assert_cancel_metrics(gateway, "deadline")

    def test_regular_bytes_do_not_renew_the_total_deadline(self):
        for mode in ("drip_stream", "drip_body"):
            with self.subTest(mode=mode), _Gateway(mode, timeout=0.25) as gateway:
                started = time.monotonic()
                client = gateway.client(stream=mode == "drip_stream")
                data = receive_to_eof(client)
                gateway.assert_released(self)
                gateway.assert_transport_closed(self)
                self.assertGreaterEqual(gateway.control.chunks, 3)
                self.assertLess(time.monotonic() - started, 1)
                self.assertFalse(data.endswith(b"0\r\n\r\n"))
                self.assert_cancel_metrics(gateway, "deadline")

    def test_deadline_also_interrupts_a_downstream_that_stops_reading(self):
        with _Gateway("flood", timeout=0.25) as gateway:
            client = socket.socket()
            gateway.clients.append(client)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            client.settimeout(2)
            client.connect(gateway.server.server_address)
            client.sendall(request_bytes(stream=True))
            self.assertTrue(gateway.control.parked.wait(1))
            # Never read a response byte: TCP flow control eventually blocks
            # wfile.write. The same deadline must wake that write as well.
            gateway.assert_released(self)
            gateway.assert_transport_closed(self)
            self.assertGreater(gateway.control.chunks, 3)
            self.assert_cancel_metrics(gateway, "deadline")

    def test_tcp_write_half_close_is_explicitly_treated_as_cancellation(self):
        with _Gateway("headers") as gateway:
            client = gateway.client()
            self.assertTrue(gateway.control.parked.wait(1))
            client.shutdown(socket.SHUT_WR)
            self.assertEqual(receive_to_eof(client), b"")
            gateway.assert_released(self)
            gateway.assert_transport_closed(self)
            self.assert_cancel_metrics(gateway, "client_cancel")

    @unittest.skipUnless(hasattr(select, "POLLRDHUP"), "FIN behind queued bytes needs POLLRDHUP")
    def test_disconnect_is_detected_behind_unread_pipelined_bytes(self):
        with _Gateway("headers") as gateway:
            client = gateway.client()
            self.assertTrue(gateway.control.parked.wait(1))
            client.sendall(request_bytes())
            client.close()
            gateway.assert_released(self)
            gateway.assert_transport_closed(self)
            self.assertEqual(len(gateway.control.requests), 1)
            self.assert_cancel_metrics(gateway, "client_cancel")

    def test_pipelined_request_is_not_consumed_and_readability_does_not_spin(self):
        guards = []

        def guard_factory(*args, **kwargs):
            guard = RequestCancellation(*args, **kwargs)
            guard._disconnected = Mock(wraps=guard._disconnected)
            guards.append(guard)
            return guard

        with patch("redesign.gateway.server.RequestCancellation", side_effect=guard_factory):
            with _Gateway("gated") as gateway:
                client = gateway.client()
                self.assertTrue(gateway.control.parked.wait(1))
                # Sent only after rfile finished the first request, so these
                # bytes remain in the kernel socket for the watcher to see.
                client.sendall(request_bytes())
                time.sleep(0.22)
                self.assertIsNone(guards[0].reason)
                self.assertLess(guards[0]._disconnected.call_count, 15)
                gateway.control.allow_body.set()
                with client.makefile("rb") as file:
                    self.assertEqual(read_response(file), (200, b"{}"))
                    gateway.assert_released(self, single=False)
                    self.assertEqual(read_response(file), (200, b"{}"))
                    gateway.assert_released(self)
                self.assertEqual(len(gateway.control.requests), 2)
                self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="relayed"), 2)
                self.assertTrue(all(not guard._thread.is_alive() for guard in guards))
                self.assertEqual(gateway.control.errors, [])

    def test_successful_keepalive_survives_the_previous_requests_deadline(self):
        with _Gateway(timeout=0.15) as gateway:
            conn = gateway.connection()
            for attempt in range(2):
                conn.request("POST", PATH, b'{"prompt":"hi"}', {"Content-Type": "application/json"})
                response = conn.getresponse()
                self.assertEqual((response.status, response.read()), (200, b"{}"))
                gateway.assert_released(self)
                if attempt == 0:
                    original_socket = conn.sock
                    time.sleep(0.22)
                self.assertIs(conn.sock, original_socket)
            self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="relayed"), 2)
            self.assertEqual(metric(gateway.registry, "client_disconnects_total"), 0)
            self.assertEqual(metric(gateway.registry, "request_deadlines_total"), 0)
            self.assertEqual(gateway.control.errors, [])

    def test_offbox_wrapper_inherits_cancellation_context_during_fallback(self):
        with _Gateway("headers") as gateway:
            gateway.service.engine = Mock(
                supports_request_cancellation=True,
                proxy=Mock(side_effect=OSError("local down")),
            )
            gateway.service.offbox = OffBoxClient(gateway.url, model="remote-test")
            client = gateway.client()
            self.assertTrue(gateway.control.parked.wait(1))
            client.close()
            gateway.assert_released(self)
            self.assertTrue(gateway.control.peer_closed.wait(1))
            self.assertEqual(gateway.control.requests[0]["model"], "remote-test")
            self.assert_cancel_metrics(gateway, "client_cancel")


class TerminalAndRecorderTest(unittest.TestCase):
    def test_upstream_reset_during_body_is_not_a_downstream_disconnect(self):
        for stream in (False, True):
            with self.subTest(stream=stream), _Gateway("reset_body") as gateway:
                conn = gateway.connection()
                conn.request("POST", PATH, json.dumps({"prompt": "hi", "stream": stream}))
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                self.assertTrue(gateway.engine.reading.wait(1))
                gateway.control.allow_body.set()
                with self.assertRaises(http.client.IncompleteRead):
                    response.read()
                gateway.assert_released(self)
                self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="upstream_error"), 1)
                self.assertEqual(metric(gateway.registry, "upstream_transport_errors_total"), 1)
                self.assertEqual(metric(gateway.registry, "client_disconnects_total"), 0)
                self.assertEqual(gateway.control.errors, [])

    def test_upstream_http_5xx_and_transport_failures_are_not_client_cancellation(self):
        for mode, status, terminal in (
            ("success", 503, "upstream_http_5xx"),
            ("drop_headers", 200, "upstream_error"),
            ("truncated", 200, "upstream_error"),
            ("truncated", 503, "upstream_error"),
        ):
            for stream in (False, True):
                with self.subTest(mode=mode, status=status, stream=stream):
                    with _Gateway(mode, status=status) as gateway:
                        conn = gateway.connection()
                        conn.request("POST", PATH, json.dumps({"prompt": "hi", "stream": stream}))
                        response = conn.getresponse()
                        self.assertEqual(response.status, 502 if mode == "drop_headers" else status)
                        if mode == "truncated":
                            with self.assertRaises(http.client.IncompleteRead):
                                response.read()
                        else:
                            response.read()
                        gateway.assert_released(self)
                        self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome=terminal), 1)
                        self.assertEqual(metric(gateway.registry, "engine_errors_total"), 1)
                        self.assertEqual(metric(gateway.registry, "client_disconnects_total"), 0)
                        self.assertEqual(metric(gateway.registry, "request_deadlines_total"), 0)
                        self.assertEqual(metric(gateway.registry, "upstream_http_5xx_total"), int(status == 503))
                        self.assertEqual(metric(gateway.registry, "upstream_transport_errors_total"), int(mode != "success"))
                        self.assertEqual(gateway.control.errors, [])

    def test_http_200_without_sse_done_is_only_transport_relayed(self):
        with _Gateway("sse_eof") as gateway:
            conn = gateway.connection()
            conn.request("POST", PATH, b'{"prompt":"hi","stream":true}')
            response = conn.getresponse()
            self.assertEqual(response.read(), SSE)
            gateway.assert_released(self)
            self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="relayed", upstream_status="200"), 1)
            self.assertNotIn("k3_gateway_protocol_completions_total", gateway.registry.render())
            self.assertEqual(metric(gateway.registry, "protocol_incomplete_total"), 1)

    def test_recorder_oserror_is_nonfatal_counted_and_releases_only_its_lease(self):
        with _Gateway() as gateway:
            self.assertTrue(gateway.budget.try_acquire(INTERACTIVE))
            gateway.service.recorder.record = Mock(side_effect=OSError("disk full"))
            conn = gateway.connection()
            conn.request("POST", PATH, b'{"prompt":"hi"}')
            response = conn.getresponse()
            self.assertEqual((response.status, response.read()), (200, b"{}"))
            gateway.assert_released(self)
            self.assertEqual(gateway.budget.in_flight(INTERACTIVE), 1)
            self.assertEqual(metric(gateway.registry, "recorder_errors_total"), 1)
            self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="relayed"), 1)
            self.assertEqual(gateway.control.errors, [])

    def test_recorder_oserror_on_rejection_does_not_free_another_requests_slot(self):
        with _Gateway() as gateway:
            self.assertTrue(gateway.budget.try_acquire(SHORT_CHAT))
            gateway.service.recorder.record = Mock(side_effect=OSError("disk full"))
            conn = gateway.connection()
            conn.request("POST", PATH, b'{"prompt":"hi"}')
            response = conn.getresponse()
            self.assertEqual(response.status, 429)
            response.read()
            gateway.assert_released(self, admitted=False)
            self.assertEqual(gateway.budget.in_flight(SHORT_CHAT), 1)
            self.assertEqual(metric(gateway.registry, "recorder_errors_total"), 1)
            self.assertEqual(metric(gateway.registry, "relay_terminal_total"), 0)
            self.assertEqual(gateway.control.requests, [])

    def test_legacy_proxy_signature_works_and_unsupported_cancellation_is_visible(self):
        class LegacyEngine:
            calls = 0

            def proxy(self, path, payload, stream):
                self.calls += 1
                return ProxyResponse(200, [], iter([b"legacy"]))

        with _Gateway() as gateway:
            legacy = LegacyEngine()
            gateway.service.engine = legacy
            conn = gateway.connection()
            conn.request("POST", PATH, b'{"prompt":"hi"}')
            response = conn.getresponse()
            self.assertEqual((response.status, response.read()), (200, b"legacy"))
            gateway.assert_released(self)
            self.assertEqual(legacy.calls, 1)
            self.assertEqual(metric(gateway.registry, "cancellation_unsupported_total", upstream="engine"), 1)
            self.assertEqual(gateway.control.errors, [])

    def test_legacy_proxy_typeerror_is_never_retried_as_a_signature_probe(self):
        class LegacyEngine:
            calls = 0

            def proxy(self, path, payload, stream):
                self.calls += 1
                raise TypeError("internal proxy bug")

        with _Gateway() as gateway:
            legacy = LegacyEngine()
            gateway.service.engine = legacy
            client = gateway.client()
            self.assertEqual(receive_to_eof(client), b"")
            gateway.assert_released(self)
            self.assertEqual(legacy.calls, 1)
            self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="internal_error"), 1)
            self.assertEqual(len(gateway.control.errors), 1)
            self.assertIsInstance(gateway.control.errors[0], TypeError)

    def test_blocking_legacy_proxy_cannot_claim_prompt_admission_release(self):
        entered, unblock, closed = threading.Event(), threading.Event(), threading.Event()

        class LegacyEngine:
            def proxy(self, path, payload, stream):
                entered.set()
                unblock.wait(2)
                return ProxyResponse(200, [], iter([b"legacy"]), closed.set)

        with _Gateway(timeout=0.15) as gateway:
            gateway.service.engine = LegacyEngine()
            try:
                client = gateway.client()
                self.assertTrue(entered.wait(1))
                self.assertEqual(receive_to_eof(client), b"")
                self.assertEqual(metric(gateway.registry, "cancellation_unsupported_total"), 1)
                self.assertEqual(gateway.budget.in_flight(SHORT_CHAT), 1)
                self.assertTrue(gateway.policy.released.empty())
            finally:
                unblock.set()
            gateway.assert_released(self)
            self.assertTrue(closed.is_set())
            self.assertEqual(metric(gateway.registry, "relay_terminal_total", outcome="deadline"), 1)


@contextmanager
def tcp_pair():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with socket.create_connection(listener.getsockname(), timeout=2) as client:
            peer, _ = listener.accept()
            with peer:
                peer.settimeout(2)
                yield client, peer


class OwnershipRaceTest(unittest.TestCase):
    def test_concurrent_close_interrupts_a_buffered_read_before_taking_its_lock(self):
        for stream in (False, True):
            with self.subTest(stream=stream), _Gateway("quiet" if stream else "body", connection_close=True) as gateway:
                response = gateway.engine.proxy(PATH, {}, stream)
                underlying, file, sock = gateway.engine.responses[0]
                original_close = underlying.close
                underlying.close = Mock(wraps=original_close)
                errors = []

                def read():
                    try:
                        list(response.body)
                    except (OSError, http.client.HTTPException):
                        pass
                    except BaseException as exc:
                        errors.append(exc)

                reader = threading.Thread(target=read, daemon=True)
                reader.start()
                self.assertTrue(gateway.engine.reading.wait(1))
                closers = [threading.Thread(target=response.close, daemon=True) for _ in range(4)]
                for closer in closers:
                    closer.start()
                for thread in (*closers, reader):
                    thread.join(1)
                    self.assertFalse(thread.is_alive(), "buffered close/read deadlocked")
                self.assertEqual(errors, [])
                underlying.close.assert_called_once()
                self.assertTrue(file.closed)
                self.assertEqual(sock.fileno(), -1)
                gateway.assert_transport_closed(self)

    def test_socket_published_after_cancellation_is_immediately_closed(self):
        with RequestCancellation(timeout=2) as cancellation:
            connection = http.client.HTTPConnection("127.0.0.1", 1)
            resources = UpstreamResources(connection, cancellation, 2)
            cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
            with tcp_pair() as (sock, peer):
                with self.assertRaises(RequestCancelled):
                    resources.publish_socket(sock)
                self.assertEqual(sock.fileno(), -1)
                self.assertEqual(peer.recv(1), b"")
            resources.close()
            resources.close()

    def test_cancel_between_socket_connect_and_httpconnection_assignment(self):
        connected, resume = threading.Event(), threading.Event()
        guards, sockets, errors = [], [], []
        original = UpstreamResources._create_connection

        def delayed(resources, *args):
            sock = original(resources, *args)
            sockets.append(sock)
            connected.set()
            resume.wait(2)
            return sock

        with _Gateway("headers") as gateway:
            def run():
                try:
                    with RequestCancellation(timeout=2) as cancellation:
                        guards.append(cancellation)
                        gateway.engine.proxy(PATH, {}, False)
                except BaseException as exc:
                    errors.append(exc)

            with patch.object(UpstreamResources, "_create_connection", delayed):
                worker = threading.Thread(target=run, daemon=True)
                worker.start()
                try:
                    self.assertTrue(connected.wait(1))
                    self.assertIsNone(gateway.engine.connections[0].sock)
                    guards[0].cancel(CancelReason.CLIENT_DISCONNECT)
                finally:
                    resume.set()
                worker.join(1)
                self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RequestCancelled)
            self.assertEqual(sockets[0].fileno(), -1)
            self.assertIsNone(gateway.engine.connections[0].sock)
            self.assertEqual(gateway.control.requests, [])
            self.assertFalse(guards[0]._thread.is_alive())

    def test_standalone_engine_deadline_closes_an_unstarted_response(self):
        with _Gateway("quiet", connection_close=True) as gateway:
            gateway.engine._timeout = 0.15
            response = gateway.engine.proxy(PATH, {}, True)
            self.assertTrue(gateway.control.peer_closed.wait(1))
            # Peer EOF can precede closing the buffered reader by a few lines.
            deadline = time.monotonic() + 1
            _, file, sock = gateway.engine.responses[0]
            while (not file.closed or sock.fileno() != -1) and time.monotonic() < deadline:
                time.sleep(0.005)
            gateway.assert_transport_closed(self)
            with self.assertRaises(RequestCancelled) as caught:
                next(response.body)
            self.assertIs(caught.exception.reason, CancelReason.DEADLINE)
            response.close()
            response.close()

    def test_a_slow_os_dns_lookup_does_not_hold_the_request_past_its_deadline(self):
        unblock, entered, returned = threading.Event(), threading.Event(), threading.Event()

        def resolve(*args):
            entered.set()
            unblock.wait(2)
            returned.set()
            return []

        engine = EngineClient("http://dns-test.invalid", timeout=0.1)
        started = time.monotonic()
        with patch("redesign.gateway.cancellation.socket.getaddrinfo", side_effect=resolve):
            try:
                with self.assertRaises(RequestCancelled) as caught:
                    engine.proxy(PATH, {}, False)
                self.assertIs(caught.exception.reason, CancelReason.DEADLINE)
                self.assertTrue(entered.is_set())
                self.assertFalse(returned.is_set())
                self.assertLess(time.monotonic() - started, 1)
            finally:
                unblock.set()
                self.assertTrue(returned.wait(1))

    def test_connect_can_try_a_second_address_without_closing_the_winner(self):
        with _Gateway() as gateway, socket.socket() as refused:
            # Bound but not listening guarantees a loopback refusal without
            # assuming that any fixed port on the host is unused.
            refused.bind(("127.0.0.1", 0))
            addresses = [
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", address)
                for address in (refused.getsockname(), gateway.upstream.server_address)
            ]
            with patch.object(UpstreamResources, "_resolve", return_value=addresses):
                response = gateway.engine.proxy(PATH, {}, False)
                self.assertEqual(b"".join(response.body), b"{}")
                response.close()
            for _, file, sock in gateway.engine.responses:
                self.assertTrue(file.closed)
                self.assertEqual(sock.fileno(), -1)

    def test_context_is_removed_after_success_and_after_exceptions(self):
        self.assertIsNone(current_cancellation())
        with self.assertRaisesRegex(RuntimeError, "test"):
            with RequestCancellation(timeout=1) as cancellation:
                self.assertIs(current_cancellation(), cancellation)
                raise RuntimeError("test")
        self.assertIsNone(current_cancellation())
        self.assertFalse(cancellation._thread.is_alive())


class SnapshotAndGaugeTest(unittest.TestCase):
    HEALTH = (
        'vllm:kv_cache_usage_perc{engine="0",model_name="model with spaces"} 0.6\n'
        'vllm:num_requests_running{engine="0"} 7\n'
        'vllm:num_requests_waiting{engine="0"} 0\n'
        'vllm:num_preemptions_total{engine="0"} 0\n'
    )

    def snapshot(self, text):
        engine = EngineClient("http://offline.invalid")
        engine.metrics_text = Mock(return_value=text)
        return engine.snapshot()

    def test_current_info_token_capacity_precedes_legacy_blocks(self):
        snapshot = self.snapshot(self.HEALTH + (
            'vllm:cache_config_info{engine="0",kv_cache_size_tokens="2197339",'
            'num_gpu_blocks="2917",block_size="768",kv_cache_memory_bytes="None"} 1.0\n'
        ))
        self.assertTrue(snapshot.known)
        self.assertEqual((snapshot.kv_usage, snapshot.running, snapshot.waiting), (0.6, 7, 0))
        self.assertEqual(snapshot.kv_capacity_tokens, 2_197_339)

    def test_legacy_capacity_fallback_is_not_summed_across_info_labels(self):
        text = self.HEALTH + "".join(
            f'vllm:cache_config_info{{engine="{engine}",num_gpu_blocks="100",block_size="16"}} 1\n'
            for engine in (0, 1)
        )
        self.assertEqual(self.snapshot(text).kv_capacity_tokens, 1600)

    def test_unknown_health_and_missing_capacity_are_not_known_zero(self):
        for text in (
            "", "unrelated_metric 0\n", "vllm:kv_cache_usage_perc 0\n",
            self.HEALTH.replace('vllm:num_requests_running{engine="0"} 7\n', ""),
            self.HEALTH.replace('vllm:num_requests_waiting{engine="0"} 0\n', ""),
            self.HEALTH.replace('vllm:num_preemptions_total{engine="0"} 0\n', ""),
            self.HEALTH.replace("0.6", "NaN"),
            self.HEALTH.replace("0.6", "+Inf"),
        ):
            with self.subTest(text=text):
                self.assertFalse(self.snapshot(text).known)
        healthy = self.snapshot(self.HEALTH)
        self.assertTrue(healthy.known)
        self.assertIsNone(healthy.kv_capacity_tokens)
        for labels in ('num_gpu_blocks="None",block_size="16"', 'kv_cache_size_tokens="NaN"'):
            self.assertIsNone(self.snapshot(self.HEALTH + f"vllm:cache_config_info{{{labels}}} 1\n").kv_capacity_tokens)

    def test_failed_metrics_http_status_is_not_a_zero_health_sample(self):
        with _Gateway() as gateway:
            gateway.control.metrics_status = 503
            with self.assertRaisesRegex(http.client.HTTPException, "metrics HTTP 503"):
                gateway.engine.snapshot()

    def test_first_positive_preemption_counter_requires_a_rate_baseline(self):
        engine = EngineClient("http://offline.invalid", snapshot_ttl=0)
        engine.metrics_text = Mock(return_value=self.HEALTH.replace(
            'vllm:num_preemptions_total{engine="0"} 0',
            'vllm:num_preemptions_total{engine="0"} 10',
        ))
        with patch("redesign.gateway.engine.time.monotonic", return_value=100) as clock:
            self.assertFalse(engine.snapshot().known)
            clock.return_value = 160
            snapshot = engine.snapshot()
        self.assertTrue(snapshot.known)
        self.assertEqual(snapshot.preemptions_per_minute, 0)

    def test_slow_health_sample_is_unknown_for_borrowing(self):
        engine = EngineClient("http://offline.invalid")
        with patch("redesign.gateway.engine.time.monotonic", return_value=100) as clock:
            def slow_metrics():
                clock.return_value = 103
                return self.HEALTH
            engine.metrics_text = slow_metrics
            self.assertFalse(engine.snapshot().known)

    def test_stalled_standalone_health_request_returns_unhealthy_at_its_deadline(self):
        with _Gateway("health_headers") as gateway:
            with patch("redesign.gateway.engine.HEALTH_TIMEOUT_SECONDS", 0.1):
                self.assertFalse(gateway.engine.healthy())
            gateway.assert_transport_closed(self)

    def test_parser_sums_labels_with_spaces_and_ignores_nonfinite_samples(self):
        values = parse_prometheus('m{model="a b"} 2 123\nm{model="c"} 3\nn NaN\n')
        self.assertEqual(values, {"m": 5.0})

    def test_gauges_replace_values_and_render_zero_without_changing_counters(self):
        registry = Registry()
        registry.increment("requests_total", 2)
        registry.set_gauge("admission_in_flight", 4, traffic_class=SHORT_CHAT.name)
        registry.set_gauge("admission_in_flight", 0, traffic_class=SHORT_CHAT.name)
        registry.set_gauge("admission_in_flight", 1, traffic_class=INTERACTIVE.name)
        registry.set_gauge("admission_ceiling", 64)
        self.assertEqual(metric(registry, "admission_in_flight", traffic_class=SHORT_CHAT.name), 0)
        self.assertEqual(metric(registry, "admission_in_flight"), 1)
        self.assertEqual(metric(registry, "admission_ceiling"), 64)
        self.assertEqual(metric(registry, "requests_total"), 2)
        self.assertEqual(registry.render().count("# TYPE k3_gateway_admission_in_flight gauge"), 1)

    def test_default_deadline_and_build_service_configuration(self):
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 600)
        service = build_service(
            "http://127.0.0.1:1", 262_144, 96, None, None, False,
            request_timeout=0.125,
        )
        self.assertEqual(service.request_timeout, 0.125)
        self.assertEqual(service.engine._timeout, 0.125)
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                RequestCancellation(timeout=timeout)


if __name__ == "__main__":
    unittest.main()
