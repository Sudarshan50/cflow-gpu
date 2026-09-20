"""HTTP shell for the gateway. Register items D2, B2, E2 and Z3 capture.

    python3 -m redesign.gateway.server --port 8002

Stdlib only, threaded. At the observed request rate (~0.2 req/s, tens of
concurrent streams) a thread per request is comfortable, and it keeps the branch
dependency-free and runnable on a laptop. The policy core is transport-agnostic,
so if SSE relay throughput becomes the bottleneck only this module changes --
that is the tripwire recorded in SYSTEM-DESIGN.md D3.

This layer owns tenancy. It never models KV; it reads what the engine publishes
and refuses classes when the engine says it is in trouble.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .backpressure import CircuitBreaker, ClassBudget
from .capture import JsonlSink, MemorySink, TraceRecorder
from .classification import ALL_CLASSES, Classifier
from .clamping import TokenClamp
from .engine import EngineClient
from .metrics import Registry
from .models import Outcome, RequestEnvelope
from .policy import GatewayPolicy
from .tokens import TokenEstimator, build_estimator

PROXIED_PATHS = ("/v1/chat/completions", "/v1/completions")
MAX_BODY_BYTES = 64 * 1024 * 1024

# Distinguishes "parse failed, error already sent" from a literal null body.
_INVALID = object()

BATCH_HEADER = "x-k3-batch"
CUSTOMER_HEADER = "x-k3-customer"


class GatewayService:
    """Everything a request handler needs, assembled once at startup."""

    def __init__(
        self,
        policy: GatewayPolicy,
        engine: EngineClient,
        estimator: TokenEstimator,
        recorder: TraceRecorder,
        registry: Registry,
        send_priority: bool,
    ) -> None:
        self.policy = policy
        self.engine = engine
        self.estimator = estimator
        self.recorder = recorder
        self.registry = registry
        self.send_priority = send_priority

    def envelope(self, payload: dict, headers, path: str) -> RequestEnvelope:
        return RequestEnvelope(
            customer=headers.get(CUSTOMER_HEADER, "anonymous"),
            prompt_tokens=self.estimator.estimate(payload),
            requested_max_tokens=_coerce_int(payload.get("max_tokens")),
            streaming=bool(payload.get("stream")),
            has_tools=bool(payload.get("tools") or payload.get("functions")),
            batch_hint=headers.get(BATCH_HEADER, "").lower() in ("1", "true", "yes"),
            path=path,
        )


def _coerce_int(value) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Without this a client that declares a body and never sends it pins a
    # thread forever.
    timeout = 30
    service: GatewayService

    def log_message(self, fmt: str, *args) -> None:
        """Silenced: usage attribution is the trace sink's job, not stderr's."""

    def do_GET(self) -> None:
        if self.path == "/health":
            healthy = self.service.engine.healthy()
            self._json(200 if healthy else 503, {"engine": "up" if healthy else "down"})
        elif self.path == "/metrics":
            self._raw(200, self.service.registry.render().encode(), "text/plain")
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request"}})

    def do_POST(self) -> None:
        if self.path not in PROXIED_PATHS:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request"}})
            return

        payload = self._read_payload()
        if payload is _INVALID:
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": {
                "message": "body must be a JSON object",
                "type": "invalid_request_error",
            }})
            return

        started = time.monotonic()
        envelope = self.service.envelope(payload, self.headers, self.path)
        decision = self.service.policy.decide(envelope)

        self.service.registry.increment(
            "requests_total",
            traffic_class=decision.traffic_class,
            outcome=decision.outcome.name.lower(),
        )
        if decision.rescued_from_rejection:
            self.service.registry.increment(
                "rescued_from_rejection_total", traffic_class=decision.traffic_class
            )

        self.service.recorder.record(decision)

        if not decision.admitted:
            self._refuse(decision)
            return

        payload["max_tokens"] = decision.clamp.granted
        if self.service.send_priority:
            payload["priority"] = int(decision.priority)

        try:
            self._relay(payload, decision, started)
        finally:
            self.service.policy.release(decision)

    def _relay(self, payload: dict, decision, started: float) -> None:
        try:
            response = self.service.engine.proxy(
                self.path, payload, stream=decision.envelope.streaming
            )
        except (OSError, http.client.HTTPException) as exc:
            self.service.registry.increment(
                "engine_errors_total", traffic_class=decision.traffic_class
            )
            self._json(502, {"error": {
                "message": f"engine unreachable: {exc}",
                "type": "upstream_error",
            }})
            return

        self.send_response(response.status)
        for key, value in response.headers:
            self.send_header(key, value)
        self.send_header("x-k3-class", decision.traffic_class)
        self.send_header("x-k3-max-tokens-granted", str(decision.clamp.granted))
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        first_byte: float | None = None
        try:
            for chunk in response.body:
                if not chunk:
                    continue
                if first_byte is None:
                    first_byte = time.monotonic()
                    self.service.registry.observe_ttft(
                        decision.traffic_class, first_byte - started
                    )
                self.wfile.write(f"{len(chunk):X}\r\n".encode())
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            # Client hung up. Abandon the upstream response rather than draining
            # it -- the slot is released by do_POST's finally.
            self.service.registry.increment(
                "client_disconnects_total", traffic_class=decision.traffic_class
            )
            self.close_connection = True
            return
        except (OSError, http.client.HTTPException) as exc:
            # The stream died after headers were sent, so the status line is
            # already 200. Omit the terminating chunk and drop the connection:
            # a truncated body must not be framed as a complete one.
            self.service.registry.increment(
                "engine_errors_total", traffic_class=decision.traffic_class
            )
            self.log_error("upstream stream failed: %s", exc)
            self.close_connection = True
            return
        finally:
            response.close()

        self.service.registry.observe_total(
            decision.traffic_class, time.monotonic() - started
        )

    def _refuse(self, decision) -> None:
        status = 503 if decision.outcome is Outcome.REJECT_SHED else 429
        if decision.clamp.granted == 0:
            status = 400
        headers = {}
        if decision.retry_after_seconds:
            headers["Retry-After"] = str(decision.retry_after_seconds)
        self._json(
            status,
            {"error": {
                "message": decision.reason,
                "type": "rate_limit_error" if status != 400 else "invalid_request_error",
                "k3_class": decision.traffic_class,
            }},
            extra_headers=headers,
        )

    def _read_payload(self):
        """Returns the parsed body, or _INVALID having already sent an error.

        A sentinel rather than None: a literal `null` body parses to None, and
        conflating the two made do_POST return without writing any response,
        hanging the client until its own timeout.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": {
                "message": "invalid Content-Length", "type": "invalid_request_error",
            }})
            return _INVALID
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(400, {"error": {
                "message": "missing or oversized body", "type": "invalid_request_error",
            }})
            return _INVALID
        try:
            return json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"error": {
                "message": f"invalid JSON: {exc}", "type": "invalid_request_error",
            }})
            return _INVALID

    def _json(self, status: int, body: dict, extra_headers: dict | None = None) -> None:
        self._raw(status, json.dumps(body).encode(), "application/json", extra_headers)

    def _raw(self, status, body: bytes, content_type: str, extra_headers=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def build_service(
    engine_url: str,
    max_model_len: int,
    concurrency_ceiling: int,
    trace_path: str | None,
    model_path: str | None,
    send_priority: bool,
    offbox_configured: bool = False,
) -> GatewayService:
    engine = EngineClient(engine_url)
    sink = JsonlSink(Path(trace_path)) if trace_path else MemorySink()

    policy = GatewayPolicy(
        classifier=Classifier(),
        clamp=TokenClamp(max_model_len=max_model_len),
        budget=ClassBudget.from_classes(
            concurrency_ceiling, ALL_CLASSES, offbox_configured=offbox_configured
        ),
        breaker=CircuitBreaker(engine),
        offbox_configured=offbox_configured,
    )
    return GatewayService(
        policy=policy,
        engine=engine,
        estimator=build_estimator(model_path),
        recorder=TraceRecorder(sink),
        registry=Registry(),
        send_priority=send_priority,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.gateway.server", description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--engine-url", default=os.environ.get(
        "K3_ENGINE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get(
        "K3_MAX_MODEL_LEN", 262_144)))
    parser.add_argument("--ceiling", type=int, default=int(os.environ.get(
        "K3_ADMISSION_CEILING", 75)))
    parser.add_argument("--trace-path", default=os.environ.get("K3_TRACE_PATH"))
    parser.add_argument("--model-path", default=os.environ.get("K3_MODEL_PATH"))
    parser.add_argument("--send-priority", action="store_true",
                        default=os.environ.get("K3_SEND_PRIORITY", "") == "1",
                        help="attach a priority field to every upstream request; "
                             "only useful if the engine honours priority scheduling")
    args = parser.parse_args(argv)

    Handler.service = build_service(
        engine_url=args.engine_url,
        max_model_len=args.max_model_len,
        concurrency_ceiling=args.ceiling,
        trace_path=args.trace_path,
        model_path=args.model_path,
        send_priority=args.send_priority,
    )

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True

    print(f"gateway on {args.bind}:{args.port} -> {args.engine_url}", file=sys.stderr)
    print(f"  max_model_len={args.max_model_len} ceiling={args.ceiling} "
          f"priority={'on' if args.send_priority else 'off'}", file=sys.stderr)
    print(f"  traces -> {args.trace_path or '(memory only)'}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
