"""HTTP front door: classify, clamp, admit, then relay.

    python3 -m redesign.gateway.server --port 8002
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import time
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .admission import AdmissionController, AdmissionLimits
from .backpressure import CircuitBreaker, ClassBudget
from .capture import JsonlSink, MemorySink, TraceRecorder
from .capacity import AdaptiveCapacityController, CapacityLimits
from .classification import ALL_CLASSES, Classifier
from .clamping import TokenClamp
from .engine import EngineClient
from .inspection import Inspection, InspectionError, PromptInspector, UsageObserver
from .media import normalize_payload
from .metrics import Registry
from .models import Outcome, RequestEnvelope
from .offbox import OffBoxClient, should_fallback
from .policy import GatewayPolicy
from .tokens import TokenEstimator, build_estimator, has_images
from .workload import WorkloadBudget, WorkloadLimits

PROXIED_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/chat/completions",
    "/completions",
)
MAX_BODY_BYTES = 64 * 1024 * 1024

# Distinguishes "parse failed, error already sent" from a literal null body.
_INVALID = object()

BATCH_HEADER = "x-k3-batch"
CUSTOMER_HEADER = "x-k3-customer"
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def _coerce_int(value) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def batch_authorized(headers, peer: str, allowlist: frozenset[str]) -> bool:
    # Public clients cannot self-declare batch; loopback and the allowlist can.
    hinted = headers.get(BATCH_HEADER, "").lower() in ("1", "true", "yes")
    if not hinted:
        return False
    if peer in _LOOPBACK:
        return True
    return headers.get(CUSTOMER_HEADER, "") in allowlist


def overcommitted_replicas(payload: dict) -> bool:
    return any(payload.get(key) is not None and
               (type(payload[key]) is not int or payload[key] != 1)
               for key in ("n", "best_of"))


def apply_granted_tokens(payload: dict, granted: int) -> None:
    payload["max_tokens"] = granted
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = granted


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
        offbox: OffBoxClient | None = None,
        batch_customers: frozenset[str] = frozenset(),
        inspector: PromptInspector | None = None,
        max_model_len: int = 262_144,
        admission_limits: AdmissionLimits | None = None,
        capacity_controller: AdaptiveCapacityController | None = None,
    ) -> None:
        self.policy = policy
        self.engine = engine
        self.estimator = estimator
        self.recorder = recorder
        self.registry = registry
        self.send_priority = send_priority
        self.offbox = offbox
        self.batch_customers = batch_customers
        self.inspector = inspector
        self.max_model_len = max_model_len
        self.admission = AdmissionController(policy, admission_limits or AdmissionLimits())
        self.capacity_controller = capacity_controller

    def close(self) -> None:
        if self.capacity_controller is not None:
            self.capacity_controller.close()

    def envelope(
        self, payload: dict, headers, path: str, peer: str = "", inspection: Inspection | None = None
    ) -> RequestEnvelope:
        # vLLM gives the newer alias precedence when both are supplied.
        requested = _coerce_int(payload.get("max_completion_tokens"))
        if requested is None:
            requested = _coerce_int(payload.get("max_tokens"))
        return RequestEnvelope(
            customer=headers.get(CUSTOMER_HEADER, "anonymous"),
            prompt_tokens=inspection.tokens if inspection is not None else self.estimator.estimate(payload),
            requested_max_tokens=requested,
            streaming=bool(payload.get("stream")),
            has_tools=bool(payload.get("tools") or payload.get("functions")),
            has_images=has_images(payload),
            batch_hint=batch_authorized(headers, peer, self.batch_customers),
            path=path,
        )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Without this a client that declares a body and never sends it pins a
    # thread forever.
    timeout = 30
    service: GatewayService

    def log_message(self, fmt: str, *args) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            healthy = self.service.engine.healthy()
            self._json(200 if healthy else 503, {"engine": "up" if healthy else "down"})
        elif self.path == "/metrics":
            text = self.service.registry.render()
            budget = self.service.policy.workload_budget
            if budget is not None:
                text += "".join(f"k3_gateway_workload_{key} {value}\n" for key, value in budget.state().items())
            text += "".join(f"k3_gateway_admission_{key} {value}\n" for key, value in self.service.admission.state().items())
            controller = self.service.capacity_controller
            if controller is not None:
                state = controller.state()
                state_number = {"cold": 0, "green": 1, "warm": 2, "pressure": 3, "stale": 4}
                text += f"k3_gateway_capacity_state {state_number[state['state']]}\n"
                for key in ("borrow_limit", "snapshot_age_seconds", "samples", "errors"):
                    value = state[key]
                    if value is not None:
                        text += f"k3_gateway_capacity_{key} {value}\n"
            if self.service.inspector is not None:
                text += f"k3_gateway_tokenizer_quarantined {int(self.service.inspector.quarantined)}\n"
            self._raw(200, text.encode(), "text/plain")
        elif self.path == "/diagnostics/admission":
            if self.client_address[0] not in _LOOPBACK:
                self._json(403, {"error": {"message": "local diagnostics only"}})
            else:
                self._json(200, self.service.admission.recent())
        elif self.path == "/diagnostics/prefix" and self.service.inspector is not None:
            if self.client_address[0] not in _LOOPBACK:
                self._json(403, {"error": {"message": "local diagnostics only"}})
            else:
                self._json(200, self.service.inspector.recent())
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

        normalize_payload(payload)
        if overcommitted_replicas(payload):
            self._json(400, {"error": {
                "message": "n/best_of must be the integer 1 on this replica",
                "type": "invalid_request_error",
            }})
            return

        started = time.monotonic()
        peer = self.client_address[0] if self.client_address else ""
        inspection = None
        reserved_prompt = None
        if self.service.inspector is not None:
            try:
                try:
                    epoch = self.service.engine.snapshot().cache_epoch
                except Exception:
                    epoch = None
                inspection = self.service.inspector.inspect(payload, epoch)
            except InspectionError as exc:
                self.service.registry.increment("inspection_failures_total", reason=exc.reason)
                self._json(exc.status, {"error": {"message": exc.public_message, "type": "invalid_request_error"
                           if exc.status == 400 else "service_unavailable"}},
                           extra_headers={"Retry-After": "5"} if exc.status == 503 else None)
                return
            if inspection is None:
                prompt = payload.get("prompt")
                batch = len(prompt) if isinstance(prompt, list) and prompt and not all(type(p) is int for p in prompt) else 1
                reserved_prompt = self.service.max_model_len * batch
                self.service.registry.increment("conservative_inspections_total")
        envelope = self.service.envelope(payload, self.headers, self.path, peer, inspection)
        decision = self.service.admission.acquire(
            envelope, reserved_prompt_tokens=reserved_prompt,
            body_bytes=getattr(self, "_request_body_bytes", 0), cancelled=self._client_disconnected,
        )
        self._usage_observer = UsageObserver(envelope.streaming) if self.service.inspector is not None else None
        self._upstream_status = 0

        try:
            if decision.admission_queued:
                self.service.registry.observe_admission_wait(decision.traffic_class, decision.admission_wait_seconds)
            self.service.registry.increment(
                "requests_total",
                traffic_class=decision.traffic_class,
                outcome=decision.outcome.name.lower(),
            )
            if decision.rescued_from_rejection:
                self.service.registry.increment(
                    "rescued_from_rejection_total", traffic_class=decision.traffic_class
                )
            if "adaptive class borrow" in decision.notes:
                self.service.registry.increment(
                    "adaptive_borrow_total", traffic_class=decision.traffic_class
                )

            self.service.recorder.record(decision)

            if not decision.admitted:
                if decision.admission_reason == "client_disconnected":
                    self._upstream_status = 499
                    self.close_connection = True
                    return
                if decision.reason.startswith("workload budget: "):
                    self.service.registry.increment("workload_rejections_total", reason=decision.reason.split(": ", 1)[1])
                self._refuse(decision)
                return

            apply_granted_tokens(payload, decision.clamp.granted)
            if self.service.send_priority:
                payload["priority"] = int(decision.priority)

            self._relay(payload, decision, started)
        finally:
            self.service.admission.release(decision)
            if self.service.inspector is not None:
                complete = self._usage_observer.finish()
                event = self.service.inspector.record(inspection, http_status=self._upstream_status,
                    usage=self._usage_observer.usage, complete=complete,
                    outcome=decision.outcome.name, granted=decision.clamp.granted)
                if event["verified_usage"]:
                    self.service.registry.increment("inspected_prompt_tokens_total", event["actual_prompt_tokens"])
                    self.service.registry.increment("inspected_cached_tokens_total", event["cached_tokens"])
                    if event.get("prior_completed_prefix_tokens", 0) > event["cached_tokens"]:
                        self.service.registry.increment("prior_completed_prefix_misses_total")
                elif complete and inspection is not None and event["actual_prompt_tokens"] is not None:
                    if event["actual_prompt_tokens"] != inspection.tokens:
                        self.service.registry.increment("token_count_mismatches_total")

    def _client_disconnected(self) -> bool:
        """Check an idle downstream socket without consuming pipelined data."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if readable:
                return self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
        except (BlockingIOError, InterruptedError):
            pass
        except (OSError, ValueError):
            return True
        return False

    def _relay(self, payload: dict, decision, started: float) -> None:
        offbox = (
            self.service.offbox
            if self.service.offbox and "routed off-box" in decision.notes
            else None
        )
        used_fallback = False
        try:
            if offbox:
                response = offbox.proxy(
                    self.path, payload, stream=decision.envelope.streaming
                )
            else:
                response = self.service.engine.proxy(
                    self.path, payload, stream=decision.envelope.streaming
                )
        except (OSError, http.client.HTTPException) as exc:
            fallback = (
                self.service.offbox
                if self.service.offbox
                and offbox is None
                and should_fallback(decision.priority)
                else None
            )
            if fallback is not None:
                try:
                    response = fallback.proxy(
                        self.path, payload, stream=decision.envelope.streaming
                    )
                    used_fallback = True
                except (OSError, http.client.HTTPException):
                    fallback = None
            if fallback is None:
                self._upstream_status = 502
                self.service.registry.increment(
                    "engine_errors_total", traffic_class=decision.traffic_class
                )
                self._json(502, {"error": {
                    "message": f"engine unreachable: {exc}",
                    "type": "upstream_error",
                }})
                return

        first_byte: float | None = None
        upstream_error_counted = False
        try:
            if used_fallback:
                self.service.registry.increment(
                    "offbox_fallback_total", traffic_class=decision.traffic_class
                )
            if 500 <= response.status < 600:
                self.service.registry.increment(
                    "engine_errors_total", traffic_class=decision.traffic_class
                )
                upstream_error_counted = True

            self.send_response(response.status)
            self._upstream_status = response.status
            for key, value in response.headers:
                self.send_header(key, value)
            self.send_header("x-k3-class", decision.traffic_class)
            self.send_header("x-k3-max-tokens-granted", str(decision.clamp.granted))
            self.send_header("x-k3-admission-wait-ms", str(round(decision.admission_wait_seconds * 1000)))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            for chunk in response.body:
                if not chunk:
                    continue
                if getattr(self, "_usage_observer", None) is not None:
                    self._usage_observer.feed(chunk)
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
            # A disconnect during a blocking upstream read is still undetected
            # until that read returns or times out; this is not active cancellation.
            self.service.registry.increment(
                "client_disconnects_total", traffic_class=decision.traffic_class
            )
            self.close_connection = True
            return
        except (OSError, http.client.HTTPException) as exc:
            # Headers may already have been sent. Omit the terminating chunk:
            # a truncated body must not be framed as a complete one.
            if not upstream_error_counted:
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
        self._upstream_status = status
        headers = {}
        if decision.retry_after_seconds:
            headers["Retry-After"] = str(decision.retry_after_seconds)
        headers["x-k3-admission-wait-ms"] = str(round(decision.admission_wait_seconds * 1000))
        self._json(
            status,
            {"error": {
                "message": decision.reason,
                "type": "service_unavailable" if status == 503 else (
                    "invalid_request_error" if status == 400 else "rate_limit_error"),
                "code": decision.admission_reason or decision.outcome.name.lower(),
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
        self._request_body_bytes = length
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
    offbox_url: str | None = None,
    offbox_api_key: str | None = None,
    offbox_model: str = "offbox",
    snapshot_ttl: float = 2.0,
    workload_limits: WorkloadLimits | None = None,
    batch_customers: frozenset[str] = frozenset(),
    admission_limits: AdmissionLimits | None = None,
    capacity_limits: CapacityLimits | None = None,
) -> GatewayService:
    engine = EngineClient(engine_url, snapshot_ttl=snapshot_ttl)
    sink = JsonlSink(Path(trace_path)) if trace_path else MemorySink()
    offbox = OffBoxClient(offbox_url, offbox_api_key, offbox_model) if offbox_url else None
    routed = bool(offbox) or offbox_configured
    controller = None
    health_source = engine
    if (
        workload_limits is not None
        and workload_limits.adaptive_borrow_requests > 0
    ):
        if capacity_limits is None:
            capacity_limits = CapacityLimits(
                max_borrow_requests=workload_limits.adaptive_borrow_requests,
                stop_kv=workload_limits.adaptive_borrow_kv_limit,
                green_kv=min(.55, workload_limits.adaptive_borrow_kv_limit * .8),
                stop_itl=workload_limits.adaptive_borrow_itl_limit,
                green_itl=min(.08, workload_limits.adaptive_borrow_itl_limit * .75),
            )
        controller = AdaptiveCapacityController(engine, capacity_limits)
        controller.start()
        health_source = controller

    policy = GatewayPolicy(
        classifier=Classifier(),
        clamp=TokenClamp(max_model_len=max_model_len),
        budget=ClassBudget.from_classes(
            concurrency_ceiling, ALL_CLASSES, offbox_configured=routed
        ),
        breaker=CircuitBreaker(health_source),
        offbox_configured=routed,
        workload_budget=(
            WorkloadBudget(workload_limits, health_source)
            if workload_limits is not None else None
        ),
        capacity_controller=controller,
    )
    return GatewayService(
        policy=policy,
        engine=engine,
        estimator=build_estimator(model_path),
        recorder=TraceRecorder(sink),
        registry=Registry(),
        send_priority=send_priority,
        offbox=offbox,
        batch_customers=batch_customers,
        inspector=PromptInspector(engine_url) if workload_limits is not None else None,
        max_model_len=max_model_len,
        admission_limits=admission_limits,
        capacity_controller=controller,
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
        "K3_ADMISSION_CEILING", 96)))
    parser.add_argument("--trace-path", default=os.environ.get("K3_TRACE_PATH"))
    parser.add_argument("--model-path", default=os.environ.get("K3_MODEL_PATH"))
    parser.add_argument("--send-priority", action="store_true",
                        default=os.environ.get("K3_SEND_PRIORITY", "") == "1",
                        help="attach a priority field to every upstream request; "
                             "only useful if the engine honours priority scheduling")
    parser.add_argument("--offbox-url", default=os.environ.get("K3_OFFBOX_URL"),
                        help="external OpenAI-compatible endpoint for P1 and E1 fallback")
    parser.add_argument("--offbox-api-key", default=os.environ.get("K3_OFFBOX_API_KEY"))
    parser.add_argument("--offbox-model", default=os.environ.get("K3_OFFBOX_MODEL", "offbox"))
    parser.add_argument("--batch-customers", default=os.environ.get("K3_BATCH_CUSTOMERS", ""),
                        help="comma-separated customers allowed to set X-K3-Batch")
    args = parser.parse_args(argv)
    workload_limits = None
    if os.environ.get("K3_WORKLOAD_GUARD") == "1":
        workload_limits = WorkloadLimits(
            large_context_tokens=int(os.environ.get("K3_LARGE_CONTEXT_TOKENS", "65536")),
            large_context_requests=int(os.environ.get("K3_LARGE_CONTEXT_MAX", "4")),
            long_output_tokens=int(os.environ.get("K3_LONG_OUTPUT_TOKENS", "2048")),
            long_output_requests=int(os.environ.get("K3_LONG_OUTPUT_MAX", "2")),
            reserved_tokens=int(os.environ.get("K3_RESERVED_TOKEN_BUDGET", "1048576")),
            projected_kv_limit=float(os.environ.get("K3_PROJECTED_KV_LIMIT", "0.85")),
            long_output_burst_requests=int(os.environ.get("K3_LONG_OUTPUT_BURST_MAX",
                                                         os.environ.get("K3_LONG_OUTPUT_MAX", "2"))),
            long_output_burst_kv_limit=float(os.environ.get("K3_LONG_OUTPUT_BURST_KV", "0.55")),
            long_output_burst_running_limit=int(os.environ.get("K3_LONG_OUTPUT_BURST_RUNNING", "8")),
            long_output_burst_prompt_limit=int(os.environ.get("K3_LONG_OUTPUT_BURST_PROMPT", "32768")),
            engine_queue_tolerance=int(os.environ.get("K3_ENGINE_QUEUE_TOLERANCE", "0")),
            adaptive_borrow_requests=int(os.environ.get("K3_ADAPTIVE_BORROW_MAX", "0")),
            adaptive_borrow_kv_limit=float(os.environ.get("K3_ADAPTIVE_BORROW_KV", "0.70")),
            adaptive_borrow_itl_limit=float(os.environ.get("K3_ADAPTIVE_BORROW_ITL", "0.12")),
        )
    max_waiters = int(os.environ.get("K3_ADMISSION_MAX_WAITERS", "16"))
    admission_limits = AdmissionLimits(
        wait_seconds=float(os.environ.get("K3_ADMISSION_WAIT_SECONDS", "0")),
        max_waiters=max_waiters,
        max_per_customer=int(os.environ.get("K3_ADMISSION_MAX_PER_CUSTOMER", str(max_waiters))),
        max_queued_tokens=int(os.environ.get("K3_ADMISSION_QUEUED_TOKENS", "2097152")),
        max_queued_bytes=int(os.environ.get("K3_ADMISSION_QUEUED_BYTES", "67108864")),
    )

    Handler.service = build_service(
        engine_url=args.engine_url,
        max_model_len=args.max_model_len,
        concurrency_ceiling=args.ceiling,
        trace_path=args.trace_path,
        model_path=args.model_path,
        send_priority=args.send_priority,
        offbox_url=args.offbox_url,
        offbox_api_key=args.offbox_api_key,
        offbox_model=args.offbox_model,
        batch_customers=frozenset(
            c.strip() for c in args.batch_customers.split(",") if c.strip()
        ),
        workload_limits=workload_limits,
        admission_limits=admission_limits,
    )

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True

    print(f"gateway on {args.bind}:{args.port} -> {args.engine_url}", file=sys.stderr)
    print(f"  max_model_len={args.max_model_len} ceiling={args.ceiling} "
          f"priority={'on' if args.send_priority else 'off'}", file=sys.stderr)
    print(f"  traces -> {args.trace_path or '(memory only)'}", file=sys.stderr)
    print(f"  admission_wait={admission_limits.wait_seconds:g}s "
          f"queue_limit={admission_limits.max_waiters}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        Handler.service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
