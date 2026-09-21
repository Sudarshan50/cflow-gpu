"""HTTP front door: classify, clamp, admit, then relay.

    python3 -m redesign.gateway.server --port 8002
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import signal
import sys
import time
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .backpressure import CircuitBreaker, ClassBudget
from .cancellation import (
    DEFAULT_TIMEOUT_SECONDS,
    CancelReason,
    RequestCancellation,
    RequestCancelled,
)
from .capture import JsonlSink, MemorySink, RotatingJsonlSink, TraceRecorder
from .completions import CompletionObserver, CompletionSummary
from .classification import ALL_CLASSES, Classifier
from .clamping import InvalidTokenLimit, TokenClamp, requested_output_tokens, validate_token_limits
from .engine import EngineClient
from .engine_tokens import EngineTokenEstimator, TokenCountUnavailable
from .prompt_protocol import normalize_engine_prompt, TokenCountInputError
from .media import normalize_payload, media_cache_stats, MediaValidationError, MediaBusyError
from .metrics import Registry
from .models import Outcome, RequestEnvelope
from .offbox import OffBoxClient, should_fallback
from .policy import GatewayPolicy
from .tokens import TokenEstimator, build_estimator, has_images

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


@dataclass
class _RelayState:
    # "relayed" means upstream HTTP EOF and a written final downstream chunk.
    # It does NOT assert valid JSON, an SSE [DONE], or semantic model completion.
    terminal: str = "internal_error"
    status: int | None = None
    engine_error_counted: bool = False
    observer: CompletionObserver | None = None


class _DownstreamError(Exception):
    pass


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
    return any((_coerce_int(payload.get(key)) or 1) > 1 for key in ("n", "best_of"))


def apply_granted_tokens(payload: dict, granted: int) -> None:
    payload["max_tokens"] = granted
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = granted


def output_policy_headers(decision) -> dict[str, str]:
    # Requested is this gateway's input, which may already be clamped by
    # tenancy. Never reconstruct an "original" limit from caller metadata.
    clamp = decision.clamp
    return {
        "x-k3-output-policy-stage": "gateway",
        "x-k3-class": decision.traffic_class,
        "x-k3-max-tokens-requested": str(clamp.requested) if clamp.requested is not None else "none",
        "x-k3-max-tokens-granted": str(clamp.granted),
        "x-k3-output-tokens-default": str(clamp.default_output_tokens),
        "x-k3-output-default-applied": str(clamp.default_applied).lower(),
        "x-k3-clamp-reason": clamp.reason,
        "x-k3-prompt-tokens": str(decision.envelope.prompt_tokens),
        "x-k3-prompt-count-source": decision.envelope.prompt_count_source,
        "x-k3-prompt-count-cache-hit": str(decision.envelope.prompt_count_cached).lower(),
    }


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
        request_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        preprocessing_limit: int = 8,
    ) -> None:
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        self.policy = policy
        self.engine = engine
        self.estimator = estimator
        self.recorder = recorder
        self.registry = registry
        self.send_priority = send_priority
        self.offbox = offbox
        self.batch_customers = batch_customers
        self.request_timeout = request_timeout
        if type(preprocessing_limit) is not int or preprocessing_limit <= 0:
            raise ValueError("preprocessing_limit must be positive")
        self.preprocessing = threading.BoundedSemaphore(preprocessing_limit)

    def publish_admission_metrics(self) -> None:
        snapshot = self.policy.budget_snapshot()
        gauges = {
            "admission_in_flight": snapshot.in_flight,
            "admission_global_limit": snapshot.ceiling,
            "admission_borrowed_in_flight": snapshot.borrowed_in_flight,
            "admission_borrowing_enabled": int(snapshot.borrowing_enabled),
            "admission_p0_reserve": snapshot.p0_reserve,
            "admission_context_work_tokens": snapshot.context_tokens_in_flight,
            "admission_context_budget_enabled": int(snapshot.context_token_budget is not None),
            "admission_context_work_limit": snapshot.effective_context_token_budget or 0,
            "admission_kv_capacity_known": int(snapshot.kv_capacity_tokens is not None),
            "admission_kv_capacity_tokens": snapshot.kv_capacity_tokens or 0,
        }
        for name, value in gauges.items():
            self.registry.set_gauge(name, value)
        for name, count in snapshot.by_class.items():
            self.registry.set_gauge("admission_class_in_flight", count, traffic_class=name)
            self.registry.set_gauge("admission_class_soft_limit", snapshot.class_limits[name], traffic_class=name)
            self.registry.set_gauge("admission_class_borrowed", snapshot.borrowed_by_class[name], traffic_class=name)
        if isinstance(self.estimator, EngineTokenEstimator):
            for name, value in self.estimator.snapshot().items():
                setter = self.registry.set_counter if name.endswith(("_total", "_sum")) else self.registry.set_gauge
                setter("tokenize_" + name, value)
        media_gauges = {"entries", "bytes", "pending", "max_entries", "max_bytes", "ttl_seconds", "workers_limit", "workers_in_flight", "workers_waiting", "request_images_limit"}
        for name, value in media_cache_stats().items():
            if name in media_gauges:
                self.registry.set_gauge("media_" + name, value)
            else:
                self.registry.set_counter("media_" + name + "_total", value)

    def close(self) -> None:
        if isinstance(self.estimator, EngineTokenEstimator):
            self.estimator.close()

    def envelope(
        self, payload: dict, headers, path: str, peer: str = ""
    ) -> RequestEnvelope:
        # vLLM gives the newer alias precedence when both are supplied.
        requested = requested_output_tokens(payload)
        count = self.estimator.count(payload) if isinstance(self.estimator, EngineTokenEstimator) else None
        return RequestEnvelope(
            customer=headers.get(CUSTOMER_HEADER, "anonymous"),
            prompt_tokens=count.count if count is not None else self.estimator.estimate(payload),
            requested_max_tokens=requested,
            streaming=bool(payload.get("stream")),
            has_tools=bool(payload.get("tools") or payload.get("functions")),
            has_images=has_images(payload),
            batch_hint=batch_authorized(headers, peer, self.batch_customers),
            path=path,
            tools_disabled=(payload.get("tool_choice") == "none" or payload.get("function_call") == "none"),
            prompt_count_source=count.source if count is not None else "heuristic",
            prompt_count_cached=count.cache_hit if count is not None else False,
            model_context_limit=count.max_model_len if count is not None else None,
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
            self.service.publish_admission_metrics()
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

        try:
            validate_token_limits(payload)
        except InvalidTokenLimit as exc:
            self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return

        started = time.monotonic()
        if overcommitted_replicas(payload):
            self._json(400, {"error": {
                "message": "n/best_of > 1 is not served on a single replica",
                "type": "invalid_request_error",
            }})
            return

        # The body has been read; monitoring the raw socket now cannot consume
        # it or interfere with rfile. The watcher is joined before keep-alive
        # dispatches another request. A None socket supports in-process fakes.
        with RequestCancellation(
            timeout=self.service.request_timeout,
            downstream=getattr(self, "connection", None),
            started=started,
        ) as cancellation:
            decision = None
            state = None
            try:
                cancellation.check()
                peer = self.client_address[0] if self.client_address else ""
                if not self.service.preprocessing.acquire(blocking=False):
                    raise TokenCountUnavailable("request preprocessing is busy", reason="preprocessing_busy")
                try:
                    normalize_payload(payload)
                    if isinstance(self.service.estimator, EngineTokenEstimator):
                        payload = normalize_engine_prompt(payload, self.path)
                    envelope = self.service.envelope(payload, self.headers, self.path, peer)
                finally:
                    self.service.preprocessing.release()
                decision = self.service.policy.decide(envelope)
                if decision.admitted:
                    state = _RelayState()
                self.service.registry.increment(
                    "requests_total",
                    traffic_class=decision.traffic_class,
                    outcome=decision.outcome.name.lower(),
                )
                if decision.metadata.get("borrowed"):
                    self.service.registry.increment("admission_borrowed_total", traffic_class=decision.traffic_class)
                if decision.metadata.get("admission_reason"):
                    self.service.registry.increment(
                        "admission_rejections_total", traffic_class=decision.traffic_class,
                        reason=decision.metadata["admission_reason"],
                    )
                if decision.rescued_from_rejection:
                    self.service.registry.increment(
                        "rescued_from_rejection_total", traffic_class=decision.traffic_class
                    )

                try:
                    self.service.recorder.record(decision)
                except OSError:
                    # A full/unwritable trace volume must not break serving or
                    # strand an admission reservation.
                    self.service.registry.increment(
                        "recorder_errors_total", traffic_class=decision.traffic_class
                    )
                cancellation.check()

                if not decision.admitted:
                    self._write_downstream(cancellation, self._refuse, decision)
                    cancellation.finish()
                    return

                apply_granted_tokens(payload, decision.clamp.granted)
                if self.service.send_priority:
                    payload["priority"] = int(decision.priority)

                self._relay(payload, decision, started, cancellation, state)
            except (MediaValidationError, TokenCountInputError) as exc:
                reason = getattr(exc, "reason", type(exc).__name__)
                self.service.registry.increment("input_rejections_total", reason=reason)
                self._write_downstream(cancellation, self._json, 400, {"error": {
                    "message": str(exc), "type": "invalid_request_error", "code": reason,
                    "param": getattr(exc, "param", None)}})
            except (MediaBusyError, TokenCountUnavailable) as exc:
                self.service.registry.increment("preprocessing_rejections_total", reason=getattr(exc, "reason", "media_busy"))
                self._write_downstream(cancellation, self._json, 503, {"error": {
                    "message": str(exc), "type": "service_unavailable"}}, {"Retry-After": "2"})
            except RequestCancelled:
                self.close_connection = True
            except _DownstreamError:
                self.close_connection = True
                if state is not None:
                    state.terminal = "downstream_error"
            finally:
                try:
                    reason = cancellation.finish()
                    traffic_class = decision.traffic_class if decision else "unclassified"
                    if reason is not None:
                        self.close_connection = True
                        if state is not None:
                            state.terminal = reason.value
                        self.service.registry.increment(
                            "client_disconnects_total"
                            if reason is CancelReason.CLIENT_DISCONNECT
                            else "request_deadlines_total",
                            traffic_class=traffic_class,
                        )
                    if state is not None:
                        summary = state.observer.finish() if state.observer else CompletionSummary()
                        if summary.first_token_seconds is not None:
                            self.service.registry.observe_ttft(traffic_class, summary.first_token_seconds)
                        if state.terminal == "relayed" and state.status is not None and 200 <= state.status < 300:
                            if summary.protocol_complete:
                                self.service.registry.increment("protocol_completions_total", traffic_class=traffic_class)
                                for finish in summary.finish_reasons:
                                    self.service.registry.increment("finish_reasons_total", traffic_class=traffic_class, reason=finish)
                            elif summary.protocol_complete is False:
                                self.service.registry.increment("protocol_incomplete_total", traffic_class=traffic_class)
                        if summary.prompt_tokens is not None and decision.envelope.prompt_count_source == "engine_rendered":
                            self.service.registry.increment("token_count_comparisons_total")
                            if summary.prompt_tokens != decision.envelope.prompt_tokens:
                                self.service.registry.increment("token_count_mismatches_total")
                        try:
                            self.service.recorder.record_completion(
                                decision, summary, time.monotonic() - started, state.terminal, state.status)
                        except OSError:
                            self.service.registry.increment("recorder_errors_total", traffic_class=traffic_class)
                        self.service.registry.increment(
                            "relay_terminal_total",
                            traffic_class=traffic_class,
                            outcome=state.terminal,
                            upstream_status=str(state.status) if state.status is not None else "none",
                        )
                        # Preserve the latency series' full-relay population;
                        # short cancellations must not improve its quantiles.
                        if state.terminal in ("relayed", "upstream_http_5xx"):
                            self.service.registry.observe_total(
                                traffic_class, time.monotonic() - started
                            )
                finally:
                    # Includes recorder/metrics failures, cancellation before
                    # headers, all body paths, rejection and fallback.
                    if decision is not None:
                        self.service.policy.release(decision)

    def _proxy(self, target, payload, decision, cancellation, upstream):
        cancellation.check()
        # The production off-box wrapper delegates synchronously to EngineClient
        # and therefore inherits the context. Legacy test/application engines
        # still receive exactly one call with the historical proxy signature.
        transport = target.engine if isinstance(target, OffBoxClient) else target
        if getattr(transport, "supports_request_cancellation", False) is not True:
            self.service.registry.increment(
                "cancellation_unsupported_total",
                traffic_class=decision.traffic_class,
                upstream=upstream,
            )
        return target.proxy(self.path, payload, stream=decision.envelope.streaming)

    def _write_downstream(self, cancellation, write, *args) -> None:
        cancellation.check()
        try:
            write(*args)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
            cancellation.check()
            raise
        except OSError as exc:
            cancellation.check()
            # E.g. a slow downstream write timeout is not an engine failure or
            # proof of a client disconnect.
            raise _DownstreamError(str(exc)) from exc
        cancellation.check()

    def _engine_error(self, decision, state) -> None:
        if not state.engine_error_counted:
            self.service.registry.increment(
                "engine_errors_total", traffic_class=decision.traffic_class
            )
            state.engine_error_counted = True

    def _relay(
        self, payload: dict, decision, started: float,
        cancellation: RequestCancellation, state: _RelayState,
    ) -> None:
        offbox = (
            self.service.offbox
            if self.service.offbox and "routed off-box" in decision.notes
            else None
        )
        used_fallback = False
        try:
            response = self._proxy(
                offbox or self.service.engine, payload, decision, cancellation,
                "offbox" if offbox else "engine",
            )
        except (OSError, http.client.HTTPException) as exc:
            cancellation.check()
            fallback = (
                self.service.offbox
                if self.service.offbox
                and offbox is None
                and should_fallback(decision.priority)
                else None
            )
            if fallback is not None:
                try:
                    response = self._proxy(
                        fallback, payload, decision, cancellation, "offbox"
                    )
                    used_fallback = True
                except (OSError, http.client.HTTPException):
                    cancellation.check()
                    fallback = None
            if fallback is None:
                state.terminal = "upstream_error"
                self._engine_error(decision, state)
                self.service.registry.increment(
                    "upstream_transport_errors_total", traffic_class=decision.traffic_class
                )
                self._write_downstream(cancellation, self._json, 502, {"error": {
                    "message": f"engine unreachable: {exc}",
                    "type": "upstream_error",
                }})
                cancellation.finish()
                return

        try:
            state.status = response.status
            content_type = next((value for name, value in response.headers if name.lower() == "content-type"), "")
            state.observer = CompletionObserver(stream="text/event-stream" in content_type, started=started)
            if 500 <= response.status < 600:
                self._engine_error(decision, state)
                self.service.registry.increment(
                    "upstream_http_5xx_total", traffic_class=decision.traffic_class,
                    status=str(response.status),
                )
            if used_fallback:
                self.service.registry.increment(
                    "offbox_fallback_total", traffic_class=decision.traffic_class
                )

            def send_headers():
                self.send_response(response.status)
                for key, value in response.headers:
                    self.send_header(key, value)
                for key, value in output_policy_headers(decision).items():
                    self.send_header(key, value)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            self._write_downstream(cancellation, send_headers)
            body = iter(response.body)
            while True:
                try:
                    chunk = next(body)
                except StopIteration:
                    break
                except (OSError, http.client.HTTPException) as exc:
                    cancellation.check()
                    state.terminal = "upstream_error"
                    self._engine_error(decision, state)
                    self.service.registry.increment(
                        "upstream_transport_errors_total", traffic_class=decision.traffic_class
                    )
                    self.log_error("upstream body failed: %s", exc)
                    # A truncated body must not receive the terminating chunk.
                    self.close_connection = True
                    return
                cancellation.check()
                if not chunk:
                    continue
                state.observer.feed(chunk)
                self._write_downstream(
                    cancellation, self.wfile.write, f"{len(chunk):X}\r\n".encode()
                )
                self._write_downstream(cancellation, self.wfile.write, chunk)
                self._write_downstream(cancellation, self.wfile.write, b"\r\n")
            self._write_downstream(cancellation, self.wfile.write, b"0\r\n\r\n")
            cancellation.finish()
            cancellation.check()
            state.terminal = "upstream_http_5xx" if 500 <= response.status < 600 else "relayed"
        finally:
            try:
                response.close()
            except OSError:
                self.service.registry.increment(
                    "upstream_close_errors_total", traffic_class=decision.traffic_class
                )

    def _refuse(self, decision) -> None:
        status = 503 if decision.outcome is Outcome.REJECT_SHED else 429
        if decision.clamp.granted == 0:
            status = 400
        headers = output_policy_headers(decision)
        if decision.retry_after_seconds:
            headers["Retry-After"] = str(decision.retry_after_seconds)
        self._json(
            status,
            {"error": {
                "message": decision.reason,
                "type": "rate_limit_error" if status != 400 else "invalid_request_error",
                "k3_class": decision.traffic_class,
                "k3_admission_reason": decision.metadata.get("admission_reason"),
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
    offbox_url: str | None = None,
    offbox_api_key: str | None = None,
    offbox_model: str = "offbox",
    snapshot_ttl: float = 2.0,
    batch_customers: frozenset[str] = frozenset(),
    request_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    borrowing_enabled: bool = False,
    p0_reserve: int = 4,
    context_token_budget: int | None = None,
    token_estimator: str = "heuristic",
    tokenize_timeout: float = 2.0,
    tokenize_revision: str = "1",
    completion_trace_path: str | None = None,
) -> GatewayService:
    engine = EngineClient(engine_url, timeout=request_timeout, snapshot_ttl=snapshot_ttl)
    sink = JsonlSink(Path(trace_path)) if trace_path else MemorySink()
    offbox = OffBoxClient(offbox_url, offbox_api_key, offbox_model) if offbox_url else None
    routed = bool(offbox) or offbox_configured
    if completion_trace_path and trace_path and Path(completion_trace_path) == Path(trace_path):
        raise ValueError("Completion and admission traces must use separate files")
    completion_sink = RotatingJsonlSink(Path(completion_trace_path)) if completion_trace_path else None
    if token_estimator not in ("engine", "heuristic"):
        raise ValueError("token_estimator must be engine or heuristic")

    policy = GatewayPolicy(
        classifier=Classifier(),
        clamp=TokenClamp(max_model_len=max_model_len),
        budget=ClassBudget.from_classes(
            concurrency_ceiling, ALL_CLASSES, offbox_configured=routed,
            borrowing_enabled=borrowing_enabled, p0_reserve=p0_reserve,
            context_token_budget=context_token_budget,
        ),
        breaker=CircuitBreaker(engine),
        offbox_configured=routed,
    )
    estimator = (EngineTokenEstimator(engine_url, timeout=tokenize_timeout, revision=tokenize_revision)
                 if token_estimator == "engine" else build_estimator(model_path))
    return GatewayService(
        policy=policy,
        engine=engine,
        estimator=estimator,
        recorder=TraceRecorder(sink, completion_sink=completion_sink),
        registry=Registry(),
        send_priority=send_priority,
        offbox=offbox,
        batch_customers=batch_customers,
        request_timeout=request_timeout,
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
    parser.add_argument("--request-timeout", type=float, default=float(os.environ.get(
        "K3_REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        help="wall-clock seconds after body receipt, including admission and relay (default: 600)")
    parser.add_argument("--borrow-capacity", action="store_true",
                        default=os.environ.get("K3_ADMISSION_BORROWING", "0") == "1")
    parser.add_argument("--p0-reserve", type=int, default=int(os.environ.get("K3_P0_RESERVED_SLOTS", "4")))
    parser.add_argument("--context-token-budget", type=int,
                        default=int(os.environ.get("K3_CONTEXT_TOKEN_BUDGET", "0")),
                        help="in-flight estimated input + granted output work limit; 0 disables")
    parser.add_argument("--token-estimator", choices=("heuristic", "engine"), default=os.environ.get("K3_TOKEN_ESTIMATOR", "heuristic"))
    parser.add_argument("--tokenize-timeout", type=float, default=float(os.environ.get("K3_TOKENIZE_TIMEOUT_SECONDS", "2")))
    parser.add_argument("--tokenize-revision", default=os.environ.get("K3_TOKENIZER_REVISION", "1"))
    parser.add_argument("--completion-trace-path", default=os.environ.get("K3_COMPLETION_TRACE_PATH"))
    args = parser.parse_args(argv)
    if args.context_token_budget < 0 or args.p0_reserve < 0:
        parser.error("context token budget and reserved slots must be nonnegative")

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
        request_timeout=args.request_timeout,
        borrowing_enabled=args.borrow_capacity,
        p0_reserve=args.p0_reserve,
        context_token_budget=args.context_token_budget or None,
        token_estimator=args.token_estimator,
        tokenize_timeout=args.tokenize_timeout,
        tokenize_revision=args.tokenize_revision,
        completion_trace_path=args.completion_trace_path,
    )

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    server.daemon_threads = True

    print(f"gateway on {args.bind}:{args.port} -> {args.engine_url}", file=sys.stderr)
    print(f"  max_model_len={args.max_model_len} ceiling={args.ceiling} "
          f"priority={'on' if args.send_priority else 'off'}", file=sys.stderr)
    print(f"  traces -> {args.trace_path or '(memory only)'}", file=sys.stderr)
    print(f"  borrowing={args.borrow_capacity} p0_reserve={args.p0_reserve} "
          f"context_work_budget={args.context_token_budget or 'disabled'}", file=sys.stderr)

    def stop_gateway(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_gateway)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    finally:
        Handler.service.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
