"""Bounded, work-conserving admission waiting. Never retries an inference.

Each lane is FIFO. Independent resource classes can make progress while a
heavy lane waits; new arrivals cannot steal a waiting request's lane. Waiting
owns queue capacity only, never an engine/class/token reservation.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Callable

from .models import Decision, RequestEnvelope
from .policy import GatewayPolicy


@dataclass(frozen=True)
class AdmissionLimits:
    wait_seconds: float = 0.0
    max_waiters: int = 16
    max_per_customer: int = 8
    max_queued_tokens: int = 2_097_152
    max_queued_bytes: int = 64 * 1024 * 1024
    poll_seconds: float = .2

    def __post_init__(self):
        if not math.isfinite(self.wait_seconds) or not 0 <= self.wait_seconds <= 60:
            raise ValueError("Admission wait must be between zero and 60 seconds")
        if min(self.max_waiters, self.max_per_customer, self.max_queued_tokens, self.max_queued_bytes) <= 0:
            raise ValueError("Admission queue limits must be positive")
        if not math.isfinite(self.poll_seconds) or not 0 < self.poll_seconds <= 1:
            raise ValueError("Admission poll interval must be between zero and one second")


@dataclass(eq=False)
class _Ticket:
    lane: tuple[str, bool, bool]
    customer: str
    tokens: int
    body_bytes: int
    started: float


class AdmissionController:
    def __init__(self, policy: GatewayPolicy, limits: AdmissionLimits):
        self.policy = policy
        self.limits = limits
        self._cv = threading.Condition()
        self._waiting: list[_Ticket] = []
        self._probing: set[tuple[str, bool, bool]] = set()
        self._per_customer: dict[str, int] = {}
        self._tokens = self._bytes = 0
        self._counters = {key: 0 for key in (
            "queued_total", "queued_admitted_total", "queue_timeouts_total",
            "queue_full_total", "cancelled_total",
        )}
        self._events: deque[dict] = deque(maxlen=512)

    def _enqueue(self, lane, envelope, charge, body_bytes, started):
        """Called with the condition locked."""
        reason = ""
        queue_limit = self.policy.admission_queue_limit(self.limits.max_waiters)
        if len(self._waiting) >= queue_limit:
            reason = "queue_full"
        elif self._per_customer.get(envelope.customer, 0) >= self.limits.max_per_customer:
            reason = "customer_queue_full"
        elif self._tokens + charge > self.limits.max_queued_tokens:
            reason = "queued_token_limit"
        elif self._bytes + body_bytes > self.limits.max_queued_bytes:
            reason = "queued_byte_limit"
        if reason:
            self._counters["queue_full_total"] += 1
            return None, reason
        ticket = _Ticket(lane, envelope.customer, charge, body_bytes, started)
        self._waiting.append(ticket)
        self._per_customer[envelope.customer] = self._per_customer.get(envelope.customer, 0) + 1
        self._tokens += charge
        self._bytes += body_bytes
        self._counters["queued_total"] += 1
        return ticket, ""

    def _remove(self, ticket):
        """Called with the condition locked; also safe on exception cleanup."""
        if ticket not in self._waiting:
            return
        self._waiting.remove(ticket)
        self._tokens -= ticket.tokens
        self._bytes -= ticket.body_bytes
        count = self._per_customer[ticket.customer] - 1
        if count:
            self._per_customer[ticket.customer] = count
        else:
            del self._per_customer[ticket.customer]

    def _finish(self, decision, ticket, reason):
        waited = max(0.0, time.monotonic() - ticket.started) if ticket else 0.0
        decision = replace(decision, admission_wait_seconds=waited,
                           admission_queued=ticket is not None, admission_reason=reason)
        with self._cv:
            if ticket and decision.admitted:
                self._counters["queued_admitted_total"] += 1
            if reason == "queue_timeout":
                self._counters["queue_timeouts_total"] += 1
            elif reason == "client_disconnected":
                self._counters["cancelled_total"] += 1
            self._events.append({
                "timestamp": time.time(), "traffic_class": decision.traffic_class,
                "prompt_tokens": decision.envelope.prompt_tokens,
                "granted_max_tokens": decision.clamp.granted,
                "queued_tokens": ticket.tokens if ticket else 0,
                "wait_seconds": waited, "admitted": decision.admitted,
                "admission_reason": reason, "policy_reason": decision.reason,
            })
        return decision

    @staticmethod
    def _disconnected(cancelled):
        try:
            return cancelled is not None and cancelled()
        except (OSError, ValueError):
            # A closed/invalid socket must not leak a lease obtained just
            # before this check, or send work to an abandoned request.
            return True

    def acquire(
        self, envelope: RequestEnvelope, reserved_prompt_tokens: int | None = None,
        *, body_bytes: int = 0, cancelled: Callable[[], bool] | None = None,
    ) -> Decision:
        wait_seconds = self.policy.admission_wait_seconds(self.limits.wait_seconds)
        if wait_seconds == 0:
            return self.policy.decide(envelope, reserved_prompt_tokens)
        if body_bytes < 0:
            raise ValueError("Request byte count cannot be negative")
        started = time.monotonic()
        lane, charge = self.policy.admission_shape(envelope, reserved_prompt_tokens)
        if lane is None:  # invalid input or off-box routing never waits locally
            return self.policy.decide(envelope, reserved_prompt_tokens)
        if self._disconnected(cancelled):
            return self._finish(self.policy.admission_rejection(envelope, "client disconnected before admission"),
                                None, "client_disconnected")

        ticket = None
        full_reason = ""
        with self._cv:
            fast = lane not in self._probing and not any(t.lane == lane for t in self._waiting)
            if fast:
                self._probing.add(lane)
            else:
                ticket, full_reason = self._enqueue(lane, envelope, charge, body_bytes, started)

        last = self.policy.admission_rejection(envelope, "waiting for admission capacity")
        if fast:
            try:
                last = self.policy.decide(envelope, reserved_prompt_tokens)
                if last.admitted and self._disconnected(cancelled):
                    self.policy.release(last)
                    last = self.policy.admission_rejection(envelope, "client disconnected before admission")
                    return self._finish(last, None, "client_disconnected")
                if last.admitted or not self.policy.can_wait(last, reserved_prompt_tokens):
                    return last
                with self._cv:
                    ticket, full_reason = self._enqueue(lane, envelope, charge, body_bytes, started)
            finally:
                with self._cv:
                    self._probing.discard(lane)
                    self._cv.notify_all()

        if full_reason:
            rejected = self.policy.admission_rejection(envelope, "admission queue: " + full_reason)
            return self._finish(rejected, None, full_reason)

        assert ticket is not None
        deadline = started + wait_seconds
        try:
            while True:
                # Pressure can shorten an existing wait, but recovery cannot
                # silently extend the deadline promised when it was enqueued.
                deadline = min(
                    deadline,
                    started + self.policy.admission_wait_seconds(self.limits.wait_seconds),
                )
                if self._disconnected(cancelled):
                    rejected = self.policy.admission_rejection(envelope, "client disconnected before admission")
                    return self._finish(rejected, ticket, "client_disconnected")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._finish(last, ticket, "queue_timeout")
                with self._cv:
                    first = next(t for t in self._waiting if t.lane == lane)
                    if first is not ticket or lane in self._probing:
                        self._cv.wait(min(self.limits.poll_seconds, remaining))
                        continue
                    self._probing.add(lane)
                try:
                    decision = self.policy.decide(envelope, reserved_prompt_tokens)
                finally:
                    with self._cv:
                        self._probing.discard(lane)
                        self._cv.notify_all()
                if decision.admitted:
                    # The health scrape may have consumed the remaining wait
                    # deadline. Never dispatch abandoned/expired work, and
                    # return an acquired lease before leaving this path.
                    disconnected = self._disconnected(cancelled)
                    if disconnected or time.monotonic() >= deadline:
                        self.policy.release(decision)
                        if disconnected:
                            last = self.policy.admission_rejection(envelope, "client disconnected before admission")
                        return self._finish(last, ticket, "client_disconnected" if disconnected else "queue_timeout")
                    return self._finish(decision, ticket, "admitted_after_wait")
                last = decision
                if not self.policy.can_wait(last, reserved_prompt_tokens):
                    return self._finish(last, ticket, "not_retryable")
                with self._cv:
                    self._cv.wait(min(self.limits.poll_seconds, max(0.0, deadline - time.monotonic())))
        finally:
            with self._cv:
                self._remove(ticket)
                self._cv.notify_all()

    def release(self, decision: Decision) -> None:
        self.policy.release(decision)
        with self._cv:
            self._cv.notify_all()

    def state(self) -> dict[str, int]:
        with self._cv:
            return {"queued_requests": len(self._waiting), "queued_tokens": self._tokens,
                    "queued_bytes": self._bytes, **self._counters}

    def recent(self) -> dict:
        with self._cv:
            events = list(self._events)
        return {"limits": asdict(self.limits), "state": self.state(), "events": events}
