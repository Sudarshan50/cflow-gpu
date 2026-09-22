"""Orchestrates classification, clamping and backpressure into one decision."""

from __future__ import annotations

from .backpressure import (
    DEFAULT_RETRY_AFTER_SECONDS,
    CircuitBreaker,
    ClassBudget,
    StaticHealthSource,
)
from .classification import ALL_CLASSES, Classifier
from .clamping import PROMPT_TOO_LONG, TokenClamp
from .models import Decision, EngineSnapshot, Outcome, RequestEnvelope
from .workload import WorkloadBudget


class GatewayPolicy:
    def __init__(
        self,
        classifier: Classifier,
        clamp: TokenClamp,
        budget: ClassBudget,
        breaker: CircuitBreaker,
        offbox_configured: bool = False,
        workload_budget: WorkloadBudget | None = None,
        capacity_controller=None,
    ) -> None:
        self._classifier = classifier
        self._clamp = clamp
        self._budget = budget
        self._breaker = breaker
        # A class is only exempt from the local budget if there is somewhere
        # else for it to go. Without an off-box target, honouring
        # `served_off_box` would admit that traffic to this engine while
        # skipping its concurrency slot -- worse than having no class at all.
        self._offbox_configured = offbox_configured
        self.workload_budget = workload_budget
        self.capacity_controller = capacity_controller

    def admission_wait_seconds(self, configured: float) -> float:
        if self.capacity_controller is None:
            return configured
        return self.capacity_controller.queue_wait_seconds(configured)

    def admission_queue_limit(self, configured: int) -> int:
        if self.capacity_controller is None:
            return configured
        return self.capacity_controller.queue_limit(configured)

    def _is_offbox(self, traffic_class) -> bool:
        return traffic_class.served_off_box and self._offbox_configured

    def _preview(self, envelope: RequestEnvelope):
        traffic_class = self._classifier.classify(envelope)
        clamp = self._clamp.apply(
            envelope.prompt_tokens, envelope.requested_max_tokens, traffic_class
        )
        return traffic_class, clamp

    def admission_shape(self, envelope: RequestEnvelope, reserved_prompt_tokens: int | None = None):
        """Pure queue classification; waiting never owns an execution slot."""
        traffic_class, clamp = self._preview(envelope)
        reserved = max(envelope.prompt_tokens, reserved_prompt_tokens or 0)
        charge = reserved + clamp.granted
        if clamp.granted == 0 or self._is_offbox(traffic_class):
            return None, charge
        large = long = False
        if self.workload_budget is not None:
            charge, large, long = self.workload_budget.requirements(reserved, clamp.granted, envelope.has_images)
            if charge > self.workload_budget.limits.reserved_tokens:
                return None, charge
        if self._budget.limit_for(traffic_class) <= 0:
            return None, charge
        return (traffic_class.name, large, long), charge

    def admission_rejection(self, envelope: RequestEnvelope, reason: str) -> Decision:
        traffic_class, clamp = self._preview(envelope)
        return self._reject(envelope, traffic_class, clamp, Outcome.REJECT_SHED, reason)

    def can_wait(self, decision: Decision, reserved_prompt_tokens: int | None = None) -> bool:
        if decision.admitted or decision.clamp.granted == 0:
            return False
        traffic_class = self._classifier.classify(decision.envelope)
        if self._budget.limit_for(traffic_class) <= 0:
            return False
        if self.workload_budget is not None:
            reserved = max(decision.envelope.prompt_tokens, reserved_prompt_tokens or 0)
            if reserved + decision.clamp.granted > self.workload_budget.limits.reserved_tokens:
                return False  # an individually impossible reservation will not clear
        return True

    def decide(self, envelope: RequestEnvelope, reserved_prompt_tokens: int | None = None) -> Decision:
        traffic_class, clamp = self._preview(envelope)

        if clamp.reason == PROMPT_TOO_LONG:
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_BUDGET,
                f"prompt of {envelope.prompt_tokens:,} tokens leaves no room for output",
                retry_after=None,
            )

        if self._is_offbox(traffic_class):
            return self._admit(envelope, traffic_class, clamp, ("routed off-box",))

        shed = self._breaker.should_shed(traffic_class)
        if shed.distressed:
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_SHED,
                "engine distressed: " + ", ".join(shed.reasons),
            )

        reserved = max(envelope.prompt_tokens, reserved_prompt_tokens or 0)
        borrow_limit = None
        borrowed = False
        if self.workload_budget is not None:
            dynamic_limit = self.workload_budget.borrow_limit(
                traffic_class.name, reserved, clamp.granted, envelope.has_images
            )
            if dynamic_limit > 0:
                borrow_limit = dynamic_limit
                borrowed = (
                    self._budget.in_flight(traffic_class)
                    >= self._budget.limit_for(traffic_class)
                )

        if not self._budget.try_acquire(traffic_class, borrow_limit=borrow_limit):
            limit = max(
                self._budget.limit_for(traffic_class),
                min(self._budget.ceiling, borrow_limit or 0),
            )
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_BUDGET,
                f"{traffic_class.name} at its concurrency limit of {limit}",
            )

        lease = None
        if self.workload_budget is not None:
            try:
                lease, reason = self.workload_budget.acquire(reserved, clamp.granted, envelope.has_images)
            except Exception:
                self._budget.release(traffic_class)
                raise
            if lease is None:
                self._budget.release(traffic_class)
                return self._reject(envelope, traffic_class, clamp, Outcome.REJECT_SHED,
                                    "workload budget: " + reason)
        notes = ("adaptive class borrow",) if borrowed else ()
        return self._admit(envelope, traffic_class, clamp, notes, workload_lease=lease)

    def release(self, decision: Decision) -> None:
        if decision.admitted:
            traffic_class = self._classifier.classify(decision.envelope)
            if not self._is_offbox(traffic_class):
                if decision.workload_lease is not None and self.workload_budget is not None:
                    if not self.workload_budget.release(decision.workload_lease):
                        return
                self._budget.release(traffic_class)

    def _admit(self, envelope, traffic_class, clamp, notes=(), workload_lease=None) -> Decision:
        return Decision(
            envelope=envelope,
            traffic_class=traffic_class.name,
            priority=traffic_class.priority,
            clamp=clamp,
            outcome=Outcome.ADMIT,
            reason="admitted",
            notes=tuple(notes),
            workload_lease=workload_lease,
        )

    def _reject(
        self, envelope, traffic_class, clamp, outcome, reason,
        retry_after: int | None = DEFAULT_RETRY_AFTER_SECONDS,
    ) -> Decision:
        return Decision(
            envelope=envelope,
            traffic_class=traffic_class.name,
            priority=traffic_class.priority,
            clamp=clamp,
            outcome=outcome,
            reason=reason,
            retry_after_seconds=retry_after,
        )


def build_default(
    max_model_len: int,
    concurrency_ceiling: int,
    offbox_configured: bool = False,
) -> GatewayPolicy:
    return GatewayPolicy(
        classifier=Classifier(),
        clamp=TokenClamp(max_model_len=max_model_len),
        budget=ClassBudget.from_classes(
            concurrency_ceiling, ALL_CLASSES, offbox_configured=offbox_configured
        ),
        breaker=CircuitBreaker(StaticHealthSource(EngineSnapshot.healthy())),
        offbox_configured=offbox_configured,
    )
