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


class GatewayPolicy:
    def __init__(
        self,
        classifier: Classifier,
        clamp: TokenClamp,
        budget: ClassBudget,
        breaker: CircuitBreaker,
    ) -> None:
        self._classifier = classifier
        self._clamp = clamp
        self._budget = budget
        self._breaker = breaker

    def decide(self, envelope: RequestEnvelope) -> Decision:
        traffic_class = self._classifier.classify(envelope)
        clamp = self._clamp.apply(
            envelope.prompt_tokens, envelope.requested_max_tokens, traffic_class
        )

        if clamp.reason == PROMPT_TOO_LONG:
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_BUDGET,
                f"prompt of {envelope.prompt_tokens:,} tokens leaves no room for output",
                retry_after=None,
            )

        if traffic_class.served_off_box:
            return self._admit(envelope, traffic_class, clamp, ("routed off-box",))

        shed = self._breaker.should_shed(traffic_class)
        if shed.distressed:
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_SHED,
                "engine distressed: " + ", ".join(shed.reasons),
            )

        if not self._budget.try_acquire(traffic_class):
            limit = self._budget.limit_for(traffic_class)
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_BUDGET,
                f"{traffic_class.name} at its concurrency limit of {limit}",
            )

        return self._admit(envelope, traffic_class, clamp)

    def release(self, decision: Decision) -> None:
        if decision.admitted:
            traffic_class = self._classifier.classify(decision.envelope)
            if not traffic_class.served_off_box:
                self._budget.release(traffic_class)

    def _admit(self, envelope, traffic_class, clamp, notes=()) -> Decision:
        return Decision(
            envelope=envelope,
            traffic_class=traffic_class.name,
            priority=traffic_class.priority,
            clamp=clamp,
            outcome=Outcome.ADMIT,
            reason="admitted",
            notes=tuple(notes),
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


def build_default(max_model_len: int, concurrency_ceiling: int) -> GatewayPolicy:
    return GatewayPolicy(
        classifier=Classifier(),
        clamp=TokenClamp(max_model_len=max_model_len),
        budget=ClassBudget.from_classes(concurrency_ceiling, ALL_CLASSES),
        breaker=CircuitBreaker(StaticHealthSource(EngineSnapshot.healthy())),
    )
