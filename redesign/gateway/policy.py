"""Orchestrates classification, clamping and backpressure into one decision."""

from __future__ import annotations

from dataclasses import replace

from .backpressure import (
    DEFAULT_RETRY_AFTER_SECONDS,
    ENCODER_RETRY_AFTER_SECONDS,
    BudgetSnapshot,
    CircuitBreaker,
    ClassBudget,
    StaticHealthSource,
)
from .classification import ALL_CLASSES, Classifier
from .clamping import PROMPT_TOO_LONG, TokenClamp
from .models import (
    AdmissionReason, CountSource, Decision, EngineSnapshot, Outcome, RequestEnvelope,
)


class GatewayPolicy:
    def __init__(
        self,
        classifier: Classifier,
        clamp: TokenClamp,
        budget: ClassBudget,
        breaker: CircuitBreaker,
        offbox_configured: bool = False,
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

    def _is_offbox(self, traffic_class) -> bool:
        return traffic_class.served_off_box and self._offbox_configured

    def decide(self, envelope: RequestEnvelope) -> Decision:
        traffic_class = self._classifier.classify(envelope)
        clamp_policy = self._clamp
        if envelope.prompt_count_source == CountSource.ENGINE_RENDERED:
            maximum = min(self._clamp.max_model_len, envelope.model_context_limit or self._clamp.max_model_len)
            clamp_policy = replace(self._clamp, max_model_len=maximum, reserve_tokens=0)
        clamp = clamp_policy.apply(
            envelope.prompt_tokens, envelope.requested_max_tokens, traffic_class
        )

        if clamp.reason == PROMPT_TOO_LONG:
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_BUDGET,
                f"prompt of {envelope.prompt_tokens:,} tokens leaves no room for output",
                retry_after=None,
            )

        if self._is_offbox(traffic_class):
            return self._admit(envelope, traffic_class, clamp, ("routed off-box",))

        snapshot = self._breaker.snapshot()
        shed = self._breaker.should_shed(traffic_class, snapshot)
        if shed.distressed:
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_SHED,
                "engine distressed: " + ", ".join(shed.reasons),
                metadata={"admission_reason": AdmissionReason.PRESSURE},
            )

        admission = self._budget.try_acquire_lease(
            traffic_class,
            context_tokens=max(0, envelope.prompt_tokens) + max(0, clamp.granted),
            snapshot=snapshot,
            has_images=envelope.has_images,
            prompt_tokens=envelope.prompt_tokens,
            prompt_count_source=envelope.prompt_count_source,
        )
        if admission.lease is None:
            metadata: dict[str, object] = {"admission_reason": admission.reason}
            retry_after: int | None = DEFAULT_RETRY_AFTER_SECONDS
            if admission.reason in (AdmissionReason.ENCODER, AdmissionReason.ENCODER_COUNT):
                metadata.update({
                    "encoder_tokens_in_flight": admission.snapshot.encoder_tokens_in_flight,
                    "encoder_token_budget": admission.snapshot.encoder_token_budget,
                })
                if admission.reason == AdmissionReason.ENCODER:
                    retry_after = ENCODER_RETRY_AFTER_SECONDS
                else:
                    # A heuristic cannot be made safe by waiting for capacity.
                    retry_after = None
                    metadata.update({
                        "prompt_count_source": envelope.prompt_count_source,
                        "required_prompt_count_source": CountSource.ENGINE_RENDERED,
                    })
            return self._reject(
                envelope, traffic_class, clamp, Outcome.REJECT_BUDGET,
                admission.detail,
                retry_after=retry_after,
                metadata=metadata,
            )

        lease = admission.lease
        gauges = admission.snapshot
        notes = ()
        if lease.borrowed:
            notes = (
                f"borrowed class capacity: {gauges.borrowed_by_class[lease.traffic_class]} "
                f"class / {gauges.borrowed_in_flight} total borrowed in flight",
            )
        return self._admit(
            envelope, traffic_class, clamp, notes,
            admission_id=lease.admission_id,
            metadata={
                "borrowed": lease.borrowed,
                "context_tokens": lease.context_tokens,
                "in_flight": gauges.in_flight,
                "class_in_flight": gauges.by_class[lease.traffic_class],
                "borrowed_in_flight": gauges.borrowed_in_flight,
                "class_borrowed_in_flight": gauges.borrowed_by_class[lease.traffic_class],
                "context_tokens_in_flight": gauges.context_tokens_in_flight,
                "effective_context_token_budget": gauges.effective_context_token_budget,
                "encoder_tokens": lease.encoder_tokens,
                "encoder_tokens_in_flight": gauges.encoder_tokens_in_flight,
                "encoder_token_budget": gauges.encoder_token_budget,
            },
        )

    def prefill_complete(self, decision: Decision) -> bool:
        """Parent calls after the first meaningful streaming delta proves prefill.

        Headers, heartbeats and role-only deltas are not proof. Non-streaming
        requests conservatively retain encoder work until terminal release().
        Does not release the request's concurrency, context or borrowed slots.
        """
        if (
            decision.admitted and decision.admission_id is not None
            and decision.envelope.streaming
        ):
            return self._budget.release_encoder_lease(decision.admission_id)
        return False

    def release(self, decision: Decision) -> None:
        """Idempotent for local admissions, including copies of a decision."""
        if decision.admitted and decision.admission_id is not None:
            self._budget.release_lease(decision.admission_id)

    def budget_snapshot(self) -> BudgetSnapshot:
        """Safe gauges for metrics integration; does not collect engine health."""
        return self._budget.snapshot()

    def _admit(
        self, envelope, traffic_class, clamp, notes=(), *,
        admission_id: str | None = None, metadata: dict[str, object] | None = None,
    ) -> Decision:
        return Decision(
            envelope=envelope,
            traffic_class=traffic_class.name,
            priority=traffic_class.priority,
            clamp=clamp,
            outcome=Outcome.ADMIT,
            reason="admitted",
            notes=tuple(notes),
            admission_id=admission_id,
            metadata=metadata if metadata is not None else {},
        )

    def _reject(
        self, envelope, traffic_class, clamp, outcome, reason,
        retry_after: int | None = DEFAULT_RETRY_AFTER_SECONDS,
        *, metadata: dict[str, object] | None = None,
    ) -> Decision:
        return Decision(
            envelope=envelope,
            traffic_class=traffic_class.name,
            priority=traffic_class.priority,
            clamp=clamp,
            outcome=outcome,
            reason=reason,
            retry_after_seconds=retry_after,
            metadata=metadata if metadata is not None else {},
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
