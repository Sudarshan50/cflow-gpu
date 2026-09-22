"""Class assignment and max_tokens clamp for the LiteLLM hop."""

from __future__ import annotations

from dataclasses import dataclass

from redesign.gateway.classification import Classifier
from redesign.gateway.clamping import PROMPT_TOO_LONG, TokenClamp
from redesign.gateway.models import RequestEnvelope
from redesign.gateway.server import apply_granted_tokens
from redesign.gateway.tokens import build_estimator, has_images
from redesign.gateway.inspection import inspectable

OFFBOX_MODEL = "p1-offbox"


@dataclass(frozen=True)
class TenancyDecision:
    traffic_class: str
    priority: int
    granted_max_tokens: int
    model: str
    reject: str = ""
    routed_off_box: bool = False


class TenancyPolicy:
    def __init__(
        self,
        max_model_len: int = 262_144,
        offbox_configured: bool = False,
        estimator=None,
        classifier: Classifier | None = None,
        clamp: TokenClamp | None = None,
        defer_local_context_check: bool = False,
    ) -> None:
        self.offbox_configured = offbox_configured
        self.estimator = estimator or build_estimator(None)
        self.classifier = classifier or Classifier()
        self.clamp = clamp or TokenClamp(max_model_len=max_model_len)
        self.defer_local_context_check = defer_local_context_check

    def decide(self, payload: dict, original_model: str) -> TenancyDecision:
        # Match vLLM and GatewayService.envelope: the modern alias wins.
        # Selecting max_tokens first can expand a smaller max_completion_tokens
        # before the gateway ever sees it. apply_granted_tokens synchronizes
        # both aliases only AFTER clamping this effective request limit.
        requested = payload.get("max_completion_tokens")
        if not isinstance(requested, int) or requested <= 0:
            requested = payload.get("max_tokens")
            if not isinstance(requested, int) or requested <= 0:
                requested = None
        envelope = RequestEnvelope(
            customer="litellm",
            prompt_tokens=self.estimator.estimate(payload),
            requested_max_tokens=requested,
            streaming=bool(payload.get("stream")),
            has_tools=bool(payload.get("tools") or payload.get("functions")),
            has_images=has_images(payload),
            batch_hint=bool(payload.get("k3_batch")),
        )
        traffic_class = self.classifier.classify(envelope)
        offbox = traffic_class.served_off_box and self.offbox_configured
        context_tokens = envelope.prompt_tokens
        if self.defer_local_context_check and not offbox and inspectable(payload):
            # The gateway validates the rendered prompt before local inference.
            # This hop still applies the existing class/output allowance.
            context_tokens = 0
        clamp = self.clamp.apply(
            context_tokens, envelope.requested_max_tokens, traffic_class
        )
        if clamp.reason == PROMPT_TOO_LONG:
            return TenancyDecision(
                traffic_class.name,
                int(traffic_class.priority),
                0,
                original_model,
                reject=f"prompt of {envelope.prompt_tokens:,} tokens leaves no room for output",
            )
        return TenancyDecision(
            traffic_class.name,
            int(traffic_class.priority),
            clamp.granted,
            OFFBOX_MODEL if offbox else original_model,
            routed_off_box=offbox,
        )

    def apply(self, payload: dict, original_model: str) -> TenancyDecision:
        decision = self.decide(payload, original_model)
        if not decision.reject:
            apply_granted_tokens(payload, decision.granted_max_tokens)
            payload["model"] = decision.model
        return decision
