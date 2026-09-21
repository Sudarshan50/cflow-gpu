"""Class assignment and max_tokens clamp for the LiteLLM hop."""

from __future__ import annotations

from dataclasses import dataclass, replace

from redesign.gateway.classification import Classifier
from redesign.gateway.clamping import PROMPT_TOO_LONG, TokenClamp, requested_output_tokens
from redesign.gateway.models import RequestEnvelope
from redesign.gateway.server import apply_granted_tokens
from redesign.gateway.tokens import build_estimator, has_images

OFFBOX_MODEL = "p1-offbox"


@dataclass(frozen=True)
class TenancyDecision:
    """Hop-local limits, not a claim about the original caller's budget.

    At the deployment hook, requested_max_tokens includes Router defaults and
    Responses conversion. On reapplication it is the already-clamped input.
    """

    traffic_class: str
    priority: int
    granted_max_tokens: int
    model: str
    reject: str = ""
    routed_off_box: bool = False
    requested_max_tokens: int | None = None
    default_output_tokens: int | None = None
    clamp_reason: str = ""

    @property
    def default_applied(self) -> bool:
        return self.requested_max_tokens is None


class TenancyPolicy:
    def __init__(
        self,
        max_model_len: int = 262_144,
        offbox_configured: bool = False,
        estimator=None,
        classifier: Classifier | None = None,
        clamp: TokenClamp | None = None,
    ) -> None:
        self.offbox_configured = offbox_configured
        self.estimator = estimator or build_estimator(None)
        self.classifier = classifier or Classifier()
        self.clamp = clamp or TokenClamp(max_model_len=max_model_len)

    def decide(self, payload: dict, original_model: str) -> TenancyDecision:
        # Match vLLM and GatewayService.envelope: the modern alias wins.
        # Selecting max_tokens first can expand a smaller max_completion_tokens
        # before the gateway ever sees it. apply_granted_tokens synchronizes
        # both aliases only AFTER clamping this effective request limit.
        requested = requested_output_tokens(payload)
        envelope = RequestEnvelope(
            customer="litellm",
            prompt_tokens=self.estimator.estimate(payload),
            requested_max_tokens=requested,
            streaming=bool(payload.get("stream")),
            has_tools=bool(payload.get("tools") or payload.get("functions")),
            has_images=has_images(payload),
            batch_hint=bool(payload.get("k3_batch")),
            tools_disabled=(payload.get("tool_choice") == "none" or payload.get("function_call") == "none"),
        )
        traffic_class = self.classifier.classify(envelope)
        clamp = self.clamp.apply(
            envelope.prompt_tokens, envelope.requested_max_tokens, traffic_class
        )
        reject = (
            f"prompt of {envelope.prompt_tokens:,} tokens leaves no room for output"
            if clamp.reason == PROMPT_TOO_LONG else ""
        )
        offbox = not reject and traffic_class.served_off_box and self.offbox_configured
        return TenancyDecision(
            traffic_class.name,
            int(traffic_class.priority),
            clamp.granted,
            OFFBOX_MODEL if offbox else original_model,
            reject=reject,
            routed_off_box=offbox,
            requested_max_tokens=clamp.requested,
            default_output_tokens=clamp.default_output_tokens,
            clamp_reason=clamp.reason,
        )

    def apply(self, payload: dict, original_model: str, *, route_model: bool = True) -> TenancyDecision:
        decision = self.decide(payload, original_model)
        if not route_model:
            # A deployment hook is past Router selection: changing an alias
            # would still use the already selected API base/client. The gateway
            # owns off-box dispatch at this point; do not claim we rerouted it.
            decision = replace(decision, model=original_model, routed_off_box=False)
        if not decision.reject:
            apply_granted_tokens(payload, decision.granted_max_tokens)
            payload["model"] = decision.model
        return decision
