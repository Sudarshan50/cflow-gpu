"""Choose defaults or preserve explicit output budgets within class/context bounds."""

from __future__ import annotations

from dataclasses import dataclass

from .classification import TrafficClass
from .models import ClampResult

UNCHANGED = "unchanged"
APPLIED_DEFAULT = "default"
CLASS_CEILING = "class_ceiling"
CONTEXT_WINDOW = "context_window"
PROMPT_TOO_LONG = "prompt_exceeds_window"

# Headroom against the engine's own accounting, which counts template and
# special tokens the gateway does not see.
RESERVE_TOKENS = 256

# Do not truncate a larger budget below this. Explicit smaller budgets that
# fit are still useful (e.g. max_tokens=1) and must not require 64 free tokens.
MIN_VIABLE_OUTPUT_TOKENS = 64


class InvalidTokenLimit(ValueError):
    """Invalid caller limits must become HTTP 400, never an implicit default."""


def _validate_limit(value: object, field: str) -> None:
    # Match responses_bridge's strict positive-integer validation; bool is not
    # a token count even though isinstance(True, int) is true in Python.
    if value is not None and (type(value) is not int or value <= 0):
        raise InvalidTokenLimit(f"{field}: must be a positive integer")


def validate_token_limits(payload: dict) -> None:
    for field in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
        _validate_limit(payload.get(field), field)


def requested_output_tokens(payload: dict) -> int | None:
    """Read canonical Chat limits at this hop, after any Responses conversion.

    vLLM's modern alias wins. Validate both aliases before selecting so an
    invalid value cannot disappear into the default/fallback path.
    """
    validate_token_limits(payload)
    modern = payload.get("max_completion_tokens")
    return modern if modern is not None else payload.get("max_tokens")


@dataclass(frozen=True)
class TokenClamp:
    max_model_len: int
    reserve_tokens: int = RESERVE_TOKENS
    min_viable_output_tokens: int = MIN_VIABLE_OUTPUT_TOKENS

    def available_for_output(self, prompt_tokens: int) -> int:
        return self.max_model_len - prompt_tokens - self.reserve_tokens

    def exceeds_window(self, prompt_tokens: int, requested: int | None) -> bool:
        """What an unguarded engine rejects: prompt + max_tokens past the window."""
        if requested is None:
            return False
        return prompt_tokens + requested > self.max_model_len

    def apply(
        self,
        prompt_tokens: int,
        requested: int | None,
        traffic_class: TrafficClass,
    ) -> ClampResult:
        _validate_limit(requested, "max_tokens")
        exceeded = self.exceeds_window(prompt_tokens, requested)
        available = self.available_for_output(prompt_tokens)
        default = traffic_class.default_output_tokens
        if default is None:
            default = traffic_class.max_output_tokens
        target = default if requested is None else requested
        minimum = min(self.min_viable_output_tokens, target, traffic_class.max_output_tokens)
        if available < minimum:
            return ClampResult(
                granted=0,
                requested=requested,
                reason=PROMPT_TOO_LONG,
                exceeded_window=exceeded,
                default_output_tokens=default,
            )

        granted = min(target, traffic_class.max_output_tokens, available)
        if granted < target:
            reason = CLASS_CEILING if traffic_class.max_output_tokens <= available else CONTEXT_WINDOW
        else:
            reason = APPLIED_DEFAULT if requested is None else UNCHANGED

        return ClampResult(
            granted=granted,
            requested=requested,
            reason=reason,
            exceeded_window=exceeded,
            default_output_tokens=default,
        )
