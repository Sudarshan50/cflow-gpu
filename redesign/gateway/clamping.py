"""Bounds max_tokens so the engine does not reject the request.

Register item D2, and the largest single defect in the observed traffic: 75 of
651 requests (11.5%) returned 400 because the engine reserves
prompt + max_tokens against one shared window, and clients send a fixed
max_tokens regardless of prompt length.

Clamping is not a workaround. The reservation is real, so a request asking for
more output than the window can hold was never servable; the only question is
whether the caller learns that as a 400 or as a shorter completion.
"""

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

# Below this, a completion is not worth serving; reject rather than truncate.
MIN_VIABLE_OUTPUT_TOKENS = 64


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
        exceeded = self.exceeds_window(prompt_tokens, requested)
        available = self.available_for_output(prompt_tokens)
        if available < self.min_viable_output_tokens:
            return ClampResult(
                granted=0,
                requested=requested,
                reason=PROMPT_TOO_LONG,
                exceeded_window=exceeded,
            )

        if requested is None:
            granted = min(traffic_class.max_output_tokens, available)
            reason = APPLIED_DEFAULT
        elif requested <= traffic_class.max_output_tokens and requested <= available:
            granted, reason = requested, UNCHANGED
        elif traffic_class.max_output_tokens <= available:
            granted, reason = traffic_class.max_output_tokens, CLASS_CEILING
        else:
            granted, reason = available, CONTEXT_WINDOW

        return ClampResult(
            granted=granted,
            requested=requested,
            reason=reason,
            exceeded_window=exceeded,
        )
