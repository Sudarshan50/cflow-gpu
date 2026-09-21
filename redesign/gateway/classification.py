"""Assigns a traffic class. Rules are ordered; the first match wins.

Classes are defined in docs/SYSTEM-DESIGN.md 5. Add a class by appending a
ClassRule to DEFAULT_RULES.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from .models import Priority, RequestEnvelope

INTERACTIVE_CEILING_TOKENS = 32_768
SHORT_CHAT_CEILING_TOKENS = 8_192


@dataclass(frozen=True)
class TrafficClass:
    name: str
    priority: Priority
    max_output_tokens: int
    ttft_target_seconds: float | None
    kv_budget_share: float
    served_off_box: bool = False
    # None retains the historical max-as-default for custom/positional classes.
    default_output_tokens: int | None = None


INTERACTIVE = TrafficClass("P0-interactive", Priority.INTERACTIVE, 8_192, 3.0, 0.50,
                           default_output_tokens=2_048)
SHORT_CHAT = TrafficClass("P1-short-chat", Priority.SHORT_CHAT, 4_096, 1.0, 0.0,
                          served_off_box=True, default_output_tokens=1_024)
LONG_CONTEXT = TrafficClass("P2-long-context", Priority.LONG_CONTEXT, 16_384, 60.0, 0.25,
                            default_output_tokens=4_096)
BATCH = TrafficClass("P3-batch", Priority.BATCH, 32_768, None, 0.25,
                     default_output_tokens=8_192)

# Tool/vision requests use LONG_CONTEXT priority so overload protection can shed them.
AGENTIC = TrafficClass("P2-agentic", Priority.LONG_CONTEXT, 16_384, 15.0, 0.45,
                       default_output_tokens=2_048)

ALL_CLASSES = (INTERACTIVE, SHORT_CHAT, LONG_CONTEXT, BATCH, AGENTIC)


class ClassRule(abc.ABC):
    def __init__(self, traffic_class: TrafficClass) -> None:
        self.traffic_class = traffic_class

    @abc.abstractmethod
    def matches(self, envelope: RequestEnvelope) -> bool: ...


class BatchHintRule(ClassRule):
    """An explicit batch header wins over every length heuristic."""

    def matches(self, envelope: RequestEnvelope) -> bool:
        return envelope.batch_hint


class AgenticRule(ClassRule):
    """Enabled tools or images, at any prompt length."""

    def matches(self, envelope: RequestEnvelope) -> bool:
        return envelope.has_images or (envelope.has_tools and not envelope.tools_disabled)


class ShortChatRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return (
            envelope.prompt_tokens <= SHORT_CHAT_CEILING_TOKENS
            and (not envelope.has_tools or envelope.tools_disabled)
        )


class InteractiveRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return envelope.prompt_tokens <= INTERACTIVE_CEILING_TOKENS


class LongContextRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return True


DEFAULT_RULES: tuple[ClassRule, ...] = (
    BatchHintRule(BATCH),
    AgenticRule(AGENTIC),
    ShortChatRule(SHORT_CHAT),
    InteractiveRule(INTERACTIVE),
    LongContextRule(LONG_CONTEXT),
)


class Classifier:
    def __init__(self, rules: tuple[ClassRule, ...] = DEFAULT_RULES) -> None:
        if not rules:
            raise ValueError("at least one rule is required")
        self._rules = rules

    def classify(self, envelope: RequestEnvelope) -> TrafficClass:
        for rule in self._rules:
            if rule.matches(envelope):
                return rule.traffic_class
        raise ValueError("no rule matched; the last rule must be a catch-all")
