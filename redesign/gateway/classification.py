"""Assigns a traffic class. Rules are ordered; the first match wins.

Classes are defined in docs/SYSTEM-DESIGN.md 5. Add a class by appending a
ClassRule to DEFAULT_RULES.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from .models import Priority, RequestEnvelope

SHORT_CHAT_CEILING_TOKENS = 8_192
MEDIUM_CONTEXT_CEILING_TOKENS = 32_768
EXPRESS_OUTPUT_CEILING_TOKENS = 512


@dataclass(frozen=True)
class TrafficClass:
    name: str
    priority: Priority
    max_output_tokens: int
    ttft_target_seconds: float | None
    kv_budget_share: float
    served_off_box: bool = False


INTERACTIVE = TrafficClass("P0-interactive", Priority.INTERACTIVE, 512, 1.0, 0.50)
SHORT_CHAT = TrafficClass("P1-short-chat", Priority.SHORT_CHAT, 2_048, 3.0, 0.0,
                          served_off_box=True)
MEDIUM_CONTEXT = TrafficClass("P2-medium-context", Priority.LONG_CONTEXT, 1_536, 15.0, 0.25)
LONG_CONTEXT = TrafficClass("P2-long-context", Priority.LONG_CONTEXT, 1_536, 60.0, 0.25)
BATCH = TrafficClass("P3-batch", Priority.BATCH, 32_768, None, 0.25)

# Tool/vision requests use LONG_CONTEXT priority so overload protection can shed them.
# 1536 covers a high-effort think plus a short answer or tool call. It stays
# under the 2048 long-output gate, so these turns do not take those scarce slots.
AGENTIC = TrafficClass("P2-agentic", Priority.LONG_CONTEXT, 1536, 15.0, 0.45)

ALL_CLASSES = (INTERACTIVE, SHORT_CHAT, MEDIUM_CONTEXT, LONG_CONTEXT, BATCH, AGENTIC)


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
    """Tools or images, at any prompt length."""

    def matches(self, envelope: RequestEnvelope) -> bool:
        return envelope.has_tools or envelope.has_images


class ShortChatRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return (
            envelope.prompt_tokens <= SHORT_CHAT_CEILING_TOKENS
            and not envelope.has_tools
        )


class InteractiveRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return (
            envelope.prompt_tokens <= SHORT_CHAT_CEILING_TOKENS
            and envelope.requested_max_tokens is not None
            and envelope.requested_max_tokens <= EXPRESS_OUTPUT_CEILING_TOKENS
        )


class MediumContextRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return envelope.prompt_tokens <= MEDIUM_CONTEXT_CEILING_TOKENS


class LongContextRule(ClassRule):
    def matches(self, envelope: RequestEnvelope) -> bool:
        return True


DEFAULT_RULES: tuple[ClassRule, ...] = (
    BatchHintRule(BATCH),
    AgenticRule(AGENTIC),
    InteractiveRule(INTERACTIVE),
    ShortChatRule(SHORT_CHAT),
    MediumContextRule(MEDIUM_CONTEXT),
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
