"""Prompt-length estimation for the clamp.

The clamp needs a prompt length before the engine has seen the request, and the
gateway has no tokenizer by default. The estimate is therefore deliberately
*conservative* -- it over-counts, so the clamp cuts harder than strictly needed.
Under-counting would re-create the 400s the clamp exists to prevent.

Every decision records both the estimate and, once the response arrives, the
engine's own `usage.prompt_tokens`. Z3 calibrates the divisor from that, so the
heuristic is a starting point with a measured correction, not a guess left in
place.
"""

from __future__ import annotations

import json
from typing import Protocol

# English averages roughly 4 characters per token. 3.5 biases the estimate high.
CONSERVATIVE_CHARS_PER_TOKEN = 3.5

# Per-message framing the chat template adds and the character count misses.
PER_MESSAGE_OVERHEAD_TOKENS = 4


class TokenEstimator(Protocol):
    def estimate(self, payload: dict) -> int: ...


def extract_text(payload: dict) -> tuple[str, int]:
    """Returns concatenated prompt text and the number of messages."""
    chunks: list[str] = []
    messages = payload.get("messages")

    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            for key in ("name", "role"):
                if isinstance(message.get(key), str):
                    chunks.append(message[key])
        count = len(messages)
    else:
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            chunks.append(prompt)
        elif isinstance(prompt, list):
            chunks.extend(p for p in prompt if isinstance(p, str))
        count = 1

    # Tool schemas are part of the prompt and are frequently larger than the
    # conversation itself in agentic traffic.
    for key in ("tools", "functions"):
        if payload.get(key):
            chunks.append(json.dumps(payload[key]))

    return "".join(chunks), count


class HeuristicEstimator:
    def __init__(
        self,
        chars_per_token: float = CONSERVATIVE_CHARS_PER_TOKEN,
        per_message_overhead: int = PER_MESSAGE_OVERHEAD_TOKENS,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token
        self._per_message_overhead = per_message_overhead

    def estimate(self, payload: dict) -> int:
        text, messages = extract_text(payload)
        return int(len(text) / self._chars_per_token) + messages * self._per_message_overhead


class TokenizerEstimator:
    """Exact counts when a tokenizer is importable in the deployment image."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def estimate(self, payload: dict) -> int:
        text, messages = extract_text(payload)
        return len(self._tokenizer.encode(text)) + messages * PER_MESSAGE_OVERHEAD_TOKENS


def build_estimator(model_path: str | None = None) -> TokenEstimator:
    """Exact tokenizer when the image provides one, heuristic otherwise."""
    if model_path:
        try:
            from transformers import AutoTokenizer

            return TokenizerEstimator(
                AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            )
        except Exception:  # noqa: BLE001 -- any failure means fall back
            pass
    return HeuristicEstimator()
