"""Prompt-length estimation for the clamp.

The estimate is deliberately conservative: over-counting cuts harder than
needed; under-counting recreates the 400s the clamp exists to prevent.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Protocol
from unicodedata import east_asian_width

from .media import png_size

# English averages roughly 4 characters per token. 3.5 biases the estimate high.
CONSERVATIVE_CHARS_PER_TOKEN = 3.5

# CJK / fullwidth characters tokenize at ~1 token each.
CJK_CHARS_PER_TOKEN = 1.0

# Measured on live portal traffic 2026-09-20: prompt_tokens_details reported
# 3,764 / 9,410 / 13,174 multimodal tokens per request. 1024 under-counted far
# enough to classify 60k-token vision turns as interactive. 4096 is the floor
# when the image size is unknown; known dimensions bill at ~192 px/token,
# capped at the encoder-cache budget so a 4K screenshot cannot hide as P0.
IMAGE_TOKENS = 4096
IMAGE_TOKENS_MAX = 16_384
IMAGE_PIXELS_PER_TOKEN = 192

# Per-message framing the chat template adds and the character count misses.
PER_MESSAGE_OVERHEAD_TOKENS = 4


class TokenEstimator(Protocol):
    def estimate(self, payload: dict) -> int: ...


@dataclass(frozen=True)
class PromptParts:
    text: str
    messages: int
    images: int
    token_ids: int
    image_tokens: int = 0


def _is_cjk(char: str) -> bool:
    if east_asian_width(char) in {"W", "F"}:
        return True
    code = ord(char)
    return (
        0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK unified
        or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility
        or 0x20000 <= code <= 0x2CEAF  # CJK extensions B-F
    )


def _text_tokens(text: str, chars_per_token: float) -> int:
    cjk = sum(1 for char in text if _is_cjk(char))
    rest = len(text) - cjk
    return int(cjk / CJK_CHARS_PER_TOKEN) + int(rest / chars_per_token)


def _is_image_part(part: dict) -> bool:
    part_type = part.get("type")
    return (
        part_type in {"image_url", "image", "input_image"}
        or "image_url" in part
        or "image" in part
    )


def tokens_for_image_size(width: int, height: int) -> int:
    billed = (width * height) // IMAGE_PIXELS_PER_TOKEN
    return max(IMAGE_TOKENS, min(IMAGE_TOKENS_MAX, billed))


def tokens_for_image_part(part: dict) -> int:
    """Bill a known PNG from its IHDR; otherwise the conservative floor."""
    url = _image_url(part)
    size = _png_size_from_data_url(url) if url else None
    if size:
        return tokens_for_image_size(*size)
    return IMAGE_TOKENS


def _image_url(part: dict) -> str:
    for key in ("image_url", "image", "input_image"):
        value = part.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for inner in ("url", "data", "image_url"):
                item = value.get(inner)
                if isinstance(item, str) and item.strip():
                    return item.strip()
    source = part.get("source")
    if isinstance(source, dict):
        data = source.get("data")
        if isinstance(data, str) and data.startswith("data:"):
            return data
    return ""


def _png_size_from_data_url(url: str) -> tuple[int, int] | None:
    if not url.startswith("data:") or "base64" not in url[:80].lower():
        return None
    _, _, payload = url.partition(",")
    prefix = "".join(payload[:64].split())
    pad = "=" * (-len(prefix) % 4)
    try:
        raw = base64.b64decode(prefix + pad, validate=False)
    except (ValueError, TypeError):
        return None
    return png_size(raw)


def extract_text(payload: dict) -> PromptParts:
    chunks: list[str] = []
    images = 0
    image_tokens = 0
    token_ids = 0
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
                    if not isinstance(part, dict):
                        continue
                    if _is_image_part(part):
                        images += 1
                        image_tokens += tokens_for_image_part(part)
                    if isinstance(part.get("text"), str):
                        chunks.append(part["text"])
            for key in ("name", "role"):
                if isinstance(message.get(key), str):
                    chunks.append(message[key])
            for key in ("tool_calls", "function_call"):
                if message.get(key):
                    chunks.append(json.dumps(message[key]))
        count = len(messages)
    else:
        prompt = payload.get("prompt")
        if isinstance(prompt, str):
            chunks.append(prompt)
        elif isinstance(prompt, list):
            for item in prompt:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, int):
                    token_ids += 1
        count = 1

    for key in ("tools", "functions"):
        if payload.get(key):
            chunks.append(json.dumps(payload[key]))

    return PromptParts("".join(chunks), count, images, token_ids, image_tokens)


def has_images(payload: dict) -> bool:
    """Classification only, so it stops at the first image.

    extract_text concatenates every prompt chunk; at a 30k-token mean that is
    not worth paying twice per request.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and _is_image_part(part):
                return True
    return False


class HeuristicEstimator:
    def __init__(
        self,
        chars_per_token: float = CONSERVATIVE_CHARS_PER_TOKEN,
        per_message_overhead: int = PER_MESSAGE_OVERHEAD_TOKENS,
        image_tokens: int = IMAGE_TOKENS,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token
        self._per_message_overhead = per_message_overhead
        self._image_tokens = image_tokens

    def estimate(self, payload: dict) -> int:
        parts = extract_text(payload)
        image_bill = parts.image_tokens or parts.images * self._image_tokens
        return (
            _text_tokens(parts.text, self._chars_per_token)
            + parts.messages * self._per_message_overhead
            + image_bill
            + parts.token_ids
        )


class TokenizerEstimator:
    """Tokenize extracted text; template and media overhead remain estimates."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def estimate(self, payload: dict) -> int:
        parts = extract_text(payload)
        encoded = len(self._tokenizer.encode(parts.text)) if parts.text else 0
        image_bill = parts.image_tokens or parts.images * IMAGE_TOKENS
        return (
            encoded
            + parts.messages * PER_MESSAGE_OVERHEAD_TOKENS
            + image_bill
            + parts.token_ids
        )


def build_estimator(model_path: str | None = None) -> TokenEstimator:
    """Use a tokenizer for extracted text when available, otherwise heuristics."""
    if model_path:
        try:
            from transformers import AutoTokenizer

            return TokenizerEstimator(
                AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            )
        except Exception:  # noqa: BLE001 -- any failure means fall back
            pass
    return HeuristicEstimator()
