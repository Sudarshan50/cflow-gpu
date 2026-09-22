"""Rendered text counts and bounded, process-keyed prefix diagnostics.

No prompt text, token IDs, response text, or credentials are retained in the
diagnostic history. A seen fingerprint establishes prior completion, not cache
residency. Multimodal and nonstandard requests use conservative reservations.
"""
from __future__ import annotations

import array
import hashlib
import hmac
import json
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from dataclasses import dataclass

CHAT_FIELDS = ("model", "messages", "tools", "chat_template", "chat_template_kwargs",
               "add_generation_prompt", "continue_final_message", "add_special_tokens")
UNINSPECTED_FIELDS = ("prompt_embeds", "prompt_embedding", "multi_modal_data", "multi_modal_uuids",
                     "truncate_prompt_tokens", "documents", "functions", "function_call")


def tokenize_projection(payload: dict) -> dict:
    fields = CHAT_FIELDS if "messages" in payload else ("model", "prompt", "add_special_tokens")
    body = {key: payload[key] for key in fields if key in payload}
    if "messages" in body:
        messages = []
        for original in body["messages"]:
            message = dict(original)
            # ChatCompletionRequest normalizes this alias before validating its
            # typed message union. TokenizeChatRequest does not, so it would
            # silently discard preserved thinking history without this copy.
            reasoning = message.pop("reasoning_content", None)
            if reasoning is not None and message.get("reasoning") is None:
                message["reasoning"] = reasoning
            messages.append(message)
        body["messages"] = messages
        kwargs = dict(body.get("chat_template_kwargs") or {})
        effort = payload.get("reasoning_effort")
        if effort not in (None, "auto"):
            kwargs["reasoning_effort"] = effort
        if effort is not None and "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = effort != "none"
        if kwargs:
            body["chat_template_kwargs"] = kwargs
    body["return_token_strs"] = False
    return body


def inspectable(payload: dict) -> bool:
    if payload.get("messages") is not None and payload.get("prompt") is not None:
        return False
    if any(payload.get(key) is not None for key in UNINSPECTED_FIELDS):
        return False
    # /tokenize has no equivalents for generation-side constrained/forced
    # prefixes. Preserve those requests through the conservative path.
    if payload.get("tool_choice") not in (None, "auto") or payload.get("response_format") is not None:
        return False
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                return False
            if message.get("tools") is not None or message.get("task") is not None:
                return False
            content = message.get("content")
            if content is None or isinstance(content, str):
                continue
            if not isinstance(content, list) or any(
                not isinstance(part, dict) or part.get("type") not in ("text", "input_text")
                for part in content
            ):
                return False
        return True
    prompt = payload.get("prompt")
    return isinstance(prompt, str) or (isinstance(prompt, list) and bool(prompt)
            and all(type(token) is int and 0 <= token < 2**32 for token in prompt))


class InspectionError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(reason)
        self.status, self.reason, self.public_message = status, reason, message


@dataclass(frozen=True)
class Inspection:
    tokens: int
    scope: str
    prefix_768: str | None
    prefix_3840: str | None
    replay_boundary: int
    replay_fingerprint: str | None
    prior_completed_prefix_tokens: int


class PromptInspector:
    def __init__(self, engine_url: str, *, timeout: float = 10, max_inflight: int = 2):
        self.url = engine_url.rstrip("/") + "/tokenize"
        self.timeout = timeout
        self._slots = threading.BoundedSemaphore(max_inflight)
        self._lock = threading.Lock()
        self._secret = secrets.token_bytes(32)
        self._scope = secrets.token_hex(8)
        self._epoch = None
        self._quarantined = False
        self._completed: OrderedDict[tuple[str, str], tuple[int, float]] = OrderedDict()
        self._events = deque(maxlen=1024)

    def inspect(self, payload: dict, cache_epoch: float | None = None) -> Inspection | None:
        with self._lock:
            if self._quarantined:
                return None
        if not inspectable(payload):
            return None
        if not self._slots.acquire(timeout=.5):
            raise InspectionError(503, "tokenizer_busy", "Prompt inspection is busy; retry shortly")
        try:
            if isinstance(payload.get("prompt"), list) and "messages" not in payload:
                tokens = payload["prompt"]
            else:
                body = tokenize_projection(payload)
                request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        raw = response.read(16 * 1024 * 1024 + 1)
                    if len(raw) > 16 * 1024 * 1024:
                        raise ValueError("response size")
                    result = json.loads(raw)
                    tokens = result.get("tokens")
                    if not isinstance(tokens, list) or result.get("count") != len(tokens):
                        raise ValueError("token count")
                except urllib.error.HTTPError as exc:
                    status = 400 if 400 <= exc.code < 500 else 503
                    raise InspectionError(status, "tokenizer_rejected" if status == 400 else "tokenizer_unavailable",
                                          f"Active model tokenizer returned HTTP {exc.code}") from None
                except (OSError, ValueError, TypeError):
                    raise InspectionError(503, "tokenizer_unavailable", "Prompt inspection is temporarily unavailable") from None
            if not all(type(token) is int and 0 <= token < 2**32 for token in tokens):
                raise InspectionError(503, "invalid_tokenizer_response", "Invalid prompt inspection response")
            result = self._fingerprint(tokens, str(payload.get("model") or "FW-Kimi-K3"), cache_epoch)
            with self._lock:
                return None if self._quarantined else result
        finally:
            self._slots.release()

    def _fingerprint(self, tokens, model: str, epoch) -> Inspection:
        packed = array.array("I", tokens)
        if sys.byteorder != "little":
            packed.byteswap()
        raw = memoryview(packed).cast("B")
        now = time.monotonic()
        with self._lock:
            if epoch is not None and self._epoch is not None and epoch != self._epoch:
                self._completed.clear()
                self._secret = secrets.token_bytes(32)
                self._scope = secrets.token_hex(8)
            if epoch is not None:
                self._epoch = epoch
            while self._completed and next(iter(self._completed.values()))[1] < now - 900:
                self._completed.popitem(last=False)
            secret, scope, completed = self._secret, self._scope, dict(self._completed)
        hasher = hmac.new(secret, model.encode() + b"\x00", hashlib.sha256)
        prefix_768 = prefix_3840 = replay = None
        boundary = (len(tokens) - 1) // 768 * 768 if tokens else 0
        seen = 0
        for end in range(768, len(tokens) + 1, 768):
            hasher.update(raw[(end - 768) * 4:end * 4])
            fingerprint = hasher.copy().hexdigest()
            if end == 768:
                prefix_768 = fingerprint
            if end == 3840:
                prefix_3840 = fingerprint
            if end == boundary:
                replay = fingerprint
            record = completed.get((scope, fingerprint))
            if record is not None and record[1] >= now - 900:
                seen = max(seen, record[0])
        return Inspection(len(tokens), scope, prefix_768, prefix_3840, boundary, replay, seen)

    def record(self, inspection: Inspection | None, *, http_status: int, usage: dict | None,
               complete: bool, outcome: str, granted: int):
        prompt = usage.get("prompt_tokens") if usage else None
        details = (usage.get("prompt_tokens_details") or {}) if usage else {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        verified = (inspection is not None and complete and 200 <= http_status < 300
                    and prompt == inspection.tokens and type(cached) is int and 0 <= cached <= prompt)
        event = {"timestamp": time.time(), "outcome": outcome, "http_status": http_status,
                 "count_mode": "engine" if inspection else "conservative",
                 "prompt_tokens": inspection.tokens if inspection else None,
                 "actual_prompt_tokens": prompt if type(prompt) is int else None,
                 "cached_tokens": cached if type(cached) is int else None,
                 "completion_tokens": usage.get("completion_tokens") if usage and type(usage.get("completion_tokens")) is int else None,
                 "granted_max_tokens": granted, "complete": complete, "verified_usage": verified}
        if inspection:
            event.update(scope=inspection.scope, prefix_768=inspection.prefix_768,
                         prefix_3840=inspection.prefix_3840, replay_boundary=inspection.replay_boundary,
                         replay_fingerprint=inspection.replay_fingerprint,
                         prior_completed_prefix_tokens=inspection.prior_completed_prefix_tokens)
        with self._lock:
            self._events.append(event)
            if (inspection is not None and complete and 200 <= http_status < 300
                    and type(prompt) is int and prompt != inspection.tokens):
                self._quarantined = True
                self._completed.clear()
            # A non-block-aligned prompt forces its final full checkpoint; an
            # exactly block-aligned prompt has the previously observed replay gap.
            if (verified and not self._quarantined and inspection.scope == self._scope and inspection.tokens % 768
                    and inspection.replay_fingerprint):
                key = (inspection.scope, inspection.replay_fingerprint)
                self._completed.pop(key, None)
                self._completed[key] = (inspection.replay_boundary, time.monotonic())
                while len(self._completed) > 4096:
                    self._completed.popitem(last=False)
        return event

    def recent(self):
        with self._lock:
            return {"scope": self._scope, "events": list(self._events), "history_entries": len(self._completed),
                    "counting_quarantined": self._quarantined}

    @property
    def quarantined(self) -> bool:
        with self._lock:
            return self._quarantined


class UsageObserver:
    """Read only terminal usage; never delay or change the relayed bytes."""
    def __init__(self, streaming: bool):
        self.streaming = streaming
        self.buffer = bytearray()
        self.usage = None
        self.finished = self.done = self.failed = False
        self.overflow = False

    def feed(self, chunk: bytes):
        if self.overflow:
            return
        self.buffer.extend(chunk)
        if self.streaming:
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    self.done = True
                else:
                    self._event(data)
        if len(self.buffer) > 1024 * 1024:
            self.buffer.clear()
            self.overflow = True

    def _event(self, raw):
        try:
            event = json.loads(raw)
            if not isinstance(event, dict):
                return
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
                details = usage.get("prompt_tokens_details") or {}
                self.usage = {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens")}
                self.usage["prompt_tokens_details"] = {
                    "cached_tokens": details.get("cached_tokens") if isinstance(details, dict) else None}
            self.finished |= any(choice.get("finish_reason") for choice in event.get("choices", []) if isinstance(choice, dict))
            self.failed |= bool(event.get("error"))
        except (ValueError, TypeError):
            self.failed = True

    def finish(self) -> bool:
        if not self.streaming and not self.overflow:
            self._event(self.buffer)
            self.done = True
        self.buffer.clear()
        return bool(self.done and self.finished and self.usage and not self.failed and not self.overflow)
