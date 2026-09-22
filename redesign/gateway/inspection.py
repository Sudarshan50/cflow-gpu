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
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

CHAT_FIELDS = ("model", "messages", "tools", "chat_template", "chat_template_kwargs",
               "add_generation_prompt", "continue_final_message", "add_special_tokens")
UNINSPECTED_FIELDS = ("prompt_embeds", "prompt_embedding", "multi_modal_data", "multi_modal_uuids",
                     "truncate_prompt_tokens", "documents", "functions")
OPAQUE_PROMPT_FIELDS = ("prompt_embeds", "prompt_embedding", "multi_modal_data",
                        "multi_modal_uuids", "documents")


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
    # Kimi's constrained-generation paths add a prefix that /tokenize does not
    # report (47 tokens on the deployed template). Keep those requests on the
    # estimator path; unlike the old behavior, that path no longer reserves the
    # entire model window.
    if (
        payload.get("tool_choice") not in (None, "auto")
        or payload.get("function_call") not in (None, "auto")
        or payload.get("response_format") is not None
    ):
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


def conservative_prompt_reservation(payload: dict, estimated_tokens: int,
                                    max_model_len: int) -> int:
    """Bound unknown prompt KV without charging every request a full window.

    The local estimator already over-counts text, tool schemas, message
    framing, and normalized media. Only opaque fields whose contents it cannot
    inspect retain the full-window fallback.
    """
    prompt = payload.get("prompt")
    batched_token_prompts = (
        isinstance(prompt, list)
        and bool(prompt)
        and all(isinstance(item, list) for item in prompt)
    )
    if batched_token_prompts or any(
        payload.get(key) is not None for key in OPAQUE_PROMPT_FIELDS
    ):
        batch = (
            len(prompt)
            if isinstance(prompt, list) and prompt
            and not all(type(item) is int for item in prompt)
            else 1
        )
        return max_model_len * batch
    return max(0, estimated_tokens)


class InspectionError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        super().__init__(reason)
        self.status, self.reason, self.public_message = status, reason, message


@dataclass(frozen=True)
class Inspection:
    tokens: int
    scope: str
    cache_block_size: int
    prefix_first_block: str | None
    prefix_five_blocks: str | None
    replay_boundary: int
    replay_fingerprint: str | None
    prior_completed_prefix_tokens: int

    @property
    def prefix_768(self) -> str | None:
        return self.prefix_first_block if self.cache_block_size == 768 else None

    @property
    def prefix_3840(self) -> str | None:
        return self.prefix_five_blocks if self.cache_block_size == 768 else None


class PromptInspector:
    def __init__(self, engine_url: str, *, timeout: float = 10, max_inflight: int = 2,
                 slot_wait_seconds: float = .5, block_size: int | None = None,
                 state_path: str | Path | None = None):
        self.url = engine_url.rstrip("/") + "/tokenize"
        self.timeout = timeout
        self._slots = threading.BoundedSemaphore(max_inflight)
        self.slot_wait_seconds = slot_wait_seconds
        self._lock = threading.Lock()
        self._secret = secrets.token_bytes(32)
        self._scope = secrets.token_hex(8)
        self._epoch = None
        self._block_size = block_size if block_size and block_size > 0 else None
        self._state_path = Path(state_path) if state_path else None
        self._quarantined = False
        self._completed: OrderedDict[tuple[str, str], tuple[int, float]] = OrderedDict()
        self._events = deque(maxlen=1024)
        self._load_state()

    def inspect(self, payload: dict, cache_epoch: float | None = None, *,
                cache_block_size: int | None = None,
                deadline: float | None = None, cancelled=None) -> Inspection | None:
        with self._lock:
            if self._quarantined:
                return None
        if not inspectable(payload):
            return None
        until = time.monotonic() + self.slot_wait_seconds
        if deadline is not None:
            until = min(until, deadline)
        while True:
            if cancelled is not None and cancelled():
                raise InspectionError(499, "client_disconnected", "Client disconnected before inspection")
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise InspectionError(503, "tokenizer_busy", "Prompt inspection admission deadline exceeded")
            if self._slots.acquire(timeout=min(.1, remaining)):
                break
        try:
            if isinstance(payload.get("prompt"), list) and "messages" not in payload:
                tokens = payload["prompt"]
            else:
                body = tokenize_projection(payload)
                request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
                try:
                    timeout = min(self.timeout, max(.001, deadline - time.monotonic())) if deadline is not None else self.timeout
                    with urllib.request.urlopen(request, timeout=timeout) as response:
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
            result = self._fingerprint(
                tokens,
                str(payload.get("model") or "FW-Kimi-K3"),
                cache_epoch,
                cache_block_size,
            )
            with self._lock:
                return None if self._quarantined else result
        finally:
            self._slots.release()

    def _load_state(self) -> None:
        if self._state_path is None:
            return
        try:
            state = json.loads(self._state_path.read_text())
            secret = bytes.fromhex(state["secret"])
            scope = state["scope"]
            epoch = state.get("epoch")
            block_size = state.get("block_size")
            rows = state.get("completed")
            if (
                state.get("version") != 1
                or len(secret) != 32
                or not isinstance(scope, str)
                or len(scope) != 16
                or epoch is not None and not isinstance(epoch, (int, float))
                or type(block_size) is not int
                or block_size <= 0
                or not isinstance(rows, list)
            ):
                raise ValueError("invalid prefix state")
            now = time.time()
            completed = OrderedDict()
            for row in rows[-4096:]:
                fingerprint, boundary, observed_at = row
                if (
                    isinstance(fingerprint, str)
                    and len(fingerprint) == 64
                    and type(boundary) is int
                    and boundary > 0
                    and isinstance(observed_at, (int, float))
                    and now - 900 <= observed_at <= now + 5
                ):
                    completed[(scope, fingerprint)] = (boundary, float(observed_at))
            self._secret = secret
            self._scope = scope
            self._epoch = epoch
            self._block_size = block_size
            self._completed = completed
        except (OSError, ValueError, TypeError, KeyError):
            # Diagnostics must never make serving depend on local state.
            return

    def _persist_locked(self) -> None:
        if self._state_path is None:
            return
        state = {
            "version": 1,
            "secret": self._secret.hex(),
            "scope": self._scope,
            "epoch": self._epoch,
            "block_size": self._block_size,
            "completed": [
                [fingerprint, boundary, observed_at]
                for (scope, fingerprint), (boundary, observed_at) in self._completed.items()
                if scope == self._scope
            ],
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._state_path.with_name(
                f".{self._state_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            temporary.write_text(json.dumps(state, separators=(",", ":")))
            os.chmod(temporary, 0o600)
            os.replace(temporary, self._state_path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass

    def _reset_scope_locked(self, epoch, block_size: int) -> None:
        self._completed.clear()
        self._secret = secrets.token_bytes(32)
        self._scope = secrets.token_hex(8)
        self._epoch = epoch
        self._block_size = block_size
        self._persist_locked()

    def _fingerprint(self, tokens, model: str, epoch,
                     cache_block_size: int | None = None) -> Inspection:
        packed = array.array("I", tokens)
        if sys.byteorder != "little":
            packed.byteswap()
        raw = memoryview(packed).cast("B")
        now = time.time()
        with self._lock:
            block_size = (
                cache_block_size
                if type(cache_block_size) is int and cache_block_size > 0
                else self._block_size
            )
            if block_size is None:
                return Inspection(len(tokens), self._scope, 0, None, None, 0, None, 0)
            if (
                self._block_size is not None
                and block_size != self._block_size
            ) or (
                epoch is not None
                and self._epoch is not None
                and epoch != self._epoch
            ):
                self._reset_scope_locked(epoch, block_size)
            else:
                changed = self._block_size != block_size or (
                    epoch is not None and self._epoch != epoch
                )
                self._block_size = block_size
                if epoch is not None:
                    self._epoch = epoch
                if changed:
                    self._persist_locked()
            while self._completed and next(iter(self._completed.values()))[1] < now - 900:
                self._completed.popitem(last=False)
            secret, scope, completed = self._secret, self._scope, dict(self._completed)
        hasher = hmac.new(secret, model.encode() + b"\x00", hashlib.sha256)
        prefix_first = prefix_five = replay = None
        boundary = (len(tokens) - 1) // block_size * block_size if tokens else 0
        seen = 0
        for end in range(block_size, len(tokens) + 1, block_size):
            hasher.update(raw[(end - block_size) * 4:end * 4])
            fingerprint = hasher.copy().hexdigest()
            if end == block_size:
                prefix_first = fingerprint
            if end == block_size * 5:
                prefix_five = fingerprint
            if end == boundary:
                replay = fingerprint
            record = completed.get((scope, fingerprint))
            if record is not None and record[1] >= now - 900:
                seen = max(seen, record[0])
        return Inspection(
            len(tokens), scope, block_size, prefix_first, prefix_five,
            boundary, replay, seen,
        )

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
            event.update(scope=inspection.scope, cache_block_size=inspection.cache_block_size,
                         prefix_first_block=inspection.prefix_first_block,
                         prefix_five_blocks=inspection.prefix_five_blocks,
                         prefix_768=inspection.prefix_768, prefix_3840=inspection.prefix_3840,
                         replay_boundary=inspection.replay_boundary,
                         replay_fingerprint=inspection.replay_fingerprint,
                         prior_completed_prefix_tokens=inspection.prior_completed_prefix_tokens)
        with self._lock:
            self._events.append(event)
            if (inspection is not None and complete and 200 <= http_status < 300
                    and type(prompt) is int and prompt != inspection.tokens):
                self._quarantined = True
                self._completed.clear()
                self._persist_locked()
            # A non-block-aligned prompt forces its final full checkpoint; an
            # exactly block-aligned prompt has the previously observed replay gap.
            if (verified and not self._quarantined and inspection.scope == self._scope
                    and inspection.cache_block_size > 0
                    and inspection.tokens % inspection.cache_block_size
                    and inspection.replay_fingerprint):
                key = (inspection.scope, inspection.replay_fingerprint)
                self._completed.pop(key, None)
                self._completed[key] = (inspection.replay_boundary, time.time())
                while len(self._completed) > 4096:
                    self._completed.popitem(last=False)
                self._persist_locked()
        return event

    def recent(self):
        with self._lock:
            return {"scope": self._scope, "events": list(self._events), "history_entries": len(self._completed),
                    "cache_block_size": self._block_size,
                    "state_persistent": self._state_path is not None,
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
