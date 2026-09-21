"""Validate media once at the gateway, preserving the original image bytes.

``normalize_controls`` is the cheap, idempotent pre-bridge/callback interface.
Only ``normalize_payload`` downloads and validates media. The success cache is
content-addressed, never URL-addressed, and retains no decoded Pillow images.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import io
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

from .cancellation import (
    CancelReason,
    RequestCancellation,
    RequestCancelled,
    UpstreamResources,
    current_cancellation,
)

IMAGE_PART_TYPES = {
    "image_url",
    "image",
    "input_image",
    "file",
    "input_file",
    "file_url",
    "document",
}
UNSUPPORTED_MEDIA_PART_TYPES = {
    "audio", "input_audio", "output_audio", "audio_url",
    "video", "input_video", "video_url",
}
DEFAULT_MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 16_000_000
DEFAULT_MAX_REQUEST_IMAGES = int(os.environ.get("K3_MAX_REQUEST_IMAGES", "8"))
if not 1 <= DEFAULT_MAX_REQUEST_IMAGES <= 8:
    raise ValueError("K3_MAX_REQUEST_IMAGES must be in [1, 8]")
DEFAULT_MAX_REQUEST_PIXELS = 32_000_000
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024
DEFAULT_MEDIA_WORKERS = 2
DEFAULT_MEDIA_WAIT_SECONDS = 1.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 10.0
DEFAULT_DOWNLOAD_READ_TIMEOUT_SECONDS = 1.0
DEFAULT_CACHE_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_CACHE_MAX_ENTRIES = 128
DEFAULT_CACHE_TTL_SECONDS = 300.0

# Bump whenever the accepted image/validation policy changes. Only these
# single-frame formats are part of the gateway's K3 compatibility contract.
MEDIA_POLICY_REVISION = "k3-original-images-v1"
SUPPORTED_IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
_CHUNK_BYTES = 64 * 1024
_WAIT_POLL_SECONDS = 0.05
_MAX_REDIRECTS = 5
# A per-request salt partitions the GPU prefix cache. One YYDS/portal key
# serving a pool must share prefixes; inbound salts are dropped here.
# Client `priority` must not reach LiteLLM (Redis scheduler heap). The
# gateway stamps engine priority after tenancy.
PREFIX_BUSTERS = ("cache_salt", "prompt_cache_key", "kv_cache_salt", "priority")
# OpenAI/YYDS default is medium. K3 encoding_k3._VALID_THINKING_EFFORTS is
# {low, high, max} and 500s on anything else.
TOOL_KEYS = ("tools", "functions")
THINKING_KEYS = ("thinking_effort", "reasoning_effort")
THINKING_ALLOWED = {"low", "high", "max"}
THINKING_ALIASES = {
    "medium": "high",
    "default": "high",
    "adaptive": "high",
    "minimal": "low",
    "none": "low",
    "off": "low",
    "xhigh": "max",
    "extra_high": "max",
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


class MediaValidationError(ValueError):
    """A required media part is invalid; safe to include in an HTTP 400."""

    def __init__(self, message_index: int, part_index: int, *, reason: str = "invalid_media") -> None:
        self.message_index = message_index
        self.part_index = part_index
        self.param = f"messages[{message_index}].content[{part_index}]"
        self.reason = reason if reason in {"invalid_media", "image_count_limit"} else "invalid_media"
        message = (f"At most {DEFAULT_MAX_REQUEST_IMAGES} image(s) per request are currently supported. Field: {self.param}."
                   if self.reason == "image_count_limit" else f"Invalid, unsupported, or oversized media at {self.param}.")
        super().__init__(message)


class MediaBusyError(RuntimeError):
    """The bounded media workers could not be acquired; map to HTTP 503."""

    def __init__(self) -> None:
        super().__init__("Media processing is busy; try again.")


class _InvalidMedia(ValueError):
    """Internal failures carry no input, URL, decoder text, or response body."""


@dataclass(frozen=True)
class _ValidatedImage:
    raw: bytes
    width: int
    height: int
    mime_type: str


class _ValidationCache:
    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_CACHE_MAX_BYTES,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
        ttl: float = DEFAULT_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, tuple[float, _ValidatedImage]] = OrderedDict()
        self._pending: dict[bytes, threading.Event] = {}
        self._bytes = 0
        self._counts = dict.fromkeys((
            "hits", "misses", "coalesced", "evictions", "expirations",
            "validation_attempts", "validations", "errors", "busy", "cancelled",
            "downloads", "download_bytes", "workers_in_flight", "workers_waiting",
        ), 0)

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += amount

    def _expire(self, now: float) -> None:
        # LRU order is not expiry order: a hit does not extend validation TTL.
        for key, (expires, value) in list(self._entries.items()):
            if expires <= now:
                del self._entries[key]
                self._bytes -= len(value.raw)
                self._counts["expirations"] += 1

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            self._expire(self._clock())
            return {
                **self._counts,
                "entries": len(self._entries),
                "bytes": self._bytes,
                "pending": len(self._pending),
                "max_entries": self._max_entries,
                "max_bytes": self._max_bytes,
                "ttl_seconds": self._ttl,
                "workers_limit": DEFAULT_MEDIA_WORKERS,
                "request_images_limit": DEFAULT_MAX_REQUEST_IMAGES,
            }

    def validate(self, raw: bytes, pixel_limit: int) -> _ValidatedImage:
        digest = hashlib.sha256(MEDIA_POLICY_REVISION.encode("ascii") + b"\0")
        for offset in range(0, len(raw), _CHUNK_BYTES):
            _check_cancellation()
            digest.update(memoryview(raw)[offset:offset + _CHUNK_BYTES])
        key = digest.digest()
        deadline = time.monotonic() + DEFAULT_MEDIA_WAIT_SECONDS
        first_lookup = True
        while True:
            _check_cancellation()
            with self._lock:
                self._expire(self._clock())
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    self._counts["hits"] += 1
                    value = cached[1]
                    _check_pixels(value.width, value.height, pixel_limit)
                    return value
                if first_lookup:
                    self._counts["misses"] += 1
                done = self._pending.get(key)
                owner = done is None
                if owner:
                    done = self._pending[key] = threading.Event()
                elif first_lookup:
                    self._counts["coalesced"] += 1
            first_lookup = False
            if owner:
                break
            # A concurrent successful validation is shared. Failed/cancelled
            # validations only wake waiters; they never become negative entries.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MediaBusyError()
            done.wait(min(_WAIT_POLL_SECONDS, remaining))

        try:
            self.count("validation_attempts")
            value = _validate_image(raw, pixel_limit)
            _check_cancellation()
            with self._lock:
                now = self._clock()
                self._expire(now)
                self._counts["validations"] += 1
                if len(raw) <= self._max_bytes and self._max_entries > 0:
                    while (len(self._entries) >= self._max_entries
                           or self._bytes + len(raw) > self._max_bytes):
                        _, (_, evicted) = self._entries.popitem(last=False)
                        self._bytes -= len(evicted.raw)
                        self._counts["evictions"] += 1
                    self._entries[key] = (now + self._ttl, value)
                    self._bytes += len(raw)
            return value
        finally:
            with self._lock:
                self._pending.pop(key, None)
                done.set()


_CACHE = _ValidationCache()
_WORKERS = threading.BoundedSemaphore(DEFAULT_MEDIA_WORKERS)


def media_cache_stats() -> dict[str, int | float]:
    """Thread-safe numeric snapshot, without content hashes or user labels.

    Gauges: entries, bytes, pending, workers_in_flight, workers_waiting and
    configured limits/TTL. Other fields are cumulative counters. Errors count
    failed requests once; validations count successfully decoded images.
    """
    return _CACHE.stats()


def _check_cancellation() -> None:
    cancellation = current_cancellation()
    if cancellation is not None:
        cancellation.check()


@contextmanager
def _media_worker() -> Iterator[None]:
    deadline = time.monotonic() + DEFAULT_MEDIA_WAIT_SECONDS
    acquired = False
    _CACHE.count("workers_waiting")
    try:
        while not acquired:
            _check_cancellation()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MediaBusyError()
            acquired = _WORKERS.acquire(timeout=min(_WAIT_POLL_SECONDS, remaining))
    finally:
        _CACHE.count("workers_waiting", -1)
    _CACHE.count("workers_in_flight")
    try:
        _check_cancellation()
        yield
    finally:
        _CACHE.count("workers_in_flight", -1)
        _WORKERS.release()


def normalize_payload(payload: dict) -> dict:
    """Apply controls and validate every required media part after the bridge.

    Limits apply to occurrences, including repeated/cache-hit images. Message
    rewrites are published only after the complete media request is accepted.
    """
    try:
        _check_cancellation()
        normalize_controls(payload)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload
        parts = []
        for message_index, message in enumerate(messages):
            _check_cancellation()
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                _check_cancellation()
                if not isinstance(part, dict) or not _looks_like_media(part):
                    continue
                part_type = part.get("type")
                if len(parts) >= DEFAULT_MAX_REQUEST_IMAGES:
                    raise MediaValidationError(message_index, part_index, reason="image_count_limit")
                if ((isinstance(part_type, str) and part_type in UNSUPPORTED_MEDIA_PART_TYPES)
                        or any(key in part for key in UNSUPPORTED_MEDIA_PART_TYPES)):
                    raise MediaValidationError(message_index, part_index)
                parts.append((message_index, part_index, part))

        rewritten = list(messages)
        raw_bytes = pixels = 0
        for message_index, part_index, part in parts:
            _check_cancellation()
            try:
                with _media_worker():
                    url = _extract_url(part)
                    if not url:
                        raise _InvalidMedia()
                    raw = _bytes_from_url(url, DEFAULT_MAX_REQUEST_BYTES - raw_bytes)
                    value = _CACHE.validate(raw, DEFAULT_MAX_REQUEST_PIXELS - pixels)
                    raw_bytes += len(raw)
                    pixels += value.width * value.height
                    encoded = base64.b64encode(value.raw).decode("ascii")
                    _check_cancellation()
                    normalized = {
                        "type": "image_url",
                        "image_url": {"url": f"data:{value.mime_type};base64,{encoded}"},
                    }
                    detail = part.get("image_url")
                    if isinstance(detail, dict) and "detail" in detail:
                        normalized["image_url"]["detail"] = detail["detail"]
                    elif "detail" in part:
                        normalized["image_url"]["detail"] = part["detail"]
            except _InvalidMedia:
                raise MediaValidationError(message_index, part_index) from None
            if rewritten[message_index] is messages[message_index]:
                rewritten[message_index] = dict(messages[message_index])
                rewritten[message_index]["content"] = list(messages[message_index]["content"])
            rewritten[message_index]["content"][part_index] = normalized
        if parts:
            payload["messages"] = rewritten
        return payload
    except MediaValidationError:
        _CACHE.count("errors")
        raise
    except MediaBusyError:
        _CACHE.count("busy")
        raise
    except RequestCancelled:
        _CACHE.count("cancelled")
        raise


def normalize_controls(payload: dict) -> dict:
    """Lightweight thinking, prefix controls and top-level tool ordering only."""
    _bind_thinking_controls(payload)
    for key in PREFIX_BUSTERS:
        payload.pop(key, None)
    extra = payload.get("extra_body")
    if isinstance(extra, dict):
        for key in PREFIX_BUSTERS:
            extra.pop(key, None)
        _normalize_thinking(extra)
        nested = extra.get("chat_template_kwargs")
        if isinstance(nested, dict):
            _normalize_thinking(nested)
    _normalize_thinking(payload)
    kwargs = payload.get("chat_template_kwargs")
    if isinstance(kwargs, dict):
        _normalize_thinking(kwargs)
    for key in TOOL_KEYS:
        if key in payload:
            payload[key] = _canonical_tools(payload[key])
    return payload


def _canonical_tools(tools: Any) -> Any:
    """Order the tool array by name so a reshuffle is not a total cache miss.

    K3 renders tool declarations ahead of all conversation content, and its
    `deep_sort_dict` sorts each schema's *keys* but preserves array order. So
    key order costs nothing and array order costs everything: measured on this
    engine, rotating three tools drops the shared prefix from 1142 tokens to 35
    (eval/harness/prefix_probe.py). Declaration order carries no meaning to the
    caller, and 56% of live requests reused zero tokens.

    Sorting also makes growth cheap: a new tool lands at its sorted position
    instead of shifting the whole blob.
    """
    if not isinstance(tools, list) or len(tools) < 2:
        return tools
    names = []
    for tool in tools:
        _check_cancellation()
        if not isinstance(tool, dict):
            return tools
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if not isinstance(name, str):
            return tools
        names.append(name)
    # Duplicate or missing names mean the intended order is not recoverable
    # from names alone; leave the caller's order untouched rather than guess.
    if len(set(names)) != len(names):
        return tools
    return [tool for _, tool in sorted(zip(names, tools), key=lambda pair: pair[0])]


def _bind_thinking_controls(payload: dict) -> None:
    """Translate effort into the controls K3's tokenizer actually consumes.

    K3 defaults to thinking=True, thinking_effort=max. Renaming an OpenAI
    effort value at the request root does not change that tokenizer default.
    Explicit template controls take precedence; extra_body overrides root
    fields while retaining other explicitly supplied template settings.
    """
    extra = payload.get("extra_body")
    extra = extra if isinstance(extra, dict) else {}
    root_template = payload.get("chat_template_kwargs")
    extra_template = extra.get("chat_template_kwargs")
    if any(value is not None and not isinstance(value, dict)
           for value in (root_template, extra_template)):
        return
    # Remove invalid strings before overlaying envelopes. Otherwise an invalid
    # extra-body value hides a valid root value until the second hop. Keep raw
    # none/off aliases here so native thinking=False can still be bound.
    for body in (payload, extra, root_template, extra_template):
        if isinstance(body, dict):
            _normalize_thinking(body, map_aliases=False)
    template = {**(root_template or {}), **(extra_template or {})}
    effective = {**payload, **extra}
    effort = next((body[key] for body in (template, effective)
                   for key in THINKING_KEYS if isinstance(body.get(key), str)
                   and THINKING_ALIASES.get(body[key], body[key]) in THINKING_ALLOWED), None)
    changed = False
    if "thinking" not in template and isinstance(template.get("enable_thinking"), bool):
        template["thinking"] = template["enable_thinking"]
        changed = True
    if effort is not None:
        mapped = THINKING_ALIASES.get(effort, effort)
        if mapped in THINKING_ALLOWED:
            template["thinking_effort"] = mapped
            if effort in {"none", "off"} and "thinking" not in template:
                template["thinking"] = False
            changed = True
    if changed:
        target = extra if isinstance(extra_template, dict) else payload
        target["chat_template_kwargs"] = template


def _normalize_thinking(body: dict, *, map_aliases: bool = True) -> None:
    for key in THINKING_KEYS:
        value = body.get(key)
        if not isinstance(value, str):
            continue
        mapped = THINKING_ALIASES.get(value, value)
        if mapped in THINKING_ALLOWED:
            if map_aliases:
                body[key] = mapped
        else:
            body.pop(key, None)


def _looks_like_media(part: dict) -> bool:
    part_type = part.get("type")
    return (
        (isinstance(part_type, str)
         and part_type in IMAGE_PART_TYPES | UNSUPPORTED_MEDIA_PART_TYPES)
        or any(key in part for key in (
            "image_url", "image", "input_image", "file", "file_data", "file_id", "file_url",
            *UNSUPPORTED_MEDIA_PART_TYPES,
        ))
        or isinstance(part.get("source"), dict)
    )


def _extract_url(part: dict) -> str:
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
        media = source.get("media_type") or source.get("mediaType") or "image/png"
        if isinstance(data, str) and data.strip():
            data = data.strip()
            if data.startswith("data:"):
                return data
            return f"data:{media};base64,{data}"
        href = source.get("url")
        if isinstance(href, str) and href.strip():
            return href.strip()
    file_obj = part.get("file")
    for file_obj in (file_obj, part):
        if not isinstance(file_obj, dict):
            continue
        data = file_obj.get("file_data") or file_obj.get("data")
        if isinstance(data, str) and data.strip():
            data = data.strip()
            if data.startswith("data:"):
                return data
            return f"data:application/octet-stream;base64,{data}"
        href = file_obj.get("file_url")
        if isinstance(href, str) and href.strip():
            return href.strip()
    direct = part.get("url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return ""


def _bytes_from_url(url: str, max_bytes: int) -> bytes:
    if url.startswith("data:"):
        header, sep, payload = url.partition(",")
        mime, *parameters = header[5:].lower().split(";")
        if (not sep or "base64" not in parameters
                or not (mime.startswith("image/") or mime in {"", "application/octet-stream"})):
            raise _InvalidMedia()
        return _b64decode(payload, max_bytes)
    if url.startswith(("http://", "https://")):
        return _download(url, max_bytes)
    return _b64decode(url, max_bytes)


def _b64decode(text: str, max_bytes: int) -> bytes:
    # Bound compact base64 before allocating decoded bytes. Check cancellation
    # while consuming whitespace-heavy inputs as well as real encoded data.
    pieces = []
    length = 0
    max_encoded = 4 * ((max_bytes + 2) // 3)
    for offset in range(0, len(text), _CHUNK_BYTES):
        _check_cancellation()
        piece = "".join(text[offset:offset + _CHUNK_BYTES].split())
        length += len(piece)
        if length > max_encoded:
            raise _InvalidMedia()
        pieces.append(piece)
    compact = "".join(pieces)
    pad = "=" * (-len(compact) % 4)
    try:
        raw = base64.b64decode(compact + pad, validate=True)
    except (binascii.Error, ValueError):
        raise _InvalidMedia() from None
    _check_cancellation()
    if not raw or len(raw) > max_bytes:
        raise _InvalidMedia()
    return raw


def _download(url: str, max_bytes: int) -> bytes:
    """Fetch afresh, with one total deadline across DNS, redirects and reads.

    UpstreamResources bounds DNS/connect waits and shuts down blocked sockets.
    The child deadline only bounds this download, never the parent's inference
    budget. Parent cancellation is propagated rather than turned into a 400.
    """
    _check_cancellation()
    parent = current_cancellation()
    cancellation = RequestCancellation(timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS).start()
    unregister = (parent.on_cancel(lambda: cancellation.cancel(
        parent.reason or CancelReason.CLIENT_DISCONNECT
    )) if parent is not None else lambda: None)
    limit = min(DEFAULT_MAX_DOWNLOAD_BYTES, max_bytes)
    _CACHE.count("downloads")
    try:
        for _ in range(_MAX_REDIRECTS + 1):
            _check_cancellation()
            cancellation.check()
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise _InvalidMedia()
            connection_type = (http.client.HTTPSConnection if parsed.scheme == "https"
                               else http.client.HTTPConnection)
            timeout = min(DEFAULT_DOWNLOAD_READ_TIMEOUT_SECONDS, cancellation.remaining)
            conn = connection_type(parsed.hostname, parsed.port, timeout=timeout)
            resources = UpstreamResources(conn, cancellation, timeout)
            try:
                with resources.io_lock:
                    resources.check()
                    conn.connect()
                    # HTTPS wraps the socket created by UpstreamResources.
                    # Publish that wrapper before headers/body can block.
                    if conn.sock is not None:
                        resources.publish_socket(conn.sock)
                    path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
                    conn.request("GET", path, headers={
                        "User-Agent": "k3-media", "Accept-Encoding": "identity",
                    })
                    resources.check()
                    response = resources.response = conn.getresponse()
                    resources.check()
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.getheader("Location")
                        if not location:
                            raise _InvalidMedia()
                        url = urljoin(url, location)
                        continue
                    if not 200 <= response.status < 300:
                        raise _InvalidMedia()
                    declared = response.getheader("Content-Length")
                    if declared is not None and (not declared.isdecimal() or int(declared) > limit):
                        raise _InvalidMedia()
                    raw = bytearray()
                    while True:
                        _check_cancellation()
                        resources.check()
                        chunk = response.read1(min(_CHUNK_BYTES, limit + 1 - len(raw)))
                        _check_cancellation()
                        resources.check()
                        if not chunk:
                            # HTTPResponse.read1 can return early EOF without
                            # raising for a truncated Content-Length response.
                            if isinstance(response.length, int) and response.length > 0:
                                raise _InvalidMedia()
                            break
                        _CACHE.count("download_bytes", len(chunk))
                        raw.extend(chunk)
                        if len(raw) > limit:
                            raise _InvalidMedia()
                    if not raw:
                        raise _InvalidMedia()
                    return bytes(raw)
            finally:
                resources.close()
        raise _InvalidMedia()
    except (RequestCancelled, OSError, ValueError, http.client.HTTPException):
        _check_cancellation()
        raise _InvalidMedia() from None
    finally:
        unregister()
        cancellation.finish()


def png_size(raw: bytes) -> tuple[int, int] | None:
    """Untrusted IHDR dimensions for estimation only, NOT image validation."""
    if len(raw) < 24 or not raw.startswith(PNG_MAGIC) or raw[12:16] != b"IHDR":
        return None
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _check_pixels(width: int, height: int, request_remaining: int) -> None:
    if width <= 0 or height <= 0 or width * height > min(DEFAULT_MAX_IMAGE_PIXELS, request_remaining):
        raise _InvalidMedia()


def _check_png_end(raw: bytes) -> None:
    # Pillow verify() stops at the IEND tag, before reading its CRC. load()
    # also accepts a missing/corrupt IEND CRC. Walk chunk boundaries so a
    # footer-like byte sequence inside compressed pixels cannot stand in for
    # the complete, zero-length IEND chunk.
    offset = len(PNG_MAGIC)
    while offset + 12 <= len(raw):
        _check_cancellation()
        length = int.from_bytes(raw[offset:offset + 4], "big")
        end = offset + length + 12
        if end > len(raw):
            break
        if raw[offset + 4:offset + 8] == b"IEND":
            if length == 0 and raw[offset + 8:end] == b"\xaeB`\x82":
                return
            break
        offset = end
    raise _InvalidMedia()


def _validate_image(raw: bytes, pixel_limit: int) -> _ValidatedImage:
    try:
        # Lazy import keeps controls/text-only requests independent of Pillow.
        # Never fall back to magic/header acceptance if Pillow is unavailable.
        from PIL import Image

        _check_cancellation()
        with Image.open(io.BytesIO(raw), formats=list(SUPPORTED_IMAGE_FORMATS)) as image:
            width, height = image.size
            _check_pixels(width, height, pixel_limit)
            mime_type = SUPPORTED_IMAGE_FORMATS.get(image.format)
            if mime_type is None or getattr(image, "n_frames", 1) != 1:
                raise _InvalidMedia()
            if image.format == "PNG":
                _check_png_end(raw)
            image.verify()
        _check_cancellation()
        # verify() checks container integrity (including PNG CRCs); reopening
        # and load() also validates compressed pixels. Neither is a save,
        # resize, EXIF transpose, colour conversion or alpha flattening.
        with Image.open(io.BytesIO(raw), formats=list(SUPPORTED_IMAGE_FORMATS)) as image:
            image.load()
        _check_cancellation()
        return _ValidatedImage(raw, width, height, mime_type)
    except RequestCancelled:
        raise
    except Exception:  # noqa: BLE001 -- decoder messages can contain user data
        raise _InvalidMedia() from None
