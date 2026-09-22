"""Normalize agent/multimodal parts into vLLM-readable PNG data URIs."""

from __future__ import annotations

import base64
import binascii
import io
import json
import subprocess
import tempfile
import threading
from fractions import Fraction
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

IMAGE_PART_TYPES = {
    "image_url",
    "image",
    "input_image",
    "file",
}
VIDEO_PART_TYPES = {
    "video_url",
    "video",
    "input_video",
}
NOTE = "[attached media could not be decoded; continuing with the text]"
VIDEO_LIMIT_NOTE = "[additional video omitted; one video is allowed per request]"
MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 48 * 1024 * 1024
DOWNLOAD_TIMEOUT = 10
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
# Longest edge after decode. Live vision turns billed 3.7k–13k image tokens;
# the encoder cache on this build is 16,817. 1568 bounds a square image to
# ~12k tokens at 192 px/token without wrecking screenshot OCR.
MAX_IMAGE_EDGE = 1568
# vLLM's current Kimi-K3 adapter is image-only even though the checkpoint is
# video-native. Keep the compatibility path bounded: eight 768px frames are
# about 6.3k media tokens in the worst square case, versus an unbounded 8 FPS
# checkpoint default.
MAX_VIDEO_FRAMES = 8
MAX_VIDEO_EDGE = 768
MAX_VIDEOS_PER_REQUEST = 1
VIDEO_DECODE_TIMEOUT = 20
_VIDEO_SLOTS = threading.BoundedSemaphore(2)
# Fixed so the same JPEG always becomes the same PNG. PIL defaults can move
# and would bust the prefix cache on every hop.
PNG_COMPRESS_LEVEL = 6


def normalize_payload(payload: dict) -> dict:
    _normalize_reasoning_request(payload)
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
    messages = payload.get("messages")
    if isinstance(messages, list):
        video_budget = [MAX_VIDEOS_PER_REQUEST]
        payload["messages"] = [
            _normalize_message(m, video_budget) if isinstance(m, dict) else m
            for m in messages
        ]
    return payload


def _normalize_reasoning_request(payload: dict) -> None:
    """Map common Chat reasoning objects onto K3's native template controls.

    LiteLLM supports ``reasoning_effort`` for Chat, while OpenAI Responses and
    several compatible clients send an object.  K3 consumes neither object
    directly; it reads ``chat_template_kwargs``.
    """
    reasoning = payload.get("reasoning")
    thinking = payload.get("thinking")
    enable_thinking = payload.get("enable_thinking")
    objects = [value for value in (reasoning, thinking) if isinstance(value, dict)]
    if (
        not objects
        and not isinstance(thinking, bool)
        and not isinstance(enable_thinking, bool)
    ):
        return

    template = payload.get("chat_template_kwargs")
    if template is not None and not isinstance(template, dict):
        return
    template = dict(template or {})

    if "reasoning_effort" not in payload and "thinking_effort" not in payload:
        effort = next(
            (
                value.get("effort")
                for value in objects
                if isinstance(value.get("effort"), str)
            ),
            None,
        )
        if effort is not None:
            payload["reasoning_effort"] = effort

    if "thinking" not in template and "enable_thinking" not in template:
        enabled: bool | None = (
            enable_thinking
            if isinstance(enable_thinking, bool)
            else thinking if isinstance(thinking, bool) else None
        )
        for value in objects:
            if isinstance(value.get("enabled"), bool):
                enabled = value["enabled"]
                break
            kind = value.get("type")
            if kind in {"disabled", "off", "none"}:
                enabled = False
                break
            if kind in {"enabled", "adaptive"}:
                enabled = True
                break
        if enabled is not None:
            template["thinking"] = enabled

    if template:
        payload["chat_template_kwargs"] = template
    if isinstance(reasoning, dict):
        payload.pop("reasoning", None)
    if isinstance(thinking, (dict, bool)):
        payload.pop("thinking", None)
    if isinstance(enable_thinking, bool):
        payload.pop("enable_thinking", None)


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
    template = {**(root_template or {}), **(extra_template or {})}
    effective = {**payload, **extra}
    effort = next((body[key] for body in (template, effective)
                   for key in THINKING_KEYS if isinstance(body.get(key), str)), None)
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


def _normalize_thinking(body: dict) -> None:
    for key in THINKING_KEYS:
        value = body.get(key)
        if not isinstance(value, str):
            continue
        mapped = THINKING_ALIASES.get(value, value)
        if mapped in THINKING_ALLOWED:
            body[key] = mapped
        else:
            body.pop(key, None)


def _normalize_message(message: dict, video_budget: list[int] | None = None) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    rewritten: list[Any] = []
    for part in content:
        if isinstance(part, dict) and _looks_like_media(part):
            url = _extract_url(part)
            if _looks_like_video(part, url):
                if video_budget is not None and video_budget[0] <= 0:
                    rewritten.append({"type": "text", "text": VIDEO_LIMIT_NOTE})
                    continue
                if video_budget is not None:
                    video_budget[0] -= 1
            rewritten.extend(_normalize_part(part))
        else:
            rewritten.append(part)
    out = dict(message)
    out["content"] = rewritten
    return out


def _looks_like_media(part: dict) -> bool:
    part_type = part.get("type")
    return (
        part_type in IMAGE_PART_TYPES
        or part_type in VIDEO_PART_TYPES
        or "image_url" in part
        or "image" in part
        or "video_url" in part
        or "video" in part
        or "input_video" in part
        or isinstance(part.get("source"), dict)
        or isinstance(part.get("file"), dict)
    )


def _normalize_part(part: dict) -> list[dict]:
    url = _extract_url(part)
    raw = _bytes_from_url(url) if url else None
    if _looks_like_video(part, url):
        frames = _video_frames(raw) if raw else []
        if not frames:
            return [{"type": "text", "text": NOTE}]
        rewritten: list[dict] = []
        for index, (png, timestamp) in enumerate(frames, start=1):
            rewritten.extend([
                {
                    "type": "text",
                    "text": (
                        f"[video frame {index}/{len(frames)}"
                        f" at {_format_timestamp(timestamp)}]"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + base64.b64encode(png).decode("ascii")
                        )
                    },
                },
            ])
        return rewritten
    png = _as_png(raw) if raw else None
    if png is None:
        return [{"type": "text", "text": NOTE}]
    return [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")},
    }]


def _extract_url(part: dict) -> str:
    for key in (
        "image_url",
        "image",
        "input_image",
        "video_url",
        "video",
        "input_video",
    ):
        value = part.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for inner in ("url", "data", "image_url", "video_url"):
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
    if isinstance(file_obj, dict):
        data = file_obj.get("file_data") or file_obj.get("data")
        name = str(file_obj.get("filename") or "upload.bin")
        mime = "image/png" if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")) else "application/octet-stream"
        if isinstance(data, str) and data.strip():
            data = data.strip()
            if data.startswith("data:"):
                return data
            return f"data:{mime};base64,{data}"
    direct = part.get("url")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return ""


def _looks_like_video(part: dict, url: str) -> bool:
    if part.get("type") in VIDEO_PART_TYPES:
        return True
    if any(key in part for key in ("video_url", "video", "input_video")):
        return True
    source = part.get("source")
    if isinstance(source, dict) and str(
        source.get("media_type") or source.get("mediaType") or ""
    ).lower().startswith("video/"):
        return True
    file_obj = part.get("file")
    if isinstance(file_obj, dict):
        name = str(file_obj.get("filename") or "").lower()
        if name.endswith((".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")):
            return True
    return url.lower().startswith("data:video/")


def _bytes_from_url(url: str) -> bytes | None:
    if url.startswith("data:"):
        return _decode_data_url(url)
    if url.startswith(("http://", "https://")):
        return _download(url)
    if _looks_like_base64(url):
        return _b64decode(url)
    return None


def _decode_data_url(url: str) -> bytes | None:
    _, _, payload = url.partition(",")
    if not payload:
        return None
    return _b64decode(payload)


def _looks_like_base64(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 32 or len(compact) % 4 not in {0, 2, 3}:
        return False
    return all(c.isalnum() or c in "+/=" for c in compact)


def _b64decode(text: str) -> bytes | None:
    compact = "".join(text.split())
    pad = "=" * (-len(compact) % 4)
    try:
        return base64.b64decode(compact + pad, validate=False)
    except (binascii.Error, ValueError):
        return None


def _download(url: str) -> bytes | None:
    try:
        with urlopen(Request(url, headers={"User-Agent": "k3-media"}), timeout=DOWNLOAD_TIMEOUT) as resp:
            raw = resp.read(MAX_DOWNLOAD_BYTES + 1)
            return raw if len(raw) <= MAX_DOWNLOAD_BYTES else None
    except (URLError, TimeoutError, OSError, ValueError):
        return None


def _video_frames(raw: bytes) -> list[tuple[bytes, float]]:
    """Decode uniformly sampled frames without allowing ffmpeg network access."""
    if not raw or len(raw) > MAX_VIDEO_BYTES:
        return []
    if not _VIDEO_SLOTS.acquire(timeout=1):
        return []
    try:
        with tempfile.TemporaryDirectory(prefix="k3-video-") as directory:
            source = f"{directory}/input"
            pattern = f"{directory}/frame-%03d.png"
            with open(source, "wb") as stream:
                stream.write(raw)

            duration, source_fps = _video_metadata(source)
            target_fps = MAX_VIDEO_FRAMES / duration if duration > 0 else 1.0
            # Never manufacture duplicate frames from low-frame-rate clips.
            fps = min(target_fps, source_fps) if source_fps > 0 else target_fps
            scale = (
                f"scale=w='min({MAX_VIDEO_EDGE},iw)':"
                f"h='min({MAX_VIDEO_EDGE},ih)':force_original_aspect_ratio=decrease"
            )
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    source,
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    f"fps={fps:.8f},{scale}",
                    "-frames:v",
                    str(MAX_VIDEO_FRAMES),
                    "-compression_level",
                    str(PNG_COMPRESS_LEVEL),
                    pattern,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=VIDEO_DECODE_TIMEOUT,
                check=False,
            )
            if result.returncode:
                return []
            frames: list[tuple[bytes, float]] = []
            for index in range(1, MAX_VIDEO_FRAMES + 1):
                try:
                    with open(f"{directory}/frame-{index:03d}.png", "rb") as stream:
                        png = stream.read(MAX_DOWNLOAD_BYTES + 1)
                except OSError:
                    break
                if len(png) > MAX_DOWNLOAD_BYTES or png_size(png) is None:
                    return []
                timestamp = (index - 1) / fps if fps > 0 else 0.0
                frames.append((png, min(timestamp, duration) if duration > 0 else timestamp))
            return frames
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    finally:
        _VIDEO_SLOTS.release()


def _video_metadata(source: str) -> tuple[float, float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration:stream=avg_frame_rate",
                "-of",
                "json",
                source,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode:
            return 0.0, 0.0
        metadata = json.loads(result.stdout)
        duration = float(metadata.get("format", {}).get("duration", 0))
        streams = metadata.get("streams") or []
        raw_rate = streams[0].get("avg_frame_rate", "0/1") if streams else "0/1"
        source_fps = float(Fraction(raw_rate))
        duration = duration if 0 < duration <= 24 * 60 * 60 else 0.0
        source_fps = source_fps if 0 < source_fps <= 240 else 0.0
        return duration, source_fps
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        json.JSONDecodeError,
    ):
        return 0.0, 0.0


def _format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def png_size(raw: bytes) -> tuple[int, int] | None:
    """Width, height from a PNG IHDR. None if the bytes are not a PNG header."""
    if len(raw) < 24 or not raw.startswith(PNG_MAGIC) or raw[12:16] != b"IHDR":
        return None
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _as_png(raw: bytes) -> bytes | None:
    # Re-encoding a valid in-budget PNG changes bytes and therefore the
    # vision-token prefix. 56% of live requests already reuse nothing; do
    # not add a second miss on every multimodal turn.
    size = png_size(raw)
    if size and max(size) <= MAX_IMAGE_EDGE:
        return raw
    try:
        from PIL import Image
    except ImportError:
        if raw.startswith(PNG_MAGIC) or raw.startswith(JPEG_MAGIC):
            return raw
        return None
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        if max(image.size) > MAX_IMAGE_EDGE:
            resample = getattr(Image, "Resampling", Image)
            image.thumbnail(
                (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE),
                getattr(resample, "LANCZOS", Image.LANCZOS),
            )
        elif raw.startswith(PNG_MAGIC) and image.mode in {"RGB", "RGBA", "L", "P"}:
            return raw
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False)
        return out.getvalue()
    except Exception:  # noqa: BLE001 -- unreadable media becomes a text note
        return None
