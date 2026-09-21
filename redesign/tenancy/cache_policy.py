"""Exact response caching, independent of the engine's shared prefix cache.

LiteLLM 1.102.0 runs the proxy hook before routing, then the SDK deployment
hook BEFORE LLMCachingHandler._async_get_cache. Capture auth at the first
barrier; finalize the key at the second, after deployment defaults are merged.
Do not move this to a during-call or success hook (both are too late).

K3_RESPONSE_CACHE_MODE defaults to "static": eligible text chat is cached
automatically, including streaming. "opt_in" requires cache={"use-cache": true};
"off" disables response caching. Explicit opt-outs always win. Tools, tool
history, media and stateful interactions always bypass response caching.
Team pooling requires K3_RESPONSE_CACHE_TEAM_IDS (comma-separated
authenticated team IDs), never request metadata or mere team membership.
Bump K3_RESPONSE_CACHE_REVISION when engine defaults/templates/weights change
behind an unchanged endpoint. Explicit request/deployment settings are hashed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass


MAX_TTL = 300
CACHE_MODES = frozenset({"static", "opt_in", "off"})
CONTEXT_ATTR = "_k3_response_cache_context"
_OVERRIDES = frozenset({
    "cache_key", "preset_cache_key", "redis_namespace", "namespace",
    "caching_groups",
})
# These are proxy bookkeeping/transport objects, not generation parameters.
# Deliberately NOT an allowlist of OpenAI parameters: unknown provider settings,
# extra_body, templates, API target/version and credential all enter the digest.
# Unknown non-JSON values fail closed rather than using unstable repr(object).
_BOOKKEEPING = frozenset({
    "cache", "caching", "cache_key", "preset_cache_key", "metadata",
    "litellm_metadata", "litellm_params", "litellm_logging_obj",
    "proxy_server_request", "client", "http_client", "logger_fn",
    # The live proxy's route_request() injects an aiohttp.ClientSession AFTER
    # the auth hook. This is a per-worker connection pool, not conversation
    # state or generation input. Hashing it fails JSON serialization and
    # silently disables every otherwise eligible response-cache lookup.
    "shared_session",
    "litellm_call_id", "litellm_trace_id", "litellm_session_id", "completion_call_id",
    "parent_otel_span", "model_info", "request_timeout", "timeout",
    "stream_timeout", "num_retries", "max_retries", "fallbacks",
    "fallback_depth", "max_fallbacks", "retry_policy", "retry_strategy",
    "rpm", "tpm", "max_parallel_requests", "litellm_request_debug",
    "acompletion", "async_call", "litellm_model_name", "model_group", "k3_batch",
}) | _OVERRIDES
_MUTABLE = frozenset({
    "tools", "functions", "tool_choice", "function_call", "parallel_tool_calls",
    "previous_response_id", "conversation", "conversation_id", "session_id",
    "thread_id", "assistant_id", "attachments", "file_ids", "files",
    "web_search_options", "search_parameters", "retrieval", "vector_store_ids",
    "prompt_id", "prompt", "store", "background", "mcp_servers",
})


def _digest(value: object) -> str:
    # JSON canonicalization preserves array order and value types. No default=str:
    # object addresses, NaN and lossy coercions must never produce a cache key.
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _field(auth: object, name: str):
    return auth.get(name) if isinstance(auth, Mapping) else getattr(auth, name, None)


def authenticated_scope(auth: object, team_pool_ids: frozenset[str] = frozenset()) -> str | None:
    """Use only the authenticated object; never body/metadata/header identities."""
    team = _field(auth, "team_id")
    if isinstance(team, str) and team and team in team_pool_ids:
        return _digest(["team", _field(auth, "org_id"), team])
    # DB virtual keys carry token; master/custom auth can carry only api_key.
    # LiteLLM normally supplies SHA256 already. Normalize a raw sk- key to the
    # same hash so cold auth and cached auth resolve to the same pool.
    key = _field(auth, "token") or _field(auth, "api_key")
    if not isinstance(key, str) or not key.strip():
        return None
    key_hash = key.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", key) else hashlib.sha256(key.encode()).hexdigest()
    return _digest(["key", key_hash])


def is_static_text_chat(data: dict, call_type: str) -> bool:
    if call_type not in {"completion", "acompletion"}:
        return False
    for body in (data, data.get("extra_body")):
        if isinstance(body, dict) and any(body.get(key) for key in _MUTABLE):
            return False
    # LiteLLM can synthesize metadata.session_id and litellm_session_id for
    # every request (missing_session_id=generate). They correlate logs; they
    # do not retrieve conversation state on chat/completions. Do not depend on
    # a client-spoofable "generated" marker. Actual body session/conversation
    # handles and tool history still disqualify the request.
    for name in ("metadata", "litellm_metadata"):
        metadata = data.get(name)
        if isinstance(metadata, dict) and any(metadata.get(key) for key in _MUTABLE - {"session_id"}):
            return False
    extra = data.get("extra_body")
    if isinstance(extra, dict) and "messages" in extra:
        # Provider extra_body can replace messages after the proxy normalizer.
        if not is_static_text_chat({"messages": extra["messages"]}, call_type):
            return False
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    # Includes tool-call history with tools omitted; media is checked BEFORE
    # normalization can replace an unavailable remote file with a text note.
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant"}:
            return False
        if any(message.get(key) for key in ("tool_calls", "function_call", "tool_call_id")):
            return False
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not all(
            isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            for part in content
        ):
            return False
    return True


def strip_cache_overrides(data: dict) -> None:
    """Scrub only request envelopes, not user text, schemas or tool arguments."""
    for key in _OVERRIDES:
        data.pop(key, None)
    control = data.get("cache")
    if isinstance(control, dict):
        for key in _OVERRIDES:
            control.pop(key, None)
    for name in ("metadata", "litellm_metadata", "litellm_params", "extra_body"):
        nested = data.get(name)
        if isinstance(nested, dict):
            strip_cache_overrides(nested)
            # extra_body is merged into provider kwargs later; it must not
            # smuggle a second set of cache controls around the proxy barrier.
            nested.pop("cache", None)
            nested.pop("caching", None)
    # This is an internal logging envelope, not a completion argument. Keeping
    # even a server-authored preset here leaks litellm_params into the OpenAI
    # provider body in 1.102.0. The handler's cache_key path is the safe API.
    data.pop("litellm_params", None)


def disable_response_cache(data: dict) -> None:
    strip_cache_overrides(data)
    data["caching"] = False
    data["cache"] = {"use-cache": False, "no-cache": True, "no-store": True}


@dataclass(frozen=True)
class CacheContext:
    scope: str
    revision: str
    ttl: float
    max_age: float


def _explicit_opt_out(data: dict) -> bool:
    control = data.get("cache")
    return data.get("caching") is False or (
        isinstance(control, dict) and (
            control.get("use-cache") is False or bool(control.get("no-cache") or control.get("no-store"))
        )
    )


def capture_context(
    data: dict,
    auth: object,
    call_type: str,
    *,
    team_pool_ids: frozenset[str] = frozenset(),
    revision: str = "1",
    mode: str = "static",
) -> CacheContext | None:
    if mode not in CACHE_MODES or mode == "off":
        return None
    if _explicit_opt_out(data):
        return None
    proxy_request = data.get("proxy_server_request")
    original_control = {}
    if isinstance(proxy_request, dict):
        original = proxy_request.get("body")
        # LiteLLM key-level cache controls can replace data.cache AFTER this
        # proxy snapshot. A caller's explicit opt-out must still win.
        if isinstance(original, dict) and _explicit_opt_out(original):
            return None
        if isinstance(original, dict) and isinstance(original.get("cache"), dict):
            original_control = original["cache"]
        headers = proxy_request.get("headers")
        if isinstance(headers, Mapping):
            for name, value in headers.items():
                if str(name).lower() == "cache-control" and isinstance(value, str):
                    directives = {part.split("=", 1)[0].strip().lower() for part in value.split(",")}
                    if directives & {"no-cache", "no-store"}:
                        return None
    control = data.get("cache", {})
    if control is None:
        control = {}
    if not isinstance(control, dict):
        return None
    if "use-cache" in control and control["use-cache"] is not True:
        return None
    if mode == "opt_in" and control.get("use-cache") is not True and original_control.get("use-cache") is not True:
        return None
    scope = authenticated_scope(auth, team_pool_ids)
    if scope is None or not is_static_text_chat(data, call_type):
        return None
    ttl = control.get("ttl", MAX_TTL)
    max_age = control.get("s-maxage", control.get("s-max-age", MAX_TTL))
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (ttl, max_age)):
        return None
    return CacheContext(scope, revision, min(ttl, MAX_TTL), min(max_age, MAX_TTL))


def finalize_response_cache(data: dict, context: CacheContext | None, call_type: str) -> None:
    """Called at the deployment barrier, before both lookup and cache writes."""
    disable_response_cache(data)
    if context is None or not is_static_text_chat(data, call_type):
        return
    effective = {key: value for key, value in data.items() if key not in _BOOKKEEPING}
    try:
        request_hash = _digest([context.revision, call_type, effective])
    except (TypeError, ValueError, UnicodeError):
        return
    namespace = f"k3-response-v1:{context.scope}"
    key = f"{namespace}:{request_hash}"
    data["caching"] = True
    data["cache"] = {
        "use-cache": True, "namespace": namespace, "ttl": context.ttl,
        "s-maxage": context.max_age,
    }
    # LLMCachingHandler honors this before get_cache_key and copies it to the
    # logging object's internal preset_cache_key for subsequent cache writes.
    data["cache_key"] = key


def cached_usage_details(original_response: object) -> dict:
    """Recover saved usage details lost by LiteLLM 1.102.0 stream replay.

The cache handler saves str(cached_result) on the server's Logging object
before replay converts usage to just three totals. Read that snapshot once,
off-loop, without another Redis lookup or any evaluation of executable code.
No response text or credentials are logged, including on malformed snapshots.
"""
    if isinstance(original_response, str):
        try:
            try:
                original_response = json.loads(original_response)
            except ValueError:
                original_response = ast.literal_eval(original_response)
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return {}
    if not isinstance(original_response, dict):
        return {}
    usage = original_response.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        name: usage[name] for name in ("prompt_tokens_details", "completion_tokens_details")
        if isinstance(usage.get(name), dict)
    }
