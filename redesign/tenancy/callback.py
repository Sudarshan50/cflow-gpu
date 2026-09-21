"""LiteLLM callback. Loaded only by the proxy process."""

from __future__ import annotations

import asyncio
import copy
import os

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import Usage

from redesign.gateway.media import normalize_payload

from .cache_policy import (
    CACHE_MODES,
    CONTEXT_ATTR,
    cached_usage_details,
    capture_context,
    disable_response_cache,
    finalize_response_cache,
)
from .policy import TenancyPolicy

_POLICY = TenancyPolicy(
    max_model_len=int(os.environ.get("K3_MAX_MODEL_LEN", "262144")),
    offbox_configured=bool(os.environ.get("K3_OFFBOX_URL", "").strip()),
)


class K3TenancyCallback(CustomLogger):
    def __init__(
        self,
        policy: TenancyPolicy | None = None,
        *,
        team_pool_ids: frozenset[str] | None = None,
        cache_revision: str | None = None,
        cache_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy or _POLICY
        self.team_pool_ids = team_pool_ids if team_pool_ids is not None else frozenset(
            team.strip()
            for team in os.environ.get("K3_RESPONSE_CACHE_TEAM_IDS", "").split(",")
            if team.strip()
        )
        self.cache_revision = cache_revision or os.environ.get("K3_RESPONSE_CACHE_REVISION", "1")
        self.cache_mode = cache_mode if cache_mode is not None else os.environ.get("K3_RESPONSE_CACHE_MODE", "static")
        if self.cache_mode not in CACHE_MODES:
            raise ValueError("K3_RESPONSE_CACHE_MODE must be static, opt_in, or off")

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if not isinstance(data, dict):
            return data
        if data.get("stream") is True:
            options = data.get("stream_options")
            if options is None:
                data["stream_options"] = {"include_usage": True}
            elif isinstance(options, dict):
                data["stream_options"] = {"include_usage": True, **options}
        context = capture_context(
            data, user_api_key_dict, call_type,
            team_pool_ids=self.team_pool_ids, revision=self.cache_revision, mode=self.cache_mode,
        )
        disable_response_cache(data)
        logging_obj = data.get("litellm_logging_obj")
        if isinstance(logging_obj, Logging):
            # A server-only attribute, not serializable request metadata. It
            # survives routing on the request's logging object and cannot be
            # forged with JSON. Missing context at the SDK barrier fails closed.
            setattr(logging_obj, CONTEXT_ATTR, context)
        # Downloads and Pillow decode/resize are blocking. Normalize only a
        # detached payload: cancellation must not leave a worker mutating the
        # request (or deep-copy LiteLLM's logging/client/lock objects).
        normalized = copy.deepcopy({
            key: data[key] for key in (
                "messages", "tools", "functions", "extra_body", "chat_template_kwargs",
                "thinking_effort", "reasoning_effort", "cache_salt", "prompt_cache_key",
                "kv_cache_salt", "priority",
            ) if key in data
        })
        original_keys = tuple(normalized)
        await asyncio.to_thread(normalize_payload, normalized)
        for key in original_keys:
            data.pop(key, None)
        data.update(normalized)
        decision = self.policy.apply(data, str(data.get("model") or "FW-Kimi-K3"))
        if decision.reject:
            # A bare exception surfaces as 500, and a portal reads that as the
            # model being down and retries. An unservable prompt is the
            # caller's input error.
            raise HTTPException(status_code=400, detail=decision.reject)
        data["metadata"] = {
            **(data.get("metadata") or {}),
            "k3_class": decision.traffic_class,
            "k3_priority": decision.priority,
            "k3_offbox": decision.routed_off_box,
        }
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        logging_obj = kwargs.get("litellm_logging_obj")
        context = getattr(logging_obj, CONTEXT_ATTR, None) if isinstance(logging_obj, Logging) else None
        finalize_response_cache(kwargs, context, call_type)
        return kwargs

    async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
        logging_obj = request_data.get("litellm_logging_obj")
        cache_details = logging_obj.caching_details if isinstance(logging_obj, Logging) else None
        details = {}
        if cache_details and cache_details.get("cache_hit") is True:
            details = await asyncio.to_thread(
                cached_usage_details, logging_obj.model_call_details.get("original_response"),
            )
        async for chunk in response:
            usage = getattr(chunk, "usage", None)
            if details and isinstance(usage, Usage):
                # Fill only missing fields: an upstream fix should take priority.
                missing = {name: value for name, value in details.items() if getattr(usage, name, None) is None}
                if missing:
                    chunk.usage = Usage(**{**usage.model_dump(), **missing})
            yield chunk


tenancy = K3TenancyCallback()
