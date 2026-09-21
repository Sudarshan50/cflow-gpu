"""LiteLLM callback. Loaded only by the proxy process."""

from __future__ import annotations

import asyncio
import copy
import os

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import Usage

from redesign.gateway.clamping import InvalidTokenLimit, validate_token_limits
from redesign.gateway.media import normalize_payload, normalize_controls, MediaValidationError, MediaBusyError
from redesign.gateway.clamping import requested_output_tokens

from .cache_policy import (
    CACHE_MODES,
    CONTEXT_ATTR,
    cached_usage_details,
    capture_context,
    disable_response_cache,
    finalize_response_cache,
    is_static_text_chat,
)
from .policy import TenancyPolicy
from .responses_bridge import (
    BridgeRequestError,
    prepare_chat_request,
    prepare_responses_request,
    validate_responses_request,
)

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
        gateway_policy_url: str | None = None,
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
        self.gateway_policy_url = (gateway_policy_url if gateway_policy_url is not None
                                   else os.environ.get("K3_GATEWAY_POLICY_URL", "")).rstrip("/")
        if self.cache_mode not in CACHE_MODES:
            raise ValueError("K3_RESPONSE_CACHE_MODE must be static, opt_in, or off")

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if not isinstance(data, dict):
            return data
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
        if call_type in ("acompletion", "aresponses"):
            try:
                validate_token_limits(data)
                if call_type == "aresponses":
                    validate_responses_request(data)
            except (BridgeRequestError, InvalidTokenLimit) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return data

    async def async_pre_call_deployment_hook(self, kwargs, call_type):
        logging_obj = kwargs.get("litellm_logging_obj")
        context = getattr(logging_obj, CONTEXT_ATTR, None) if isinstance(logging_obj, Logging) else None
        # The bridge reuses Logging(call_type="aresponses") for its inner Chat
        # call. Dispatch by the hook argument, not Logging.call_type. This flag
        # only DISABLES caching, so a caller forging it cannot gain eligibility.
        bridged = kwargs.get("_skip_responses_api_bridge") is True
        if bridged or not is_static_text_chat(kwargs, call_type):
            context = None
        disable_response_cache(kwargs)
        try:
            if call_type in ("acompletion", "aresponses"):
                validate_token_limits(kwargs)
            if call_type == "aresponses":
                prepare_responses_request(kwargs)
                return kwargs
            if call_type != "acompletion":
                return kwargs
            prepare_chat_request(kwargs)
        except (BridgeRequestError, InvalidTokenLimit) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Router defaults and Responses conversion are now complete. Check
        # eligibility above before media failures can turn images into text.
        try:
            await self._apply_chat_policy(kwargs, bridged=bridged)
        except MediaValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except MediaBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finalize_response_cache(kwargs, context, call_type)
        return kwargs

    async def _apply_chat_policy(self, data, *, bridged):
        gateway_owned = bool(self.gateway_policy_url) and str(data.get("api_base") or "").rstrip("/") == self.gateway_policy_url
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
        await asyncio.to_thread(normalize_controls if gateway_owned else normalize_payload, normalized)
        for key in original_keys:
            data.pop(key, None)
        data.update(normalized)
        if data.get("stream") is True:
            options = data.get("stream_options")
            if options is None or isinstance(options, dict):
                data["stream_options"] = {"include_usage": True, **(options or {})}
        if gateway_owned:
            # Preserve the post-conversion caller/deployment limit. The gateway
            # counts the actual rendered input before applying the sole final
            # default/clamp/admission policy. A heuristic here must not reject or
            # shrink a request that the authoritative counter could have served.
            requested = requested_output_tokens(data)
            metadata_key = "litellm_metadata" if bridged else "metadata"
            data[metadata_key] = {
                **(data.get(metadata_key) or {}),
                "k3_class": None,
                "k3_priority": None,
                "k3_offbox": False,
                "k3_output_policy": {"stage": "gateway_pending", "requested_max_tokens": requested},
            }
            return
        decision = self.policy.apply(data, str(data.get("model") or "FW-Kimi-K3"), route_model=False)
        if decision.reject:
            # A bare exception surfaces as 500, and a portal reads that as the
            # model being down and retries. An unservable prompt is the
            # caller's input error.
            raise HTTPException(status_code=400, detail=decision.reject)
        metadata_key = "litellm_metadata" if bridged else "metadata"
        data[metadata_key] = {
            **(data.get(metadata_key) or {}),
            "k3_class": decision.traffic_class,
            "k3_priority": decision.priority,
            "k3_offbox": decision.routed_off_box,
            "k3_output_policy": {
                "stage": "tenancy",
                "requested_max_tokens": decision.requested_max_tokens,
                "granted_max_tokens": decision.granted_max_tokens,
                "default_output_tokens": decision.default_output_tokens,
                "default_applied": decision.default_applied,
                "clamp_reason": decision.clamp_reason,
                "traffic_class": decision.traffic_class,
            },
        }

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
