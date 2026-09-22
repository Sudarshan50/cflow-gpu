"""Render the LiteLLM config. Off-box is added only when K3_OFFBOX_URL is set."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PRICING = """      # USD per token: $3.30 input, $16.50 output, $0.33 cached input per 1M.
      input_cost_per_token: 0.0000033
      output_cost_per_token: 0.0000165
      cache_read_input_token_cost: 0.00000033
"""

MODEL_INFO = f"""    model_info:
      base_model: azure_ai/FW-Kimi-K3
{PRICING}      mode: chat
      supports_vision: true
      supports_function_calling: true
      supports_tool_choice: true
      supports_parallel_function_calling: true
      supports_reasoning: true
      max_input_tokens: 262144
      max_output_tokens: 32768
"""

BASE = f"""model_list:
  - model_name: FW-Kimi-K3
    litellm_params:
      model: openai/FW-Kimi-K3
      api_base: http://127.0.0.1:8002/v1
      api_key: not-needed
{PRICING}      # The gateway serves Chat; native /responses would 404 and trigger cooldown.
      use_chat_completions_api: true
{MODEL_INFO}  - model_name: kimi-k3
    litellm_params:
      model: openai/kimi-k3
      api_base: http://127.0.0.1:8002/v1
      api_key: not-needed
{PRICING}      use_chat_completions_api: true
{MODEL_INFO}  - model_name: moonshotai/Kimi-K3
    litellm_params:
      model: openai/moonshotai/Kimi-K3
      api_base: http://127.0.0.1:8002/v1
      api_key: not-needed
{PRICING}      use_chat_completions_api: true
{MODEL_INFO}  - model_name: default
    litellm_params:
      model: openai/FW-Kimi-K3
      api_base: http://127.0.0.1:8002/v1
      api_key: not-needed
{PRICING}      use_chat_completions_api: true
{MODEL_INFO}"""

OFFBOX = """
  - model_name: p1-offbox
    litellm_params:
      model: openai/{model}
      api_base: {url}
      api_key: {key}
"""

SETTINGS = """
litellm_settings:
  callbacks:
    - redesign.tenancy.callback.tenancy
  drop_params: true
  enable_caching_on_provider_specific_optional_params: true
  request_timeout: 900
{cache}

router_settings:
  num_retries: 0
  timeout: 900
{router_redis}{fallbacks}

general_settings:
  store_model_in_db: true
  allow_requests_on_timeout: false
  # Preserve streaming usage chunks for portals; explicit include_usage:false wins.
  always_include_stream_usage: true
{coordination}
"""

REDIS = """
  cache: true
  enable_redis_auth_cache: true
  cache_params:
    type: redis
    # The authenticated callback enables eligible static text by default.
    # Keep the SDK default off until that callback has built a scoped key.
    # Auth/coordination Redis remains active for all requests.
    mode: default_off
    supported_call_types: [acompletion]
    ttl: 300
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
    password: os.environ/REDIS_PASSWORD
"""

ROUTER_REDIS = """  redis_host: os.environ/REDIS_HOST
  redis_port: os.environ/REDIS_PORT
  redis_password: os.environ/REDIS_PASSWORD
"""

COORDINATION = """  use_redis_transaction_buffer: true
  coordination_redis:
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
    password: os.environ/REDIS_PASSWORD
"""


def render(
    offbox_url: str = "",
    offbox_key: str = "not-needed",
    offbox_model: str = "offbox",
    redis_host: str = "",
) -> str:
    text = BASE
    fallbacks = ""
    if offbox_url.strip():
        text += OFFBOX.format(
            model=offbox_model or "offbox",
            url=offbox_url.rstrip("/"),
            key=offbox_key or "not-needed",
        )
        fallbacks = (
            "  fallbacks:\n"
            "    - FW-Kimi-K3: [p1-offbox]\n"
            "    - kimi-k3: [p1-offbox]\n"
            "    - moonshotai/Kimi-K3: [p1-offbox]\n"
            "    - default: [p1-offbox]\n"
        )
    redis = bool(redis_host.strip())
    text += SETTINGS.format(
        cache=REDIS if redis else "",
        router_redis=ROUTER_REDIS if redis else "",
        fallbacks=fallbacks,
        coordination=COORDINATION if redis else "",
    )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.tenancy.render_config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(
            os.environ.get("K3_OFFBOX_URL", ""),
            os.environ.get("K3_OFFBOX_API_KEY", "not-needed"),
            os.environ.get("K3_OFFBOX_MODEL", "offbox"),
            os.environ.get("REDIS_HOST", ""),
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
