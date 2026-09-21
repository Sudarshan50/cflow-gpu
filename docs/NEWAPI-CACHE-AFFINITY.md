# NewAPI channel affinity and K3 cache accounting

Research snapshot: **2026-09-20**. This document records findings and proposed
configuration; it does not mean the remote portal has been configured or tested.

NewAPI source is pinned to commit
[`9a0be8750a6d736d9692535ed2cd68f8eec46529`][commit], the upstream `main`
revision inspected during research. **The remote portal URL, authentication,
version, and effective settings are unknown.** Check its version against this
schema before applying the example. A current upstream default is not evidence
of the remote portal's configuration.

## 1. CACHE first: distinguish three mechanisms

| Mechanism | What is reused | Evidence needed |
|---|---|---|
| NewAPI request/channel affinity | A session key's selected channel ID | Portal-side affinity-selection diagnostics |
| vLLM GPU prefix cache | Previously computed prompt state | Fresh engine execution and actual cached-token usage |
| LiteLLM response replay | A previously generated response | Effective response-cache controls and cache-hit/execution diagnostics |

The inspected local engine configuration, `/scratch/hf/config.yaml` mounted as
`/hf/config.yaml`, enables `enable-prefix-caching`,
`enable-prompt-tokens-details`, and `prefix-match-unit: 128`.
`enable-force-include-usage` was unset. The generated LiteLLM configuration at
`/scratch/deploy-state/litellm.yaml` enables Redis response caching as well as
authentication caching. Installed LiteLLM was **1.102.0**.

A fast response is not proof of GPU prefix reuse. A replay can carry the cached
usage from its original execution without running the engine again. Conversely,
a stable channel does not guarantee a prefix hit: token-prefix identity, cache
residency, and any backend routing behind that channel still matter.

## 2. Request direction and ownership

The inspected request path is:

```text
client -> remote NewAPI portal [selects its upstream channel]
       -> our nginx :443 -> LiteLLM :4000 -> K3 gateway :8002 -> vLLM :8001

response/usage travels back in the reverse direction
```

Our endpoint is an **upstream provider from NewAPI's perspective**. The local
LiteLLM is downstream of the calling portal in request order. There was no
configured local LiteLLM -> remote NewAPI model route in the inspected deployment.

Container, systemd, and process-name inventories found no local NewAPI/OneAPI
instance. The local `/` UI is LiteLLM. The calling NewAPI portal is external to
this inspected stack; its actual host and access method were not established.
Local changes cannot enforce that remote portal's channel choice.

Local evidence from the research snapshot:

- `/etc/nginx/conf.d/k3.conf:12-17,50-65`: LiteLLM upstream, gateway backup,
  and `/v1/` forwarding.
- `/usr/local/lib/k3/redesign/tenancy/render_config.py`: model aliases point to
  `http://127.0.0.1:8002/v1`; generated config contained no off-box model.
- Deployed `redesign/gateway/media.py`, `server.py`, and `engine.py` differed
  from the repository copies. These findings used deployed files, not an
  assumption that working-tree changes were running.

## 3. Valid NewAPI affinity setting

### Defaults do not match incoming Kimi Chat requests

The pinned [setting schema and defaults][settings] enable affinity with a
3600-second default TTL and 100000 maximum entries. The two supplied rules are:

| Rule | Incoming model regex | Incoming path regex | Key source |
|---|---|---|---|
| Codex | `^gpt-.*$` | `/v1/responses` | Body `prompt_cache_key` |
| Claude | `^claude-.*$` | `/v1/messages` | Body `metadata.user_id` |

An incoming `FW-Kimi-K3` request to `/v1/chat/completions` matches neither.
Matching uses the **model and path received by NewAPI**, before the selected
channel's model mapping and request conversion. If an incoming Claude alias is
mapped to Kimi later, the incoming Claude alias is what the rule sees. Add the
portal's actual incoming aliases to the rule; do not infer them from vLLM's
served model name.

### Full setting value: Kimi sessions, prefer mode

The following is a complete example **value of `channel_affinity_setting`**,
matching `ChannelAffinitySetting` and `ChannelAffinityRule` in the pinned Go
source. It is not an HTTP update envelope or a local LiteLLM configuration.
Merge this rule into the remote portal's existing settings. The example's
`rules` array contains only Kimi; replacing an existing array with it would
remove the portal's other rules.

```json
{
  "enabled": true,
  "session_mode": "prefer",
  "switch_on_success": true,
  "keep_on_channel_disabled": false,
  "max_entries": 100000,
  "default_ttl_seconds": 3600,
  "rules": [
    {
      "name": "kimi-chat-session",
      "model_regex": ["^(FW-Kimi-K3|kimi-k3|moonshotai/Kimi-K3)$"],
      "path_regex": ["^/v1/chat/completions$"],
      "user_agent_include": [],
      "key_sources": [
        {"type": "request_header", "key": "X-Session-ID"},
        {"type": "gjson", "path": "prompt_cache_key"}
      ],
      "value_regex": "",
      "ttl_seconds": 3600,
      "skip_retry_on_failure": false,
      "session_mode": "prefer",
      "include_using_group": true,
      "include_model_name": true,
      "include_rule_name": true
    }
  ]
}
```

The upstream UI's **Rules JSON** editor expects only the `rules` array, not the
whole object. It serializes rule objects without its UI-only `id` field;
see the [editor source][editor]. Configure global fields through the matching
settings controls/API for the portal's version. This document does not assume
an unknown remote admin endpoint's authentication or update payload.

### Schema and ordering nuances

Verified against [key extraction and lookup][affinity] and
[`EffectiveSessionMode`][policy]:

- Source types are exactly `request_header`, `gjson`, `context_int`, and
  `context_string`. Headers use `key`; JSON body extraction uses `path`.
  `X-Session-ID` is our chosen header name, not a built-in magic NewAPI header.
- Sources are ordered fallbacks: the **first nonempty** value wins. They are
  not concatenated. If both example fields are present, the header wins.
- Keep the identifier **stable for the whole session**, not fresh per request.
  Generate a new opaque identifier for a new session. If both fields are sent,
  keep their values equal. A changing request ID defeats affinity; a single
  shared portal/user ID intentionally collapses many sessions into one binding.
- Group/model/rule scoping does not automatically add a user or token identity.
  Use unique opaque session IDs across callers sharing that scope.
- The binding suffix is `rule:model:using_group:session_value`, under
  `new-api:channel_affinity:v1`. Changing model alias or group creates a separate
  binding. For auto-group routing, the lookup's group can be `auto`, rather than
  the concrete group eventually selected.
- The **first qualifying rule** owns the lookup, even on a binding-cache miss.
  Place this specific rule before any broader overlapping rule. An empty model
  regex list does not mean all models. These regexes are anchored intentionally.
- `value_regex` validates the chosen value; it does not extract a capture group.
  If the first nonempty source fails validation, lookup proceeds to another
  rule, not another source in the same rule. The example leaves it unrestricted.
- Header matching is case-insensitive, but underscores and hyphens differ:
  `Session_id` is not `Session-Id`. Default Codex header passthrough lists are
  separate from its affinity extractor; forwarding session headers does not
  make them default affinity keys.
- Explicit rule `session_mode: "prefer"` wins over the legacy
  `skip_retry_on_failure` boolean. Setting the boolean false also expresses the
  intended legacy retry behavior. Merely changing the global mode does not
  override an existing legacy strict rule; `"inherit"` explicitly uses global
  mode, while an absent rule mode uses the legacy boolean.

### Availability and storage

[Channel selection][selection] checks an explicit pin first, then initial-attempt
affinity, then normal eligible-channel selection. A preferred channel must be
enabled, satisfy request filters, and serve the requested group/model.

`prefer` is useful for live availability: an unusable binding can be cleared and
another eligible channel selected. `strict` can instead return
`503 strict_session_binding_unavailable`. Prefer mode permits ordinary retries;
it does not grant extra retries or guarantee a successful alternative. The
upstream default retry count is zero. `switch_on_success: true` allows a later
successful retry channel to become the binding.

Bindings target channel IDs, not a particular credential or engine worker.
Recording refreshes their TTL. The affinity cache uses Redis when enabled,
otherwise process-local memory. A multi-instance portal needs consistent shared
affinity storage; our LiteLLM Redis is not evidence that the remote portal has it.

## 4. Local stripping occurs after remote selection

At the research snapshot, deployed
`/usr/local/lib/k3/redesign/tenancy/callback.py:25-41` calls
`redesign.gateway.media.normalize_payload`. The deployed normalizer removes
`cache_salt`, `prompt_cache_key`, `kv_cache_salt`, and client `priority`, both at
top level and inside `extra_body`. It also canonicalizes tool order and
normalizes media/thinking parameters. The gateway normalizes again.

The gateway's `engine.py:185-204` builds new outbound headers rather than
forwarding incoming session headers. Its response path relays body bytes,
including usage/SSE data, while filtering transport-related headers. LiteLLM
also has its own request/response conversion; generic client-header forwarding
is opt-in and was not enabled in the inspected generated configuration.

These operations happen **after NewAPI has selected this endpoint**. Removing a
field here cannot erase the earlier request body/header stored in NewAPI's
request context. Restoring the field here cannot create a missing remote rule
or force this channel to be selected. The remote rule can consume the session
key without forwarding it to the engine. Likewise, a post-selection NewAPI
parameter override cannot create a missing key for its own earlier lookup.

The inspected vLLM Chat protocol defines `cache_salt` as a GPU prefix-cache salt
(`entrypoints/openai/chat_completion/protocol.py:437-446`). Its Responses
`prompt_cache_key` field explicitly says it is unimplemented and ignored
(`entrypoints/openai/responses/protocol.py:205-210`); the inspected Chat protocol
has no declared field by that name. Do not treat these fields as interchangeable
without tracing the deployed engine version.

Local failures can influence subsequent remote retries or binding behavior,
but no causal path was established from local key stripping to missing remote
preselection affinity.

## 5. Usage, pricing, and streaming

### Standard response contract

For OpenAI-compatible Chat Completions, preserve this usage structure
(illustrative counts):

```json
{
  "usage": {
    "prompt_tokens": 1024,
    "completion_tokens": 8,
    "total_tokens": 1032,
    "prompt_tokens_details": {"cached_tokens": 896}
  }
}
```

`prompt_tokens` includes the cached subset; do not add cached tokens to it.
For streaming, explicitly request `"stream_options": {"include_usage": true}`.
A normal ending has a finish-reason chunk, a cumulative usage chunk (normally
`choices: []`), then `data: [DONE]`. The inspected vLLM emits cached-token details
in that final usage chunk when usage is requested.

NewAPI's [normal Chat relay][compatible] defaults to forcing upstream stream
usage for supported channel types (`FORCE_STREAM_OPTION=true` in
[`common/init.go`][init]). **Body passthrough forwards the stored body instead
of this reconstructed request**, so the injected stream options may not reach
the provider. Explicit usage in the original request covers that case. Also
inspect the actual channel adapter: conversion behavior depends on channel type.

Installed LiteLLM's `llms/openai/openai.py:1189-1197` forwards explicit stream
options but does not add them by default for every custom API base. Missing
upstream usage can cause NewAPI to estimate token counts, which cannot establish
actual GPU cached-token reuse. Its [stream handler][stream] examines final usage
and has a limited second-last-frame fallback; do not rely on arbitrary earlier
usage frames surviving as the final accounting snapshot.

### What the portal metrics mean

From [affinity usage counters][affinity], [quota accounting][quota], and
[log fields][loginfo]:

- Affinity usage `hit` means a usage observation reported positive cached tokens
  or `prompt_cache_hit_tokens`. It is **not a channel-affinity lookup hit**.
- Usage `total` counts observations with the required affinity context, not all
  traffic to the engine. Usage can be positive even when the binding lookup
  missed; absence of a matching rule can prevent these per-session stats.
- `cached_tokens` prefers `prompt_tokens_details.cached_tokens`, then
  `input_tokens_details.cached_tokens`. `prompt_cache_hit_tokens` is a separate
  signal; do not double-count two representations of the same reused tokens.
- OpenAI token-rate mode is cached/prompt; Claude mode is
  cached/(prompt+cached). The counters are TTL-backed with TTL refreshed on
  observation, not fixed time buckets. Their aggregation key is rule/group/key
  fingerprint, without model, even if the routing binding includes model.
- `other.cache_tokens` is the normalized cached-token count used in logging.
- **`other.cache_ratio` is a pricing multiplier, not a measured hit rate.**
  The [fallback cache-price ratio][ratio] is `1.0`. A correctly reported GPU hit
  therefore need not produce a discount unless portal pricing is configured.
- Full-response replay can reuse an old usage object. Verify engine execution
  separately before attributing a reported count to this call's GPU work.

### Failed streams are not ordinary HTTP failures

At the pinned revision, [retry decisions][retry] suppress retries for strict
sessions and otherwise consult retry budget, constraints and error/status rules.
But the OpenAI Chat stream handler normally returns usage and no relay error
after scanning, even when stream status records an abnormal end. Request-policy
logging can mark `stream_not_successful`, while the
[distributor][distributor] records affinity based on HTTP status `<400`, not that
success flag. Thus an HTTP-200 failed stream can leave or refresh a binding
without triggering ordinary relay retry or affinity invalidation. Confirm the
remote version before relying on different behavior.

## 6. Bounded two-turn verification

This is a procedure to run deliberately, not a test executed while writing this
document. It submits **two sequential Chat requests**, changes only a short
synthetic suffix, uses one session identifier, requests usage, and asks for no
response-cache reads or writes. It prints only status and numeric usage/terminal
facts, never generated text, credentials, or customer identifiers.

Prerequisites:

1. Obtain the remote portal's actual URL and an inference credential through the
   operator. Admin diagnostics may use separate authentication; it is unknown.
2. Verify the remote version/schema, incoming alias, effective rule order,
   group/model eligibility, and absence of a pin that overrides affinity.
3. Bound portal/provider automatic retries for this probe. The script makes no
   client retries, but that alone cannot cap attempts inside the portal. Observe
   the queue first; stop on any failure or queue growth. Do not flush caches or
   induce production faults.
4. Confirm response-cache bypass reaches LiteLLM. The body `cache` object below
   is a **LiteLLM extension**, not an affinity-setting field or a guaranteed
   NewAPI Chat passthrough field. NewAPI conversion may drop it. Use a verified
   test-scoped passthrough/parameter override or effective test-key controls at
   LiteLLM, and check engine execution. Do not enable body passthrough blindly:
   it also bypasses normal model/body conversion. The HTTP `Cache-Control`
   header alone is not proof of LiteLLM response-cache bypass.

For an isolated test channel using normal conversion, NewAPI's documented
[parameter override operations][channel-docs] can insert the LiteLLM extension
after conversion. This is a **separate test-channel parameter override**, not a
field to add to the affinity-setting object. Merge with existing operations if
needed; do not overwrite unrelated channel configuration:

```json
{
  "operations": [
    {
      "mode": "set",
      "path": "cache",
      "value": {"no-cache": true, "no-store": true}
    }
  ]
}
```

LiteLLM `proxy/litellm_pre_call_utils.py:1666-1674` can replace request cache
controls with key-level controls. Verify the effective settings, not merely the
payload sent. The changed suffix and fresh synthetic run prefix additionally
avoid replaying an identical prior test request.

The example uses the local Python environment's installed `httpx`. Replace the
placeholder environment values, with `NEWAPI_BASE_URL` ending in `/v1`:

```bash
export NEWAPI_BASE_URL='https://REPLACE-WITH-PORTAL-HOST/v1'
export NEWAPI_API_KEY='REPLACE-WITH-TEST-INFERENCE-CREDENTIAL'
export NEWAPI_MODEL='FW-Kimi-K3'

/usr/local/lib/k3/venv/bin/python -B - <<'PY'
import asyncio
import json
import os
import uuid

import httpx


async def main():
    base = os.environ["NEWAPI_BASE_URL"].rstrip("/")
    key = os.environ["NEWAPI_API_KEY"]
    if "REPLACE-" in base or "REPLACE-" in key:
        raise SystemExit("Replace the portal URL and test credential placeholders.")

    # New once per two-turn run; unchanged between the requests.
    session = "synthetic-" + uuid.uuid4().hex
    prefix = ("Synthetic cache probe " + session + ".\n"
              + "The blue square is beside the green circle.\n" * 64)
    headers = {
        "Authorization": "Bearer " + key,
        "X-Session-ID": session,
        "Cache-Control": "no-cache, no-store",
    }
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
        for turn in (1, 2):
            payload = {
                "model": os.environ.get("NEWAPI_MODEL", "FW-Kimi-K3"),
                "messages": [{
                    "role": "user",
                    "content": prefix + f"Probe turn {turn}: reply with OK.",
                }],
                "prompt_cache_key": session,
                "stream": True,
                "stream_options": {"include_usage": True},
                "cache": {"no-cache": True, "no-store": True},
                "max_tokens": 16,
                "temperature": 0,
            }
            usage = None
            done = False
            finished = False
            size = 0
            # Total wall-clock bound per turn, including connection and SSE.
            async with asyncio.timeout(30):
                async with client.stream(
                    "POST", base + "/chat/completions",
                    headers=headers, json=payload,
                ) as response:
                    print(json.dumps({"turn": turn, "status": response.status_code}))
                    if response.status_code != 200:
                        raise RuntimeError("non-200 response; stopped")
                    async for line in response.aiter_lines():
                        size += len(line.encode("utf-8"))
                        if size > 131072:
                            raise RuntimeError("SSE size bound exceeded; stopped")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        if not data:
                            continue
                        event = json.loads(data)
                        if event.get("error"):
                            raise RuntimeError("SSE error; stopped")
                        for choice in event.get("choices", []):
                            finished |= bool(choice.get("finish_reason"))
                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]
            details = (usage or {}).get("prompt_tokens_details") or {}
            counts = {
                name: (usage or {}).get(name)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            counts["cached_tokens"] = details.get("cached_tokens")
            # Whitelist numeric usage only, even if a remote response is malformed.
            counts = {k: v if type(v) is int else None for k, v in counts.items()}
            print(json.dumps({"turn": turn, "done": done,
                              "finish_reason_present": finished, **counts}))
            if not done or not finished or any(v is None for v in counts.values()):
                raise RuntimeError("missing terminal/usage facts; stopped")
            if turn == 1:
                await asyncio.sleep(1)


try:
    asyncio.run(main())
except Exception as exc:
    # Do not print exception bodies, response text, or request URLs/credentials.
    raise SystemExit("Probe stopped: " + type(exc).__name__) from None
PY
```

Interpretation:

- Turn 1 should match the rule and establish a binding; turn 2 should show an
  actual affinity selection for the same synthetic key/group/model.
- Same channel twice is insufficient if there is only one eligible channel or
  random routing happened to repeat. Check the portal's admin affinity details
  and request-policy events. A changed channel can be legitimate prefer-mode
  failover; inspect eligibility and retry evidence.
- Positive turn-2 `cached_tokens` with verified fresh engine execution supports
  GPU prefix reuse. Missing counts are not equivalent to zero; neither result
  by itself proves whether the portal's affinity lookup worked.
- The probe exercises the header extractor because it is first. Body-only
  fallback is a separate optional two-request run with a fresh session and the
  `X-Session-ID` header omitted, not an automatic extra workload.
- Remote read-only diagnostics at the pinned revision include
  `GET /api/option/channel_affinity_cache` and
  `GET /api/log/channel_affinity_usage_cache` (query fields `rule_name`,
  `using_group`, `key_fp`). Use only the synthetic session's diagnostics;
  do not export prompts, credentials, or production customer identifiers.

## 7. Local work versus remote access

Local work can make usage reporting explicit, preserve truthful usage fields,
keep prompt/tool serialization stable, and distinguish response replay from
GPU execution. It cannot force remote NewAPI affinity or select our channel.
Keeping `prompt_cache_key` locally for future observability is separate from
whether it is required by the engine; it is not an established portal-routing
fix.

Remote operator access is needed to verify and configure the portal's incoming
aliases, ordered affinity rules, pins, channel eligibility/type, model mapping,
retry policy, body conversion/passthrough, pricing, and shared affinity storage.
Confirm those facts before attributing a production affinity failure to any
local transformation.

## Sources

All GitHub links below are pinned to the inspected source revision. The public
channel-management documentation is a live documentation page and may change.

[commit]: https://github.com/QuantumNous/new-api/commit/9a0be8750a6d736d9692535ed2cd68f8eec46529
[settings]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/setting/operation_setting/channel_affinity_setting.go
[affinity]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/service/channel_affinity.go
[policy]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/service/request_policy.go
[selection]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/service/channel_select.go
[editor]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/web/src/features/system-settings/general/channel-affinity/index.tsx
[compatible]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/relay/compatible_handler.go
[init]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/common/init.go
[stream]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/relay/channel/openai/relay-openai.go
[quota]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/service/text_quota.go
[loginfo]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/service/log_info_generate.go
[ratio]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/setting/ratio_setting/cache_ratio.go
[retry]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/service/relay_error.go
[distributor]: https://github.com/QuantumNous/new-api/blob/9a0be8750a6d736d9692535ed2cd68f8eec46529/middleware/distributor.go
[channel-docs]: https://docs.newapi.ai/en/docs/guide/console/channel-management
