# Gateway workload recovery and prefix diagnostics

## Active policy

The recovery guard budgets local requests by rendered input length and granted
output allowance. It complements the existing per-class admission policy.

| Control | Initial value |
|---|---:|
| Large-context threshold | 65536 tokens |
| Concurrent large-context requests | 4 |
| Long-output threshold | 2048 granted tokens |
| Concurrent long-output requests | 2 |
| Total reserved input + output | 1048576 tokens |
| Projected KV soft limit for heavy work | 85% |
| Projected KV limit for small/short work | 97% |

A request can consume both the large-context and long-output budgets. Input
reservations cover the full context: seeing a previous fingerprint does not
establish that its GPU cache entries remain resident. Heavy requests are also
deferred when an engine queue or unowned engine work is observed. Rejections
include `Retry-After`; output allowances are not reduced to simulate a speedup.

The shared proxy-key ceiling was temporarily reduced from 42 to 6 during
recovery and restored to **42** after deployment. The engine's 4096-token batch
profile and existing output-class ceilings remain the reference settings.

## Accurate counting and the preserved-thinking bug

Standard text/tool requests are counted with the local engine's `/tokenize`
endpoint using the effective normalized payload. LiteLLM defers the local
context-window decision to this gateway path while retaining class-output caps.
The gateway enforces the context window before inference.

Live validation exposed a critical difference in the installed vLLM build:

- `ChatCompletionRequest._normalize_messages_before` maps `reasoning_content`
  to `reasoning` **before** validating message types.
- `TokenizeChatRequest` does not perform that mapping, so preserved thinking
  can be discarded during validation.
- A live request counted as 67018 tokens reached inference with 241476 tokens.

`tokenize_projection()` now mirrors inference's normalization on a detached
message dictionary, including explicit `reasoning` precedence and the native
reasoning-effort template kwargs. The actual inference payload is preserved.

Synthetic parity checks against real inference established:

| Case | Uncorrected `/tokenize` | Corrected count | Inference input |
|---|---:|---:|---:|
| Preserved reasoning history | 133 | 4334 | 4334 |
| Tool history with reasoning | 238 | 4439 | 4439 |
| Tool history with null content | 234 | 4435 | 4435 |

Explicit-primary reasoning and root-effort controls also matched. Additional
gateway checks matched a 70137-token thinking history and accepted a valid
approximately 150k-token code request through LiteLLM that the former character
estimate rejected.

If completed inference contradicts a token count, the inspector enters
conservative quarantine and discards its learned history. Subsequent requests
receive conservative reservations until the discrepancy is reviewed and the
gateway is reloaded. This prevents another undercount from silently expanding
admitted work.

## Conservative formats

Multimodal inputs, forced/constrained generation formats, truncation, dynamic
message-level tools and other unsupported inspection shapes retain the existing
inference behavior but reserve a full configured context window. Completion
batches reserve that amount for each prompt, including batches of token-ID
lists. These cases are labeled `conservative`; token-ID fingerprints are not
presented as proof of multimodal prefix identity.

Replica controls require the integer 1, so string/bool coercions cannot bypass
the single-replica admission assumption.

## Diagnostics and metrics

`GET http://127.0.0.1:8002/diagnostics/prefix` is a loopback-only endpoint with a
bounded ring of 1024 events. It records counts, completion status and
process-keyed HMAC fingerprints, never prompt text, token IDs or response text.

Recent completed-prefix history is limited to 4096 entries and 15 minutes.
Fingerprints are scoped to the gateway/model and reset when an engine epoch
change is observed. Recorded replay boundaries require a completed response
with matching prompt usage; incomplete streams do not teach history.

`prior_completed_prefix_tokens` means a matching prefix completed earlier in
this diagnostic scope. It does **not** mean that the corresponding KV/KDA state
must still be cached. Comparing this field with actual cached-token usage helps
separate previously seen inputs from entirely new prefixes. Exact miss causes
still require interpreting eviction/checkpoint behavior and engine restarts.

Prometheus additions include:

- `k3_gateway_workload_active_requests`
- `k3_gateway_workload_reserved_tokens`
- `k3_gateway_workload_large_context_requests`
- `k3_gateway_workload_long_output_requests`
- `k3_gateway_workload_rejections_total{reason=...}`
- `k3_gateway_inspected_prompt_tokens_total`
- `k3_gateway_inspected_cached_tokens_total`
- `k3_gateway_token_count_mismatches_total`
- `k3_gateway_tokenizer_quarantined`
- `k3_gateway_prior_completed_prefix_misses_total`

## Deployment

Enable `K3_WORKLOAD_GUARD=1` on both gateway and LiteLLM. The remaining controls
are defined in `redesign/deploy/profiles/workload-guard.conf`. The active systemd
overrides are:

- `/etc/systemd/system/k3-gateway.service.d/80-workload-guard.conf`
- `/etc/systemd/system/k3-litellm.service.d/80-workload-guard.conf`

Reload both services after draining. Enable/disable the two sides together so
the proxy's deferred context check agrees with gateway inspection. The earlier
UI/database and pricing configuration is retained.

Original serving modules were backed up under
`/root/.config/k3-operator/workload-recovery-20260921T162940Z/`. The temporary
proxy limit has a restore lease at
`/root/.config/k3-operator/heavy-recovery-20260921.json` (restored). Temporary
NGINX inference-admission pauses were removed after checks.

## Verification and observed limits

- **304 tests passed**, including concurrent admission/release, duplicate
  release, batching, malformed replica controls, protocol completeness,
  diagnostic privacy, epoch changes and counting quarantine.
- Ten bounded real gateway checks plus a true-context-limit rejection passed:
  text cold/repeat/changed-prefix, automatic tools, default thinking, preserved
  thinking/tool history, valid large input, forced tool and image requests.
- A real admission test accepted four simultaneous 70157-token, one-output-token
  requests, rejected the fifth at the large-context limit, and released all
  reservations. Engine counters matched the four accepted requests exactly.
- The repeated-text probe reused 2304 tokens; changing its early prefix changed
  the fingerprint and produced a miss.
- An independent Docker restart event at **16:45 UTC**, outside the gateway
  reload commands, reset the GPU cache during observation. Engine startup time
  must be considered when interpreting post-rollout cache rates.
- The initial live rollout exposed the reasoning-count mismatch above; its
  measurements are not accepted as evidence of successful capacity control.
- After correction, the first 90-second live window was light: one 92-input-token
  request generated 4096 tokens, with zero queue/preemptions and about 13.8 ms
  mean decode token time. This is an availability check, **not** a matched
  full-load throughput comparison. A 92-token prompt cannot produce a full
  768-token prefix-cache hit.

Evidence is retained under `/tmp/opencode/`, including
`reasoning-token-parity-20260921.json`,
`workload-guard-verification-20260921.json`, and
`live-heavy-limit-20260921.json`.

## Read-only follow-up: 17:45–17:46 UTC

The deployed guard was rechecked on both serving containers. The gateway still
has four large-context slots, two long-output slots, a 1048576-token reservation
budget and an 85% heavy-work projected-KV limit. All three readiness endpoints
returned HTTP 200. The primary key was independently read back at **42** parallel
requests, 3000000 TPM and 3000 RPM, with its recovery lease marked restored.
NGINX configuration validation passed and the temporary pause include is absent.
Database-management support and the requested pricing were verified for all four
local K3 aliases.

There were still 19 diagnostic events, zero completed token-count mismatches and
no counting quarantine. The latest event remained the 17:34:36 UTC verification
request. All workload reservations and active-request gauges were zero. The only
cumulative workload rejection was the previously verified fifth large-context
probe; the follow-up window recorded no new gateway counter activity.

The 30.094-second observation from 17:45:37 to 17:46:07 UTC contained 16 samples:
zero running requests, zero queued requests, zero reported active KV usage and
no preemptions, generated tokens or completions. The accompanying database
snapshot found no completions in its last five minutes or failures in its last
15 minutes. Engine startup remained 16:45:20 UTC; gateway and LiteLLM startup
remained 17:20:55 UTC. No inference probes were introduced in this follow-up.

This confirms an available, idle deployment with released reservations. It adds
**no full-load throughput or cache-hit comparison**; representative arriving
traffic is still needed to assess those outcomes.

Additional evidence under `/tmp/opencode/`:

- `workload-recovery-followup-live-20260921T1746.json`
- `workload-recovery-followup-db-20260921T1746.json`
- `workload-recovery-followup-verification-20260921T1746.json`
