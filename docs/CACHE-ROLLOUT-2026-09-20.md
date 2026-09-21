# Cache-first control-plane rollout — 2026-09-20 UTC

## Result

Deployed cache correctness, streaming usage preservation, nonblocking normalization,
global admission enforcement, and upstream cleanup fixes. Verified actual Redis
response replay and GPU prefix reuse separately, through both loopback LiteLLM
and the public HTTPS endpoint. The model was not restarted, preserving its warm
GPU cache.

Remote NewAPI channel affinity remains a portal-side configuration investigation:
there is no NewAPI instance on this host and its remote URL/version/settings were
not provided. See [NEWAPI-CACHE-AFFINITY.md](NEWAPI-CACHE-AFFINITY.md) for the
source-pinned rule schema, exact Kimi example and verification procedure.

## Deployed files and behavior

Source paths below are relative to `/root/cflow-gpu`; installed counterparts are
under `/usr/local/lib/k3`.

- `redesign/tenancy/cache_policy.py`: new canonical exact-response cache policy.
  Keys include effective request/provider generation settings, model/target,
  backend revision, and a hashed authenticated key scope. Untrusted namespace
  and preset-key overrides are removed. Arrays retain their semantic order;
  dictionary serialization is canonical. Unknown non-JSON generation settings
  bypass caching rather than produce an unstable key.
- `redesign/tenancy/callback.py`: authenticated context is captured before
  routing, then keys are finalized after deployment defaults are merged and
  before lookup. Blocking media normalization runs off the async event loop on
  detached request data. Streaming defaults to `include_usage: true` while
  honoring explicit false. Cached-stream token details dropped by LiteLLM
  1.102.0 are restored from the server-side cached response snapshot.
- `redesign/tenancy/render_config.py`: retains Redis authentication and
  coordination, enables provider-specific cache parameters, and configures an
  SDK cache that defaults off until the authenticated callback enables eligible
  requests. Cache TTL is bounded to 300 seconds.
- `redesign/tenancy/policy.py` and `redesign/gateway/server.py`: use
  `max_completion_tokens` precedence consistently, so conflicting token fields
  no longer expand the engine-effective caller limit. Replica checks inspect
  both `n` and `best_of`.
- `redesign/gateway/backpressure.py`: enforces an atomic global ceiling in
  addition to class limits. Runtime ceiling remains **160**; source defaults
  are not evidence of the active setting.
- `redesign/gateway/engine.py`: reads the deployed
  `vllm:kv_cache_usage_perc` metric and gives responses explicit connection
  ownership for cleanup before/after body iteration and setup failure.
- `redesign/gateway/server.py`: release protection covers trace/other
  post-admission exceptions; upstream cleanup covers response-header failures;
  HTTP 5xx responses increment the engine-error counter exactly once.
- `redesign/probe/cache_salt.py`: drains/validates responses and uses the second
  request's actual `usage.prompt_tokens_details.cached_tokens` for the verdict.
  Global hit-counter changes are diagnostic only. Failed/incomplete requests
  cannot pass because unrelated live traffic generated cache hits.
- `/etc/nginx/conf.d/k3.conf` and the source/installed
  `redesign/deploy/deploy.sh` generator: removed the unauthenticated gateway
  backup from the tenancy upstream. Requests no longer bypass authentication
  and cache policy when LiteLLM restarts.

### Response-cache modes

`K3_RESPONSE_CACHE_MODE`:

- `static` (active default): cache eligible fully specified text chat, including
  streaming, without requiring a custom client opt-in.
- `opt_in`: require `cache: {"use-cache": true}` as well as eligibility.
- `off`: disable full-response replay.

Explicit `caching: false`, `use-cache: false`, `no-cache`, `no-store`, and HTTP
`Cache-Control: no-cache/no-store` opt out. Tool/tool-history, media and
stateful requests bypass full-response replay. This does not disable their GPU
prefix caching. Generated LiteLLM logging session IDs and connection pools are
not model-generation parameters and do not partition cache entries.

Authenticated API keys are separate response-cache scopes. Team pooling is
available only through operator allowlist `K3_RESPONSE_CACHE_TEAM_IDS`; no teams
are pooled in the current deployment. `K3_RESPONSE_CACHE_REVISION=1` is current;
bump it after backend weights/templates/defaults change behind the same URL.

The existing GPU salt/tool-normalization policy and engine cache configuration
were not changed in this release. Preexisting source-only media resizing/token
estimation changes were not rolled into this deployment.

## Timeline and validation

- **20:15:46:** installed seven control-plane Python files after source tests.
- **20:16:43:** gateway and LiteLLM restarted. Nginx configuration validated and
  reloaded. Engine remained at its **17:59:06** start.
- **20:21:** first six bounded live synthetic calls proved prefix reuse but
  exposed missing response-cache hits. Actual proxy dispatch injects an
  `aiohttp.ClientSession` in `shared_session`; treating this transport object
  as generation input made key serialization fail and caching bypass.
- Added a regression using actual `route_request()` dispatch and initialized
  connection pools. Excluded only the transport field from the digest.
- **20:28:36:** LiteLLM restarted with the correction; gateway/engine stayed up.
  Offline installed-code tests ran during startup instead of readiness polling.
- **20:28:47–57:** internal bounded probes passed.
- **20:29:22–33:** public HTTPS probes passed.
- **20:34:05:** read-only DB/Redis/runtime verification completed.
- **20:36:40:** installed repaired probe tooling; it is invoked separately and
  required no service restart. The repaired standalone probe was not executed
  against production; actual cache behavior was already established by the
  bounded authenticated smoke checks below.

### Tests

- Initial combined gateway/tenancy/cache/regression suites: **139 tests passed**.
- Installed-layout cache tests and serving-invariant tests passed before first
  restart (23 each at that version).
- After the live connection-pool discovery: **24 cache tests passed**, including
  actual proxy dispatch, request-key stability across pools, tenant/settings
  isolation, explicit bypass, streaming usage and namespace-spoofing cases.
- **19 new prefix-probe tests passed**; related legacy counter and connection
  ownership tests also passed in the focused worker run.
- `git diff --check`, shell syntax validation and `nginx -t` passed.

These suite counts overlap; they are not summed into a unique-test total.
Source tests contain mocked failure-gate output; those expected failure paths
are not a production model correctness-gate run.

## Controlled live cache results

Smoke tool: `/tmp/opencode/k3_cache_smoke.py`. It uses an existing local
authenticated master credential privately, emits only timing/status/usage,
and submits six sequential synthetic requests per invocation with at most
32 output tokens. Actual outputs used eight tokens each. No customer prompts
or credentials were printed. There were three invocations: initial diagnostic,
corrected internal, corrected public. No saturation/load benchmark was run.

### Corrected internal path, `http://127.0.0.1:4000/v1`

| Check | First request | Second request |
|---|---:|---:|
| Prefix test with response reads/writes bypassed | 0/989 cached prompt tokens | **896/989** |
| Exact nonstream response | 1.632 s | **0.009 s** |
| Exact streaming response | 1.704 s | **0.013 s** |

### Public HTTPS path, `https://api.cflowx.in/v1`

| Check | First request | Second request |
|---|---:|---:|
| Prefix test with response reads/writes bypassed | 0/988 cached prompt tokens | **896/988** |
| Exact nonstream response | 1.684 s | **0.014 s** |
| Exact streaming response | 4.681 s | **0.013 s** |

All requests returned HTTP 200 with valid finish reasons. Streams delivered
usage including `prompt_tokens_details.cached_tokens` and `[DONE]`. Replayed
responses preserved response IDs. The two prefix-test requests changed a short
suffix and explicitly bypassed response caching, separating prefix reuse from
identical-response replay.

DB verification confirmed **two response-cache hits in each corrected six-call
cohort**. Hit keys had the expected authenticated scope and two distinct request
hashes per cohort. A bounded Redis SCAN found eight scoped entries across two
scopes, with TTLs of 110–265 seconds. No cache values or key names were dumped.

**Logging limitation:** DB miss keys are independently generated log keys and
do not prove equality with actual lookup keys. One cached-stream hit record
omitted cached-token details even though the public/client SSE contained the
restored details. Client transport correctness and spend-log completeness are
separate follow-ups.

## Post-deploy live traffic

Read-only DB snapshot at 20:34:05.600 UTC, request starts in
**[20:28:36, 20:33:00)**, excluding all authenticated-master probe records:

- 64 retained ended non-master requests, 64 marked success and zero failure.
- Two actual response-cache hits, 59 false, three other/missing flags.
- 1,816,685 prompt tokens; 1,299,968 cached prompt tokens; 17,612 completion
  tokens. These are logged completed cohorts, not interval engine throughput.
- No callback-marked errors or recorded 500s.
- One success row also carried error 499, so success alone does not establish
  complete delivery to the client.

Nginx recorded 14 inference 502s during **20:28:36–44** around the second restart:
11 sub-second failures and three requests started before restart. From
**20:28:45 through 20:33:00**, there were 56 completed inference HTTP 200s and no
inference errors in that log window. This total includes public probes and uses
completion timestamps, unlike the DB cohort.

Passive engine window **20:29:35–20:30:55**, 80.016 seconds:

- Prefix lookup hits/queries: **80.78%**.
- Generation counter rate: **85.16 tokens/s**.
- Event-weighted mean inter-token latency: **107.71 ms**.
- Running 9 → 9; waiting 0 → 0; KV occupancy 27.61% → 13.37%; no preemptions.

Baseline 19:56:43–19:57:58 had 77.57% prefix hits, 71.71 tokens/s and 204.98 ms
ITL. Workload/concurrency and observation cohorts changed, and the later window
may include probes. **These differences are not a causal production speedup
claim.** The directly verified improvement is repeat-request response replay
with correct scoping/settings and intact client-visible cache usage.

## NewAPI affinity: what is known and what remains

Current pinned upstream NewAPI defaults match GPT `/v1/responses` and Claude
`/v1/messages`, not incoming Kimi `/v1/chat/completions`. If the remote portal
uses those defaults, a Kimi rule is needed. Rules evaluate the incoming model
and route before channel mapping; portal-visible aliases must be used.

Use a stable per-session identifier such as `X-Session-ID` or
`prompt_cache_key`, an explicit Kimi Chat rule, and intentional group/model/rule
namespacing. A request ID that changes each turn cannot yield session affinity.
The full verified JSON example is in `NEWAPI-CACHE-AFFINITY.md`.

Local removal of a field after NewAPI has already selected the channel cannot
cause that earlier selection to fail. Likewise, LiteLLM's generated session IDs
in local logs do not prove what NewAPI originally received. Its URL, version,
ordered rules and selection diagnostics are needed to verify the actual cause.

Also distinguish NewAPI's `cache_ratio` pricing multiplier from a measured cache
hit percentage. A stable channel, GPU prefix reuse and response replay are
three distinct outcomes. The local streaming-usage fix improves the evidence
returned to the portal; it cannot configure the remote routing policy.

## Follow-ups and recovery records

- Blocking upstream reads still do not actively watch client disconnects;
  connection ownership cleanup is fixed, full cancellation propagation is not.
- The blanket 512-token agentic cap, omitted model-rendered input fields,
  Responses compatibility and other earlier audit issues require separate
  focused patches. This release does not claim to fix the whole audit.
- Initial seven-file pre-release backups and digest manifest:
  `/tmp/opencode/k3-cache-release-20260920T201546Z`.
- Probe-tool backup:
  `/tmp/opencode/k3-cache-release-20260920T203640Z`.
- Installer/helper: `/tmp/opencode/k3_install_cache_release.py`. Its guarded
  rollback requires the current digest to match the release manifest. The
  subsequent shared_session correction changed cache_policy.py after the first
  manifest; reconcile that one known change before rolling back the initial
  bundle. Nginx/generator edits were separately applied and must be considered
  separately from the Python bundle. Do not restore the unauthenticated gateway
  fallback as part of a cache-only rollback.
- No git commit was created. Substantial preexisting user changes remain in the
  checkout; the full source tree was not copied into production.
