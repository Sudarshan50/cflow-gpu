# Log — complete portal admission and workload logic

**Purpose:** Explain every custom admission/workload check in the serving path, the quota and preparation checks around it, their exact order, their parameters, and the errors they produce.
**Verified runtime snapshot:** **2026-09-22 19:58:52 UTC**.
**Public local service:** `https://api.cflowx.in/v1`
**Repository:** `/root/cflow-gpu`
**This file:** `/root/cflow-gpu/docs/log.md`

## Read this first

The running configuration changed after the earlier conversation and the earlier TTFT document. The current gateway has a **16-request resting baseline**, a **24-request expansion ceiling**, and an **8-request noncritical pressure floor**. Its engine still has a **64-sequence ceiling**.

The current classification/output caps also changed: P0 is now an express short-input/short-output class; P1 allows 2,048 output tokens; medium-context and long-context requests normally allow 1,536 output tokens.

There are **two independent kinds of dynamic behavior**:

1. **Gateway adaptive execution:** changes how many requests may start, using engine KV, queue, ITL, TTFT, and prefill observations.
2. **LiteLLM dynamic RPM/TPM enforcement:** the main key's metadata makes its key-level RPM/TPM limits conditional on the router's recent deployment-failure signal. This does not use GPU KV or the gateway controller's state directly.

All substantive explanations and findings from this admission/workload audit are collected here. Historical numbers are explicitly labeled. Code conditions describe the inspected implementation; recommendations near the end are proposed changes.

## Contents

1. [Runtime snapshot](#1-runtime-snapshot)
2. [Definitions and the different queues](#2-definitions-and-the-different-queues)
3. [Full request lifecycle and check order](#3-full-request-lifecycle-and-check-order)
4. [Edge transport and authentication checks](#4-edge-transport-and-authentication-checks)
5. [LiteLLM quota checks](#5-litellm-quota-checks)
6. [Tenancy preprocessing and response-cache eligibility](#6-tenancy-preprocessing-and-response-cache-eligibility)
7. [Gateway request validation and token inspection](#7-gateway-request-validation-and-token-inspection)
8. [Classification and output clamping](#8-classification-and-output-clamping)
9. [Engine observations and adaptive capacity](#9-engine-observations-and-adaptive-capacity)
10. [Circuit breaker and execution-slot checks](#10-circuit-breaker-and-execution-slot-checks)
11. [Every workload-budget check](#11-every-workload-budget-check)
12. [Every admission-queue check](#12-every-admission-queue-check)
13. [Forwarding, completion, cancellation, and release](#13-forwarding-completion-cancellation-and-release)
14. [Historical and inactive restrictions](#14-historical-and-inactive-restrictions)
15. [Complete error and outcome map](#15-complete-error-and-outcome-map)
16. [Parameter dictionary and precedence](#16-parameter-dictionary-and-precedence)
17. [Configuration validation](#17-configuration-validation)
18. [Worked examples](#18-worked-examples)
19. [Metrics, diagnostics, and saved error evidence](#19-metrics-diagnostics-and-saved-error-evidence)
20. [Remaining interactions and improvement priorities](#20-remaining-interactions-and-improvement-priorities)
21. [Source map and evidence](#21-source-map-and-evidence)

---

## 1. Runtime snapshot

### 1.1 Processes and evidence boundary

| Component | Process/container start, UTC | Role |
|---|---|---|
| vLLM engine | 19:47:29 | Prefill, decode, batching, physical KV allocation, prefix cache |
| Gateway | 19:53:05 | Inspection, classification, clamping, execution admission, workload leases, waiting queue |
| LiteLLM | 19:53:05 | Authentication, quota enforcement, normalization, first-hop clamping, exact-response cache |

At the snapshot the engine was idle, with zero running/waiting requests and zero active KV usage. The gateway reported execution limit 16, zero queue entries, zero active workload leases, and no tokenizer quarantine. There were no request traces in this gateway process epoch at that point.

This is a configuration/logic snapshot, not a throughput benchmark or evidence of a production-wide error-rate improvement.

All **21 inspected serving-source files** matched between the repository and installed control-plane directory. The snapshot artifact records hashes and file timestamps.

The external NewAPI/distributor, when used, is another hop before this host. Its currently enforced settings were not established by this local inspection. Error messages from that hop must not automatically be attributed to a local gateway check.

### 1.2 Active limits

| Control | Current effective value |
|---|---:|
| Gateway workload guard | Enabled |
| Throughput-first mode | Enabled |
| Engine sequence ceiling | 64 |
| Gateway hard global slot ceiling | 64 |
| Adaptive resting/initial execution limit | **16** |
| Adaptive expansion maximum | **24** |
| Adaptive noncritical minimum | **8** |
| Effective limit during a start pause | **0** |
| Admission waiting budget | **60 seconds** |
| Maximum gateway waiters | **128** |
| Maximum waiters per customer label | **128** |
| Queue shape-token bound | **67,108,864 tokens** |
| Queue accounted-body bound | **1,073,741,824 bytes = 1 GiB** |
| Inspection operations in parallel | **4**, in the single gateway process |
| Individual tokenizer HTTP timeout | Up to **10 seconds**, limited by remaining admission time |
| Context window | **262,144 tokens** |
| Engine scheduled-token budget per iteration | **4,096** |
| Measured KV-token capacity | **1,639,906** |
| Aggregate reservation fraction | **92%** |
| Aggregate reservation ceiling | **1,508,713 tokens** |
| nginx per-IP/per-authorization outstanding-request limit | **256** each |
| Gateway TCP listen backlog setting | **256** |
| LiteLLM worker processes | **32** |
| Off-box model target | Not configured |

The current KV capacity differs slightly from the earlier 1,641,413-token observation because the engine was restarted and its pool was sized again. The budget is calculated from the measured capacity, not permanently fixed to the earlier number.

### 1.3 Main key settings

The key responsible for most recent recorded traffic has these database settings:

```text
rpm_limit             = 3,000,000
tpm_limit             = 3,000,000
max_parallel_requests = 500
metadata.rpm_limit_type = dynamic
metadata.tpm_limit_type = dynamic
```

These are key settings, not measured engine capacity. Their dynamic semantics are detailed in section 5. Other keys have different configured limits and metadata.

---

## 2. Definitions and the different queues

### 2.1 Admission versus workload checking

**Admission** asks whether a request can enter execution now, whether it can wait, or whether it must be refused.

**Workload checking** asks whether this request's input/output commitment fits the gateway's resource accounting alongside already admitted work.

An **execution slot** is a gateway bookkeeping permit. It is not one GPU, a fixed chunk of VRAM, or a reserved GPU stream. One TP8 model replica uses all eight GPUs together.

A **workload lease** records the resource charge for one admitted local request so that it can be released exactly once.

### 2.2 Queue and occupancy map

```text
HTTP/TCP arrival
    ↓
nginx and LiteLLM processing
    ↓
Inspection-slot waiting
    ↓
Gateway admission waiting queue, if needed
    ↓
Gateway execution slot + workload lease acquired
    ↓
vLLM scheduler waiting, if needed
    ↓
Prefill and generation
    ↓
Response finishes/fails/is abandoned
    ↓
Lease and execution slot released
```

| Waiting/occupancy location | Owner | What it is waiting for | Gateway execution slot held? |
|---|---|---|---|
| TCP acceptance backlog | OS / `ThreadingHTTPServer` | The server accepting a connection | No |
| Proxy/preprocessing work | nginx / LiteLLM / preparation threads | Authentication, hooks, media work, transport | No |
| Inspection-slot wait | `PromptInspector` semaphore | One of four tokenizer-operation permits | No |
| Gateway admission queue | `AdmissionController` | Execution permission and a valid workload reservation | No |
| vLLM waiting queue | Engine scheduler | Engine scheduling / physical resources | **Yes** |
| Active prefill/decode/relay | Engine and gateway | Generation and response delivery finishing | **Yes**, until gateway cleanup |

LiteLLM's `max_parallel_requests` is an **outstanding-request gauge**, not a waiting queue. Its slot can be held while the request waits farther downstream.

There is no durable job queue here. Gateway waiters are in-memory request handlers retaining open HTTP connections. A process restart does not persist/replay their queued jobs.

### 2.3 Queue count, queue deadline, and execution capacity differ

```text
128 queue positions = how many requests can wait
60-second deadline = how long a request can wait for admission
16 current slots   = how many requests may hold execution permits now
```

A queue containing one request can still time out if no execution opportunity becomes available before its deadline.

The engine queue can be zero while the gateway has many waiters: those requests have not reached vLLM yet.

### 2.4 Token quantities also differ

| Quantity | Meaning |
|---|---|
| Actual rendered prompt tokens | Input length after the model's template/tokenization |
| Estimated prompt tokens | Local heuristic when exact inspection is not used |
| Reserved prompt tokens | Value used for workload commitment; may exceed the estimate for opaque input |
| Requested output | Caller's allowance visible at this hop |
| Granted output | Allowance after clamping |
| Predicted output | Historical-length estimate; telemetry in throughput mode |
| Generated output | Actual completion tokens, including reasoning where reported |
| TPM reservation | Proxy quota accounting over a rate-limit window |
| Workload reservation | Gateway commitment held for the request lifetime |
| Queued shape tokens | Accounting for waiting entries; not allocated GPU KV |
| Cached prompt tokens | Actual reused prefix reported after inference |

---

## 3. Full request lifecycle and check order

### 3.1 Across components

```text
1. nginx accepts the permitted host/route, applies transport bounds,
   requires a nonempty Authorization header, and forwards to LiteLLM.

2. LiteLLM authenticates and runs applicable quota/budget and custom hooks.
   The K3 hook normalizes input, classifies, and applies its output clamp.
   Exact-response-cache eligibility is finalized before provider execution.

3. A response-cache hit can return without gateway or GPU inference.
   A miss/live request reaches gateway :8002.

4. Gateway validates route, declared body length, and the JSON object.

5. Gateway normalizes payload, checks n/best_of, and starts the throughput
   admission deadline.

6. Gateway performs eligible exact token inspection, or uses fallback counting.

7. Gateway constructs RequestEnvelope and its reservation/queue shape.

8. AdmissionController makes a real policy attempt, or queues behind an
   already waiting/aged request according to its lane rules.

9. GatewayPolicy checks context, off-box routing, circuit breaker,
   shared execution capacity, and then WorkloadBudget, in that order.

10. On admission: write granted output/priority to payload and send to vLLM.

11. Relay the response. Observe terminal usage when available.

12. Release the workload lease and execution slot; notify queue waiters.
```

**Ordering precision:** `n/best_of` is checked after gateway normalization, but before the admission timer starts. The body/route checks precede normalization. LiteLLM's own callbacks run in registered order; its quota algorithm's internal order is described separately below. Do not assume its token estimate is always based on the original unmodified client payload.

### 3.2 Exact order inside one `GatewayPolicy.decide()` attempt

| Order | Check/action | First blocking result |
|---|---|---|
| 1 | Classify and calculate clamp | Establishes class and grant |
| 2 | Prompt leaves insufficient output space | Context rejection; eventual HTTP 400 |
| 3 | Class is off-box and target is configured | Admit off-box without local slot/lease |
| 4 | Circuit breaker for this priority | `engine distressed: ...` |
| 5 | Acquire shared/global execution slot | `shared execution capacity: N` |
| 6 | Acquire workload lease | `workload budget: REASON` |
| 7 | Return admitted decision | Request can be dispatched |

If step 6 fails or raises, the temporary execution slot from step 5 is returned.

This order means an execution-capacity failure can hide a later memory failure. A single returned reason is the first blocking result of that attempt, not a list of all potentially binding checks.

The admission controller can retry this policy check while waiting. A policy refusal is not necessarily sent immediately to the client.

Sources: [`server.py`](../redesign/gateway/server.py), [`policy.py`](../redesign/gateway/policy.py), [`admission.py`](../redesign/gateway/admission.py).

---

## 4. Edge transport and authentication checks

### 4.1 nginx checks

| Check | Condition | Result |
|---|---|---|
| Accepted TLS host | Request must reach the accepted server name | Unmatched TLS host rejected; default HTTP server can close with 444 |
| Authorization presence | Header must be nonempty in `/v1/` | HTTP 401 if absent |
| Per-IP outstanding work | `limit_conn k3_ip_conn 256` | HTTP 503 when exceeded |
| Per-authorization outstanding work | `limit_conn k3_conn 256` | HTTP 503 when exceeded |
| Request body | At most 64 MiB | HTTP 413 when too large |
| Upstream read/send | Configured 900-second timeouts | Timeout/connection error, commonly 504 for upstream read timeout |

For HTTP/2, nginx's connection-limiting accounting treats concurrent requests according to the module's request semantics; this is not a GPU-sequence count.

The old zones are still declared:

```text
k3_ip_req: 20 requests/second
k3_req:    100 requests/second
```

But the active `/v1/` location does not apply those `limit_req` buckets. Declaration alone does not enforce a request-rate gate.

nginx overwrites `X-K3-Customer` with its configured customer label and clears `X-K3-Batch`. It forwards authorization to LiteLLM. It has no unauthenticated gateway backup in its inference upstream.

`proxy_buffering off` supports prompt streaming delivery; it does not change admission capacity.

The upstream retry directive includes connection/timeout/502/503 conditions. With a single configured upstream and ordinary POST replay protections, it should not be interpreted as an unrestricted inference-retry loop.

### 4.2 Authentication and account checks

LiteLLM validates the credential and applicable access/account state. Examples include missing/invalid/expired/blocked keys, model access restrictions, and configured spending budgets.

- Authentication failure: typically **401**.
- Access/ownership restriction: typically **403**.
- Applicable budget/rate restriction: **429**.

These checks determine whether the request reaches the gateway. They do not reserve physical engine KV.

The custom gateway trusts the local authenticated proxy path; it does not implement the LiteLLM virtual-key database itself.

---

## 5. LiteLLM quota checks

### 5.1 Active limiter and defaults

The installed proxy uses `_PROXY_MaxParallelRequestsHandler_v3`. `LEGACY_MULTI_INSTANCE_RATE_LIMITING` is unset, so the v3 handler is selected.

| Setting | Effective default/value |
|---|---:|
| `LITELLM_RATE_LIMIT_WINDOW_SIZE` | 60 seconds |
| `LITELLM_TPM_TOKEN_RESERVATION_ENABLED` | `true` |
| `DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE` | 1 |
| Parallel-request slot expiration | 3,600 seconds, library constant |
| Combined-TPM input heuristic | 4 characters/token |
| No-output-cap baseline floor | 4,096 / 4 = 1,024 tokens |

Redis coordinates counters between the 32 proxy workers. Some fast-path mirrors and the dynamic failure signal are process-local; those are distinct from the shared quota counters.

### 5.2 What `dynamic` RPM/TPM really means

The main key has both rate-limit types set to `dynamic` in **authenticated key metadata**.

The exact predicate is:

```python
if limit_type == "dynamic":
    enforce_this_key_rate_limit = model_has_failures
else:
    enforce_this_key_rate_limit = True
```

The failure check looks at the router's deployments for the requested model. It activates enforcement if any relevant deployment has:

```text
failure_count > DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE
```

With the default threshold 1, this means **more than one tracked failure**.

| Key rate-limit mode | Key RPM/TPM enforcement in this predicate |
|---|---|
| `dynamic`, failure signal false | The key-level limit is omitted from the enforced descriptor |
| `dynamic`, failure signal true | The configured key-level limit is enforced |
| `guaranteed_throughput` | Enforced when a numeric limit exists |
| `best_effort_throughput` | Enforced when a numeric limit exists in this handler |
| Unset/other mode | Enforced when a numeric limit exists |

The mode names do not allocate dedicated GPU throughput. In this handler, the special conditional branch is specifically for `dynamic`.

Additional details:

- If no router or no matching deployment list exists, the failure check returns false.
- If looking up failures raises an exception, it returns true and enforces the configured rate limit.
- The failure signal reads a **local router cache** with a 60-second TTL, not a fleet-wide GPU-health measurement.
- With 32 workers, different workers can have different recent local failure observations.
- The router's deployment-failure callback increments this signal before evaluating cooldown policy. A gateway refusal surfaced as a deployment failure can therefore affect later quota enforcement.
- This mechanism does not gradually change 3 million into a computed safe TPM value. It conditionally enables/disables the configured key-level RPM/TPM checks.
- **`max_parallel_requests: 500` is not disabled by this dynamic RPM/TPM predicate.**
- Other applicable user/team/model/project/org limits can still bind independently.

Therefore the main key's 3-million RPM/TPM settings are **not unconditional hard ceilings** under its current dynamic metadata. If the desired contract is a hard quota, the mode and accounting policy must reflect that explicitly.

Dependency source: `litellm/proxy/hooks/parallel_request_limiter_v3.py`, especially `_should_enforce_rate_limit`, `_get_enforced_limit`, `_create_rate_limit_descriptors`, and `_check_model_has_recent_failures`.

### 5.3 Applicable quota descriptors

The library constructs descriptors when their corresponding identities and settings exist:

- API key RPM, TPM, and maximum parallel requests.
- User RPM/TPM.
- Team RPM/TPM.
- Team-member RPM/TPM.
- End-user RPM/TPM.
- Model-specific key and team RPM/TPM.
- Configured per-key request-tag RPM.
- Model-specific project RPM/TPM.
- Project input-token/output-token quotas when configured.
- Organization and model-specific organization RPM/TPM.
- Agent/session limits when configured for those routes.

These are conditional library capabilities, not evidence that every one is configured for the main K3 key. The inspected local custom gateway does not add a second per-tenant RPM/TPM implementation.

If budget throttling is configured and selected during authentication, the key's numerical RPM/TPM can be scaled before the dynamic-mode predicate:

```text
scaled limit = max(1, floor(configured limit × throttle percentage))
```

Absent a throttle percentage, the configured value is retained. The custom serving YAML does not declare a global budget-throttle percentage.

### 5.4 Rate-limit check order

Within the ordinary v3 pre-call limiter:

1. Claim request accounting state.
2. Determine dynamic-mode failure state.
3. Build applicable descriptors.
4. Check/increment configured windowed request counters.
5. Acquire any configured parallel-request gauge slots.
6. Reserve tokens against enforced combined-TPM descriptors.
7. Reserve separately configured project input/output token quotas.
8. Continue to provider routing/inference if all applicable checks pass.

TPM is skipped in step 4 when reservation mode is enabled, so it is not accidentally charged once as a request counter and again as a token reservation.

If TPM reservation fails after a parallel slot was acquired, that slot is released before the 429 is raised.

### 5.5 RPM counter behavior

The ordinary request-counter Lua path uses a stored window start and counter. It starts/resets the window when missing or expired, otherwise increments the counter. The returned value is over limit when:

```text
counter > configured requests_per_unit
```

The code calls this a sliding-window mechanism, but this path is a window-start/TTL counter, not a log of every arrival over every possible rolling 60-second interval.

A request charged at this step can later fail a different check. RPM attempt accounting and successful-completion accounting are therefore different. The failure refund path primarily settles token reservations and parallel slots, not a blanket refund of every request attempt.

### 5.6 Parallel-request gauge behavior

The shared Redis path uses per-request slot IDs in a sorted set:

```text
remove expired slot IDs
if in_flight + 1 > limit:
    reject
else:
    register this request's slot ID
```

The main key's limit is 500. This includes time spent waiting downstream, not just GPU-running time.

Completion/failure/disconnect cleanup removes that specific slot ID. Releasing an already removed ID is a no-op. Old leaked slots can expire after 3,600 seconds.

The limiter checks a local mirror before its shared Redis acquire path. Redis/Lua failures have in-process fallback paths. Local fallback is not equivalent to a globally coordinated multi-worker capacity owner.

### 5.7 Combined-TPM estimation

For ordinary text Chat, the input estimate is approximately:

```text
estimated_input = max(1, extracted_text_characters // 4)
```

When an explicit output cap exists, it is used as the output estimate. This library helper takes the maximum supported cap alias it finds, whereas the K3 local clamp gives `max_completion_tokens` precedence. The effective payload at each hook therefore matters.

When no explicit cap exists:

```text
output_floor = min(1024, max(1, smallest_enforced_TPM_limit // 4))
```

If no operator output estimate is configured, the fallback output estimate is generally:

```text
estimated_output = max(estimated_input, output_floor)
```

Special cases include embeddings with no generation, empty inputs, configured output estimates, and Responses API minimum output-cap handling. Output-candidate counts can multiply the estimate; the local gateway later allows only `n/best_of = 1`.

The reserved total is at least one token when the combined-TPM reservation path applies:

```text
quota_reservation = max(1, estimated_input + estimated_output)
```

For very small configured TPM allowances and an omitted output cap, the library can also inject a bounded implicit output cap. This is distinct from the K3 class clamp.

Reservation is checked atomically per descriptor in Redis Lua. If a later descriptor fails, previously applied descriptor increments are rolled back. This closes the race in which many concurrent requests all see the same unused token allowance.

### 5.8 TPM settlement, cache accounting, and failure

On success, the installed default total-token path computes actual usage from the response and subtracts reported cached input tokens. For an already reserved scope:

```text
settlement increment = actual quota tokens − reserved quota tokens
```

For an unreserved scope, the actual amount is charged rather than refunding a reservation that never existed.

On failure, the limiter settles at recovered partial usage when available, or refunds the reservation when there is no recovered usage. Request accounting flags guard repeated cleanup. Project input/output reservation paths additionally track window identities for guarded adjustments.

These token-quota operations do not free GPU memory. The gateway's workload lease is a different object with a different lifetime and release path.

### 5.9 Quota error message and retry headers

A typical library message is:

```text
Rate limit exceeded for api_key: [identity].
Limit type: max_parallel_requests.
Current limit: 500, Remaining: 0.
Limit resets at: [time]
```

The type can also be `requests` or `tokens` for RPM/TPM.

The helper raises **HTTP 429** and sets:

```text
retry-after: 60
rate_limit_type: [requests | tokens | max_parallel_requests]
reset_at: [formatted time]
```

The helper calculates its advertised reset time as current time plus the configured window. This is not an exact forecast of when a parallel request will finish or the remaining TTL of every counter.

---

## 6. Tenancy preprocessing and response-cache eligibility

### 6.1 K3 pre-call hook

The custom LiteLLM callback:

1. Defaults `stream_options.include_usage` to true for streaming, preserving an explicit false.
2. Captures authenticated response-cache scope and original cache eligibility.
3. Disables response caching until the deployment hook finalizes an eligible key.
4. Normalizes a detached copy of relevant payload fields in a worker thread.
5. Applies `TenancyPolicy` classification and output clamping.
6. Raises HTTP 400 for its own unservable-context rejection.
7. Adds class/priority/off-box metadata.

The hook records `k3_class`, `k3_priority`, and `k3_offbox`. It does not currently establish a trusted downstream customer identity in the gateway's `X-K3-Customer` header.

Source: [`tenancy/callback.py`](../redesign/tenancy/callback.py).

### 6.2 Normalization that affects workload

| Operation | Behavior and relevance |
|---|---|
| Cache salt / client priority fields | Removes `cache_salt`, `prompt_cache_key`, `kv_cache_salt`, and inbound `priority`; gateway assigns engine priority later |
| Thinking controls | Maps common reasoning/thinking objects and aliases onto K3 template controls |
| Effort names | Accepted native values are `low`, `high`, `max`; aliases include `medium → high`, `minimal → low`, `xhigh → max` |
| Thinking off | Supported explicit controls can set template thinking false; generic `none/off` effort maps to low and disables thinking when not otherwise explicitly set |
| Tool ordering | Sorts uniquely named valid tool/function declarations; duplicate/missing names preserve caller order |
| Allowed-tools selection | Normalizes supported Responses-style `allowed_tools` into Chat-compatible tool choice |
| Numeric message content | Converts bare int/float content to text |
| Image/video input | Converts supported formats to bounded image/frame representations |

These transformations can alter token counts, prefix identity, class, and output behavior. They are not execution admission themselves.

### 6.3 Media bounds and preparation waits

| Control | Value / behavior |
|---|---|
| Remote download | 15 MiB maximum, 10-second timeout |
| Video decode input | 48 MiB maximum raw video |
| Video count | One per request |
| Video frames | Up to eight |
| Frame longest edge | 768 px |
| Image longest edge | 1,568 px in the normal decoder path |
| Video decode permits | Two per Python process |
| Wait for video permit | One second |
| `ffprobe` timeout | Five seconds |
| `ffmpeg` timeout | 20 seconds |
| PNG encoding compression | Level 6 |

The two video permits are process-local. With multiple LiteLLM workers, this is not one globally shared two-video limit across the entire proxy fleet.

Remote video downloads still pass through the 15 MiB downloader before the separate 48 MiB decode-size check. The larger decode limit does not make remote downloads accept 48 MiB.

Media failure or unavailable decode capacity can replace the part with:

```text
[attached media could not be decoded; continuing with the text]
```

Additional videos are replaced with:

```text
[additional video omitted; one video is allowed per request]
```

Those are payload changes, not necessarily HTTP errors. They can also remove the image signal that would otherwise determine the gateway class.

Gateway normalization is performed before its 60-second admission clock starts. LiteLLM normalization is an earlier additional stage.

Source: [`media.py`](../redesign/gateway/media.py).

### 6.4 Exact-response cache gates

The current response-cache mode defaults to `static`, with maximum TTL 300 seconds and revision `amd-5f3007-base-v1`. No team-pool allowlist was configured in the inspected environment.

Cache eligibility requires:

- A supported Chat completion call type.
- Authenticated scope derived from the authenticated key, or an explicitly allowed authenticated team.
- Static text Chat content with supported message roles.
- No disqualifying tools/tool history/media/stateful request fields.
- No caller opt-out such as `caching: false`, `use-cache: false`, `no-cache`, `no-store`, or applicable HTTP cache-control directives.
- Valid positive finite TTL/max-age settings, capped at 300 seconds.
- A serializable canonical effective request for the cache key.

The deployment hook rechecks eligibility after routing defaults are merged. Unknown nonserializable fields bypass caching rather than producing unstable keys.

**Cache ineligibility means live inference, not request rejection.** A cache hit can return without acquiring a gateway execution slot or workload lease.

Response replay is different from GPU prefix reuse. A prefix hit still needs admission and generation; it does not automatically bypass memory accounting.

Source: [`tenancy/cache_policy.py`](../redesign/tenancy/cache_policy.py).

---

## 7. Gateway request validation and token inspection

### 7.1 Handled request validity checks

| Check | Exact local rule | Failure |
|---|---|---|
| POST path | `/v1/chat/completions`, `/v1/completions`, `/chat/completions`, `/completions` | 404 `not found` |
| Content-Length parse | Must parse as integer | 400 `invalid Content-Length` |
| Declared body length | `0 < length <= 64 × 1024 × 1024` | 400 `missing or oversized body` |
| JSON decoding | Body must decode as JSON | 400 `invalid JSON: ...` |
| Top-level JSON type | Must be an object/dict | 400 `body must be a JSON object` |
| `n` / `best_of` | If non-null, exact Python integer type and value 1 | 400 `n/best_of must be the integer 1 on this replica` |

The `n/best_of` rule rejects bools, numeric strings, floats, and multi-candidate values. Null values are treated as unspecified by this check.

The gateway handler's socket timeout is 30 seconds. Its TCP backlog is 256. Neither setting is the admission-queue count or a promise of a 30-second total request duration.

Public Responses calls use LiteLLM's Chat conversion. The gateway does not directly implement `/v1/responses`.

### 7.2 Admission timer start

In throughput mode:

```text
started = monotonic time after normalization and n/best_of validation
admission_deadline = started + 60 seconds
```

This deadline is passed to inspection and admission. It is not reset when the request enters the gateway waiting queue.

### 7.3 Exact inspection eligibility

`inspectable(payload)` declines exact inspection if:

1. Both `messages` and `prompt` are non-null.
2. Any of these fields is non-null:

   ```text
   prompt_embeds
   prompt_embedding
   multi_modal_data
   multi_modal_uuids
   truncate_prompt_tokens
   documents
   functions
   ```

3. `tool_choice` or `function_call` is neither null/absent nor `auto`.
4. `response_format` is non-null.
5. A Chat message is not an object.
6. A message has non-null message-level `tools` or `task`.
7. Content is not null, a string, or an array consisting only of `text`/`input_text` objects.
8. A non-Chat prompt is neither a string nor a nonempty flat list of exact nonnegative integer token IDs below `2**32`.

Failing this predicate does **not** reject the request. It selects fallback counting/reservation. Native request validation may still reject the request later.

Constrained-generation formats are excluded because the deployed inference path can add prompt material that `/tokenize` does not report identically.

### 7.4 Tokenizer projection

The Chat projection includes:

```text
model, messages, tools, chat_template, chat_template_kwargs,
add_generation_prompt, continue_final_message, add_special_tokens
```

The completion projection includes:

```text
model, prompt, add_special_tokens
```

Important parity handling:

- Copies messages rather than changing the inference object.
- Maps retained `reasoning_content` to `reasoning` where the primary field is absent.
- Carries applicable reasoning-effort/template flags.
- Sets `return_token_strs = false`.

For a supported flat token-ID prompt, the inspector can use the supplied IDs directly rather than calling `/tokenize`.

### 7.5 Inspection semaphore and response checks

1. If quarantined, return no exact inspection and use fallback accounting.
2. If the payload is not inspectable, use fallback accounting.
3. Wait for one of four permits, checking cancellation every at-most-100 ms wait step.
4. Permit waiting is bounded by the smaller of the inspection-slot budget and admission deadline.
5. For an HTTP tokenizer request, use up to 10 seconds, limited by remaining admission time.
6. Reject a tokenizer response larger than 16 MiB.
7. Require a token list and `count == len(tokens)`.
8. Require every returned token to be an exact integer in `[0, 2**32)`.
9. Build diagnostic fingerprints and release the inspection permit in `finally`.

### 7.6 Inspection errors

| Internal reason | Public message | HTTP | Retry-After |
|---|---|---:|---:|
| `client_disconnected` | Client disconnected before inspection | No usable response; internally 499 | — |
| `tokenizer_busy` | `Prompt inspection admission deadline exceeded` | 503 | 5 |
| `tokenizer_rejected` | `Active model tokenizer returned HTTP 4xx` | 400 | — |
| `tokenizer_unavailable` | Tokenizer HTTP 5xx or `Prompt inspection is temporarily unavailable` | 503 | 5 |
| `invalid_tokenizer_response` | `Invalid prompt inspection response` | 503 | 5 |

The inspection response body includes message and type. These internal reason names are metric/exception labels; this branch does not insert each one as a public JSON `error.code`.

### 7.7 Heuristic token estimate

When exact counting is unavailable, the current gateway has no configured local tokenizer model path, so it uses `HeuristicEstimator`:

```text
estimated prompt =
    floor(CJK/fullwidth characters / 1)
  + floor(other extracted characters / 3.5)
  + 4 × message count
  + estimated image tokens
  + supplied flat token-ID count
```

Extracted text includes message content, names/roles, serialized tool-call history, and serialized tool/function declarations. It is not identical to rendering the native model template. In particular, it does not cover every retained-reasoning/template field handled by exact inspection.

For a known PNG size:

```text
image_tokens = max(4096, min(16384, floor(width × height / 192)))
```

Unknown image dimensions use 4,096 tokens per image. Video frames become images before this accounting.

The heuristic is intended to be conservative, but earlier measurements show both over- and under-counting. Its intention must not be treated as a mathematical upper-bound guarantee.

### 7.8 Conservative prompt reservation

Current fallback reservation is:

```text
if prompt is a nonempty list of token-list batches
   OR any opaque prompt field is non-null:
       reserve max_model_len × batch_count
else:
       reserve max(0, estimated_prompt_tokens)
```

Opaque fields are:

```text
prompt_embeds, prompt_embedding, multi_modal_data,
multi_modal_uuids, documents
```

`batch_count` is the prompt-list length when the list is nonempty and not entirely a flat list of exact integers; otherwise it is one.

Ordinary images, forced tool choice, `response_format`, and `truncate_prompt_tokens` do not by themselves force a full-window reservation in the current helper. That is different from earlier deployments.

### 7.9 Token-count quarantine and prefix diagnostics

After a completed successful response, if actual prompt usage disagrees with an exact inspection count:

- Set counting quarantine.
- Clear learned completed-prefix history.
- Future requests use fallback counting/reservation.

The in-memory quarantine flag starts false on a new inspector instance; it is not restored from the persisted prefix-history JSON.

Prefix fingerprints are diagnostic evidence of prior matching completed work, **not proof of current cache residency**. They do not lower the active workload reservation.

History is bounded to 4,096 entries and 15 minutes. The diagnostic event ring holds 1,024 entries. An engine epoch or cache-block-size change resets the fingerprint scope/history.

Sources: [`inspection.py`](../redesign/gateway/inspection.py), [`tokens.py`](../redesign/gateway/tokens.py), [`server.py`](../redesign/gateway/server.py).

---

## 8. Classification and output clamping

### 8.1 Request envelope

The gateway constructs:

```text
customer             = X-K3-Customer, default "anonymous"
prompt_tokens        = exact inspection count, else estimator
requested_max_tokens = positive integer max_completion_tokens,
                       else positive integer max_tokens,
                       else unspecified
streaming            = bool(stream)
has_tools            = bool(tools or functions)
has_images           = image presence after normalization
batch_hint           = gateway batch authorization decision
path                 = request path
```

Unlike `n/best_of`, the output-cap helper uses `isinstance(value, int)`, so Python bools can pass that integer test. Zero, negative, string, and float output values are treated as absent by this local helper rather than uniformly rejected here. Native validation/other proxy handling can differ.

### 8.2 Current first-match rules

| Order | Class | Predicate | Engine priority | Output cap | Declared TTFT target |
|---|---|---|---:|---:|---:|
| 1 | P3-batch | Authorized batch hint | 3 | 32,768 | None |
| 2 | P2-agentic | Tools or images, any length | 2 | 1,536 | 15 s |
| 3 | P0-interactive | Prompt ≤8,192 and an explicit effective requested output ≤512 | 0 | 512 | 1 s |
| 4 | P1-short-chat | Prompt ≤8,192, no tools; prior rules did not match | 1 | 2,048 | 3 s |
| 5 | P2-medium-context | Prompt ≤32,768; prior rules did not match | 2 | 1,536 | 15 s |
| 6 | P2-long-context | Remaining requests | 2 | 1,536 | 60 s |

Lower priority numbers run earlier under the engine's priority scheduler. Priority does not bypass gateway slots, workload commitments, or queue deadlines.

The per-class TTFT targets are declarations/observability intent. The adaptive controller uses its own global mean-latency thresholds, not these individual targets.

### 8.3 Batch and off-box rules

Gateway batch authorization requires `X-K3-Batch` equal to `1`, `true`, or `yes`, plus a loopback peer or an allowlisted customer. The current allowlist is empty; nginx clears the public batch header.

LiteLLM's `TenancyPolicy` separately reads the `k3_batch` payload field. These are different mechanisms, so a first-hop hint does not by itself establish that the final gateway envelope is authorized as batch.

P1 is marked as an off-box-capable class. It only bypasses local slots/leases when an off-box target is actually configured. No target was configured at this inspection.

### 8.4 Clamp formula and reasons

Let:

```text
L = 262,144 model context
P = gateway prompt count used for context validation
M = effective requested output, or unspecified
H = class output ceiling
A = L − P − 256
```

The clamp executes:

```text
if A < 64:
    G = 0
    reason = prompt_exceeds_window
elif M is unspecified:
    G = min(H, A)
    reason = default
elif M <= H and M <= A:
    G = M
    reason = unchanged
elif H <= A:
    G = H
    reason = class_ceiling
else:
    G = A
    reason = context_window
```

`G` is the granted output allowance.

Important boundaries:

- The 256 tokens are context-validation headroom. They are not added again to every workload charge.
- The 64-token rule is minimum remaining context space, not a minimum requested output. A request for one output token can be admitted.
- Even a one-output-token request is rejected by this policy if fewer than 64 output-space tokens remain after the reserve.
- A class clamp or context clamp normally changes the request rather than throwing an HTTP error.
- Only `G == 0` becomes the local context rejection.
- `exceeded_window` is separately calculated using `P + M > L`; it does not include the 256-token reserve in that flag.

### 8.5 Two-hop clamping

LiteLLM estimates and classifies first. When the workload guard is enabled and the payload is inspectable/local, it defers the context-window calculation by passing zero prompt tokens into its clamp, but still applies the class output cap.

The gateway later classifies again using its own count and clamps again. An estimate-to-exact class change can therefore make the final effective grant depend on both hops. The second hop cannot recover an original larger allowance already reduced by the first hop.

The payload sent to vLLM always receives `max_tokens = G`; an existing `max_completion_tokens` field is synchronized to the same grant.

The gateway response includes `x-k3-class` and `x-k3-max-tokens-granted`. Forwarding/preservation of those headers through an external portal is a separate integration concern.

### 8.6 Output budget and reasoning

The output allowance includes generated reasoning tokens. A short class cap can finish with `length` before a useful visible answer appears.

This can return HTTP 200. Reducing class caps changes output behavior and service time; it is not merely an internal admission optimization.

Sources: [`classification.py`](../redesign/gateway/classification.py), [`clamping.py`](../redesign/gateway/clamping.py), [`tenancy/policy.py`](../redesign/tenancy/policy.py).

---

## 9. Engine observations and adaptive capacity

### 9.1 What is measured

The background controller reads engine metrics for:

- Running requests.
- Engine waiting requests.
- Active KV fraction.
- Preemptions.
- Measured KV-token capacity and block size.
- Native ITL, TTFT, and prefill sum/count metrics.
- Engine/cache epoch.

For this vLLM deployment, the principal names include:

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:num_preemptions_total
vllm:inter_token_latency_seconds_sum / _count
vllm:time_to_first_token_seconds_sum / _count
vllm:request_prefill_time_seconds_sum / _count
vllm:cache_config_info
```

The parser sums values with the same metric name across labels. KV capacity is extracted from `kv_cache_size_tokens` labels; if multiple values appear, the minimum is selected. This parser is built around the present single-instance model and must not be assumed to combine arbitrary multi-replica gauges correctly.

### 9.2 Sampling, freshness, and smoothing

| Mechanism | Current behavior |
|---|---|
| Background interval | 0.5 seconds between completed sampling calls |
| Controller snapshot freshness | At most 2 seconds old |
| Engine health HTTP timeout | 5 seconds |
| Ordinary engine-client snapshot TTL | 2 seconds |
| Refresh/reconciliation TTL | At most 0.25 seconds |
| Request-side reconciliation | Waits for background generation to change, up to `0.5 × 1.5 = 0.75 s` |
| Latency sample | Difference of metric sums divided by difference of counts |
| Latency smoothing | `new_mean = 0.5 × new_sample + 0.5 × old_mean` |
| No new latency observations | Retain old mean for up to about 10 seconds, then use `None` |

An engine metrics scrape failure can make the controller pause before a five-second HTTP operation returns because the two-second freshness allowance has already elapsed.

The preemption rate is:

```text
max(0, counter_delta) / elapsed_minutes
```

It is not a rolling one-minute total. One preemption over a short polling interval can yield a large per-minute rate for that sample.

### 9.3 Observation validation and missing signals

The throughput controller rejects observations with:

- Missing/nonpositive KV-token capacity.
- Non-finite KV usage, or usage outside `[0, 1]`.
- Negative running, waiting, or preemption values.
- Non-finite preemption rate.
- A present latency value that is negative or non-finite.

On a failed observation it sets stale state, pauses starts, clears pressure/healthy/recovery streaks, and records `engine_health_unavailable` as a controller reason.

There is an important implementation limitation: the engine parser converts missing running/waiting/KV gauges to zero and missing preemption information to no observed increment. The controller then sees those defaults as valid numbers. Missing latency values are allowed and do not count as exceeding a threshold. Thus validation is not a complete required-metric-presence check.

### 9.4 Derived capacity settings

The runtime constructor calculates:

```text
maximum = min(K3_ADMISSION_CEILING, K3_ADAPTIVE_BORROW_MAX)
        = min(64, 24) = 24

minimum = min(configured minimum, maximum) = 8

initial = max(minimum, min(configured initial, maximum)) = 16
```

The engine's 64-sequence capacity does not override this configured maximum of 24.

### 9.5 Pressure predicates

Let `active = running + waiting`.

| Controller reason | Predicate |
|---|---|
| `kv_pressure` | Observed KV `>= 0.88` |
| `engine_queue` | Engine waiting `> 2` |
| `decode_latency` | `active > 0` and mean ITL `> 0.08 s` |
| `first_token_latency` | Active work, supporting load, and mean TTFT `> 5 s` |
| `prefill_latency` | Active work, supporting load, and mean prefill `> 10 s` |
| `preemption` | Observed preemption rate `> 0` |

Supporting load means at least one of:

```text
engine waiting > 0
OR mean ITL > 0.04 seconds
OR active >= initial_requests  # currently 16
```

Thus a slow isolated cold request can be a warning without triggering the TTFT/prefill pressure-reduction branch. It can still prevent healthy expansion.

### 9.6 Healthy/green predicate

```text
no pressure reasons
AND engine waiting == 0
AND KV <= 0.75
AND (
    no active work
    OR every available latency satisfies:
         ITL <= 0.04 s
         TTFT <= 2 s
         prefill <= 2 s
)
```

`None` latency does not violate the comparison. Healthy polling does not necessarily mean that each poll contains a fresh latency event.

### 9.7 State transitions and changes to new starts

Each successful observation updates:

```text
pressure_streak += 1 if pressure reasons exist, else reset to 0
healthy_streak  += 1 if green, else reset to 0
recovery_streak += 1 if no pressure reasons, else reset to 0
```

The main decision order is:

1. **Idle, green, no queued demand:** reset target to initial 16 and unpause.
2. **Critical:** KV `>= 0.97` or any preemption immediately pauses starts.
3. **Three pressure observations:** enter pressure; pause if engine waiting `> 2`; if adjustment is due, reduce target by 20% with floor 8.
4. **Four observations without pressure:** unpause; state becomes green or warm according to the current green predicate.
5. Under healthy recovery, restore a below-baseline target directly to 16 when an adjustment is due, even if no gateway requests are queued.
6. With queued demand and healthy recovery, grow above baseline by four, up to 24.
7. Other transient/warning observations freeze growth; an earlier pause normally remains until a recovery branch clears it.

Adjustments are rate-bounded to once every two seconds.

Typical noncritical reductions are:

```text
24 → 19 → 15 → 12 → 9 → 8
```

The effective limit exposed to admission is:

```text
execution_limit = target if snapshot is fresh and starts are not paused
                  else 0
```

The floor of 8 does not prevent a critical/stale pause from returning zero.

### 9.8 Recovery nuances

The current code explicitly restores the configured baseline after healthy observations without requiring queue demand. Above-baseline growth still requires queued demand.

There is a separate no-queue contraction branch that uses `max(initial, observed active)` when its conditions run. However, it is an `elif` after the healthy-adjustment branch. A fully green active workload already above baseline can retain its expanded target without queue demand because it enters that earlier branch but makes no upward change. The unambiguous reset occurs when the engine is idle/green and the gateway queue is empty.

The controller never terminates a healthy running request merely because its target falls. Running/owned requests can temporarily exceed the new limit; replacements wait until occupancy permits them.

### 9.9 What this controller does not calculate

It does not currently use these as scheduling-cost inputs:

- Per-request expected remaining generation time.
- Per-request verified cached-prefix residency.
- Per-customer fairness weight.
- Caller-specific deadline.
- Queue-age percentiles.
- A measured useful-completion-rate objective.

It receives the total number of gateway waiters as demand. Its thresholds are global means rather than the individual class TTFT targets.

Sources: [`engine.py`](../redesign/gateway/engine.py), [`capacity.py`](../redesign/gateway/capacity.py), [`throughput.py`](../redesign/gateway/throughput.py).

---

## 10. Circuit breaker and execution-slot checks

### 10.1 Circuit breaker predicates

`CircuitBreaker.evaluate()` reports distress for:

```text
KV > 0.97
OR (KV > 0.90 AND engine waiting > 8)
OR preemptions_per_minute > 1.0
```

The comparisons are strict `>` here. The adaptive controller's critical KV comparison is `>= 0.97`, so their exact boundary behavior differs.

If its health source raises an exception, the breaker itself fails open and does not report distress. That does not make the full system fail open: the adaptive controller can independently return execution limit zero for stale/unavailable health.

### 10.2 Which classes the breaker can shed

Only priorities 2 and 3 are sheddable:

- P2-agentic.
- P2-medium-context.
- P2-long-context.
- P3-batch.

P0 and P1 are exempt from this specific breaker. They still obey execution capacity, reservations, and admission deadlines.

The policy reason is:

```text
engine distressed: kv_usage ...
engine distressed: kv_usage ... with ... queued
engine distressed: preemptions .../min
```

A breaker decision is `REJECT_SHED`. It can still be retained in the bounded admission queue if `can_wait()` permits the request. HTTP 503 is sent when that decision is ultimately refused; it is not necessarily an immediate rejection at the first distress observation.

### 10.3 Shared execution acquisition

In throughput mode, under a lock:

```text
C = min(hard_global_ceiling, controller.execution_limit())
N = sum(in_flight counts across all local classes)

if C <= 0 or N >= C:
    no slot
else:
    increment this class's in_flight count
    slot acquired
```

Failure reason:

```text
shared execution capacity: C
```

This is a global request-count gate. It does not directly evaluate KV, prompt length, output length, or RPM/TPM; those are handled elsewhere.

### 10.4 The misleading `kv_budget_share` name

Class definitions retain `kv_budget_share`. `ClassBudget` uses it as:

```text
legacy per-class request cap = int(global ceiling × class share)
```

It is not an actual partition of GPU KV bytes.

At global ceiling 64, the retained baseline caps are:

| Class | Retained share | Calculated legacy cap |
|---|---:|---:|
| P0 | 0.50 | 32 |
| P1 local fallback | 0.25 | 16 |
| P2-medium-context | 0.25 | 16 |
| P2-long-context | 0.25 | 16 |
| P3-batch | 0.25 | 16 |
| P2-agentic | 0.45 | 28 |

These caps do not sum to a dedicated 64-slot partition. The global ceiling is separately enforced.

Throughput-mode acquisition ignores these positive per-class caps in favor of the shared limit. The values still affect zero-cap eligibility checks and the `adaptive class borrow` note/metric. That note does not mean physical KV was borrowed from a separate partition.

Source: [`backpressure.py`](../redesign/gateway/backpressure.py).

---

## 11. Every workload-budget check

### 11.1 Variables and units

Use the following notation:

```text
P = prompt count used by classification/context clamping
R = max(P, conservative reserved prompt override or 0)
G = granted output tokens after clamping
Q = incoming execution charge
T = sum of existing execution-token charges
K = measured engine KV-token capacity
V = observed active KV fraction
N = number of workload leases owned by this gateway
E = engine running + engine waiting
B = applicable execution-token reservation ceiling
```

In current throughput mode:

```text
Q = R + G
B = max(1, floor(K × 0.92))
  = 1,508,713 at this snapshot
```

`R` is used for memory commitment and workload flags. `P`, not the full conservative `R` override, is used to decide whether the actual request appears to fit the model context. An opaque request can therefore reserve a full window while its estimated prompt still passes context clamping.

### 11.2 Workload flags

```text
large = (R >= 65,536) OR has_images
long  = (G >= 2,048)
heavy = large OR long
```

The thresholds are inclusive.

These flags remain meaningful even though the old request-count caps are disabled:

- They determine the queue lane shape.
- Heavy requests trigger engine-ownership reconciliation.
- They determine telemetry and output-history collection.

The naming can be confusing:

- A request classified as `P2-long-context` with 40,000 input tokens is not `large` by the 65,536-token workload threshold.
- An image request is `large` for workload purposes even if its estimated token count is small.
- A 2,048-output-token P1 request is `long`; a 2,047-output-token request is not.
- Under current caps, normal agentic/medium/long-context output grants of 1,536 do not set the `long` flag, although input length/images can still make them heavy.

### 11.3 Input sanity

`WorkloadBudget.acquire()` requires:

```text
R >= 0
G > 0
```

Otherwise it raises `ValueError("Invalid workload token reservation")`.

Normal gateway flow rejects a zero grant before reaching this method. This exception is an internal invariant check, not the normal client overload response.

### 11.4 Obtain and reconcile health

`_snapshot(heavy)` first obtains a snapshot from the controller.

- On snapshot failure it returns no snapshot.
- For a heavy request, if the engine reports more running/waiting requests than the gateway owns, it requests a fresher background observation.
- The current controller's refresh method waits for the background sampler; the request thread does not issue its own metrics scrape.
- If refresh fails, the old reading is retained rather than replaced by an optimistic empty-engine reading.

In throughput mode, no snapshot produces:

```text
workload budget: engine_health_unavailable
```

Often a stale/paused controller has already blocked the request at the earlier shared-slot check with `shared execution capacity: 0`. The workload health error can appear if health becomes unavailable between checks or the method is called through another path.

### 11.5 Calculate commitment and projected KV

Under the workload lock:

```text
projected = V
if K is known:
    projected = max(V, T / K) + Q / K
```

Using `max(V, T/K)` avoids directly adding two measurements of overlapping existing work. The incoming charge is then added.

The gateway keeps the full grant reserved for the request lifetime. It does not periodically reduce a lease as generation progresses, and it does not subtract a diagnostic cache-hit prediction.

### 11.6 Ordered rejection checks

After calculation, the checks execute in this order:

| Order | Condition | Throughput mode? | Reason returned |
|---|---|---|---|
| 1 | `large` and existing large count reaches configured maximum | **Skipped** | `large_context_slots` in legacy mode |
| 2 | `long` and aggregate predicted output exceeds selected budget | **Skipped** | `long_output_budget` in legacy mode |
| 3 | `T + Q > B` | **Enforced for all requests** | `reserved_tokens` |
| 4 | Heavy work with engine waiters beyond legacy tolerance | **Skipped** | `engine_queue` in legacy mode |
| 5 | Heavy work and `E > N` after reconciliation | **Enforced** | `untracked_engine_work` |
| 6 | `projected > 0.92` | **Enforced for all requests** | `kv_headroom` |
| 7 | All checks pass | Create lease | `admitted` |

The final public policy message prepends `workload budget: ` to a workload rejection reason.

Because `reserved_tokens` is checked before `kv_headroom`, a request exceeding the summed reservation ceiling normally reports `reserved_tokens`, even though the projected-KV calculation also exceeds the limit.

### 11.7 `reserved_tokens`

This check protects aggregate promised capacity:

```text
existing committed tokens + incoming commitment > allowed commitment
```

It can reject while actual active KV is low because reservations include future output and conservative prompt commitments that the engine has not yet materialized.

This is not the LiteLLM TPM counter. Waiting for a quota window reset does not release a gateway lease. A lease is released by the admitted request's lifecycle cleanup.

### 11.8 `kv_headroom`

This check protects against observed usage being higher than gateway commitment bookkeeping.

For example, even if the gateway has only 200,000 committed tokens, observed KV can already be 91%. Adding a 20,000-token request projects above 92% at the measured capacity and is refused.

Sources of differences include engine work outside the current gateway, differing allocation geometry, and observation/lifecycle lag. The reason alone does not identify which source applies to an individual request.

### 11.9 `untracked_engine_work`

For heavy requests:

```text
engine running + waiting > owned workload leases
```

means the gateway cannot fully reconcile the engine's request count with its lease registry.

Possible explanations include:

- A stale gauge just after a request finished.
- A gateway restart while old engine requests remain.
- Direct engine callers.
- Another independent gateway serving the same engine.
- Delayed engine cancellation after local release.

The code refreshes before refusing, but does not prove which explanation is responsible.

This specific count-discrepancy check is only applied to heavy requests. It is not a full distributed ownership protocol. Multiple independent gateways would not automatically share the same global slot/commitment registry.

### 11.10 Lease creation

An admitted lease stores:

```text
lease_id -> (
    execution token charge,
    large flag,
    long flag,
    predicted long-output tokens,
    class name,
    full granted output
)
```

The lease ID is a monotonic integer within the process. It is not a timed lease with automatic expiration.

The method increments:

- Total reserved tokens.
- Active large count, if large.
- Active long count, if long.
- Aggregate predicted long-output tokens, if long.

It also updates legacy-shaped telemetry such as `burst_admissions_total`. Those counters can change even when the associated old rejection budget is disabled.

### 11.11 Output prediction

Per class, retain up to 128 valid completed long-output observations.

```text
if fewer than 8 history samples:
    estimate = 2,048
else:
    estimate = nearest-rank 95th percentile of retained output lengths

prediction = min(G, max(1, ceil(estimate × 1.25)))
```

The percentile index is `max(0, ceil(sample_count × 0.95) - 1)` in sorted history.

History is updated only for a lease marked long and an exact integer actual output satisfying:

```text
0 < actual_output <= granted_output
```

The recorded completion-token count includes reasoning work when the engine reports it that way.

**Current throughput mode does not replace `G` with this prediction for memory admission.** The full grant remains committed. Legacy mode can use the prediction for a long request's charge and predicted-output budget.

### 11.12 Single-request feasibility

`can_fit_alone()` asks whether one request's charge could fit an otherwise empty reservation budget.

- Current throughput mode compares full `R + G` against `B`.
- Legacy long-output mode can compare `R + predicted_output` instead.
- If health cannot be obtained, it returns true so a bounded request can wait for health recovery rather than being permanently declared impossible.

This check controls queue eligibility. It does not grant a lease and does not guarantee current fit.

The fallback `K3_RESERVED_TOKEN_BUDGET=1,048,576` remains in `reservation_limit()` when measured capacity is unavailable. In the active controller path, missing capacity normally causes a pause, so this fallback should not be interpreted as permission to admit freely without valid engine capacity.

Source: [`workload.py`](../redesign/gateway/workload.py).

---

## 12. Every admission-queue check

### 12.1 Queue ticket and lane

A ticket contains:

```text
lane = (class name, large flag, long flag)
customer label
queued shape-token charge
accounted body bytes
admission acquire start time
```

The lane does not contain customer identity. FIFO ordering is principally within workload lanes, not a weighted per-tenant scheduling algorithm.

The queue's token charge uses full reserved prompt plus grant. It represents waiting shape, not an execution reservation or a GPU allocation.

### 12.2 Entry and deadline checks

`AdmissionController.acquire()`:

1. Gets the effective wait budget from policy. Throughput mode returns the configured 60 seconds unchanged.
2. If the wait budget is zero, directly calls `policy.decide()` without queue machinery.
3. Rejects negative internally supplied body-byte accounting with `ValueError`.
4. Calculates the smaller of the passed deadline and `acquire_start + wait_budget`.
5. If already expired, returns an admission deadline rejection with code `queue_timeout`.
6. Calculates the queue lane and full shape charge.
7. If no queueable lane exists, calls the policy directly.
8. Checks whether the client disconnected before queueing.

No queueable lane is returned for a zero output grant, configured off-box routing, a zero-cap class, or an individually impossible reservation.

### 12.3 Initial fast path

A request gets an immediate real policy attempt when:

```text
no older queued ticket has the same lane
AND no globally aged waiter currently has priority
```

Concurrent initial probes are allowed. Merely having another capacity check in progress does not automatically consume a queue position.

If the attempt is admitted, cancellation and deadline are checked again before dispatch. If either applies, the acquired lease/slot is released.

If it is refused and cannot wait, return the refusal. Otherwise try to enqueue it.

If an older same-lane/aged ticket already has priority, the new arrival first tries to enqueue rather than stealing that opportunity.

### 12.4 Queue bounds, in exact order

Under the admission condition lock:

| Order | Rejection predicate before append | Admission reason |
|---|---|---|
| 1 | Existing global waiting count `>= 128` | `queue_full` |
| 2 | Existing count for this customer `>= 128` | `customer_queue_full` |
| 3 | Existing queued tokens + incoming charge `> 67,108,864` | `queued_token_limit` |
| 4 | Existing accounted bytes + incoming bytes `> 1,073,741,824` | `queued_byte_limit` |

Only the first applicable reason is returned. These become:

```text
admission queue: queue_full
admission queue: customer_queue_full
admission queue: queued_token_limit
admission queue: queued_byte_limit
```

They are `REJECT_BUDGET` decisions and currently map to HTTP 429.

With the customer bound equal to the global bound, a single customer filling the entire queue normally hits `queue_full` first. Seeing an older `customer_queue_full` error does not establish that the current 128/128 settings caused it.

### 12.5 What queue-byte accounting actually measures

The gateway records the original declared request-body length in `_request_body_bytes` and passes that value into queue accounting.

It is not a precise measurement of retained Python object memory, normalized media expansion, or all preprocessing threads waiting before admission.

The inspection waiting area and video preparation occur before this admission-queue bound. The 128-entry limit is therefore not a global bound on every request or every byte held throughout the entire portal.

### 12.6 Successful enqueue

On append:

- Add ticket to the waiting list.
- Increment this customer's waiting count.
- Add queued token and body-byte charges.
- Increment `queued_total`.
- Send the new total queued count to the capacity controller as demand.

No execution slot or workload lease is retained while simply waiting.

### 12.7 FIFO, probing, and aging

A queued request may make a policy attempt only when:

1. It is the first waiting ticket in its lane.
2. Its lane is not already making a queued policy probe.
3. If an aged waiter exists, this ticket is that aged waiter.

The oldest ticket becomes aged when:

```text
now − ticket.started >= 5 seconds
```

This gives the oldest ticket the next cross-lane opportunity. It does not mean the ticket has been promised a start within five seconds.

When the aged request cannot currently fit, other lanes can be blocked even if a smaller request would fit. The implementation does not perform cost-aware bounded backfilling around that aged request.

FIFO describes queued-lane ordering. Concurrent initial fast probes mean it is not a strict global FIFO order over all network arrivals.

### 12.8 Wait loop

Each queued loop iteration:

```text
check client disconnection
check remaining admission time
check lane/aged priority
if eligible:
    retry policy.decide()
    release lane-probing marker and notify waiters

    if admitted:
        recheck deadline and cancellation
        release acquired capacity if expired/cancelled
        otherwise finish as admitted_after_wait

    retain this attempt as the last blocking decision
    if can_wait() is false:
        finish as not_retryable

wait for notification or up to 200 ms / remaining deadline
```

Lease release, queue removal, and completion of a queued probe notify the condition variable. Controller health changes are discovered by subsequent probes; the queue also polls at the bounded interval.

The fixed admission deadline is not reset on retries or controller-state changes.

### 12.9 Which refusals can wait?

`policy.can_wait()` returns false for:

- An already admitted decision.
- Zero granted output.
- A zero-cap class.
- A request that cannot fit alone under the applicable reservation model.

Otherwise it can return true. It does not categorically exclude `REJECT_SHED`; genuine engine-distress decisions can wait for recovery within the deadline.

An unknown-health feasibility check can allow waiting, while the execution controller still refuses new starts until health recovers.

### 12.10 Timeout and reason preservation

When time expires, the controller normally returns the last policy refusal with:

```text
admission_reason = queue_timeout
```

Examples:

```text
policy reason:    shared execution capacity: 16
admission reason: queue_timeout
```

or:

```text
policy reason:    workload budget: reserved_tokens
admission reason: queue_timeout
```

A queued request that never gets a chance to probe can retain the initial message `waiting for admission capacity` instead of a more specific resource reason.

The HTTP status is derived from the retained decision, not from the word `queue_timeout` alone.

### 12.11 Removal and cancellation

Queue cleanup runs in `finally`:

- Remove the ticket if still present.
- Subtract its token/byte charges.
- Decrement/remove the customer counter.
- Update queued-demand feedback.
- Notify waiters.

An admitted decision obtained just after a disconnect or after the deadline is released before inference dispatch.

The client-disconnect check peeks at the socket without consuming input. While already blocked in an upstream read, active-generation cancellation has different limitations, described in section 13.

### 12.12 Wait-duration measurement

For a ticket, reported admission wait is measured from entry into `AdmissionController.acquire()`, before its initial shape/probe work. It excludes earlier prompt inspection and normalization, and can include the initial failed policy attempt before actual enqueue.

When no ticket exists, `_finish()` reports zero wait even for an already-expired preparation deadline.

Therefore:

```text
x-k3-admission-wait-ms
```

is not complete end-to-end waiting time and does not account for every preparation stage.

Source: [`admission.py`](../redesign/gateway/admission.py).

---

## 13. Forwarding, completion, cancellation, and release

### 13.1 What reaches the engine

After admission the gateway:

- Writes `max_tokens = granted output`.
- Synchronizes an existing `max_completion_tokens` alias.
- Adds the class-derived integer `priority`, because `K3_SEND_PRIORITY=1`.
- Sends the request to the engine.

The engine can still reject native input, context, model, tool, or sampling validation. Gateway admission is not a guarantee that all engine-side validation has succeeded.

The engine's 4,096-token iteration budget and 64-sequence ceiling are different from the gateway's current 24-request adaptive maximum. The engine's own queue can hold admitted work not yet scheduled; more than two waiters is a controller pressure signal, not a hard two-entry queue bound.

### 13.2 Streaming and nonstreaming transport

The gateway uses an upstream socket timeout of 600 seconds. This is not the 60-second admission budget or a single guaranteed end-to-end deadline.

- Streaming uses `read1(8192)` to forward available chunks without waiting for an entire 8 KiB buffer.
- Nonstreaming reads the complete upstream body.
- The gateway adds class, granted-output, and admission-wait headers.
- It frames the outgoing body as chunked transfer.

The execution lease lasts through the gateway's relay/cleanup lifecycle, not merely until the first generated token.

### 13.3 Usage observation

`UsageObserver` watches terminal usage without changing relayed content.

For a complete usable observation it requires:

- A completion marker (`[DONE]` for streaming; completed body for nonstreaming).
- At least one finish reason.
- A usage object.
- No detected error or parsing/overflow failure.

The observer's buffer is bounded to 1 MiB. An overflow stops diagnostic usage observation; it is not itself an instruction to truncate the forwarded model response.

When a complete usage object contains a valid completion-token count, the count can be used for output-history updates at release.

### 13.4 Workload release

Under the workload lock:

```text
item = leases.pop(lease_id, None)
if item is absent:
    return false  # already released or unknown

subtract token charge
subtract large/long counters
subtract predicted long-output tokens
record valid actual output history for a long lease
return true
```

`GatewayPolicy.release()` releases the associated class/shared slot only after a successful workload-lease removal. If the lease was already removed, it returns without decrementing the slot a second time.

The admission controller then notifies waiters that capacity may be available.

### 13.5 Failure cases and cleanup

| Event | Behavior |
|---|---|
| Workload refusal after execution-slot acquire | Return the temporary execution slot immediately |
| Workload method raises after slot acquire | Return the slot, then propagate the exception |
| Deadline/disconnect discovered just after admission | Release lease and slot before dispatch |
| Engine connection fails before response headers | Return HTTP 502 and release capacity in request cleanup |
| Engine returns 4xx/5xx | Relay status/body; cleanup releases capacity |
| Client disconnect during a write | Close upstream response and release through cleanup |
| Upstream stream breaks after headers | Omit normal terminating chunk, close connection, release capacity |
| Duplicate release | Lease removal is a no-op and prevents a second slot decrement |
| Gateway process stops | In-memory queue/lease state is lost; existing engine work must be reconciled by the new process |

If the client disconnects while the handler is blocked reading upstream, detection can be delayed until that read returns or times out. This is not a fully proactive engine-abort mechanism.

An engine request may also take time to abort after its HTTP connection is closed. That contributes to possible engine-versus-lease count discrepancies.

### 13.6 Off-box behavior

No off-box target is active in this snapshot.

If configured:

- A class marked off-box can be admitted without a local execution slot/workload lease.
- The off-box client replaces the model name and removes local priority.
- Eligible P0/P1 connection failures before response delivery can use the configured fallback path.

The current local queue does not transparently turn into an external durable overflow queue.

### 13.7 Successful HTTP versus useful completion

These are separate measurements:

```text
admitted by gateway
HTTP 200 headers sent
complete stream delivered
valid terminal usage observed
nonempty useful answer/tool call produced
completion within caller deadline
```

An HTTP 200 can still end with a truncated stream or output allowance exhausted entirely in reasoning. Admission success alone does not establish useful throughput.

Sources: [`server.py`](../redesign/gateway/server.py), [`engine.py`](../redesign/gateway/engine.py), [`inspection.py`](../redesign/gateway/inspection.py), [`offbox.py`](../redesign/gateway/offbox.py).

---

## 14. Historical and inactive restrictions

### 14.1 Fixed per-class concurrency

Legacy `ClassBudget` acquisition checks:

```text
class_in_flight >= class_limit
OR total_in_flight >= global_ceiling
```

Without borrowing, a P1 request could hit its 16-request cap while other capacity was unused. The returned message was:

```text
P1-short-chat at its concurrency limit of 16
```

The shared throughput-mode branch replaces positive per-class acquisition caps with the global adaptive limit.

### 14.2 Conditional legacy borrowing

When not in throughput mode, borrowing is eligible only for P1-short-chat and P2-agentic and requires:

- Positive configured adaptive-borrow maximum.
- Positive controller borrow allowance, if a controller is present.
- Available health and measured KV-token capacity.
- No engine waiters.
- No observed preemptions.
- Present mean ITL no higher than the borrow threshold.
- Projected KV no higher than the borrow KV threshold.

The effective class limit becomes the larger of its retained baseline cap and allowed borrowing, still bounded by the global ceiling.

This request-shape borrowing path is bypassed by the current shared execution-pool policy.

### 14.3 Fixed large-context count

The legacy branch refuses a large request when:

```text
large_count >= K3_LARGE_CONTEXT_MAX  # retained value 4
```

Reason: `large_context_slots`.

Current throughput mode skips this count gate. It retains the large flag and memory commitment.

### 14.4 Fixed long-output count

Earlier deployments enforced two simultaneous long-output requests and returned `long_output_slots`.

The current `WorkloadLimits` still contains deprecated compatibility fields such as `long_output_requests=2` and `long_output_burst_requests`, but the count rejection itself is not enforced by the current workload code, even when reviewing its non-throughput branch.

Changing an old count parameter therefore does not necessarily change current admission.

### 14.5 Predicted-output budget

The retained non-throughput policy uses:

```text
base predicted-output budget  = 12,288
burst predicted-output budget = 49,152
```

The burst allowance requires:

- Burst budget greater than base.
- Controller permits burst, if it supplies that predicate.
- Incoming request is not large.
- Health and KV capacity are available.
- Reserved input at most 32,768 tokens.
- Finite projected KV between zero and 55% inclusive.
- Zero engine waiters and preemptions.
- `max(engine running, owned leases) < 16`.

Otherwise the base budget applies.

If no long request is active, the selected budget is raised to at least the incoming prediction so one otherwise valid completion is not made impossible solely by that aggregate prediction budget.

Legacy refusal:

```text
existing predicted long tokens + incoming prediction > selected budget
→ long_output_budget
```

Because the selected budget depends on the arriving request while existing predicted tokens are global, one request shape can lose burst eligibility and be blocked while another continues. This was an important historical starvation interaction.

Current throughput mode computes some prediction telemetry but skips this rejection and uses full granted output for commitment.

### 14.6 Legacy engine-queue gate

For large/long requests, the retained legacy workload branch can reject when engine waiting exceeds `_queue_tolerance()`.

Tolerance falls to zero when:

- Projected KV is invalid or above 55%.
- Any preemption is observed.
- `max(engine running + waiting, owned leases) >= 16`.

Otherwise it uses the configured tolerance, currently retained as 2.

Reason: `workload budget: engine_queue`.

Current throughput mode skips this workload rejection. Engine queue pressure instead affects the adaptive starts controller, whose separate `tolerated_waiters=2` currently has the same numerical value.

### 14.7 Legacy controller shrank waiting under pressure

The older `AdaptiveCapacityController` implementation can return:

| State | Allowed waiting | Queue-count bound |
|---|---|---|
| Green | Configured waiting | Configured count |
| Warm | At most 1 second | At most 8 |
| Pressure/stale | Zero waiting | At most 4; zero-wait path skips queueing |

The current `ThroughputCapacityController` overrides both methods and always returns the configured 60-second wait and 128-entry count.

### 14.8 Historical full-context fallback

Earlier unsupported inspection formats routinely reserved a full 262,144-token window. Against the old 1,048,576-token global budget, four such requests with a 1,536-token grant required:

```text
4 × (262,144 + 1,536) = 1,054,720 > 1,048,576
```

The fourth could be refused even though the large-request count limit was four and actual prompts were much smaller.

Current ordinary multimodal/constrained requests use estimates instead; opaque formats retain full-window handling. Accurate accounting remains necessary because the estimator is not universally conservative.

---

## 15. Complete error and outcome map

### 15.1 Decision types

The gateway has three primary policy outcomes:

```text
ADMIT         = 0
REJECT_BUDGET = 1
REJECT_SHED   = 2
```

Transport mapping in `_refuse()` is:

```text
503 if outcome == REJECT_SHED
429 otherwise
400 overrides the above if granted output == 0
```

JSON type is:

```text
503 → service_unavailable
400 → invalid_request_error
429 → rate_limit_error
```

JSON code is:

```text
admission_reason, when nonempty
otherwise lowercase outcome name
```

Thus a gateway 429 is not proof that RPM/TPM was exceeded.

### 15.2 Admission/workload refusal inventory

| Message / internal reason | Trigger | Can wait? | Current final HTTP |
|---|---|---|---|
| `prompt of ... tokens leaves no room for output` | Clamp leaves less than 64 available tokens | No normal queue | 400 |
| `shared execution capacity: N` | Global owned slots reach the adaptive allowance, or allowance is zero | Usually yes | 429 |
| `workload budget: engine_health_unavailable` | No workload health snapshot in throughput mode | Usually yes | 429 |
| `workload budget: reserved_tokens` | Existing plus incoming token commitment exceeds budget | Yes if it can fit alone | 429 |
| `workload budget: untracked_engine_work` | Heavy request, engine work exceeds owned leases after reconciliation | Usually yes | 429 |
| `workload budget: kv_headroom` | Projected KV exceeds allowed fraction | Usually yes | 429 |
| `engine distressed: ...` | Breaker threshold and sheddable priority | Usually yes if queueable | 503 if that refusal is retained |
| `admission queue: queue_full` | 128 waiting entries already present | No additional queue entry | 429 |
| `admission queue: customer_queue_full` | Customer waiting bound reached | No additional queue entry | 429 |
| `admission queue: queued_token_limit` | Aggregate queued shape charge would exceed bound | No additional queue entry | 429 |
| `admission queue: queued_byte_limit` | Accounted queued body bytes would exceed bound | No additional queue entry | 429 |
| `waiting for admission capacity` | Initial placeholder retained when no eligible attempt occurs | Already waited | Usually 429 on timeout |
| `admission deadline exceeded` | Deadline elapsed before/just after admission | No dispatch | Usually 429; zero-grant override can make 400 |
| `client disconnected before admission` | Downstream gone | Cancel/remove | No client response; internally 499 |

When queue insertion itself fails, the new `REJECT_BUDGET` queue-bound decision can replace an earlier distress refusal. Therefore not every request that encounters engine distress ultimately returns 503.

Historical-only/legacy messages include:

- `CLASS at its concurrency limit of N`.
- `workload budget: large_context_slots`.
- `workload budget: long_output_budget`.
- `workload budget: engine_queue`.
- Historical `workload budget: long_output_slots` from removed count-gate deployments.

### 15.3 Admission outcome codes

| `admission_reason` | Meaning |
|---|---|
| Empty | Direct policy result; no finished queued outcome attached |
| `admitted_after_wait` | Obtained execution capacity after queue waiting |
| `queue_timeout` | Admission deadline expired |
| `queue_full` | Global waiting count prevented enqueue |
| `customer_queue_full` | Customer waiting count prevented enqueue |
| `queued_token_limit` | Queued token charge prevented enqueue |
| `queued_byte_limit` | Accounted queued bytes prevented enqueue |
| `client_disconnected` | Request abandoned before dispatch |
| `not_retryable` | Queue loop determined the request can no longer wait/fit alone |

The internal name `not_retryable` does not automatically remove the generic retry header. The current HTTP error contract can still present a budget-style 429 with retry guidance even when the request shape cannot fit alone. Clients need stable reason semantics; this is an improvement opportunity.

### 15.4 Example: timed-out execution admission

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 2
x-k3-admission-wait-ms: 59990
```

```json
{
  "error": {
    "message": "shared execution capacity: 16",
    "type": "rate_limit_error",
    "code": "queue_timeout",
    "k3_class": "P0-interactive"
  }
}
```

The wait number is illustrative. The request's output class does not guarantee an execution opportunity before the 60-second deadline.

### 15.5 Retry headers

- Throughput-mode policy refusals with a retry value use **`Retry-After: 2`**.
- Legacy policy refusals generally use the decision's default **30 seconds**.
- Normal prompt-too-long policy rejection sets no retry value.
- Inspection-unavailable 503 uses **5 seconds**.
- The installed LiteLLM quota helper generally uses its **60-second window**.

These are independently generated headers. Their existence locally does not establish that an external distributor preserved them.

### 15.6 Errors outside admission/workload

| Error family | Typical surface |
|---|---|
| Missing/invalid/expired authentication | 401 |
| Forbidden model/management/response ownership | 403 |
| Missing route | 404 |
| Malformed body / invalid `n` / invalid native input | 400 |
| nginx body exceeds limit | 413 |
| Tokenizer rejected input | 400 |
| Tokenizer unavailable / busy / invalid response | 503 |
| nginx outstanding-request bound | 503 |
| Engine connection failure before response | Gateway 502 `engine unreachable: ...` |
| Proxy unavailable / refused connection | Often nginx 502 |
| Upstream read timeout at nginx | 504 |
| Native generation/internal conversion exception | 500 or stream failure, depending on when it occurs |
| Upstream interruption after HTTP 200 | Incomplete stream/body; status cannot be changed retroactively |
| Client cancellation | Connection close / internally logged 499 |
| Class/output allowance exhausted | Often 200 with `finish_reason: length` |
| Media fallback note inserted | Often 200 after inference, not a rejection |

Unexpected exceptions are not all routed through `_refuse()`. They can surface as proxy 500s or connection failures rather than one of the structured admission messages.

---

## 16. Parameter dictionary and precedence

### 16.1 How effective values are selected

1. Base systemd unit supplies environment/default command arguments.
2. Drop-ins such as `60-`, `80-`, `90-`, and `95-` override repeated environment variables in order.
3. The container launcher passes selected environment variables into the control-plane container.
4. `server.py` converts environment strings to numbers and builds limit objects.
5. Throughput mode selects different code branches; an environment variable can remain present but no longer enforce its old gate.
6. Live `/diagnostics/capacity` and `/diagnostics/admission` show constructed controller/queue settings.

Python dataclass defaults, CLI defaults, repository reference YAML, and live process values are not interchangeable.

### 16.2 Active mode, execution, and queue variables

| Variable | Current value | Meaning |
|---|---:|---|
| `K3_WORKLOAD_GUARD` | 1 | Builds workload budget and inspector; also makes eligible LiteLLM context validation defer to gateway |
| `K3_THROUGHPUT_FIRST` | 1 | Selects shared adaptive admission rather than legacy class/workload gates |
| `K3_ADMISSION_CEILING` | 64 | Absolute gateway request-slot ceiling |
| `K3_ADAPTIVE_BORROW_MAX` | 24 | Throughput controller's maximum execution target |
| `K3_ADAPTIVE_INITIAL_EXECUTION` | 16 | Resting/start/recovery baseline |
| `K3_ADAPTIVE_MIN_EXECUTION` | 8 | Noncritical reduction floor |
| `K3_ADAPTIVE_BORROW_KV` | 0.88 | Throughput pressure threshold for observed KV |
| `K3_ADAPTIVE_GREEN_ITL` | 0.04 s | ITL expansion threshold |
| `K3_ADAPTIVE_BORROW_ITL` | 0.08 s | ITL pressure threshold |
| `K3_PROJECTED_KV_LIMIT` | 0.92 | Reservation ceiling fraction and projected memory limit |
| `K3_ADMISSION_WAIT_SECONDS` | 60 | Shared inspection/admission deadline budget after gateway normalization |
| `K3_ADMISSION_MAX_WAITERS` | 128 | Global queued ticket count |
| `K3_ADMISSION_MAX_PER_CUSTOMER` | 128 | Per-label queued count |
| `K3_ADMISSION_QUEUED_TOKENS` | 67,108,864 | Queued shape-token bound |
| `K3_ADMISSION_QUEUED_BYTES` | 1,073,741,824 | Accounted queued request-body bytes |
| `K3_ADMISSION_FAIRNESS_AGE_SECONDS` | 5 | Oldest-ticket cross-lane aging threshold |
| `K3_SEND_PRIORITY` | 1 | Adds class-derived priority to engine payload |
| `K3_MAX_MODEL_LEN` | 262,144 | Context clamp and opaque fallback basis |
| `K3_ENGINE_URL` | `http://127.0.0.1:8001` | Engine/inspection/metrics destination |

If the workload guard is off, merely leaving `K3_THROUGHPUT_FIRST=1` in the environment does not create the throughput controller. Throughput construction requires workload limits and a positive adaptive maximum.

### 16.3 Retained workload variables

| Variable | Current retained value | What still uses it |
|---|---:|---|
| `K3_LARGE_CONTEXT_TOKENS` | 65,536 | Large flag, lanes, reconciliation, telemetry |
| `K3_LARGE_CONTEXT_MAX` | 4 | Legacy count gate only; disabled in throughput mode |
| `K3_LONG_OUTPUT_TOKENS` | 2,048 | Long flag and prediction fallback |
| `K3_RESERVED_TOKEN_BUDGET` | 1,048,576 | Legacy/fallback reservation budget; measured-capacity fraction is used normally in throughput mode |
| `K3_LONG_OUTPUT_BURST_KV` | 0.55 | Legacy burst eligibility/queue tolerance and retained prediction telemetry |
| `K3_LONG_OUTPUT_BURST_RUNNING` | 16 | Legacy burst eligibility/queue tolerance |
| `K3_LONG_OUTPUT_BURST_PROMPT` | 32,768 | Legacy output-burst eligibility |
| `K3_LONG_OUTPUT_BASE_TOKEN_BUDGET` | 12,288 | Legacy prediction budget and retained telemetry |
| `K3_LONG_OUTPUT_BURST_TOKEN_BUDGET` | 49,152 | Legacy prediction burst budget |
| `K3_ENGINE_QUEUE_TOLERANCE` | 2 | Legacy heavy-work queue gate, not the source of the throughput controller's separate waiter threshold |
| `K3_OUTPUT_HISTORY_SIZE` | 128 | Maximum output observations per class |
| `K3_OUTPUT_MIN_SAMPLES` | 8 | Samples required before using empirical output quantile |
| `K3_OUTPUT_QUANTILE` | 0.95 | Historical output-length quantile |
| `K3_OUTPUT_SAFETY_FACTOR` | 1.25 | Prediction multiplier |

### 16.4 Constructor constants not exposed by the current environment parser

| Field/constant | Value | Role |
|---|---:|---|
| Controller `poll_seconds` | 0.5 | Background sampling |
| `freshness_seconds` | 2 | Maximum controller snapshot age |
| `increase_step` | 4 | Healthy expansion step |
| `green_kv` | 0.75 | Derived from the configured stop threshold, capped at 0.75 |
| `critical_kv` | 0.97 | Immediate pause threshold |
| `green_ttft` / `stop_ttft` | 2 / 5 s | Expansion / pressure TTFT thresholds |
| `green_prefill` / `stop_prefill` | 2 / 10 s | Expansion / pressure prefill thresholds |
| `pressure_samples` / `recovery_samples` | 3 / 4 | State confirmation |
| `adjustment_seconds` | 2 | Minimum interval between target adjustments |
| `decrease_factor` | 0.8 | Noncritical multiplicative decrease |
| Throughput `tolerated_waiters` | 2 | Engine queue pressure threshold |
| Admission `poll_seconds` | 0.2 | Queue condition wait interval |
| Context reserve | 256 tokens | Clamp headroom |
| Minimum remaining output space | 64 tokens | Context rejection threshold |
| Breaker KV distress | >0.97 | Shedding predicate |
| Breaker combined pressure | KV >0.90 and waiters >8 | Shedding predicate |
| Breaker preemption threshold | >1/minute | Shedding predicate |

Changing an environment variable that the parser never reads does not change these fields. Their current entry points are constructor/code settings unless explicitly wired into configuration.

### 16.5 Prefix/cache and timeout values

| Setting | Value |
|---|---|
| Physical cache block size | 768 tokens, measured engine setting |
| Engine prefix matching override | None |
| Engine prefix caching | Enabled |
| Engine Mamba cache mode | `align` |
| Prefix diagnostic event/history limits | 1,024 events / 4,096 completed-history entries |
| Prefix history age | 900 seconds |
| Exact-response cache mode | Default `static` |
| Exact-response cache maximum TTL | 300 seconds |
| Exact-response revision | `amd-5f3007-base-v1` |
| Gateway request socket timeout | 30 seconds |
| Engine upstream socket timeout | 600 seconds |
| Engine health scrape timeout | 5 seconds |
| nginx/LiteLLM configured request/upstream timeouts | 900 seconds |

Each timeout applies to its own operation. They do not combine into one consistent caller deadline automatically.

---

## 17. Configuration validation

These checks run when constructing the service/limit objects. They raise startup `ValueError` exceptions, not ordinary per-request overload JSON.

### 17.1 `AdmissionLimits`

| Requirement | Error message |
|---|---|
| Finite wait within `[0, 60]` seconds | `Admission wait must be between zero and 60 seconds` |
| Positive global/customer/token/byte queue limits | `Admission queue limits must be positive` |
| Finite polling interval in `(0, 1]` | `Admission poll interval must be between zero and one second` |
| Finite positive fairness age | `Admission fairness age must be positive` |

Setting a 120-second admission wait in the current environment alone fails validation. Supporting it requires changing the validated policy.

### 17.2 `WorkloadLimits`

| Requirement | Error message |
|---|---|
| Positive large threshold/count, long threshold/deprecated count, fallback token budget | `Workload limits must be positive` |
| Projected KV limit strictly between 0 and 1 | `Projected KV limit must be between zero and one` |
| Burst KV limit strictly between 0 and 1 | `Burst KV limit must be between zero and one` |
| Burst KV limit no greater than projected KV limit | `Burst KV limit cannot exceed the heavy-work limit` |
| Positive burst-running/prompt/token budgets and nonnegative legacy queue tolerance | `Invalid burst running limit or engine queue tolerance` |
| Burst token budget at least base token budget | `Long-output burst token budget cannot be below base` |
| Nonnegative adaptive maximum | `Adaptive borrow request limit cannot be negative` |
| Throughput mode requires positive adaptive maximum | `Throughput-first admission requires adaptive capacity` |
| Adaptive KV threshold strictly between zero and projected limit | `Adaptive borrow KV limit must be below the heavy-work limit` |
| Finite positive adaptive ITL limit | `Adaptive borrow ITL limit must be positive` |
| Positive history size and `1 <= min_samples <= history_size` | `Invalid output prediction history` |
| Quantile in `(0, 1]` | `Output prediction quantile must be between zero and one` |
| Finite prediction multiplier at least 1 | `Output prediction safety factor must be at least one` |

Some inactive legacy fields are still validated. Setting a retained limit to zero as an assumed disable switch can therefore prevent startup even when its runtime gate would be bypassed.

### 17.3 Base `CapacityLimits`

| Requirement | Error message |
|---|---|
| Positive maximum and increase step | `Adaptive request limits must be positive` |
| `0 < poll <= 5`, `poll < freshness <= 30` | `Invalid adaptive sampling intervals` |
| Each green/stop threshold pair finite and `0 < green < stop` | `Adaptive green thresholds must be below stop thresholds` |

### 17.4 `ThroughputLimits`

| Requirement | Error message |
|---|---|
| `1 <= minimum <= initial <= maximum` | `Minimum and initial capacity must fit the execution ceiling` |
| `stop_kv < critical_kv < 1` | `Critical KV threshold must be above the pressure threshold` |
| Pressure/recovery observations at least 2 | `Pressure and recovery require multiple observations` |
| Finite adjustment interval at least polling interval | `Execution adjustment interval must cover a sampling interval` |
| Decrease factor in `(0, 1)` and nonnegative tolerated waiters | `Invalid throughput decrease factor or queue tolerance` |

### 17.5 Other relevant constructor checks

- Empty classifier rules raise `at least one rule is required`.
- A classifier with no matching rule raises `no rule matched; the last rule must be a catch-all`.
- Nonpositive heuristic characters-per-token raises `chars_per_token must be positive`.
- Invalid response-cache mode raises `K3_RESPONSE_CACHE_MODE must be static, opt_in, or off`.
- Invalid numeric environment strings can fail `int()`/`float()` parsing before limit construction.

---

## 18. Worked examples

The following are illustrative applications of the inspected rules, not additional live load tests.

### 18.1 A request with no available execution slot

Assume the current target is 16 and all 16 slots are owned. Incoming request:

```text
prompt = 100 tokens
requested output = 1 token
tools/images = none
```

It classifies as P0, but shared-slot acquisition fails before workload memory checks. It may wait in the gateway queue. Its tiny size and priority do not create a free slot.

### 18.2 Queue has space, but a short request still times out

Assume the 16 running P1 requests each actually generate 2,048 tokens at 20 tokens/second:

```text
generation duration ≈ 2,048 / 20 = 102.4 seconds
```

If the controller stays at 16, a seventeenth request can wait alone in the 128-position queue and expire after 60 seconds before those requests finish.

An ITL around 50 ms is already above the 40 ms expansion threshold, so spare queue space and low KV usage do not automatically make the current controller expand.

If some running request finishes sooner or the controller expands, the waiter can start earlier. A single long request only blocks later work when the relevant admission resource is unavailable.

### 18.3 Grant-based long-output boundary

For a 100-token text prompt without tools/images:

| Requested output | Current class/grant | `long` workload flag |
|---|---|---|
| 1 | P0 / 1 | False |
| 512 | P0 / 512 | False |
| 513 | P1 / 513 | False |
| 2,047 | P1 / 2,047 | False |
| 2,048 | P1 / 2,048 | **True** |
| 8,192 | P1 / clamped to 2,048 | **True** |
| Omitted | P1 / default 2,048 | **True** |

The long flag uses the grant, not the original requested maximum or actual eventual output.

### 18.4 Prompt-length class boundaries

Without tools/images/batch and requesting 100 output tokens:

| Prompt count | Class |
|---|---|
| 8,192 | P0-interactive |
| 8,193 | P2-medium-context |
| 32,768 | P2-medium-context |
| 32,769 | P2-long-context |
| 65,536 | P2-long-context and workload `large=true` |

Adding tools or an image makes these agentic before the length rules run, unless an authorized batch hint matched first.

### 18.5 Reservation limit with low observed KV

For six opaque requests each reserving a full prompt window plus a 1,536-token grant:

```text
per-request charge = 262,144 + 1,536 = 263,680

five requests = 1,318,400 <= 1,508,713
six requests  = 1,582,080 >  1,508,713
```

With otherwise available slots and consistent ownership, the sixth is blocked by `reserved_tokens`. Low actual KV does not override those commitments.

### 18.6 Observed KV binds before commitment sum

Assume:

```text
existing commitments = 200,000 tokens
observed KV = 91%
incoming charge = 20,000 tokens
K = 1,639,906
```

The summed commitment 220,000 fits the budget, but:

```text
projected = max(0.91, 200,000 / K) + 20,000 / K
          ≈ 0.9222
          > 0.92
```

If earlier checks permit the attempt, the workload reason is `kv_headroom`.

### 18.7 Engine ownership discrepancy

If the engine reports four running/waiting requests but this gateway owns only two leases, a heavy incoming request requests a fresh observation. If the discrepancy persists, it is refused as `untracked_engine_work` even with low KV usage.

That refusal protects accounting; it is not proof that the engine is physically full.

### 18.8 Context reserve boundaries

```text
P = 261,824
available = 262,144 − 261,824 − 256 = 64
```

A one-token output can pass the local context clamp.

```text
P = 261,825
available = 63
```

The same one-token output is rejected because the policy requires at least 64 available tokens.

### 18.9 Dynamic quota versus execution admission

For the main key:

```text
key RPM/TPM mode = dynamic
failure signal = false
```

The key-level RPM/TPM checks can be omitted, but:

- Its 500-outstanding-request limit still applies.
- nginx's 256 per-IP/per-authorization outstanding limit still applies.
- Gateway execution is still capped by its adaptive 8–24 range or zero pause.
- Memory and queue bounds still apply.

Thus passing a dynamic quota check says nothing about whether the request can immediately obtain GPU-serving admission.

---

## 19. Metrics, diagnostics, and saved error evidence

### 19.1 Gateway diagnostic endpoints

| Endpoint | Content |
|---|---|
| `/diagnostics/capacity` | Controller state, effective/target execution limits, signals, streaks, freshness, configured limits |
| `/diagnostics/admission` | Queue limits/state and up to 512 finished queue/admission events |
| `/diagnostics/prefix` | Inspection mode, quarantine, usage/fingerprint diagnostics, bounded history |
| `/metrics` | Gateway counters, latency series, workload/queue/controller gauges |
| `/health` | Engine HTTP health result |

The diagnostic endpoints are loopback-only. A healthy `/health` response does not mean a free execution slot exists or that the controller will consider every workload green.

Read-only checks:

```bash
curl --silent http://127.0.0.1:8002/diagnostics/capacity
curl --silent http://127.0.0.1:8002/diagnostics/admission
curl --silent http://127.0.0.1:8002/metrics
```

### 19.2 Counters and how to interpret them

| Metric family | Meaning |
|---|---|
| `capacity_execution_limit` | Effective admission target, zero if paused/stale |
| `capacity_target_execution_limit` | Stored target before pause/freshness masking |
| `capacity_queued_demand` | Number of gateway waiting tickets |
| `capacity_pressure_streak` / `healthy_streak` | Consecutive controller classifications |
| `capacity_errors` | Background observation failures |
| `workload_active_requests` | Owned workload leases |
| `workload_reserved_tokens` | Full current execution commitments in throughput mode |
| `workload_large_context_requests` / `long_output_requests` | Workload flags among active leases |
| `workload_long_output_budget_enabled` | 0 in throughput mode |
| `workload_large_context_slot_limit_enabled` | 0 in throughput mode |
| `workload_long_output_slot_limit_enabled` | 0 in current code |
| `admission_queued_requests/tokens/bytes` | Current queue occupancy/charges |
| `admission_queued_total` | Requests that obtained waiting tickets |
| `admission_queued_admitted_total` | Queued requests later admitted |
| `admission_queue_timeouts_total` | Finished admission deadline expirations |
| `admission_queue_full_total` | All enqueue-bound refusals, including token/byte/customer bounds |
| `inspection_failures_total{reason=...}` | Inspection rejection/failure reason |
| `tokenizer_quarantined` | Exact-count trust disabled |

All names above carry the `k3_gateway_` prefix in actual exposition.

At startup, the retained `workload_long_output_last_token_budget` can show 12,288 before any acquisition sets it to zero for throughput mode. Check the explicit enabled flags; a nonzero initialized telemetry value does not prove the old budget gate is active.

### 19.3 Important observability gaps

- The admission diagnostic ring is not a complete list of all requests. Direct successful fast-path returns can bypass `_finish()` and its event append.
- Gateway `requests_total` is updated after admission returns. It is not a network-arrival counter and excludes earlier inspection/body failures.
- The request trace is written after the admission decision, so its timestamp is not necessarily the original arrival time.
- The trace's requested output can already have been clamped by LiteLLM.
- Current traces record class, shape, grant, reason, and admission wait, but not a full controller/lease snapshot for each failed attempt.
- The gateway's metric named TTFT measures first upstream body chunk after its local timer starts, not universally the first generated content token.
- Nonstreaming first-body timing is effectively whole-response time.
- Per-class latency series hold up to 2,048 observations and provide a 60-second live subset. Their `_count` is retained sample count, not a cumulative completion counter.
- Capacity controller inputs use native engine timing, not the gateway first-body metric.
- Counts reset with process restart unless separately persisted.

### 19.4 Customer identity

The gateway reads `X-K3-Customer`, defaulting to `anonymous`. nginx sets a label on the request to LiteLLM, but the inspected custom LiteLLM callback does not explicitly propagate authenticated identity into gateway egress headers.

The earlier 18:32–18:43 process-epoch inspection found **133/133 traces anonymous**. The newer 19:53 process epoch had no traces at the 19:58 snapshot, so there is no new traffic sample demonstrating a different result.

Authenticated response-cache scope in LiteLLM is not the same thing as authenticated tenant identity in gateway scheduling.

### 19.5 Saved error export

The previously analyzed `/root/all_logs.json` contains 10,000 distinct proxy calls with starts between **09:00:55 and 17:51:37 UTC**:

```text
success: 5,238
failure: 4,762  (47.62%)
```

Selected failure families:

| Historical reason | Count |
|---|---:|
| Tenant parallel limit 42 | 1,737 |
| `long_output_slots` | 992 |
| Class concurrency | 401 |
| `large_context_slots` | 378 |
| Legacy `engine_queue` | 319 |
| Shared execution capacity | 219 |
| `reserved_tokens` | 122 |
| `customer_queue_full` | 88 |
| `long_output_budget` | 82 |
| Incomplete streams | 68 |
| Tokenizer unavailable | 41 |
| Explicit TPM quota | 10 |

Overall, 4,387 failures matched quota/admission/capacity-control reasons. That establishes control-related failure surfaces, not that every refusal was avoidable.

These records include tests, changing settings, and earlier policies. They predate the current process/classification configuration. They cannot be used as a direct error rate for the 19:58 snapshot.

The earlier 90-second live observation at approximately 18:40 found 10 admission timeouts with a 7–16-entry gateway queue, no engine waiters, and KV below 38%, alongside genuine ITL spikes. It explains the motivation for this audit but is also an earlier-runtime observation.

---

## 20. Remaining interactions and improvement priorities

### 20.1 Why valid requests can fail with low KV

1. An adaptive execution target is occupied even though memory fits more requests.
2. The warm-state latency predicates prevent expansion.
3. A long generation holds a slot beyond the admission waiting budget.
4. Conservative full-context reservations consume commitment before physical allocation grows.
5. An ownership mismatch prevents safe heavy-work accounting.
6. An aged head-of-line request blocks smaller fitting requests.
7. An outer quota/transport check refuses the request before it reaches the engine.
8. Tokenizer or preprocessing capacity prevents timely arrival at execution admission.

Low KV is useful evidence, but it does not prove spare compute, correct accounting, or a free execution permit.

### 20.2 What dynamic behavior already exists

- A shared adaptive execution target with pressure reduction and recovery.
- Restoration of the configured baseline after healthy recovery.
- Bounded queued-demand-driven expansion above baseline.
- A reservation ceiling derived from measured KV capacity.
- Prediction history for completed long outputs.
- Conditional key RPM/TPM enforcement through LiteLLM's dynamic metadata.

These mechanisms operate independently. They are not yet one fully coordinated cost-/deadline-/tenant-aware scheduler.

### 20.3 Recommended next changes

| Priority | Change | Reason |
|---|---|---|
| 1 | Define whether tenant RPM/TPM are hard quotas; align metadata and counters with that contract | Current main-key dynamic mode conditionally omits the key-level checks |
| 2 | Use workload-/class-aware latency objectives and useful-completion feedback for execution expansion | A single strict global green threshold can strand queued demand |
| 3 | Add end-to-end caller deadlines and expected remaining work | A 60-second admission deadline is not suitable for every generation duration |
| 4 | Propagate authenticated tenant identity into gateway accounting | Current queue labels cannot reliably distinguish pooled end customers |
| 5 | Add weighted fairness and bounded backfilling around aged expensive work | Prevent short work from timing out behind a request that cannot currently fit |
| 6 | Improve exact inspection coverage and uncertainty-aware fallback counts | Full-window overreservation and heuristic undercounting are both undesirable |
| 7 | Make original requested versus granted output explicit | Class caps change observable output and reasoning completion behavior |
| 8 | Separate quota 429s from temporary serving-capacity errors consistently | Clients/distributors need distinct, actionable retry semantics |
| 9 | Add controller/lease snapshots, per-stage timing, correlation IDs, and stream-completion status | Current error reasons alone cannot reconstruct every historical admission state |
| 10 | Improve active cancellation and restart/drain coordination | Avoid unowned engine work and requests continuing after callers abandon them |

Queue count should remain bounded. Increasing 128 to a larger number does not solve a 60-second deadline or a frozen execution target when the queue was not full.

Memory commitment and expected compute cost should remain separate. Do not subtract a presumed prefix hit or replace a full output grant with a small prediction without validated engine-side memory enforcement.

### 20.4 Burst and recovery objective

```text
Normal traffic:
    meet normal latency targets

Temporary burst within tenant contract:
    retain useful requests in bounded waiting
    expand safe execution where doing so improves useful completion rate
    tolerate the agreed temporary latency increase

After burst:
    drain queued work
    return to the normal operating point

Sustained overload:
    meet deadlines through additional ready capacity,
    asynchronous processing where appropriate,
    or one clear bounded overload response
```

Approximate planning relationships:

```text
outstanding requests ≈ arrival rate × average request duration

new burst backlog ≈ max(0, burst arrival rate − service rate) × burst duration

drain time ≈ backlog / (service rate − normal arrival rate)
```

The drain relationship only works when normal offered work is below sustainable capacity. Mixed workloads need work-weighted estimates rather than treating a one-token reply and a large reasoning generation as equal.

### 20.5 Verification criteria for implementation changes

Use representative mixtures of short/long input, cold/warm prefixes, long reasoning output, tools/media, and cancellations. Compare:

- Correct useful completions within deadline.
- Terminal errors by layer/reason.
- End-to-end first generated token and first visible answer content.
- ITL and total completion time.
- Queue age and drain time.
- Tenant fairness and small-request progress.
- Output truncation/empty-answer rate.
- Reservation conservation and physical preemptions.
- Recovery after the burst ends.

The current inspection was read-only and idle at the recorded snapshot. It does not qualify a new concurrency maximum for production traffic.

### 20.6 Would removing restrictions and generating multiple keys solve the problem?

**Question:** If the custom hardcoded restrictions are removed and traffic is distributed across newly generated API keys, will that solve the throughput/error problem?

**Answer:** Not if those keys all route to the same K3 replica. Selectively relaxing an unnecessarily restrictive policy can improve acceptance and useful throughput. Creating additional credentials does not itself create additional GPU capacity, and removing all admission checks can move the backlog and failures deeper into the serving stack.

This answer uses the 19:58:52 snapshot above; it is not a new live throughput measurement.

#### A. A key is an identity/quota bucket, not an execution worker

Ten keys pointing to the same LiteLLM → gateway → vLLM route still share:

- The same TP8 model replica.
- The same prefill/decode compute.
- The same physical KV pool.
- The same gateway execution controller.
- The same gateway waiting queue and deadline policy.
- The same engine scheduler.

For example, 20 keys each permitting eight outstanding requests can offer 160 requests. That does not give this gateway 160 execution slots: the last verified controller still permits at most 24 simultaneous execution leases, or fewer under pressure. The remaining work must wait within the applicable bounds or be refused.

| Last verified control | Value | Does generating more keys change it? |
|---|---:|---|
| Main key parallel allowance | 500 | Separate keys can have separate allowances, but 500 already exceeds this backend's execution limit |
| Gateway baseline / maximum / pressure floor | 16 / 24 / 8 | No; this pool is global across keys |
| Engine configured sequence ceiling | 64 | No |
| Gateway waiting positions | 128 | No |
| Admission deadline | 60 seconds | No |
| Engine KV pool | Approximately 1.64 million reported tokens | No |
| nginx per-IP outstanding-request bound | 256 | No, if the portal requests still originate from the same IP |

The engine's 64-sequence ceiling is a configured operating limit, not a promise that every workload fits or performs well at concurrency 64. It is also not an immutable hardware constant; changing it requires workload qualification.

#### B. When multiple keys can help

1. **A restrictive per-key quota is the actual bottleneck.** Separately budgeted legitimate customers can avoid contending for one shared key allowance. The aggregate quotas still need to match the capacity and service contract. Adjusting an incorrectly small allowance directly may be simpler than adding credentials.
2. **Customer attribution and fairness.** One key per tenant enables separate usage records, quotas, budgets, and revocation. Fair scheduling also requires passing authenticated identity to the gateway and allocating capacity fairly there.
3. **Keys route to genuinely independent serving capacity.** If different routes reach additional ready model replicas or backend pools with independent capacity, load balancing can increase total service capacity. The increase comes from those backends, not from the number of keys. Different credentials for one shared provider/account capacity pool do not necessarily provide independent capacity.

The last verified main key permits 500 outstanding requests while the gateway's configured execution maximum is 24. The historical 42-key limit was important earlier, but splitting that key is not a direct fix for the later `shared execution capacity` failures.

#### C. What happens if all custom admission checks are removed?

The native engine still has its own scheduler, context validation, and resource limits. Removing the gateway gates does not remove those limits.

Possible effects include:

- More requests reach vLLM and wait in its engine queue instead of the gateway queue.
- Aggregate throughput can improve if the gateway was the binding artificial bottleneck and the engine can handle the additional work efficiently.
- TTFT can rise sharply as the engine backlog grows.
- Cold prefills can interfere more with already-running decodes.
- Memory contention can lead to preemption/recomputation rather than immediate useful progress.
- Requests can exceed caller/proxy deadlines or end with incomplete streams.

An OOM is not an inevitable consequence: vLLM has memory management and scheduling protections. However, unlimited offered work and unqualified memory/scheduling settings do not guarantee safe or useful service either.

The meaningful comparison is complete, correct responses delivered within deadline—not just fewer immediate gateway 429s.

#### D. Which restrictions should be retained or redesigned?

| Restriction | Recommended treatment |
|---|---|
| Model context validity | Retain; a credential cannot make an oversized context fit |
| Memory accounting and engine-health protection | Retain the protection; calibrate the policy and accounting accurately |
| Exact KV fractions such as 88%, 92%, 97% | Treat as configurable/qualified policy thresholds, not immutable laws of hardware |
| Fixed class concurrency/count gates | Already largely disabled in throughput mode; prefer a shared fair pool |
| Strict global latency conditions for expansion | Redesign around workload/class objectives and measured useful throughput |
| Output class caps | Review explicitly; relaxing them may improve answer completeness but increases generation duration and memory commitments |
| Waiting deadline | Make caller-/workload-aware; preserve an actual bound |
| Queue count/bytes | Keep bounded and size to useful backlog; additional queue space is not additional compute |
| Tenant RPM/TPM/spend limits | Enforce the intended tenant contract separately from temporary engine-capacity decisions |

Removing a 1,536-token output cap can help a reasoning-heavy answer finish, but a longer answer also occupies a slot longer. It is a quality/service-time tradeoff rather than a free throughput increase.

#### E. Two additional consequences of using many keys

- **Response-cache fragmentation:** the local exact-response cache normally includes authenticated key scope. Identical eligible requests sent under different keys can miss each other's response-cache entries and cause more live inference. The engine's shared prefix cache is a different mechanism and does not necessarily lose reuse just because the API key changes.
- **Fairness is not automatic:** the earlier gateway trace cohort used `anonymous` for every inspected request, and the custom callback still lacks explicit authenticated customer propagation to gateway scheduling. Merely issuing keys does not repair that identity path or create a per-tenant scheduler.

#### F. Recommended architecture and next experiment

Use separate tenant keys for identity and quota management, then route their requests through authenticated tenant attribution, bounded fair waiting, and **one shared backend-aware admission owner per engine**.

Selectively qualify less restrictive execution settings—for example, compare 16, 24, and 32 concurrent admissions on the same representative cold/warm, reasoning-heavy, tool/media workload. Preserve request/output semantics for a fair comparison and measure useful completion rate, p95 TTFT, ITL, timeouts, and preemptions. These values are candidate experiment points, not a claim that 32 is currently qualified.

Where a wait is useful, use a deadline compatible with the caller's needs. If sustained offered work exceeds the replica's sustainable service rate, use additional ready capacity or an appropriate asynchronous bulk-work path. A second full local K3 copy is not a free option on this already TP8-sized node.

**Decision:** use multiple keys for customer isolation and accounting; use selective, measured admission changes to improve utilization; add independent backend capacity when the hardware/service-rate limit is binding. Blanket removal of checks plus more keys is not a complete solution.

---

## 21. Source map and evidence

### 21.1 Repository source

| Component | Source |
|---|---|
| Request handling and final error mapping | [`redesign/gateway/server.py`](../redesign/gateway/server.py) |
| Policy ordering and release | [`redesign/gateway/policy.py`](../redesign/gateway/policy.py) |
| Workload commitments and predictions | [`redesign/gateway/workload.py`](../redesign/gateway/workload.py) |
| Queue admission and cleanup | [`redesign/gateway/admission.py`](../redesign/gateway/admission.py) |
| Adaptive shared execution controller | [`redesign/gateway/throughput.py`](../redesign/gateway/throughput.py) |
| Base controller, freshness, legacy waiting | [`redesign/gateway/capacity.py`](../redesign/gateway/capacity.py) |
| Circuit breaker and slots | [`redesign/gateway/backpressure.py`](../redesign/gateway/backpressure.py) |
| Classes and priorities | [`redesign/gateway/classification.py`](../redesign/gateway/classification.py) |
| Context/output clamp | [`redesign/gateway/clamping.py`](../redesign/gateway/clamping.py) |
| Inspection/fallback/quarantine | [`redesign/gateway/inspection.py`](../redesign/gateway/inspection.py) |
| Heuristic counting | [`redesign/gateway/tokens.py`](../redesign/gateway/tokens.py) |
| Metrics scrape and upstream transport | [`redesign/gateway/engine.py`](../redesign/gateway/engine.py) |
| Media and reasoning normalization | [`redesign/gateway/media.py`](../redesign/gateway/media.py) |
| Value types | [`redesign/gateway/models.py`](../redesign/gateway/models.py) |
| Off-box routing | [`redesign/gateway/offbox.py`](../redesign/gateway/offbox.py) |
| Request trace | [`redesign/gateway/capture.py`](../redesign/gateway/capture.py) |
| Gateway metrics | [`redesign/gateway/metrics.py`](../redesign/gateway/metrics.py) |
| LiteLLM custom hook | [`redesign/tenancy/callback.py`](../redesign/tenancy/callback.py) |
| First-hop clamp | [`redesign/tenancy/policy.py`](../redesign/tenancy/policy.py) |
| Exact-response-cache eligibility | [`redesign/tenancy/cache_policy.py`](../redesign/tenancy/cache_policy.py) |
| LiteLLM configuration renderer | [`redesign/tenancy/render_config.py`](../redesign/tenancy/render_config.py) |
| Throughput profile | [`redesign/deploy/profiles/throughput-admission.conf`](../redesign/deploy/profiles/throughput-admission.conf) |
| Edge throughput bounds | [`redesign/deploy/profiles/throughput-edge.inc`](../redesign/deploy/profiles/throughput-edge.inc) |

### 21.2 Installed dependency source

Dependency root:

```text
/usr/local/lib/k3/venv/lib/python3.12/site-packages/
```

Relevant files below that root:

```text
litellm/proxy/hooks/__init__.py
litellm/proxy/hooks/parallel_request_limiter_v3.py
litellm/proxy/hooks/max_budget_limiter.py
litellm/proxy/auth/budget_throttle.py
litellm/proxy/utils.py
litellm/router.py
litellm/router_utils/router_callbacks/track_deployment_metrics.py
litellm/constants.py
```

The quota behavior documented here is specific to the inspected installed library. Library upgrades can change it independently of the custom gateway source.

### 21.3 Runtime configuration

```text
/etc/nginx/conf.d/k3.conf
/etc/systemd/system/k3-gateway.service.d/60-amd-optimized.conf
/etc/systemd/system/k3-gateway.service.d/80-workload-guard.conf
/etc/systemd/system/k3-gateway.service.d/90-admission-queue.conf
/etc/systemd/system/k3-gateway.service.d/95-throughput-admission.conf
/etc/systemd/system/k3.service.d/60-amd-optimized.conf
/scratch/deploy-state/amd-optimized/config-base.yaml
/scratch/deploy-state/litellm.yaml
```

### 21.4 Snapshot and previous analysis artifacts

Current read-only snapshot:

```text
/tmp/opencode/k3-log-document-snapshot-20260922T195852Z.json
```

Snapshot script:

```text
/tmp/opencode/k3_log_document_snapshot_20260922.py
```

Earlier export/live-analysis artifacts, with their own timestamps:

```text
/tmp/opencode/k3-current-logic-export-20260922.json
/tmp/opencode/k3-current-logic-live-20260922.json
/tmp/opencode/k3-current-policy-mechanisms-20260922.json
```

Related documents:

- [Earlier TTFT and dynamic-limit explanation](TTFT-THROUGHPUT-DYNAMIC-LIMITS-2026-09-22.md)
- [Error catalog](ERROR-CATALOG-2026-09-22.md)
- [Historical burst analysis](TRAFFIC-BURST-ANALYSIS-2026-09-22.md)
- [Earlier throughput rollout](THROUGHPUT-ADMISSION-2026-09-22.md)
- [Admission recovery](ADMISSION-RECOVERY-2026-09-22.md)
- [Workload recovery](WORKLOAD-RECOVERY-2026-09-21.md)

### 21.5 Update log

| Evidence time | Entry |
|---|---|
| 2026-09-22 19:58:52 UTC | Created this detailed admission/workload reference from matching repository/installed source and a read-only runtime snapshot. Recorded baseline 16, maximum 24, floor 8, revised class/output rules, measured KV capacity 1,639,906, and main-key dynamic RPM/TPM metadata. |
| Follow-up discussion based on the same snapshot | Added section 20.6 explaining why removing restrictions and generating multiple keys does not add capacity to one backend, when separate keys help, and which controls should be retained or redesigned. |

Future observations should identify their deployment epoch and timestamp so that changed settings and historical failure counts remain interpretable.

### 21.6 Document verification

The Markdown structure/navigation check verified **21 numbered sections**, **94 closed fenced blocks**, and **72 table-of-contents or local-file links**. It also checked for unfinished drafting markers. The checker is `/tmp/opencode/k3_verify_log_markdown_20260922.py`; it validates this document, not production serving performance.
