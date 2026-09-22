# TTFT, throughput, admission limits, and dynamic burst handling

**System:** Kimi-K3 serving stack — nginx → LiteLLM → gateway → vLLM
**Evidence date:** 22 September 2026
**Live inspection:** approximately 18:40–18:43 UTC
**Repository:** `/root/cflow-gpu`

## Executive summary

Many observed failures came from admission restrictions before the engine exhausted its memory capacity. However, there is also genuine prefill/decode contention. The best improvement is a workload-aware admission controller rather than simply raising every limit.

The system already has dynamic admission. Its current latency rules can still prevent expansion while requests wait and eventually time out.

The main findings are:

1. The current controller starts at **8 concurrent executions**, can grow to **64**, and retains a **128-request queue with a 60-second admission deadline**.
2. Most historical fixed class, long-output-count, and large-context-count gates are disabled in throughput mode.
3. Expansion requires every available latency signal to satisfy a strict global healthy threshold. A TTFT of 2.3 seconds or an ITL of 45 ms can freeze expansion even with low KV usage and waiting demand.
4. During a 90-second passive observation, **10 requests expired in admission**, although the queue held only **7–16 requests** and active KV usage remained below **38%**. The blocking reason was shared execution capacity.
5. That same observation contained real decode-latency spikes. Low KV occupancy does not establish that compute is idle.
6. Effective customer fairness is missing: all 133 inspected gateway traces from the current process epoch used the `anonymous` customer label.
7. Output class ceilings remain active. HTTP 200 does not necessarily mean a complete, useful answer, especially when reasoning consumes the output allowance.

The desired objective is:

> Maximize correct, useful completions within caller deadlines, subject to memory safety, tenant quotas, agreed latency bounds, and customer fairness.

## Scope and evidence

This document records the analysis performed against:

- Repository source under `/root/cflow-gpu`.
- Installed gateway source under `/usr/local/lib/k3`.
- Live container and systemd configuration.
- Live gateway diagnostics and engine metrics.
- Read-only LiteLLM database queries.
- `/root/all_logs.json`.
- `/root/fuck.md`, the external endpoint defect report.
- The repository's error catalog and prior admission/performance investigations.
- Four CPU-only reproductions of the current controller's expansion rules.

**“Current” in this document means the inspected deployment at the timestamp above.** The exported logs end before the latest gateway process started at 18:32:40 UTC. Historical failures therefore cannot all be attributed to the inspected policy.

The recommendations below are proposed follow-up work. The observations and policy reproductions establish existing behavior; they do not establish a newly qualified maximum production capacity.

## Contents

1. [How the parameters connect](#1-how-the-parameters-connect)
2. [Current request path and restrictions](#2-current-request-path-and-restrictions)
3. [Exact dynamic-controller behavior](#3-exact-dynamic-controller-behavior)
4. [What the saved errors show](#4-what-the-saved-errors-show)
5. [Remaining live bottlenecks](#5-remaining-live-bottlenecks)
6. [Recommended dynamic burst-handling design](#6-recommended-dynamic-burst-handling-design)
7. [Prefix caching and engine tuning](#7-prefix-caching-and-engine-tuning)
8. [Implementation order and validation](#8-implementation-order-and-validation)
9. [Source map and analysis artifacts](#9-source-map-and-analysis-artifacts)

---

## 1. How the parameters connect

These parameters control different resources. Treating them as interchangeable causes misleading capacity conclusions.

| Parameter | What it measures or controls | Connection to performance |
|---|---|---|
| **RPM** | Requests admitted per rate-limit window | Controls arrival volume, regardless of whether a request is cheap or expensive. |
| **TPM** | Token allowance per window | Controls token consumption/reservations. It does not directly describe GPU processing capacity. |
| **Execution concurrency** | Requests currently admitted to the engine | More concurrency can improve aggregate throughput, but can slow each request. |
| **KV cache** | Model state retained for active contexts and generation | Limits how much context/generation can coexist in memory. |
| **Prefix cache** | Reusable state for matching prompt prefixes | Saves prefill computation. It does not eliminate subsequent generation. |
| **TTFT** | Time until the first generated token | Includes waiting and prompt processing, depending on where it is measured. |
| **ITL** | Inter-token latency during generation | Determines generation speed after the first token. |
| **Admission queue** | Accepted work waiting outside the engine | Absorbs bursts by exchanging additional latency for fewer rejections. |
| **Engine batch-token budget** | Tokens scheduled in an engine iteration | Affects prefill efficiency and interference with ongoing decoding. |

### 1.1 Customer-visible TTFT is a sum of stages

Approximately:

```text
Customer TTFT =
    network/proxy overhead
  + media preparation/tokenization
  + gateway admission waiting
  + engine waiting
  + prefill
  + first-token generation/delivery
```

For a reasoning model, distinguish first generated token from first answer content:

```text
Time to first answer content
  ≈ time to first generated token
    + time spent generating reasoning before the answer
```

The defect report's **54.5-second time to first content** is not the same measurement as the native engine TTFT used by the controller.

The gateway's metric named `ttft` measures the first upstream body chunk. For nonstreaming requests, that is effectively whole-response latency. Even for streaming, the first chunk need not contain a generated token.

Relevant source: [`server.py`](../redesign/gateway/server.py#L387).

### 1.2 Long outputs hold capacity even when KV usage is low

For an illustrative generation rate of 30 tokens/second:

```text
4,096 generated tokens ÷ 30 ≈ 137 seconds
```

That request holds an execution slot for more than two minutes. A request behind it may exceed the current 60-second admission deadline even when the waiting queue is far below its count limit.

Reasoning tokens count toward this generation work.

### 1.3 A quota is not a throughput guarantee

For a stable workload, Little's Law gives:

```text
Average outstanding requests ≈ arrival rate × average request duration
```

For example:

```text
3,000 RPM = 50 requests/second
50 × 30-second average duration = 1,500 outstanding requests
```

An engine configured for 64 sequences cannot sustain that particular workload merely because a key permits 3,000 RPM.

Similarly, 3 million TPM of mostly cached input is very different from 3 million TPM of newly generated output. Prompt processing, cached-prompt reuse, and autoregressive decoding have different resource costs.

### 1.4 Low KV usage does not prove spare compute

Three percentages must remain distinct:

- **`gpu-memory-utilization: 0.88`:** engine startup memory-budget setting.
- **`kv_cache_usage_perc`:** active usage of the engine's KV pool.
- **GPU compute utilization:** how busy the accelerators are executing work.

Low KV occupancy can coexist with substantial prefill/decode interference.

Also, idle reusable prefix-cache blocks may remain on the engine's free list. Low active KV usage does not mean that all reusable prefixes have been erased.

---

## 2. Current request path and restrictions

The local serving path is:

```text
Client / external portal or NewAPI
              ↓
        nginx :443
              ↓
       LiteLLM :4000
  authentication, quotas, response cache
              ↓
        Gateway :8002
  normalization → inspection → classification
  → output clamp → admission queue/reservations
              ↓
         vLLM :8001
      prefix lookup → prefill → decode
```

The engine is **one TP8 replica**. Eight GPUs do not mean eight independent replicas or eight independent 64-request pools.

### 2.1 nginx: transport bounds

The inspected inference location applies:

- **256 outstanding requests per IP**.
- **256 outstanding requests per authorization value**.
- **64 MiB request-body limit**.
- **900-second upstream read timeout**.
- Response buffering disabled for streaming.

The previous **20 requests/second shared-IP limiter is no longer applied to the inference location**. Its zone declaration remains in the configuration, but the active throughput include applies connection limits instead.

An exhausted connection limit returns **HTTP 503**. An oversized body can be rejected by nginx with **HTTP 413**.

Sources:

- `/etc/nginx/conf.d/k3.conf`
- [`throughput-edge.inc`](../redesign/deploy/profiles/throughput-edge.inc)

### 2.2 LiteLLM: authenticated quota enforcement

The busiest key's current database settings were:

| Setting | Inspected value |
|---|---:|
| `rpm_limit` | **3,000,000** |
| `tpm_limit` | **3,000,000** |
| `max_parallel_requests` | **500** |

The RPM value differs from the **3,000 RPM** in the defect report. The historical **42-parallel-request** limit also differs from this setting. Other keys have separate allowances, and applicable higher-level quotas may also constrain requests.

The installed limiter:

1. Checks request-rate and outstanding-request limits.
2. Reserves estimated input tokens plus output allowance against TPM.
3. Adjusts that reservation against actual usage on completion.
4. Releases the outstanding-request slot on completion/failure.

Its default window is **60 seconds**. Its combined-TPM upfront input estimate is character-based rather than the gateway's exact rendered-token count.

In the installed default total-token reconciliation path, reported cached input is subtracted. That is quota accounting, separate from GPU memory accounting and billing.

Consequences:

- Requests waiting downstream still occupy LiteLLM outstanding-request slots.
- TPM reservations can be exhausted before that amount of work has actually been generated.
- A large output allowance can reserve more quota than the request eventually consumes.

Quota failures return **HTTP 429**. The installed limiter supplies `Retry-After`; propagation through the external portal needs end-to-end verification.

The inspected local router has `num_retries: 0`. External client/portal retries are a separate layer.

Sources:

- `/scratch/deploy-state/litellm.yaml`
- `/usr/local/lib/k3/venv/lib/python3.12/site-packages/litellm/proxy/hooks/parallel_request_limiter_v3.py`
- [`render_config.py`](../redesign/tenancy/render_config.py)

### 2.3 Prompt inspection: exact counts where supported

The gateway calls the engine's **`/tokenize` endpoint** for supported text/tool requests.

Current behavior:

- **4 concurrent inspections**.
- Up to **10 seconds** for an individual tokenizer HTTP operation, bounded by the remaining admission deadline.
- Inspection-slot waiting shares the admission deadline.
- A token-count disagreement with completed inference quarantines exact inspection.

| Condition | Response |
|---|---|
| Inspection slot unavailable until deadline | **503**, `Prompt inspection admission deadline exceeded` |
| Tokenizer rejects input with 4xx | **400**, `Active model tokenizer returned HTTP ...` |
| Tokenizer unavailable or timed out | **503**, `Prompt inspection is temporarily unavailable` |
| Invalid tokenizer response | **503**, related inspection-response error |

This prompt inspection uses the engine tokenizer; it is not a separate moderation service. The 41 inspection-unavailable failures in the export occurred during the earlier engine-reload window.

#### Conservative fallback changed after the earlier error catalog

Current code uses estimates for ordinary multimodal/constrained formats. Full-window reservations remain for opaque inputs such as prompt embeddings and certain batched token inputs.

Therefore, the earlier statement that all multimodal/constrained requests reserve an entire 262,144-token window is historical, not the inspected behavior.

Source: [`inspection.py:59–119`](../redesign/gateway/inspection.py#L59).

### 2.4 Classification and output clamping

Classification is first-match-wins:

| Order | Request shape | Class | Output ceiling |
|---|---|---|---:|
| 1 | Authorized batch hint | P3-batch | 32,768 |
| 2 | Tools or images | P2-agentic | **1,536** |
| 3 | Prompt ≤8,192 tokens | P1-short-chat | **4,096** |
| 4 | Prompt ≤32,768 tokens | P0-interactive | 8,192 |
| 5 | Larger prompt | P2-long-context | 16,384 |

The clamp is approximately:

```text
available = 262,144 − prompt_tokens − 256

if available < 64:
    reject with 400
else:
    granted = min(
        requested output, or class default,
        class output ceiling,
        available
    )
```

The 64-token rule concerns **remaining context space**, not a minimum requested output. One-token requests are allowed when sufficient context space remains.

Clamping happens at both LiteLLM and the gateway. The newer `max_completion_tokens` alias takes precedence in the local policy when both supported aliases are supplied.

**Throughput mode removed several admission gates, but did not remove these output ceilings.** A request asking for 32,000 tokens can still receive a 4,096-token grant in P1.

This normally returns **HTTP 200**, not an HTTP error. The gateway reports the grant through `x-k3-max-tokens-granted`. Generation reaching the allowance can end with `finish_reason: length`.

If reasoning consumes the allowance before answer content appears, a successful HTTP response may contain no answer text.

The gateway also requires `n` and `best_of`, when supplied, to be the integer 1; otherwise it returns **HTTP 400**.

Sources:

- [`classification.py:28–37`](../redesign/gateway/classification.py#L28)
- [`clamping.py:49–79`](../redesign/gateway/clamping.py#L49)
- [`tenancy/policy.py:43–92`](../redesign/tenancy/policy.py#L43)
- [`server.py`](../redesign/gateway/server.py)

### 2.5 Execution and KV reservations

The active engine configuration is:

```text
max-model-len:          262,144
max-num-seqs:           64
max-num-batched-tokens: 4,096
prefix caching:        enabled
```

The gateway currently reserves:

```text
request reservation = reserved prompt tokens + full granted output
```

Measured engine KV-token capacity:

```text
1,641,413 tokens
```

Current reservation ceiling:

```text
floor(1,641,413 × 0.92) = 1,510,099 tokens
```

Admission checks both:

```text
existing reservations + incoming reservation <= reservation ceiling
```

and:

```text
projected KV =
    max(observed KV usage, existing reservations / capacity)
  + incoming reservation / capacity

projected KV <= 92%
```

It also checks for engine work that the gateway does not own, particularly for heavy requests.

These checks produce:

- `workload budget: reserved_tokens`
- `workload budget: kv_headroom`
- `workload budget: untracked_engine_work`
- `workload budget: engine_health_unavailable`

These reservations are separate from TPM quotas.

The token-based reservation is a conservative admission model. K3 has hybrid attention/recurrent-state cache geometry, so measured engine memory pressure still matters.

In throughput mode, the historical predicted-output budget and large-context request-count limit do not gate execution. Output predictions remain available as telemetry, while memory reservations cover the full granted output.

If a workload reservation fails after a class/shared execution slot is acquired, the policy releases that execution slot. Completed, failed, or cancelled admitted requests release their leases; lease release is idempotent.

Sources:

- [`workload.py:182–248`](../redesign/gateway/workload.py#L182)
- [`policy.py:148–192`](../redesign/gateway/policy.py#L148)
- `/scratch/deploy-state/amd-optimized/config-base.yaml`

### 2.6 Admission queue

Current bounds:

| Control | Value |
|---|---:|
| Maximum waiting requests | **128** |
| Maximum per customer label | **128** |
| Queued shape-token accounting | **67,108,864** |
| Accounted queued request-body bytes | **1 GiB** |
| Admission deadline | **60 seconds** |
| Queue polling interval | 200 ms |
| Cross-lane aging threshold | 5 seconds |

Queued requests hold **queue capacity**, not engine/KV execution reservations.

The gateway retries admission checks until capacity becomes available, the deadline expires, or the client disconnects. It does not replay inference as part of queue waiting.

Each traffic-class/resource lane is FIFO. After five seconds, the oldest waiter receives cross-lane priority. An individually impossible reservation should not enter the queue and indefinitely block other work.

The current 60-second deadline starts **after gateway normalization**. It is not a complete end-to-end deadline covering all outer proxy and media-preparation time.

Source: [`admission.py`](../redesign/gateway/admission.py).

---

## 3. Exact dynamic-controller behavior

The inspected controller starts at **8 executions**, can grow to **64**, and samples health every **500 ms**.

### 3.1 Expansion conditions

All applicable conditions must be satisfied:

| Signal | Expansion requirement |
|---|---|
| Gateway demand | Requests waiting |
| Engine queue | Zero waiters |
| Observed KV | ≤75% |
| Mean ITL | ≤40 ms |
| Mean engine TTFT | ≤2 seconds |
| Mean prefill time | ≤2 seconds |
| Confirmation | Four healthy observations |

Then:

```text
execution limit += 4
```

This adjustment happens at most once every **2 seconds**, up to 64. Even under continuously healthy demand, growing from 8 to 64 takes approximately **28 seconds**.

The latency comparisons apply when their values are available. Missing latency values are not treated as threshold violations by the comparison helper.

### 3.2 Pressure conditions

Pressure includes:

- KV ≥88%.
- More than 2 engine waiters.
- Mean ITL >80 ms.
- Mean TTFT >5 seconds, with supporting load evidence.
- Mean prefill >10 seconds, with supporting load evidence.

Supporting load evidence includes an engine waiter, ITL above 40 ms, or at least 8 observed active requests.

After three pressure observations, ordinary pressure reduces the limit approximately as:

```text
limit = max(8, floor(limit × 0.8))
```

This decrease happens at most every two seconds. A sustained engine queue pauses additional starts.

### 3.3 Immediate pause conditions

New starts pause immediately for:

- KV ≥97%.
- A detected preemption.
- Unavailable/stale capacity observations.

The controller's snapshot freshness limit is **2 seconds**.

Already-running requests continue. Consequently, observed running requests can temporarily exceed a newly reduced admission limit.

### 3.4 Recovery

- Four observations without overload allow starts to resume.
- Stronger healthy conditions are required for expansion.
- Once idle and healthy, the controller returns to the **8-request baseline**.
- Queue count and waiting deadline remain available during pressure.

### 3.5 What is dynamic versus fixed

| Dynamic today | Fixed in the inspected policy |
|---|---|
| Execution concurrency | Per-class output ceilings |
| KV admission ceiling derived from measured engine capacity | Queue count/token/byte bounds |
| Execution pauses and recovery | 60-second maximum admission wait |
| Shared use of available execution capacity | Global latency thresholds |
| | Tenant quota settings |
| | Engine 4,096-token batch budget |

Sources:

- [`throughput.py:70–211`](../redesign/gateway/throughput.py#L70)
- [`capacity.py:18–30`](../redesign/gateway/capacity.py#L18)
- [`throughput-admission.conf`](../redesign/deploy/profiles/throughput-admission.conf)
- `/etc/systemd/system/k3-gateway.service.d/95-throughput-admission.conf`

---

## 4. What the saved errors show

### 4.1 Export population

`/root/all_logs.json` contains **10,000 distinct proxy call records**, covering request starts from **09:00:55 to 17:51:37 UTC**:

- **5,238 successes**.
- **4,762 failures — 47.62%**.

These are exported attempts, including tests and multiple deployment versions. They are not a current-policy production error rate and do not include every failure rejected before reaching LiteLLM.

### 4.2 Main restriction-related failure families

| Logged reason | Count | Interpretation |
|---|---:|---|
| Tenant `max_parallel_requests: 42` | **1,737** | Historical shared-key outstanding-request limit |
| `long_output_slots` | **992** | Historical fixed long-generation count limit |
| Class concurrency limit | **401** | Mostly P1 reaching its 16-request allocation |
| `large_context_slots` | **378** | Historical fixed large-context count limit |
| `engine_queue` | **319** | Historical workload guard refusing heavy work |
| `shared execution capacity` | **219** | Adaptive execution capacity unavailable after waiting |
| `reserved_tokens` | **122** | Reservation accounting exhausted |
| `customer_queue_full` | **88** | Customer-labelled queue bound |
| `long_output_budget` | **82** | Historical predicted-output budget |
| Actual TPM limit | **10** | Explicit 3-million-token quota rejection |

No explicit RPM-limit failures were identified in this export.

Overall, **4,387 failures — 92.1% of failures — matched quota/admission/capacity-control reasons**. That does not establish that all were avoidable: some controls were responding to real contention.

### 4.3 Why the historical restrictions rejected too aggressively

Earlier policies stacked several gates:

1. **P1 could hit 16 requests while other class capacity was unused.**
2. **Only two long-output requests were allowed** under the original workload guard.
3. **Only four large-context requests were allowed.**
4. Predicted-output budgets switched between **12,288 and 49,152 tokens** depending on the arriving request and controller state.
5. The old controller reduced waiting from **20 seconds → 1 second → zero** as it moved from green to warm to pressure.
6. Conservative formats previously reserved an entire **262,144-token context**.

The output-budget switch was especially problematic: existing work could fit the larger allowance, while a newly arriving long-context request was evaluated against the smaller allowance. That could block one lane despite spare memory.

These mechanisms could reject requests well before actual KV exhaustion. Most of these gates are disabled in current throughput mode.

The export also contains **124 failures saying `shared execution capacity: 1`**. That was an earlier controller behavior; the inspected noncritical floor is 8.

Historical detail:

- [Traffic burst analysis](TRAFFIC-BURST-ANALYSIS-2026-09-22.md)
- [Admission recovery](ADMISSION-RECOVERY-2026-09-22.md)
- [Workload recovery](WORKLOAD-RECOVERY-2026-09-21.md)

### 4.4 Current direct-gateway error semantics

| Current condition | HTTP / error |
|---|---|
| Tenant RPM/TPM/parallel quota | **429**, LiteLLM rate-limit error |
| Shared execution capacity unavailable | **429**, `shared execution capacity: N` |
| Reservation/headroom constraint | **429**, `workload budget: ...` |
| Queue count/token/byte bound | **429**, `admission queue: ...` |
| Admission deadline expires | Usually **429**, code `queue_timeout`, retaining the last blocking reason |
| Circuit breaker sheds eligible traffic | **503**, `engine distressed: ...` |
| Token inspection unavailable | **503** |
| Prompt/context invalid | **400** |
| Engine connection failure | **502** |
| Upstream stream breaks after headers | Broken stream; HTTP may already be **200** |

Specific queue-capacity reasons include:

- `queue_full`
- `customer_queue_full`
- `queued_token_limit`
- `queued_byte_limit`

The circuit breaker sheds eligible long-context/batch-priority traffic for severe KV/preemption distress. The shared controller can independently pause starts across classes.

Gateway admission refusals currently include a wait-duration header and generally `Retry-After: 2`. Inspection-unavailable responses use `Retry-After: 5`.

This differs from the earlier throughput document, which described ordinary capacity errors as 503. The installed [`server.py:451–470`](../redesign/gateway/server.py#L451) is authoritative for the inspected behavior.

Application codes such as `reject_budget` and `queue_timeout` are separate from HTTP status. The export's `error_code` field therefore must not be treated as a clean HTTP-status column.

### 4.5 Other errors are not fixed by increasing limits

The export includes:

- **68 incomplete-stream errors**.
- **41 tokenizer-unavailable errors**.
- **34 connection-related errors**.
- Request-format, tool-choice, and other compatibility failures.

The external portal also reported opaque upstream errors and unavailable-channel errors. End-to-end request correlation is needed to establish their underlying local causes rather than inferring them solely from the portal's final status/message.

The broader error-family inventory is in [ERROR-CATALOG-2026-09-22.md](ERROR-CATALOG-2026-09-22.md). Its snapshot-specific settings should be read alongside this later inspection.

---

## 5. Remaining live bottlenecks

### 5.1 The queue is available, but execution expansion still stalls

During the **90-second passive observation**, approximately 18:40:01–18:41:31 UTC:

| Measurement | Observed |
|---|---:|
| Admission-limit range | **8–17** |
| Engine running requests | 11–17 |
| Gateway waiters | **7–16** |
| Engine waiters | **0 throughout** |
| Active KV | **9.5–37.9%** |
| New admission timeouts | **10** |
| Queue-full events | **0** |
| Preemptions | **0** |
| Generated tokens | 26,882, approximately **298 tokens/s** |

The failed requests were blocked by shared execution capacity rather than queue space or KV reservations.

However, the controller also observed significant ITL spikes, up to approximately **570 ms** in its smoothed signal. There was genuine generation interference.

Both facts matter: capacity was being withheld by policy, and some pressure signals were real. This observation does not prove that raising concurrency to 64 would improve this workload.

### 5.2 The warm state can freeze growth indefinitely

The current controller was reproduced offline with:

```text
KV = 6%
Engine waiters = 0
Gateway waiters = 32
Starting limit = 8
```

| Signal combination | Limit after 60 simulated seconds |
|---|---:|
| ITL 30 ms, TTFT 1 s, prefill 0.5 s | **64** |
| Same, but TTFT 2.3 s | **8** |
| Same, but ITL 45 ms | **8** |
| Same, but prefill 2.6 s | **8** |

These are policy reproductions, not GPU performance predictions.

They demonstrate the central issue:

> A request mix can be acceptable for throughput-oriented burst service but fail the controller's strict all-green test, preventing expansion while its queue expires.

The per-class TTFT targets in `classification.py` do not drive this controller. It uses global thresholds instead.

### 5.3 Queue aging can create head-of-line blocking

After five seconds, the oldest waiter receives the next opportunity across lanes.

If that request cannot currently fit, other lanes can be blocked even when a smaller request could run. Aging protects large jobs from starvation, but the present mechanism can delay cheap requests behind expensive ones.

In the live diagnostic history, even a **95-input-token, one-output-token request** expired after approximately 60 seconds.

### 5.4 Customer fairness is not currently effective

All **133 gateway traces** inspected from the current process epoch used the **`anonymous` customer label**.

The scheduler therefore cannot reliably distinguish customers behind the shared provider key.

Also, the per-customer queue allowance equals the entire global queue allowance. It does not provide meaningful subdivision of that queue.

### 5.5 Output clamping can make successful throughput misleading

In the exported non-replayed K3 successes with reasoning details, roughly **77% of generated tokens were reasoning tokens**.

That work still:

- Consumes decode compute.
- Holds concurrency.
- Extends queue waiting.
- Consumes output allowance.

The defect report's empty-answer finding is consistent with reasoning exhausting a small allowance. Its exact 32% empty-answer rate belongs to that report's test population; the spend-log export alone does not contain enough response content to independently reproduce it.

Because LiteLLM can clamp the allowance before the gateway sees it, a gateway trace saying `clamp_reason: unchanged` does not by itself establish that the original caller's allowance was preserved.

### 5.6 The latest reservation fallback needs accurate accounting

The previous full-context fallback was overly restrictive for many ordinary requests. Current code has relaxed it.

But the heuristic is not a proven upper bound for all formats. Existing repository experiments show both overestimation and underestimation, and the fallback extractor does not account for all the same fields as exact rendered inspection.

The right improvement is **better token-count coverage and measured uncertainty**, rather than assuming every estimate is conservative.

---

## 6. Recommended dynamic burst-handling design

The intended behavior is:

```text
Normal traffic:
    low waiting and low TTFT

Temporary burst:
    buffer valid requests
    use additional safe execution capacity
    allow an agreed temporary latency increase

After burst:
    drain backlog
    return to normal latency

Sustained overload:
    bounded waiting, useful overload response, or additional capacity
```

This needs coordinated controls in five areas.

### 6.1 Keep tenant quotas separate from serving capacity

Recommended semantics:

- **429:** actual customer RPM/TPM/spend quota restriction.
- **503:** temporary serving-capacity exhaustion.
- **400:** invalid input.
- Stable machine-readable reasons and meaningful retry guidance.

Changing status alone does not improve throughput. It helps clients and distributors choose the correct response to a failure.

First reconcile the intended RPM contract with the busiest key's current **3,000,000 RPM** database value.

For burst shaping, a token bucket can be useful. However, if the contract requires a strict maximum in every rolling minute, enforce that separately: an ordinary token bucket can permit more than one minute's refill rate over a short interval.

Quota reservation and reconciliation should use a consistent effective request shape, preserve the relevant accounting-window identity, and release reservations exactly once.

Outer outstanding-request limits must accommodate execution, queueing, and bounded preparation without defeating the intended gateway buffer.

### 6.2 Make the controller workload-aware and deadline-aware

Each request should carry:

```text
authenticated tenant
end-to-end deadline
accurate input count / uncertainty
estimated uncached prefill work
expected output, including reasoning
memory commitment
latency class
```

Use distinct budgets for:

| Budget | Purpose |
|---|---|
| **KV/memory** | Prevent unsafe memory commitments |
| **Prefill work** | Limit cold-prompt interference |
| **Decode work** | Manage long-running generation occupancy |
| **Queue/deadline** | Decide whether waiting can still produce a useful result |

The current output predictor mostly provides telemetry in throughput mode. It should help estimate service time and scheduling cost, while memory safety remains separately enforced.

Do not reduce memory reservations solely because a prompt was seen before. A prior prefix fingerprint does not prove current residency or physical sharing.

Expected service work and hard memory commitments should remain separate. Adopting incremental memory reservations would require engine-supported progress/accounting and a validated enforcement mechanism; output-length prediction alone is insufficient.

### 6.3 Permit controlled expansion in the acceptable warning band

Instead of requiring every latency signal to be below a universal green threshold:

1. Compare latency against the request class and allowed burst SLO.
2. Separate intrinsic cold-prefill time from congestion-induced waiting.
3. Expand gradually when doing so improves useful completions.
4. Reduce cold-prefill admission when it harms existing decodes.
5. Retain immediate memory/preemption protection.

The optimization objective should be:

```text
maximize correct, useful completions within caller deadlines
```

subject to:

```text
memory safety
tenant quotas
agreed TTFT and generation-latency bounds
fairness
```

A 2.3-second cold TTFT should not automatically prevent all expansion if that workload permits it.

Likewise, distinguish **three fresh latency observations** from three controller polls reusing the same retained signal. Current latency estimates can persist between observations.

After the burst, reduce borrowed capacity with recovery hysteresis while allowing running work to finish. The normal operating point and burst envelope should be selected from representative measurements rather than assumed to be universally 8 or 64.

### 6.4 Use fair queues with bounded backfilling

A better scheduler should support:

- Authenticated per-tenant weighted fairness.
- Different treatment for cheap interactive work and expensive cold work.
- Borrowing of unused capacity.
- Aging for expensive requests.
- **Bounded backfilling:** running a smaller fitting request without indefinitely delaying the older large one.

This is more effective than introducing another fixed two-long-requests-only restriction.

The queue should remain available during ordinary congestion.

### 6.5 Size waiting against actual service capacity

For an approximate burst calculation:

```text
backlog added ≈ max(0, burst arrival rate − service rate) × burst duration
```

After the burst:

```text
drain time ≈ backlog / (service rate − normal arrival rate)
```

This requires normal arrival rate to be below sustainable service rate. For mixed workloads, work-weighted estimates are more informative than treating every request as equal.

A larger queue only helps when its contents can complete before callers give up. The observed live queue was far below 128, so raising the count limit further is not the first improvement.

For latency-tolerant requests, longer deadlines may be appropriate. The current `AdmissionLimits` validator explicitly caps waiting at 60 seconds, so this requires a code-level policy change rather than only setting an environment variable to 120.

Minute-scale bulk workloads are better served through an asynchronous job path.

If sustained offered work exceeds the qualified capacity of this replica, queueing cannot make the backlog disappear. Additional warm capacity or controlled rejection is then necessary. No off-box target was configured in the inspected deployment.

### 6.6 Coordinate retries, cancellation, and error reporting

- Choose one retry owner across SDK, portal, and service layers.
- Preserve `Retry-After`, use jitter, and bound the total retry budget.
- Respect the caller's remaining deadline.
- Avoid replaying partially delivered generation.
- Cancel/reconcile abandoned execution promptly.
- Carry a request ID and stable reason through every hop.
- Distinguish completed streams from responses that only began with HTTP 200.

These measures reduce retry amplification and prevent retries from consuming the capacity intended to recover from a burst.

---

## 7. Prefix caching and engine tuning

### 7.1 Two different caches

| | Exact response cache | Prefix cache |
|---|---|---|
| Location | LiteLLM/Redis | vLLM |
| Reuses | Entire previous response | Prompt-processing state |
| Avoids generation? | Yes, on replay | No |
| Inspected behavior | Eligible static text, up to 300-second TTL | Enabled; engine-managed eviction |
| Main optimization | Explicit cache policy and correct accounting | Stable prompt prefixes |

Tools, tool history, media, and stateful interactions bypass the local exact-response cache under its current policy.

### 7.2 Improve prefix reuse at the prompt-layout level

- Keep stable instructions/reference material before changing runtime fields.
- Keep tool declarations consistently ordered.
- Measure actual cached tokens rather than assuming a repeated request must hit.
- Account for K3's recurrent-state checkpoints: matching tokens alone do not guarantee reusable state.

The existing prefix experiment measured **96.1% reuse with stable text before a changing date, versus zero reuse with the date first**.

The inspected engine reports a **768-token physical cache block size** and Mamba `align` mode. The tested finer prefix-matching override was not selected for the active baseline.

See [prefix-cache experiment results](../experiments/2026-09-21-prefix-cache/RESULTS.md).

### 7.3 Engine tuning has workload-dependent tradeoffs

- The current 4,096-token batch budget affects both prefill efficiency and decode interference.
- The earlier 2,048-token experiment reduced some streaming gaps but **reduced aggregate output throughput by approximately 25–26%** on completed mixed-workload comparisons.
- Warm-only measurements showed concurrency 16 delivering more aggregate output than concurrency 8, while individual streams slowed.

These results support a workload-dependent operating point, rather than a universal assumption that 8 or 64 is optimal.

See [mixed-prefill experiment results](../experiments/2026-09-21-mixed-prefill/RESULTS.md).

---

## 8. Implementation order and validation

### 8.1 Recommended order

| Priority | Change | Main files |
|---|---|---|
| **1** | Replace strict all-green expansion with class-/workload-aware burst control and useful-throughput feedback | `throughput.py`, `capacity.py`, `engine.py` |
| **2** | Add end-to-end deadlines and bounded backfilling; remove unnecessary global head-of-line blocking | `admission.py`, `policy.py` |
| **3** | Propagate authenticated customer identity and implement actual tenant fairness | `tenancy/callback.py`, `server.py`, `admission.py` |
| **4** | Improve constrained/media/retained-reasoning token accounting | `inspection.py`, `tokens.py`, `workload.py` |
| **5** | Make requested versus granted output limits explicit and review the 1,536/4,096 class caps | `classification.py`, `clamping.py`, `tenancy/policy.py` |
| **6** | Align overload semantics, retry ownership, cancellation, and stream-completion reporting | `server.py`, proxy/portal integration |
| **7** | Persist controller state, blocking reason, queue age, token-count mode, configuration epoch, and stage timings per request | `capture.py`, `metrics.py` |

### 8.2 Validation workloads

Validate with workload-matched bursts containing:

- Cold and warm long contexts.
- Short interactive requests arriving while long generations are active.
- Long reasoning generations.
- Tools and media.
- Cache hits and misses.
- Cancellation, deadline expiry, and recovery after the burst.

Use representative concurrency points to locate the useful-throughput/latency tradeoff. Existing tiny-request canaries do not establish sustained performance on expensive customer workloads.

### 8.3 Acceptance measurements

Compare:

- Correct, useful completions within deadline.
- Terminal failures by layer and reason.
- End-to-end TTFT and time to first answer content.
- ITL and full completion duration.
- Queue-age distribution and drain time.
- Tenant and workload-class fairness.
- Actual cached tokens and prefill work.
- Output completeness, tool-call correctness, and truncation.
- Memory-reservation conservation and preemptions.
- Successful recovery to normal latency after burst demand stops.

## Conclusion

The system has progressed from rigid count limits to dynamic concurrency, but the controller still optimizes narrow latency thresholds without enough knowledge of request cost, queue deadlines, or customer fairness.

The next improvement should coordinate those decisions. That is how the service can absorb bursts, accept a controlled temporary TTFT increase, and recover afterward without hiding failures behind low engine occupancy.

---

## 9. Source map and analysis artifacts

### Serving source

| Area | Source |
|---|---|
| HTTP request lifecycle and error mapping | [`redesign/gateway/server.py`](../redesign/gateway/server.py) |
| Dynamic throughput controller | [`redesign/gateway/throughput.py`](../redesign/gateway/throughput.py) |
| Controller sampling/freshness and legacy policy | [`redesign/gateway/capacity.py`](../redesign/gateway/capacity.py) |
| Admission queue and aging | [`redesign/gateway/admission.py`](../redesign/gateway/admission.py) |
| Policy ordering and release | [`redesign/gateway/policy.py`](../redesign/gateway/policy.py) |
| Shared execution slots and circuit breaker | [`redesign/gateway/backpressure.py`](../redesign/gateway/backpressure.py) |
| KV/output workload reservations | [`redesign/gateway/workload.py`](../redesign/gateway/workload.py) |
| Token inspection and fallback reservation | [`redesign/gateway/inspection.py`](../redesign/gateway/inspection.py) |
| Heuristic token counting | [`redesign/gateway/tokens.py`](../redesign/gateway/tokens.py) |
| Classification and output ceilings | [`redesign/gateway/classification.py`](../redesign/gateway/classification.py) |
| Context/output clamping | [`redesign/gateway/clamping.py`](../redesign/gateway/clamping.py) |
| Engine metrics and streaming transport | [`redesign/gateway/engine.py`](../redesign/gateway/engine.py) |
| Tenancy normalization and cache hooks | [`redesign/tenancy/callback.py`](../redesign/tenancy/callback.py) |
| First-hop classification/clamping | [`redesign/tenancy/policy.py`](../redesign/tenancy/policy.py) |
| Exact-response caching | [`redesign/tenancy/cache_policy.py`](../redesign/tenancy/cache_policy.py) |

### Analysis artifacts

The following local artifacts contain aggregate evidence and policy reproductions:

- `/tmp/opencode/k3-current-logic-export-20260922.json`
- `/tmp/opencode/k3-current-logic-live-20260922.json`
- `/tmp/opencode/k3-current-policy-mechanisms-20260922.json`

Analysis scripts:

- `/tmp/opencode/k3_current_logic_audit_20260922.py`
- `/tmp/opencode/k3_policy_mechanism_review_20260922.py`

The artifacts contain aggregate counts, request shapes, and policy reasons rather than prompt/response content or credentials. Local temporary artifacts should be retained with the investigation if long-term reproducibility is required.
