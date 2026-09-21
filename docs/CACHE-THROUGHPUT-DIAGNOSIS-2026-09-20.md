# Cache and per-request throughput investigation — 2026-09-20

## Executive finding

There are three distinct problems:

1. **Reproduced hybrid prefix reuse limitation:** an identical shared prefix can
   miss until a KDA/Mamba state checkpoint is materialized at a reusable boundary.
   Large prefill chunks can skip those intermediate checkpoints.
2. **Measured slow engine decoding under mixed production traffic:** the engine
   itself reaches roughly 5–6 tokens/s per active request in the observed window.
   Excellent prefix reuse does not prevent slow subsequent decoding.
3. **Reproduced protocol routing defect:** LiteLLM sends native Responses calls
   to a Chat-only gateway. Its 404s trigger deployment cooldowns, with subsequent
   requests receiving router 429s. This can encourage upstream channel failover.

No periodic GPU cache flush was found. The remote NewAPI selection settings and
the Azure comparison deployment were not available for inspection.

## Scope and effective configuration

Inspected repository changes and serving-path modules, deployed gateway/tenancy
code, nginx, systemd/container settings, benchmark records, current vLLM cache and
scheduler source, runtime metrics/logs, GPU state, and aggregated LiteLLM DB logs.
No production prompts or credentials are included in this report.

Effective engine config is `/scratch/hf/config.yaml`, not repository `config.yaml`:

| Setting | Running deployment |
|---|---|
| Hardware / parallelism | 8 × MI355X, TP8, EP enabled |
| Context | 262144 |
| Sequence ceiling | 160; repository config says 96 |
| Gateway admission ceiling | 160 |
| Batch token budget / long-prefill threshold | 16384 / 16384 |
| Prefix caching / matching unit | enabled / 128 |
| Physical attention block / recurrent cache mode | 768 / align |
| KV dtype / watermark | auto (BF16) / 0.02 |
| Stream interval | 4 |
| Async scheduling | enabled |
| Speculative decoding / host KV offload | absent / disabled |
| LiteLLM workers / response-cache TTL | 32 / 300 seconds |

AITER and A8W4 controls are present and recognized by this installed build.
GPU observation showed 94–100% busy, junction temperatures 58–62°C, and clocks
mostly 2400 MHz. This snapshot does not indicate obvious thermal throttling;
it does not identify which GPU kernels dominate execution.

## 1. Cache lifecycle and the reproduced checkpoint effect

Current vLLM `v1/core/block_pool.py:647–740` reallocates free blocks in eviction
order. Cached, unreferenced blocks remain reusable until selected for eviction;
they are not invalidated by a wall-clock TTL. `reset_prefix_cache` logs a reset
when called. The current engine log contained **zero** successful/failed reset
messages or reset endpoint references, and no engine errors. The engine had run
since 17:59:06 UTC. Gateway/LiteLLM restarted at 20:16:43, independently of it.

**Important metric distinction:** `BlockPool.get_usage()` counts blocks excluded
from the free list. The free list includes idle cached blocks. Consequently a
low `kv_cache_usage_perc`, or zero preemptions, does **not** establish that idle
prefix entries cannot be evicted. The observed drop to 0% when active requests
disappeared is not proof that the prefix cache was flushed.

The **300-second TTL** applies to LiteLLM's exact response replay. It does not
control GPU KV lifetime. Extending it would not repair agentic prefix reuse;
tools/media/stateful requests intentionally bypass exact response replay.

### Actual shared-prefix probe

Three sequential engine requests used a fresh run marker, the same repeated
text prefix, a changing branch number, and the same repeated suffix text.
Each prompt was 7675 tokens, with one requested output token.

| Request | Cached tokens | Elapsed |
|---|---:|---:|
| First branch | 0 | 2.675 s |
| Second branch, same prefix | 0 | 1.271 s |
| Third branch, same prefix | 3840 | 0.923 s |

The source explains this behavior:

- `single_type_kv_cache_manager.py`, `MambaManager.allocate_new_blocks`, allocates
  the running state at the end of a chunk and leaves skipped positions null.
- `MambaManager.find_longest_cache_hit` needs a cached recurrent state, not only
  attention KV for the matching token prefix.
- `kv_cache_coordinator.py:810–817` detects when attention has a longer matching
  prefix than the recurrent-state group.
- `sched/scheduler.py:357–432` introduces a stop at that shared-prefix junction
  during recomputation, making it reusable by later requests.

Thus `prefix-match-unit: 128` is a lookup granularity, **not a promise to retain
every 128-token recurrent state**. The optional sparse-retention environment
variable was unset; the observed effect does not require a TTL or an explicit
sparse-retention override.

Changing tools/system text can independently destroy a prefix. Tool ordering is
already canonicalized in deployed code; tool contents and tool-set changes still
matter. Dense checkpointing everywhere would consume more state memory and
cannot be assumed to be a free optimization.

## 2. Measured throughput and prefill interference

A 35.1-second production window ending at 20:24:16 UTC showed:

- 4201.7 generated tokens/minute = **70.0 generated tokens/s aggregate**;
- mean **12.43 running**, no sampled queue, mean active KV usage 27%;
- zero preemptions;
- histogram-estimated median inter-token latency **177 ms** (~5.65 tokens/s);
- window prefix-token hit ratio **19.02%** (small number of new requests).

The latency histogram contained 2453 observations. Completed-request shape and
TTFT statistics had only 2–3 observations and are too small for strong tail claims.
Dividing aggregate output by average active requests gives 5.63 tokens/s, a rough
cross-check rather than an individual request measurement.

A separate 20-minute DB sample of 154 successful non-replayed requests measured
median **5.41 output tokens/s end-to-end** and **6.40 output tokens/s after the
recorded completion start**; mean prompt 35630 tokens, mean output 225 tokens.
Another overlapping sample had cached-token details on all 166 non-replayed
successes, 44 with zero cached tokens, and 47.87% aggregate prompt-token reuse.
These rolling samples are not identical cohorts.

Engine logs show decode collapses during prefill bursts, for example:

| Engine log time | Prompt tokens/s | Generated tokens/s | Running |
|---|---:|---:|---:|
| 20:23:25 | 19007.5 | 6.2 | 14 |
| 20:23:35 | 3699.5 | 80.5 | 15 |
| 20:23:55 | 0 | 71.0 | 12 |

The scheduler allows an in-progress prefill up to the entire 16384-token batch
budget; running requests share execution steps. The threshold equal to the batch
budget provides little per-request isolation. This makes the optimization a
strong candidate for latency regression, but its isolated effect still requires
a matched A/B. Long-context decode also remains expensive after a prefix hit.

### Cache-hit versus generation probe

Four sequential requests, 2102 input / 64 output tokens, amid changing production
load (these are observations, **not a controlled gateway overhead A/B**):

| Path / request | Cached | TTFT | E2E output tokens/s |
|---|---:|---:|---:|
| Direct engine, cold | 0 | 0.774 s | 13.666 |
| Direct engine, identical | 2048 | 0.329 s | 11.406 |
| Gateway, identical after 20-second pause | 2048 | 0.284 s | 5.851 |
| Direct engine, new short suffix | 2048 | 0.345 s | 9.192 |

The gateway probe had a maximum chunk gap of 4.003 seconds despite 97.4% prompt
reuse. Good cache reuse improved startup but did not guarantee decode speed.
Nginx buffering is off and the gateway uses `read1()` for streaming. Interval 4
adds delivery batching; changing it to 1 improves granularity, not 4× compute.
Some synthetic completions spent their entire 64-token budget on reasoning.
Visible answer speed therefore also depends on reasoning settings and accounting.

## 3. Responses 404 → cooldown → apparent channel instability

Deployed `redesign/gateway/server.py` accepts Chat/Completions paths, not
`/v1/responses`. Generated LiteLLM models use the OpenAI provider, whose native
Responses path is selected unless conversion is requested.

The observed rolling log sample contained:

- **7 `aresponses` NotFoundError failures**;
- **20 RouterRateLimitError failures** mentioning cooldown/no deployments:
  17 Responses calls and 3 Chat calls;
- additional input/context errors and connection failures around the earlier
  gateway/LiteLLM restart. Those are separate from steady-state decode speed.

Installed `litellm/router_utils/cooldown_handlers.py:318–403` can cooldown a
single deployment on a non-retryable 404. This is an availability defect, not
evidence of GPU saturation. Remote NewAPI failover remains a plausible downstream
effect; its actual selection events were not inspected.

**Prepared repository fix:** `use_chat_completions_api: true` in each local
model's `litellm_params` in `redesign/tenancy/render_config.py`. A local Chat-only
stub reproduced native `/v1/responses` → 404, then the bridge used
`/v1/chat/completions` successfully with the requested 16-token limit and cached
usage preserved. Added `redesign/tests/test_responses_bridge.py` covers all four
aliases, streaming terminal usage, and ordinary Chat routing.

**Validation:** 13 bridge/tenancy tests and 24 cache-policy tests passed with the
installed LiteLLM. This establishes basic create/stream conversion; full stateful
Responses lifecycle, hosted tools, and remote NewAPI adapters need separate tests.
The fix is in the repository; deployed files/services were not changed here.

## 4. Other optimization/deployment gaps

- Deployed `gateway/media.py` still decodes/re-encodes every image on both local
  hops and has no repository 1568-pixel edge cap. Deployed `tokens.py` also lacks
  the repository's dimension-aware image estimate. This adds avoidable CPU work
  and can underestimate vision load. Deterministic re-encoding alone is not proof
  of per-request cache instability. Roll out the existing changes with a vision
  quality check rather than invent another normalizer.
- Admission is request-count/KV-pressure based, not decode-latency based. It can
  admit far more work than an interactive tokens/s target tolerates even before
  KV fills. Merely reducing 160 to 96 will not help a window with 12 active calls.
- Agentic output is capped at 512, including reasoning. A cap is not faster
  per-token execution and may cause truncation/repeated turns. Audit finish
  reasons before treating reduced output counts as a throughput improvement.
- 32 LiteLLM workers resulted in about 6.4k threads, mostly separate Prisma
  engines. Proxy CPU was low in the snapshot, so this is excess footprint, not
  the established source of 177-ms engine token latency.
- Native host KV offload previously demonstrated no read reuse and is disabled.
  Earlier FP8 experiments had compatibility/correctness problems. Neither is a
  justified immediate fix based on the current evidence.

## Prioritized optimization plan

1. **Deploy the tested Responses bridge** through a drained LiteLLM rollout;
   verify create/stream usage through the actual portal. Engine restart is not
   required for this routing change.
2. **A/B latency-oriented prefill settings.** First test batch budget 8192 against
   16384 with matched traffic/cache conditions; then test 4096 and a per-request
   long-prefill threshold around 3072–4096. Smaller chunks can both reduce decode
   stalls and create more recurrent-state checkpoints. Validate multimodal
   scheduling, queue growth, correctness, and total throughput in each arm.
3. **Test stream interval 1 separately** for smoother token delivery. Measure
   content/reasoning token timing, not SSE chunk count.
4. **Budget the long-prefill workload for a latency target**, not maximum sequence
   count. Bound simultaneous cold long/vision work or route it to separate
   capacity. Adjust admission using measured TTFT/ITL and queue signals; avoid
   arbitrary aggressive rejection that merely causes upstream retries.
5. **Deploy the existing media/estimator improvements**, preserving a stable
   tool/system prefix and checking screenshot quality.
6. **Profile the engine's decode kernels/communication** if the matching
   low-load, warm-prefix test still misses the target. Compare AITER/MLA/MoE
   variants one at a time with correctness gates. Speculation is a separate
   engine-compatibility project, not a cache-setting fix.

The Azure comparison needs the same actual model, context/tools/images,
reasoning settings, output length, cache warmth, concurrency, and speed formula.
The current machine's ~70 aggregate tokens/s is not comparable to another
service's 70 tokens/s for one request. No validated change here establishes a
68–70 tokens/s per-request capability on this deployment.

## Follow-up: current traffic at 21:03 UTC — maximum-thinking default

The follow-up identified a request-control defect in addition to prefill/decode
contention. **Explicit low reasoning effort was not being translated into K3's
native thinking-effort control.**

### Current traffic, not historical benchmark throughput

The 15-minute completed-request sample contained 120 successful, non-replayed
calls. Median prompt was 26850 tokens, mean 40948, p95 129380; median output
395.5 tokens. Median end-to-end output rate was **4.49 tokens/s** and median
post-completion-start rate **5.07 tokens/s**.

The 60.1-second engine window ending 21:03:52 UTC had:

| Measurement | Value |
|---|---:|
| Prompt tokens/minute | 333767.6 |
| Generated tokens/minute | 7325.2 (~122.1 tokens/s aggregate) |
| Mean active requests | 38.08 |
| Mean queued / preemptions | 0 / 0 |
| Mean active KV usage | 57% |
| Window prefix-token reuse | 33.59% |
| Histogram-estimated inter-token latency p50 / p95 | 184 ms / 1.689 s |

Subsequent observations reached 43–48 running requests and up to 10 waiting.
GPUs were 94–97% busy. This is a busy shared engine, despite the absence of
preemption and despite TPM quota not being exhausted.

An overlapping 157-request cohort, grouped by cache reuse:

| Prompt cached fraction | Requests | Median decode tokens/s | Median reported TTFT |
|---|---:|---:|---:|
| >=90% | 44 | 8.04 | 0.931 s |
| >0%, <90% | 64 | 5.79 | 1.941 s |
| Zero/unknown | 49 | 2.87 | 3.935 s |

These are observational cohorts, not matched cache A/Bs. They show that even
very well-cached production calls fall far short of 60–70 tokens/s.
Completed-request samples also omit still-running calls and may underrepresent
the slowest newly started requests.

### Reasoning dominates generated work

In a subsequent 155-request cohort, 140 reported reasoning-token details:

- Total output in those 140 calls: **47895 tokens**.
- Reasoning output: **35851 tokens (74.85%)**.
- **55** calls spent at least 95% of output on reasoning.
- **47** calls generated exactly 512 total tokens with >=500 reasoning tokens.
- Median non-reasoning output among calls with details: **31.5 tokens**.

The 512-token agentic cap includes reasoning. This is strong evidence of a
budget/effort mismatch, not evidence that 512 tokens is generous for this traffic.
Stored response bodies were empty, so actual `finish_reason` and subsequent
client retries could not be established from these DB rows. Do not label all
512-token calls failed without that evidence.

### Exact source of the control mismatch

The loaded model's `tokenization_kimi.py:357–389` defaults to `thinking=True` and
explicitly applies `kwargs.setdefault("thinking_effort", "max")`.
`encoding_k3.py:618–633` renders that native `thinking_effort` instruction near
the beginning of the prompt.

vLLM's `ChatCompletionRequest.build_chat_params()` forwards `reasoning_effort`
and may set `enable_thinking`; it does not rename effort to K3's
`thinking_effort`. The old normalizer mapped values such as medium→high but
left them in the request-root field. It also mapped none/off→low without
setting the native `thinking=False` flag.

Verified using the live `/tokenize` endpoint with a short synthetic message:

| Request control | Prompt tokens | Observation |
|---|---:|---|
| Omitted | 91 | Tokenizer maximum-thinking default |
| Root `reasoning_effort: low` | 91 | Exactly the same token IDs as omitted |
| Native template `thinking_effort: low` | 92 | Correctly changes encoded prompt |
| Native template `thinking: false` | 24 | Removes thinking instruction/prefix |

### Prepared fix and verification

`redesign/gateway/media.py` now binds explicit effort to native
`chat_template_kwargs.thinking_effort`, translates none/off into native thinking
disablement, and handles the `enable_thinking` alias. Explicit native template
settings take precedence; controls in `extra_body` are included. Requests with
no controls retain the model default; changing that default is a separate
quality/latency choice.

Live tokenizer checks proved:

- normalized low produces exactly the native-low token IDs;
- normalized none produces exactly the native-thinking-off token IDs;
- normalized low differs from the old ineffective mapping.

**55 tests passed**, covering thinking controls, media, cache policy, tenancy,
and Responses conversion. Added `redesign/tests/test_thinking_controls.py`.
The implementation is prepared in the repository, not deployed. No measured
post-rollout throughput gain is claimed.

### Revised first actions

1. Deploy the explicit-effort translation fix. For routine tasks, evaluate a
   stable low-effort service profile using
   `chat_template_kwargs={"thinking": true, "thinking_effort": "low"}`.
   For tasks not requiring reasoning, evaluate `{"thinking": false}` instead.
   Keep deep reasoning available where needed; do not silently disable it for
   every caller. Stable per-session effort also avoids repeatedly changing the
   early prompt and losing cache reuse.
2. Pair effort with output budgets. Deep-reasoning agentic calls need room to
   finish; merely increasing the 512 cap increases work and is not a speed fix.
   Record reasoning tokens, non-reasoning tokens, and actual finish reasons.
3. Then run the smaller-prefill-batch and latency-aware admission experiments
   described above. Lowering the sequence cap from 160 to 96 alone would not
   bind the observed 38–48 active requests.
4. Compare the Azure endpoint using the exact model/version and reasoning mode.
   Equal TPM is an admission quota, not equal per-request decode capacity.

Removing unnecessary reasoning can shorten requests and reduce shared load,
but does not directly accelerate every GPU decode step. Likewise, the 74.85%
reasoning share is not a measured 4× speedup opportunity: answer length,
quality, batching efficiency, and load will change. At 38 simultaneous decoding
streams, 60 tokens/s each would require 2280 output tokens/s aggregate; this
sample delivered ~122.1. That arithmetic is a capacity target, not a hardware
replica-count estimate. A 60–70 tokens/s promise needs a matched benchmark and
possibly substantial engine/capacity changes beyond request normalization.
