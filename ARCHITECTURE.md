# Architecture — Kimi-K3 on 8× MI355X

**Status:** live deployment as of 22 September 2026, after the throughput-admission change that set the gateway’s resting concurrency to 32.

This document describes the system that is serving `https://api.cflowx.in/v1`. It is the request path, the settings that are actually loaded, and the methods those settings implement.

Two older documents are not this deployment:

- `docs/ARCHITECTURE.md` describes the retired nginx customer-key allowlist and Authorization swap.
- `docs/SYSTEM-DESIGN.md` and `docs/K3-DEPLOYMENT.md` record earlier measurements. Several of their live numbers (expert parallel on, `max-num-seqs` 427 or 512, prefill budget 8192 or 16384, stock vLLM image) are not what the engine is running now.

Where a number below is a measurement, the date and the workload are named. A setting and a measurement are not the same thing.

---

## 1. What this system is

One bare-metal node serves **Moonshot Kimi-K3** as an OpenAI-compatible API.

| Fact | Value |
|---|---|
| Model | `moonshotai/Kimi-K3`, snapshot `f831ab66814297da540d832a5235f8e904f29d06` |
| Shape | about 2.8 trillion parameters total, about 104 billion active, mixture-of-experts |
| Weights | MXFP4, about 1.5 TB on disk, about 187 GB per GPU once loaded |
| Layers | 93. **24** are full attention (MLA). **69** are KDA, a linear / Mamba-like state whose size does not grow with prompt length |
| Public name | `FW-Kimi-K3`. Aliases `kimi-k3` and `moonshotai/Kimi-K3` are accepted so existing clients do not 404 |
| Context window | 262,144 tokens, input plus generated output together |
| Replica count | **one**. A second copy of the weights does not fit |

Customers do not talk to the GPU process. They talk to a portal (NewAPI, channel 285), which talks to this host. Four local processes then handle the request. Each one owns a different decision.

```text
client
  → NewAPI portal (off this box)
  → nginx            :443   TLS, connection caps, usage log
  → LiteLLM          :4000  who the caller is, spend, exact-response replay
  → gateway          :8002  how many requests may run, memory reservations, priority
  → vLLM engine      :8001  batching, prefix cache, token generation
```

Nothing in that chain is allowed to do another hop’s job.

| Hop | Owns | Does not own |
|---|---|---|
| nginx | TLS, “is there a Bearer token”, how many connections one IP may hold, the usage log | token counts, customer budgets, GPU scheduling |
| LiteLLM | virtual keys, spend, response-cache replay, worker processes | how many sequences the GPU may run |
| gateway | execution slots, the waiting room, KV reservations, traffic class, priority | the customer database, the CUDA/HIP batch |
| vLLM | which requests share a GPU step, the prefix cache, KV and KDA allocation, preemption | who the customer is |

---

## 2. Hardware and why the shape is fixed

| Item | Value |
|---|---|
| GPUs | 8× AMD Instinct MI355X VF, `gfx950`, about 288 GiB each, about 2,304 GB total |
| CPU / RAM | 192 vCPU, about 2,015 GiB |
| Boot disk | 2 TB. `/scratch` is a directory on this disk, not a separate volume, so a droplet snapshot carries the weights |
| Engine image | `johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8` |
| Image selected | 20 September 2026. The plain optimized profile. DSpark speculative decoding is on the image and **not** enabled |

Tensor parallel is 8 because there are 8 GPUs and the weights only fit when split across all of them. Expert parallel is **off** (`no-enable-expert-parallel: true` in the active engine profile). The AMD recipe that qualified this image omits expert parallel. The older stock image had it on and measured a decode speedup; that result does not transfer to this image by editing one flag.

### 2.1 Why there is no second replica

Kimi-K3’s weights occupy about 65% of every card before one token of cache exists. A second replica, a separate prefill worker, or a rolling restart onto a spare copy would need another 1.5 TB. The node has 2.3 TB. Those designs are impossible on this hardware. A restart of the engine is several minutes of downtime. The gateway and LiteLLM can restart without reloading weights.

### 2.2 Why the KV cache is smaller than the cards suggest

MLA stores each token’s KV as one latent vector, effectively one KV head. Tensor parallel shards KV by head. With one head and eight ranks, each GPU holds a full copy of that cache. The pool the engine reports is about **1.64 million tokens** (measured on this image at `gpu-memory-utilization: 0.88`: 2177 blocks, about 1,639,906 tokens).

That replication is a property of MLA under tensor parallel. It is not fixed by raising a concurrency number. De-duplicating it means a different parallelism mode. SGLang’s hybrid DP-attention was tried on 20 September 2026 and failed bring-up (hybrid Mamba OOM). It is not the live engine.

Host memory is not a second cache. A 64 GB host offload was measured: the GPU wrote blocks out and never read them back (`external_prefix_cache_hits` stayed 0). It is disabled.

FP8 KV would cut MLA bytes per token. On this stack at tensor-parallel 8 the FP8 MLA decode kernel asserts `batch_size=1` and the workers die during profiling. KV dtype stays `auto`, which is BF16.

### 2.3 Why 262,144 is the context window

Clients send a fixed `max_tokens`. vLLM reserves **prompt + max_tokens** against one window. At a 131,072 window and a 128,000 `max_tokens`, only about 3,000 tokens of prompt remained and almost every agent request was rejected. 262,144 leaves room for a long prompt beside a large reservation.

The window is a limit, not a reservation. KV pages are allocated as tokens are computed. One request above 262,144 receives HTTP 400 and the engine stays up. That was verified. `gpu-memory-utilization: 0.96` was not safe: a long prefill killed the engine while KV use was only about 8%, because the driver evicted GPU buffers (`svm_range_evict_svm_bo_worker` in `dmesg`). The live value is **0.88**. Do not raise it to 0.96.

---

## 3. The request, hop by hop

### 3.1 Portal

NewAPI (the remote portal) selects this host as channel 285 and sends the customer’s key. This repository does not contain that portal’s configuration. A limit configured there (requests in flight, TPM, RPM) rejects work before nginx sees it. Those rejections do not appear in this host’s engine metrics.

### 3.2 nginx

File: `/etc/nginx/conf.d/k3.conf`. Server name `api.cflowx.in`. TLS on 443, port 80 redirects to HTTPS. An unmatched host is the default server and is dropped (`ssl_reject_handshake` / `return 444`).

`location /v1/` does four things:

1. Requires `Authorization`. An empty header is **401**.
2. Applies `redesign/deploy/profiles/throughput-edge.inc`: at most **256** concurrent connections per IP and per authorization value. Excess is **503**. There is no per-IP request-rate bucket on this location. The old 20 requests/s zone is still declared in the file and is not used by `/v1/`.
3. Writes `/var/log/k3/usage.log` (`k3usage`: address, status, bytes, request time, time to first upstream byte, class header).
4. Proxies to LiteLLM at `127.0.0.1:4000` with buffering off and a 900 second read timeout. The client’s `Authorization` header is forwarded. nginx does not swap it for an upstream key.

`/metrics` and `/health` on the public port are loopback-only and proxy to the gateway. The operator pages `/ops/` and `/api/state` proxy to the dashboard on `127.0.0.1:8080`.

vLLM itself only authenticates some path prefixes. Routes such as `/invocations` would run inference with no key if they were reachable. nginx never forwards them. That allowlist is why the proxy exists.

### 3.3 LiteLLM

Unit `k3-litellm.service`. Listens on `127.0.0.1:4000`. **32** worker processes (`K3_LITELLM_WORKERS`). Config is rendered to `/scratch/deploy-state/litellm.yaml`.

LiteLLM is the tenancy hop:

- Virtual keys, spend, and key-level parallel or TPM caps, when a key has them set in LiteLLM’s database.
- A Redis exact-response cache, TTL **300 seconds**, revision `amd-5f3007-base-v1`. This replays an identical completion. It does not keep GPU prefix state. Tool and multimodal calls are not served from it.
- The callback `redesign.tenancy.callback` runs `normalize_payload` and `TenancyPolicy` before the request is forwarded to the gateway at `127.0.0.1:8002`.

`normalize_payload` (`redesign/gateway/media.py`) is the prefix-safety pass:

- Drops `cache_salt`, `prompt_cache_key`, and `kv_cache_salt`. A per-user salt makes an identical prompt miss the GPU cache on purpose. One portal key serving many end users must share prefixes.
- Sorts the `tools` and `functions` arrays by name (`_canonical_tools`). K3 renders tool schemas **before** the conversation and keeps array order. `write, read, run` and `read, run, write` are different prefixes. Duplicate or unnamed tools are left in the caller’s order.
- Rewrites thinking controls onto `chat_template_kwargs`. K3 accepts `low`, `high`, and `max`. `medium` is mapped to `high` because this build returns HTTP 500 on any other effort.
- Normalizes images and at most one video into a bounded PNG/frame form so media bytes do not change between hops and blow the prefix.

`TenancyPolicy` classifies the request and clamps `max_tokens` / `max_completion_tokens` to the class ceiling. When the workload guard is on, the accurate context-window check is deferred to the gateway, which counts tokens with the engine’s `/tokenize` endpoint. A prompt that cannot fit is **400**, not 500, so the portal does not treat it as an outage and retry it.

LiteLLM has **no** off-box model configured. `K3_OFFBOX_URL` is unset. `P1-short-chat` is marked `served_off_box` in code and stays on this GPU until a URL exists.

### 3.4 Gateway

Unit `k3-gateway.service`. Listens on `127.0.0.1:8002`. This is the admission controller. Code is `redesign/gateway/`.

On each request it:

1. Normalizes the payload again.
2. Counts tokens. Ordinary chat is counted with the engine tokenizer, including reasoning fields the raw `/tokenize` schema would drop. Multimodal and other shapes it cannot count reserve a full context window and are labeled conservative.
3. Classifies (section 4).
4. Clamps output to the class ceiling. `max_completion_tokens` wins over `max_tokens` when both are set.
5. Asks the throughput controller how many executions are allowed right now.
6. Either starts the request, parks it in the waiting room, or refuses it.
7. If it starts, reserves a KV charge, stamps `priority`, and proxies to `127.0.0.1:8001`.
8. Releases the slot and the reservation when the response finishes or the client disconnects.

Loopback diagnostics:

- `GET /diagnostics/capacity` — controller state, live execution limit, KV, running, waiting, token gap, first-token time, prefill time.
- `GET /diagnostics/admission` — recent admission outcomes.
- `GET /diagnostics/prefix` — counts and fingerprints, never prompt text.

Prometheus metrics are on the gateway’s `/metrics`. Names that matter for this design include `k3_gateway_capacity_execution_limit`, `k3_gateway_capacity_starts_paused`, `k3_gateway_workload_reserved_tokens`, and `k3_gateway_workload_throughput_first`.

### 3.5 Engine

Unit `k3.service`, overridden by `k3.service.d/60-amd-optimized.conf`. Docker container `k3`, host network, devices `/dev/kfd` and `/dev/dri`.

The process is:

```text
vllm serve <local Kimi-K3 snapshot> --config /trial/config-base.yaml
```

Active files, bind-mounted read-only:

| Role | Path |
|---|---|
| Serve profile | `/scratch/deploy-state/amd-optimized/config-base.yaml` (repo copy: `experiments/2026-09-20-amd-optimized/config-base.yaml`) |
| Kernel environment | `/scratch/deploy-state/amd-optimized/optimized.env` |
| AITER JIT cache | `/scratch/deploy-state/amd-optimized/cache` → `/root/.cache` |

`/scratch/hf/config.yaml` and the repo-root `config.yaml` are the older stock profile. The running container does not load them.

The engine binds `127.0.0.1:8001` and still has its own `VLLM_API_KEY`. Reaching it without going through nginx requires a local process and that key.

---

## 4. Traffic classes

Assignment is first-match in `redesign/gateway/classification.py`. The same rules run in LiteLLM’s tenancy policy and again in the gateway.

| Order | Class | Matches when | Priority | Output ceiling | First-token target |
|---|---|---|---|---|---|
| 1 | `P3-batch` | explicit batch hint | batch | 32,768 | none |
| 2 | `P2-agentic` | any tools or any image | long-context | 1,536 | 15 s |
| 3 | `P0-interactive` | prompt ≤ 8,192 and requested output ≤ 512 | interactive | 512 | 1 s |
| 4 | `P1-short-chat` | prompt ≤ 8,192 and no tools | short-chat | 2,048 | 3 s |
| 5 | `P2-medium-context` | prompt ≤ 32,768 | long-context | 1,536 | 15 s |
| 6 | `P2-long-context` | everything else | long-context | 1,536 | 60 s |

`P0-interactive` is never shed by the circuit breaker. `P2-*` and `P3-batch` are sheddable. `P1-short-chat` would leave the box if an off-box URL were configured. It is not.

The output ceiling is a clamp. A client that asks for 128,000 tokens on an agentic call is granted 1,536. That grant is what the gateway is supposed to reserve. See section 6.3 for the case where the full client grant is reserved instead.

Priority is sent to vLLM because the engine profile sets `scheduling-policy: priority`. A higher-priority arrival can preempt a lower-priority running request. Batch and distill work is the work that should lose that contest. Customer interactive work should not.

---

## 5. Engine methods

These are the methods the GPU process is actually using. Each one is a setting in `config-base.yaml` or `optimized.env`.

### 5.1 Parallelism and memory

| Setting | Live value | Method |
|---|---|---|
| `tensor-parallel-size` | 8 | One shard of the weights on each GPU. Collective communication on every layer. |
| `no-enable-expert-parallel` | true | Experts are not spread across GPUs as a second mesh. Required by the qualified AMD profile. |
| `gpu-memory-utilization` | 0.88 | Fraction of each card vLLM may allocate. The rest is workspace for graphs and long-prefill activations. |
| `kv-cache-dtype` | auto (BF16) | KV is not quantized. |
| `max-model-len` | 262144 | Hard context window. |
| `max-num-seqs` | 64 | Hard cap on sequences resident in the engine. The gateway ceiling is the same number. |
| `max-num-batched-tokens` | 4096 | How many prompt tokens one prefill step may chew. A trial at 2048 made mixed workloads 25–36% slower and was reverted. |
| `scheduling-policy` | priority | Honor the gateway’s priority field. |
| `enable-prefix-caching` | true | Reuse KV for an identical token prefix. Load-bearing. See section 7. |
| `enable-prompt-tokens-details` | true | Report cached-token counts upstream. |
| `load-format` | auto | Weights load from the local snapshot. `HF_HUB_OFFLINE=1`. |

`block-size` is not set. This is a hybrid model. vLLM forces the attention page to 768 tokens so it is at least the Mamba page, then pads. Setting 128 does nothing.

### 5.2 Kernels

`optimized.env` selects the AMD AITER path. These variables are read from the environment. They cannot be expressed as vLLM CLI flags.

| Variable | Value | What it selects |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | 1 | AITER kernels. Without this, MXFP4 MoE has no backend and the process refuses to start. |
| `VLLM_ROCM_USE_AITER_FP4BMM` | 1 | FP4 batched GEMM used by the AMD recipe. |
| `AITER_SITUV2_A8W4` | 1 | FP8 activations × MXFP4 weights. |
| `AITER_BF16_FP8_MOE_BOUND` | 0 | FP8 activations at every batch size. **Must stay paired with the line above.** Without it, small batches feed BF16 activations to interleaved A8W4 weights and the model writes fluent nonsense at full speed. |
| `VLLM_ROCM_USE_KIMI_K3_PREROUTE_FP8` | 1 | FP8 pre-route kernel. |
| `VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8` | 1 | FP8 latent-tail kernel. |
| `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION` | 1 | Fused latent MoE tail. |
| `KIMI_K3_*_FP8_WEIGHT_CACHE_MODIFIER` | 2 | Weight-cache layout modifiers from the AMD recipe. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | 0 | Kept beside this recipe. `torch.compile` does not apply to this model class. |
| `AITER_JIT_DIR` | `/root/.cache/aiter-5f3007` | Reuse the compiled kernels from the qualified run so the next start is not a cold JIT. |
| `HIP_FORCE_DEV_KERNARG`, `HSA_NO_SCRATCH_RECLAIM`, `HSA_ENABLE_IPC_MODE_LEGACY` | 1 | AMD runtime flags from the same recipe. |

FP8 in those names is compute precision. It is not an FP8 KV cache.

DSpark (speculative draft model `Inferact/Kimi-K3-DSpark`) was measured on this image and left off. At one request it was about 21% faster. At 16 concurrent requests the gain was about 3%, tail latency was worse, and the KV pool shrank about 24%. The live profile does not load the draft model.

### 5.3 What the historical 2.5 million TPM run actually was

On 18 September 2026 the stock image, with expert parallel on, `max-num-seqs` 512, prefill budget 8192, and `gpu-memory-utilization` 0.92, sustained about **2.50 million tokens per minute** on a synthetic workload: 12,288 input tokens, 512 output tokens, about 85% of the input a shared prefix, offered at 3.26 requests/s. Steady state absorbed that with queue depth 0. The run average including ramp-up was about 2.16 million.

Of those tokens, **96% were prompt tokens**. Generated tokens were about 1,440 per second, roughly 86,000 per minute. The prefix cache hit rate in that run was 81%. KV use fell to about 49% because the shared prefix was stored once.

That number is not a property of the cards. It is a property of that workload on that image. Live customer traffic is long, only partly shared, and often agentic. A later production window on the stock image saw the hit rate at 6–13%, KV near full, and throughput in the hundreds of thousands of TPM. The AMD image’s controlled warm test at 16 streams produced about **384 output tokens per second** when every prompt was already cached, and about **8 tokens per second per stream** once two cold long prompts were mixed in.

2 million stable TPM for real clients therefore means: enough requests actually finish, their shared prefixes hit, and cold prefills do not stall the ones that are decoding. It does not mean a flag called “2 million”.

---

## 6. Admission methods

The gateway’s throughput-first mode is on (`K3_THROUGHPUT_FIRST=1`). Installed from `redesign/deploy/profiles/throughput-admission.conf` as `k3-gateway.service.d/95-throughput-admission.conf`. The gateway process must be restarted to read a change. The engine does not.

### 6.1 The three concurrency numbers

These are different limits. Only the live execution limit is enforced at a given moment.

| Setting | Live value | Meaning |
|---|---|---|
| `K3_ADAPTIVE_INITIAL_EXECUTION` | **32** | Resting concurrency. The controller starts here and returns here when the waiting room is empty and the machine is idle. |
| `K3_ADAPTIVE_MIN_EXECUTION` | **16** | Floor. Under pressure the cap is multiplied by 0.8 every 2 seconds and is not allowed below this. |
| `K3_ADAPTIVE_BORROW_MAX` and engine `max-num-seqs` | **64** | Ceiling. The controller cannot admit a 65th sequence. The engine would refuse it anyway. |
| Live `execution_limit` | moves between 16 and 64 | The number of requests allowed to be inside the engine right now. |

Changed on 22 September 2026 at about 18:11 UTC, from a resting value of 8 and a floor of 8. Before that change the gateway was refusing work with `shared execution capacity: 8` while the KV cache was under 20%.

The controller (`ThroughputCapacityController` in `redesign/gateway/throughput.py`) samples the engine every **500 ms**. It never cancels a request that is already running. It only decides whether a new one may start.

| Signal | Threshold | Effect |
|---|---|---|
| KV cache | `K3_ADAPTIVE_BORROW_KV` **0.88** | At or above 88%, the cap starts shrinking. |
| KV cache | critical **0.97** (hardcoded) | New starts pause immediately. |
| Token gap (ITL) | `K3_ADAPTIVE_BORROW_ITL` **0.12** s | Above 120 ms while any request is active, three samples in a row shrink the cap. |
| Token gap | `K3_ADAPTIVE_GREEN_ITL` **0.08** s | The cap may grow only while the gap stays under 80 ms. |
| Engine queue | more than **2** waiters (`tolerated_waiters`) | Treated as pressure. Above that, new starts pause so the engine can drain. |
| Preemption | any | Pause new starts. |
| Growth step | **+4** slots, at most once per **2** seconds | Only while healthy, and only while something is waiting in the gateway queue. |
| Shrink | **×0.8** | After three pressure samples, down toward the floor of 16. |

A request that cannot get a slot waits in the admission queue.

| Queue setting | Value |
|---|---|
| `K3_ADMISSION_WAIT_SECONDS` | 60 |
| `K3_ADMISSION_MAX_WAITERS` | 128 |
| `K3_ADMISSION_MAX_PER_CUSTOMER` | 128 |
| `K3_ADMISSION_QUEUED_TOKENS` | 64 Mi tokens of accounted shape |
| `K3_ADMISSION_QUEUED_BYTES` | 1 GiB of accounted body bytes |
| Fairness age | 5 seconds, then the oldest waiter gets the next cross-lane turn |

When the wait expires the gateway returns **429** with `shared execution capacity: N`, where N is the live cap. LiteLLM forwards that text as a `RateLimitError`. It is not LiteLLM’s `max_parallel_requests` and it is not a NewAPI cap. nginx is not rate-limiting `/v1/`.

### 6.2 First-token time still freezes growth

These four numbers are **not** in the env file. They are defaults on `CapacityLimits` in `redesign/gateway/capacity.py`, and the throughput constructor does not override them.

| Field | Value | What it does today |
|---|---|---|
| `green_ttft` | 2.0 s | Average first-token time above 2 seconds means “not healthy”, so the cap cannot grow. |
| `green_prefill` | 2.0 s | Average prefill above 2 seconds does the same. |
| `stop_ttft` | 5.0 s | Above 5 seconds, and only when the pool already looks busy, counts as pressure and the cap shrinks. |
| `stop_prefill` | 10.0 s | Same for prefill. |

A long prompt takes several seconds to produce a first token even when the cache is mostly empty. That fails `green_ttft`. Shortly after the resting cap was raised to 32, a live sample showed state `warm`, execution limit **16**, 15 running, KV about **30%**, token gap **82 ms**, first-token time **3.4 s**. The 82 ms gap is under the 120 ms shrink line and just over the 80 ms growth line. The 3.4 s first token is what keeps the state from being green, so the cap does not climb back from 16 to 32.

That behavior is still in the code. The env change did not remove it.

### 6.3 Memory reservation

`K3_PROJECTED_KV_LIMIT=0.92`. In throughput mode the budget is **92% of the engine’s measured KV-token capacity**, not the fallback `K3_RESERVED_TOKEN_BUDGET` (1,048,576), which applies only when the engine has not reported a pool.

The charge, in `WorkloadBudget.acquire`, is **full prompt + the full granted output**. Predicted output is computed for telemetry and then not used as the charge while throughput mode is on. A client whose class ceiling is still large, or a conservative reservation of the whole window, can fill this paper budget while the real cache is far under 92%. The refusal reason is `workload budget: reserved_tokens`, returned as **429**.

`K3_WORKLOAD_GUARD=1` remains set. Its old gates are **not** applied in throughput mode:

- large-context slot cap (`K3_LARGE_CONTEXT_MAX=4`)
- long-output token budgets (12,288 base / 49,152 burst)
- fixed class concurrency shares

Those gates were the main source of a 37% error rate on 22 September between 10:27 and 14:27 UTC, while median KV was 2%. They stay in the older drop-in `90-admission-queue.conf`. The `95-` drop-in overrides the variables it sets. The code path `if not throughput_first` is what disables the gates. Both have to stay true.

### 6.4 Circuit breaker

`CircuitBreaker` returns **503** (`REJECT_SHED`) only when the engine is in distress, and only for sheddable priorities. It fails open if metrics cannot be read.

| Condition | Threshold |
|---|---|
| KV alone | above **97%** |
| KV and an engine queue together | KV above **90%** and more than **8** engine waiters |
| Preemptions | above **1 per minute** |

P0 is not shed. A 503 from this breaker means “the engine is actually in trouble”, which is a different statement from the 429 “the gateway’s current slot count is full”.

### 6.5 Status codes a client can see

| Code | Who emits it | Typical cause on this deployment |
|---|---|---|
| 401 | nginx or LiteLLM | missing or unknown key |
| 400 | gateway / tenancy | prompt does not fit the window, or the body is invalid |
| 429 | gateway, forwarded by LiteLLM | live execution cap full after waiting, or `reserved_tokens` |
| 503 | gateway shed, or nginx connection cap | engine distress, or more than 256 connections from one IP or key |
| 500 | LiteLLM / engine | a real server fault. Oversized prompts used to be 500 and were moved to 400 so the portal would not retry them as outages |

---

## 7. Prefix cache

Prefix caching is the method that made the 2.5 million TPM result possible, and it is the method most easily destroyed by a prompt that looks harmless.

vLLM hashes **complete blocks** and only reuses a prefix when the token ids from the start of the prompt match a prefix already in GPU memory. The first difference ends the match. Everything after that difference is recomputed and stored again.

On this hybrid model there is a second condition. KDA keeps a recurrent state, not only attention KV. A matching token prefix is reusable only when a KDA checkpoint exists at that boundary. The prefill chunk size of 4096 produces a natural checkpoint near 3840 tokens. Finer `prefix-match-unit: 128` was measured: it saved a few extra tokens and added a prefill step, and cold first-token time rose **27–31%**. The live profile does not set it.

### 7.1 What this codebase already stabilizes

| Input | What the code does | Why |
|---|---|---|
| Tool array order | sorted by function name in `normalize_payload` | Tools are rendered first. Order is the prefix. |
| `cache_salt` and related keys | stripped | A salt is a different cache namespace for identical text. |
| Image re-encode | fixed PNG compression | A different encoding would be a different prefix. |
| Exact response replay | LiteLLM Redis, 300 s, identical body only | Saves a GPU call when the whole request is a duplicate. Does not create a GPU prefix. |

### 7.2 What the codebase does not rewrite

The date, session id, and tool results inside `messages` are the client’s text. This service does not move them. The measured layout effect, on 21 September 2026:

| Layout | Reused tokens |
|---|---|
| Date before the stable instructions | 0 / 8795 |
| Stable instructions, then the date | 8448 / 8794 (96%) |

The corresponding first-token times were about 0.94 s and 0.21 s. That change belongs in the client template: fixed instructions and reference text first, volatile fields after them, tool results in later messages. Reordering the `messages` array by role, or scanning customer text inside `normalize_payload` to relocate a date, changes the answer and is not part of this design.

### 7.3 Salting and tenancy

Cross-tenant prefix caching leaks whether a guessed prefix is already resident (documented timing side channel). The mitigation is a `cache_salt`. Salting per API key on a portal that shares one key fragments the only cache this box has, and capacity falls off a cliff as the number of salts grows. This deployment drops inbound salts so the portal pool shares one namespace. That is a deliberate choice for a single mutually trusted portal key, not a general multi-tenant isolation design.

---

## 8. Supervision and ports

systemd owns restarts. Docker’s restart policy is not used for the engine. Docker resets its backoff once a container has run 10 seconds, and this model takes minutes to load, so a crash would hot-loop a multi-minute start forever.

| Unit | Bind | Restart cost |
|---|---|---|
| `nginx` | 0.0.0.0:443 and :80 | seconds |
| `k3-litellm` | 127.0.0.1:4000 | seconds. In-flight calls fail |
| `k3-gateway` | 127.0.0.1:8002 | seconds. In-flight streams fail. Engine weights stay loaded |
| `k3` (container `k3`) | 127.0.0.1:8001 | several minutes of weight load. `TimeoutStartSec` is 1800 on the AMD drop-in |
| dashboard | 127.0.0.1:8080, public via `/ops/` | seconds. It is a monitor and does not have to stop when the engine stops |

Engine restart policy: 30 s, then 60, 120, 240, 480, then stop after 5 failures in 30 minutes.

Other local pieces, present and not on the request path: LiteLLM’s Postgres and Redis, an alerts scraper, a distill unit that is not serving customers, and the AMD metrics exporter on 127.0.0.1:5000. `gfx_activity` reads ~100% even when idle on these virtual functions. It is not a utilization signal. Power, clocks, and used VRAM are.

---

## 9. Methods that were tried and are not in the live path

Kept here so a later change does not “turn them on” as if they were unfinished.

| Method | Result | Live state |
|---|---|---|
| DSpark speculative decoding | +21% at 1 request, about +3% at 16, worse tail, −24% KV | off |
| Prefill budget 2048 | mixed completion 25–36% slower | 4096 |
| `prefix-match-unit: 128` | +27–31% cold first-token time | unset |
| FP8 KV cache | workers die at batch > 1 under TP=8 | BF16 / auto |
| Host KV offload | writes, never reads | off |
| `gpu-memory-utilization` 0.96 | engine death mid-prefill, KV was not full | 0.88 |
| Expert parallel on this image | not part of the qualified profile | off |
| SGLang DP-attention (KV de-duplication) | hybrid Mamba OOM at bring-up | not deployed |
| Class shares, long-output slot caps, 20 r/s nginx bucket | rejected traffic while KV was ~2% | disabled on `/v1/` and in throughput mode |
| Second replica, prefill/decode split | weights do not fit twice | impossible on this node |
| `VLLM_BATCH_INVARIANT` | silently half-applies on ROCm and costs a large fraction of throughput where it does work | unset |

---

## 10. How a token becomes TPM

TPM is tokens that **finish**, per minute, prompt tokens and generated tokens together.

A request that waits 60 seconds and returns 429 contributes zero. A request whose prefix hits contributes its prompt tokens cheaply and then spends time on the unique tail and the output. A request whose prefix misses occupies KV and a prefill slot for the whole prompt.

The levers, in the order they bind on this deployment:

1. **Admission.** If the live execution limit is 16 while KV is 30%, the GPUs are not the ceiling. The controller is. The resting value is 32 and the ceiling is 64. First-token time above 2 seconds still stops the climb back up (section 6.2).
2. **Prefix identity.** Stable instructions first, tools in a stable order, no per-user salt. This is the difference between storing a long prompt once and storing it on every request.
3. **Prefill versus decode.** One batch budget of 4096 tokens is shared. Two cold long prompts were measured to cut warm decode from about 32 tokens/s per stream to about 8. There is no separate cap on how many cold prefills may run at once. Adding one, of 1 or 2, is the next GPU-side method. It has to admit the request into a queue, not reject it while the machine is idle.
4. **Paper KV versus real KV.** Reservations use the full granted output. Real generation is often far shorter. That refusal throws away TPM the cards had room to produce.
5. **Sequence ceiling 64.** Enough for 2 million TPM only when prefixes hit and requests finish quickly. The 2.5 million run needed on the order of 400 concurrent sequences because each 12k-token request took about 110 seconds. Raising `max-num-seqs` and `K3_ADAPTIVE_BORROW_MAX` together is useful only after the live cap actually stays at 32 and the cache is not near full. The old 512 over-admitted, filled KV, and preempted finished prefill.
6. **KV pool size.** `gpu-memory-utilization` 0.88 is the qualified value. 0.92 was safe on the older image and grew the pool. 0.96 crashed this hardware. Any move needs a concurrent long-prompt test that watches `dmesg` for amdgpu eviction, and an engine restart.
7. **A different engine that does not replicate MLA KV eight times.** That is the only structural capacity increase left. It is a migration, not a setting, and the one attempt failed.

---

## 11. File map for the live path

| Concern | File |
|---|---|
| Public edge | `/etc/nginx/conf.d/k3.conf` and `redesign/deploy/profiles/throughput-edge.inc` |
| Gateway env | `redesign/deploy/profiles/throughput-admission.conf`, installed as `/etc/systemd/system/k3-gateway.service.d/95-throughput-admission.conf` |
| Admission controller | `redesign/gateway/throughput.py`, `admission.py`, `policy.py`, `workload.py` |
| Hardcoded first-token gates | `redesign/gateway/capacity.py` (`green_ttft`, `green_prefill`, `stop_ttft`, `stop_prefill`) |
| Classes | `redesign/gateway/classification.py` |
| Payload normalize | `redesign/gateway/media.py`, called from `redesign/gateway/server.py` and the tenancy callback |
| Tenancy | `redesign/tenancy/callback.py`, `policy.py`, `cache_policy.py` |
| Engine profile | `experiments/2026-09-20-amd-optimized/config-base.yaml`, live at `/scratch/deploy-state/amd-optimized/config-base.yaml` |
| Kernel env | `experiments/2026-09-20-amd-optimized/optimized.env`, live beside the profile |
| Engine unit override | `/etc/systemd/system/k3.service.d/60-amd-optimized.conf` |
| Measurements | `docs/K3-DEPLOYMENT.md` (18 Sep stock benchmark), `experiments/2026-09-20-amd-optimized/RESULTS.md`, `experiments/2026-09-21-mixed-prefill/RESULTS.md`, `experiments/2026-09-21-prefix-cache/RESULTS.md`, `docs/THROUGHPUT-ADMISSION-2026-09-22.md`, `docs/TRAFFIC-BURST-ANALYSIS-2026-09-22.md` |

---

## 12. Invariants

These are the rules a change has to keep, because each one was paid for once already.

1. One replica. Do not plan a second local copy of Kimi-K3 on this node.
2. A YAML boolean `false` is dropped by vLLM’s config parser and becomes the flag’s default. Disable a boolean with the positive negative form, as `no-enable-expert-parallel: true`.
3. `AITER_SITUV2_A8W4=1` without `AITER_BF16_FP8_MOE_BOUND=0` produces fluent wrong answers. Run the correctness gate after any AITER change.
4. Do not set `gpu-memory-utilization` to 0.96.
5. Do not set an FP8 KV cache on this tensor-parallel-8 image.
6. Do not set `prefix-match-unit: 128` to chase a higher hit rate.
7. Do not re-enable host KV offload.
8. Do not put DSpark on the concurrent production profile.
9. Do not salt the prefix cache per end user of the shared portal key.
10. Do not restore per-class concurrency shares or the nginx 20 r/s IP bucket on `/v1/`. They reject work while the GPUs are idle.
11. An oversized prompt is 400. A full execution cap is 429. Engine distress is 503. Mixing those up makes the portal retry the wrong failures.
12. Editing repo-root `config.yaml` does not change the running engine. The container loads `/trial/config-base.yaml`.
13. Gateway env changes apply on `k3-gateway` restart. They do not apply to a process that was already running.
