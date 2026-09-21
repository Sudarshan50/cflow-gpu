# Kimi-K3 logic audit and historical deployment handoff

Evidence date **2026-09-20 UTC**, before the optimized-image rollout.
The deployment details below are historical. For the active image, profiles and
validation results, use [the optimized rollout results](../experiments/2026-09-20-amd-optimized/RESULTS.md).

Repo: `/root/cflow-gpu`
Installed control plane: `/usr/local/lib/k3`
Effective engine config: `/scratch/hf/config.yaml` (NOT repo `config.yaml`)
Public edge: `https://api.cflowx.in/v1`
Dashboard: `:8080` (`/api/state`)
Remote portal (NewAPI / yydsrouter.cc) is off-box. Channel 285 points here.

---

## 0. What this system is

One bare-metal node serving **Moonshot Kimi-K3** (`KimiK3ForConditionalGeneration`,
text `KimiLinearForCausalLM`) to agentic + vision traffic from a remote portal.

- **2.8T MoE MXFP4**, ~1.5 TB weights, **16 of 896 experts** active
- **93 layers**: 24 full attention (MLA), **69 KDA** (linear / Mamba-like state)
- Tokenizer mode `kimi_k3`. Defaults: `thinking=True`, `thinking_effort=max`
- Native thinking efforts: `low | high | max` only. `medium` 500s the engine
- Checkpoint: `moonshotai/Kimi-K3` snapshot `f831ab66814297da540d832a5235f8e904f29d06`
- One replica only. Weights do not fit twice. Prefill/decode disagg is impossible
  (needs ~3 TB, node has 2.3 TB)

### Hardware (live)

| Item | Value |
|---|---|
| GPUs | 8 × AMD Instinct MI355X VF, gfx950, ~288 GiB each |
| CPU / RAM | 192 vCPU, ~2015 GiB |
| Host driver | 6.19.14.31400000, kernel 6.8.0-137-generic |
| Engine image | `vllm/vllm-openai-rocm@sha256:5aa7e626…` (tag `kimi-k3`) |
| vLLM | `0.1.dev19253+g5f76ae224.d20260727` |
| Container ROCm | 7.2.3 |
| PyTorch | 2.11.0+git, HIP 7.2.53211 |

### Hard constraints another AI must not violate

1. **One replica.** Do not propose a second local copy, disagg, or rolling N+1.
2. **YAML `false` is a footgun.** vLLM config parser emits nothing for boolean
   false, so you get the DEFAULT, not disabled. Use `no-enable-*: true`.
3. **Do not re-enable A3 host KV offload** without a proven CPU→GPU read path.
4. **Do not invent `K3_OFFBOX_URL`.** Off-box is armed and empty.
5. **Do not raise `gpu-memory-utilization` above 0.92** without a concurrent
   unique-prompt >186k test. 0.96 crashed (`svm_range_evict_svm_bo_worker`).
6. **Do not treat dashboard Peak TPM as tokens-in-a-minute.** See §10.
7. **Equal TPM vs Azure ≠ equal per-request t/s.** TPM is a quota. This box
   shares one TP8 replica.
8. Repo `config.yaml` / `max-num-seqs: 96` is **not** what the engine is
   running until an engine restart loads it.

---

## 1. Request path (exact)

```
client
  → remote NewAPI portal (yydsrouter.cc, channel 285, affinity rule 3)
  → nginx :443  api.cflowx.in
  → LiteLLM :4000  (32 workers, tenancy callback, Redis response cache)
  → gateway :8002  (classify, clamp, admit, normalize, priority)
  → vLLM :8001     (TP=8, EP on, GPU prefix cache, no host offload, no spec)
```

Nginx also exposes the dashboard at `/` and `/api/`. Inference is only `/v1/*`.

### 1.1 nginx (`/etc/nginx/conf.d/k3.conf`)

- Upstream tenancy: `127.0.0.1:4000` only. **No unauthenticated gateway backup**
  (removed 2026-09-20; a LiteLLM restart used to bypass auth/cache).
- `proxy_buffering off` (streaming TTFT).
- Per-key: `100 r/s burst 200 nodelay`, **96 connections**.
- Per-IP: `20 r/s burst 40`, 96 connections.
- Timeouts 900s at edge. Gateway engine timeout **600s**.
- Usage log `/var/log/k3/usage.log` (status, bytes, latency; no customer name
  after key swap).

### 1.2 LiteLLM (`/scratch/deploy-state/litellm.yaml`)

Models: `FW-Kimi-K3`, `kimi-k3`, `moonshotai/Kimi-K3`, `default` →
`http://127.0.0.1:8002/v1` as `openai/<name>`.

**Live YAML is missing `use_chat_completions_api: true`.** Repo renderer has it.
Native `/v1/responses` therefore 404s on the Chat-only gateway and LiteLLM can
cooldown the only local deployment.

Settings (live):

- callback `redesign.tenancy.callback.tenancy`
- `drop_params: true`
- `enable_caching_on_provider_specific_optional_params: true`
- Redis cache `mode: default_off`, TTL **300s**, `acompletion` only
- `request_timeout` / router timeout **900**
- `num_retries: 0` locally (portal retries separately)
- `always_include_stream_usage: true`
- 32 workers → large Prisma thread footprint; CPU was not the 177ms ITL cause

Key-level limits observed on one key (not proven to be channel 285):
`max_parallel_requests=64`, **3,000,000 TPM**. Other keys unset.

### 1.3 Tenancy callback (`redesign/tenancy/callback.py`) — first hop logic

Order on every chat call:

1. If `stream=true`, default `stream_options.include_usage=true` (explicit false wins).
2. Capture authenticated response-cache context (key scope, revision, mode).
3. Disable SDK cache until finalize (after routing merges defaults).
4. Deep-copy only payload fields, `asyncio.to_thread(normalize_payload)`:
   messages, tools, functions, extra_body, chat_template_kwargs,
   thinking/reasoning effort, cache salts, priority.
5. `TenancyPolicy.apply`: estimate tokens, classify, clamp `max_tokens` /
   `max_completion_tokens` (`max_completion_tokens` wins if both valid ints).
6. Unservable prompt → HTTP **400** (not 500; portal used to retry 500s as outages).
7. Stamp `metadata.k3_class`, `k3_priority`, `k3_offbox`.
8. After routing: `finalize_response_cache` builds the Redis key.

### 1.4 Payload normalize (`redesign/gateway/media.py`)

**Deployed** (`/usr/local/lib/k3/.../media.py`):

- Drop prefix busters: `cache_salt`, `prompt_cache_key`, `kv_cache_salt`, `priority`
- Alias-map thinking strings at request root / extra_body / chat_template_kwargs:
  `medium→high`, `none/off→low`, `xhigh→max`, etc. **Does NOT bind into
  `chat_template_kwargs.thinking_effort`.** K3 tokenizer ignores root
  `reasoning_effort`. Default remains **max thinking**.
- Sort `tools` array by function name (K3 `deep_sort_dict` sorts keys but
  **preserves array order**; rotating 3 tools dropped shared prefix 1142→35).
- Decode every image, re-encode PNG compress=6 (deterministic). **No 1568-px
  edge cap in deployed file.**

**Repo only (not deployed):**

- `_bind_thinking_controls()`: copy explicit effort into
  `chat_template_kwargs.thinking_effort`; `none/off` + no thinking flag →
  `thinking=False`. Native template settings win. `enable_thinking` alias.
- `MAX_IMAGE_EDGE = 1568` (~12k tokens at 192 px/token, encoder cache 16817).

### 1.5 Classification (`redesign/gateway/classification.py`)

First match wins:

| Order | Rule | Class | Priority | max_output | kv share | Off-box? |
|---|---|---|---|---|---|---|
| 1 | `x-k3-batch` + allowlist/loopback | P3-batch | 3 | 32768 | 0.25 | no |
| 2 | tools OR images | **P2-agentic** | 2 | **512** | 0.45 | no |
| 3 | prompt ≤ 8192, no tools | P1-short-chat | 1 | 4096 | 0 | **yes if URL set** |
| 4 | prompt ≤ 32768 | P0-interactive | 0 | 8192 | 0.50 | no |
| 5 | else | P2-long-context | 2 | 16384 | 0.25 | no |

Live portal traffic is almost all **P2-agentic** (tools/screenshots).
512 includes reasoning tokens. With max thinking, many turns hit 512 and stop.

P0 is never shed. Agentic was created so tool loops are sheddable.

### 1.6 Token estimate (`redesign/gateway/tokens.py`)

Used only for classify + clamp, not billing.

- Text: 3.5 chars/token; CJK 1.0
- +4 tokens per message
- **Deployed:** flat **4096 tokens/image**
- **Repo:** dimension-aware PNG IHDR, 192 px/token, floor 4096, cap 16384

Live vision samples were 3764 / 9410 / 13174 multimodal tokens. 1024 used to
mis-class 60k vision as P0. 4096 still under-counts large screenshots.

### 1.7 Clamp (`redesign/gateway/clamping.py`) — D2

Engine reserves `prompt + max_tokens` against `max_model_len` (262144).
Clients often send Foundry-style 128000. Unguarded → 400.

```
available = 262144 - prompt - 256
if available < 64: reject PROMPT_TOO_LONG
else granted = min(requested or class_ceiling, class_ceiling, available)
```

Applied on **both** LiteLLM hop and gateway hop. Gateway also rejects `n` or
`best_of` > 1.

### 1.8 Gateway admit (`backpressure.py` + `policy.py` + `server.py`)

Live: `K3_ADMISSION_CEILING=160`, `K3_SEND_PRIORITY=1`, `K3_ENGINE_URL=:8001`,
timeout 600s.

1. Snapshot vLLM `/metrics` (2s TTL): running, waiting, `kv_cache_usage_perc`,
   preemption rate.
2. Shed P2/P3 if distressed:
   - KV > **0.97**, or
   - KV > **0.90 AND waiting > 8**, or
   - preemptions/min > 1.0
   - P0 never shed
3. ClassBudget: atomic global ceiling **and** `ceiling * class.kv_budget_share`.
   Agentic share 0.45 → 72 slots at ceiling 160; global 160 still binds.
4. If admitted: set `max_tokens`, stamp `priority` int for vLLM, proxy.
5. Stream with `read1()` (not `read(n)` — that invented seconds of TTFT).
6. Client disconnect does **not** cancel the GPU job; slot held until upstream
   returns or 600s. Portal shows many `client_gone`.
7. Off-box only if `K3_OFFBOX_URL` set (empty).

HTTP: shed → 503 + Retry-After 30; class full → 429; oversize → 400; engine
down → 502.

### 1.9 vLLM engine (live `/scratch/hf/config.yaml`)

| Knob | Live | Repo `config.yaml` |
|---|---|---|
| TP | 8 | 8 |
| Expert parallel | **true** | true |
| max-model-len | 262144 | 262144 |
| max-num-seqs | **160** | 96 (needs restart) |
| max-num-batched-tokens | 16384 | 16384 |
| long-prefill-token-threshold | **16384** | 16384 |
| gpu-memory-utilization | 0.92 | 0.92 |
| kv-cache-dtype | auto (BF16) | (omitted, same) |
| watermark | 0.02 | (omitted; live file has it) |
| async-scheduling | true | (omitted; live file has it) |
| prefix caching | true, match unit 128 | true, 128 |
| scheduling-policy | priority | priority |
| stream-interval | **4** | (omitted → default 1) |
| mm-processor-cache-gb | 8 | (omitted) |
| cudagraph | FULL_DECODE_ONLY | FULL_DECODE_ONLY |
| custom_ops | +fused_rms_norm_gated | same |
| speculation | none | none |
| host KV offload | none | none |
| enable-prompt-tokens-details | true | true |

Env (engine PID 1):

```
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_USE_AITER_MOE=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
AITER_SITUV2_A8W4=1
AITER_BF16_FP8_MOE_BOUND=0
AITER_ROCM_ARCH=gfx942;gfx950
HF_HUB_OFFLINE=1
SAFETENSORS_FAST_GPU=1
```

`VLLM_ROCM_USE_AITER_MLA` unset (AITER MLA on).
`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset (dense retention).
`VLLM_USE_V2_MODEL_RUNNER` unset (K3 excluded from ROCm V2 default).

### 1.10 Hybrid KV facts (why capacity knobs mislead)

- Attention block forced to **768** tokens so attention page ≥ Mamba page;
  Mamba page padded +8.68%. Explicit `block-size: 128` is a no-op.
- Shared `BlockPool` across attention + KDA. No separate KDA pool flag.
- KV tokens ~**2,197,339**; max concurrency at 262144 ≈ **8.38×**
- TP0 KV budget ~**58.9 GiB** after ~200 GiB weights/non-torch
- `kv_cache_usage_perc` / `BlockPool.get_usage()` counts blocks **not on the
  free list**. Idle cached blocks sit on the free list, so 0% usage ≠ cache flush.
- No GPU cache TTL. LiteLLM 300s TTL is **exact response replay only**.
- KDA recurrent state is allocated at **chunk end**. Intermediate positions
  can be null. `prefix-match-unit: 128` is lookup granularity, not a promise
  to keep every 128-token state. Large 16k chunks skip checkpoints.
  Reproduced: identical prefix 0 → 0 → 3840 cached tokens.

MLA KV is **replicated** across TP8 (single latent head). The ~8× de-dupe
lever is SGLang DP-attention, which OOM’d on hybrid here. vLLM DCP is 1.

---

## 2. Response cache vs GPU prefix cache (two different things)

| | Redis exact-response | GPU prefix KV |
|---|---|---|
| Layer | LiteLLM | vLLM BlockPool |
| TTL | 300s | none (eviction only) |
| Hits | identical static text chat | shared token prefix |
| Skips | tools, tool history, media, stateful, opt-out | salts / tool reorder / system change |
| Mode | `K3_RESPONSE_CACHE_MODE=static` | always on |
| Scope | hashed API key (no team pool live) | one engine, salts dropped |

Cache key (`tenancy/cache_policy.py`): model, target, revision
`K3_RESPONSE_CACHE_REVISION=1`, hashed auth scope, canonical generation
settings. Bookkeeping excluded (`shared_session`, timeouts, rpm/tpm, …).
Unknown non-JSON settings **bypass** rather than unstable-hash.
`shared_session` must stay excluded or every lookup fails closed.

---

## 3. Dashboard TPM (do not confuse with capacity)

`dashboard/server.py` + `index.html`

- **Total TPM**: trailing **60s** `(Δprompt + Δgen) / dt * 60`. Honest.
- **Peak / burst TPM**: max of **2s scrape rate × 60** over ~20 min chart.
  One fat prefill → millions. Chart clips p98; **KPI does not**.
- Live example: Total **535k**, Peak **6.93M**, and that peak was
  **6.93M input / 389 output**.

LiteLLM TPM quota is a 60s reservation window. Compare it to Total TPM, not Peak.

---

## 4. Remote portal (NewAPI) — not in this repo

Channel 285: `https://api.cflowx.in`, model `FW-Kimi-K3`, priority 10, weight 5.
Affinity enabled, rule 3 `prompt prefix` / conversation, TTL 14400s,
`switch_on_success=true`, `skip_retry_on_failure=false`.
Nominal fresh-selection share ~25% vs other priority-10 channels.

Portal retries: 10× on 429/500, backoff 500ms×1.6 cap 4s. **502 is not** in
the ordinary retry list. Observed: 429s from gateway shed, LiteLLM 64-parallel,
3M TPM reserve, and Responses 404→cooldown.

---

## 5. Optimization register — every item and actual status

IDs from `docs/SYSTEM-DESIGN.md`. Status is **live measured**, not “in a doc”.

### Tier A — KV capacity

| ID | Idea | Status | Evidence |
|---|---|---|---|
| A1 | SGLang DP-attention, de-dupe MLA KV ~8× | **FAIL** hybrid Mamba OOM. Reverted vLLM | SYSTEM-DESIGN 6.1 |
| A2 | fp8 KV e4m3 | **FAIL** pool tokens 2.20M→4.32M but seats did not rise (KDA pages). AITER MLA `batch_size=1` assert unless `VLLM_ROCM_USE_AITER_MLA=0`. Needle regressions. Off. | K3-STATUS §6 |
| A3 | 64 GB native host L2 | **FAIL read path**. GPU→CPU ~806 GB/wave, CPU→GPU 0, L2 hits 0. Disabled. Do not raise. | a3-readpath |
| DCP>1 | vLLM decode-context parallel | Build can do hybrid; live **DCP=1**. Untested on ROCm K3; forces PIECEWISE graphs | K3-STATUS §9 |
| EPLB | Expert load balance | CLI exists, live N/A, untested | |

### Tier B — waste / scheduling

| ID | Idea | Status |
|---|---|---|
| B0 | Drop inbound cache salts | **LIVE.** Prefixes share across portal keys |
| B1 / G8 | Right-size max-num-seqs | Live **160**. Repo **96**, needs engine restart. 160 was not binding at offered 96 (cold seats ~42, warm ~66). Production still ran 38–74 |
| B2 | `scheduling-policy: priority` | **LIVE.** Gateway stamps priority |
| B3 | long-prefill threshold | **LIVE but inert.** Threshold **=** batch 16384, so one long prefill can take the whole step. This is the decode-collapse bug |

### Tier C — compute

| ID | Idea | Status |
|---|---|---|
| C1 | AITER + `AITER_BF16_FP8_MOE_BOUND=0` | **LIVE.** Bound=0 was required to stop silent A8W4 garbage |
| C2 | batch tokens 8192→16384 | **LIVE.** Helps prefill round-trips; hurts decode isolation |
| C3 | A8W4 / fused KDA decode | **PARKED.** Historical bundled TPOT 57.8→39.8 ms, not isolated |
| C4 / DSpark | 7-token speculation | **PARKED.** Recipe in `experiments/2026-09-20-amd-optimized/`. AMD withdrew 3× claim. Needs optimized image |
| Graphs | FULL_AND_PIECEWISE → FULL_DECODE_ONLY | **LIVE.** Freed ~11.7 GiB into KV |
| torch.compile | Requested | **No-op on K3** (“model does not support it”) |
| Stream interval | 1→4 | **LIVE.** SSE batching, not 4× compute |
| Async sched | on | **LIVE** already in baseline; explicit flag was a no-op |
| Watermark | 0.02 | **LIVE.** Keep 2% free to cut preemption |
| MM processor cache | 8 GiB | **LIVE.** Untested isolated gain |
| ReplaySSM | CLI exists | K3 AMD path does not opt in |
| Model runner V2 | available | K3 excluded on ROCm default |

### Tier D / G — traffic / control plane (mostly LIVE)

| ID | Idea | Status |
|---|---|---|
| D2 | Clamp max_tokens | **LIVE** LiteLLM + gateway |
| D1 / E1 | Off-box P1 | Armed, **empty URL** |
| G1 | P2-agentic class | **LIVE** |
| G2 | Portal 1.5M TPM / 400 RPM / 32–48 conc | Proposed; local key seen at 3M/64. Portal not owned here |
| G3 | Oversize → 400 not 500 | **LIVE** |
| G4 | Image estimate 1024→4096 | **LIVE** deployed; repo has dimension-aware upgrade **not deployed** |
| G5 | Agentic output 512 | **LIVE.** Painful with max thinking |
| G6 | Shed KV>90% and queue>8 | **LIVE** (queue depth 8) |
| G7 | Don’t send client priority to LiteLLM | **LIVE** (stripped in normalize; gateway re-stamps for engine) |
| G9 | Sort tools by name | **LIVE** |
| Distill | P3 loopback | **PARKED** |

### Cache-control rollout 20:16–20:28Z (LIVE, no engine restart)

- Canonical Redis response cache + streaming usage restore
- Nonblocking media normalize
- Global admission ceiling atomic
- Nginx backup-upstream removed
- `max_completion_tokens` precedence
- Connection ownership / 5xx counted once

### Prepared in **repo**, **not deployed**

1. `use_chat_completions_api: true` on all local models
   (`redesign/tenancy/render_config.py`). Tests: `test_responses_bridge.py`
2. `_bind_thinking_controls` in `media.py`. Tests: `test_thinking_controls.py`
3. Image 1568-px cap + dimension-aware estimator
4. Repo `max-num-seqs: 96` (engine still 160)

### AMD optimized candidates (source only, not live)

`experiments/2026-09-20-amd-optimized/`

- Image `johnqin2025/kimi-k3-dspark:1.1.0-mi355x-…` + Infera overlay
- **EP off**, batch **4096**, seqs **64**, util **0.88**, context still 262144
- Optional DSpark: method `dspark`, 7 tokens, `ROCM_AITER_MLA`, draft
  `Inferact/Kimi-K3-DSpark@cf6b824…`
- No local runtime validation. Do not claim 3×.

---

## 6. Failed / do-not-retry without new evidence

| Attempt | Why it died |
|---|---|
| SGLang hybrid DP | Hybrid state cache OOM |
| FP8 KV + AITER MLA | `mla_gluon[bh16bn128] requires batch_size=1` |
| FP8 KV + TRITON MLA | More tokens, fewer seats; needle fail |
| Host KV 64 GB | Write-only |
| gpu-util 0.96 + 256k | EngineDeadError at 186624 computed tokens, amdgpu evict |
| flashkda prefill | `shared Kimi GDN layer only supports Triton KDA` |
| A8W4 without BOUND=0 | Silent bad answers |
| MXFP4 without AITER | `No MXFP4 MoE backend supports the deployment` |
| ReplaySSM / V2 runner | Not active for this K3 AMD path |
| Raising A3 size | Same dead read path |

Correctness gate still **fails needle-128k-depth50** on current BF16/AITER
(observed `<|open|>` or `PHOENIX-447` vs `PHOENIX-4471`).

---

## 7. Live performance (use these, not marketing)

### Isolated / controlled

| Setup | Result |
|---|---|
| Direct engine, ~2k in / 64 out, cold | ~13.7 out t/s |
| Same, cached | ~11.4 out t/s |
| Gateway after pause | ~5.9 out t/s |
| Synthetic 96 conc, 30k prompt, 64 out, cold BF16 | **29.4 out t/s aggregate**, TTFT p50 115s, TPOT p50 1.19s/token |
| Historical shared-prefix bench | high **aggregate** TPM; not per-request agentic speed |

This hardware’s **per-request decode ceiling on the live image is ~12–20 t/s**,
not 60–70.

### Production windows 2026-09-20

| Window | Running | Gen aggregate | Per-req ITL/decode | Notes |
|---|---|---|---|---|
| ~20:24 | 12.4 | 70 t/s | ITL p50 177ms (~5.7 t/s) | KV 27%, 0 preempt |
| ~21:03 | 38 | 122 t/s | ITL p50 184ms | 75% reasoning, 512-cap pileup |
| ~22:00 (this session) | 10.8 | 76 t/s | ITL p50 128ms (~7.8 t/s) | 78% prefix hit, KV 19% |
| LiteLLM last 15m @22:01 | — | — | e2e 5.47 / decode 6.31 t/s | 84% reasoning, 32/120 hit 512 |
| ≥90% cached cohort | — | — | **5.7 t/s** | Cache ≠ decode speed |

Engine log pattern (still live before restart):

- Decode-only: 120–174 gen t/s / 8–13 req ≈ **13–21 t/s each**
- Prefill burst 13k–18k prompt t/s: gen falls to **9–11 t/s total**

GPUs 94–100% busy at 58–67°C. Not thermal-bound. Clock one GPU dipped ~1670 MHz.

### Thinking mismatch (verified on live `/tokenize`)

| Request | Encoded | Effect |
|---|---|---|
| omitted | 91 tok | default max thinking |
| root `reasoning_effort: low` | **same 91 ids** | **no-op on deployed gateway** |
| template `thinking_effort: low` | 92 | works |
| template `thinking: false` | 24 | works |

Repo bind makes normalized low == native low. **Not on the box.**

### Why Azure 60–70 t/s is the wrong comparison

- Public hosted K3 is ~37–62 t/s (Artificial Analysis), dedicated fleet
- Same TPM quota ≠ same GPUs, isolation, or thinking mode
- 11 concurrent 40k decodes × 60 t/s = 660 out t/s; this box delivers 76–170
- Moonshot serving note: 64+ accelerator supernodes, not 8

---

## 8. Code map

```
/root/cflow-gpu/
  config.yaml                         # repo engine intent; NOT live
  dashboard/{server.py,index.html}    # TPM / Peak TPM
  docs/                               # STATUS, SYSTEM-DESIGN, K3-STATUS,
                                      # CACHE-*, NEWAPI-CACHE-AFFINITY
  eval/harness/
    engine_window.py                  # 60s Prometheus rates
    prod_stats.py                     # nginx + engine
    prefix_probe.py / agentic_load.py
  experiments/2026-09-20-amd-optimized/   # candidate image+DSpark
  experiments/2026-09-19-throughput/
  redesign/
    gateway/   server, media, tokens, classification, clamping,
               backpressure, policy, engine, metrics, offbox
    tenancy/   callback, cache_policy, policy, render_config
    edge/      nginx key helpers
    deploy/    deploy.sh, litellm yaml renderer, systemd, containers
    tests/     serving invariants, cache, thinking, responses, media
  nginx/

Installed live copies: /usr/local/lib/k3/redesign/...
Engine:            /scratch/hf/config.yaml  + docker k3
LiteLLM yaml:      /scratch/deploy-state/litellm.yaml
Gateway unit:      /etc/systemd/system/k3-gateway.service
```

Request lifecycle functions (read these first):

1. `tenancy/callback.py` `async_pre_call_hook`
2. `gateway/media.py` `normalize_payload`
3. `tenancy/policy.py` `TenancyPolicy.decide`
4. `gateway/server.py` `Handler.do_POST`
5. `gateway/policy.py` `GatewayPolicy.decide`
6. `gateway/engine.py` `EngineClient.proxy` + `_drain`
7. vLLM scheduler: batch budget, long-prefill, Mamba checkpoint, watermark

---

## 9. Remaining optimizations (for the next AI to prioritize)

Already established, not hypothetical:

### Control plane (no engine restart) — high confidence, limited t/s win

1. **Deploy thinking bind.** Routine profile
   `chat_template_kwargs={thinking: true, thinking_effort: low}`; keep max
   where wanted. Cuts 80%+ reasoning volume and 512-cap retries. Does **not**
   make each GPU step 8× faster. Stable per-session effort also protects prefix.
2. **Deploy Responses bridge** `use_chat_completions_api: true`. Stops 404→cooldown.
3. **Deploy image 1568 + dimension estimator.** Less vision prefill / mis-class.
4. **Cancel on client_gone** (today GPU keeps running 600s). Portal ~half of
   streams were client_gone in one window. Wasted decode seats.
5. **Dashboard Peak TPM** = max of true 60s windows; keep 2s rate as “prefill spike”.
6. **Do not raise 64-parallel / 3M TPM** while engine ITL is 128–180ms. That
   admits more work into a busy decode batch.

### Engine scheduling (restart, A/B required)

7. **Make B3 real:** `max-num-batched-tokens` 8192 then 4096;
   `long-prefill-token-threshold` **3072–4096**, not equal to the batch.
   Goal: stop 170→9 gen t/s collapse; also more KDA checkpoints.
8. **stream-interval 1** separately (delivery, not compute).
9. **Admit on ITL/TTFT/queue**, not only KV% and request count. KV 19% still
   ran 11 decodes at 8 t/s. Cutting 160→96 alone will not bind today’s occupancy.
10. Optional sparse prefix retention `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`
    (unset = dense). Untested correctness.

### Compute / image (only path toward 60–70 t/s)

11. AMD optimized image vs live, **matched** cold/warm, low conc then 64.
    Changes kernels + EP off + 4096 batch together — isolate if possible.
12. DSpark 7-token on that image. Measure acceptance, short agentic outs,
    unique long prompts, vision. No synthetic accept.
13. Isolated C3 (A8W4 / KDA fused) with BOUND=0 and needle gate.
14. SGLang again **only** if hybrid DP-attention memory is actually sized;
    last try OOM’d. This is the real ~8× KV lever if it ever works.

### Will not get 60–70 t/s

- TPM / Peak TPM cosmetics
- Redis 300s TTL
- Host offload retry
- FP8 KV retry on this image
- Prefix cache alone (already 50–78% and well-cached calls are 5.7 t/s)
- Believing 8–9 t/s is a quota bug

### Honest ceiling

On **this** image and 8 GPUs, isolated decode is ~13–15 t/s. Contended
agentic+vision+max-thinking is ~5–8 t/s. Azure 60–70 needs a different
serving stack and/or more isolated capacity, plus effort control.

---

## 10. How to measure (do not invent)

```bash
# 60s engine window
python3 /root/cflow-gpu/eval/harness/engine_window.py --seconds 60 --label x --out /tmp/w.json

# dashboard
curl -s http://127.0.0.1:8080/api/state | python3 -c \
  "import json,sys; d=json.load(sys.stdin); w=d.get('window') or {}; \
   print(w.get('total_tpm'), w.get('prompt_tpm'), w.get('gen_tpm'))"

# engine 10s lines
docker logs k3 --since 5m 2>&1 | grep 'Avg prompt throughput'

# thinking encode (engine must be up)
# POST /tokenize with/without chat_template_kwargs.thinking_effort
```

Compare Azure only with same model revision, thinking effort, tools/images,
output length, cache warmth, concurrency, and the same speed formula
(output tokens / decode time, not e2e including thinking-as-delay).

---

## 11. Current process state at handoff

Observed **2026-09-20 ~22:12Z**: container `k3` **restarted** (~3 min up) and
was still loading weights/AITER JIT. Dashboard `server.state=unreachable` and
KPI numbers frozen from the previous engine (start had been 17:59:06).
Gateway/LiteLLM were still up (started ~20:16 / 20:28). After engine comes
back, prefix cache is **cold** until traffic rebuilds it.

Do not change live services from this document. Implement from repo, deploy
control plane first, then one engine A/B at a time with the needle gate.
