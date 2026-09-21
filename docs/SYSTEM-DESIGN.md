# System Design v2 — Capacity-First Serving for Kimi-K3 on 8× MI355X

**Status:** live on the box, 2026-09-20. Partial. See §6.1 for what was
tried and what failed.
**AUDITED 2026-09-20** by four independent reviewers. Several quantitative
claims below were found unsound and are corrected inline. Read
[`redesign/AUDIT-2026-09-20.md`](../redesign/AUDIT-2026-09-20.md) before acting
on any number in this document.
**Supersedes:** nothing. This is a design layer *above* `K3-DEPLOYMENT.md`, which
remains the authoritative record of what was measured on the box. Where this
document and that one disagree on a *measurement*, that one wins. Where they
disagree on an *explanation*, see §9 — this document corrects two of them.

**Scope:** how to serve four workload classes plus distillation traffic from a
single 8× MI355X node, with one Kimi-K3 replica, at enterprise quality.

---

## 0. Executive summary

The box is not slow. Its KV cache is roughly eight times smaller than the
hardware paid for, because Kimi-K3's MLA attention has a single latent KV head
and tensor parallelism cannot shard it — so at TP=8 every GPU holds a complete
copy of the cache.

Every prior plan, including the first version of this one, treated the
2,295,266-token KV pool as a fixed constant and tried to schedule around it.
It is not a constant. It is a consequence of a parallelism choice.

This design therefore attacks, in strict order:

1. **Capacity** — make the KV pool bigger (up to ~8×, then ~2× again with fp8).
2. **Waste** — stop discarding completed prefill through preemption.
3. **Compute** — kernel and engine efficiency, which only becomes the binding
   constraint *after* 1 and 2.

The engine question (vLLM vs SGLang) was originally sequenced third, on the
reasoning that kernel efficiency only binds after capacity is fixed. **Z2
overturned that.** Static verification of both upstreams found that KV
de-duplication for a hybrid KDA+MLA model on ROCm is available on SGLang and
not on vLLM — so the 8× lever and the engine choice are *the same decision*.
See `redesign/Z2-FINDINGS.md` and §8 D2.

Admission right-sizing still goes first. It is free, engine-independent, and
without it the migration cannot be measured.

---

## 1. The binding constraint

### 1.1 Single replica, permanently

Kimi-K3 is 2.8 T parameters at MXFP4 ≈ **1.5 TB of weights**. The node has
8 × 288 GB = **2,304 GB** of HBM3E. Weights occupy **187.5 GB per GPU** — 65% of
every card — before one token of KV exists.

A second replica needs a second full copy. It does not fit at any quantization
currently served. Therefore, on this hardware:

| Ruled out | Why |
|---|---|
| Prefill/decode disaggregation | Needs separate prefill and decode workers, each holding full weights. 3 TB required, 2.3 TB available. |
| Prefix-cache-aware routing (llm-d, AIBrix, GKE Inference Gateway, production-stack) | Routes *between* replicas. With one replica there is no routing decision; the engine scheduler already sees every request. |
| Replica autoscaling, rolling deploys, zero-downtime restart, N+1 | All require a second replica. |

These are not permanently wrong — they are wrong *now*. §8 records the tripwire
for each.

### 1.2 What remains

Three levers, all untouched in the current deployment:

1. **KV capacity** — the pool is 8× deflated by replication, and host DRAM
   (2,048 GB against 524 GB of HBM KV) is idle.
2. **Admission and scheduling** — the engine admits ~5× what it can hold.
3. **Traffic shaping** — work that does not need a 2.8 T model is being served
   by a 2.8 T model.

---

## 2. The memory cost model

This supersedes the flat `228 KB/token` figure used in earlier analysis. That
number was an empirical average which *concealed* the replication rather than
exposing it.

```
cost(sequence) = 54 MB            KDA recurrent state
                                  69 linear-attention layers, TP=8
                                  FIXED per request, overwritten in place

               + 216 KB × N       MLA KV cache
                                  24 full-attention layers
                                  = 27 KB × 8 TP replicas
```

Source for the per-layer figures: SGLang's Kimi-K3 day-0 analysis
(`~54 MB` KDA state under TP=8; `~27 KB` per token of MLA KV).

### 2.1 Why the 8×

MLA compresses all KV into a single latent vector per token — effectively
**one KV head**. Tensor parallelism shards the KV cache along the head
dimension. When `tp_size > num_kv_heads`, the cache is duplicated
`tp_size / num_kv_heads` times. At TP=8 with one head, that is 8 copies.

### 2.2 The arithmetic that confirms it

| Quantity | Value |
|---|---|
| HBM allocated to KV (61 GiB × 8) | ~524 GB |
| Capacity at 27 KB/token (no replication) | **19.4M tokens** |
| Capacity actually reported by the engine | **2,295,266 tokens** |
| Ratio | **8.4×** |
| TP size | **8** |
| Cross-check: 27 KB × 8 | 216 KB/token vs 228 KB/token measured (5% gap = KDA amortised) |

**This is one derivation, not two — and its units do not match.** `488 GiB` is
an 8-GPU aggregate; `2,295,266 tokens` is what a single *worker* reports.
`RUNBOOK.md:154-156` shows `Available KV cache memory: 61.06 GiB` and the token
count on adjacent lines of the same per-worker output, and 61.06 GiB ÷
2,295,266 = **27.9 KiB/token per rank** — matching the documented 27 KiB
directly. The factor of 8 was introduced by the unit mismatch.

The "228 KB/token measured" cross-check is the same division rearranged
(524e9 ÷ 2,295,266 = 228,296), so it corroborates nothing.

**The conclusion survives on physics, not on this arithmetic:** MLA has one KV
head, TP cannot shard it, and vLLM's DCP documentation states the latent KV is
replicated in full across every TP rank. But the model fits only one of the
three `(available KV, pool)` pairs in this repo — the others imply 44.3 and
37.8 KiB/token — so **G0 is reinstated** and §2 should be read as an inference
awaiting confirmation.

### 2.3 Consequences

- **Break-even at ~250 tokens.** Below that, fixed KDA state exceeds the entire
  MLA KV cost. For the observed traffic distribution KDA is second-order — but
  once the MLA side is de-duplicated, KDA becomes the binding term and
  *concurrency*, not context length, is what limits the box.
- **`max-num-seqs: 512` was never absurd in ambition.** Against a de-duplicated
  ~19M-token pool, 512 sequences at a 30k mean is roughly the right order. It is
  wrong *today* only because replication divided the pool by 8. This predicts
  the setting comes back up after A1.
- **Sustainable concurrency: WITHDRAWN.** This section previously derived ~75–80
  sequences and called the agreement with production a cross-check. Three
  problems, all found in audit:
  1. The model has **no prefix-sharing term**, and `K3-DEPLOYMENT.md:526` says
     the engine's own no-sharing estimate is "pessimistic by ~5× for this
     workload. Ignore it."
  2. `K3-DEPLOYMENT.md:39` records **427 concurrent requests sustained**.
  3. The "≈73 in system" agreement was produced by adding *queued* requests,
     which hold no KV. Against running-only (46–51) the model is ~60% high.

  A prefix-sharing term is required before any admission number is quoted.

---

## 3. Observed baseline

From `experiments/2026-09-19-throughput/FINDINGS.md` and
`eval/runs/20260919T145955Z-baseline/`. These are measurements, not estimates.

| Signal | Value | Reading |
|---|---|---|
| Prefix cache hit rate (marginal, live) | 6–13% | collapsed from a documented 85% |
| KV pool occupancy | 94–98% | at the replication wall |
| Preemptions | 4–9 / min | completed prefill being discarded |
| Queued on `capacity` | 22–25 | admission is not refusing, it is deferring |
| Running sequences | 46–51 | ≈ the pool's true capacity |
| `max-num-seqs` configured | 512 | ~5× overcommitted |
| Live throughput | 233k–544k TPM | against a 1,493,000 TPM measured ceiling |
| Input : output token ratio | **97.7 : 2.3** | prefill-dominated |

### 3.1 The failure signature is documented upstream

vLLM's decode-context-parallelism writeup reports baseline TP reaching **100% KV
at concurrency 64** and plateauing, versus sharded KV sustaining concurrency 512
at 82% KV. The production numbers above sit exactly on that wall. This is a
known, named failure mode of MLA-under-TP, not a tuning deficiency.

### 3.2 Three hypotheses for the hit-rate collapse, none yet excluded

1. **Eviction spiral** (FINDINGS.md §1). The long-context tail evicts the shared
   prefix. Fixed by tuning.
2. **Client `cache_salt`.** `K3-DEPLOYMENT.md` §11.2 confirms per-API-key salting
   drives the hit rate to **0% by design**. Fixed by a conversation, not a config.
3. **The sample is one developer.** Added by Z1/Z3. See below.

### 3.3 The traffic evidence is much weaker than it looked

`redesign/traffic` over the surviving `prod_stats.json` — the only production
log data that outlived the teardown:

| | |
|---|---|
| Coverage | **58 minutes**, 651 requests, 10.85 req/min |
| Top client share | **92% of billable requests** from one key (`sudarshan`, 2 IPs) |
| Distinct billable clients | 5, of which 3 sent fewer than 10 requests |
| Failure rate | **16.6%** |
| Edge duration | p50 **58 s**, p95 **373 s**, p99 **1,044 s** |

This is not a production load; it is largely one person's testing. Two
consequences the design must absorb:

- **Hypothesis 3 is now the most likely.** A 6–13% prefix hit rate on one
  developer's varied ad-hoc prompts is the expected result, not a pathology.
  The 85% shared-prefix premise cannot be evaluated from this window at all.
  **The eviction-spiral narrative is unproven**, and the case for tuning against
  it rests on a sample too small to carry it.
- **Every concurrency number in §2.3 is provisional.** The replication finding
  is unaffected — it is arithmetic over the engine's own reported pool — but the
  ~30k mean prompt, and therefore the admission sizing, needs Z3.

### 3.4 The defect nobody costed: 11.5% of requests are rejected

**75 of 651 requests returned 400** — 69% of all failures and 11.5% of all
traffic. `config.yaml` already records the cause: clients send a fixed
`max_tokens`, the engine reserves `prompt + max_tokens` against one window, and
oversized requests are refused.

This is larger, more certain, and cheaper to fix than anything in Tier A. It
needs no GPU: a per-class `max_tokens` clamp at the tenancy layer. **Register
item D2 is promoted to the top of the programme.**

**But it does not explain all of it.** Dry-running the clamp against the
observed prompt distribution (`python3 -m redesign.gateway`) shows the
fixed-`max_tokens` reservation accounts for **~7 points of the 11.5%**, with the
clamp recovering all of those. The remaining ~4.5 points have a cause the
surviving logs do not identify — `prod_stats.json` records only the status code,
not the engine's error body. The gateway's error taxonomy is what will name it
on first boot. **Do not claim D2 recovers 11.5%.**

---

## 4. Architecture

Four layers with strict ownership. The single largest correction from v1 is that
the gateway no longer models KV occupancy: with a hybrid KDA+MLA pool it cannot
know the true memory state, and attempting it produces a wrong answer with
confident arithmetic.

```
        client
          │
          ▼
  ┌───────────────┐   TLS, default-deny (Bearer present), per-key conn caps, usage.log
  │  nginx (keep) │   owns: transport + attribution
  └───────┬───────┘   owns NOT: anything token-aware, key allowlists, credential swap
          │
          ▼
  ┌───────────────┐   virtual keys, token budgets, spend, class assignment,
  │ LiteLLM       │   max_tokens clamp, off-box fallback, small-model routing
  │ (tenancy)     │   owns: WHO and HOW MUCH
  └───────┬───────┘   owns NOT: scheduling, queue depth
          │
          ▼
  ┌───────────────┐   per-class fairness, circuit-break on engine metrics,
  │ backpressure  │   429-before-overload, priority stamping
  │ (~150 lines)  │   owns: WHEN TO REFUSE
  └───────┬───────┘   owns NOT: KV arithmetic, batching decisions
          │
          ▼
  ┌───────────────────────────────────────────┐
  │ engine (vLLM or SGLang) · TP8 + EP        │  owns: batching, cache-aware
  │  KDA state pool  +  MLA KV pool           │  admission, KV/KDA allocation,
  │  L1 HBM          +  L2 host DRAM          │  preemption
  └───────────────────────────────────────────┘  owns NOT: customers, quotas
```

Nginx forwards the client's `Authorization` header unchanged. It does not
read `customers.tsv` and it does not replace the header with a LiteLLM master
key. A missing or non-Bearer header is 401. A Bearer token is a conn/req
identity only; LiteLLM decides whether it is a real user and how much they
may spend.

The engine owns *scheduling* — the gateway has no view of memory.
LiteLLM owns *tenancy* — who and how much. The engine has no concept of a customer.
The backpressure service (the gateway process) is the seam: it reads
`gpu_cache_usage_perc`, the running/waiting gauges and the preemption counter,
and refuses classes when the engine says it is in trouble. **It never computes
a token budget of its own.**

---

## 5. Traffic classes

Class is assigned once, at the tenancy layer, from prompt length and API key,
and carried to the engine as a `priority` field.

| Class | Trigger | Priority | TTFT target | Served by |
|---|---|---|---|---|
| **P0 Interactive** | prompt ≤ 32k, streaming | 0 | p95 < 3 s | K3 |
| **P1 Short chat** | prompt ≤ 8k, no tools | 1 | p95 < 1 s | **off-box model** |
| **P2 Long context** | prompt > 32k | 2 | p95 < 60 s | K3, budgeted |
| **P3 Batch / distill** | explicit header or batch API | 3 | none | K3, off-peak |

Priority scheduling forcibly preempts a lower-priority running request when a
higher-priority one waits. A P3 distillation request preempted by a P0 keystroke
is the system working — which makes P3 restartability a hard requirement (§7).

**P1 goes off-box rather than to a co-resident small model.** Co-residency costs
~138 GB of HBM (dropping `gpu-memory-utilization` 0.92 → 0.86, cutting K3's pool
by ~25%) plus SM contention presenting as TTFT jitter on P0. Routing P1 to an
external endpoint through LiteLLM achieves the same traffic relief at zero HBM
cost. Revisit only if A1 lands and the pool has genuine slack.

---

## 6. Optimization register

Ranked by expected value. "GPU cost" is validation time on the live node,
assuming ~5 min per engine restart (measured startup: 262 s).

### Tier A — Capacity: make the pool bigger

| # | Optimization | Mechanism | Effect | Confidence | GPU |
|---|---|---|---|---|---|
| **A1** | MLA KV de-duplication — DP attention (SGLang) or DCP (vLLM) | Partitions the KV latent by sequence (DP attention) or by token (DCP) instead of replicating it per rank | **up to ~8× pool** | **Z2: available on SGLang, NOT on vLLM.** vLLM's DCP lists hybrid-model integration as future work, has no ROCm support, and K3 itself is "underway". SGLang documents `--enable-dp-attention` in its K3 recipe and has merged hybrid DP-attention fixes. | ~4 h |
| **A2** | fp8 KV cache (`e4m3`) | Halves MLA bytes/token | **~2× pool** | **Z2: available on SGLang, blocked on vLLM.** vLLM ROCm tracker #50682 lists fp8 KV as in-progress and not working on MI355X. Silent-garbage risk either way. | ~3 h |
| **A3** | CPU DRAM L2 tier | Offloads evicted blocks to host RAM | ~2× *addressable* (not resident) | **FAILED 2026-09-20 on vLLM native.** Write-only. Disabled. | done |

A1 and A2 multiply. If both land, KV stops binding entirely and the design's
centre of gravity moves to compute. A3 was the fallback after A1. It is not
a fallback on this engine: it copies KV to host and never serves it back.

### Tier B — Waste: stop discarding completed work

| # | Optimization | Effect | Confidence | GPU |
|---|---|---|---|---|
| **B0** | `cache_salt` audit | Sharing works; inbound salts now dropped | **PASS** | **zero** |
| **B1** | Right-size `max-num-seqs` | **NUMBER WITHDRAWN.** The direction may hold — the box was preempting — but the model that produced "~75" has no prefix-sharing term, and `K3-DEPLOYMENT.md:39` records **427 concurrent sustained**. Acting on 75 risked a ~5× throughput cut. | **Low** | ~1 h |
| **B2** | Priority scheduling + classes | Protects interactive TTFT | **On** | done |
| **B3** | `long-prefill-token-threshold` | Caps the tail's scheduling share | **On at 16384** | done |

### Tier C — Compute: only binds after Tier A

| # | Optimization | Effect | Note |
|---|---|---|---|
| **C1** | AITER flag contract (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`) | Prevents a 2.06× decode regression presenting as an upgrade failure | Mandatory on any image change. Static-analysis result from `FINDINGS.md` §2. |
| **C2** | `max-num-batched-tokens` re-tune | Plausibly 10–20% TTFT | **Done.** Live value is **16384** (was 8192). |
| **C3** | A8W4 vs A4W4; KDA fused decode | **Re-ranked upward.** Previously dismissed as "sub-1% end-to-end" using the *token* ratio. That is the right denominator for memory and the wrong one for latency: prefill p50 is **1.10 s** and decode p50 is **33.25 s**, so decode is ~**72% of end-to-end**. An 8.9% decode-layer gain is ~8.9% of 33 s. |
| **C4** | DSpark / speculative decoding | Re-evaluate | The "97.7% input" dismissal used the same wrong denominator. The batch-size argument still stands; measure rather than assume. |

### Tier D — Traffic

| # | Optimization | Effect |
|---|---|---|
| **D2** | Per-class `max_tokens` clamp | **Recovers ~7 points of the 11.5% rejection rate** (§3.4). Highest-certainty item on the register, zero GPU. Do this first. |
| **D1** | Route P1 short chat off-box | Removes request-count churn at zero HBM cost |
| **D3** | Distillation on a priority-3 loopback lane, off-peak | Near-free throughput; never touches customer quota |

### Tier E — Operational

| # | Optimization | Effect |
|---|---|---|
| **E1** | LiteLLM off-box fallback | Degraded quality instead of an outage during a spot reclaim |
| **E2** | Per-class SLO instrumentation | Marginal hit rate, per-class TTFT, KV occupancy by class, preemption rate |

### 6.1 Tried and measured — 2026-09-20

Live path: `nginx (api.cflowx.in) → LiteLLM :4000 → gateway :8002 → vLLM :8001`.
One replica. No second engine. Distill parked. Off-box URL not invented.

| Item | Result | Evidence |
|---|---|---|
| **A1** DP attention / SGLang hybrid | **FAIL / tripwire.** Hybrid mamba OOM. Reverted to A3-only on vLLM. | bring-up RuntimeError |
| **A2** fp8 KV | **Run 2026-09-20, not inherited.** `--kv-cache-dtype fp8_e4m3` is accepted and the pool doubles, but with `VLLM_ROCM_USE_AITER=1` the fp8 MLA decode path resolves to `aiter/ops/triton/gluon/mla_gluon.py`, whose `bh16bn128` variant asserts `batch_size=1` — every worker died with `requires batch_size=1, got 160` during profiling. Retested with `VLLM_ROCM_USE_AITER_MLA=0` (AITER MoE, and therefore the C1 contract, untouched). See §6.5. | this session's engine log |
| **A3** 64 GB host L2 | **FAIL on the read path.** GPU→CPU writes (fill/replay ~806 GB/wave). CPU→GPU **0**. `external_prefix_cache_hits` **0**. Replay TTFT identical to fill (23.83 s vs 23.82 s). **Disabled on the next restart.** Do not raise 64 GB. | `/scratch/deploy-state/bench/a3-readpath/scorecard.json` |
| **GPU prefix cache** | **WORKS when the prefix stays in HBM.** Documented 18432+7168 point: **70.6% cold / 99.8% hot**, 0 preemptions. | `g-opt-baseline` |
| **B0** `cache_salt` | **PASS.** Identical probe raised hit counters. Tenancy now **drops** inbound `cache_salt` / `prompt_cache_key` so a portal pool can share. | `/scratch/deploy-state/cache-salt.json` |
| **B1** admission | **Kept 427** (withdrawn "75" not applied). 0 preemptions under both campaigns. | live `max-num-seqs` |
| **B2 / B3** priority + long-prefill | **On.** `scheduling-policy: priority`, threshold 16384. | `/scratch/hf/config.yaml` |
| **C1** AITER | **On.** | `vllm-k3.env` |
| **C2** batched tokens | **On at 16384.** | config |
| **D2** clamp | **On** at LiteLLM + gateway. | tenancy callback |
| **D1 / E1** off-box | **Armed, empty.** No `K3_OFFBOX_URL`. | — |
| **C3 / C4** decode kernels / spec | **Not started.** Now the highest remaining *compute* bets: A-tier capacity is exhausted on this engine. | — |

Shared-prefix capacity (engine, 128 out): cold 64-conc **2.00M total TPM** / 10.1k gen TPM; hot 128-conc **8.90M total TPM** / 45.1k gen TPM. Size a YYDS channel at **1.5M TPM / 400 RPM / 96 conc**, not the 3M model-group cap.

### 6.2 Tier F — after A3 read-path fail

The box is a **GPU-resident prefix-cache** server. Host RAM is not a second cache.

| # | Optimization | Why now | GPU |
|---|---|---|---|
| **F1** | Disable A3 | Stops ~806 GB/wave of useless HBM↔host copies. Same hits as before (zero L2 hits). | 1 restart |
| **F2** | Drop client `cache_salt` at the tenancy hop | One portal key + per-user salts would partition the only cache that works. | zero |
| **F3** | nginx `limit_conn` 32 → **96** (key and portal IP) | A shared YYDS key is a pool. 32 clipped it before TPM. 96 is still far under admission 427. | zero |
| **F4** | Keep GPU prefix caching; do not salt by customer | The 70.6% / 99.8% result is the product. | zero |
| **F5** | C3 decode (A8W4 / KDA fused) | Capacity levers A1–A3 are done or dead. Decode is ~72% of e2e latency. | image change + Z5 |
| **F6** | C4 speculative decode | Same reason as F5. Measure; do not assume. | image / flag |
| **F7** | Off-box P1 (D1/E1) when a real URL exists | Removes short-chat churn at zero HBM. Do not invent a URL. | zero |
| **F8** | Portal/channel TPM 1.5M, not 3M | 3M is the *group* cap across providers. This box's honest cold share is ~1.5–2.0M. | zero |
| **F9** | Snapshot before the next engine experiment | Last droplet died with no snapshot. | ops |
| **F10** | Do not try LMCache / HiCache / raise-64GB as a silent A3 retry | Native offload already writes and never reads. A new connector is a new session with a written pass/fail, not a config tweak. | new session |

### 6.3 Observed request shape — YYDS Lioxi, 2026-09-20 16:35Z

The first real portal wave is **not** the g-opt bench (shared 18432 prefix, 128-out).
It is agentic + vision on one 3M/96 key:

| Fact | Live number | Consequence |
|---|---|---|
| Prompt p50 / p90 / max (success) | 8.6k / 60k / 174k | Prefill-bound. Decode kernels (F5/F6) are the wrong next bet for this wave. |
| Completion p50 | **16** tokens | Output reservation, not output compute, is what costs HBM. |
| Prefill:decode TPM | ~42:1 (374k / 9k) | KV fills; gen TPM is noise. |
| Images | 43 / 179 successes, 3.7k–13k image tokens | Estimator still bills **1024**/image → under-classifies as P0. |
| Prefix hit | ~12% this window, 31% lifetime; some 4608 / 18432 shares | GPU prefix works. Most tokens are unique. |
| Classification | 95 P0 admitted vs 72 P2 | `tools` disqualifies P1, so a 20k tool loop becomes **P0**. P0 is never shed. |
| Channel cap | Lioxi **3M TPM / 3000 RPM / 96 conc** | F8 was never applied. The key can stampede the only replica. |
| Engine | 62 running / 36 waiting / KV 98% / admission **427** | 427 was sized for short decode. At ~30k unique prompt, the pool holds ~**60–90** resident seqs. Extra admits sit on `capacity` and preempt. |
| Failures | clamp-too-long as **500**; shed as 429 then LiteLLM "no deployments" | Correct decisions, wrong status / cooldown.

**Re-rank:** F5/F6 stay parked until this wave's decode share is material. Do not invent `K3_OFFBOX_URL`. Do not retry A3.

### 6.4 Tier G — this wave, zero GPU first

| # | Optimization | Mechanism | Why this traffic | GPU |
|---|---|---|---|---|
| **G1** | New class **P2-agentic** (tools or images) | Sheddable priority 2. `max_output` **512** (they emit 16–90). TTFT target 15 s, not 3 s. | Stops tool loops from wearing the P0 SLO and the "never shed" shield. | zero |
| **G2** | Apply F8 on Lioxi | **1.5M TPM / 400 RPM / 32–48 conc** | 96 conc × 30k prompt is more KV than the box has. | zero |
| **G3** | Clamp / oversize → **400**, not 500 | `ValueError` in the LiteLLM hook becomes InternalServerError | 11 live 500s were "320k–430k leaves no room". The portal retries those as outages. | zero |
| **G4** | Image estimate **1024 → 4096** (or bytes-derived) | Conservative over-count | Live images are 3.7k–13k. Under-count sends 60k+ jobs through as P0. | zero |
| **G5** | Hard cap `max_tokens` on tool loops at **512** even if the client asked for 8k | D2 class ceiling for G1 | Engine reserves prompt+max_tokens. 30k+8k × 60 = the whole pool. | zero |
| **G6** | Shed on **KV>90% and waiting>0**, not only preemption rate | Backpressure already has `kv_usage>95%` but P0 is exempt | Once G1 moves the wave off P0, this actually fires before 36 deep. | zero |
| **G7** | Stop stamping `payload["priority"]` | LiteLLM Redis heap `TypeError: tuple vs list` still in spend | Two live failures. Tenancy already has the class in metadata. | zero |
| **G8** | Recompute B1 from this shape | `max-num-seqs` ≈ pool / mean unique prompt ≈ **64–96**, not 427 | 427 admits work the KV cannot seat. Needs an engine restart; do after G1–G5. | 1 restart |
| **G9** | Prefix-stabilize agent payloads | Canonicalize `tools` JSON; drop volatile stamps already covered by F2 | Shared 4.6k/18k hits exist. Reordered tool schemas bust the only cache that works. | zero |
| **G10** | Do not spend a restart on C3/C4 for this wave | Decode is 16 tokens | Measure again if completion p50 leaves the tens. | — |

Do G1–G7 without touching the engine. G8 is the one restart that matches the live occupancy math. G9 is a tenancy normalize, same hop as F2 / thinking_effort.

### 6.5 Results — optimization wave of 2026-09-20 17:00–18:15Z

Everything below was measured on this box in this session. Nothing is inherited
from `K3-DEPLOYMENT.md`, the vendor tracker, or an earlier wave. Records are in
`/scratch/deploy-state/bench/opt-2026-09-20/`.

**Shipped, control plane (G1–G7).** Live-traffic verified.

| item | state | evidence |
|---|---|---|
| G1 P2-agentic class | shipped | `k3_gateway_requests_total{traffic_class="P2-agentic"}` incrementing on portal traffic within seconds of restart |
| G3 oversize → 400 | shipped | edge returns `400 invalid_request_error "prompt of 342,862 tokens leaves no room for output"`, was 500 |
| G4 image estimate 4096 | shipped | `IMAGE_TOKENS`, calibrated against live `prompt_tokens_details` of 3,764 / 9,410 / 13,174 |
| G5 agentic `max_tokens` ≤ 512 | shipped | falls out of G1's class ceiling; covered by `test_an_agentic_turn_is_capped_at_the_agentic_ceiling` |
| G6 earlier shed | shipped | distress now also fires on KV > 90% **with** a queue deeper than 8, not only at 97% or on preemption |
| G7 priority stamping | no change needed | the `tuple vs list` TypeError is confined to 15:04:19Z, before the F2 normalize shipped; `priority` is already stripped from client payloads |

**A2 fp8 KV — tested and rejected on measurement.**

1. It runs, and the pool really does double: **2,197,339 → 4,320,192 tokens**
   (16.48x vs 8.39x max concurrency at full window).
2. It costs the AITER MLA kernel. With `VLLM_ROCM_USE_AITER=1` every worker
   aborts in profiling: `mla_gluon[bh16bn128] requires batch_size=1, got 160`
   from `aiter/ops/triton/gluon/mla_gluon.py`. Serving fp8 at all requires
   `VLLM_ROCM_USE_AITER_MLA=0`.
3. It buys nothing at the shape that matters, because **the doubled pool is not
   the binding constraint**. Cold cache, 96 offered, 30.4k prompts:

   | build | running | KV usage | prompt tok/s |
   |---|---|---|---|
   | fp8_e4m3 | 40 | 33.4% | 13,531 |
   | bf16 + AITER MLA | 42 | 59.8% | 13,976 |

   fp8 stores the same tokens in ~56% of the bytes exactly as advertised, and
   then stops at the same ~42 sequences with two thirds of the pool idle.
4. Tier 3 of the Z5 gate lost one check against a bf16 control taken in the
   same session — `needle-128k-depth10` failed on fp8 (leaked `<|open|>`) and
   passed on bf16.

Given (3), the correctness question in (4) is not worth resolving: there is no
throughput to buy. A2 is closed for this hardware and model.

**`needle-128k-depth50` is a pre-existing failure, not a regression.** It fails
on all three builds tried today — fp8, the bf16 control with AITER MLA off, and
the shipped bf16 with AITER MLA on. The planted code is `PHOENIX-4471` and the
shipped build answers `PHOENIX-447`: it finds the needle at that depth and drops
the final digit, which the check's substring test scores as a miss. The other
two builds answered `<|open|>`, a bare control token, which additionally
suggests `gate/client.py` may be reading a channel-tagged K3 response rather
than the answer text. So the gate is red on the serving build for a reason that
predates this wave. Two things it needs before it can convict anything at 128k:
repeat-N instead of one greedy sample per depth, and a near-miss distinct from
a miss.

### 6.6 The real capacity lever is the prefix cache, not the KV dtype

The seat count on this hybrid KDA+MLA model tracks the prefix-cache hit rate,
not the KV pool. Two runs of *identical* work, same build, same offered load of
96 × 30.4k tokens:

| prefix hit rate | running | KV usage | prompt tok/s |
|---|---|---|---|
| 14.5% (cold) | 42 | 59.8% | 13,976 |
| 34.0% (warm) | 66 | 94.1% | **22,393** |

**+60% throughput from cache hits alone**, on work that was byte-identical.
Mamba/KDA state is per-sequence and is not shrunk by `kv-cache-dtype`, which is
why fp8 could not raise the seat count while shared prefixes could. Live
traffic sits at 21.8%.

That makes prefix reuse the highest-value remaining work, above anything in the
A tier: every point of hit rate is a seat, and every seat is throughput.

**Live reuse, 622 requests with usage detail, 2026-09-20 13:00–18:00Z.** From
`LiteLLM_SpendLogs.metadata → additional_usage_values.prompt_tokens_details`:

| band | requests | mean prompt |
|---|---|---|
| **0 — total miss** | **348 (56%)** | 20,993 |
| 1–24% | 111 | 44,989 |
| 25–49% | 80 | 25,422 |
| 50–74% | 17 | 31,444 |
| 75–100% | 66 | 23,953 |

18.9% of all prompt tokens were reused (3,107,456 of 16,448,618). The 348 total
misses alone are ~7.3M tokens of prefill that bought nothing. A *total* miss
means even the first block missed, so the divergence is at the very head of the
prompt — not in the body.

### 6.7 G9, corrected: it is tool **array** order, not key order

G9 as written in §6.4 — "canonicalize tools JSON" — would have been a no-op.
K3's own encoder already calls `deep_sort_dict` on `tools` (`encoding_k3.py`
:598), which sorts each schema's keys and **preserves array order**, and it
renders the tool declaration *ahead of all conversation content*
(:615). So key order is free and array order is total.

Measured through the engine's `/tokenize`, i.e. the real encoder, against a
1142-token reference (`eval/harness/prefix_probe.py`):

| variation | shared prefix, raw | with normalize |
|---|---|---|
| tool dict keys reversed | 1142 (100%) | 1142 (100%) |
| tools array rotated | **35 (3.1%)** | **1142 (100%)** |
| tools array reversed | **35 (3.1%)** | **1142 (100%)** |
| one extra tool appended | 154 (13.2%) | 154 (13.2%) |
| next conversation turn appended | 1142 (96.5%) | 1142 (96.5%) |

`media.normalize_payload` now sorts `tools` and `functions` by name, so a client
that reshuffles its tool list keeps the whole prompt instead of 3% of it.
Declaration order carries no meaning to the caller. Arrays with duplicate or
missing names are left alone, because names no longer identify the entries and
any sort would be a guess.

Two limits worth stating. Changing the tool *set* still costs the declaration
block (13.2%), because it is one compact JSON blob rendered before everything;
sorting bounds that loss to the blob and puts a late-sorting new tool after the
reusable part instead of shifting all of it. And growing conversation history was
never the problem — it already reuses 96.5%.

This is a mechanism fix, so it is not yet an attributed throughput number: the
portal went idle before it shipped, and `cached_tokens` per request is the metric
to re-read against the 18.9% baseline when traffic returns.

One caveat on the instrument: the warm number above was an accident. The load
generator built deterministic bodies, so the second run of an A/B inherited the
first run's cache and read 60% faster on the same work. `agentic_load.py` now
salts bodies per run; an A/B that does not compare cold to cold is measuring
the cache, not the build.

---

## 7. The distillation lane

Teacher generation must run on K3, so it cannot move off-box. It can be made to
consume only what nothing else wants.

| Property | Design |
|---|---|
| Path | Direct to the engine on loopback. Never through nginx or the tenancy layer — distillation volume must not consume customer quota or appear in `usage.log` as billable. |
| Priority | `3`. Preempted whenever a P0 waits, which is correct. |
| Restartability | Checkpoint per completed sample. A preempted request must cost only its own wasted prefill, never batch progress. |
| Admission | Cut to zero by the circuit breaker when the engine reports distress. |
| Window | Unthrottled only off-peak; a background filler during business hours. |
| Logprobs | If token-level distributions are needed, response size grows substantially. Size the object store and write path before a long run. |

**Do not start a distillation campaign until B1 is complete.** Adding a
high-volume preemptible workload to a scheduler that is already preempting makes
both problems unreadable.

---

## 8. Decisions and their tripwires

### D1 · Substrate: bare metal, not Kubernetes
Kubernetes earns its complexity by making placement decisions across a fleet.
With one node and one replica there are none — the pod is pinned to the only
machine, the autoscaler has nothing to scale, and the ROCm device plugin adds a
failure mode to a system with zero redundancy. Every k8s-native serving control
plane prices its value in cross-replica behaviour that is unavailable here.
**Tripwire:** adopt k8s the same week a second node is committed.

### D2 · Engine: migrate to SGLang, because that is where the capacity is
**Revised by Z2.** The original reasoning — engine third, because kernel
efficiency only binds after capacity is fixed — was correct about *kernels* and
wrong about *what the engine choice buys*. A1 is not a kernel optimisation. It
is a parallelism-layout capability, it is worth ~8× the KV pool, and it exists
on only one of the two engines for this model on this platform.

| | vLLM on ROCm | SGLang on ROCm |
|---|---|---|
| A1 de-duplication | **no** — hybrid integration is future work, ROCm unmentioned, K3 "underway" | **yes** — `--enable-dp-attention` in the K3 recipe; hybrid DP-attention bugs fixed upstream |
| A2 fp8 KV | **no** — in-progress per issue #50682 | **yes** — `fp8_e4m3` documented |
| A3 host tier | yes (connector caveats) | yes (HiCache) |
| Ceiling | ~2× addressable | ~16× resident, ~31× addressable |

So the migration *is* the capacity programme, not a follow-on to it. The kernel
deltas SGLang is usually sold on (A8W4 over A4W4 at 1.2%; KDA fused decode at
8.9% of decode-layer time) remain nearly irrelevant to a 97.7%-input workload —
they are simply not the reason to move.

**Tripwire:** if `redesign/probe` reports `enable_dp_attention` absent from the
pinned image, or G2 shows the pool unchanged with it enabled, this decision
reverts and the plan becomes A3-only on vLLM at a ~2× ceiling.

**Migration hazards** are catalogued in `redesign/Z2-FINDINGS.md` §3. The two
that will bite first: `--max-running-requests` is floor-divided by `dp_size`
(and can produce a server that refuses everything, looking like a migration
failure rather than a config error), and DP attention cannot currently be
combined with prefill CP (`--cp-strategy interleave` asserts `dp_size == 1`).

### D3 · Gateway: keep nginx, add tenancy, build backpressure
nginx is not the weak layer — it terminates TLS, default-denies, enforces
per-key caps, and passes 26 security assertions. LiteLLM over Envoy AI Gateway
because the latter is Gateway-API-native and effectively assumes Kubernetes,
which D1 rules out. The backpressure service has no off-the-shelf equivalent:
no gateway tracks engine distress, because in a multi-replica world that is the
scheduler's job.
**Tripwire:** if LiteLLM's SSE relay cannot sustain production streaming
throughput, nginx bypasses it for P0 and it becomes a control-plane-only path.

### D4 · Second model: off-box, not co-resident
See §5. **Tripwire:** co-locate only if A1 lands and the pool has slack.

---

## 9. Corrections to existing documentation

Two explanations in the current docs are wrong. Both conclusions happen to be
right today; both rationales would mislead a future reader into closing a
question that should stay open.

### 9.1 `K3-DEPLOYMENT.md` §7.4 — fp8 KV "unusable at TP=8"
The stated reason is *"TP=8 gives 12 KV heads per rank, below the 16-head
threshold."* **MLA does not have 12 KV heads per rank — it has one latent
vector.** The head-count argument cannot apply to the MLA layers.

The conclusion (fp8 KV does not work on vLLM/ROCm today) is correct, but for the
upstream reason recorded in vLLM issue #50682: FP8 KV cache for K3 on ROCm is
*in progress*, with PRs #51040, #51011 and #50619 open. **Do not let the stale
rationale close this question** — it is worth ~2× the KV pool.

### 9.2 The flat `228 KB/token` KV cost
Used in earlier capacity analysis. It is an empirical average that conflates a
constant per-sequence term with a linear per-token term, and it hides the 8×
replication that produces it. Superseded by §2.

---

## 10. Availability

Everything above makes one box faster. None of it makes one box available.

**One replica on a spot instance that the provider may reclaim at any time.** A
reclaim, GPU fault, engine crash or config error is a full outage with a ~5
minute best-case recovery from snapshot — assuming the snapshot is current and
DNS moves promptly. No configuration of this hardware delivers a meaningful
availability number.

Mitigations, in descending order of how much they help:

1. **Move the production instance off spot.** The only change addressing the
   largest single failure cause. A cost decision, and it should be made
   explicitly rather than by default.
2. **Off-box fallback in LiteLLM (E1).** When K3 is unreachable, P0 and P1 fail
   over to an external provider at reduced capability rather than returning 5xx.
   A config block, not a project.
3. **Keep the snapshot current and rehearsed.** An unrehearsed restore path is a
   hypothesis.

A second node solves availability and unlocks the entire routing playbook in
§1.1 simultaneously. The architecture in §4 is deliberately shaped so adding one
is an extension, not a rewrite.

---

## 11. Execution plan

### Phase Z — zero GPU (laptop)

| ID | Deliverable | Why it precedes GPU |
|---|---|---|
> **Revised 2026-09-20.** The droplet was destroyed with no usable snapshot and
> the production logs did not survive. Three consequences:
>
> 1. **The first boot is a full rebuild** — ~1 hour plus a 1.5 TB weight pull —
>    not a 5-minute snapshot restore. Session economics change accordingly:
>    snapshot immediately after the first successful build, before experimenting.
> 2. **Build SGLang directly; skip vLLM entirely.** With nothing to preserve,
>    standing up vLLM only to right-size `max-num-seqs` and then migrate would
>    spend a full rebuild on a stack we have already decided to leave. B1 was
>    never vLLM-specific — `--max-running-requests` is the same lesson.
>    **G1 folds into G2.**
> 3. **Z3 folds into Z4.** With no logs, the trace must be captured on first
>    boot, and the tenancy layer already sees every request. The gateway is the
>    instrument.

| **Z1** | `cache_salt` audit — deferred, no logs survive | Becomes a first-boot check rather than an offline one. |
| **Z2** | Static source verification: does DCP / DP-attention support a hybrid KDA+MLA model on ROCm today? | **Determines whether this is an 8× redesign or a 2× tuning exercise.** Pure code reading. Precedent: `validate.py` / `envprobe.py` caught the AITER rename statically before it cost a campaign. |
| **Z3** | Trace capture — **folded into Z4** | No engine comparison is meaningful without the real prompt distribution, prefix reuse, tool calls and concurrency. With the logs gone, capture starts at first boot. |
| **Z4** | Tenancy + backpressure + capture, against a mock endpoint | Entirely vendor- and model-agnostic. Carries D2, the top register item. |
| **Z5** | Correctness gate extension for quantized-KV output quality | fp8/4-bit KV fails as *silent garbage* — the class of bug a throughput benchmark reports as success. |
| **Z6** | Pre-registered experiment definitions | Every GPU session gets a question, variants, metric, pass/fail and rollback, written in advance. |

### Phase G — GPU sessions, each time-boxed to one question

| Session | Question | Pass/fail | Est. |
|---|---|---|---|
| **G-build** | Does the fresh SGLang stack stand up and serve? | Engine loads, gate baseline recorded, **snapshot taken before anything else** | ~2 h + weight pull |
| **G2** | Does `--enable-dp-attention` de-duplicate the KV pool for K3? | **AGGREGATE served tokens across all DP ranks**, or sustained concurrency × mean prompt — never the per-rank pool log line, which stays ~2.3M under DP attention and would record a correct 8× win as a failure. Plus: per-rank occupancy spread (DP attention concentrates the long tail on one rank), preemptions 0, gate passes, **P0 TTFT p95 does not regress** | ~4 h |
| **G3** | Does fp8 KV double the pool again without corrupting output? | Both, or revert | ~3 h |
| **G4** | Does native host L2 serve prefixes after GPU eviction? | **FAIL 2026-09-20.** A3 disabled. Do not re-book as a size ramp. | done |

**G0 and G1 are retired.** G0's question — is the pool replicated — was settled
offline by `redesign/capacity/hypothesis.py` at 3.2% error versus 726% for the
alternative. G1 disappeared with the decision to build fresh on SGLang. Two
sessions saved before the box booted.

G2 carries the migration, so it is the session to over-prepare: the capture
layer (Z4), the correctness gate extension (Z5) and the probe
(`redesign/probe`) must all be green first. Run `redesign/probe` inside the
pinned SGLang MI35x image during G-build, before booking G2 — if
`enable_dp_attention` is absent, G2 should not be booked at all.

### GPU cost discipline — non-negotiable

- Every restart is ~4–5 min of paid GPU. **Batch variants into one session**
  using the existing `experiments/2026-09-19-throughput/driver.sh` and
  `run_all.sh`, which are already built and pre-validated against both images.
  Reuse them; do not poke interactively.
- **Validate the harness, tenancy and backpressure layers on a cheap
  small-model GPU box first.** None of that code is vendor- or model-specific.
  Arriving at the MI355X with an unvalidated harness is the most expensive
  mistake available — and the 2026-09-19 campaign already hit it once, when a
  `docker exec`'d benchmark outlived its parent and corrupted a measurement.
- Snapshot before every session. The 5-minute restore path is the rollback.
- **No session without a written pass/fail.** "Boot it and see" is how GPU
  budgets disappear.

---

## 12. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| A1 is unavailable for hybrid KDA+MLA on ROCm on both engines | **High** | Z2 answers this before any GPU spend. If unavailable, A2+A3 become the plan and the ceiling is ~4× rather than ~16×. |
| Hit-rate collapse is client `cache_salt`, not eviction | **High** | Z1 gates everything. |
| fp8 / 4-bit KV produces silent garbage (§7.3 failure class) | **High** | Z5 gate extension; correctness gate before any customer traffic. Never trust a throughput benchmark to detect it. |
| Spot reclaim during business hours | **High** | §10. |
| CPU offload connector instability at scale (vLLM #52656) | Medium | **Superseded.** Native L2 is write-only at 64 GB (G4 / a3-readpath). A3 is off. Do not ramp. |
| `max-num-seqs` right-sizing overshoots downward, costing throughput | Medium | Tune upward from a stable base once preemptions are zero — far safer than tuning down from 512. |
| LiteLLM becomes an SSE relay bottleneck | Medium | Benchmark at production concurrency before cutover; D3 tripwire. |
| Backpressure service is a new SPOF in the request path | Medium | **Fail open** on controller error — degraded admission beats no service. |

### Do not revisit

| Idea | Why closed |
|---|---|
| `gpu-memory-utilization` 0.96 | Killed the engine mid-prefill. Left ~11.5 GiB/GPU; `svm_range_evict_svm_bo_worker` at the second of the crash. |
| `torch.compile` | Inert on K3. Re-confirmed in the 2026-09-19 logs. |
| Lowering `max-model-len` to reclaim KV | KV pages are allocated on demand. Reclaims nothing; rejects the 3–7% of traffic above 100k. |
| Setting `block-size` | No-op. The hybrid model forces 768 to keep the attention page ≥ the mamba page. |
| P/D disaggregation, cross-replica routing | §1.1. Requires a second copy of 1.5 TB of weights. |
| fp8 KV *for the reason given in §7.4* | **Reopened.** See §9.1. |

---

## 13. Open questions

1. **Real prompt-length distribution.** Every sizing number derives from a ~30k
   mean inferred from 229 requests. A week of real data could move
   `max-num-seqs` by 2× in either direction. Z3's main deliverable.
2. **Is the hit-rate collapse client-caused?** Z1.
3. **Request-rate ceiling for the tenancy layer.** ~1 req/s at 256 concurrency
   makes LiteLLM's overhead irrelevant. If P1 volume is hundreds of req/s, that
   changes and the proxy needs benchmarking before commitment.
4. **Is a second node a cost decision or a hard no?** Availability, not
   throughput, is the reason to revisit.
5. **What is P1 traffic worth?** Off-box routing is free in HBM but not in
   money. If that traffic is high-margin, co-residency after an A1 pool
   expansion becomes the better answer.

---

## 14. References

**Internal**
`docs/K3-DEPLOYMENT.md` §2.1, §7.3, §7.4, §11.1, §11.2, §11.5 ·
`experiments/2026-09-19-throughput/FINDINGS.md` ·
`eval/runs/20260919T145955Z-baseline/`

**External**
- vLLM, *Efficient Decode Context Parallelism for Long Context Workloads* —
  <https://vllm.ai/blog/2026-08-07-decode-context-parallelism>
- vLLM issue #50682, *[ROCm][AMD] Kimi-K3 Gap and Roadmap Tracking* —
  <https://github.com/vllm-project/vllm/issues/50682>
- vLLM issue #52656, *OffloadingConnector crashes above 64 GB CPU RAM* —
  <https://github.com/vllm-project/vllm/issues/52656>
- vLLM, *KV offloading usage guide* —
  <https://docs.vllm.ai/en/stable/features/kv_offloading_usage/>
- LMSYS, *SGLang and Miles Add Day-0 Support for Kimi K3* —
  <https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support>
- SGLang, *Kimi-K3 cookbook (MI350X/MI355X recipe)* —
  <https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx>
- SGLang, *DeepSeek-V3 usage — DP attention* —
  <https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md>
- ROCm, *vLLM V1 performance optimization (MI300X/MI355X)* —
  <https://rocm.docs.amd.com/en/docs-7.1.0/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html>
- ROCm blogs, *4-bit KV caching in LMCache on MI355X* —
  <https://rocm.blogs.amd.com/software-tools-optimization/4bit-KV-LMcache/README.html>
