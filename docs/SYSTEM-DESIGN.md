# System Design v2 — Capacity-First Serving for Kimi-K3 on 8× MI355X

**Status:** proposed, unimplemented. Written 2026-09-20.
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

Two independent derivations agree. **This is the headline finding and G0 exists
to confirm or refute it on the live box before anything is built on it.**

### 2.3 Consequences

- **Break-even at ~250 tokens.** Below that, fixed KDA state exceeds the entire
  MLA KV cost. For the observed traffic distribution KDA is second-order — but
  once the MLA side is de-duplicated, KDA becomes the binding term and
  *concurrency*, not context length, is what limits the box.
- **`max-num-seqs: 512` was never absurd in ambition.** Against a de-duplicated
  ~19M-token pool, 512 sequences at a 30k mean is roughly the right order. It is
  wrong *today* only because replication divided the pool by 8. This predicts
  the setting comes back up after A1.
- **Sustainable concurrency today:** 524 GB ÷ (54 MB + 30k × 216 KB) ≈ **80
  sequences**. Observed steady state is 46–51 running with 22–25 queued ≈ 73 in
  system. The pool's real capacity is already asserting itself — through
  preemption rather than through admission.

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
  ┌───────────────┐   TLS, default-deny, per-key conn caps, usage.log
  │  nginx (keep) │   owns: transport + attribution
  └───────┬───────┘   owns NOT: anything token-aware
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

The engine owns *scheduling* — the gateway has no view of memory.
The gateway owns *tenancy* — the engine has no concept of a customer.
The backpressure service is the seam: it reads `gpu_cache_usage_perc`, the
running/waiting gauges and the preemption counter, and refuses classes when the
engine says it is in trouble. **It never computes a token budget of its own.**

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
| **A3** | CPU DRAM L2 tier | Offloads evicted blocks to host RAM | ~2× *addressable* (not resident) | Medium. Connector instability above 64 GB documented (vLLM #52656). | ~6 h |

A1 and A2 multiply. If both land, KV stops binding entirely and the design's
centre of gravity moves to compute. A3 is a different axis — it raises hit rate,
not concurrency — and is the fallback if A1 proves unavailable.

### Tier B — Waste: stop discarding completed work

| # | Optimization | Effect | Confidence | GPU |
|---|---|---|---|---|
| **B0** | `cache_salt` audit | Potentially the entire hit-rate collapse | High | **zero** |
| **B1** | Right-size `max-num-seqs` | Eliminates preemption. At 97.7% input, a preemption discards completed *prefill* — the most expensive thing on the box. | High | ~1 h |
| **B2** | Priority scheduling + classes | Protects interactive TTFT | High | ~2 h |
| **B3** | `long-prefill-token-threshold` | Caps the tail's scheduling share, breaking the eviction spiral | Medium-high | shared |

### Tier C — Compute: only binds after Tier A

| # | Optimization | Effect | Note |
|---|---|---|---|
| **C1** | AITER flag contract (`VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`) | Prevents a 2.06× decode regression presenting as an upgrade failure | Mandatory on any image change. Static-analysis result from `FINDINGS.md` §2. |
| **C2** | `max-num-batched-tokens` re-tune | Plausibly 10–20% TTFT | Currently 8192, tuned against a 12.8k-prompt benchmark. Observed mean is ~30k. Almost certainly mis-sized. |
| **C3** | A8W4 vs A4W4; KDA fused decode | ~1.2%; ~8.9% of decode-layer time | Real but decode-side, against a 2.3%-decode workload. Sub-1% end-to-end. |
| **C4** | DSpark / speculative decoding | Likely negative | Verification tokens compete under load; this workload is high-batch and 97.7% input. Defer. |

### Tier D — Traffic

| # | Optimization | Effect |
|---|---|---|
| **D2** | Per-class `max_tokens` clamp | **Recovers 11.5% of all traffic currently rejected with 400** (§3.4). Highest-certainty item on the register, zero GPU. Do this first. |
| **D1** | Route P1 short chat off-box | Removes request-count churn at zero HBM cost |
| **D3** | Distillation on a priority-3 loopback lane, off-peak | Near-free throughput; never touches customer quota |

### Tier E — Operational

| # | Optimization | Effect |
|---|---|---|
| **E1** | LiteLLM off-box fallback | Degraded quality instead of an outage during a spot reclaim |
| **E2** | Per-class SLO instrumentation | Marginal hit rate, per-class TTFT, KV occupancy by class, preemption rate |

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
| **Z1** | `cache_salt` audit from existing logs | May end the investigation. Costs nothing. |
| **Z2** | Static source verification: does DCP / DP-attention support a hybrid KDA+MLA model on ROCm today? | **Determines whether this is an 8× redesign or a 2× tuning exercise.** Pure code reading. Precedent: `validate.py` / `envprobe.py` caught the AITER rename statically before it cost a campaign. |
| **Z3** | Trace capture + replay harness | No engine A/B is meaningful without the real prompt distribution, prefix reuse, tool calls and concurrency. |
| **Z4** | Tenancy + backpressure layers against a mock endpoint | Entirely vendor- and model-agnostic. |
| **Z5** | Correctness gate extension for quantized-KV output quality | fp8/4-bit KV fails as *silent garbage* — the class of bug a throughput benchmark reports as success. |
| **Z6** | Pre-registered experiment definitions | Every GPU session gets a question, variants, metric, pass/fail and rollback, written in advance. |

### Phase G — GPU sessions, each time-boxed to one question

| Session | Question | Pass/fail | Est. |
|---|---|---|---|
| **G1** | Does right-sized `max-num-seqs` drive preemptions to zero on the current vLLM stack? | Preemption rate 0 across a full replay | ~2 h |
| **G2** | Does SGLang + `--enable-dp-attention` de-duplicate the KV pool for K3? | Reported pool grows toward ~19M tokens; correctness gate passes; **P0 TTFT p95 does not regress** | ~4 h |
| **G3** | Does fp8 KV double the pool again without corrupting output? | Both, or revert | ~3 h |
| **G4** | Is a host KV tier still needed after A1+A2? | Only run if G2/G3 leave KV binding | ~6 h |

**G0 is retired.** The replication question it was booked to answer was settled
offline by `redesign/capacity/hypothesis.py` — 3.2% error under the replicated
hypothesis versus 726% under the de-duplicated one. That is one GPU session
saved before the box booted.

G2 carries the migration, so it is the session to over-prepare: the trace
harness (Z3), the correctness gate extension (Z5) and the probe
(`redesign/probe`) must all be green first.

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
| CPU offload connector instability at scale (vLLM #52656) | Medium | Ramp 64 → 128 → 256 → 512 GB with a soak per step; first instability is the ceiling. `SimpleCPUOffloadConnector` over `OffloadingConnector`. |
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
