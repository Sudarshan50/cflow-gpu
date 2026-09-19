# Z2 — Is MLA KV de-duplication available for Kimi-K3 on ROCm today?

**Date:** 2026-09-20 · **Method:** static, upstream documentation and issue
trackers. No GPU, no engine started.
**Question:** `docs/SYSTEM-DESIGN.md` A1 is worth up to **8× the KV pool** and is
the only item on the optimisation register with that magnitude. Is it
collectable, on which engine, and at what cost?

---

## Answer

**Yes on SGLang. No on vLLM.**

And that inverts the sequencing in the original design, which placed the engine
decision third. The 8× lever and the engine choice are **the same decision**.

---

## 1. vLLM — DCP is blocked three ways

vLLM's mechanism is Decode Context Parallelism: `--decode-context-parallel-size`,
which shards the MLA latent along the token dimension instead of replicating it
per rank. The constraint arithmetic works for us — for MLA,
`tp_size >= dcp_size` and `tp_size % dcp_size == 0`, so TP=8 admits dcp up to 8,
exactly the factor we need.

It is nonetheless unavailable:

| Blocker | Evidence |
|---|---|
| **Hybrid models unsupported** | The DCP announcement lists "integrating it with hybrid models" under **future work**. K3 is 69 KDA + 24 MLA layers. |
| **Kimi-K3 not shipped** | Same source: DCP supports DeepSeek-V2/V3/R1, Qwen3-235B, Llama-family, with work "underway for GLM-5.2 and Kimi K3". |
| **ROCm never mentioned** | All DCP benchmarks are 8× **NVIDIA B200**. The only ROCm DCP work is RFC #57228, scoped to DeepSeek-V4, not K3. |

Separately, vLLM issue #50682 (*[ROCm][AMD] Kimi-K3 Gap and Roadmap Tracking*)
lists **fp8 KV cache as in-progress and not working on MI355X** — so A2 is
blocked on vLLM as well, with PRs #51040, #51011 and #50619 open.

**Net: on vLLM/ROCm today, neither A1 nor A2 is collectable. The ceiling is A3
alone — roughly 2× addressable, not resident.**

## 2. SGLang — DP attention is documented for K3, and exercised on hybrids

SGLang's mechanism is different and, for this architecture, better suited.
Rather than sharding one sequence's KV across ranks (DCP), **DP attention
partitions whole sequences across attention-DP ranks.** Each rank holds the KV
for the requests it owns, so nothing is replicated.

That distinction matters for a hybrid model: both the KDA recurrent state
(54 MiB, per sequence) and the MLA KV (27 KiB/token, per sequence) are
*per-sequence* quantities, so they follow the sequence to its DP rank naturally.
DCP, which splits a single sequence's KV across ranks, has to solve the harder
problem of splitting a recurrent state that is inherently sequential — which is
plausibly why hybrid support is still future work there.

| Evidence | Source |
|---|---|
| `--enable-dp-attention` and `--dp-size` documented **in the Kimi-K3 recipe itself** | SGLang K3 cookbook |
| DP attention + hybrid is a **live, exercised path** | PR #34535, *"Fix hybrid-SSM DP attention failures at low concurrency"* — you do not fix bugs in an unsupported combination |
| KDA is a first-class kernel/layer/backend in SGLang | KDA components landed for `KimiLinearForCausalLM`, reused by later Kimi models |
| Hybrid memory is a designed subsystem | Dual-pool design, hybrid radix tree for prefix caching, state transfer channel for PD |
| fp8 KV (A2) supported | `fp8_e4m3` documented in the K3 recipe |
| HiCache (A3) supported | `--enable-hierarchical-cache`, `--hicache-ratio`, `--hicache-storage-backend` |

**Net: on SGLang, A1 + A2 + A3 are all reachable. Modelled ceiling ~16×
resident (A1×A2), ~31× addressable with A3.**

---

## 3. Caveats that must shape the migration

These are not reasons against SGLang. They are the things that will bite during
G2/G3 if they are not designed for in advance.

### 3.1 `max-running-requests` is floor-divided by `dp_size`

The K3 cookbook states the concurrency ceiling is *"server-wide and floor-divided
by `attn_dp_size`"*. PR #34535 exists because that floor division could produce
**"Hybrid (mamba/linear-attention) state cache is too small to serve any
requests"** at low concurrency.

This interacts directly with **B1**. Set `--max-running-requests` to the
*server-wide* figure from `capacity_model.py`, not the per-rank figure, and
verify the post-division value is still above the hybrid pool's minimum. Getting
this backwards produces a server that refuses everything, and it will look like
a migration failure rather than a config error.

### 3.2 CP and DP attention are mutually exclusive today

`--cp-strategy interleave` asserts `dp_size == 1`. So prefill context
parallelism (`--enable-prefill-cp`, `--attn-cp-size`) and DP attention cannot
currently be combined. **Pick DP attention** — it is the one that de-duplicates
the resident KV, which is our binding constraint. Prefill CP addresses prefill
latency, which is not what is failing.

### 3.3 The hybrid pool split may be static

SGLang's general hybrid support uses a **physically isolated dual-pool** design
with the split set statically by `--mamba-full-memory-ratio`. That is precisely
the "guess the workload composition up front" problem that the K3 day-0 writeup
claims to have solved with a *unified* pool filling KDA from one end and MLA
blocks from the other.

**Unresolved:** whether K3 on the current build gets the unified pool or the
static dual pool. It matters — a static split mis-set is a self-inflicted
capacity ceiling, and `capacity_model.py`'s numbers assume a single pool.
`probe_dedup_support.py` checks for `mamba_full_memory_ratio` in the CLI surface
for exactly this reason. **Resolve before G2.**

### 3.4 DP attention is a throughput optimisation, not a latency one

Upstream guidance: *"not recommended for low-latency, small-batch use cases;
optimized for high-throughput scenarios with large batch sizes."* This workload
is high-batch (46–51 running, 22–25 queued), so the fit is right — but **P0
interactive TTFT is the metric to watch during G2**, not aggregate throughput.
If P0 p95 regresses, that is the tradeoff appearing, and it is the one thing
that could make A1 not worth taking.

### 3.5 Correctness surface changes wholesale

Migrating means changing the engine, the MoE kernel path (FlyDSL A8W4), the
attention backend (Triton MLA), the KV dtype (fp8), and the parallelism layout
— **simultaneously**. The three-tier correctness gate was recorded against vLLM
behaviour; its baseline almost certainly needs re-recording, which weakens it at
exactly the moment it matters most. §7.3's silent-garbage failure is the class
of bug that a throughput benchmark reports as a success.

**Z5 exists to address this and is now a hard prerequisite for G3.**

---

## 4. What this changes in the plan

| Was | Now |
|---|---|
| Engine decision sequenced third, after capacity work | **Engine decision *is* the capacity work.** A1 requires SGLang. |
| "Expected outcome: SGLang, on fp8 KV and unified memory" | Confirmed, and for a bigger reason than stated: **A1 is ~8×, four times larger than the fp8 argument.** |
| G4 (engine A/B) last | G4 folds into G2. There is no A/B — there is a migration, measured against the vLLM baseline. |
| A3 (CPU tier) as lever #2 | Demoted. A1 is 8× resident; A3 is ~2× addressable. Do A1 first, and A3 may prove unnecessary. |

**B1 still goes first**, and that is unchanged. Reasons:

1. It is free and engine-independent.
2. Without it, the migration cannot be measured — preemption noise will dominate
   whatever the engine difference is.
3. SGLang has its own version of the same hazard (§3.1), so the sizing work
   transfers directly and you want to understand it *before* you meet it under
   a floor division.

---

## 5. Revised order

```
Z1  cache_salt audit                          zero GPU, may end the investigation
Z3  trace capture and replay                  zero GPU, gates all measurement
Z5  correctness gate for quantized KV         zero GPU, gates G3
G1  B1 on vLLM: right-size max-num-seqs       ~2 h, proves the measurement loop
G2  SGLang migration + DP attention (A1)      ~4 h, THE main event
G3  fp8 KV on SGLang (A2)                     ~3 h, stacks to ~16x
G4  HiCache L2 (A3), only if still needed     ~6 h, likely unnecessary after A1
```

---

## 6. Confidence and how to falsify

Everything above is **static evidence: documentation, release notes, issue
trackers and merged PR titles.** None of it has been executed. The specific ways
it could be wrong:

- The K3 cookbook may document `--enable-dp-attention` generically without it
  being validated for K3's KDA layers.
- PR #34535 proves hybrid-SSM DP attention exists; it does not prove *KDA*
  specifically is covered, and KDA is not the same as Mamba-style SSM.
- ROCm support for DP attention is inferred from the recipe being an MI35x
  recipe, not stated directly for the DP-attention path.

`probe_dedup_support.py` closes the first of these cheaply — run it inside the
pinned SGLang MI35x image before booking G2. The other two need G2 itself.

**Falsifier:** if `probe_dedup_support.py` reports `enable_dp_attention` absent,
or G2 shows the pool unchanged with DP attention enabled, this finding is wrong
and the plan reverts to A2+A3 on vLLM with a ~2× ceiling.

---

## 7. Sources

- vLLM, *Efficient Decode Context Parallelism for Long Context Workloads* — <https://vllm.ai/blog/2026-08-07-decode-context-parallelism>
- vLLM, *Context parallel deployment* — <https://github.com/vllm-project/vllm/blob/main/docs/serving/context_parallel_deployment.md>
- vLLM issue #50682, *[ROCm][AMD] Kimi-K3 Gap and Roadmap Tracking* — <https://github.com/vllm-project/vllm/issues/50682>
- vLLM issue #57228, *[RFC][ROCm] DCP for DeepSeek-V4* — <https://github.com/vllm-project/vllm/issues/57228>
- SGLang, *Kimi-K3 cookbook* — <https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3>
- SGLang PR #34535, *Fix hybrid-SSM DP attention failures at low concurrency* — <https://github.com/sgl-project/sglang/pull/34535>
- SGLang, *DeepSeek-V3 usage — DP attention* — <https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md>
- PyTorch blog, *Hybrid Models Meet SGLang: More than Full Attention* — <https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/>
- LMSYS, *SGLang and Miles Add Day-0 Support for Kimi K3* — <https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support>
