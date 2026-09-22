# KDA prefix caching — measured investigation

2026-09-21 UTC; eight MI355X GPUs; pinned AMD optimized plain Kimi K3.

## Decision

**Keep the baseline engine matching configuration.** The tested 128-token
override improved tail reuse but added a prefill step and increased cold/branching
first-token latency by 27–31%. Its small replay savings do not justify that
tradeoff on these fixtures. The original profile is selected for deployment;
the fixed-capacity control is an experiment setting only.

The strongest demonstrated cache improvement is the application-owned prompt
layout: stable text first, changing runtime context afterward.

## Scope

The Git checkout and installed control-plane code were restored to
`9f951e066358870466ce9875da63b174d5818b95` before this investigation. The subsequent
four-optimization work is preserved on `archive/four-optimizations-20260921` at
`c28434d666468ef7b9d0d20d128767cdecf41f9a`.

The engine image is
`johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8`.
TP8, EP off, DSpark off, context 262144, 64 sequences, 4096 batch tokens,
auto/BF16 KV and 0.88 memory utilization were retained. Physical cache blocks
are 768 tokens, with Mamba `align` mode.

## 1. Moving the changing date after stable text works

`baseline.json` contains three dates in each layout. The date-aware task returned
the correct requested date in all six requests.

| Baseline layout, changed-date followups | Common rendered prefix | Actual reused tokens | Reused fraction |
|---|---:|---:|---:|
| Date before stable text | 18 | 0 / 8795 | 0% |
| Stable text before date | 8756 | 8448 / 8794 | 96.1% |

The two changed-date followups had median first-token times of approximately
0.940 s and 0.206 s respectively. This is an observation from the small layout
probe, not a production throughput result.

With `prefix-match-unit=128`, the stable-first followups reused **8704 / 8792
tokens (99.0%)**; date-first still reused zero. All six candidate date answers
were correct. The exploratory runs use different synthetic UUIDs, which account
for their slightly different token counts. The later latency benchmark uses
identical token IDs in both arms.

The mechanism is causal prefix identity: block hashes include their parent
hashes. Moving the volatile field later preserves more of the preceding model
state. Reversing cache-search direction cannot make a suffix independent of a
changed prefix. The installed Mamba lookup already searches right to left.

## 2. A matching token prefix can lack a KDA checkpoint

Each sibling fixture is 7675 tokens and changes one token at the specified
boundary. The observed cached-token sequences were identical in both settings:

| Divergence position | Request 1 → 2 → 3 cached tokens |
|---|---|
| 2304 | **0 → 0 → 2304** |
| 3840 | **0 → 3840 → 3840** |
| 4608 | **0 → 3840 → 4608** |

The default 4096-token budget produces a natural checkpoint at 3840. Other
interior checkpoints are materialized when the engine discovers a shared-prefix
junction during recomputation. Finer matching does not create all these states
in advance.

An explicit **2305-token priming request** made **2304 tokens reusable by the
first subsequent full-length sibling** in both arms. This demonstrates targeted
checkpoint placement using the existing engine. It also requires an additional
prefill request; the priming cost must be included when evaluating adoption.

## 3. Finer matching helps some tails, with boundary exceptions

| Exact prompt length | Baseline cached tokens, requests 1 → 2 → 3 | 128-token matching |
|---|---|---|
| 988 | 0 → 768 → 768 | **0 → 896 → 896** |
| 1024 | 0 → 768 → 768 | 0 → 768 → 768 |
| 7680 | 0 → 3840 → 6912 | 0 → 3840 → 6912 |

Replay must leave a token to compute logits. A checkpoint at the entire prompt
length therefore need not be usable for that identical prompt. The tested build
also does not retain an arbitrary earlier partial checkpoint just because a
finer matching unit is enabled.

Salt controls reused **0 → 1536 → 0** tokens for same-salt seed/repeat followed
by a changed salt. A stable token prefix does not override namespace isolation.

Tool-rendering checks found that dictionary-key permutations retained identical
token IDs, reversing the tool array reduced its common prefix, and the existing
baseline normalizer restored canonical order. These were tokenizer checks,
separate from the GPU cache measurements.

## 4. Correctness of partial-tail reuse

`tail128-validation-isolated.json` passed **16/16** checks:

- 988- and **133468-token** prompts.
- Retrieval of an earlier archive code plus a changing runtime code.
- Three independently salted cold references per length.
- Shared-cache sequence **A, B, A, C, B**, including return to previous branches.
- Exact output hashes equal to the corresponding cold reference, correct values,
  `finish_reason=stop`, complete streams and reconciled engine counters.
- The four cache-hit branches at each length reused **896** and **133376** tokens
  respectively. Each answer generated 19 tokens.

The first validation attempt, `tail128-validation.json`, is deliberately retained
as **invalid/incomplete**. Other traffic entered the engine at approximately
08:17 UTC; channel 285 had been re-enabled externally. The counter check stopped
the run. After the user authorized disabling it again, the channel was verified
disabled at **08:20:32 UTC**, with routing settings unchanged, and validation was
restarted using fresh synthetic namespaces.

## 5. Timing methodology and rejected initial comparison

`latency.py` runs six fixed-input cases, three warmups and eight measured requests
per case: **66 requests per arm**, including 48 measured requests. It checks
intended cache states and exact model inputs. `compare.py` additionally requires
matching output hashes, complete case coverage, attributable engine work, equal
cache geometry/capacity and configuration differences limited to the matching
unit.

The first timing runs are retained as `baseline-latency.json` and
`tail128-latency.json`. Their strict pair comparison was **rejected** because
automatic sizing produced **2176 versus 2177 cache blocks** (1639153 versus
1639906 reported tokens). Those runs do not serve as the qualified matched pair.

The final pair fixes **2176 blocks in both arms** with the common experimental
control `num-gpu-blocks-override: 2176`. The runner verifies this control is
actually loaded. The pair otherwise differs only by `prefix-match-unit: 128`.

The equal-capacity comparison **passed**, including exact prompt and output
hashes for all **66 request pairs**. Both arms reported 1639153 cache tokens.
Results are in `comparison.json`, derived from
`baseline-latency-fixed2176.json` and `tail128-latency-fixed2176.json`.

| Case | Cached tokens, baseline → 128 | Baseline median TTFT | 128 median TTFT | Change |
|---|---:|---:|---:|---:|
| Cold 7675-token prefill | 0 → 0 | 0.700 s | 0.890 s | **+27.2%** |
| Warm 2304-prefix sibling | 2304 → 2304 | 0.660 s | 0.864 s | **+30.9%** |
| Exact 988-token replay | 768 → 896 | 0.211 s | 0.208 s | −1.6% |
| Exact 1024-token replay | 768 → 768 | 0.208 s | 0.202 s | −2.7% |
| Exact 7680-token replay | 6912 → 6912 | 0.196 s | 0.204 s | +4.2% |
| Stable-first 8796-token Chat | 8448 → 8704 | 0.200 s | 0.212 s | +6.3% |

Positive changes mean slower first-token delivery. The 988-token replay saved
only **3.35 ms** despite reusing 128 more tokens. The stable-first Chat case saved
256 input-token computations but took **12.49 ms longer** to deliver its first
token. Results are point estimates from eight measured requests per case; small
millisecond differences are not statistical proof of a generalized improvement.
The JSON includes observed minima/maxima, including occasional latency outliers.

## 6. Why additional reuse can cost more time

`scheduler-trace.json` executes the reviewed text-prefill splitting function from
the pinned engine, without importing vLLM or modifying its source.

| Text-prefill case | Baseline chunk ends | 128-token matching chunk ends |
|---|---|---|
| Cold 7675 tokens | 3840, 6912, 7675 | 3840, 6912, **7552**, 7675 |
| Warm 2304-prefix sibling | 6144, 6912, 7675 | 6144, 6912, **7552**, 7675 |
| Cold 988 tokens | 768, 988 | 768, **896**, 988 |
| Warm 988-token replay | 988 | 988 |
| Cold 8796-token Chat | 3840, 7680, 8448, 8796 | 3840, 7680, 8448, **8704**, 8796 |

Saving the partial-tail state requires an extra prefill step in these cold or
branching cases. A warmed replay still takes one step with either setting.
Increasing cached-token fraction therefore does not imply a proportional
latency improvement. This trace explains scheduling; GPU timings are measured
separately.

## Application implications

1. Put stable instructions/reference material before volatile runtime fields in
   application-owned templates, preserving message roles and answer semantics.
2. Keep tool declarations and namespace identity stable when their meanings are
   stable. The baseline tool canonicalization is already effective.
3. For frequently reused, known shared boundaries, consider explicit priming or
   targeted checkpoint placement. Measure its upfront cost against subsequent
   reuse. The 2304-token priming probe establishes feasibility, not a broad
   workload speedup.
4. Evaluate cache settings using complete, correct responses and actual cached
   tokens together with latency. A high hit fraction alone is insufficient.

## Final deployment verification

The baseline engine restarted at **08:48:50 UTC**. At **08:52 UTC**,
`final-smoke.json` and `final-state.json` verified:

- Git branch `redesign/capacity-first`, HEAD `9f951e0`.
- All **100 installed control-plane files**, including the complete file set,
  match the baseline Git archive; only the baseline `60-amd-optimized.conf`
  service overrides are active.
- The engine profile equals the committed baseline. Both experimental overrides
  were removed; effective `prefix_match_unit=None` and
  `num_gpu_blocks_override=None`.
- Auto-sized cache: **2177 blocks / 1639906 reported tokens**; physical block
  size 768 and Mamba `align` mode.
- Two fresh 988-token archive/runtime-code checks returned the correct values
  with complete streams, normal termination and exact engine-counter accounting.
  The second request reused **768 tokens**, confirming restored baseline behavior.
- Engine, gateway and LiteLLM health/readiness returned **HTTP 200**.
- Gateway/LiteLLM images are `k3-python:24.04`, response-cache revision is
  `amd-5f3007-base-v1`, and the engine was idle at final verification.
- NewAPI **channel 285 remains disabled (`status=2`)**.

**11 CPU measurement/comparison contract tests passed.** The source-based
scheduler trace covers 12 cases. The runtime decision is supported by the
equal-capacity measurements, the 16-request partial-cache correctness suite and
the post-restart smoke checks. The archive branch retains the four-optimization
work independently of this experiment.
