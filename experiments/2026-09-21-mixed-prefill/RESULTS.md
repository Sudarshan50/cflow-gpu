# Further optimization tests — 2026-09-21

## Decision

**Retain `max-num-batched-tokens: 4096`.** The 2048 candidate reduced some
individual streaming gaps but made the completed mixed workloads take longer:
warm-request completion regressed 33–36%, cold TTFT regressed about 55–56%, and
aggregate output throughput fell 25–26%.

The broader service is **not established as optimal**. A separate tokenizer
experiment confirmed false-positive context rejections, and live traffic still
frequently hit output limits. Those are better targets than adopting this
smaller global prefill budget.

Service resumed on the baseline profile at **12:04 UTC**. Channel 285 is enabled;
the existing **42-request proxy concurrency limit** is preserved. A fresh public
endpoint inference check passed at **12:06:58 UTC**.

## Traffic before testing

The 42-request limit was verified from the active LiteLLM key configuration.
In the 10:20:21–10:21:21 live window:

- Mean 4.5 running requests, maximum six; zero queue and zero preemptions.
- Mean active KV usage 27.5%, maximum 40.7%.
- 69.7 generated tokens/s aggregate, about twelve completed requests/minute.
- Mean engine TTFT 2.23 seconds, with a long-prefill tail.
- Seven of twelve completed requests ended at the output-length limit.

The preceding five-minute completed-request cohort contained 97 non-replayed
successes, median 46823 input tokens, p95 102613 input tokens, 86.61% cached input,
median 9.22-second duration and 27.57 generated tokens/s after the first proxy
completion event. Seven rejected attempts shared a heuristic estimate of 611024
input tokens. These observations reflect the current workload, not a controlled
32-versus-42 concurrency experiment.

## Test design

- Pre-four-optimization source checkpoint `9f951e0`; the archived four-step work
  was not reintroduced.
- Same AMD image `sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8`,
  target, TP8, EP off, DSpark off, BF16/auto KV and default prefix matching.
- Equal effective KV capacity: 2177 blocks / 1639906 reported tokens. The
  candidate was explicitly pinned to the baseline's existing allocation.
- Deterministic synthetic 49244-token warm contexts and 131164-token cold
  contexts, verified with the actual `/tokenize` endpoint.
- Concurrency 4, 8 and 16. Mixed rounds inject two cold requests after every warm
  stream has emitted content. Warm/cold outputs are fixed at 256/32 tokens.
- Two measured rounds per scenario, following cache priming and a decode warmup.
- Complete SSE, exact token budgets, intended cached-token counts, quiet phase
  boundaries and reconciled engine output/completion counters were required.
- Ten correctness checks passed on each profile: arithmetic, JSON, a forced tool
  call, one tiny image, retrieval, and cold/repeated 133k retrieval.

Fixed-length performance outputs are intentionally separate from correctness
checks. Only 3 of the 56 included performance output hashes matched exactly
across profiles. Inputs, output-token counts and cache states matched, but these
measurements do not establish bitwise generation equivalence or broad model
quality equivalence.

## Completed mixed-workload comparisons

`comparison-completed-phases.json` explicitly contains only the completed paired
4- and 8-stream cohorts: **56 matched-work request pairs**. It retains the failed
candidate-trial status and excludes concurrency 16.

| Metric | c=4, 4096 | c=4, 2048 | c=8, 4096 | c=8, 2048 |
|---|---:|---:|---:|---:|
| Warm-request median duration | 31.76 s | **43.15 s** | 32.77 s | **43.51 s** |
| Warm per-request rate after first token | 8.16 tok/s | **5.98 tok/s** | 7.91 tok/s | **5.94 tok/s** |
| Cold median TTFT | 19.44 s | **30.21 s** | 19.53 s | **30.55 s** |
| Aggregate output throughput | 34.23 tok/s | **25.20 tok/s** | 64.39 tok/s | **48.51 tok/s** |
| Median of per-stream p95 visible gaps | 372 ms | 258 ms | 378 ms | 291 ms |

The smaller budget improves the upper tail of individual gaps while introducing
more slow prefill iterations. That is an unfavorable trade for completion time
and throughput in these mixed workloads. A lower p95 gap alone would have given
a misleading optimization verdict.

## Warm-only behavior and concurrency

Under the baseline, the all-warm workloads delivered approximately:

| Concurrent warm streams | Per-request output rate | Aggregate output rate |
|---:|---:|---:|
| 4 | 32 tok/s | 121 tok/s |
| 8 | 32 tok/s | 242 tok/s |
| 16 | 25 tok/s | 384 tok/s |

Adding two cold long-context requests reduced baseline warm-stream rates to
roughly 8 tok/s across these concurrency levels, despite no preemptions. This
demonstrates prefill interference independently of KV exhaustion.

The 2048 warm-only candidate improved c=4 aggregate throughput by 13.6% in this
small comparison, while c=8 was essentially unchanged (-0.6%). Both warm-only
and mixed results must be considered; these measurements do not establish a
universal best concurrency ceiling or justify changing the user's 42 limit.

## Interrupted c=16 qualification

The first candidate c=16 warmup was invalidated by an unrelated 81-token request:
the engine reported 1105 generated tokens and seventeen completions instead of
1024 tokens and sixteen completions. Channel 285 had been re-enabled externally.
The original failed report was retained.

After additional operator approval, channel and gateway routing were paused.
The resumed c=16 attempt passed workload-accounting checks but failed the fully
warmed-prefix precondition. It was not scored as a valid latency comparison.
The cause of the missing expected warm hit was not established. Both failed
attempts remain available; no successful c=16 candidate result is claimed.

The completed-phase comparison checks the earlier valid phases independently
and is explicitly marked partial. The normal full-comparison mode continues to
reject incomplete/failed trials.

## Confirmed context-counting improvement opportunity

Six synthetic tokenization checks compared the current heuristic with the
engine's rendered token counts. No generation requests were used for this audit.

| Input | Heuristic count | Actual rendered count | Current policy |
|---|---:|---:|---|
| English reference text | 15727 | 11026 | Accept |
| Indented Python | **300018** | **150030** | **Reject incorrectly** |
| JSON lines | 50013 | 75026 | Accept; underestimates by 33% |
| CJK reference text | 48015 | 30027 | Accept |
| Emoji sequences | 65727 | 140025 | Accept; underestimates by 53% |
| Unicode tool documentation | **288055** | **84094** | **Reject incorrectly** |

The two rejected examples fit the 262144-token window with both a 512-token
output allowance and the current 256-token reserve. Actual tokenization took
about 110 ms and 30 ms respectively on this node. This is direct evidence that
engine-aligned counting can improve valid-request acceptance. It also shows the
heuristic is not consistently conservative.

Production request bodies are not retained, so this does not establish how many
of the observed live rejections were false positives. The tested alternative
remains an experiment; the production counting policy was preserved.

## Next priorities

1. Use actual rendered token counts for context acceptance/clamping, particularly
   near the limit, while preserving payload/model identity between counting and
   inference. Retain inexpensive estimates only where their uncertainty is
   acceptable.
2. Budget cold/long-context concurrency against KV headroom and an interactive
   latency target. The mixed test shows the impact of cold prefills even before
   memory is exhausted.
3. Review the 512-token agentic ceiling using representative tool-call completion
   and reasoning requirements. Live length finishes show the cap is binding;
   they do not prove every capped response is a failure.

## Restoration and verification

- Baseline engine restarted **12:00:28 UTC**, gateway restored **12:04:29 UTC**.
- At 12:04, all 100 installed serving files matched the baseline archive; only
  the baseline service overrides were active.
- Effective batch budget is 4096, default prefix matching, automatic cache
  sizing; the experimental block override was removed.
- Two fresh baseline cache/answer probes passed, with 0 then 768 cached tokens.
- Engine, gateway and LiteLLM health/readiness returned HTTP 200.
- Channel 285 was enabled after verification. Its then-current weight/priority
  were preserved, including an independently changed weight of 4.
- The active proxy concurrency setting was rechecked as 42.
- A fresh authenticated public-endpoint request returned HTTP 200, the expected
  answer and a normal stop at 12:06:58 UTC.
- Five CPU stream/accounting contract tests passed.

Raw evidence is under `/tmp/opencode/`:

- `traffic-42-live-20260921T1020.json`, `traffic-42-db-20260921T1020.json`
- `mixed-baseline-20260921.json`
- `mixed-batch2048-20260921.json` (interrupted original trial)
- `mixed-batch2048-complete-20260921.json` (failed c=16 resumption)
- `mixed-context-audit-20260921.json`
- `mixed-final-smoke-20260921.json`, `mixed-final-state-20260921.json`
- `mixed-proxy-smoke-20260921.json`
