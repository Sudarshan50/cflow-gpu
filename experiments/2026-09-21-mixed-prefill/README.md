# Mixed long-context prefill/decode experiment

**Result:** retain 4096. Completed c=4/c=8 comparisons showed longer mixed
completion times and lower throughput with 2048; the candidate c=16 comparison
did not qualify. The separate context audit reproduced two false-positive
rejections. See `RESULTS.md` for results, limitations and restored serving state.

Source baseline: `9f951e066358870466ce9875da63b174d5818b95`.
Pinned AMD image: `sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8`.

The operator approved temporarily disabling channel 285, draining active work,
running isolated tests, and restoring service with the proxy concurrency limit
of 42 preserved. The earlier four-optimization branch remains separate.

## Hypothesis and decision criteria

Reducing the global prefill budget from 4096 to 2048 may shorten decode stalls
during cold, long-context bursts. It may also increase cold-request latency and
reduce throughput, so cache-hit fraction alone is not a success criterion.

Evaluate streaming gap tails, warm-request completion times, cold-request TTFT,
and total completed-output throughput. A useful default should improve mixed
latency materially without a substantial throughput/cold-latency regression.
Correctness, complete streams, intended cache states, and exact engine-work
accounting are prerequisites. No global-optimality claim follows from this
bounded workload.

## Controls

- Same target revision, image, TP8/EP-off settings, tuning environment and 262144
  context; DSpark remains off and prefix matching remains the baseline default.
- Baseline's existing pool is 2177 blocks. Pin the candidate to the same effective
  2177 blocks with `num-gpu-blocks-override: 2177`; this controls auto-profiling
  capacity variation. The candidate also sets `max-num-batched-tokens: 2048`.
- The runner checks the on-disk profile, effective cache geometry and startup
  log's loaded scheduler budget. Both arms have the same effective KV capacity.
- Synthetic deterministic inputs with matching token hashes, fresh cache salts,
  and direct engine access. No customer prompt capture or response-cache replay.
- 49244-token warm inputs; 131164-token cold inputs.
- Concurrency 4, 8 and 16; two rounds per warm/mixed case. A mixed round injects
  two cold requests only after all warm streams have produced content.
- Warm streams generate exactly 256 tokens; cold requests exactly 32. EOS is
  ignored only for fixed-work performance measurement. This deliberately fixed
  output workload is separate from correctness tests.
- Eight pre-four smoke checks cover arithmetic, retrieval, JSON, a forced tool
  call and one tiny image. Two additional 133k retrieval checks cover cold/repeat
  behavior. Their answers must finish normally.

The 2048-token budget may align to 1536-token chunks on the 768-token hybrid
grid; the effective workload, not the nominal setting alone, matters.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/2026-09-21-mixed-prefill -p 'test_*.py' -v

# With the baseline loaded and traffic drained:
python3 experiments/2026-09-21-mixed-prefill/bench.py \
  --variant baseline --cache-blocks 2177 --out /tmp/opencode/mixed-baseline.json

# After loading the candidate described above:
python3 experiments/2026-09-21-mixed-prefill/bench.py \
  --variant batch2048 --cache-blocks 2177 --out /tmp/opencode/mixed-batch2048.json
```

Every output filename must be new. Failed or contaminated phases remain recorded
as failures. Keep the channel disabled until the selected serving profile has
been reloaded and verified, then restore its previously enabled state.

`compare.py --completed-only` explicitly reports only fully paired completed
concurrency cohorts, preserves source-trial failures and identifies excluded
concurrencies. Its default mode requires a complete, successful trial.

`context_audit.py` compares the baseline policy with actual `/tokenize` counts on
synthetic text/tool schemas. Run it between performance phases so its CPU work
does not affect the latency measurements.
