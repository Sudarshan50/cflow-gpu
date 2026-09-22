# Kimi-K3 gfx950 AITER GEMM campaign

This experiment keeps the qualified Kimi-K3 model, vLLM configuration and
gateway policy fixed while testing ROCm/AITER BF16 GEMM dispatch on 8× MI355X.

## Invariants

- Production base image: digest `5f3007aff1bc…`.
- Candidate AITER revision: `5337bf2cdb88`, containing ROCm/AITER PR 5702.
- Model snapshot and `config-base.yaml` do not change.
- `AITER_SITUV2_A8W4=1` and `AITER_BF16_FP8_MOE_BOUND=0` remain paired.
- Never edit `/tmp/aiter_configs`; it is generated at process import.
- Candidate rows enter through `AITER_CONFIG_GEMM_BF16`.

## Traffic boundary

Image construction and passive metric capture are safe while traffic is live.
Before profiling, tuning, or restarting the model:

1. Disable NewAPI traffic.
2. Wait for gateway queued requests and engine running/waiting requests to reach
   zero.
3. Stop `k3-traffic.timer`, `k3-gateway`, and `k3-litellm`.

Only restore traffic after either the candidate passes every offline gate or the
qualified base image has been restored and smoke-tested.

## Runtime artifacts

Artifacts live under `/scratch/deploy-state/amd-optimized/gemm/`:

- `live-baseline-*/`: production metric windows.
- `profiles/`: rocprofv3 summaries.
- `shapes/`: captured untuned GEMM rows.
- `bf16-candidate-complete.csv`: materialized complete dispatch table.
- `cache/`: candidate-only AITER JIT cache.
- qualification and comparison JSON.

## Candidate selection

The trial systemd drop-in is
`systemd/65-gemm-tuning-candidate.conf`. It is copied to
`/etc/systemd/system/k3.service.d/` only for the isolated trial. Removing that
one file and restarting `k3` restores the qualified `60-amd-optimized.conf`.

Rows are accepted only when:

1. Production dispatch passes numerical validation (`err_ratio <= 0.05`).
2. Median kernel time improves by at least 3% over fallback across repeated
   measurements, or Torch is explicitly retained as the winner.
3. There are no duplicate dispatch keys or unrecognized kernels.
4. End-to-end correctness passes.
5. Aggregate concurrency-1 and concurrency-16 throughput does not regress by
   more than 2%.

## Required gates

- Eight-check `qualify.py` smoke and isolated c1/c16 workload.
- Tier-1 and tier-3 corruption gates.
- Forced tool call, structured output, image, and long-context retrieval.
- Mixed cold-prefill/active-decode benchmark.
- Prefix-cache accounting, zero preemptions, and complete stream accounting.
- Matched live canary before promotion.
