# KDA prefix-cache investigation

This experiment starts from `redesign/capacity-first` at
`9f951e066358870466ce9875da63b174d5818b95`. The later four-optimization work is
preserved on `archive/four-optimizations-20260921` and is outside this study.

**Result:** retain the baseline engine profile. Finer matching improved tail
reuse but increased cold/branching first-token latency by 27–31% in the
equal-capacity comparison. Stable-first prompt layout produced the largest
demonstrated reuse gain. See `RESULTS.md` for measurements and final verification.

## Runtime

- Eight MI355X GPUs, TP8, EP off, DSpark off.
- Image `johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8`.
- Kimi K3 revision `f831ab66814297da540d832a5235f8e904f29d06`.
- Context 262144, 64 sequences, 4096 batch tokens, memory utilization 0.88,
  auto/BF16 KV, prefix caching and priority scheduling.
- Physical hybrid block size 768; Mamba cache mode `align`.
- Active profile: `/scratch/deploy-state/amd-optimized/config-base.yaml`.
- Baseline: no `prefix-match-unit` override. Candidate: add only
  `prefix-match-unit: 128`, followed by an engine restart.

The runners check the checkout, image, profile, effective `/metrics` cache
configuration and container identity. Changing YAML without restarting does not
qualify as a candidate run. The control-plane source remains at the baseline.

## What is being measured

1. **Causal prefix identity.** Moving a changing date after an application-owned
   stable prompt increases the common rendered token prefix. Both layouts must
   still return the requested date.
2. **Checkpoint materialization.** Controlled token IDs place sibling divergence
   exactly at 2304, 3840 or 4608. Repeating three siblings distinguishes an
   immediately reusable checkpoint from one created during the second request.
3. **Partial-tail matching.** Exact 988-, 1024- and 7680-token replays exercise
   fine-grained matching and the final-token replay limit.
4. **Explicit boundary priming.** A 2305-token seed creates the checkpoint at
   2304 before the first full-length sibling. The seed consumes model work and
   is a separate request.
5. **Cache namespace controls.** Identical prompts with a new synthetic salt
   must be cold. Real tenant isolation remains part of cache identity.
6. **Tool rendering.** Dictionary and tool-array permutations are tokenized;
   token-prefix equality is reported separately from actual GPU reuse.

Every inference request goes directly to `127.0.0.1:8001`. All input is synthetic
text. Responses require a finish reason, terminal usage and SSE `[DONE]`.
Inference prompt counts must equal `/tokenize` counts. Each probe requires an
idle engine and reconciles its output/completion counters with engine metrics.

### Correctness checks

`validate.py` retrieves an earlier archive code and a changing runtime code at
988 and 133468 input tokens. Three independently salted cold references are
compared with shared-cache branches A, B, A, C, B. Returning to A and B exercises
copy-on-write isolation. The checks require exact cold-output equality, correct
answers, normal termination and the intended checkpoint reuse.

### Matched timing checks

`latency.py` uses fixed rendered/token-ID fixtures, three warmup requests and
eight measured requests per case. It includes cold 7675-token prefills, distinct
2304-prefix siblings, exact replays at three lengths and a stable-first Chat
fixture. Cache salts isolate each run without changing model input tokens.

`compare.py` rejects incomplete cases, unmatched prompt/output hashes, different
workloads, unexplained engine work, changed cache capacity and configuration
changes beyond the single matching-unit flag. Timings are sequential c=1
first-token microbenchmarks. They characterize cache tradeoffs, not production
throughput or tail latency.

For the final matched comparison, both profiles additionally use
`num-gpu-blocks-override: 2176`, and both latency commands use
`--cache-blocks 2176`. This common experimental control removes a one-block
auto-profiling difference between starts. It is removed again for the final
baseline deployment.

## Mechanism in the pinned engine

Paths below are relative to the container's
`/usr/local/lib/python3.12/dist-packages/vllm/`.

- `v1/core/kv_cache_utils.py:558-623`: block identity hashes the parent hash,
  current token IDs and extra keys, including salt, LoRA and multimodal identity.
  An early change propagates through the subsequent hash chain.
- `v1/core/single_type_kv_cache_manager.py:1249-1324`: the Mamba manager already
  searches **right to left**, stopping at the longest available state checkpoint.
- `v1/core/kv_cache_coordinator.py:685-817`: the hybrid coordinator reconciles
  attention and recurrent-state hits and detects an uncached common prefix.
- `v1/core/sched/scheduler.py:357-432`: prefill steps stop at cacheable boundaries
  and discovered shared-prefix junctions. The 4096-token budget normally produces
  3840-token aligned chunks on the 768-token grid.
- `v1/core/single_type_kv_cache_manager.py:1500-1604`: skipped recurrent-state
  positions are null; the manager retains the running state at actual chunk ends.
- `v1/core/sched/scheduler.py:403-422` and
  `v1/core/single_type_kv_cache_manager.py:1648-1681`: fine matching adds a stop
  and a copy-on-write checkpoint at the prompt's last partial hash boundary.
  It does not produce recurrent checkpoints at every interior 128-token boundary.
- `v1/core/single_type_kv_cache_manager.py:1337-1382`: sparse retention controls
  which available states are retained. Default dense retention does not create
  states skipped during a forward pass.

Thus a long common token prefix is an upper bound, not evidence that a reusable
KDA state exists. Identical suffix text following a different causal prefix is
also not an interchangeable model state.

`trace_scheduler.py` extracts just the reviewed splitting function from the
installed source and runs it with text-only request metadata. Its recorded
`scheduler-trace.json` explains the extra partial-tail scheduling step without
importing vLLM or modifying the engine. It is a CPU scheduling trace, separate
from the measured GPU requests.

For application-owned templates, place stable instructions before volatile
runtime context while preserving message roles and meaning, for example:

```text
<stable instructions and reference material>
RUNTIME_CONTEXT current_date=2031-04-12
```

The measured date task establishes this layout's behavior for the synthetic
fixture. Application adoption should verify its own rendered token prefix and
answer semantics.

## Reproduction

Coordinate a quiet maintenance window. NewAPI channel 285 must be disabled and
the engine idle. The selected variant must already be loaded. Each output path
must be new; earlier evidence is preserved, including interrupted runs.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/2026-09-21-prefix-cache -p 'test_*.py' -v

python3 experiments/2026-09-21-prefix-cache/study.py \
  --variant baseline --out /tmp/opencode/prefix-baseline-new.json

# After loading the one-knob candidate:
python3 experiments/2026-09-21-prefix-cache/study.py \
  --variant tail128 --out /tmp/opencode/prefix-tail128-new.json
python3 experiments/2026-09-21-prefix-cache/validate.py \
  --variant tail128 --out /tmp/opencode/prefix-tail128-validation-new.json

# Load each timing profile with the same num-gpu-blocks-override: 2176.
# The two loaded profiles differ only in prefix-match-unit.
python3 experiments/2026-09-21-prefix-cache/latency.py \
  --variant tail128 --cache-blocks 2176 \
  --out /tmp/opencode/prefix-tail128-latency-new.json

# After loading baseline plus the same fixed-capacity control:
python3 experiments/2026-09-21-prefix-cache/latency.py \
  --variant baseline --cache-blocks 2176 \
  --out /tmp/opencode/prefix-baseline-latency-new.json
python3 experiments/2026-09-21-prefix-cache/compare.py \
  /tmp/opencode/prefix-baseline-latency-new.json \
  /tmp/opencode/prefix-tail128-latency-new.json \
  --out /tmp/opencode/prefix-comparison-new.json

# Restore the selected deployment profile and restart after the comparison.
# A baseline deployment has neither experimental override.
```

See `RESULTS.md` and the adjacent JSON artifacts for measurements and the runtime
selection. Credentials and administrative status-change helpers stay outside Git.
