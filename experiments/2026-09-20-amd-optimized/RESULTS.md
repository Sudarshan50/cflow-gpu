# AMD optimized Kimi-K3 and DSpark — measured local results

Date: 2026-09-20 UTC. Hardware: one node, eight MI355X GPUs.

## Decision

**Use the AMD optimized image without DSpark for the current concurrent,
long-context service.** DSpark is implemented, runnable, and passed the tested
correctness checks, but its measured concurrent gain is too small to justify
the worse tail latency and reduced KV capacity for this workload.

This is not a finding that DSpark never helps. It improved the isolated
single-request workload by about 21%, and remains available in
`config-dspark.yaml` for a low-concurrency deployment or further qualification.

The original stock engine/config remains available for rollback. The optimized
plain profile is now live; final rollout verification is recorded below.

## Artifact and recipe

- AMD documentation supplied by the operator:
  https://rocm.docs.amd.com/projects/infera/en/latest/recipes/kimi-k3-optimized.html
- Infera source pin: `625a950b109371aaf8ebedfdc05757b57ac32eab`.
- Image: `johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8`.
- Target: `moonshotai/Kimi-K3`, revision `f831ab66814297da540d832a5235f8e904f29d06`.
- Draft: `Inferact/Kimi-K3-DSpark`, revision `cf6b8244620e7ea4b0651d214f28e89eac75bed6`.
- TP8, EP disabled, 262144 context, 64 sequences, 4096 batch tokens, memory
  utilization 0.88, auto/BF16 KV, prefix caching, priority scheduling.
- Required AMD FP8 pre-route/shared-expert/latent-tail kernel settings were used.
  FP8 compute here does **not** mean FP8 KV cache.

The Infera overlay was also inspected at its pinned digest. Its seven engine
patches concern external KV/RDMA connectors. The single-node trial used direct
vLLM with no external connector, retaining the existing gateway/LiteLLM stack.

The optimized image retains the stock-looking `g5f76ae224` version string but
has different kernels and correctness fixes. The stock image was not patched
in place.

## Matched comparison

Same pinned image, target, sampling, tuning environment and workload revision;
the numerical configuration differs only by the DSpark mapping. The intrinsic
runner change (plain V1 versus DSpark V2) is part of enabling DSpark.

- Four synthetic tasks: cache explanation, Python streaming parser, matrix
  explanation, HTTP-stream testing checklist.
- Mean measured input: **1568 tokens**; fixed **256 generated tokens/request**.
- Temperature 0, seed 42, thinking disabled, response caching bypassed by direct
  engine access. Workload is warm-prefix, not a replay of customer traffic.
- Full measured concurrency and output length warmed before timing.
- Each worker completed balanced four-task cycles, so both arms sampled each
  task equally. At least 60 seconds of arrivals plus cycle/completion drain.
- Gateway was stopped during the experiment. Engine generated-token and
  completion-counter deltas exactly matched test requests; boundary running and
  waiting counts were zero.
- Responses required valid final usage, finish reason and SSE `[DONE]`.

| Measure | Optimized plain | Optimized + DSpark | DSpark change |
|---|---:|---:|---:|
| c=1 output tokens/s | **67.96** | **82.22** | **+20.98%** |
| c=1 request p95 | 4.893 s | 3.719 s | lower |
| c=1 TTFT p95 | 0.235 s | 0.752 s | higher |
| c=16 output tokens/s | **480.74** | **493.62** | **+2.68%** |
| c=16 request p95 | **9.045 s** | **10.901 s** | **+20.51%** |
| c=16 TTFT p95 | 1.020 s | 1.059 s | slightly higher |
| Reported KV pool | **1,639,906 tokens** | **1,245,361 tokens** | **-24.06%** |
| c=16 prefix lookup hit/query fraction | 97.96% | 48.98% | lower in this short-prompt test |

Sample counts: plain c=1 **16**, DSpark c=1 **20**; c=16 **128 each**.
These are point estimates from one matched comparison, not statistical proof of
a 2.68% improvement. In particular, small-sample p95 values are noisy.

**No measured result supports a 3× aggregate-throughput claim here.** AMD's
current recipe README withdrew its earlier throughput figures due to
non-reproducible short/no-warmup measurements. Its guidance also says DSpark's
advantage generally declines with concurrency.

The earlier stock smoke timings included live traffic and are **not** a matched
stock-versus-optimized throughput baseline. Do not infer a causal kernel speedup
by dividing these isolated numbers by a historical live-traffic rate.

## DSpark was genuinely active

- Seven proposed tokens per draft; greedy drafting, standard real rejection.
- No synthetic acceptance configured.
- Draft CUDA graphs captured; **zero `running the draft eagerly` messages**.
- Mean accepted length, including bonus: **2.923 tokens/round**.
- Accepted draft-token fraction: **27.47%** in both workloads.
- No request errors or preemptions in either measured arm.

The image's draft attention has a documented causal approximation: it does not
provide full noncausal visibility across the proposed draft block. That can cost
acceptance. The target verifier correctly excludes future tokens, which is the
critical distinction for valid rejection sampling. This experiment did not
isolate how much of the observed acceptance loss is caused by that approximation.

## Correctness and long-context checks

Both arms passed all eight synthetic smoke checks:

- Four arithmetic/basic reasoning answers.
- Exact marker retrieval from **20,642 prompt tokens**.
- Strict JSON output.
- Forced tool invocation with validated JSON arguments.
- Main-colour identification on a generated image.

Output hashes matched between arms for every smoke check. The stock engine
also passed these checks before replacement.

Both arms then passed two exact-marker checks at **133,442 prompt tokens**:

| Long-context check | Plain | DSpark |
|---|---:|---:|
| First request time | 12.859 s | 13.576 s |
| First request cached tokens | 7,680 | 9,216 |
| Identical repeat time | 1.028 s | 0.453 s |
| Repeat cached tokens | 132,864 | 132,096 |

The first requests were not wholly cold: earlier smoke inputs shared an initial
prefix. Both returned the exact marker and `finish_reason=stop`. This validates
the exercised 133k retrieval case, not the entire 262k context range or broad
model quality.

The short-prompt cache fraction above should not be generalized as “DSpark halves
all caching”: the long-context repeat lost only one additional 768-token block.

## Startup work and operational changes

- Plain trial: engine started around **22:09:32**, ready **22:18:48**.
- DSpark trial: started **22:23:11**, ready **22:28:11**.
- During initial loading, protocol-completeness tests, rollback-ownership tests,
  profile validation, and comparison review were completed.
- **14 AITER libraries, 41,098,632 bytes**, were saved from the qualified plain
  runtime to an image-specific cache. Subsequent startup logs confirmed imports
  from `/root/.cache/aiter-5f3007`. Triton/other cache files also persist under the
  mounted `/root/.cache`.
- The second startup was shorter, but runner/draft work differs; the whole time
  difference cannot be attributed solely to caching.
- The named systemd drop-in selects the optimized image; the original
  `/etc/systemd/system/k3.service` and `/scratch/hf/config.yaml` remain untouched.
- NewAPI settings and channel weights were not changed by this experiment.

## Reproduction and artifacts

Source tools in this directory:

- `qualify.py`: smoke and balanced sustained workloads.
- `compare.py`: refuses unmatched provenance, unbalanced tasks, errors or
  unexplained engine output/completions.
- `check_long_context.py`: the two 133k retrieval/cache checks.
- `test_qualify.py`: incomplete/error SSE must not count as success.

Runtime artifacts: `/scratch/deploy-state/amd-optimized/`

- `optimized-base-qualification.json`
- `optimized-dspark-qualification.json`
- `comparison.json`
- `optimized-base-long-context.json`
- `optimized-dspark-long-context.json`
- Per-phase state/log JSON files and final-smoke results.

Additional CPU/operational checks:

- `/tmp/opencode/k3_opt_cpu_audit.py`: 5,464 target-window checks, mixed/padded
  metadata, recurrent strides and wide-addressing checks without GPU execution.
- `/tmp/opencode/test_k3_candidate_watch.py`: rollback must preserve concurrent
  operator replacements and record shutdown failures.

## Final live configuration and verification

The selected optimized plain engine started **22:33:02 UTC**, reached readiness
**22:37:02**, and passed a fresh eight-check smoke qualification at **22:37:12**.
This last start reused the compiled caches and took approximately four minutes.

Active configuration:

- Engine: `/scratch/deploy-state/amd-optimized/config-base.yaml`.
- Environment: `/scratch/deploy-state/amd-optimized/optimized.env`.
- Service override: `/etc/systemd/system/k3.service.d/60-amd-optimized.conf`.
- Gateway override: `/etc/systemd/system/k3-gateway.service.d/60-amd-optimized.conf`,
  setting the global admission ceiling to **64**, matching the qualified profile.
- LiteLLM override: `/etc/systemd/system/k3-litellm.service.d/60-amd-optimized.conf`,
  setting response-cache revision **`amd-5f3007-base-v1`** so replay is not mixed
  with completions from the prior engine build.
- Engine/gateway/LiteLLM use `Restart=always` again; qualification-only disabling
  of engine automatic restarts has been removed.

The gateway resumed at **22:37:48** and LiteLLM restarted at **22:37:51**. Engine,
gateway and LiteLLM readiness checks all returned **HTTP 200 at 22:38:39**.
The two GPU-less inspection containers were removed; the pinned images and draft
weights remain available for reproduction.

Six sequential checks through **https://api.cflowx.in/v1**, completed
**22:39:32–22:39:36**, all returned HTTP 200 and valid terminal responses:

| Check | First | Repeat |
|---|---:|---:|
| Prefix reuse with response replay bypassed | 0/988 cached tokens | **768/988** |
| Exact nonstream response | 0.418 s | **0.037 s**, same response ID |
| Exact streaming response | 0.536 s | **0.015 s**, same response ID |

Streams retained cached-token details, a finish reason, and `[DONE]`. These are
small functionality probes, not production-wide performance claims. The recipe's
default cache matching differs from the former explicit 128-token matching unit;
the observed 768-token reuse must not be reported as 896-token reuse from an older
deployment's probe.

The old service file and `/scratch/hf/config.yaml` are retained for rollback and
are **not the active optimized profile**. Restoring stock requires coordinating
the three named service overrides and restarting the affected services; removing
only an unrelated source YAML is not a rollback. The DSpark profile remains
tested but disabled in production.

## Next targeted work

The demonstrated concurrent gain does not justify DSpark's memory/tail cost as
the default for today's traffic. Future work can evaluate a fully noncausal
draft backend and a broader reasoning-enabled workload, using the same
correctness and matched-comparison discipline. Avoid replacing real acceptance
with a synthetic acceptance target to reproduce a marketing throughput number.
