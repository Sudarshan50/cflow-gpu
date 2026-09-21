# AMD optimized Kimi-K3 candidates — 2026-09-20

Profiles for direct vLLM behind the existing LiteLLM endpoint on one
8× MI355X node. `config-base.yaml` is optimized plain decoding;
`config-dspark.yaml` adds only the four-key `speculative-config` mapping.
Both use `optimized.env`. See the qualification artifacts described below for
runtime status; the profiles alone do not establish a performance result.

**Final result:** optimized plain is deployed; DSpark is retained as a tested
alternative. See [RESULTS.md](RESULTS.md) for measured throughput, memory/tail
tradeoffs, active configuration paths and public API verification.

## Pinned provenance

AMD source: **AMD-AGI/Infera commit
`625a950b109371aaf8ebedfdc05757b57ac32eab`**.

- [AMD rendered recipe](https://rocm.docs.amd.com/projects/infera/en/latest/recipes/kimi-k3-optimized.html)
- [Pinned README](https://github.com/AMD-AGI/Infera/blob/625a950b109371aaf8ebedfdc05757b57ac32eab/examples/recipes/kimi-k3-optimized/README.md)
- [Plain raw manifest](https://raw.githubusercontent.com/AMD-AGI/Infera/625a950b109371aaf8ebedfdc05757b57ac32eab/examples/recipes/kimi-k3-optimized/aggregated/deploy.yaml)
- [DSpark raw manifest](https://raw.githubusercontent.com/AMD-AGI/Infera/625a950b109371aaf8ebedfdc05757b57ac32eab/examples/recipes/kimi-k3-optimized/aggregated-dspark/deploy.yaml)

Artifacts named by that recipe:

| Artifact | Exact reference |
| --- | --- |
| Optimized base image | `johnqin2025/kimi-k3-dspark:1.1.0-mi355x-rocm7.2.3-20260802@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8` |
| Infera overlay (v0.2.2) | `inferaimage/infera-overlay@sha256:6918eff34f201548a738dd592d2a1ece0627354d2e88f24a87cfa8f787a72a44` |
| Community draft | `Inferact/Kimi-K3-DSpark`, revision `cf6b8244620e7ea4b0651d214f28e89eac75bed6` |

The target is `moonshotai/Kimi-K3`; the deployment helper must supply its exact
local snapshot path as the positional model argument in both runs and record its
revision. The draft path is already pinned under `/hf/hub` in the DSpark YAML.
Model provisioning and image/overlay inspection belong to the deployment helper.
AMD reports the same vLLM commit (`g5f76ae224`) for stock and optimized images;
the optimized artifact adds kernel changes. An image name containing “dspark”
does not enable speculation in the plain profile.

## Translation from AMD's manifest

| Setting | Candidate / rationale |
| --- | --- |
| Worker execution | Direct `vllm serve` for LiteLLM instead of `infera.engine.vllm` and the Infera router. Overlay digest is recorded for provenance; inspect its launcher/payload before deciding how to use it with direct vLLM. |
| Endpoint | Bind `127.0.0.1:8001`; aliases `FW-Kimi-K3`, `kimi-k3`, `moonshotai/Kimi-K3`, in that order. First alias remains the response model name. |
| TP / EP | TP=8, EP disabled, matching AMD's omitted/default-false EP rather than the existing EP-enabled configuration. `no-enable-expert-parallel: true` explicitly emits the negative flag. |
| Context | **262144**, a deliberate reduction from AMD's 1048576 to preserve the local external contract. This does not establish long-context capacity at the new settings. |
| Admission / memory | AMD's 64 sequences, 4096 batch tokens, 0.88 GPU memory utilization in both arms. AMD documents draft allocation failure at 0.95. |
| Loading / multimodal / KV | AMD's `auto` load format, `data` multimodal encoder TP mode, `auto` KV dtype and enabled prefix caching. FP8 compute kernels do not imply FP8 KV. |
| Parsers | AMD's `kimi_k3` reasoning/tool parsers and `enable-auto-tool-choice: true`. “Auto” tool choice is distinct from parser selection. |
| Priority | **`scheduling-policy: priority` is a gateway-required deviation.** The gateway always sends priority; retaining the compatible policy is intentional. |
| Usage / logging | Retain `enable-prompt-tokens-details: true` for cached-token accounting and `no-enable-log-requests: true` for the local logging contract. |
| Environment | Copy every GPU/kernel key and value from AMD's worker block, including both exact `KIMI_K3_*_WEIGHT_CACHE_MODIFIER` names, the HIP/HSA settings, and the paired SiTU/MoE settings. Add `HF_HOME=/hf`; retain `HF_HUB_OFFLINE=1`. |

The direct-vLLM environment omits Kubernetes `POD_*` metadata and Infera's
`INFERA_ENGINE_READY_TIMEOUT=7200`, which only controls its worker supervisor.
A future direct launcher must budget startup time separately; AMD reports
10–14 minutes for weight load, AITER JIT and graph capture on local storage.
`optimized.env` contains no credentials. Public authentication remains in
LiteLLM; the engine stays on loopback. Retain the GPU device/runtime access
needed by the node. Do not merge the old tuning file into these profiles:
explicit async scheduling, graph compilation, watermark, prefix-match-unit,
stream interval and long-prefill-threshold overrides are absent from AMD's
manifest. The live-only `VLLM_ROCM_USE_AITER_MOE` override is likewise not copied.
The inspected effective defaults and actual runtime results are in RESULTS.md.

The config parser can drop YAML boolean `false` values. Negative flags here use
the positive boolean form (`no-…: true`); validate those exact options against the
pinned image's real CLI before deployment.

## DSpark sampling and comparison scope

The speculative mapping matches AMD: method `dspark`, seven draft tokens,
`ROCM_AITER_MLA`, and a local model path. Inspected sampling/rejection defaults
are **greedy drafting and standard rejection**. Runtime measurements confirmed
genuine drafting and acceptance. **No synthetic acceptance mode or synthetic
acceptance length is configured.**

**No 3× throughput or local speedup claim is made.** AMD withdrew its throughput
figures because a short sweep without specified warm-up did not reproduce
(165.99 tok/s cold and 270.4 warm versus 241.69 published). Its remaining latency
and crash-fix observations concern its own manifests and hardware. In particular,
the roughly 49 ms versus 17 ms TPOT comparison involves 16-GPU disaggregated
DSpark versus 8-GPU aggregated plain serving; it is not evidence for a 3× gain
from adding DSpark to this node.

A useful later comparison is **optimized plain versus optimized DSpark** on the
same eight GPUs, exact image, target snapshot, environment, context, priority
mix, request sampling parameters, prompt/output lengths and cache conditions.
Use stated warm-up and sustained measurement windows; measure cold and warm
cache cases separately. Report completed-request goodput, output tokens/s,
TTFT/TPOT distributions, errors, preemptions and actual draft acceptance, across
low concurrency through the 64-sequence setting. Confirm seven-token speculation
and the actual draft backend/graph behavior in runtime evidence. Short agentic
outputs, unique long prompts and vision need representation. The crossover where
drafting ceases to help is workload-dependent. Comparing the live profile to
either candidate also changes kernels, EP, batching and memory, so it cannot
isolate the contribution of speculation.

## Source validation

Validation uses available PyYAML only, without importing vLLM or instantiating an
engine. Check both YAML mappings, the exact shared settings, and that removing
`speculative-config` from the DSpark mapping gives the plain mapping. Check the
four speculative fields and absence of synthetic acceptance settings. Compare
the env key/value mapping with AMD's pinned worker GPU block, plus the two HF
settings. Real image CLI/default inspection, model availability, startup,
correctness and performance remain deployment-stage checks.

## Local qualification work

The exact images and Inferact draft snapshot were acquired. CPU-only inspection
of the pinned base established:

- Target verification is causally masked in the Gluon kernel; the old stock
  image's future-token visibility bug is fixed in this artifact.
- Mixed/padded batch metadata, KDA strides, and 64-bit KV addressing passed the
  source/CPU regression in `/tmp/opencode/k3_opt_cpu_audit.py`.
- Draft attention deliberately uses a causal approximation rather than full
  parallel noncausal attention. This can reduce acceptance; acceptance and actual
  throughput must be measured rather than inferred from the DSpark model card.
- Real defaults are greedy drafting and standard rejection, with no synthetic
  acceptance. The actual CLI accepted both configs with device discovery stubbed
  to CPU for argument parsing only; no GPU engine was created by that check.
- The exact Infera overlay's seven startup patches concern external KV/RDMA
  connectors. The aggregated trial uses ordinary vLLM in the base image and no
  external connector, so it does not require the Infera router/operator overlay.

`qualify.py` runs eight synthetic checks (arithmetic, 20k-token retrieval, strict
JSON, forced tool use and a generated image), then warm-prefix fixed-output
streaming workloads at concurrency 1 and 16 for 60 seconds each plus completion
drain. It validates SSE finish/usage/DONE. This is a repeatable decode comparison,
not a replay of customer traffic or validation of the entire 262k context window.

The stock runtime passed all eight smoke checks before replacement. Its artifact
is `/tmp/opencode/k3-stock-smoke.json`; its timings include simultaneous customer
traffic and are not an isolated performance baseline.

Staged runtime configs and artifacts live in `/scratch/deploy-state/amd-optimized`.
The original `k3.service` and `/scratch/hf/config.yaml` are untouched; a named
`/etc/systemd/system/k3.service.d/60-amd-optimized.conf` drop-in selects the trial.
The gateway is stopped during isolated qualification. A background candidate
watcher (`/tmp/opencode/k3_run_candidate.py`) waits for readiness, executes the
checks, and removes only its unchanged trial drop-in/restores the original engine
and gateway if qualification fails. It does not automatically declare success
based on health alone. Its state/result JSON files record the actual outcome.

While the engine loads, local tests and deployment review continue. No ready-loop
wall-clock estimate is reported as measured inference performance.
