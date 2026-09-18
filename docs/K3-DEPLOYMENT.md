# Kimi-K3 on 8× MI355X — Deployment & Benchmark Reference

Single-node vLLM deployment of `moonshotai/Kimi-K3` on AMD MI355X, tuned for a
96:4 input:output serving profile with 85% prefix-cache hits.

**Result: 2,504,145 tokens/min sustained in steady state (100.2% of the 2.5M target);
2,159,399 TPM averaged over a full 780-request run including ramp-up. 3.15× the
untuned baseline. 780/780 requests, zero failures.**

Date: 2026-09-18 · Host: 8× AMD Instinct MI355X VF

> **Historical record.** This documents the box as measured on the date above
> and is left unchanged so the numbers keep their context. One thing has since
> moved: `/scratch` was the 40 TB `/dev/vdc1` volume then, and is a directory on
> the 2 TB boot disk now, so that a droplet snapshot carries the weights
> ([`SNAPSHOT.md`](SNAPSHOT.md)). Storage-setup commands below are superseded by
> [`DEPLOY.md`](DEPLOY.md); the tuning, benchmarks and findings still stand.

---

## 1. Headline numbers

| config | TPM | % of 2.5M | RPM | TTFT p50 | TPOT p50 |
|---|---|---|---|---|---|
| baseline (W4A16, no cache) | 686,478 | 27.5% | 53.63 | 314,185 ms | 237.96 ms |
| + expert parallel, no cache | — | — | — | — | 57.82 ms (warmup) |
| + A8W4 + 85% cache, `FULL_AND_PIECEWISE` | 2,021,815 | 80.9% | 157.95 | 11,131 ms | 190.78 ms |
| **+ `FULL_DECODE_ONLY` (production)** | **2,159,399** | **86.4%** | **168.70** | **937 ms** | 213.65 ms |
| production, steady-state 60 s window | **2,504,145** | **100.2%** | — | — | — |

The gap between 86.4% and 100.2% is ramp-up. Concurrency has to climb from 0 to
~427; the run average includes that. Once warm, the queue depth was **0** — the
server fully absorbed the offered 3.26 req/s.

### Production run, full metrics

```
completed                  780          failed                     0
duration                   277.41 s     max_concurrent_requests    427
request_throughput         2.8117 r/s   output_throughput          1439.60 tok/s
total_token_throughput     35,989.99 tok/s
total_input_tokens         9,584,640    total_output_tokens        399,360
input share                0.9600       prefill                    34,550 tok/s
mean_ttft_ms               1412.11      median_ttft_ms             936.88
p90_ttft_ms                2109.65      p99_ttft_ms                8429.31
mean_tpot_ms               193.81       median_tpot_ms             213.65
p99_tpot_ms                247.62
median_itl_ms              96.82        p99_itl_ms                 628.05
median_e2el_ms             111,689.24   p99_e2el_ms                127,753.36
prefix cache hit rate      81.00%       GPU mean utilization       87.01%
```

---

## 2. What actually moved the needle

Ranked by measured contribution. This ordering is **not** what the initial
analysis predicted, and the reasons matter for future tuning.

### 2.1 The 85% prefix cache — largest single factor

The naive reading is "85% cached saves 85% of prefill compute." That is true but
nearly irrelevant here: prefill was only ~2.5% of wall time. The real mechanism is
**KV memory**.

With a shared prefix, vLLM stores those 10,445 tokens of KV **once** rather than
per request. Cost per additional concurrent sequence drops from 12,800 tokens to
1,843 unique input + 512 generated = **2,355 tokens**, a 5.4× reduction.

```
concurrency ≈ (KV_pool_tokens − prefix_tokens) / (unique_in + max_out)
            = (1,696,278 − 10,445) / 2,355
            ≈ 715 sequences
```

Since decode throughput scales with concurrency, this is the dominant lever.
Observed KV utilization fell from 98.6% (pinned, thrashing) to 49.4%.

### 2.2 A8W4 fp8 activations — 2.06× decode speedup

`AITER_SITUV2_A8W4=1` alone produces **garbage output**. It requires
`AITER_BF16_FP8_MOE_BOUND=0` as a partner. See §7.3 for the root cause.

Median TPOT across identical 20-prompt warmups:

| stage | TPOT p50 | total tok/s | TTFT p50 |
|---|---|---|---|
| baseline W4A16 | 82.02 ms | 4,628.16 | 1,386.86 ms |
| + expert parallel | 57.82 ms | 5,995.78 | 2,002.19 ms |
| + A8W4 + cache | 39.75 ms | 6,783.99 | 307.36 ms |
| + `FULL_DECODE_ONLY` | 39.50 ms | 6,810.62 | 303.07 ms |

### 2.3 `FULL_DECODE_ONLY` — a KV windfall, not compilation

This was applied intending to enable `torch.compile`. **It did not.** vLLM logs:

```
`torch.compile` is turned on, but the model ... does not support it.
Please open an issue on GitHub if you want it to be supported.
```

Kimi-K3's model class lacks the `@support_torch_compile` decorator, so
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` and `mode=VLLM_COMPILE` are inert.

The +5.5pp gain came from a **side effect**: `FULL_DECODE_ONLY` captures far fewer
CUDA graph variants than `FULL_AND_PIECEWISE`, freeing **11.7 GiB** that became KV.

| | `FULL_AND_PIECEWISE` | `FULL_DECODE_ONLY` |
|---|---|---|
| Available KV memory | 49.41 GiB | **61.09 GiB** |
| GPU KV cache size | 1,169,808 tok | **1,696,278 tok** |
| Max concurrency (uncached) | 91.39× | 132.52× |

This was decisive because at 512 concurrent with a shared prefix you need
~1.22M KV tokens, which the old 1.17M pool could not hold — KV was the binding
constraint, not `--max-num-seqs`.

Note this config also flipped `custom_ops` from `'all'` to `'none'`. Measured cost:
none (warmup TPOT 39.50 vs 39.75 ms, within noise), because the AITER kernels are
still selected via `ir_op_priority`.

### 2.4 Expert parallelism

`--enable-expert-parallel` moved TPOT 82.02 → 57.82 ms. Worthwhile, but third in
magnitude behind caching and A8W4.

---

## 3. Environment

### Hardware

| | |
|---|---|
| GPU | 8× AMD Instinct MI355X VF, `gfx950`, PCI `0x75b3` |
| VRAM | 308,902,100,992 B/card (287.7 GiB) = **2,304 GB** total |
| CPU | 192 vCPU |
| RAM | 2,015 GiB |
| Scratch | `/dev/vdc1`, 40 TB ext4 on `/scratch` |
| Kernel | `6.8.0-137-generic` |
| amdgpu | `6.19.14.31400000` |
| ROCm | `/opt/rocm` (7.14) |

### Image

```
vllm/vllm-openai-rocm:kimi-k3   id 5aa7e626ff73   57.2 GB
vLLM v0.1.dev19253+g5f76ae224.d20260727
```

> **Note the hyphen.** The image is `vllm-openai-rocm`, *not* `vllm-openai_rocm`.
> The underscore form does not exist and yields a pull failure.

### Model

| | |
|---|---|
| Repo | `moonshotai/Kimi-K3` |
| Params | 2.8 T total / 104 B active |
| Quantization | MXFP4 weights, MXFP8 activations |
| Files | 96 safetensors, **1.5 TB** on disk |
| Download | 1,561,025,666,513 B in ~700 s ≈ **2.23 GB/s** |
| Load time | weights 140.15 s; total 143.30 s / 192.29 GiB; engine init 74.15 s |

### Selected backends (confirm these in logs on every launch)

```
Using ROCM_AITER_FA MLA prefill backend.
Using ROCM_AITER_MLA backend out of potential backends:
    ['ROCM_AITER_MLA', 'TRITON_MLA', 'ROCM_AITER_TRITON_MLA'].
quantization=mxfp4   kv_cache_dtype=auto (bf16)
cudagraph_mode=FULL_DECODE_ONLY   51 capture sizes, largest 512, 0.34 GiB
```

---

## 4. Production deployment

### 4.1 Host prep (one time)

```bash
# Scratch disk — SUPERSEDED, see DEPLOY.md §4.1. /scratch is now a directory on
# the boot disk; mounting a volume here would hide the weights.
mkfs.ext4 -L DOSCRATCH /dev/vdc1        # destructive; only on first setup
echo 'LABEL=DOSCRATCH /scratch ext4 discard,errors=remount-ro 0 2' >> /etc/fstab
mkdir -p /scratch && mount /scratch
mkdir -p /scratch/hf /scratch/results

# TCP tuning for the 1.5 TB pull (values currently in effect)
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456
sysctl -w net.ipv4.tcp_rmem="4096 87380 268435456"
sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"
sysctl -w net.core.default_qdisc=fq_codel
sysctl -w net.ipv4.tcp_congestion_control=cubic
```

### 4.2 Fetch weights

```bash
apt-get install -y python3.12-venv jq          # venv pkg is NOT preinstalled
python3 -m venv /opt/hfv
/opt/hfv/bin/pip install -U "huggingface_hub[cli]" hf_transfer

export HF_HOME=/scratch/hf
export HF_TOKEN=<token>
/opt/hfv/bin/hf download moonshotai/Kimi-K3
```

`huggingface_hub` 1.32+ deprecates `hf_transfer` in favour of Xet and will warn.
Harmless — Xet delivered 2.23 GB/s unaided. `HF_XET_HIGH_PERFORMANCE=1` was not
needed.

### 4.3 Launch (production config)

```bash
docker rm -f k3 2>/dev/null
docker run -d --name k3 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --network host --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 64g -v /scratch/hf:/hf -v /scratch/results:/results \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e AITER_SITUV2_A8W4=1 \
  -e AITER_BF16_FP8_MOE_BOUND=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e SAFETENSORS_FAST_GPU=1 \
  --entrypoint vllm vllm/vllm-openai-rocm:kimi-k3 serve moonshotai/Kimi-K3 \
    --served-model-name kimi-k3 --trust-remote-code \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --load-format fastsafetensors \
    --max-model-len 12800 --max-num-seqs 512 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 --block-size 128 \
    --enable-prefix-caching --prefix-match-unit 128 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+fused_rms_norm_gated"]}' \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
    --no-enable-log-requests --host 0.0.0.0 --port 8000
```

Ready in roughly **6 minutes** (140 s weights + 74 s engine init + overhead).

> `--entrypoint vllm` is **required**. The image's default entrypoint is
> `/bin/bash`, so without this the serve arguments are handed to bash.

### 4.4 Environment variables

| variable | value | purpose |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | `1` | Master switch for AITER kernels. Without it the MXFP4 MoE oracle finds no viable backend and startup fails. |
| `VLLM_ROCM_USE_AITER_MOE` | `1` | AITER MoE path. |
| `AITER_SITUV2_A8W4` | `1` | fp8 activations × int4 weights MoE GEMM. **Must** pair with the next var. |
| `AITER_BF16_FP8_MOE_BOUND` | `0` | Forces fp8 activations at all batch sizes. Without it, small batches silently use bf16 activations against interleaved A8W4 weights → garbage output. |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `0` | Intended to enable torch.compile. **Inert on K3** (see §2.3); retained only because it accompanies the `FULL_DECODE_ONLY` config that won. |
| `SAFETENSORS_FAST_GPU` | `1` | Faster weight load. |
| `HF_HOME` | `/hf` | Mapped to `/scratch/hf`. |
| `HF_HUB_OFFLINE` | `1` | No network calls at startup. |

### 4.5 Serve flags

| flag | value | rationale |
|---|---|---|
| `--tensor-parallel-size` | `8` | All GPUs. |
| `--enable-expert-parallel` | — | TPOT 82.02 → 57.82 ms. |
| `--load-format` | `fastsafetensors` | 1.5 TB in 140 s. |
| `--max-model-len` | `12800` | 12,288 in + 512 out exactly. Do not inflate; it directly scales KV per sequence. |
| `--max-num-seqs` | `512` | Concurrency ceiling. Peak observed 427. |
| `--max-num-batched-tokens` | `8192` | Prefill chunk. |
| `--gpu-memory-utilization` | `0.92` | Headroom for graphs + AITER workspaces. |
| `--block-size` | `128` | Matches `--prefix-match-unit`. |
| `--enable-prefix-caching` | — | **Essential.** The whole 85%-cache benefit depends on it. |
| `--prefix-match-unit` | `128` | Cache-key granularity, aligned to block size. |
| `--compilation-config` | `FULL_DECODE_ONLY` | Frees 11.7 GiB into KV (§2.3). |
| `--no-enable-log-requests` | — | Replaces the removed `--disable-log-requests`. |

---

## 5. Benchmark methodology

### 5.1 Workload shape

Target profile: 96:4 input:output, 12,288 in / 512 out, **85% of input cached**.

`vllm bench serve`'s random dataset generates the fixed prefix **once** outside the
request loop (`benchmarks/datasets/datasets.py:564`, `# Generate prefix once`), so
it is genuinely shared and cacheable. Total input is
`prefix_len + input_len` (line 759).

```
prefix_len = round(12288 × 0.85) = 10,445   ← shared, cacheable
input_len  = 12288 − 10445       =  1,843   ← unique per request
total                             = 12,288  ✓  10,445/12,288 = 85.0%
```

**Realistic ceiling is 84.4%, not 85%.** The 128-token match unit floors the
prefix to 81 blocks × 128 = 10,368 tokens. Observed steady-state: **81.00%**.

### 5.2 Warmup (also populates the cache and autotunes AITER kernels)

```bash
docker exec k3 vllm bench serve \
  --backend openai --base-url http://localhost:8000 \
  --model moonshotai/Kimi-K3 --served-model-name kimi-k3 \
  --tokenizer moonshotai/Kimi-K3 --trust-remote-code \
  --dataset-name random \
  --random-prefix-len 10445 --random-input-len 1843 --random-output-len 512 \
  --random-range-ratio 0 --ignore-eos \
  --request-rate 1 --num-prompts 20
```

### 5.3 Target run

3.26 req/s × 12,800 tokens × 60 = 2.50M TPM offered.

```bash
docker exec k3 vllm bench serve \
  --backend openai --base-url http://localhost:8000 \
  --model moonshotai/Kimi-K3 --served-model-name kimi-k3 \
  --tokenizer moonshotai/Kimi-K3 --trust-remote-code \
  --dataset-name random \
  --random-prefix-len 10445 --random-input-len 1843 --random-output-len 512 \
  --random-range-ratio 0 --ignore-eos \
  --request-rate 3.26 --burstiness 1.0 --num-prompts 780 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir /results --result-filename run.json
```

Verify the shape held: `input share` must read exactly `0.9600`.

### 5.4 Confirm cache is working

```bash
curl -s http://localhost:8000/metrics \
  | grep -E "^vllm:(prefix_cache_queries_total|prefix_cache_hits_total)" \
  | awk '{print $NF}' | paste -sd' ' \
  | awk '{printf "hit_rate=%.2f%%\n", 100*$2/$1}'
```

Expect ~81%. A reading near 0% means `--enable-prefix-caching` is missing or the
dataset has no shared prefix — the benchmark is then meaningless for this profile.

### 5.5 Measure steady state, not just the run average

The run average is dragged down by ramp-up. For capacity planning, read the
trailing-window TPM once concurrency has plateaued (queue depth 0, ~90 s in).
Run average 2,159,399 vs steady state 2,504,145 — a 16% difference.

---

## 6. Correctness gate

**Run this after every launch, before trusting any number.** Several
configurations start cleanly, serve at full speed, and emit fluent nonsense
(§7.3). Large-batch benchmarks do **not** expose that failure; only correctness
probes do.

```bash
/scratch/gate.sh             # full gate, ~21 s
/scratch/gate.sh --quick     # tier 1 only, ~5 s
/scratch/gate.sh --baseline  # re-record the baseline after an intended change
```

Implementation is `/scratch/gate.py` (stdlib only, no dependencies).
Exit 0 = pass, 1 = fail. Three tiers:

| tier | checks | time |
|---|---|---|
| 1 CORRUPTION | 3 factual probes scored by rank+margin, numeric continuation, semantic stability | ~5 s |
| 2 REASONING | 16 arithmetic word problems, exact numeric answers, run 8-way parallel | ~10 s |
| 3 LONGCTX | needle retrieval at ~12k prompt tokens (production shape) | ~6 s |

Healthy baseline on the production config:

```
[PASS] top1(Paris)         -0.245  margin +3.56 over ' {'
[PASS] top1(Jupiter)       -0.578  margin +2.38 over ' ______'
[PASS] top1(Celsius)       -0.624  margin +1.38 over ' cent'
[PASS] counting            10 11 12 13 14 15
[PASS] semantic stability  facts consistent across 3 samples (3 distinct texts)
-> 16/16 correct (floor 14)
[PASS] needle retrieval    {'prompt_tokens': 12026, 'got': 74812}
GATE PASS in 20.6s
```

Three design points, each learned from a false result during construction:

- **Rank + margin, not an absolute logprob.** An absolute floor is not portable
  across prompts: `" Paris"` sits near -0.24 but `" Celsius"` near -0.70, purely
  because it competes with `" cent"` and `" Cent"`. Both are rank-1 and both are
  correct. A -0.6 floor failed a perfectly healthy model. What actually signals
  corruption is the right token losing its lead, so the gate requires rank-1 with
  a margin >= 0.8 over the runner-up.
- **Semantic stability, not bitwise determinism.** vLLM is *not* bitwise
  deterministic at temperature 0 — continuous batching varies batch composition
  and float reduction is not associative. Measured here: 5 identical requests
  produced 3 distinct texts on a healthy server. Asserting byte equality
  guarantees a false failure; the gate asserts the facts are stable instead.
- **Tier 3 must fit `max_model_len`.** Filler tokenizes at ~17 tokens/repeat;
  overshooting returns HTTP 400, which reads like a model failure but is a gate
  bug. Sized to ~12,026 prompt tokens against the 12,800 limit.

Baseline is stored at `/scratch/gate-baseline.json` and the gate flags a
reasoning drop of more than 2 problems even when still above the absolute floor.

**Validated in both directions** (a gate that only passes is worthless):

| control | result |
|---|---|
| healthy server | exit 0 |
| server unreachable | exit 1 — **fails closed**, does not silently pass |
| wrong expected answer | detected |
| synthetic flat logprob distribution (margin +0.10) | detected |
| reasoning 16 -> 13 | regression flagged |

`/scratch/gate-legacy.sh` retains the original two-probe version.

---

## 7. Failure catalog

Every issue hit during bring-up, with the fix. Most cost 10+ minutes to diagnose.

### 7.1 Wrong image name
`vllm/vllm-openai_rocm:kimi-k3` does not exist. Correct: `vllm-openai-rocm` (hyphen).

### 7.2 MXFP4 MoE — no viable backend
```
NotImplementedError: No MXFP4 MoE backend supports the deployment configuration.
```
`VLLM_ROCM_USE_AITER_MOE_SITUV2` is **not a real variable**. Use
`VLLM_ROCM_USE_AITER=1` plus `AITER_SITUV2_A8W4=1`.

### 7.3 Garbage output with A8W4 — the subtle one
`AITER_SITUV2_A8W4=1` forces an interleaved weight layout **and** selects
activation dtype by batch size. Below the bf16/fp8 boundary the kernel feeds bf16
activations to interleaved A8W4 weights — a silent layout mismatch producing
fluent gibberish. Small-batch coherence checks trip it; large-batch benchmarks do not.

**Fix:** `AITER_BF16_FP8_MOE_BOUND=0` pins fp8 activations at every batch size.
Validated: `logprob(Paris) = -0.242`, counting intact.

### 7.4 fp8 KV cache is unusable at TP=8
```
mla_gluon[bh16bn128] requires batch_size=1, got 128
```
TP=8 gives 12 heads/rank, below the 16-head threshold for the persistent
batching-capable decode kernel, so fp8 KV falls back to a `batch_size=1` path.
**Drop `--kv-cache-dtype fp8`; use default bf16.** Moot anyway — KV sits at 49%,
so there is nothing to gain from halving it.

### 7.5 `flashkda` prefill backend does not exist for K3
```
AssertionError: The shared Kimi GDN layer only supports the Triton KDA
prefill backend, got 'flashkda'
```
Remove `--kda-prefill-backend` entirely at any value. The default is correct.

### 7.6 `torch.compile` is a no-op on K3
See §2.3. Not an error, but do not expect compilation gains.

### 7.7 Removed / renamed flags
`--disable-log-requests` no longer exists in vLLM 0.27+ → `--no-enable-log-requests`.

### 7.8 Missing `python3.12-venv`
Not preinstalled. `apt-get install -y python3.12-venv`.

### 7.9 Harmless warnings — safe to ignore
```
Op 'grouped_topk' not present in model, enabling with '+grouped_topk' has no effect
Op 'sparse_attn_indexer' not present in model, ...
Triton kernel JIT compilation during inference: chunk_kda_fwd_kernel_*
```
The Triton JIT messages indicate first-touch compilation latency spikes — the
warmup run covers them.

---

## 8. Tooling

| path | purpose |
|---|---|
| `/scratch/gate.sh` | Coherence gate (§6). Run after every launch. |
| `/scratch/stopbench.sh` | Kills benchmark + GPU sampler without killing the parent shell. |
| `/opt/k3dash/server.py` | Dashboard backend: scrapes vLLM Prometheus, derives windowed TPM and histogram percentiles, samples `rocm-smi`. |
| `/opt/k3dash/index.html` | Self-contained live UI, canvas charts, no external deps. |

Dashboard on **port 8080**; JSON state at `/api/state`:

```bash
curl -s http://localhost:8080/api/state | jq -c \
  '{tpm: .window.total_tpm, pct: .window.pct_of_target,
    running: .latest.running, waiting: .latest.waiting,
    kv_pct: (.latest.kv_usage*100), gpu: .gpu.mean_use}'
```

GPU sampling loop used during runs:

```bash
while true; do echo "$(date +%s) $(rocm-smi --showuse --csv 2>/dev/null | tr '\n' ' ')"; sleep 5; done
```

Instantaneous `rocm-smi` reads can show 0% by catching an idle moment between
kernels; average over the run rather than trusting single samples.

---

## 9. Operations

### Health

```bash
curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models   # expect 200
docker logs k3 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency"
bash /scratch/gate.sh
```

Expected on the production config:

```
Available KV cache memory: 61.09 GiB
GPU KV cache size: 1,696,278 tokens
Maximum concurrency for 12,800 tokens per request: 132.52x
```

`Maximum concurrency` is computed **assuming no prefix sharing** and is therefore
pessimistic by ~5× for this workload. Ignore it; watch actual KV utilization.

### Key metrics to watch in production

| metric | healthy | meaning if wrong |
|---|---|---|
| `vllm:num_requests_waiting` | ~0 | Sustained >50 means offered load exceeds capacity. |
| KV utilization | <70% | Approaching 100% causes preemption and collapse. |
| prefix cache hit rate | ~81% | A drop means traffic lost its shared prefix; capacity falls ~3×. |
| GPU utilization | 85–100% | Low with a full queue implies a stall, not saturation. |

### Results

```
/scratch/results/target.json                    baseline
/scratch/results/cfg-a8w4-cached.json           A8W4 + cache, FULL_AND_PIECEWISE
/scratch/results/cfg-compile-cached.json        production, FULL_DECODE_ONLY
/scratch/results/*-stdout.log                   full bench output
/scratch/results/k3-gpuse*.log                  GPU samples
/scratch/k3-results.tar.gz                      all of the above
```

---

## 10. Remaining headroom

Not exhausted; listed in descending expected value.

1. **Shorten ramp-up** — the only thing between 86.4% run-average and 100%. KV peaked
   at 49.4% and concurrency at 427 of a 512 ceiling, so raising `--max-num-seqs`
   (→768) and `--gpu-memory-utilization` (→0.95) should let it reach steady state sooner.
2. **SGLang A/B** on the same node — untested.
3. **Multi-node** — untested. Single-node already meets target in steady state.

Do **not** bother with: fp8 KV cache (§7.4, and KV is not the constraint at 49%),
`torch.compile` (§2.3), or `--kda-prefill-backend` (§7.5).

---

## 11. Production notes (researched 2026-09-18)

### 11.1 This image is outdated — upgrade target

K3 reached stable vLLM in v0.27.0 (2026-08-10). Verified tags in
`vllm/vllm-openai-rocm`:

| tag | digest | note |
|---|---|---|
| `kimi-k3` | `sha256:5aa7e626ff73...` | **what §4.3 runs.** Dev build, 2026-07-27 |
| `v0.28.0` | `sha256:e0a3b2bd3fe7...` | K3 on ROCm w/ V2 runner, gfx950 enablement |
| `v0.29.0` | `sha256:e5e47f6aaab6...` | **upgrade target.** V2 runner is default |

vLLM's 2026-09-13 benchmarks report v0.27.1 → main at 2.2-2.8x throughput for K3.
Our image predates v0.27.0. vLLM issue #50347 is filed against this exact build
string on 8xMI355X TP=8 (HIP 700 fault in `ROCM_AITER_MLA` on long-context
multi-turn).

> **The AITER flag contract changed.** The current K3 recipe uses
> `VLLM_ROCM_USE_AITER_MOE_SITUV2=1` and says explicitly **not** to set
> `AITER_SITUV2_A8W4=1`, because AITER checks that flag first and it overrides
> the intended path. Note §7.2 found `VLLM_ROCM_USE_AITER_MOE_SITUV2` *unrecognized*
> in the current image — the variable names differ by build. **Carrying §4.4's env
> vars forward verbatim is the most likely way to recreate the §7.3 silent-garbage
> failure.** Build the eval gate before upgrading, not after.

Pin by digest, mirror to a private registry, and keep a `docker save` tarball.
The `kimi-k3` tag is a one-off launch tag in a repo that pushes nightlies, now
superseded by versioned releases; it has no reason to survive garbage collection.

### 11.2 `cache_salt` vs the shared prefix — a real tradeoff

Cross-tenant prefix caching is a documented timing side channel (CVE-2025-46570):
TTFT reveals whether a guessed prefix is already cached, reported at ROC AUC 0.99
with only 8 tokens. The mitigation is a per-request `cache_salt` mixed into the
first block hash.

**Verified on this deployment** — same 1,500-token prompt, varying only the salt:

| request | prefix cache hit rate |
|---|---|
| salt-A, cold | 0.0% |
| salt-A, repeat | 97.7% |
| salt-B, *identical text* | **0.0%** |
| salt-B, repeat | 97.7% |
| salt-A again | 97.7% |

So salting genuinely partitions the cache — and would fragment the shared prefix
this whole deployment depends on. Cost against the 1,696,278-token KV pool is
`N x 10,445` tokens resident before any live request:

| trust groups | prefix KV | % of pool | approx max concurrency |
|---|---|---|---|
| 1 (no salt) | 10,445 | 0.6% | ~427 (observed) |
| 10 | 104,450 | 6% | ~676 |
| 50 | 522,250 | 31% | ~498 |
| 100 | 1,044,500 | 62% | ~117 |

**Decide deliberately.** Mutually trusted tenants: omit `cache_salt`, keep the
81% and the 3x. Mutually untrusted: salt **per trust boundary, never per API key**,
use >=256 bits of randomness (a guessable salt restores the attack), and expect
the cliff past ~50 groups.

### 11.3 Exposure — the one-line shape

`--api-key` guards only the `/v1`, `/v2`, and `/inference` prefixes. `/invocations`
routes to the same inference functions with **no auth**, as do `/pooling`,
`/classify`, `/score`, and `/rerank`; `/pause` and `/update_weights` are also open.
Bind vLLM to localhost and put a reverse proxy in front that **allowlists** paths
(`location / { return 404; }`), since blocklisting breaks each time vLLM adds an
endpoint. Set the read timeout above the ~112 s median E2E latency, and disable
proxy buffering or SSE streaming will batch.

`--api-key` does accept multiple keys (repeat the flag) for rotation, but gives no
per-key identity, budgets, or revocation without restart — and restart costs ~6 min.

Not exposed to CVE-2026-48746 (`Host`-header auth bypass, affects <0.22.0).

### 11.4 Tooling verdicts for this specific deployment

- **Prefix-aware routing (llm-d/KServe/AIBrix) is near-pointless here.** It
  partitions caches across replicas when many distinct prefixes compete for
  insufficient KV. With ~1 shared prefix and KV at 49%, every replica caches it
  permanently and round-robin performs identically. Also: two replicas need 3.0 TB
  against 2.304 TB HBM, so `--data-parallel-size 2` cannot fit on this node.
  Trigger to revisit: distinct prefix count above ~20, or hit rate dropping after
  scale-out.
- **LiteLLM for load balancing would hurt.** No prefix affinity (the PR adding it
  was closed unmerged) and it rewrites tool schemas by stripping
  `additionalProperties`/`strict` — which changes the token prefix, since K3
  renders tools into the prompt. Fine as a single-backend key/budget layer.
- **SkyPilot does not solve this.** `sky/clouds/do.py` hard-codes
  `SPOT_INSTANCE: 'Spot instances are not supported in DO'`, and the DO catalog
  contains only H100 — no AMD entries.
- **Red Hat AI Inference Server is blocked on scope, not just version.** 3.5.0
  ships vLLM v0.24.0, but AMD support covers "FP8 (W8A8) and GGUF only" and K3 is
  MXFP4.
- **Observability:** metric names verified against this build in §11.5.

### 11.5 Metric names verified on this build

67 `vllm:` metrics exposed. Current names (older docs use removed ones):

```
vllm:kv_cache_usage_perc                    (0-1 fraction despite "_perc")
vllm:inter_token_latency_seconds            (aggregate ITL/TPOT)
vllm:request_time_per_output_token_seconds  (per-request TPOT, distinct metric)
vllm:prefix_cache_hits_total / _queries_total
vllm:external_prefix_cache_hits_total / _queries_total
vllm:num_preemptions_total
vllm:num_requests_waiting_by_reason         (present; prefer reason="capacity")
```

> **Measured bug: prompt-token percentiles are pinned.** Histogram buckets derive
> from `max_model_len`, so with 12800 the series ends at 10000 — and **802 of 802
> requests land in `+Inf`**. Any p50/p99 prompt-size panel reads exactly 10000
> forever, which looks plausible rather than broken. Use
> `rate(vllm:prompt_tokens_total[5m]) / rate(vllm:request_success_total[5m])`
> for exact mean prompt size instead.

Alert on preemption rather than cache usage — vLLM is designed to run KV near
full, so `>0.95` pages during healthy operation while
`rate(vllm:num_preemptions_total[5m]) > 0` means work is actually being discarded.
Guard the cache-hit ratio alert with `and sum(rate(...queries_total[10m])) > 10`
or `0/0` yields NaN and flaps when idle.

---

## 12. Copy-paste quick start

```bash
# assumes weights already in /scratch/hf
docker rm -f k3 2>/dev/null
docker run -d --name k3 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --network host --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 64g -v /scratch/hf:/hf -v /scratch/results:/results \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
  -e AITER_SITUV2_A8W4=1 -e AITER_BF16_FP8_MOE_BOUND=0 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e SAFETENSORS_FAST_GPU=1 \
  --entrypoint vllm vllm/vllm-openai-rocm:kimi-k3 serve moonshotai/Kimi-K3 \
    --served-model-name kimi-k3 --trust-remote-code \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --load-format fastsafetensors \
    --max-model-len 12800 --max-num-seqs 512 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.92 --block-size 128 \
    --enable-prefix-caching --prefix-match-unit 128 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+fused_rms_norm_gated"]}' \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice \
    --no-enable-log-requests --host 0.0.0.0 --port 8000

# wait ~6 min, then:
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models)" = 200 ]; do sleep 10; done
bash /scratch/gate.sh          # MUST print GATE PASS
```

---

# Procurement review (2026-09-18)

Nine candidates evaluated against primary sources. **Six rejected, and five of the
six rejections trace to a single platform fact rather than to tool deficiencies.**

## The finding that decides Job 1

DigitalOcean's API has **no spot field at all**. `rg -ic spot specification/` across
`digitalocean/openapi` @ `a06aa8a` returns **0 matches in 2,968 files**, and
`droplet_create.yml` has exactly 14 properties, none of them capacity-tier. DO's docs
show the spot toggle only in the Control Panel section; the Spot GPU table is the one
GPU table published **without a Slug column**.

No client of that API can expose what the API does not define. Terraform, Pulumi,
Ansible, Packer, doctl and dstack all fail for this one reason. **The droplet must be
created by hand in the Control Panel; automation begins at first boot.**

dstack deserves a note: it *passes* the AMD test that killed SkyPilot — MI355X is a real
catalog entry (`gpuhunt/_internal/constraints.py:271-276`, CDNA4, device id `0x75A3`) —
but `gpuhunt/providers/digitalocean.py:108` hard-codes `spot=False` on every DO offer.

To settle the one residual doubt (an undocumented slug): `make -f deploy/Makefile sizes`

## The finding that decides Job 2

Re-downloading from HuggingFace is **optimal**, and the reason is structural, not marginal.

GPU droplet ingress has **no documented cap and is unmetered** ("only outbound traffic
counts against your monthly transfer allowance" — DO Droplet Limits). Egress is capped at
10 Gbps **and billed**. Every caching scheme must first push 1.56 TB *outbound* through
the slow metered direction — ≥1,249 s, which is 1.8x the entire download — to build a
cache whose best case is a *tie*.

| Option | Restore 1.56 TB | vs 700 s | $/mo |
|---|---|---|---|
| **HF re-download (measured)** | **700 s** | **1.00x** | **$0** |
| + `HF_XET_HIGH_PERFORMANCE=1` | ≤700 s | ≤1.00x | $0 |
| Cross-region Spaces | ≥700 s (no documented ceiling) | ≥1.00x | $29 + egress |
| Droplet snapshot | UNVERIFIED (DO publishes no figure) | — | $94 |
| Block volume @ 450 MB/s | **3,469 s** | **4.96x** | $150 |

Custom image **uploads** are dead on size: "Images must be 100 GB or less when
uncompressed" — 1,453.9 GiB is 14.5x over. Block volumes' 525 MB/s is a **60-second
burst**, not a sustained rate, so 450 MB/s is the honest number.

Spot reclamation destroys **both** disks: "Any data on the boot disk or scratch disk
associated with a Spot instance is lost upon reclamation" (Spot Preview Terms §2.5).
So re-download also wins on reclaim-survival — there is no state to lose.

## Job 3: the defect in our previous fix

**`docker update --restart unless-stopped` cannot back off for this workload.** Docker
resets its backoff to 100 ms once a container has run ≥10 s
(`moby/daemon/internal/restartmanager/restartmanager.go:64-67`), and
`MaximumRetryCount` is consulted **only** for `on-failure` (line 84). Model load takes
~6 min, so every crash clears the 10 s threshold: the known HIP 700 crash
(vLLM #50347) yields an unbounded 100ms-restart / 6-min-reload loop with no cap.

`deploy/k3.service` moves supervision to systemd: 30s → 60s → 120s → 240s → 480s, then
stops after 5 failures in 30 min. It also moves `RequiresMountsFor=/scratch` **off**
`docker.service` onto the container unit — on the daemon, `Requires=` propagates stops,
so unmounting `/scratch` would have taken down all of Docker.

## Job 4: do NOT set `VLLM_BATCH_INVARIANT=1` on this box

The env var exists in our build but is **NVIDIA + Intel XPU only**, and on ROCm it
**fails silently rather than erroring**. Verified on this machine:

    is_cuda=False  is_rocm=True  is_xpu=False  dispatch_key=CUDA

`batch_invariant.py` branches on `is_cuda()` (line 914) and `elif is_xpu()` (930).
ROCm takes neither, so no `mm`/`addmm`/`matmul`/`linear` override is installed — yet
execution continues into an unguarded block that *does* register `softmax`/`bmm`
overrides under dispatch key `CUDA`. Result: partial application, no determinism, and
numerics that differ from what we tested. `rocm_aiter_mla.py` does not declare
`supports_batch_invariance`. ROCm support is open, unmerged PR #52231 (`needs-rebase`).

Cost where it *does* work: **25-35% throughput loss**. And it would not have caught our
AITER bug — batch invariance guarantees reproducibility, not correctness. Fluent
nonsense would reproduce bitwise.

**Keep the bespoke gate.** Every off-the-shelf harness fails on specifics: lm-eval and
inspect-ai have no gating exit code (inspect exits 0 on failure *by design*); promptfoo,
DeepEval and Ragas all need a second model for semantic assertions. None can assert
logprob margins, which is our most sensitive probe.

## Job 6: `--config` is native and verified

`vllm serve --config` exists in our exact build (`cli_args.py:368`; precedence
`cli > config > defaults` at `argparse_utils.py:494`). `deploy/config.yaml` was verified
**offline** against `/proc/1/cmdline`: 19 flags, zero differences.

**FOOTGUN:** `key: false` emits *nothing* (`argparse_utils.py:571-573`), silently giving
the flag's default rather than the disabled state. Use `no-enable-log-requests: true`.

Env vars cannot live in the YAML — `envs.py` reads `os.environ` at 80 sites, and the
config path only produces argv. Hence `deploy/vllm-k3.env`.

## Summary

| Tool | Job | Verdict | Decisive fact |
|---|---|---|---|
| **doctl + Makefile** | 1 | **ADOPT** | Only candidate that can script snapshot/restore; no spot flag exists anywhere |
| **cloud-init** | 1 | **ADOPT** | `user_data` is a real droplet-create field AND accepted by the Control Panel |
| Terraform / OpenTofu | 1 | REJECT | No spot attribute; "spot" absent from provider source |
| Pulumi | 1 | REJECT | Bridges the same TF provider via git submodule |
| Ansible (community + official) | 1 | REJECT | Deprecated 2025-11-01; successor also has no spot param |
| Packer | 1 | REJECT | 22-month-stale; MI355X is spot-only so nothing to build on |
| dstack | 1 | REJECT | `spot=False` hard-coded despite real MI355X catalog entry |
| **HF re-download + Xet HP** | 2 | **ADOPT** | Ingress free/uncapped, egress capped+billed; caching pays 1.8x to tie |
| Spaces / snapshot / volumes | 2 | REJECT | 100 GB image cap; 4.96x slower; or no published figure |
| Torrent / P2P | 2 | REJECT | No primary source for any HF torrent path |
| **systemd unit** | 3 | **ADOPT** | Only layer that can brake a restart loop |
| Docker restart policy alone | 3 | REJECT | Backoff resets at 10 s uptime; 6-min load ⇒ never engages |
| Podman quadlet | 3 | REJECT | Capable incl. `/dev/kfd`, but full image copy + runtime swap for zero gain |
| supervisord / k3s | 3 | REJECT | No mount-ordering primitive / K8s disqualifier |
| **Bespoke gate (keep)** | 4 | **ADOPT** | Only thing that can assert logprob margins |
| lm-eval / inspect-ai | 4 | REJECT | No gating exit code |
| promptfoo / DeepEval / Ragas | 4 | REJECT | Require a second model we do not have |
| `VLLM_BATCH_INVARIANT` | 4 | **REJECT** | Silently half-applies on ROCm; 25-35% cost; wrong failure mode |
| **Open WebUI** | 5 | **ADOPT** | Background calls disableable via 6 named vars with early HTTP 200 return |
| LibreChat / Lobe / AnythingLLM | 5 | REJECT | 6-10 services; restrictive license; injects prompts |
| **vLLM `--config` + EnvironmentFile** | 6 | **ADOPT** | Native, verified in this build, exact argv match |
| Compose / Ansible / shell script | 6 | REJECT | Redundant once unit + YAML exist |

## Jobs with no good answer

**Job 1, the create step.** Unsolvable today. No tool can request DO spot capacity
because the API has no field for it. Ceiling is Control-Panel creation + cloud-init.
This is a platform gap, not a tooling gap.

**Job 2.** No tool beats re-download, and this is a *finding*, not a gap — the
asymmetry between free ingress and metered egress makes caching structurally worse.

Jobs 3, 4, 5 and 6 all have answers, now applied or prepared.

## Open WebUI (Job 5, not installed)

One container, SQLite, no GPU, telemetry off in the image, 300 s default timeout (clears
our 112 s median). Its background calls default **ON** and hit the **main** model
(`utils/task.py:18`), which would dilute our shared prefix. Disable all six, and set
`ENABLE_PERSISTENT_CONFIG=False` — it defaults True and snapshots env into SQLite on
first boot, silently ignoring later changes. License is BSD-3 plus a branding clause
exempting deployments under 50 users.

## Image backup (corrected 2026-09-18 13:2x)

The image is **14.5 GB**, not ~57 GB as stated in an earlier draft.

`docker save | zstd` achieved **0.36% compression** (13.5 GiB => 13.5 GiB) because
Docker layers are already compressed. Compression is pure wasted CPU here, so the
backup is now stored **uncompressed** at `/scratch/backup/k3-image.tar` (14.53 GB).

Two mistakes worth recording so they are not repeated:

1. A *completed* zstd run was misread as a stalled one. zstd's final `99.64% (...)`
   summary line looks like progress output, and the file had stopped growing because
   it was **finished**. Check for a running process (`systemctl is-active`) or an
   explicit completion marker - never infer failure from a static file size.
2. The first re-save used `docker save <digest>`, which yields `RepoTags: None`. A
   `docker load` of that archive restores the image **untagged**, and RepoDigests come
   from the registry, so `docker run <digest>` would fail on a fresh box. Always save
   by TAG (`docker save vllm/vllm-openai-rocm:kimi-k3`) and verify the restored image
   ID, which is what `deploy/launch.sh` now does.

Backups run under `systemd-run --unit=... --property=IOSchedulingClass=idle
--property=Nice=10` so they cannot compete with the server for I/O and cannot die
with the invoking shell - the original failure mode.

`deploy/launch.sh` now: restores from tarball -> falls back to registry -> **verifies
the image ID and exits 1 on mismatch** rather than serving unknown bits -> checks
weights exist -> and defers to `systemctl restart k3.service` when the unit is
installed, so it cannot fight systemd for ownership.

---

# Browser management stack (installed AND REMOVED 2026-09-18)

Prometheus (:9091), Grafana (:3000), Dozzle (:8899), Uptime Kuma (:3001) and
Cockpit (:9090) were installed and then removed the same day as **redundant**.
They worked - this was deliberate simplification, not a failure. The tool-link
`<nav>` they added to the :8080 dashboard was removed with them.

**The `amd-metrics-exporter` on 127.0.0.1:5000 stays** - it is DO-provided, not ours.

What listens now:

| Port | Service | Bind | Notes |
|---|---|---|---|
| 8000 | **nginx edge** | 0.0.0.0 | public API. Bearer key on `/v1`, everything else 403 |
| 8001 | vLLM (production) | **127.0.0.1** | no longer public. Also holds the key itself |
| 8080 | custom dashboard | 0.0.0.0 | public, **unauthenticated by choice**, read-only |
| 5000 | amd-metrics-exporter | 127.0.0.1 | DO-preinstalled, `amdgpu-exporter 1.5.1-3~24.04` |

The findings below were expensive to establish and are still true, so they are
kept even though the tools they came from are gone.

---

# CONTEXT WINDOW: 12,800 -> 131,072 (2026-09-18 14:17)

The endpoint was serving `max-model-len: 12800`, tuned for the 12,288+512
benchmark shape. Every agentic client above 12.8k got a hard 400. The model's
native ceiling is `text_config.max_position_embeddings` = **1,048,576**.

## Why long context is affordable here

`text_config.model_type` is **`kimi_linear`** - a hybrid. Of 93 layers, only the
24 in `linear_attn_config.full_attn_layers` are full attention (MLA); the other
**69 are KDA** (linear/delta attention, constant state per sequence regardless
of length). So only ~26% of layers carry length-proportional KV.

vLLM consequently overrides `block-size`: *"Setting attention block size to 768
tokens to ensure that attention page size is >= mamba page size"*
(`interface.py:911`), then pads the mamba page by 8.68% to match. **Passing
`block-size: 128` was a no-op** and has been removed from config.yaml.

## What was measured, and the crash

Raising the window also **increased** usable KV, because `max-model-len` shifts
how the hybrid allocator balances mamba state against attention pages:

| config | KV tokens | conc. @12.8k | conc. @ full len |
|---|---|---|---|
| 12,800 @ 0.92 | 1,696,278 | 132.5x | 132.5x |
| 262,144 @ 0.96 | 2,732,926 | 213.5x | 10.4x |
| **131,072 @ 0.92** | **2,256,363** | **176.3x** | **17.2x** |

So the benchmark shape gained headroom (132x -> 176x) while the window grew 10x.

**262,144 @ 0.96 is NOT safe and was reverted.** A ~250k prompt killed the engine
mid-prefill at `num_computed_tokens=186,624` with `EngineDeadError`. Diagnosis:

- **Not** KV exhaustion - `kv_cache_usage=0.084` at the moment of death.
- **Not** host OOM - no oom-kill, 77 GiB of 2015 GiB used.
- No HIP error was logged; the worker died silently and the parent only saw
  `RuntimeError: cancelled` from `shm_broadcast.py:703`, which is downstream noise.
- The real evidence was in **dmesg**: `svm_range_evict_svm_bo_worker [amdgpu]`
  fired at the exact second of the crash. The driver was **evicting GPU buffer
  objects** - VRAM headroom exhaustion. `0.96` left only ~11.5 GiB/GPU spare and
  attending over 186k tokens of KV needs more working room than that.

At `0.92` there is ~16 GiB/GPU free (measured, `rocm-smi`), and 128k is capped
below the failure point, so **an oversized request now returns a clean 400 and
the engine survives** - verified, `restarts=0` afterwards.

**Lesson: when a vLLM worker dies with no Python traceback, check `dmesg` for
`amdgpu` eviction before assuming a KV or scheduler bug.**

## Verified agentic behaviour

`/scratch/ctx-test.py`. Needle retrieval - a unique code buried at a controlled
depth, so "accepted the prompt" is distinguishable from "attended over it":

| prompt tokens | depth | latency | retrieved |
|---|---|---|---|
| 18,783 | 0.50 | 5.6s | yes |
| 60,785 | 0.50 | 16.3s | yes |
| 116,787 | 0.10 | 15.1s | yes |
| 116,787 | 0.50 | 13.8s | yes |
| 116,788 | 0.95 | 10.7s | yes |
| 186,786 | - | - | clean HTTP 400, engine survived |

Tool calling works end to end, including the multi-turn loop that actually
matters (call -> tool result fed back -> model uses the result):
`finish_reason=tool_calls`, then `read_file({"path": "config/db.yaml"})`, then
correct use of the injected port `54329`. `tool-call-parser: kimi_k3` and
`reasoning-parser: kimi_k3` were already configured.

**Gotcha when testing a reasoning model:** `max_tokens=48` produced empty
`content` and read as a retrieval failure - K3 spent the whole budget on
reasoning. Give long-context probes >=256 tokens and read `reasoning_content`
as well as `content`.

Streaming through the nginx edge is genuinely incremental (120 SSE chunks,
first at 0.21s, spread 2.24s, `X-Accel-Buffering=None`), and 6 concurrent
requests through the edge all returned 200 in ~0.97s.

## Agentic capability surface - all verified 2026-09-18

`/scratch/agentic-test.py`. These are the features a coding-agent harness
actually depends on, as opposed to "the request returned 200":

| capability | result | note |
|---|---|---|
| parallel tool calls in one message | works | 2 calls returned together |
| `tool_choice` forced named function | works | honoured exactly |
| `tool_choice: "none"` | works | suppresses calls |
| 31 tool schemas in one request | works | picks the right tool, 1,972 prompt tokens |
| multi-turn tool loop | works | call -> tool result -> uses result |
| `response_format: json_object` | works | |
| `response_format: json_schema` strict | works | enums + arrays respected |
| **vision / image input** | **works** | read `R7X-42Q` off a PNG, 217 mm tokens |
| stop sequences | works | |
| `/v1/messages` (Anthropic shape) | works | matters for Claude Code |
| extra gateway fields (`metadata`, `user`) | tolerated | no 400 |
| SSE streaming | works | incremental, `X-Accel-Buffering=None` |

Multimodal is genuinely enabled - the startup log confirms *"Encoder cache will
be initialized with a budget of 16817 tokens, and profiled with 1 image items"*.
Structured-output backends present: `xgrammar` and `llguidance 1.7.6`.

### Two findings that look like bugs and are not

1. **`reasoning_content` is always empty, and that is correct.** K3 wraps
   thinking in `<|open|>think<|sep|> ... <|close|>think<|sep|>` (3-token
   markers, unlike K2's single `<think>`). Per
   `vllm/reasoning/kimi_k3_reasoning_parser.py:20-22`, when thinking is disabled
   *"the parser returns every delta as normal content; there is simply no think
   channel to extract."* Verified across `chat_template_kwargs` of unset,
   `{"thinking": true}` and `{"thinking": false}`: `reasoning_content` was empty
   in all three and **`content` was a clean final answer every time**. No
   chain-of-thought leaks into `content`, which is what strict clients need.
   `thinking: true` does spend more tokens (401 vs 93 completion tokens) without
   exposing them.

2. **Structured output and long-context probes need a generous `max_tokens`.**
   `json_schema` at `max_tokens: 200` returned empty content and read as a
   server failure; the completion actually needed 232. Likewise a needle probe
   at `max_tokens: 48` produced empty `content` because the model spent the
   budget before answering. **Neither was a server fault.** Budget >=512 on
   structured or long-context calls before concluding anything is broken.

## Raising it further

128k is a verified stopping point, not the ceiling. To go higher, the >186k
shape must be re-tested **under concurrency**, since several simultaneous long
prefills multiply the workspace demand that already caused one crash at
single-request load. Do not raise `gpu-memory-utilization` to buy the room.

---

# AUTHENTICATED PUBLIC ENDPOINT (2026-09-18 13:55)

The endpoint is deliberately public again, but behind auth. Timeline: it was
internet-reachable with **no** authentication; `ufw` closed it at 13:31; it was
reopened at 13:52 behind an nginx edge.

## Why `--api-key` alone is NOT sufficient - read this before simplifying

vLLM's `--api-key` guards only three path prefixes:

    entrypoints/serve/utils/server_utils.py:42
    GUARDED_PREFIX = ("/v1", "/v2", "/inference")

This build exposes **25 routes**. Eleven fall outside that tuple, and the
dangerous one is not hypothetical - measured before the fix:

    POST /invocations         -> 200   # full inference, no credentials
    POST /scale_elastic_ep    -> 400   # reached the handler; can alter the deployment
    GET  /metrics             -> 200   # all operational telemetry
    POST /tokenize|/detokenize|/generative_scoring, GET /load|/version|/ping

`/invocations` is the SageMaker-compatible alias for the same engine. Opening
port 8000 with only `--api-key` set would have handed out **free inference on a
$36/hr box** via a single unguarded POST. This is the entire reason nginx exists
here; do not remove it and rely on the flag.

## Shape

    internet -> :8000 nginx  --(allowlist + Bearer check)-->  127.0.0.1:8001 vLLM
                :8080 dashboard (direct, read-only, no auth)

- **`/v1/*`** requires `Authorization: Bearer <key>`, else 401.
- **`/metrics`, `/health`, `/ping`** are `allow 127.0.0.1; deny all` - the :8080
  dashboard keeps scraping them unchanged, external callers get 403.
- **Everything else returns 403**, including `/invocations` and `/scale_elastic_ep`.
- vLLM **also** holds the key (`VLLM_API_KEY` in `vllm-k3.env`, mode 600) so a
  bypass of nginx still hits 401 on `/v1`. Defence in depth, not the only line.

Config: `/etc/nginx/conf.d/k3.conf` (mode 640). Key: `/scratch/deploy/api-key.txt`
(mode 600). `api_server.py:307` reads `VLLM_API_KEY`, so the key does **not** need
to sit in `config.yaml`, which is world-readable.

Two nginx settings that matter for this workload, both non-obvious:
- `proxy_buffering off` + `proxy_read_timeout 900s`. A 12,288-in/512-out request
  runs ~112s median; nginx's 60s default would cut streams off mid-generation.
- `worker_processes 16`, down from `auto` (=192, one per vCPU). 192 workers each
  holding `keepalive 32` to a single loopback upstream is pure waste.
- `map_hash_bucket_size 128` is **required**: `"Bearer sk-k3-<48 hex>"` exceeds the
  64-byte default and nginx refuses to start with `could not build map_hash`.

## Verification

`/scratch/deploy/verify-auth.sh` - 25 assertions, all passing. It asserts the
specific unguarded routes return 403, that `/v1` needs the key, that `/metrics`
is local-only, and that vLLM is bound to loopback.

Confirmed from a **genuinely external** client (source IPs 34.96.49.38 /
34.96.49.71 in the nginx access log): `/v1/models` 401, `/invocations` 403,
`/metrics` 403, `:8080` served the dashboard.

**Testing gotcha, hit twice now:** curl-ing your own public IP *from the box*
proves nothing. `-A ufw-before-input -i lo -j ACCEPT` accepts everything routed
via loopback, and traffic to a local address takes that path even when the
address is a public one on eth0. Probe from off-box. Most third-party fetchers
(codetabs, allorigins) return 522 on non-standard ports and cannot be used;
`https://r.jina.ai/http://<ip>:<port>/` does work and reports the status code.

Also: `ifconfig.me` reported `201.79.29.187`, which whois attributes to Claro
Brazil. That **is** correct - DO metadata confirms it as this droplet's public
IPv4 in mem1. Do not chase the geolocation.

## Current firewall

    ufw-user-input: --dport 22 ACCEPT      # ssh, key-only
                    --dport 8000 ACCEPT    # nginx edge, Bearer key required
                    --dport 8080 ACCEPT    # dashboard, unauthenticated
    INPUT policy DROP

:8080 is open unauthenticated **by explicit decision**. It is safe to expose only
because the server is GET-only (no `do_POST`) and its single `subprocess` call
takes a fixed `rocm-smi` argument list from internal callers with no `shell=True`
and nothing request-derived - so it leaks telemetry (throughput, HBM, power, model
name, KV capacity) but grants no control. If that ever changes, put it behind the
same nginx key.

## AMD exporter: profiler hazard corrected

DO's shipped `/etc/metrics/config.json` had `"ProfilerMetrics": {"all": true}` plus 50
`GPU_PROF_*` fields. AMD documents this as taking the GPU's **single** hardware
profiler slot: *"the current hardware limits a single profiler instance to be run at
any given time."* It was configured-but-inactive (`GPU_PROF_*` count 0 after 4h
uptime), so the config was corrected to `all: false` with the 50 fields removed and
the exporter **deliberately NOT restarted** - the fix lands at next boot without
touching a live GPU. Original saved at `/etc/metrics/config.json.do-original`.
The exporter itself **remains installed and running** - it is DO-provided and was
not part of the removed stack.

## gfx_activity is NOT a usable utilization signal here

`amd_gpu_gfx_activity` reads a flat **100% on all 8 GPUs** even at near-idle, and
`rocm-smi --showuse` independently agrees - so it is the VF/SR-IOV driver reporting,
not an exporter bug. **Do not build a "GPU utilization" panel.** Use `amd_gpu_used_vram`,
`amd_gpu_power_usage` (434-557 W observed) and `amd_gpu_clock` instead, which do carry
information. `amd_gpu_temperature` is absent - a VF does not expose host thermal sensors.

## Metric names: do NOT rename vLLM's shipped dashboard

**This build emits the NEW metric names** - `vllm:kv_cache_usage_perc` and
`vllm:inter_token_latency_seconds`, *not* `gpu_cache_usage_perc` /
`time_per_output_token_seconds`. vLLM's official dashboard at
`examples/observability/prometheus_grafana/grafana.json` is therefore correct **as
shipped** and needs no renaming. An earlier stale metric list caused a
wrong-direction patch that broke two working panels.

Worth keeping for any future dashboard - the prefix-cache ratio vLLM does not ship:

    sum(rate(vllm:prefix_cache_hits_total[5m])) / sum(rate(vllm:prefix_cache_queries_total[5m]))

Measured scrape cost: vLLM `/metrics` **4-5 ms**, AMD exporter **1.4 s** (its PCIe
bandwidth sampler dominates). vLLM renders its full registry every scrape and ignores
Prometheus `name[]` filters (vllm#53140), so scrape cost cannot be reduced
server-side - use **30s intervals, not 5s**, for anything that scrapes it.

## Cockpit install: the metapackage is a trap (if anyone reinstalls)

`apt-get install cockpit` **would install and start NetworkManager** on this
netplan/systemd-networkd box (verified by simulation on this host: 2 matches for
`Inst network-manager`), because its `Recommends: cockpit-networkmanager` depends on
`network-manager (>= 1.6)`. The safe install, verified to pull **0**:

    apt-get install -y --no-install-recommends -t noble-backports \
      cockpit-ws cockpit-bridge cockpit-system cockpit-storaged

Use **noble-backports (362)**, not the release pocket (314, ~53 releases stale), and
write the `cockpit.socket` drop-in BEFORE apt - postinst starts the socket, and without
the drop-in it binds `0.0.0.0:9090`.

## Dozzle: the `:ro` docker-socket mount is security theatre

Per Dozzle's own docs, `:ro` marks the socket file read-only on disk but the API still
permits create/delete. Applies to anything that mounts the docker socket, not just Dozzle.

`k3`'s log driver had **no max-size**, so it grew unbounded. `k3.service` now sets
`--log-opt max-size=100m --log-opt max-file=5`, applying at the next restart.

## Health probe design (for whatever probes k3 next)

A ~20-token probe against `/v1/chat/completions` **cannot** dilute the prefix cache.
vLLM only hashes *complete* blocks (`vllm/v1/core/kv_cache_utils.py:829` - "We only hash
full blocks") and block size here is 128, so a short probe generates zero block hashes
and inserts nothing. Recommended: `/health` every 60s for liveness, plus a keyword probe
on `/v1/chat/completions` every 5-10 min to catch "returns 200 but the model is wedged",
which `/health` and `/v1/models` both miss.

## Rejected

- **Homepage** - redundant. No Dozzle widget, `customapi` is JSON-only (no PromQL), and
  host metrics need a second Glances container.
- **LiteLLM** - not installed. Forwarding IS byte-identical with the `openai/` prefix
  (empirically verified with an echo server), but it **requires PostgreSQL** for virtual
  keys, defaults to **one worker** against 427 concurrent, and the `hosted_vllm/` prefix flattens list-form
  assistant content (`llms/hosted_vllm/chat/transformation.py:223-225`) which would change
  every downstream token hash. Virtual keys and budgets ARE free/MIT; only key
  *regeneration*, key tags and per-model budgets are enterprise. For auth alone, nginx +
  `--api-key` is strictly safer since nginx does no JSON parsing at all.
