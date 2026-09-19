# Kimi-K3 throughput investigation — 2026-09-19

Investigation into further throughput headroom for the 8× MI355X Kimi-K3
deployment, plus a partially-completed A/B campaign. Companion to
[`docs/K3-DEPLOYMENT.md`](../../docs/K3-DEPLOYMENT.md); where that file records
the 2026-09-18 tuning campaign, this one records what changed since and what
the next round of tuning should do differently.

**The campaign was stopped early by operator request.** Only the baseline
variant completed. Everything in §1–§4 is verified and standalone; §5 is one
data point rather than a comparison.

---

## 1. The premise of the existing tuning no longer holds

`K3-DEPLOYMENT.md` §2.1 names the 85% shared prefix as the single largest
factor in reaching 2.5M TPM, and derives the concurrency ceiling from it. On
2026-09-19 the live prefix cache hit rate had collapsed.

Measured over two 60-second windows against live production traffic:

| | window 1 | window 2 |
|---|---|---|
| total throughput | 232,731 TPM | 544,326 TPM |
| prefix cache hit rate (marginal) | 6.0% | 12.9% |
| preemptions | 9/min | 4/min |

KV cache usage climbed from 94% to 97.9% during observation, with 22–25
requests queued on `reason="capacity"` and 46–51 running. By the alerting rule
this repo already recommends (§11.5 — "`rate(vllm:num_preemptions_total[5m]) > 0`
means work is actually being discarded"), the box was in a degraded state.

The mechanism is a heavy tail. Prompt-size distribution over 229 requests:

| bucket | share |
|---|---|
| ≤ 10k tokens | 53% |
| ≤ 20k | 69% |
| ≤ 50k | 81% |
| ≤ 100k | 93% |
| ≤ 200k | 97% |
| > 200k | 3% |

A single >200k request consumes up to ~9% of the 2,295,266-token KV pool. That
tail evicts the shared prefix the other 93% depend on, which raises their KV
demand, which drives more eviction. It is self-reinforcing, and it explains a
6–13% hit rate on traffic that is not intrinsically low-reuse.

**Implication for tuning:** the documented benchmark profile (85% prefix,
12.8k prompts) cannot detect any of this, because it never fills KV. Tuning
against it will keep producing changes that do not help production.

---

## 2. v0.29.0 upgrade: the AITER flag contract moved, and §11.1 predicted the
   wrong name

§11.1 flags that the AITER contract changed between builds and warns that
carrying `vllm-k3.env` forward verbatim is "the most likely way to recreate the
§7.3 silent-garbage failure". That warning is correct. The specific variable it
predicts is not.

Enumerating `vllm.envs` in both images:

| variable | kimi-k3 dev build | v0.29.0 |
|---|---|---|
| `AITER_SITUV2_A8W4` | **recognised** by `vllm.envs` | **absent** from `vllm.envs` |
| `VLLM_ROCM_USE_AITER_MOE_SITUV2` | absent | absent (§11.1 predicted this name) |
| `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4` | absent | **present, defaults to `False`** |

The real v0.29.0 control is `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`. It is read at
`vllm/envs.py:1262`, cached into `_MOE_SITUV2_A8W4` at `vllm/_aiter_ops.py:1760`,
and gates a8w4 dispatch at `_aiter_ops.py:1905`.

Two layers are involved, which is what makes this subtle:

- **vLLM layer** — `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`, new in v0.29.0,
  default off.
- **aiter library layer** — `AITER_SITUV2_A8W4` is still read directly from
  `os.environ` at `aiter/fused_moe.py:629`, and its own source comment says it
  *overrides* the intended path.

So upgrading with the current env file unchanged would leave vLLM's a8w4
dispatch **disabled** while the aiter side still expects the interleaved
layout. The likely outcome is losing the 2.06× decode speedup §2.2 attributes
to A8W4, and the benchmark would read as an upgrade regression rather than a
misconfiguration.

Also load-bearing: `AITER_BF16_FP8_MOE_BOUND` defaults to **256** in v0.29.0's
aiter (`fused_moe.py:604`), not 0. That is precisely the bf16/fp8 boundary
behind the §7.3 garbage-output failure, so pinning it to 0 remains mandatory.

### Corrected env for v0.29.0

```diff
-AITER_SITUV2_A8W4=1
+VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1
 AITER_BF16_FP8_MOE_BOUND=0          # keep: v0.29.0 aiter defaults this to 256
```

This is **untested at runtime** — the campaign was stopped before the upgrade
variant ran. It is a static-analysis result, and the correctness gate must pass
before it is trusted.

---

## 3. KV offloading is viable on the current image

§10 dismisses fp8 KV cache, and §7.4's kernel-level reason (TP=8 gives 12
heads/rank, below the 16-head threshold) still stands. But its secondary
argument — "KV is not the constraint at 49%" — is now stale, since KV runs at
94–98%.

CPU KV offloading is the alternative, and the box has ~1.9 TiB of essentially
idle host RAM against 61 GiB of HBM KV per GPU.

The risk was that a KV connector would disable the hybrid KV cache manager,
which K3 requires — vLLM warns that "hybrid SSM models ... require HMA and will
fail at startup without it" (`vllm/config/vllm.py`). Verified this does not
apply:

```
OffloadingConnector:       HMA=True  (subclasses SupportsHMA)
SimpleCPUOffloadConnector: HMA=True
```

`--kv-offloading-size <GiB>` wires to `OffloadingConnector` with
`cpu_bytes_to_use` and `kv_role=kv_both` (`vllm/config/vllm.py:894-922`). No
upgrade required; it works on the current build. **Not yet benchmarked.**

---

## 4. Benchmark methodology

The documented §5.3 profile is unsuitable for this work: at 85% prefix reuse
and 12.8k prompts it leaves KV at ~49%, so no KV-pressure change can register.

The profile used here keeps long prompts (the actual stressor) while holding
the hit rate at a realistic operating point rather than the degraded live one:

```
--random-prefix-len 18432   # 24 x 768, aligned to the block size the hybrid
                            # model forces, so no partial block is wasted
--random-input-len  7168
--random-range-ratio 0.5    # unique part in [3584, 10752]
--random-output-len 600
--request-rate inf --max-concurrency 256 --num-prompts 200
```

Mean total input 25,600 tokens; nominal hit rate 72%. **Measured 70.9%** over
the baseline benchmark window, computed from the
`prefix_cache_hits_total`/`prefix_cache_queries_total` counter delta rather
than the cumulative gauge, which warmup pollutes.

### Two methodology traps hit during this work

1. **A `docker exec`'d benchmark outlives a kill of its parent.** `pkill` on the
   driver left a 200-request benchmark running inside the container; a
   subsequent probe measured its own traffic mixed with the survivor's and
   reported 19.4% instead of 67.6%, and took 270s instead of 17.6s. `driver.sh`
   now clears both sides of the container boundary before every variant.
2. **Cumulative `prefix_cache_*` counters are not a per-run measurement.**
   Always difference them across the run window.

---

## 5. Baseline result (only completed variant)

Current production config, current image, profile above:

| metric | value |
|---|---|
| successful requests | 200 / 200 |
| duration | 212.50 s |
| total input / generated tokens | 5,166,427 / 122,059 |
| **total token throughput** | **24,886 tok/s ≈ 1.49M TPM** |
| output token throughput | 574.39 tok/s |
| request throughput | 0.94 req/s |
| median / p99 TTFT | 61,524 ms / 159,881 ms |
| median / p99 TPOT | 199.63 ms / 394.96 ms |
| median ITL | 79.44 ms |
| prefix hit rate (window) | 70.9% |
| correctness gate | **PASS** (`gate_rc=0`) |
| engine startup | 262 s |
| KV pool | 2,295,266 tokens |

TTFT is large because the profile deliberately saturates at 256 concurrency;
it is a throughput benchmark, not a latency one. Compare TTFT only across
variants run with this same profile, never against §1's numbers.

---

## 6. Status and next steps

Completed: baseline. Not run: `kvoffload`, `longprefill`, `async`, `batch16k`,
and the v0.29.0 upgrade. All configs are in `configs/` and pre-validated
against **both** images by `validate.py`, so they can be resumed with
`./run_all.sh`.

Ranked by expected value for the workload actually running:

1. **Diagnose the prefix-cache collapse.** Largest and cheapest win. Candidate
   causes: clients sending a per-API-key `cache_salt` (§11.2 verified this
   drops the rate to 0% and warns to salt per trust boundary, never per key);
   genuinely diverse traffic; or the eviction spiral in §1. Only the last is
   fixed by tuning.
2. **KV offloading** (§3) — verified viable, unbenchmarked.
3. **Cap the long-prefill tail** with `--long-prefill-token-threshold`, paired
   with lowering `max-num-seqs` from 512, which the pool cannot sustain at
   current prompt sizes (hence the preemptions).
4. **`--async-scheduling`** — note it has a documented interaction with hybrid
   models when `mamba_cache_mode != "align"`.
5. **v0.29.0 upgrade** with the corrected env from §2, gated on the gate.

Do not revisit: fp8 KV cache (§7.4 kernel limit), `torch.compile` (§2.3,
re-confirmed inert in the current logs), `gpu-memory-utilization` 0.96 (crashed
the engine previously), or lowering `max-model-len` to reclaim KV — pages are
allocated on demand, so it reclaims nothing and would reject the 3–7% of real
traffic above 100k.

---

## 7. Contents

| path | what |
|---|---|
| `driver.sh` | per-variant runner: apply config, restart, wait ready, warmup, bench, hit rate, gate |
| `run_all.sh` | sequential campaign over the current-image variants |
| `upgrade.sh` | switches the pinned image via a systemd drop-in; rollback is one file removal |
| `validate.py` | parses every config through vLLM's real CLI path on both images |
| `redact.py` | strips live secrets and customer IPs; exits non-zero if any survive |
| `configs/` | the five serving configs under test |
| `results/baseline/` | baseline evidence: bench output, startup log, gate result, hit rate |
| `transcript/` | redacted agent transcript of the investigation |

> Every file here passed `redact.py`. Env files are excluded entirely: they
> carry the live API key and contain nothing reproducible.
