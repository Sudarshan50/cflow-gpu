# K3 encoder-cache / Mamba-alignment deadlock — 2026-09-21

`reproduce.py` consolidates the two CPU metadata reproductions from the read-only
source investigation. It reproduces running-request and waiting-admission
zero-progress states, a single-image control, and encoder-cache abort cleanup.

**Capacity-only proposal review: BLOCKED.** Keeping compute at 16817 while
raising cache capacity to 262144 still deadlocks when a decoder prefix hits but
the corresponding encoder entries are absent. The extended CPU checks preserve
that counterexample alongside passing large-cache controls. A deployable
`vision_scheduler.py` was not created under the requested stop-on-blocker rule.

**Scope:** these are constructed metadata states proving a source-level defect.
They neither identify nor replay historical requests, and do not establish the
exact cause of the earlier incident. Every request ID, image identifier, offset,
and placeholder mask in the harness is synthetic.

## Run

Use host Python 3.12 and access to the already-running `k3` container:

```bash
python3 -B /root/cflow-gpu/experiments/2026-09-21-encoder-deadlock/reproduce.py
```

The script performs one read-only `docker inspect` to resolve the **current PID**
and verify the image pin. It reads installed source through
`/proc/<pid>/root/usr/local/lib/python3.12/dist-packages/vllm`, prints source-file
SHA-256 fingerprints, and AST-extracts the actual methods/classes with deferred
type annotations. Only Python standard-library dependencies are imported.

Execution uses integer/list metadata: no pixel allocation, tokenizer, model,
vLLM/torch/transformers import, GPU initialization, inference, HTTP request,
container exec, file write, service restart, engine patch, or portal operation.
Assertions are the checks; run without `-O`. All output goes to stdout.
Passing assertions include successful reproduction of the proposal's blocker;
the final stdout line explicitly reports `REVIEW BLOCKED`.

## Pinned provenance and limits

- Image: `johnqin2025/kimi-k3-dspark@sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8`.
- vLLM startup version: `v0.1.dev19253+g5f76ae224.d20260727`.
- Checkpoint: `moonshotai/Kimi-K3`, snapshot `f831ab66814297da540d832a5235f8e904f29d06`.
- Startup-pinned decoder budget: **4096**; base scheduler encoder fields:
  **4096 / 4096**; effective encoder compute/cache budgets: **16817 / 16817**.
- Startup-pinned Mamba mode: **`align`**, block size **768**. Chunked MM enabled;
  no EAGLE or external encoder-cache connector in these cases.
- The 06:23:59 UTC startup selected `align`; 06:26:23 selected 768-token blocks;
  06:26:27 logged the 16817-token encoder budget.

The harness reads the mounted checkpoint's `preprocessor_config.json`: patch
size 14, merge size 2, input patch limit 65536, side patch limit 512, and no fixed
output-token count. Installed CPU sizing methods reproduce a maximum of **16817
embeddings** at **1861×7041**, and **10000 embeddings** for **2800×2800**.
Decoder/block settings are explicit startup-derived fixtures, not live scheduler
introspection. Alternative capacity/alignment controls change only local objects.

## Cases and expected proof

Both-image cases use distinct 10000-embedding items, each with a synthetic
10010-token placeholder (nine leading wrapper slots and one trailing slot):

| Item | Placeholder range |
| --- | --- |
| Image 1 | `[20, 10030)` |
| Image 2 | `[10040, 20050)` |

There are 1000 trailing metadata tokens. Precomputed cumulative embedding masks
exercise the installed `PlaceholderRange` methods without constructing tensors.

1. **Running:** progress reaches **9984**. Image 1 remains pinned and leaves
   **6817** reclaimable slots, insufficient for image 2. The encoder gate allows
   **56** more decoder tokens, but alignment floors the endpoint back to 9984:
   **zero progress**, repeated three times with the same held reference.
2. **Waiting:** seed an unreferenced cached image from a synthetic completed
   request, then supply a **9984-token prefix-cache hit**. Admission re-pins that
   image for `waiting-request` and again yields **56 → 0** tokens. The request's
   committed computed count stays zero and the touched cache reference persists;
   this constructs the zero-running waiting-head variant.
3. **Single image:** a **16817-embedding** image completes metadata prefill at
   token **17847**, with full reclaimable encoder capacity afterward.
4. **Isolation controls:** the same two-image metadata completes at token
   **21050** without alignment, or with local cache capacity **20000**.
5. **Abort cleanup, both stuck cases:** invoke installed
   `EncoderCacheManager.free(request)`, the hook used by `_free_request` on abort.
   Reclaimable capacity returns to **16817**, request references disappear, and
   allocating space for a fresh maximum-sized image evicts the old entry.
   This checks cache cleanup after a delivered abort, not cancellation delivery.

The runner mirrors the relevant commit/break order around the extracted methods;
it does not instantiate the complete scheduler or model runner. Waiting-state
queue blocking is grounded in the caller's `break` below.

## Review of the 262144-slot capacity-only proposal

The tested local change copies the scheduler-config fixture and sets only
`encoder_cache_size=262144`. Installed budget helpers confirm:

| Quantity | Original | Proposed |
| --- | ---: | ---: |
| Effective encoder compute | 16817 | 16817 |
| Effective encoder cache slots | 16817 | 262144 |
| Decoder batch tokens | 4096 | 4096 |
| Encoder profiling budget (`min(compute, cache)`) | 16817 | 16817 |
| Maximum-sized images per profiling batch | 1 | 1 |

Positive CPU controls pass: the two-image running case advances, and a shared
cache serves **12 synthetic requests**, six initially running and six initially
waiting with warm encoder entries and decoder prefix hits. Their full rendered
prompt reservation is **252600** tokens and total image cost is **240000**
embeddings. The test ledger rejects a thirteenth 21050-token request and an
understated reservation; it releases reservations at simulated prefill
completion after encoder references are freed. Global per-step compute and
decoder budgets remain 16817 and 4096. This ledger models the parent's admission
contract; it is not a test of the separately developed gateway implementation.

### Counterexample satisfying the proposed reservation bound

Use the same two 10000-embedding images, **empty encoder cache of 262144 slots**,
and a **9984-token decoder prefix-cache hit**. Reserve the full **21050-token**
prompt before admission. All 20000 embeddings fit within both reservation and
cache capacity, yet each attempt produces:

```text
encoder_inputs_planned = [0]
encoder_compute_remaining = 6817
after_encoder_gate = 56
after_mamba_alignment = 0
allocated_encoder_entries = 0
cache_freeable = 262144
```

The first image consumes 10000 of the planned compute budget. The second image
cannot fit the remaining **compute** budget, so the encoder gate truncates the
chunk to 56 tokens. Alignment removes all progress before the first image's
planned allocation is committed. The next attempt starts identically. A prior
image's unreferenced encoder entry can be evicted while its decoder KV prefix
remains reusable; reserving current prefills does not guarantee a warm encoder
entry for every decoder prefix hit.

Consequently the requested init-only `AsyncScheduler` wrapper would not provide
a complete multi-image solution. No scheduler-class configuration change is
issued. The proposed import-path CLI mechanism itself is supported at
`engine/arg_utils.py:1464–1469` and `config/scheduler.py:170–191`; the unresolved
condition is scheduler progress, not class loading.

### Runner allocation and memory review

Installed `GPUModelRunner.encoder_cache` starts as an empty dictionary
(`v1/worker/gpu_model_runner.py:571–572`). Encoder outputs are inserted by
reference at `:2986–2996,3203–3210` and evicted at `:1177–1183`. The CPU test uses
those actual insertion/lookup/eviction methods with an ordinary object sentinel.
No cache-size-shaped preallocated embedding buffer was found. The decoder input
embedding buffer uses `max_num_batched_tokens`, still 4096 (`:505,798–801`), and
profiling uses the minimum of compute and cache budgets
(`multimodal/encoder_budget.py:195–196`; runner `:6430–6466`). A scheduler-local
config copy would also leave worker configuration and compilation inputs intact.

Checkpoint hidden width is **7168**, with **BF16 / 2 bytes** per element:

- Proposed maximum logical payload: `262144 × 7168 × 2 = 3758096384` bytes,
  **3.500000 GiB per rank**.
- Additional logical payload over 16817 slots: **3.275469 GiB per rank**.
- The bounded 06:27:34 startup journal reported approximately **27.11–27.97 GiB**
  of additional memory allowance across ranks: its fully-utilized KV-memory
  recommendation minus already allocated KV memory. This is historical startup
  headroom, not a new live GPU measurement. The logged 43.11 GiB KV allocation
  on TP0 is already allocated, not spare memory.

**3.5 GiB bounds logical embedding payload, not physical allocator residency.**
`model_executor/models/kimi_k25_vit.py:838–849` splits a batched projector output
into views, which the runner stores by reference. Surviving entries can retain
storage for evicted sibling views; allocator overhead and in-flight buffers also
lie outside the slot-count calculation. A strict physical <=4 GiB guarantee is
therefore not established by a capacity-only change.

## Source paths

Paths are relative to the installed `vllm` directory; lines refer to the source
inspected in the pinned image:

| Mechanism | Source |
| --- | --- |
| Base encoder fields → raised effective budgets | `config/scheduler.py:238–239`; `v1/core/encoder_cache_manager.py:300–320`; `v1/core/sched/scheduler.py:226–237` |
| Profile maximum image dimensions | `models/kimi_k3/common/mm_preprocess.py:153–187,202–215` |
| Cache hit takes a reference; allocation gates | `v1/core/encoder_cache_manager.py:115–122,159–173` |
| Encoder rejection truncates decoder progress | `v1/core/sched/scheduler.py:1489–1508` |
| Mamba alignment rounds progress to zero | `v1/core/sched/scheduler.py:390–401` |
| Running zero-token skip / waiting zero-token break | `v1/core/sched/scheduler.py:529–568,869–896` |
| Waiting KV-failure cleanup occurs later | `v1/core/sched/scheduler.py:940–947` |
| Consumed image reference release threshold | `v1/core/sched/scheduler.py:2024–2055` |
| Delivered abort → encoder reference cleanup | `v1/core/sched/scheduler.py:2144–2224`; `v1/core/encoder_cache_manager.py:217–254` |

Operator-reported mitigation at handoff: gateway `K3_MAX_REQUEST_IMAGES=1` from
06:38 UTC; portal channel 285 remains disabled. These are operational context,
not actions or requests exercised by this reproducer.
