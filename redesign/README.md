# redesign/ — capacity-first serving work

Implementation of [`docs/SYSTEM-DESIGN.md`](../docs/SYSTEM-DESIGN.md).

Everything here is built and validated **without a GPU**. The box costs money
the moment it boots, so the rule for this directory is:

> No GPU session without a pre-written question, a pre-written pass/fail, and a
> rollback. "Boot it and see" is how GPU budgets disappear.

## Phase Z — zero-GPU deliverables

| ID | File | Status | What it answers |
|---|---|---|---|
| **Z0** | `capacity_model.py` | done | Is the KV pool 8× deflated by MLA-under-TP replication? |
| **Z1** | `cache_salt_audit.py` | todo | Are clients sending per-key `cache_salt`, which drives the hit rate to 0% by design? |
| **Z2** | `probe_dedup_support.py` | todo | Does DCP / DP-attention support a hybrid KDA+MLA model on ROCm *today*, on either engine? |
| **Z3** | `trace/` | todo | Capture and replay the real production request distribution. |
| **Z4** | `gateway/` | todo | Tenancy + backpressure layers, against a mock endpoint. |
| **Z5** | `gate_kv_quant.py` | todo | Extend the correctness gate to catch quantized-KV silent garbage. |
| **Z6** | `experiments/` | todo | Pre-registered GPU session definitions. |

## Phase G — GPU sessions

Defined in `docs/SYSTEM-DESIGN.md` §11. Each is time-boxed to one question.
Nothing in Phase G runs until the Phase Z item it depends on is green.

## Z0 result

`capacity_model.py` confirms the §2 finding offline:

```
measured pool             2,295,266 tokens
predicted if REPLICATED   2,369,005   (error 3.2%)
predicted if DE-DUPED     18,952,040  (error 725.7%)
VERDICT                   REPLICATED
```

Cross-check: the model's sustainable concurrency at the observed mean prompt
length is **78 sequences**; production steady state is 68–76 in system. The
configured `max-num-seqs: 512` is **6.6×** that.

Headroom if the replication is fixed:

| Scenario | Pool | vs today | max-num-seqs |
|---|---|---|---|
| Today (TP8 replicated, bf16) | 2.37M | 1.0× | 78 |
| + fp8 KV (A2) | 4.74M | 2.0× | 155 |
| + 512 GB CPU L2 tier (A3) | 4.68M addressable | 2.0× | 78 |
| **+ KV de-duplication (A1)** | **18.95M** | **8.0×** | **591** |
| + de-dup and fp8 (A1+A2) | 37.90M | 16.0× | 1111 |

Run it:

```bash
python3 redesign/capacity_model.py            # full report
python3 redesign/capacity_model.py --sweep    # concurrency vs prompt length
python3 redesign/capacity_model.py --verify   # hypothesis test only, exit 0 = replicated
python3 redesign/capacity_model.py --json     # machine-readable
```

No dependencies. Python 3.9+.

## Caveat on the inputs

`PROMPT_DISTRIBUTION` in `capacity_model.py` comes from a **229-request
sample**. Every sizing number derived from it is provisional until Z3 replaces
it with a real week of traffic. The replication finding itself does not depend
on the distribution — only the concurrency numbers do.
