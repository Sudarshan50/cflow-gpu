# redesign/ — capacity-first serving work

Implementation of [`docs/SYSTEM-DESIGN.md`](../docs/SYSTEM-DESIGN.md).

Everything here runs **without a GPU**. The box costs money the moment it boots,
so the rule for this directory is:

> No GPU session without a pre-written question, a pre-written pass/fail, and a
> rollback. "Boot it and see" is how GPU budgets disappear.

## Layout

```
capacity/     KV capacity model -- the arithmetic the plan rests on
  model.py        domain: Architecture, Parallelism, Precision, CapacityModel
  deployment.py   sourced constants for this deployment
  scenarios.py    optimisation scenarios; extend by appending to REGISTRY
  hypothesis.py   replication hypothesis test
  report.py       report assembly, no I/O
  renderers.py    TextRenderer / JsonRenderer behind a Renderer protocol
probe/        static engine capability probes
  base.py         EngineProbe template, ProbeResult
  engines.py      VllmProbe, SglangProbe; add an engine by subclassing
  renderers.py    TextRenderer / JsonRenderer
tests/        unittest suite over the model
```

## Run

```bash
python3 -m redesign.capacity                 # full report
python3 -m redesign.capacity --sweep         # concurrency vs prompt length
python3 -m redesign.capacity --verify        # hypothesis only; exit 0 = replicated
python3 -m redesign.capacity --json

python3 -m redesign.probe                    # both engines
python3 -m redesign.probe --engine sglang    # exit 0 = an A1 flag is present

python3 -m unittest discover -s redesign/tests -t .
```

No third-party dependencies. Python 3.10+.

## Phase Z — zero-GPU deliverables

| ID | Where | Status | Answers |
|---|---|---|---|
| **Z0** | `capacity/` | done | Is the KV pool 8× deflated by MLA-under-TP replication? |
| **Z2** | `probe/`, `Z2-FINDINGS.md` | done | Is de-duplication available for a hybrid KDA+MLA model on ROCm, and on which engine? |
| **Z1** | — | todo | Are clients sending per-key `cache_salt`, which drives the hit rate to 0% by design? |
| **Z3** | — | todo | Capture and replay the real production request distribution. |
| **Z4** | — | todo | Tenancy + backpressure layers, against a mock endpoint. |
| **Z5** | — | todo | Extend the correctness gate to catch quantized-KV silent garbage. |
| **Z6** | — | todo | Pre-registered GPU session definitions. |

## Results so far

### Z0 — the pool is replicated, confirmed offline

```
measured pool             2,295,266 tokens
predicted if REPLICATED   2,369,005   (error 3.2%)
predicted if DE-DUPED     18,952,040  (error 725.7%)
VERDICT                   REPLICATED
```

The model's sustainable concurrency at the observed mean prompt length is **75
sequences**; production steady state is 68–76 in system. Configured
`max-num-seqs` is **512**, a 6.8× overcommit — which is where the 4–9/min
preemptions come from.

| Scenario | Resident | vs today | Admission ceiling |
|---|---|---|---|
| Today (TP8 replicated, bf16) | 2.37M | 1.0× | 75 |
| A2 fp8 KV | 4.74M | 2.0× | 149 |
| A3 512 GB host tier | 2.37M (+2.31M tier) | 2.0× | 75 |
| **A1 KV de-duplication** | **18.95M** | **8.0×** | **570** |
| A1+A2 | 37.90M | 16.0× | 1074 |

This retired GPU session G0 before the box booted.

### Z2 — A1 is available on SGLang, not on vLLM

See [`Z2-FINDINGS.md`](Z2-FINDINGS.md). This inverted the original sequencing:
the engine decision is not a follow-on to the capacity work, it *is* the
capacity work.

## Caveat on the inputs

`OBSERVED_PROMPTS` in `capacity/deployment.py` comes from a **229-request
sample**. Every concurrency number derived from it is provisional until Z3
replaces it with a real week of traffic. The replication finding itself does not
depend on the distribution.
