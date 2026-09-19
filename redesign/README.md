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
traffic/      what the surviving edge logs say about production
  records.py      value types for a traffic window
  sources.py      TrafficSource protocol; ProdStatsSource
  analysis.py     Check registry; add a check by subclassing
gateway/      tenancy, backpressure and trace capture
  models.py       RequestEnvelope, Decision, EngineSnapshot, ClampResult
  classification.py  ordered ClassRule registry, first match wins
  clamping.py     TokenClamp -- register item D2
  backpressure.py CircuitBreaker, ClassBudget; fails open
  policy.py       composes the above into one Decision
  capture.py      TraceSink protocol; JsonlSink, MemorySink
tests/        unittest suite over all four
```

## Run

```bash
python3 -m redesign.capacity                 # full report
python3 -m redesign.capacity --sweep         # concurrency vs prompt length
python3 -m redesign.capacity --verify        # hypothesis only; exit 0 = replicated
python3 -m redesign.capacity --json

python3 -m redesign.probe                    # both engines
python3 -m redesign.probe --engine sglang    # exit 0 = an A1 flag is present

python3 -m redesign.traffic                  # exit 1 = a CRITICAL finding
python3 -m redesign.traffic --window 24h

python3 -m redesign.gateway                  # dry-run the policy, no engine
python3 -m redesign.gateway --ceiling 96

python3 -m unittest discover -s redesign/tests -t .
```

No third-party dependencies. Python 3.10+.

## Phase Z — zero-GPU deliverables

| ID | Where | Status | Answers |
|---|---|---|---|
| **Z0** | `capacity/` | done | Is the KV pool 8× deflated by MLA-under-TP replication? |
| **Z2** | `probe/`, `Z2-FINDINGS.md` | done | Is de-duplication available for a hybrid KDA+MLA model on ROCm, and on which engine? |
| **Z1/Z3** | `traffic/` | done | What do the surviving edge logs show? (No further log data exists — it died with the box.) |
| **Z4** | `gateway/` | done | Classification, `max_tokens` clamp, backpressure, trace capture. Carries Z3's capture too. |
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

### Z1/Z3 — the traffic evidence is weaker than the plan assumed

`python3 -m redesign.traffic` over the only production log data that survived
the teardown (58 minutes, 651 requests):

| Finding | Severity |
|---|---|
| **75 requests (11.5% of all traffic) rejected with 400** — 69% of all failures | CRITICAL |
| Edge duration p50 **58 s**, p95 373 s, p99 1,044 s | CRITICAL |
| **92% of billable requests from one key** (2 IPs) | WARN |
| 58 minutes of coverage against a 24-hour target | WARN |
| `/v1/models` 62% failure, `/v1/embeddings` 100% failure | WARN |

Two consequences for the design:

- **The eviction-spiral narrative is unproven.** A 6–13% prefix hit rate across
  one developer's ad-hoc prompts is the expected result, not a pathology. The
  85% shared-prefix premise cannot be evaluated from this window at all, so a
  third hypothesis now sits alongside eviction and `cache_salt`. The replication
  finding is unaffected — it is arithmetic over the engine's reported pool.
- **D2 is promoted to the top of the register.** An 11.5% rejection rate is
  larger, more certain and cheaper to fix than anything in Tier A, and needs no
  GPU: a per-class `max_tokens` clamp at the tenancy layer.

### Z4 — the clamp recovers ~7 of the 11.5 points, not all of them

`python3 -m redesign.gateway` dry-runs the policy against the observed prompt
distribution with the fixed `max_tokens` Foundry-style SDKs send:

```
engine rejects today, unguarded          70    7.0%
rescued by the clamp                     70    7.0%
observed production rejection rate            11.5%
NOT explained by this cause                    4.5%
```

The clamp recovers everything it can address. The residual ~4.5 points have a
cause `prod_stats.json` cannot identify — it records the status code, not the
engine's error body. The gateway's error taxonomy will name it on first boot.

## Planning changes after the teardown

The box was destroyed with **no usable snapshot**, and the logs went with it.

- **First boot is a full rebuild**, ~1 hour plus a 1.5 TB weight pull. Snapshot
  immediately after the first successful build, before experimenting.
- **Build SGLang directly.** With nothing to preserve, standing up vLLM only to
  tune it and migrate would spend a rebuild on a stack already decided against.
  **G1 folds into G2.**
- **Capture starts at first boot.** `gateway/capture.py` records request shapes
  — never content, customer labels hashed — so the distribution every admission
  number depends on gets rebuilt from day one.

## Caveat on the inputs

`OBSERVED_PROMPTS` in `capacity/deployment.py` comes from a **229-request
sample**. Every concurrency number derived from it is provisional until Z3
replaces it with a real week of traffic. The replication finding itself does not
depend on the distribution.
