# Status — `docs/SYSTEM-DESIGN.md` vs what exists

Audited 2026-09-20 against the branch, not from memory. Re-audit before booking
any GPU session.

**Nothing in Phase G has run. No GPU has been touched.**

---

## Blocking defects — all three closed

| # | Defect | Closed by |
|---|---|---|
| 1 | `gateway/server.py` missing; `deploy.sh` could not finish | `gateway/server.py`, `engine.py`, `tokens.py`, `metrics.py` |
| 2 | Engine never told to honour priority; class design inert | `ENABLE_PRIORITY_SCHEDULING` in every profile, validated against the image, and the gateway stamps a priority only when the engine can honour one |
| 3 | `edge` stage a stub | nginx server block, per-customer key map, `keys` stage, certbot behind a DNS guard |

**The stack is now deployable end to end.** What remains is unvalidated rather
than unbuilt.

---

## Optimisation register (§6)

| # | Item | State | Note |
|---|---|---|---|
| **A1** | MLA KV de-duplication | config done, **unvalidated** | `dedup` profile; G2 is the test |
| **A2** | fp8 KV | config done, **unvalidated** | Z5 now exists and gates it; G3 is the test |
| **A3** | CPU DRAM L2 tier | config done, **unvalidated** | `dedup-hicache` profile; G4 |
| **B0** | `cache_salt` audit | **deferred** | logs died with the box; no first-boot check written |
| **B1** | Right-size admission | **done** | derived from `redesign.capacity`, not hardcoded |
| **B2** | Priority scheduling + classes | **done** | gateway stamps, engine honours, flag validated pre-launch |
| **B3** | `long-prefill-token-threshold` | **done** | in all three profiles |
| **C1** | AITER flag contract | **done** | register row still names the *vLLM* variable; stale since the SGLang decision |
| **C2** | `max-num-batched-tokens` | **set, untuned** | 16384 in every profile; a starting point to sweep, not a tuned constant |
| **C3** | A8W4 vs A4W4, KDA fused decode | **partial** | A8W4 set; no A/B configured. Sub-1% here anyway |
| **C4** | DSpark | declared off, unused | deliberate; `ENABLE_DSPARK` is documentation-only |
| **D2** | `max_tokens` clamp | **done** | `gateway/clamping.py`, recovers ~7 of the 11.5 points |
| **D1** | Route P1 off-box | **half** | classifier marks `served_off_box`, but with no target configured P1 still takes a bounded local slot rather than silently skipping the budget |
| **D3** | Distillation lane | **not started** | |
| **E1** | Off-box fallback | **not started** | the availability mitigation §10 calls highest-value |
| **E2** | Per-class SLO instrumentation | **done** | `gateway/metrics.py`, per-class TTFT and totals on `/metrics`. No alerting rules yet |

---

## Architecture layers (§4)

| Layer | State |
|---|---|
| nginx edge | **done** — TLS, default-deny 444, per-key map, limits, loopback-only scrapes |
| LiteLLM tenancy | **not started** — D1 and E1 both depend on it |
| Backpressure | **done** — policy core plus HTTP shell, 21 end-to-end tests |
| Engine | **done** — `deploy.sh engine`, three profiles |

---

## Phase Z (§11)

| ID | State |
|---|---|
| Z0 capacity model | **done** — retired session G0 |
| Z1 `cache_salt` audit | **deferred** — no logs survive |
| Z2 engine capability | **done** — A1 is SGLang-only; inverted the sequencing |
| Z3 trace capture | **done** — folded into Z4, `gateway/capture.py` |
| Z4 tenancy + backpressure | **done** |
| Z5 quantised-KV gate | **done** — three tiers, wired into `deploy.sh verify` |
| Z6 pre-registered sessions | **partial** — profiles map to sessions; no written pass/fail per session |

---

## What is left

| Item | State |
|---|---|
| **A1 / A2 / A3** | Config complete, **unvalidated on hardware**. This is what G2, G3 and G4 are for. |
| **SGLang flag names** | Taken from upstream docs, not from a running build. `deploy.sh engine` probes the image and refuses to launch on an unknown flag, so this fails cheaply rather than after a weight load. |
| **D1 off-box routing / E1 fallback** | Classifier marks P1 `served_off_box`; no external endpoint configured, so P1 currently serves locally. |
| **D3 distillation lane** | Not started. The `x-k3-batch` header and priority 3 exist, so the mechanism is there; the runner is not. |
| **B0 `cache_salt` audit** | Deferred — needs first-boot traffic. |
| **Alerting rules** | Metrics are exposed; nothing consumes them. |
| **Z6** | Profiles map to sessions; per-session written pass/fail still informal. |

## What is genuinely finished

The capacity argument, and the machinery that keeps it honest: the cost model
with 27 unit tests, the hypothesis test that retired a GPU session offline, the
engine capability probe, the traffic analysis that found the 400s, the gateway
policy core with 33 tests, and a deployer whose admission ceiling is derived
from the model rather than written down — so config and model cannot drift.
