# Status — `docs/SYSTEM-DESIGN.md` vs what exists

Audited 2026-09-20 against the branch, not from memory. Re-audit before booking
any GPU session.

**Nothing in Phase G has run. No GPU has been touched.**

---

## Blocking defects

| # | Defect | Effect |
|---|---|---|
| 1 | `redesign/gateway/server.py` does not exist, but `deploy.sh` renders a unit with `ExecStart=python3 -m redesign.gateway.server` | **`./deploy.sh` cannot complete a full run.** It fails at the `gateway` stage. The policy core is complete and tested; the HTTP shell is not written. |
| 2 | Engine is never told to honour request priority | **The entire class design is inert.** The gateway classifies, stamps a priority and budgets per class — and the engine schedules FCFS, so P3 distillation competes evenly with P0 keystrokes. B2 is half-built. |
| 3 | `edge` stage is a stub | No TLS, no per-customer keys, no default-deny. The gateway must stay on loopback; the stack is **not exposable**. |

Defect 2 is the one worth pausing on: every piece of classification machinery
built in Z4 buys nothing until the engine's scheduler is told to respect it.

---

## Optimisation register (§6)

| # | Item | State | Note |
|---|---|---|---|
| **A1** | MLA KV de-duplication | config done, **unvalidated** | `dedup` profile; G2 is the test |
| **A2** | fp8 KV | config done, **blocked** | needs Z5 first — silent-garbage class |
| **A3** | CPU DRAM L2 tier | **partial** | `deploy.sh` accepts the flags, but no profile enables it, so there is no session to run |
| **B0** | `cache_salt` audit | **deferred** | logs died with the box; no first-boot check written |
| **B1** | Right-size admission | **done** | derived from `redesign.capacity`, not hardcoded |
| **B2** | Priority scheduling + classes | **half** | gateway side done; engine side not wired — see defect 2 |
| **B3** | `long-prefill-token-threshold` | **done** | in all three profiles |
| **C1** | AITER flag contract | **done** | register row still names the *vLLM* variable; stale since the SGLang decision |
| **C2** | `max-num-batched-tokens` re-tune | **absent** | not in any profile or in `deploy.sh` |
| **C3** | A8W4 vs A4W4, KDA fused decode | **partial** | A8W4 set; no A/B configured. Sub-1% here anyway |
| **C4** | DSpark | declared off, unused | deliberate; `ENABLE_DSPARK` is documentation-only |
| **D2** | `max_tokens` clamp | **done** | `gateway/clamping.py`, recovers ~7 of the 11.5 points |
| **D1** | Route P1 off-box | **half** | classifier marks `served_off_box`; nothing to route *to* |
| **D3** | Distillation lane | **not started** | |
| **E1** | Off-box fallback | **not started** | the availability mitigation §10 calls highest-value |
| **E2** | Per-class SLO instrumentation | **not started** | capture records shapes; no metrics, no alerting |

---

## Architecture layers (§4)

| Layer | State |
|---|---|
| nginx edge | **stub** — defect 3 |
| LiteLLM tenancy | **not started** — D1 and E1 both depend on it |
| Backpressure | core **done** and tested; no HTTP server — defect 1 |
| Engine | **done** — `deploy.sh engine`, three profiles |

---

## Phase Z (§11)

| ID | State |
|---|---|
| Z0 capacity model | **done** — retired session G0 |
| Z1 `cache_salt` audit | **deferred** — no logs survive |
| Z2 engine capability | **done** — A1 is SGLang-only; inverted the sequencing |
| Z3 trace capture | **done** — folded into Z4, `gateway/capture.py` |
| Z4 tenancy + backpressure | **core done**, server missing |
| Z5 quantised-KV gate | **not started** — blocks G3 *and* all customer traffic |
| Z6 pre-registered sessions | **partial** — profiles map to sessions; no written pass/fail per session |

---

## Critical path to a first boot

1. `gateway/server.py` — without it `deploy.sh` cannot finish.
2. Engine-side priority scheduling — without it the class design does nothing.
3. Z5 correctness gate — without it: no customer traffic, no fp8 KV.
4. `edge` stage — without it the stack cannot leave loopback.

1 and 2 are small and zero-GPU. 3 is the largest remaining Phase Z item. 4 is
the only one that is mostly transcription rather than design.

## What is genuinely finished

The capacity argument, and the machinery that keeps it honest: the cost model
with 27 unit tests, the hypothesis test that retired a GPU session offline, the
engine capability probe, the traffic analysis that found the 400s, the gateway
policy core with 33 tests, and a deployer whose admission ceiling is derived
from the model rather than written down — so config and model cannot drift.
