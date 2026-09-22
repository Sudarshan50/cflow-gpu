# Throughput-first admission — live 22 September 2026

## Goal and deployment

Valid requests share available execution capacity and wait through temporary
bursts. Capacity errors are returned when bounded waiting cannot obtain safe
execution capacity, rather than because an otherwise idle traffic class reached
a fixed share or long-output allowance.

Final gateway activation: **2026-09-22 15:36:55 UTC**. Nginx uses the aligned
throughput edge policy. The engine and LiteLLM container IDs were preserved
through both gateway activations.

The diagnosis motivating this change is
[TRAFFIC-BURST-ANALYSIS-2026-09-22.md](TRAFFIC-BURST-ANALYSIS-2026-09-22.md).

## Effective policy

| Control | Value / behavior |
|---|---|
| Execution pool | Shared across local classes |
| Initial execution capacity | 32 |
| Maximum execution capacity | 64, within the engine's sequence ceiling |
| Healthy expansion | +4 slots at most every 2 seconds, when demand exists |
| Pressure confirmation | Three background observations at approximately 500ms intervals |
| Recovery confirmation | Four observations without overload; stronger green conditions permit expansion |
| Ordinary pressure response | Reduce new starts; preserve running requests and bounded waiting |
| Sustained engine queue | More than two engine waiters pauses additional starts |
| Decode pressure | Mean engine inter-token latency above 120ms |
| TTFT / prefill pressure | Mean engine TTFT above 5s / prefill above 10s, with supporting load evidence |
| Supporting latency-load evidence | Engine waiters, ITL above 80ms, or at least 32 observed active requests |
| KV pressure | Observed KV at least 88% |
| Immediate start pause | KV at least 97%, preemption, or stale/incomplete health information |
| Projected memory ceiling | 92% of measured engine KV-token capacity |
| Memory accounting | Full input plus granted output; conservative-format reservations retained |
| Admission queue | 128 requests, 64M shape tokens, 1GiB accounted request-body bytes |
| Admission wait | Up to 60s, including tokenizer-slot/inspection time after normalization |
| Queue aging | After 5s, the oldest valid waiter gets the next cross-lane opportunity |
| Prompt inspection | Four concurrent inspections; waits share the admission deadline |
| Edge outstanding requests | 256 per IP and authorization value |
| Edge fixed request-rate rejection | Removed from the throughput-mode inference location |

Long cold prompts can have substantial intrinsic TTFT. Without accompanying
queueing, decode interference, or a busy pool, they freeze expansion rather than
collapse the execution limit. When the engine is observed empty with healthy KV,
the qualified starting capacity is restored immediately. This avoids leaving an
idle engine at a reduced limit after a completed expensive request.

The configured 64 is an operating ceiling, not a promise that every workload
performs best at concurrency 64. The controller searches within that envelope
using current latency and memory pressure.

## Removal of the confirmed artificial gates

In throughput mode:

- `ClassBudget` enforces one adaptive shared execution limit. Fixed class shares
  no longer reject a class while other capacity is idle.
- The 12,288/49,152 predicted-output budgets and their prompt-length/running-count
  eligibility cliffs do not gate execution.
- The fixed large-context request-count limit does not gate execution. Physical
  KV commitments determine whether those requests fit.
- Memory capacity comes from the engine's measured KV-token capacity rather than
  the old fixed one-million-token budget. Full output grants are reserved, even
  when recent observed outputs were short.
- A warm/pressured controller does not change 60s waiting to 1s/0s or shrink the
  waiting queue to eight/four positions.
- Nginx forwards valid pooled bursts to authenticated quota enforcement and the
  gateway's bounded admission system. Its previous 20r/s shared-IP bucket no
  longer rejects those requests at the edge.

The shared pool retains idempotent release, cancellation cleanup, context
validation, full-grant memory commitments, and reconciliation of engine work
outside this gateway. Independently running gateways must not be treated as
independent owners of the same engine capacity.

## Error semantics and limits of the guarantee

- **503:** temporary serving capacity unavailable after waiting, queue/transport
  bounds exhausted, or an individually impossible execution reservation.
- **429:** actual authenticated tenant quota/rate limits remain the proxy's
  responsibility.
- **400 and other input/auth errors:** request validity and authentication retain
  their normal semantics.

Overload responses include their reason, admission outcome, wait duration, and
`Retry-After: 2`. They do not silently truncate outputs or replay inference.

Finite queues and client deadlines still matter when offered work exceeds
sustainable completion capacity. Low *current* KV alone cannot override full
memory already committed to accepted requests or a genuine decode bottleneck.
Waiting deliberately trades some burst latency for successful completion.

## Implementation

- `redesign/gateway/throughput.py`: pressure confirmation, shared capacity,
  bounded expansion, and idle recovery.
- `redesign/gateway/backpressure.py`: atomic shared execution pool.
- `redesign/gateway/workload.py`: measured-capacity, full-grant memory accounting
  and opt-in removal of legacy workload-count/output-budget gates.
- `redesign/gateway/admission.py`: stable deadlines, queued-demand feedback,
  cross-lane aging, cancellation, and impossible-reservation handling.
- `redesign/gateway/inspection.py`: bounded preparation concurrency and deadline
  sharing.
- `redesign/gateway/server.py`: runtime selection, capacity diagnostics, overload
  semantics, and a 256-connection TCP acceptance backlog.
- `redesign/gateway/capture.py`: durable policy reason, admission reason, and
  wait duration in the existing content-free request trace.
- `redesign/tests/test_throughput_admission.py`: pressure, memory conservation,
  fairness, cancellation, recovery, and HTTP behavior.

Runtime overrides:

- `/etc/systemd/system/k3-gateway.service.d/95-throughput-admission.conf`, from
  `redesign/deploy/profiles/throughput-admission.conf`.
- `/etc/nginx/conf.d/k3.conf` includes
  `/usr/local/lib/k3/redesign/deploy/profiles/throughput-edge.inc` inside `/v1/`.

The edge generator in `redesign/deploy/deploy.sh` supports
`K3_THROUGHPUT_FIRST=1`; use that setting when regenerating the throughput edge.
The base profile remains available for legacy-mode operation.

## Observability

Loopback-only `GET /diagnostics/capacity` exposes controller state, current and
target execution limits, pending demand, signal values, pressure reasons, and
measurement freshness. `/diagnostics/admission` retains the bounded outcome ring.

New gauges include:

- `k3_gateway_capacity_execution_limit`
- `k3_gateway_capacity_target_execution_limit`
- `k3_gateway_capacity_starts_paused`
- `k3_gateway_capacity_queued_demand`
- `k3_gateway_capacity_pressure_streak`
- `k3_gateway_capacity_healthy_streak`
- `k3_gateway_workload_throughput_first`
- `k3_gateway_workload_long_output_budget_enabled`
- `k3_gateway_workload_large_context_slot_limit_enabled`

Existing queue, reservation, rejection, cache, and latency telemetry remains
available. The control loop uses native engine timing. A nonstreaming gateway
first-body-byte measurement is whole-response latency, not generated-token TTFT.

## Verification

**Final regression: 389 tests passed.** This includes healthy burst sharing,
simultaneous reservation conservation, full-grant physical memory bounds,
actual overload, stale telemetry, cross-lane aging, deadline expiry,
cancellation, and the corrected idle/cold-prefill recovery behavior.

Final active-policy GPU validation started **15:37:39 UTC**:

| Check | Result | Observed duration |
|---|---:|---:|
| 64 simultaneous tiny requests | 64/64 correct | p95 1.83s |
| 32 simultaneous requests generating integers 1 through 64 | 32/32 correct | approximately 4.41s for the group |
| 12 long-output-allowance requests plus four cold long contexts | 16/16 correct | p95 11.73s |

All requested/granted allowances were preserved. Maximum observed running
requests was 36; the gateway queue reached 32 and drained. No preemptions,
tokenizer quarantine, or model restart occurred. No unrelated gateway arrivals
were observed during this final 112-request check.

Immediately afterward, a **64-request public TLS burst** exercised
`nginx → authenticated LiteLLM → gateway → engine`:

- **64/64 HTTP 200 and correct answers.**
- Median end-to-end latency **1.06s**; p95 **1.40s**; maximum **1.41s**.

An earlier public check exposed slow recovery after cold work (p95 8.49s), leading
to the final controller correction. A separate temporary-gateway run overlapped
with 17 customer attempts: its 96 tiny/decode requests succeeded, while its 16
heavy requests expired under cross-instance ownership/pressure checks. That run
is retained as an invalid isolated comparison, not a successful qualification.
The final correction was regression-tested against the GPU-qualified serving
base, then validated through the single active gateway as reported above.

The final 60-second passive observation was idle: no new customer requests,
queueing, preemptions, or generated tokens. It confirms clean released state,
not a production-wide error-rate improvement. The canaries qualify the exercised
burst/correctness behavior; sustained maximum throughput on the full customer
workload still requires ongoing measurement.

## Evidence and restoration

Artifacts under `/tmp/opencode/`:

- `k3-throughput-controller-qualified-20260922.json`
- `k3-throughput-controller-tests-20260922.log`
- `k3-throughput-active-final-20260922.json`
- `k3-throughput-public-final-20260922.json`
- `k3-throughput-final-observation-20260922.json`

Snapshots:

1. `k3-throughput-rollout-20260922T152442Z`: original policy → first throughput build.
2. `k3-throughput-rollout-20260922T153639Z`: first throughput build → final correction.

`k3_throughput_rollout.py rollback --deployment <snapshot>` checks file hashes and
waits for a natural drained window. To restore the original policy using these
snapshots, roll back snapshot 2, then snapshot 1. Snapshot 2 alone restores the
first throughput build. Keep the saved manifests with their corresponding files.
