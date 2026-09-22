# Adaptive Admission Control Design

The production throughput-first profile deployed on 2026-09-22 is documented in
[THROUGHPUT-ADMISSION-2026-09-22.md](THROUGHPUT-ADMISSION-2026-09-22.md). It replaces
the legacy class/output gates and pressure-shortened waiting described below
when `K3_THROUGHPUT_FIRST=1` is enabled.

The design should use **one adaptive admission controller that coordinates concurrency, queues, and customer fairness**. It should expand when spare capacity exists and protect latency when the system is genuinely saturated.

## 1. Separate hard boundaries from adaptive limits

| Keep as hard boundaries | Make adaptive |
|---|---|
| Model context limit | Number of requests admitted concurrently |
| Memory-safety limits | Capacity allocated to each request class |
| Configured customer quota ceilings | Customer share of available serving capacity |
| Maximum queue memory | Queue depth and waiting budget within request deadlines |

Load management should preserve the caller’s generation settings. It should not silently shorten responses or disable reasoning to make throughput appear better.

## 2. Correct the admission path first

Recommended flow:

```text
Authenticate and validate
          ↓
Identify authenticated customer/key
          ↓
Atomically check and reserve available capacity
          ├── Capacity available → execute
          ├── Temporarily busy → bounded fair queue
          └── Cannot meet deadline → clear overload response
```

Two prerequisites address the confirmed flaws:

- **A capacity check in progress must not automatically consume a waiting-queue position.** Multiple checks should safely use a shared snapshot and atomic reservations.
- **Customer identity must come from authentication.** Requests should not silently share an `anonymous` customer bucket.

For the reproduced case, **12 eligible requests must be admitted when 16 suitable slots and sufficient resource budgets are available**.

## 3. Adjust concurrency using actual pressure

A background control loop should update the admission budget periodically. Requests should use its latest snapshot rather than perform blocking health checks on the critical path.

### Signals to use

- Gateway queue age and depth.
- Engine running and waiting requests.
- Projected KV-memory demand.
- Time to first token and generation-token latency.
- Prefill/tokenization congestion.
- Preemptions, failures, and measurement freshness.

Include **reasoning tokens**, not just visible output, when estimating service cost.

### Controller behavior

| Situation | Response |
|---|---|
| Waiting demand, healthy latency, spare capacity | Gradually increase admitted concurrency |
| Other request classes are idle | Let the busy class borrow unused capacity |
| Queue delay or latency keeps rising | Reduce new admissions and wait for capacity |
| Critical memory pressure or unhealthy engine | Stop admitting additional risky work |
| Measurements disagree or are stale | Refresh/reconcile and use a conservative temporary budget |

Implemented initial production rules:

- Evaluate every **500 milliseconds** outside the request path.
- Increase borrowed capacity by **four slots** per healthy observation, up to 32.
- Remove borrowed capacity immediately on an engine queue, preemption, stale
  metrics, 70% KV usage, ITL above 120 ms, TTFT above five seconds, or mean
  prefill time above ten seconds.
- Use a 24-slot intermediate ceiling in the warning band.
- Use different recovery and congestion thresholds to prevent oscillation.
- Never exceed the existing **64-request hard ceiling**; the adaptive controller
  cannot weaken token reservations, context limits, or queue-memory bounds.

Reducing the budget would stop new starts until usage falls; it would not terminate healthy running requests.

## 4. Replace rigid class partitions with fair sharing

The current short-chat class is capped at 16 even when other classes are idle.

The proposed scheduler should provide:

- Minimum capacity shares for important classes.
- Borrowing of unused capacity within the global safe envelope.
- Per-customer fairness.
- Aging so large jobs cannot wait indefinitely.

The first production controller implements protected class minimums with
work-conserving P1/P2-agentic borrowing. Downstream-customer fairness cannot be
derived from the one shared NewAPI provider key: NewAPI must propagate a
trusted customer/session identity before that guarantee can be implemented.

Request cost should consider input size, likely output—including reasoning—and observed service times. A maximum output allowance alone is a poor workload classifier.

## 5. Make queueing deadline-aware

A fixed eight-request queue and universal ten-second wait do not fit every workload.

Queue decisions should ask:

> “When can this request probably start, and is there enough time left to complete it?”

For example, four large requests taking around **43 seconds** should not cause waiting callers to repeat three ten-second attempts before the first slot becomes available.

Queue bounds should cover:

- Request count.
- Retained bytes and token reservations.
- Per-customer share.
- Remaining deadline.

Small interactive requests and large-context jobs should have separate waiting policies.

## 6. Keep accounting and retries consistent

Maintain one request lifecycle:

```text
Checking → Queued → Admitted → Running → Completed/Failed/Cancelled
```

Track queued demand separately from execution reservations. Release or reconcile reservations exactly once, including cancellation and partial streams.

Use one retry owner, bounded retries, jitter, and meaningful `Retry-After` guidance. Preserve the distinction between:

- **429:** customer quota/rate restriction.
- **503:** temporary serving-capacity shortage.
- **400:** invalid input.

## 7. Validate before enabling automatic expansion

Rollout order:

1. Fix premature queue rejection and customer identity.
2. Run the controller in **shadow mode**—record decisions without enforcing them.
3. Enable small, bounded adjustments.
4. Test mixed traffic: tiny requests, reasoning-heavy requests, large contexts, cache hits/misses, and cancellation.
5. Expand the operating ceiling only after correctness and latency checks pass.

**Success means more requests complete within their deadlines—not merely higher TPM or fewer immediate errors.** This design should remove avoidable rejections and use spare capacity better, while retaining honest backpressure when demand exceeds sustainable capacity.

## 8. Initial live rollout

On 2026-09-22 the background controller was deployed by restarting only the
gateway while it had zero active and zero queued requests. The model and
LiteLLM were not restarted. A simultaneous 20-request P1 smoke test completed
20/20 with HTTP 200 (p50 0.68 s, p95 1.65 s, max 2.37 s), with no engine queue,
preemption, gateway queue timeout, or controller scrape error afterward.

The fixed long-output request-count limit was subsequently removed. Admission
is now bounded by predicted output tokens (12,288 base; 49,152 healthy burst),
projected KV usage, engine queue/preemptions, adaptive latency state, and the
64-request global ceiling. Stale, warm, or pressured controller state disables
the larger token budget. Legacy request-count settings are accepted for
configuration compatibility but are not enforced.
