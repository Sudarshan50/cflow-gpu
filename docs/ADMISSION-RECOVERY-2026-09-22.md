# Bounded admission recovery

## Question and initial baseline

Can short bursts be admitted with substantially fewer overload rejections while
preserving the existing model, generation, cache, media and tenant policies?

The 05:46:53–06:16:53 UTC baseline on September 22 contained 263 successful API
requests, 26 admission 429s, three context-window 400s and seven response-ID
ownership 403s. The admission failures were 18 `long_output_slots`, seven
`engine_queue` and one `untracked_engine_work`. Successful end-to-end latency
was 7.356 s median and 38.434 s p95, across a mixed workload. The model engine
was using its existing 262144-token context, 4096-token batch, 64 sequences,
priority scheduling and prefix cache. Video support was already enabled.

## Deployed policy

`redesign/deploy/profiles/admission-queue.conf` enables bounded waiting on the
gateway. The final deployed profile sets `K3_LONG_OUTPUT_BURST_MAX=2`, retaining
the original two-request long-output limit. Extra burst slots are implemented
but disabled after the initial live correctness check described below. Without
the profile, waiting and burst expansion default to disabled.

- Bounded admission wait: 3 seconds; at most 16 waiting requests. The NewAPI
  provider identity is a multi-customer pool, so it may use all 16 waiter
  positions rather than being treated as one end customer.
- Waiting capacity: at most 2097152 reserved-shape tokens and 64 MiB of request
  body sizes. These are queue accounting, separate from execution reservations.
- FIFO within traffic-class/resource lanes. An independently admissible short
  request can progress while a heavy lane waits. New arrivals cannot overtake
  an already waiting request in the same lane.
- Base long-output allowance remains two. The optional, currently disabled
  expansion requires ordinary text prompts up to 32768 tokens, no engine queue
  or preemptions, fewer than eight observed/owned running requests, known KV
  capacity, and projected KV demand at most 55%.
- Large-context, multimodal and conservative full-context reservations do not
  borrow burst slots. All admitted requests retain full input plus granted
  output reservations, with the existing 1048576-token execution budget and
  85% heavy-work projected-KV limit.
- Up to two owned engine waiters are tolerated only below 55% projected KV,
  below eight observed/owned requests, and without preemptions. Under stronger
  pressure the original empty-engine-queue rule applies. Further contention
  waits in the bounded gateway queue instead of immediately failing.
- Cached engine/request-count disagreements trigger a fresh, single-flight
  scrape, rate-bounded to 250 ms. A persistent mismatch still blocks heavy
  admission; a temporary mismatch can resolve during the bounded wait.
- Expired and disconnected waiters never reach inference. Acquired leases are
  returned if cancellation/deadline is detected immediately after admission.
  Existing generation and class-output limits are used without modification.
- P1 short-chat and P2 agentic traffic may borrow otherwise idle class capacity
  up to 32 concurrent requests. Borrowing requires projected KV at or below
  70%, no engine queue or preemptions, and recent mean inter-token latency at
  or below 120 ms. The original class shares return immediately when any of
  those signals shows pressure. The global 64-request ceiling and all full
  input-plus-output reservations remain hard bounds.
- Overload responses use HTTP 503 and `service_unavailable`, preserving their
  original policy reason plus an admission outcome code and wait duration.
  Genuine context errors remain 400; class-limit 429 semantics are retained.

No inference is replayed by this mechanism. Retries are admission checks only.
LiteLLM retains ownership of its existing tenant quota/reservation accounting.
Its API retry settings are not part of this change.

## Instrumentation

- `k3_gateway_admission_*` gauges/counters: queue size, queued token/body budget,
  enqueued/admitted counts, expiry, queue-full and cancellation counts.
- Per-class `k3_gateway_admission_wait_seconds` quantiles for queued requests.
- `k3_gateway_workload_burst_admissions_total` and the last long-output limit.
- Loopback-only `/diagnostics/admission`: bounded content-free outcomes.
- Response header `x-k3-admission-wait-ms`.

Existing client-visible TTFT and end-to-end measurements include the wait;
moving failures into another status code is not considered an improvement.

## Verification and rollout criteria

1. Baseline and candidate regression suites must pass, including conservation
   of execution reservations, FIFO lanes, full queues, cancellation, deadline
   expiry, original payload/output preservation, cache, media and protocol tests.
2. The deployed model/engine configuration, media normalization, prompt
   inspection, classification, clamping, tenancy policies and nginx policy are
   fingerprinted. Their hashes and the model engine's startup epoch must agree
   before and after rollout.
3. The initial four-slot trial was withdrawn. The final rollout uses the
   original two long-output slots and 10-second waiting. Deploy the gateway
   after its work drains; the engine and LiteLLM processes remain running so
   model/KV and tenant state persist.
4. Observe natural traffic and compare admission rejection rate, all 4xx/5xx,
   throughput, cache reuse, preemptions and latency by comparable prompt/output
   buckets. New successful requests that were previously rejected are reported
   as an additional workload rather than silently removed from latency totals.
5. Target at least a 50% reduction in admission rejection rate with a meaningful
   live sample (preferably 100+ attempts). No new engine failures, token-count
   quarantine, lost leases or preemption increase is acceptable. Investigate
   more than 25% sustained latency growth in comparable, sufficiently populated
   buckets; do not infer regressions from an isolated changed traffic mix.

If serving pressure or comparable-workload latency materially regresses,
disable the feature with `admission-queue-disabled.conf`, drain, and reload the
gateway. Deployment manifests record the immediately preceding module/profile
versions; they are not necessarily the original pre-feature deployment.

## Results

Initial host regression baseline: 308 tests passed, 16 skipped. Final regression
result using the installed LiteLLM environment: **339 tests passed, no skips**.

### Concurrent rollout and first live correctness check

Another operator deployed the candidate during verification and restarted the
model and proxy while completing separate video/reasoning/Responses work. The
user confirmed that work and subsequently confirmed a stable validation window.
The earlier passive observation is not used as a matched before/after result.

The intermediate full installed-dependency regression suite passed 337 tests.
The first candidate-only live burst completed nine of nine requests correctly.
In the subsequent matched test at 07:03 UTC, strict admission served five of
nine attempts (four overload rejections); candidate admission served all nine,
but one generated sequence stopped at 32 although 64 was requested. Its
4096-token grant was intact and only 71 tokens were generated. That candidate
run is not accepted as a correctness-qualified performance result.

Burst expansion is therefore reduced to the existing two-request base during
further validation; bounded waiting and freshness reconciliation remain enabled.
The pressure-gated expansion code remains available but is not qualified for
production expansion by that failed test.

For feature rollback, apply `admission-queue-disabled.conf` as the gateway's
90-admission override and reload after draining. This disables waiting and
expansion without reverting the other operator's serving changes. The snapshot
under `k3-admission-rollout-20260922T064922Z` was taken after that operator's
deployment, so it must not be mistaken for the original pre-feature build.

### Qualified conservative burst comparison

Two matched reference-policy comparisons were run on the same warmed model
engine, with runtime fingerprints and startup IDs checked before/after. The
temporary reference listener used the current code with the original strict
admission controls (no waiting, two long-output slots, no queue tolerance),
then was removed. The customer listener was not redirected to it.

The final comparison at **07:40:46 UTC** used six simultaneous long-allowance
requests plus three small tool-capable canaries:

| Result | Strict reference | Final bounded queue |
|---|---:|---:|
| Successful requests | 5/9 | 9/9 |
| Overload rejections | 4 | 0 |
| Correct successful answers | 5/5 | 9/9 |
| Immediately admitted burst mean | 4.056 s | 4.453 s |
| Small-canary mean | 0.646 s | 1.161 s |
| Small-canary maximum | 0.797 s | 1.312 s |
| Maximum admission wait | 0 | 7.986 s |
| Cached prompt tokens per burst request | 1536 | 1536 |
| Granted output allowance | 4096 | 4096 |
| Preemptions | 0 | 0 |

Both runs had stable runtime fingerprints and no unrelated customer arrivals
during the matched phases. The extra four requests succeed by waiting, so
their total latency is higher than that of the immediately admitted requests.
The small-canary sample is only three requests; the approximately 0.5-second
change must not be represented as proof of no latency impact. An earlier
qualified two-slot comparison at 07:07 UTC also passed 9/9 and measured a
small-canary mean of 0.826 s versus 0.994 s for the reference. These small samples
support bounded burst handling, not a broad statistically matched SLO claim.

### Correctness and compatibility

- Live corruption/determinism and reasoning gate: **6/6 passed**.
- Authenticated proxy checks: **6/6 passed** (model aliases, streaming terminal
  markers and usage, function calling, image input, video input, Responses API).
- Model context, batch/sequence settings, output clamping, media, prompt
  inspection, tenancy/price configuration and nginx policy were fingerprinted.
  They did not change during the agent's final activation or validation windows.
- The final pressure-aware workload update became ready at **07:39:47 UTC**.
  The engine and LiteLLM container IDs were preserved across this activation.

### Ordinary traffic observations and remaining limits

The **07:15:23–07:25:23 UTC** window (before the final pressure refinement)
contained 60 proxy requests: 47 successes and 13 overload timeouts. Four queued
requests completed during the window. The engine reached 16 running requests,
six waiting requests and 72.3% active KV usage, with substantially less prefix
reuse than the early baseline. No preemptions or engine restarts occurred.
This window did **not** meet the broad error-reduction target. In response, the
final update restricts queue tolerance to genuinely low-pressure states and
includes both running and waiting engine requests in ownership reconciliation.

The final **07:41:11–07:46:11 UTC** window contained 503 proxy requests:

- 482 HTTP 200 responses.
- Seven admission timeouts (1.39% of attempts), all `long_output_slots`.
- Fourteen HTTP 400 input/tokenizer/structured-output validation failures.
- No preemptions, model/proxy restarts or protected runtime changes.
- Peak sampled engine work: three running, zero waiting, 0.64% active KV usage.

That last window was cache-heavy (432 response-cache hits recorded in the
database). Its traffic mix differs materially from the cold-load window, so
the change in rejection percentage is **not** an isolated measurement of the
code's effect. Fixed long-output capacity can still cause a valid request to
outwait the admission budget. Remaining work is to qualify
cost-/prefill-aware scheduling or additional capacity against representative
cold traffic; raising concurrency is not justified by this run alone. The
matched burst-handling goal is met, but a production-wide cold-load latency and
error-rate improvement has not been established by these differing workloads.

### Pooled NewAPI adaptive update

At 09:17 UTC the gateway was updated without restarting the model. Static class
shares now protect baseline capacity, while P1 and P2-agentic requests can
borrow up to 32 slots under the pressure conditions above. A 20-request
simultaneous P1 smoke test returned 20/20 HTTP 200; four requests used borrowed
capacity, with zero queueing, preemptions, or engine errors. This verifies the
borrowing path and reservation conservation, not sustained production
throughput.

### Evidence

Artifacts are in `/tmp/opencode/`:

- `k3-admission-comparison-20260922T074046Z.json` (final matched comparison).
- `k3-admission-comparison-20260922T070739Z.json` (earlier two-slot comparison).
- `k3-admission-candidate-20260922T070329Z.json` (unqualified four-slot check).
- `k3-admission-correctness-20260922.json`.
- `k3-admission-compatibility-20260922.json`.
- `k3-admission-live-observation-20260922.json` (cold-load window).
- `k3-admission-final-observation-20260922.json` (final cache-heavy window).
- `k3-admission-rollout-20260922T073815Z/` (final activation and immediate rollback snapshot).
