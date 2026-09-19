# redesign/deploy — the production deployer

One script, `deploy.sh`, stands up the whole stack from a bare host. It depends
on nothing from the pre-existing vLLM deployment; this branch is self-contained.

```bash
./deploy.sh --dry-run                       # print every mutation, perform none
ENGINE_IMAGE=repo/sglang@sha256:... ./deploy.sh
PROFILE=dedup ./deploy.sh engine verify     # one profile, named stages
```

`ENGINE_IMAGE` is required and must be a digest. A floating tag changes the
AITER contract between deploys, which is the documented route to the
silent-garbage failure.

## Stages

| Stage | Does | Idempotent because |
|---|---|---|
| `preflight` | Refuses to proceed unless the host can serve this model at all | read-only |
| `host` | Directories, TCP tuning for the weight pull | checks before mutating |
| `weights` | ~1.5 TB fetch, resumable | completion marker |
| `engine` | Validates the profile against the image, then launches | container recreated |
| `gateway` | Tenancy, backpressure, trace capture | unit rewritten, reloaded |
| `edge` | nginx, TLS, default-deny | **stub — not yet implemented** |
| `verify` | Liveness, KV pool check against the model | read-only |

## Profiles map to GPU sessions

| Profile | Register | Expected pool | Admission | Session |
|---|---|---|---|---|
| `baseline` | — | ~2.3M | 75 | G-build |
| `dedup` | A1 | ~18.9M | 570 | G2 |
| `dedup-fp8` | A1+A2 | ~37.9M | 1074 | G3 |

**Build `baseline` first and snapshot it before tuning anything.** It is the
configuration that must serve correctly and record a gate baseline; every later
profile is measured against its numbers.

## Two guards that exist to protect GPU spend

**Flag validation.** Before launching, `engine` probes the image's own CLI
surface (`redesign/probe`) and refuses to start if the profile names a flag the
build does not accept. Without it an unknown flag is discovered *after* the
weights load — several minutes of paid GPU per mistake. If
`enable_dp_attention` comes back absent, register item A1 is unavailable on that
image and session G2 must not be booked at all.

**Floor-division guard.** SGLang's `--max-running-requests` is server-wide and
floor-divided across attention-DP ranks. Set it too low and the per-rank share
drops below the hybrid state pool's minimum, and the server refuses every
request while looking like a migration failure rather than a config error. The
deployer computes the per-rank value and refuses implausible ones.

```
FAIL  max-running-requests 8 over dp_size 8 leaves 1
      per rank. SGLang floor-divides this ceiling, and too small a share means
      'Hybrid state cache is too small to serve any requests' -- a server that
      refuses everything while looking like a migration failure.
```

## The admission ceiling is derived, not written down

`deploy.sh` asks `redesign.capacity` for the ceiling that matches the pool the
chosen profile actually produces, so the two cannot drift. `preflight` also runs
the model's own hypothesis test and refuses to deploy if it fails — if the model
no longer describes the box, every number it feeds the deployer is wrong.

Override for one run with `MAX_RUNNING_REQUESTS=...`; the deployer warns when
you do.

## Not yet implemented

| Gap | Consequence |
|---|---|
| `edge` stage is a stub | The gateway must stay on loopback. No TLS, no per-customer keys, no default-deny. **Not exposable.** |
| No correctness gate (Z5) | Do not route customer traffic, and do not enable `fp8_e4m3` — quantised KV fails as silent garbage, which a throughput benchmark reports as success. |
| `redesign.gateway.server` does not exist | The gateway unit references an HTTP entry point that has not been written. The policy core is complete and tested; the ASGI shell is not. |
| No LiteLLM config | D1 off-box routing and E1 outage fallback are designed but unconfigured. |

`deploy.sh` warns about the first two at the point they matter rather than
failing silently.
