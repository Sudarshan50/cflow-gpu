# redesign/deploy — SGLang deployment tooling

`deploy.sh` provides the SGLang profile-driven deployment path. The running
optimized vLLM service uses the separate [AMD profiles and service overrides](../../experiments/2026-09-20-amd-optimized/RESULTS.md).

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
| `engine` | Validates the profile against the image, then launches | fingerprint of image+args |
| `gateway` | Backpressure, trace capture, off-box | unit rewritten, reloaded |
| `ops` | First-boot B0/Z5, D3 distill, E2 scrape, Z3 timer | units rewritten, reloaded |
| `edge` | nginx, TLS, default-deny | HTTP bootstrap, then cert, then TLS |
| `keys` | Write the pass-through Bearer map (no TSV, no swap) | nginx reload |
| `verify` | Liveness, cache_salt, correctness gate, aggregate KV | read-only |

## Profiles map to GPU sessions

The aggregate-capacity figures below are historical planning estimates, not
validated limits for the current deployment.

| Profile | Register | Expected aggregate | Admission | Session |
|---|---|---|---|---|
| `baseline` | — | ~2.3M unique | max(floor, 427) | G-build |
| `dedup` | A1 | ~18.9M unique | ~570 | G2 |
| `dedup-fp8` | A1+A2 | ~37.9M unique | ~1074 | G3 |
| `dedup-hicache` | A1+A3 | ~18.9M + L2 | ~570 | G4 |

`python3 -m redesign.sessions` prints the written pass/fail for each.

**Build `baseline` first and snapshot it before tuning anything.**

## Admission is not the zero-sharing floor

`deploy.sh` derives admission from `redesign.capacity` unless explicitly
overridden. Qualify that configuration against the chosen engine and workload.

## Two guards that exist to protect GPU spend

**Flag validation.** Before launching, `engine` probes the image and refuses to
start if the profile names a flag the build does not accept.

**Floor-division guard.** SGLang's `--max-running-requests` is server-wide and
floor-divided across attention-DP ranks. Too small a per-rank share produces a
server that refuses everything.

## Off-box (D1 / E1)

Set `K3_OFFBOX_URL` (and optionally `K3_OFFBOX_API_KEY`, `K3_OFFBOX_MODEL`)
before `gateway`. P1 routes there; P0/P1 fail over to it when K3 is down.
LiteLLM is the §4 tenancy hop (`k3-litellm.service`, `127.0.0.1:4000`).
nginx sends public requests through LiteLLM; there is no unauthenticated gateway fallback.
Install the proxy into `/usr/local/lib/k3/venv` (`pip install 'litellm[proxy]'`).
Off-box models are not added until `K3_OFFBOX_URL` exists.

## Distillation (D3)

```bash
python3 -m redesign.distill --prompts samples.jsonl --output teacher.jsonl
systemctl start k3-distill.service
```

Loopback to the engine, priority 3, checkpointed, paused when the breaker
sheds batch. Never through nginx. Seed prompts live at
`redesign/deploy/distill/prompts.jsonl`.

## First boot (B0 / Z5)

`first-boot.sh` waits for `/health`, runs `redesign.probe.cache_salt`, then
the correctness gate. `k3-first-boot.service` is the oneshot.

## Alerts (E2)

`k3-alerts.service` scrapes the gateway `/metrics` every 30s against
`deploy/alerts/k3-gateway.rules.yml` and writes
`/scratch/deploy-state/alerts/last.json`. This is the scrape the rules file
was waiting for.
