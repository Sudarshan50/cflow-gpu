# cflox.store — Kimi-K3 on 8× MI355X

This repository is the complete, rebuildable definition of a production
OpenAI-compatible inference endpoint: `moonshotai/Kimi-K3` (2.8 T total / 104 B
active, MXFP4 weights) served by vLLM on 8× AMD Instinct MI355X, published at
`https://cflox.store/v1` behind an nginx edge that terminates TLS, enforces
per-customer API keys with per-key rate and concurrency caps, and closes the
unauthenticated routes vLLM's own `--api-key` leaves open
(`docs/K3-DEPLOYMENT.md` §11 "Why `--api-key` alone is NOT sufficient").
The box runs on a DigitalOcean MI355X spot droplet whose `/scratch` disk is
**ephemeral and destroyed on reclaim**, so everything needed to stand the
service back up from nothing lives here.

---

## Quick start

Four steps. The full version, with ordering constraints and failure modes, is
[`docs/DEPLOY.md`](docs/DEPLOY.md).

1. **Create the droplet by hand in the DigitalOcean Control Panel.** This step
   is not automatable — see [What is deliberately not automated](#what-is-deliberately-not-automated).
2. **Clone this repo to `/scratch/deploy`.** `deploy.sh` finds its own
   location, but `gen-keys.sh:19` and `k3.service:27` hardcode
   `/scratch/deploy`, so use that path. The repo is private, so the clone needs
   the PAT — a fresh box has no credential helper and git otherwise fails with
   `could not read Username for 'https://github.com'`:

   ```bash
   mkdir -p /scratch
   git clone https://Sudarshan50:<PAT>@github.com/Sudarshan50/cflow-gpu.git /scratch/deploy
   ```

3. **Restore the secrets bundle** so existing customer keys keep working.
   Skipping this mints fresh credentials and every live integration starts
   returning 401 — a generated key is not the one your customer holds:

   ```bash
   cd /scratch/deploy
   ./secrets-backup.sh verify k3-secrets.enc     # confirm the passphrase first
   ```

4. **Run the deployer.** Its final `verify` stage runs the security probes and
   the correctness gate for you:

   ```bash
   sudo ./deploy.sh              # all stages; ./deploy.sh --help for the list
   ```

   The `secrets` stage picks up `k3-secrets.enc` from the repo automatically
   and prompts for the passphrase. Point it elsewhere with
   `SECRETS_BUNDLE=<file> sudo -E ./deploy.sh` — the `-E` is required there,
   because plain `sudo` drops the variable and the stage would mint fresh keys
   instead of restoring.

   It must end in `verification passed`, which includes `GATE PASS`.

Weights are ~1.5 TB and download at roughly 2.23 GB/s (~700 s)
(`docs/K3-DEPLOYMENT.md` §3 "Model"); the engine then takes about 6 minutes to
load (`docs/K3-DEPLOYMENT.md` §4.3).

`deploy.sh` is idempotent and staged (`preflight host weights secrets install
image keys services tls verify`, plus `all`); re-running it costs nothing when
the work is already done. `--dry-run` prints every mutation without performing
any. If a stage fails, fix the cause and re-run that stage alone. **Path C** in
[`docs/DEPLOY.md`](docs/DEPLOY.md) does the same work by hand.

---

## Repo layout

```
.
├── Makefile                     operator shortcuts (gate, health, cache, logs, secret-scan, deploy, plan)
├── README.md                    this file
├── cloud-init.yaml              first-boot automation for a fresh droplet (scratch, weights, unit)
├── config.yaml                  vLLM serve args, with the reasoning for every value
├── deploy.sh                    staged single-script deployment (preflight … verify)
├── gen-keys.sh                  mint/revoke customer keys; regenerate the nginx auth+limit layer
├── issue-cert.sh                Let's Encrypt issuance and the TLS server block
├── k3.service                   systemd unit for the vLLM container
├── launch.sh                    image digest verification + fallback foreground launch
├── secret-scan.sh               pre-push gate: no credential tracked, staged, or in history
├── verify-auth.sh               security assertion suite for the public edge
├── vllm-k3.env.example          template for the container env (AITER flags + VLLM_API_KEY)
├── customers.tsv.example        template for the per-customer key table
├── dashboard/
│   ├── index.html               dashboard UI
│   └── server.py                loopback monitoring server, deployed to /opt/k3dash
├── docs/
│   ├── ARCHITECTURE.md          detailed component and request-path reference
│   ├── DEPLOY.md                full deployment guide (paths A/B/C, verification, rollback)
│   ├── K3-DEPLOYMENT.md         56 KB authoritative engineering reference: benchmarks,
│   │                            tuning rationale, failure catalog. Source of truth.
│   ├── RUNBOOK.md               day-2 operations and incident response
│   └── SECRETS.md               credential inventory, rotation, handling rules
├── nginx/
│   ├── 00-default-deny.conf     drop requests that match no server_name (444 / reject handshake)
│   ├── k3.conf                  upstream definitions: k3_backend :8001, k3_dash :8080
│   ├── k3-dash.inc              basic-auth-protected dashboard locations, included by the TLS block
│   └── k3-tls.conf.template     reviewable copy of the TLS edge policy; issue-cert.sh generates the live file
├── systemd/
│   └── k3dash.service           systemd unit for the dashboard
└── tests/
    ├── gate.sh                  correctness gate entry point (delegates to gate.py)
    ├── gate.py                  three-tier gate: corruption, reasoning, long context
    ├── gate-baseline.json       recorded healthy baseline the gate compares against
    ├── agentic-test.py          tool calling, structured output, vision, against :8001
    ├── ctx-test.py              long-context needle retrieval at controlled depths
    └── ide-ready.py             end-to-end IDE-agent behaviour through TLS with a customer key
```

Not in git, created on the box (see [Secrets](#secrets)): `api-key.txt`,
`customers.tsv`, `dash-password.txt`, `vllm-k3.env`.

> `docs/RUNBOOK.md` and `docs/SECRETS.md` are authored in the same change as
> this README and were not yet present when it was written.
> `TODO(operator):` confirm both landed before treating the links to them as
> live.

---

## Architecture

```
                        internet
                            │
                            ▼
              ┌─────────────────────────────┐
              │ nginx :80  → 301 https       │  00-default-deny.conf drops
              │ nginx :443 TLS (Let's Encr.) │  anything not cflox.store (444)
              └─────────────┬───────────────┘
                            │  k3-tls.conf, generated by issue-cert.sh
        ┌───────────────────┼────────────────────┬──────────────────────┐
        │                   │                    │                      │
   location /v1/       location = /         location /api/        everything else
   $k3_customer map    + /api/ (dashboard)                         → 403
   empty → 401         basic auth via
        │              k3-dash.htpasswd            │                     │
   per-customer               │                    │
   limit_conn 200             └─────────┬──────────┘
   limit_req 100r/s                     │
   Authorization rewritten to           ▼
   the single upstream key      upstream k3_dash
        │                       127.0.0.1:8080  (k3dash.service, no auth of
        ▼                                        its own — nginx is the only
   upstream k3_backend                           thing protecting it)
   127.0.0.1:8001                                    │
   k3.service → docker "k3"                          │ scrapes
   vLLM, VLLM_API_KEY enforced  ◄─────────────────────┘
        │
        ▼
   /scratch/hf/hub/models--moonshotai--Kimi-K3   (~1.5 TB, ephemeral disk)
```

Both upstreams are defined in `nginx/k3.conf`. `/metrics` and `/health` are
`allow 127.0.0.1; deny all` (`issue-cert.sh:72-73`) so the dashboard can scrape
them and nobody else can. Every request through `/v1/` is attributed in
`/var/log/k3/usage.log` using the `k3usage` log format defined at
`gen-keys.sh:91-93`.

Detailed version: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Common operations

Run from `/scratch/deploy` as root. `make` with no target prints the target list
(`Makefile:4-5`).

| Task | Command | Source |
|---|---|---|
| Re-run any part of the deployment | `./deploy.sh <stage>` (`--help` for the list) | `deploy.sh` |
| Add a customer (mints key, reloads nginx) | `./gen-keys.sh add <name>` | `gen-keys.sh:46-54` |
| Revoke a customer | `./gen-keys.sh revoke <name>` | `gen-keys.sh:55-60` |
| Rebuild the nginx auth/limit layer from `customers.tsv` | `./gen-keys.sh` | `gen-keys.sh:45` |
| Restart the engine (~6 min to serve again) | `systemctl restart k3.service` | `launch.sh:45-48` |
| Restart the dashboard | `systemctl restart k3dash.service` | `systemd/k3dash.service` |
| Endpoint + KV summary | `make health` | `Makefile:10-12` |
| Correctness gate, full (~21 s) | `make gate` | `Makefile:6-7` |
| Correctness gate, tier 1 only (~5 s) | `make quick` | `Makefile:8-9` |
| Live prefix-cache hit rate (expect ~81%) | `make cache` | `Makefile:13-15` |
| Follow engine logs | `make logs` | `Makefile:16-17` |
| Prove the edge is still locked down | `./deploy.sh verify` (see caveat below) | `deploy.sh`, `verify-auth.sh` |
| Follow per-customer usage | `tail -f /var/log/k3/usage.log` | `systemd/k3dash.service:26` |
| Issue or renew TLS | `./issue-cert.sh` (or `--watch`) | `issue-cert.sh:8` |
| Verify/restore the pinned image | `./launch.sh` | `launch.sh:17-37` |

Three caveats, all verified on the live box:

- `verify-auth.sh` used to target the plaintext `http://127.0.0.1:8000` edge,
  which was removed when the API moved to TLS (`nginx/k3.conf:1-2`), so every
  assertion failed to connect rather than finding a real problem. It now targets
  `https://cflox.store` and pins the hostname to loopback with `curl --resolve`,
  so it exercises the real server block even before DNS propagates. It also
  distinguishes the two credential tiers: the **upstream** key must be rejected
  at the edge (401) and a **customer** key accepted (200). All 26 assertions
  pass on the live box. See
  [`docs/DEPLOY.md` §5.1](docs/DEPLOY.md#51-verify-authsh--the-security-assertions).

- `make health` curls `http://127.0.0.1:8001/v1/models` with no credentials
  (`Makefile:11`). Because `VLLM_API_KEY` is set, that returns **401**, not 200.
  A 401 still proves the engine is listening and answering; use
  `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health` (200) for
  an unambiguous liveness check.
- `make gate` still invokes the **deployed** copy at `/scratch/gate.sh`
  (`Makefile:7`), so editing `tests/gate.py` alone changes nothing until
  `deploy.sh install` copies it out. Running `bash tests/gate.sh` directly does
  use the repo copy: it now resolves `gate.py` and `gate-baseline.json` relative
  to its own location, so the two copies cannot silently score against each
  other's baseline.

- The gate's documented timings (~21s full, ~5s quick) assume an **idle**
  engine. Under live customer load the quick tier measured 98.7s; that is
  queueing, not a failure.

---

## Secrets

**No credential is committed.** Every secret is gitignored with a committed
`*.example` twin or is generated on the box.

| File | Contents | Template / origin |
|---|---|---|
| `api-key.txt` | the single upstream vLLM credential | generated; must match `VLLM_API_KEY` |
| `vllm-k3.env` | container env incl. `VLLM_API_KEY` | `vllm-k3.env.example` |
| `customers.tsv` | every live per-customer key | `customers.tsv.example` |
| `dash-password.txt` | dashboard basic-auth password | generated |
| `/etc/nginx/conf.d/00-k3-keys.conf` | `$k3_customer` map over all customer keys | generated by `gen-keys.sh:21` |
| `/etc/nginx/conf.d/k3-limits.inc` | limit zones + the upstream key rewrite | generated by `gen-keys.sh:97` |
| `/etc/nginx/k3-dash.htpasswd` | dashboard basic-auth hash | generated |

The ignore rules are in `.gitignore`. Handling, rotation and blast radius:
[`docs/SECRETS.md`](docs/SECRETS.md).

> `.gitignore:6` instructs "Verify with `make secret-scan` before every push",
> but no `secret-scan` target exists in `Makefile`. `TODO(operator):` add the
> target or amend the comment.

---

## Documentation

| Document | What it is for |
|---|---|
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Rebuilding the box from nothing: prerequisites, three deployment paths, verification, rollback, troubleshooting. |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Day-2 operations and incident response. |
| [`docs/SECRETS.md`](docs/SECRETS.md) | Credential inventory, rotation procedure, handling rules. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Component-level architecture and request path in detail. |
| [`docs/K3-DEPLOYMENT.md`](docs/K3-DEPLOYMENT.md) | The engineering reference: measured numbers, tuning rationale, failure catalog (§7), correctness gate design (§6), quick start (§12). When this README and that file disagree, that file wins. |

---

## What is deliberately not automated

- **Droplet creation.** DigitalOcean's API has no spot field at all — verified as
  0 occurrences of "spot" across 2,968 files in `digitalocean/openapi`
  (`cloud-init.yaml:4-6`, `docs/K3-DEPLOYMENT.md` §"The finding that decides Job 1").
  The MI355X spot droplet must therefore be created by hand in the Control Panel.
  Everything after creation is automated. `make sizes` (`Makefile:18-20`) probes
  whether an MI355X slug is visible to `doctl` on this account.
- **Pointing DNS at the box.** `issue-cert.sh` refuses to call certbot until a
  public resolver returns this host's own public IP for `cflox.store`
  (`issue-cert.sh:13`, `issue-cert.sh:20-34`), because a failed HTTP-01 burns
  Let's Encrypt rate limit at 5 failures per hostname per hour
  (`issue-cert.sh:4-5`). Create the A record yourself; `./issue-cert.sh --watch`
  will poll for up to 4 hours and issue as soon as it resolves.

---

## Context window: settled, with one loose end

`config.yaml` ships `max-model-len: 262144` (`config.yaml:62`). An earlier
revision of the file argued that 262,144 was unsafe, because it had crashed once
with `EngineDeadError` mid-prefill — but that crash was at
`gpu-memory-utilization: 0.96`, which left only ~11.5 GiB/GPU of workspace. At
the current `0.92` (~16.6 GiB/GPU) 256k is **verified** as of 2026-09-18: a
single 249,696-token prefill, 4×212k concurrent and 6×191k concurrent all passed
with zero evictions and zero preemptions, with prompts randomised to defeat
prefix caching, and a request above the cap returns a clean 400 rather than
killing the engine (`config.yaml:42-61`).

128k was rejected for a concrete reason: clients send a fixed `max_tokens`
(Foundry-style SDKs default to 128,000) and vLLM reserves prompt + `max_tokens`
against one shared window, so at 131,072 only 3,072 prompt tokens remained and
nearly every agentic request 400'd (`config.yaml:29-34`).

**Loose end:** the stale snapshots `config.yaml.bak-12800` and
`config.yaml.bak-131072-1635` are still on the box and are excluded from the
repo by `.gitignore`. Raising the window *further* requires re-testing the new
shape under concurrency with unique prompts while watching `dmesg` for
`svm_range_evict_svm_bo_worker` — see
[`docs/DEPLOY.md` §8.1](docs/DEPLOY.md#81-context-window-262144-and-why) and
[`docs/ARCHITECTURE.md` §10](docs/ARCHITECTURE.md).
