# Deployment guide

Rebuilding `https://cflox.store/v1` — Kimi-K3 on 8× MI355X — from nothing.

`/scratch` is a non-persistent 40 TB volume that is destroyed when the droplet
is reclaimed (`cloud-init.yaml:25`). This document assumes you have lost it and
everything on it, including the 1.5 TB of weights.

The authoritative engineering reference is
[`K3-DEPLOYMENT.md`](K3-DEPLOYMENT.md). This guide sequences and explains; it
does not restate the measurements. Where the two disagree, that file wins.

**Contents**

- [1. Prerequisites](#1-prerequisites)
- [2. Path A — fresh droplet, fully automated](#2-path-a--fresh-droplet-fully-automated)
- [3. Path B — existing box, single script](#3-path-b--existing-box-single-script)
- [4. Path C — manual step by step](#4-path-c--manual-step-by-step)
- [5. Verification](#5-verification)
- [6. Post-deploy](#6-post-deploy)
- [7. Rollback and teardown](#7-rollback-and-teardown)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### Hardware

From the box this configuration was measured on (`K3-DEPLOYMENT.md` §3
"Hardware"):

| | |
|---|---|
| GPU | 8× AMD Instinct MI355X VF, `gfx950`, PCI `0x75b3` |
| VRAM | 287.7 GiB/card, 2,304 GB total |
| CPU / RAM | 192 vCPU / 2,015 GiB |
| Scratch | `/dev/vdc1`, 40 TB ext4 on `/scratch` |
| Kernel / amdgpu / ROCm | `6.8.0-137-generic` / `6.19.14.31400000` / 7.14 |

Tensor parallelism is fixed at 8 (`config.yaml:19`) and expert parallelism is
on (`config.yaml:20`), so this config assumes exactly 8 GPUs. Fewer requires
re-tuning `tensor-parallel-size`, `gpu-memory-utilization` and `max-model-len`
together, and re-running the gate.

### Disk

| Consumer | Size |
|---|---|
| Model weights, 96 safetensors | **1.5 TB** at `/scratch/hf/hub/models--moonshotai--Kimi-K3` (`K3-DEPLOYMENT.md` §3 "Model") |
| Docker image | 57.2 GB (`K3-DEPLOYMENT.md` §3 "Image") |
| Optional image tarball | `/scratch/backup/k3-image.tar`, consumed by `launch.sh:13` |

The 40 TB scratch volume covers all of this comfortably; the constraint is
time, not space.

### DNS

`cflox.store` must have an **A record pointing at this box before TLS can be
issued**. `issue-cert.sh` hardcodes `EXPECT_IP=201.79.29.187`
(`issue-cert.sh:13`) and asks `dns.google` rather than the local resolver
(`issue-cert.sh:20-34`); it refuses to invoke certbot until that public
resolver agrees, because a failed HTTP-01 challenge burns Let's Encrypt rate
limit at 5 failures per hostname per hour (`issue-cert.sh:4-5`).

If the rebuilt droplet gets a **different public IP**, `EXPECT_IP` must be
updated or `issue-cert.sh` will never fire. Note also that `ifconfig.me`
reports `201.79.29.187` and whois attributes it to Claro Brazil; that is
correct for this droplet in `mem1` and is not a hijack
(`K3-DEPLOYMENT.md` §"Testing gotcha").

### Packages

`cloud-init.yaml:8` installs `jq`, `zstd`, `python3.12-venv`. `python3.12-venv`
is **not** preinstalled and its absence is a documented bring-up failure
(`K3-DEPLOYMENT.md` §7.8). Additionally required on the host: `docker`
(`k3.service:8-9`), `nginx`, `certbot` plus a webroot at `/var/www/certbot`
(`issue-cert.sh:117-118`), `openssl` (key minting, `gen-keys.sh:49`), plus
`curl`, `python3`, `install`.

`deploy.sh`'s `preflight` stage **checks** for all of these and refuses to
continue if any is missing (`deploy.sh:254-265`), but no stage installs
`docker`, `nginx` or `certbot`, and `cloud-init.yaml` does not either. On a
fresh droplet, install them yourself before running `deploy.sh`.

`htpasswd` (from `apache2-utils`) is **absent** on this box. The live
`/etc/nginx/k3-dash.htpasswd` is an `$apr1$` hash, and `deploy.sh` produces it
with `openssl passwd -apr1` rather than adding a package dependency
(`deploy.sh:260-263`).

### Credentials to have on hand

A Hugging Face token if the model repo requires one (`K3-DEPLOYMENT.md` §4.2
passes `HF_TOKEN`), and an email for Let's Encrypt registration
(`issue-cert.sh:14` defaults to `admin@cflox.store`).

---

## 2. Path A — fresh droplet, fully automated

### 2.1 Create the droplet by hand. This part cannot be scripted.

DigitalOcean's API has **no spot field at all**: `rg -ic spot specification/`
across `digitalocean/openapi` returns **0 matches in 2,968 files**
(`cloud-init.yaml:4-6`, `K3-DEPLOYMENT.md` §"The finding that decides Job 1").
There is therefore no `doctl` flag, no Terraform attribute and no API body
field that requests a spot MI355X instance. **The droplet must be created in
the DigitalOcean Control Panel by hand.** Everything after creation is
automated.

To check whether an MI355X slug is even visible to `doctl` on your account:

```bash
make sizes      # Makefile:18-20
```

### 2.2 Pass `cloud-init.yaml` as user data

If your account *can* create the droplet from the CLI, or you are attaching
user data to a Control-Panel-created droplet:

```bash
doctl compute droplet create <name> \
  --user-data-file cloud-init.yaml \
  ...                              # size/region/image/ssh-keys per the Control Panel
```

(`cloud-init.yaml:3`.) In the Control Panel the same content goes in the
"User data" field under Advanced Options.

### 2.3 What cloud-init does on first boot

| Step | Action | Source |
|---|---|---|
| packages | `jq`, `zstd`, `python3.12-venv` | `cloud-init.yaml:8` |
| sysctl | TCP buffers to 256 MiB, `fq_codel`, cubic — tuning for the 1.5 TB pull | `cloud-init.yaml:11-18`, `:50` |
| scratch | adds `LABEL=DOSCRATCH /scratch ext4` to fstab, mounts it, creates `hf/`, `results/`, `deploy/` | `cloud-init.yaml:26-29` |
| weights | venv at `/opt/hfv`, `hf download moonshotai/Kimi-K3` with `HF_XET_HIGH_PERFORMANCE=1` | `cloud-init.yaml:36-42` |
| serve config | installs `config.yaml` to `/scratch/hf/config.yaml` | `cloud-init.yaml:45` |
| supervision | installs and `enable --now` `k3.service` | `cloud-init.yaml:46-47` |

Re-downloading the weights every rebuild is the measured optimum, not
laziness: ingress is unmetered and uncapped at 2.23 GB/s (~700 s), while egress
is capped at 10 Gbps and billed, so any "cache it elsewhere" scheme pays ≥1249 s
outbound to build a cache that at best ties a free 700 s download
(`cloud-init.yaml:31-35`).

### 2.4 What cloud-init does **not** do

It does not clone this repo, does not create secrets, does not install nginx,
the key layer, TLS, or the dashboard. After cloud-init finishes you have a
working engine on `127.0.0.1:8001` and nothing in front of it. Continue with
**Path B** from the `secrets` stage, or with Path C §4.5 onward.

Also note `cloud-init.yaml:45-47` reads from `/scratch/deploy/`, so the repo
must be cloned there before `k3-bootstrap.sh` reaches step 3. On a genuinely
fresh boot it will not be, and that step fails; clone the repo first, then
re-run `/usr/local/bin/k3-bootstrap.sh`, which is idempotent (it skips the
download if the weights directory already exists, `cloud-init.yaml:38`).

---

## 3. Path B — existing box, single script

```bash
cd /scratch/deploy
sudo ./deploy.sh --help        # authoritative stage list for your checkout
sudo ./deploy.sh --dry-run     # print every mutation, change nothing
sudo ./deploy.sh               # all stages, in order
sudo ./deploy.sh <stage> ...   # re-run one or more stages
sudo ./deploy.sh tls --watch   # poll DNS every 60s for up to 4h, then issue
```

`deploy.sh` is staged so a failure is resumable: fix the cause, re-run that one
stage, then continue. It is idempotent — every stage no-ops when its work is
already done, and `services` uses `enable --now` rather than `restart`
specifically so re-running the script never costs the ~6 minute model reload
(`deploy.sh:631-632`). `--dry-run` is trustworthy because every mutation goes
through a single `run()` wrapper while read-only probes still execute, so the
plan reflects the box's actual state (`deploy.sh:89-90`, `:176-178`).

Run `./deploy.sh --help` for the authoritative stage list. The table below adds
the timing and success signals; §3.2 has the ordering reasoning, which is the
expensive part.

### 3.1 Stages

| # | Stage | Does | Rough time | Success looks like |
|---|---|---|---|---|
| 1 | `preflight` | `/dev/kfd` + `rocm-smi`, docker, nginx, certbot, jq, zstd, python3-venv, openssl, curl; `/scratch` mounted with ≥1600 GiB free (`MIN_FREE_GIB`) | seconds | no FATAL |
| 2 | `host` | Writes `/etc/sysctl.d/99-vllm.conf` with the values from `cloud-init.yaml`, `/scratch` mount + fstab entry, directories, `ufw allow 22/80/443` | ~1 min | `sysctl net.core.rmem_max` = 268435456 |
| 3 | `weights` | `hf download moonshotai/Kimi-K3` into `/scratch/hf`; skipped entirely if `/scratch/hf/hub/models--moonshotai--Kimi-K3` exists | **~12 min** at 2.23 GB/s | that directory exists, 96 safetensors |
| 4 | `secrets` | `api-key.txt`, `dash-password.txt`, `vllm-k3.env` from the template with `VLLM_API_KEY := api-key.txt`, and `/etc/nginx/k3-dash.htpasswd` for user `admin` (`DASH_USER`) | seconds | all four exist, none world-readable |
| 5 | `install` | `config.yaml` → `/scratch/hf/config.yaml`, `dashboard/` → `/opt/k3dash`, units → `/etc/systemd/system`, nginx fragments → `/etc/nginx/conf.d`, `tests/` → `/scratch` | seconds | `systemctl cat k3.service` succeeds |
| 6 | `image` | Verifies the pinned digest **by bits**; restores from `/scratch/backup/k3-image.tar`, else pulls; fails closed on mismatch | seconds cached, ~10 min pulling 57.2 GB | `Image verified: sha256:5aa7e626…` |
| 7 | `keys` | `gen-keys.sh` — `00-k3-keys.conf` + `k3-limits.inc` from `customers.tsv`, then reload | seconds | `generated … with N customer key(s)` |
| 8 | `services` | `daemon-reload`, `enable --now` docker / k3 / k3dash / nginx / `certbot.timer`, then polls `:8001/v1/models` **with the key** until 200 | **~6 min** for the engine | `engine ready after Ns`, then `dashboard healthy on :8080` |
| 9 | `tls` | `issue-cert.sh` (DNS-gated); no-ops if the cert and `k3-tls.conf` are already present, but still re-runs `nginx -t` | ~1 min once DNS is right | `tls in place`, or a deferral warning |
| 10 | `verify` | `verify-auth.sh` (only if the `:8000` edge exists), HTTPS edge probes, `tests/gate.sh` | ~2 min | `verification passed` |

`all` is the default and runs 1–10 in exactly that order (`deploy.sh:79`).

Three behaviours worth knowing before you read the output:

- **The engine poll doubles as a credential check.** `services` polls
  `/v1/models` *with* the key from `api-key.txt`, so a 401 is not a transient
  state — it means `VLLM_API_KEY` in `vllm-k3.env` disagrees with `api-key.txt`,
  and the script dies immediately with that diagnosis rather than waiting out
  the timeout (`deploy.sh:644-659`). It also aborts early if `k3.service` has
  given up, since the unit stops retrying after 5 failures in 30 minutes and
  will not come back on its own (`deploy.sh:660-666`, `k3.service:12-13`).
- **A deferred `tls` stage is not success.** If DNS does not yet resolve to
  `EXPECT_IP`, `issue-cert.sh` exits 1, and `deploy.sh` marks the run incomplete
  and tells you to finish with `./deploy.sh tls --watch` (`deploy.sh:700-707`).
  There is deliberately no `--force` path, because failed HTTP-01 attempts burn
  Let's Encrypt rate limit (`deploy.sh:687-689`).
- **`https /v1/models upstream key → 401` is correct on a fresh box.** The
  `verify` stage probes the edge with the *upstream* key, which is only accepted
  if it also happens to be in `customers.tsv`. With a fresh table it is not, and
  401 is the right answer (`deploy.sh:734-738`). Use a real customer key to see
  200.

The ~6 minute engine start is 140 s of weight loading plus 74 s of engine init
plus overhead (`K3-DEPLOYMENT.md` §3 "Model", §4.3). `k3.service:24` allows
`TimeoutStartSec=900`, and `deploy.sh`'s own `READY_TIMEOUT` matches at 900 s, so
neither gives up before the model is ready.

### 3.2 Ordering constraints, and why they exist

These are not stylistic. Each one was paid for once already.

**Keys before nginx validates.** `gen-keys.sh` emits the
`map $http_authorization $k3_customer` block into
`/etc/nginx/conf.d/00-k3-keys.conf` (`gen-keys.sh:21`, `:75-82`). The `/v1/`
location in `k3-tls.conf` dereferences `$k3_customer` to decide 401 vs pass
(`issue-cert.sh:76`). nginx refuses to load a config that references an
undefined map variable, so the `keys` stage must precede any `nginx -t` that
sees the TLS block. The two files are also coupled in the other direction:
`map_hash_bucket_size` is declared **only** in `00-k3-keys.conf`
(`gen-keys.sh:72`) and must not be redeclared in the TLS block, because a
duplicate directive fails `nginx -t` (`issue-cert.sh:41-42`). It is required at
all because `"Bearer sk-k3-<name>-<40 hex>"` exceeds nginx's default 64-byte
map bucket (`gen-keys.sh:71`).

**The certificate before the TLS server block.** `k3-tls.conf` names
`/etc/letsencrypt/live/cflox.store/fullchain.pem` (`issue-cert.sh:58-59`), and
`nginx -t` fails on a missing certificate file. This is why `issue-cert.sh`
only calls `write_tls_server` after certbot has succeeded and the chain is
non-empty (`issue-cert.sh:130-134`), and why it keeps a `.pre-tls.bak` and
rolls back if validation fails anyway (`issue-cert.sh:133-138`).

**`VLLM_API_KEY` must equal `api-key.txt`, exactly.** nginx rewrites every
customer's `Authorization` header to the single upstream credential read from
`api-key.txt` (`gen-keys.sh:20`, `:110`). vLLM independently enforces
`VLLM_API_KEY` from `vllm-k3.env` (`vllm-k3.env.example:19-23`). If they differ,
a valid customer key passes nginx and then collects vLLM's own 401 — which is
exactly what happened on first setup (`gen-keys.sh:105-108`). Rotating the
upstream key requires a **vLLM restart**; rotating a customer key does not
(`gen-keys.sh:108-109`, `vllm-k3.env.example:20-21`).

**`config.yaml` at `/scratch/hf/config.yaml` before the container starts.** The
unit passes `--config /hf/config.yaml` and only bind-mounts `/scratch/hf` as
`/hf` (`k3.service:34`, `:38`). A config left in `/scratch/deploy` is invisible
inside the container. `launch.sh:42` and `cloud-init.yaml:45` both install it;
`launch.sh` only does so if it is absent, so an edit to the repo copy does
**not** propagate — copy it explicitly and restart.

**Weights before `k3.service`.** `HF_HUB_OFFLINE=1` (`vllm-k3.env.example:17`)
means there is no silent re-download; a missing weights directory is a hard
failure, which `launch.sh:40-41` checks for explicitly. The unit's
`RequiresMountsFor=/scratch` (`k3.service:7`) guarantees only that the mount
exists, not that it has contents.

**Dashboard binds loopback only.** `DASH_ADDR=127.0.0.1`
(`systemd/k3dash.service:22`) is load-bearing: the dashboard has no
authentication of its own, and the basic auth in `k3-dash.inc` is the only
thing protecting it (`systemd/k3dash.service:19-21`). Do not move it to
`0.0.0.0`.

---

## 4. Path C — manual step by step

The escape hatch when a stage fails. This mirrors `K3-DEPLOYMENT.md` §4 and
§12; consult those for the measured detail.

### 4.1 Host prep

```bash
mkfs.ext4 -L DOSCRATCH /dev/vdc1        # DESTRUCTIVE; first setup only
echo 'LABEL=DOSCRATCH /scratch ext4 discard,errors=remount-ro 0 2' >> /etc/fstab
mkdir -p /scratch && mount /scratch
mkdir -p /scratch/hf /scratch/results /scratch/deploy

sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456
sysctl -w net.ipv4.tcp_rmem="4096 87380 268435456"
sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"
sysctl -w net.core.default_qdisc=fq_codel
sysctl -w net.ipv4.tcp_congestion_control=cubic
```

(`K3-DEPLOYMENT.md` §4.1; the same values are persisted by
`cloud-init.yaml:11-18`.)

### 4.2 Weights

```bash
apt-get install -y python3.12-venv jq
python3 -m venv /opt/hfv
/opt/hfv/bin/pip install -U "huggingface_hub[cli]"
export HF_HOME=/scratch/hf
export HF_TOKEN=<token>
/opt/hfv/bin/hf download moonshotai/Kimi-K3
```

(`K3-DEPLOYMENT.md` §4.2.) Expect ~700 s. The `hf_transfer` deprecation warning
from `huggingface_hub` 1.32+ is harmless; Xet delivered 2.23 GB/s unaided.

### 4.3 Secrets

```bash
cd /scratch/deploy
umask 077
openssl rand -hex 24 | sed 's/^/sk-k3-upstream-/' > api-key.txt
sed "s|^VLLM_API_KEY=.*|VLLM_API_KEY=$(cat api-key.txt)|" \
    vllm-k3.env.example > vllm-k3.env
openssl rand -base64 15 > dash-password.txt
printf 'admin:%s\n' "$(openssl passwd -apr1 "$(cat dash-password.txt)")" \
    > /etc/nginx/k3-dash.htpasswd
chmod 600 api-key.txt vllm-k3.env dash-password.txt
chmod 640 /etc/nginx/k3-dash.htpasswd && chown root:www-data /etc/nginx/k3-dash.htpasswd
```

`openssl passwd -apr1` rather than `htpasswd` because `apache2-utils` is not
installed and the live file is an `$apr1$` hash (`deploy.sh:260-263`). The
dashboard username is `admin` (`deploy.sh:63`); the mode `640 root:www-data` and
the `sk-k3-upstream-` prefix are conventions, not enforced by any script.

### 4.4 Install config, units, nginx fragments, dashboard, gate

```bash
install -m644 /scratch/deploy/config.yaml        /scratch/hf/config.yaml
install -m644 /scratch/deploy/k3.service         /etc/systemd/system/k3.service
install -m644 /scratch/deploy/systemd/k3dash.service /etc/systemd/system/k3dash.service
install -m644 /scratch/deploy/nginx/00-default-deny.conf /etc/nginx/conf.d/
install -m640 /scratch/deploy/nginx/k3.conf              /etc/nginx/conf.d/
install -m644 /scratch/deploy/nginx/k3-dash.inc          /etc/nginx/conf.d/
mkdir -p /opt/k3dash && install -m644 /scratch/deploy/dashboard/* /opt/k3dash/
install -m755 /scratch/deploy/tests/gate.sh /scratch/deploy/tests/gate.py /scratch/
install -m644 /scratch/deploy/tests/gate-baseline.json /scratch/
systemctl daemon-reload
```

The gate files go to `/scratch/` because that is where `Makefile:7` and
`tests/gate.sh:6` look for them, and `gate.py` reads its baseline from
`/scratch/gate-baseline.json`.

### 4.5 Image

```bash
cd /scratch/deploy && ./launch.sh
```

`launch.sh` compares the local image ID against
`sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b`
(`launch.sh:12`), restores from `/scratch/backup/k3-image.tar` if present,
otherwise pulls by digest, and **fails closed on a mismatch rather than serving
unknown bits** (`launch.sh:30-36`). If `k3.service` is installed it hands off to
systemd (`launch.sh:45-48`); otherwise it runs the container in the foreground.

Note the image name has a **hyphen**: `vllm-openai-rocm`. The underscore form
does not exist (`K3-DEPLOYMENT.md` §7.1).

### 4.6 Keys and the nginx auth layer

```bash
cd /scratch/deploy
./gen-keys.sh add acme          # mints and prints one customer key
./gen-keys.sh                   # or just regenerate from customers.tsv as-is
```

This writes `/etc/nginx/conf.d/00-k3-keys.conf` and
`/etc/nginx/conf.d/k3-limits.inc`, runs `nginx -t`, and reloads only if
validation passes (`gen-keys.sh:117-121`). Per-customer ceilings currently
applied (`gen-keys.sh:38-41`):

| Limit | Value | Why |
|---|---|---|
| `limit_conn k3_percust` | 200 | Sized for a key shared by a whole portal, not one developer; 200 of `max-num-seqs` 512 is ~39% of engine slots (`gen-keys.sh:25-29`) |
| `limit_req` rate | 100r/s | Must scale with the connection cap or it becomes the real limit: 200 in-flight requests at ~3 s each turn over ~67/s (`gen-keys.sh:30-33`) |
| burst | 200 | Absorbs the login-time stampede |
| `limit_conn k3_global` | 512 | Matches `max-num-seqs` so one portal cannot starve the others (`gen-keys.sh:36-37`) |

Both generated files are mode 640 because they hold live keys in cleartext;
nginx reads them as root at load, so the worker user never needs them
(`gen-keys.sh:64-66`).

### 4.7 Start the services

```bash
systemctl enable --now k3.service k3dash.service
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health)" = 200 ]; do sleep 10; done
```

Confirm the backends selected, on every launch (`K3-DEPLOYMENT.md` §3
"Selected backends"):

```bash
docker logs k3 2>&1 | grep -E "ROCM_AITER_FA MLA prefill|ROCM_AITER_MLA backend|quantization=mxfp4|cudagraph_mode"
docker logs k3 2>&1 | grep -E "GPU KV cache size|Maximum concurrency"
```

The restart policy is deliberate: `RestartSec=30s`, `RestartSteps=4`,
`RestartMaxDelaySec=10min`, giving up after 5 failures in 30 minutes
(`k3.service:12-13`, `:21-23`). Docker's own backoff cannot do this job — it
resets to 100 ms once a container has run ≥10 s, and this model takes ~6 min to
load, so every crash clears that threshold (`k3.service:17-20`).

### 4.8 TLS

```bash
cd /scratch/deploy
./issue-cert.sh            # issue now, if DNS already points here
./issue-cert.sh --watch    # poll DNS every 60s for up to 4h, issue when ready
```

It tries `cflox.store` + `www.cflox.store` and falls back to the bare domain,
because certbot fails the whole request if any single `-d` name fails HTTP-01
(`issue-cert.sh:115-128`). On fallback it drops `www` from `server_name` so
nginx does not claim a name it cannot serve.

The generated `k3-tls.conf` uses `listen 443 ssl http2` rather than the
standalone `http2 on;` directive, because nginx 1.24 rejects the latter — it
was added in 1.25.1 (`issue-cert.sh:53-55`).

---

## 5. Verification

Both suites must pass. They test different things: one proves the edge is
locked, the other proves the model is not producing fluent nonsense.

### 5.1 `verify-auth.sh` — the security assertions

```bash
cd /scratch/deploy && ./verify-auth.sh
```

Exit code is the number of failed assertions; `ALL CHECKS PASSED` is the only
acceptable output. The classes of assertion and what each proves:

| Class | Assertions | What it proves |
|---|---|---|
| The hole `--api-key` does not close | `POST /invocations`, `/scale_elastic_ep`, `/tokenize`, `/detokenize`, `/generative_scoring`, `GET /load`, `/version`, `/openapi.json`, `/docs` all → 403 (`verify-auth.sh:31-40`) | vLLM's `--api-key` guards only the prefixes `("/v1","/v2","/inference")` (`server_utils.py:42`). This build exposes 25 routes; 11 fall outside that tuple. **`POST /invocations` was measured answering 200 with no credentials** — full inference, free, on a $36/hr box (`verify-auth.sh:3-6`, `K3-DEPLOYMENT.md` §"Why `--api-key` alone is NOT sufficient"). These assertions are the proof that nginx still closes it. |
| Inference requires the key | no key → 401, wrong key → 401, valid key → 200, on `/v1/models`, `/v1/chat/completions`, `/v1/completions` (`verify-auth.sh:43-54`) | The `$k3_customer` map is actually wired into the `/v1/` location. |
| Diagnostics are local-only | `/metrics`, `/health` → 200 from 127.0.0.1 (`verify-auth.sh:57-59`) | The `allow 127.0.0.1; deny all` pair (`issue-cert.sh:72-73`) lets the dashboard scrape while refusing everyone else. |
| vLLM is not directly reachable | direct `:8001/v1/models` → 401, and `:8001` bound to `127.0.0.1` (`verify-auth.sh:62-65`) | Defence in depth: bypassing nginx still hits vLLM's own 401. |
| Dashboard is alive and honest | `:8080/`, `/api/state`, `/api/health` → 200; `server.state == "up"`; `cache.capacity_tokens > 0` (`verify-auth.sh:68-81`) | The monitor works and agrees the engine is live. It deliberately does not count nulls — empty percentile lists and a null `gpu.error` are the healthy case. |
| End to end | `17*23` answered through the authenticated edge, asserting `391` appears anywhere in `reasoning_content + content` (`verify-auth.sh:84-91`) | A real answer traverses the whole path. Written as "digits appear" rather than equality because K3 emits its chain before the answer. |

> **Known stale target.** `verify-auth.sh:9` sets `EDGE=http://127.0.0.1:8000`,
> but the plaintext `:8000` edge was removed when the API moved to TLS
> (`nginx/k3.conf:1-2`). On the live box nothing listens on 8000 (`ss -tln`
> shows only 80, 443, 8001, 8080), so every `$EDGE` assertion currently fails to
> connect — for the wrong reason. `deploy.sh`'s `verify` stage works around this
> by running the suite **only if** something is listening on `:8000`, and
> substituting HTTPS-edge probes otherwise (`deploy.sh:714-743`).
> `TODO(operator):` repoint `EDGE` at `https://cflox.store` with a customer key,
> or make it an env var, and delete that branch. Until then the `:8001` and
> `:8080` assertions are the only ones that can run standalone; use the five
> checks at `issue-cert.sh:141-153` for edge coverage.

### 5.2 `tests/gate.sh` — the correctness gate

```bash
make gate      # full, ~21 s   (Makefile:6-7)
make quick     # tier 1, ~5 s  (Makefile:8-9)
bash /scratch/gate.sh --baseline   # re-record after an INTENDED change
```

**Run it after every launch and after any change to the AITER environment
variables in `vllm-k3.env`** (`vllm-k3.env.example:7-9`, `Makefile:6`). This is
not ceremony. Several configurations start cleanly, serve at full speed, and
emit fluent nonsense; large-batch benchmarks do not expose that failure, only
correctness probes do (`K3-DEPLOYMENT.md` §6).

| Tier | Checks | Fails when |
|---|---|---|
| 1 CORRUPTION | 3 factual probes scored by rank + margin ≥ 0.8 over the runner-up, numeric continuation, semantic stability across 3 samples | The right token loses its lead — the signature of a layout/dtype mismatch |
| 2 REASONING | 16 arithmetic word problems, exact answers, floor 14/16, plus a regression guard at 2 problems below baseline | Accuracy degrades even while staying above the absolute floor |
| 3 LONGCTX | needle retrieval at ~12,026 prompt tokens | Long-context prefill corrupts logits (vLLM #51039 reports NaN logits after long prefill) |

Design points worth not relitigating (`K3-DEPLOYMENT.md` §6):
rank+margin rather than an absolute logprob floor, because `" Paris"` sits near
−0.24 and `" Celsius"` near −0.70 and both are healthy; **semantic** stability
rather than bitwise determinism, because vLLM is not bitwise deterministic at
temperature 0 — 5 identical requests produced 3 distinct texts on a healthy
server. The gate is validated in both directions: an unreachable server exits 1
rather than silently passing.

Baseline lives at `/scratch/gate-baseline.json`; the committed copy is
`tests/gate-baseline.json`, recorded `2026-09-18T12:25:45Z` at
`tier2 16/16`, `tier3_prompt_tokens 12026`.

### 5.3 Optional deeper suites

| Script | Target | Covers |
|---|---|---|
| `tests/agentic-test.py` | `127.0.0.1:8001` | Parallel tool calls, forced `tool_choice`, 31 tool schemas, `json_object`/`json_schema`, vision, stop sequences, `/v1/messages` |
| `tests/ctx-test.py` | `127.0.0.1:8001` | Needle retrieval at controlled depth and length |
| `tests/ide-ready.py` | `https://cflox.store/v1` with a **customer** key | SSE deltas, streaming tool-call assembly, tool-result round trip, usage accounting, oversized-request behaviour |

`ide-ready.py` needs a real customer key, not the upstream key
(`tests/ide-ready.py:19-27`):

```bash
K3_CUSTOMER_KEY=$(awk -F'\t' '!/^#/&&NF{print $2; exit}' /scratch/deploy/customers.tsv) \
  python3 tests/ide-ready.py
```

Budget ≥512 `max_tokens` on structured-output or long-context probes before
concluding anything is broken: a `json_schema` call at 200 returned empty
content but actually needed 232, and a needle probe at 48 spent its whole
budget reasoning (`K3-DEPLOYMENT.md` §"Two findings that look like bugs and are
not").

---

## 6. Post-deploy

### 6.1 Onboard a customer

```bash
cd /scratch/deploy
./gen-keys.sh add <name>
```

Mints `sk-k3-<name>-<40 hex>` (`gen-keys.sh:49`), appends it to
`customers.tsv` at mode 600, regenerates the nginx map and limit zones,
validates, and reloads. The key is printed **once**, to stdout — capture it
then. No vLLM restart is needed, because nginx swaps the customer key for the
upstream credential at proxy time (`gen-keys.sh:105-110`).

The `<name>` becomes the identity in the access log and the key for the rate
and concurrency zones, so it must be unique and contain no whitespace
(`customers.tsv.example:10-12`). Never hand-edit `customers.tsv`; the nginx
layer is generated from it and will not be rebuilt.

Give the customer:

```
base_url: https://cflox.store/v1
model:    FW-Kimi-K3        # or the alias kimi-k3
api_key:  <the minted key>
```

Both model names resolve. `kimi-k3` is retained as an alias because live
customer integrations still send it and dropping it 404s them
(`config.yaml:10-16`).

### 6.2 Revoke

```bash
./gen-keys.sh revoke <name>
```

Effective on the reload the script performs at the end (`gen-keys.sh:55-60`,
`:117-121`). This is the whole point of per-customer keys: before them there was
one shared key with no identity, no quota and no rate limit, so a leak by any
one customer compromised everybody and rotation broke every client at once
(`gen-keys.sh:5-12`).

### 6.3 Dashboard

Served at `https://cflox.store/` behind basic auth
(`nginx/k3-dash.inc:3-8`), proxying to `k3dash.service` on `127.0.0.1:8080`.
Password is in `dash-password.txt`; the hash nginx checks is
`/etc/nginx/k3-dash.htpasswd`. It is mounted at `/` and not `/dash/` because
`index.html` fetches the absolute path `/api/state`, which a prefix would break
(`nginx/k3-dash.inc:1-2`).

The unit runs `ProtectSystem=strict`, `ProtectHome=true`, `NoNewPrivileges=true`
(`systemd/k3dash.service:31-35`) — read-only is all it needs, since it scrapes
loopback, tails the nginx log, and shells out to `rocm-smi`. It is only
`Wants=k3.service`, not `Requires=`, so it still starts and reports the engine
as down when k3 is crash-looping (`systemd/k3dash.service:6-8`).

### 6.4 Usage log

`/var/log/k3/usage.log`, format `k3usage`, defined at `gen-keys.sh:91-93`:

```
$time_iso8601 cust=$k3_customer status=$status path=$request_uri
req_ms=$request_time up_ms=$upstream_response_time
in=$request_length out=$body_bytes_sent ip=$remote_addr
```

Written only from the `/v1/` location (`issue-cert.sh:79`), so it is a clean
per-customer billing and attribution record. The dashboard tails the same file
(`systemd/k3dash.service:26`).

`TODO(operator):` there is no logrotate rule for this file in the repo. Confirm
one exists on the box or add it — it is the billing record and it grows with
traffic.

### 6.5 What to watch

| Metric | Healthy | Meaning if wrong |
|---|---|---|
| `vllm:num_requests_waiting` | ~0 | Sustained >50 means offered load exceeds capacity |
| KV utilization | <70% | Approaching 100% causes preemption and collapse |
| prefix cache hit rate | ~81% (`make cache`) | A drop means traffic lost its shared prefix; capacity falls ~3× |
| GPU utilization | 85–100% | Low with a full queue implies a stall, not saturation |

(`K3-DEPLOYMENT.md` §9 "Key metrics to watch in production".) Ignore the
`Maximum concurrency` line in the startup log — it assumes no prefix sharing
and is pessimistic by ~5× for this workload.

---

## 7. Rollback and teardown

### 7.1 Roll back a serve-config change

```bash
cp /scratch/deploy/config.yaml.bak-131072-1635 /scratch/hf/config.yaml
systemctl restart k3.service          # ~6 min
make gate
```

The repo carries two snapshots: `config.yaml.bak-12800` (the original benchmark
window) and `config.yaml.bak-131072-1635` (the 131,072 window). The latter
differs from the live `config.yaml` **only at line 51**, the `max-model-len`
value — see §8.1.

Remember `launch.sh:42` installs `config.yaml` only if `/scratch/hf/config.yaml`
is absent, so you must copy over it explicitly.

### 7.2 Roll back the TLS block

`issue-cert.sh:133` saves `${TLSCONF}.pre-tls.bak` before overwriting and
restores it automatically if `nginx -t` fails (`issue-cert.sh:135-138`). To do
it by hand, restore that file and reload nginx.

### 7.3 Roll back the key layer

`customers.tsv` is the source of truth; `gen-keys.sh` with no arguments rebuilds
both generated nginx fragments from it. Restore the TSV, re-run `./gen-keys.sh`.

### 7.4 Roll back the image

`launch.sh` restores from `/scratch/backup/k3-image.tar` when the local image ID
does not match the pinned digest (`launch.sh:19-21`). Keeping that tarball on
scratch is what makes a rebuild independent of the upstream `kimi-k3` tag, which
is a one-off dev tag that may be garbage-collected (`launch.sh:10-12`).

### 7.5 Teardown

Stop serving, in this order:

```bash
systemctl disable --now k3.service k3dash.service
```

Then destroy the droplet in the Control Panel. `/scratch` — weights, image
tarball, results, and every secret file — is destroyed with it
(`cloud-init.yaml:25`). Nothing else needs cleaning up, which is the point of
keeping all state on the ephemeral volume. Before destroying, extract anything
you need to keep: `/var/log/k3/usage.log` (billing record) and
`customers.tsv` (so existing customer keys survive the rebuild — otherwise every
customer must be re-issued).

---

## 8. Troubleshooting

### 8.1 Context window: 262144, and why

**Status: settled by measurement on 2026-09-18. `max-model-len: 262144` at
`gpu-memory-utilization: 0.92` is the verified configuration
(`config.yaml:62`, `config.yaml:77`).**

This entry is kept because the reasoning is what protects the next person: the
value was raised, reverted, and raised again, and the file's own comments
contradicted each other for part of that history.

**Why not 131,072.** Clients send a *fixed* `max_tokens` — Foundry-style SDKs
default to 128,000 — and vLLM reserves prompt + `max_tokens` against one shared
window (`renderers/params.py`: `max_input_tokens = max_total_tokens -
max_output_tokens`). Usable prompt is therefore `max-model-len - 128,000`. At
131,072 that left **3,072** prompt tokens and 400'd nearly every agentic
request. 256k leaves 134,144. 204,800 was rejected as too tight
(`config.yaml:29-34`).

**Why 262,144 was once believed unsafe.** It was tried at
`gpu-memory-utilization: 0.96` and killed the engine mid-prefill at
`num_computed_tokens=186,624` with `EngineDeadError`. This was *not* KV
exhaustion — `kv_cache_usage` was 0.084 at death. The cause was VRAM headroom:
`dmesg` logged `svm_range_evict_svm_bo_worker` (amdgpu evicting buffer objects)
at the exact second of the crash, because 0.96 left only ~11.5 GiB/GPU spare
(`config.yaml:42-48`).

**Why it is safe now.** 0.92 leaves ~16.6 GiB/GPU, measured via `rocm-smi` with
the engine loaded and serving. An earlier note in the file claimed ~23 GiB,
which was wrong in the dangerous direction and has been retracted
(`config.yaml:69-76`). At 0.92 the following all passed with **zero new
evictions**:

| Shape | Result |
|---|---|
| single prefill, 249,696 tok | PASS, 0 new evictions |
| 4 × 212k concurrent, unique prompts (848k) | PASS, 95s, 0 evictions |
| 6 × 191k concurrent, unique prompts (1.15M) | PASS, 59s, 0 evictions, 0 preemptions |
| request > 262,144 | clean 400, engine survives |

Prompts were randomised specifically to defeat prefix caching; an earlier run
with identical prompts finished in 6s and proved nothing (`config.yaml:49-56`).
Measured KV pool at 0.92 is 2,295,266 tokens, giving 8.76× worst-case
concurrency at 256k (`config.yaml:39-40`) — confirmed live at
`capacity.capacity_tokens` on the dashboard's `/api/state`.

**Still loose:** the snapshots `config.yaml.bak-12800` and
`config.yaml.bak-131072-1635` remain on the box, and the latter no longer
matches its own name's implication now that the prose has been rewritten. Both
are excluded from the repo by `.gitignore` (`*.bak-*`). `K3-DEPLOYMENT.md`
predates this resolution and still records 131,072 @ 0.92 as the shipped
configuration; `config.yaml` is the authority.

**If you raise the window further**, the same test applies and has not been run
for any shape above 262,144 (`config.yaml:58-61`): drive a prompt above the new
cap **under concurrency with unique prompts**, not single-request, because
several simultaneous long prefills multiply the workspace demand that caused the
original crash at single-request load. Randomise the prompts or prefix caching
will make the test pass for free. Watch `dmesg` for
`svm_range_evict_svm_bo_worker` throughout. `tests/ctx-test.py` builds needle
prompts at controlled lengths and is the natural starting point; it must be
driven in parallel to make the test valid.

Do **not** raise `gpu-memory-utilization` to buy the room — that is the exact
change that caused the crash, and the ~16.6 GiB/GPU it currently leaves is what
makes 256k survivable (`config.yaml:69-77`). Expect an engine restart to cost
~6 minutes of downtime.

### 8.2 Fluent but wrong output — the AITER pairing

The most dangerous failure mode here, because nothing looks broken.
`AITER_SITUV2_A8W4=1` forces an interleaved weight layout **and** selects
activation dtype by batch size. Below the bf16/fp8 boundary the kernel feeds
bf16 activations to interleaved A8W4 weights — a silent layout mismatch that
produces fluent gibberish at full speed (`K3-DEPLOYMENT.md` §7.3).

`AITER_BF16_FP8_MOE_BOUND=0` pins fp8 activations at every batch size and is the
fix (`vllm-k3.env.example:10-13`). **Never change one of these without the
other, and always run the gate afterwards** (`vllm-k3.env.example:7-9`).

Detection: small-batch coherence checks trip it; large-batch benchmarks do not.
That is exactly what gate tier 1 is for. Symptom after a correct fix:
`logprob(Paris) = -0.242`, counting intact.

### 8.3 `EngineDeadError` mid-prefill on a long prompt

Symptom: the worker dies with no Python traceback; the parent reports only
`RuntimeError: cancelled` from `shm_broadcast.py:703`, which is downstream
noise.

Diagnosis, from the one time this happened
(`K3-DEPLOYMENT.md` §"What was measured, and the crash"):

- Not KV exhaustion — `kv_cache_usage` was 0.084 at the moment of death.
- Not host OOM — no oom-kill, 77 GiB of 2,015 GiB used.
- The evidence was in **`dmesg`**: `svm_range_evict_svm_bo_worker [amdgpu]` at
  the exact second of the crash. The driver was evicting GPU buffer objects.
  VRAM headroom, not cache capacity.

```bash
dmesg -T | grep -i 'svm_range_evict\|amdgpu'
```

**Lesson: when a vLLM worker dies with no Python traceback, check `dmesg` for
amdgpu eviction before assuming a KV or scheduler bug.** Remedy is to lower
`max-model-len` (see §8.1), not to raise `gpu-memory-utilization` — 0.96 is what
caused it (`config.yaml:57-63`).

### 8.4 Startup fails: no MXFP4 MoE backend

```
NotImplementedError: No MXFP4 MoE backend supports the deployment configuration.
```

`VLLM_ROCM_USE_AITER_MOE_SITUV2` is **not a real variable**. Use
`VLLM_ROCM_USE_AITER=1` plus `AITER_SITUV2_A8W4=1` (`K3-DEPLOYMENT.md` §7.2),
both of which are already in `vllm-k3.env.example:10-12`.

### 8.5 Other catalogued failures

| Symptom | Cause / fix | Reference |
|---|---|---|
| Image pull fails | Name is `vllm-openai-rocm` with a hyphen; `vllm-openai_rocm` does not exist | §7.1 |
| `mla_gluon[bh16bn128] requires batch_size=1, got 128` | fp8 KV cache is unusable at TP=8 (12 heads/rank, below the 16-head threshold). Drop `--kv-cache-dtype fp8`, use bf16. Moot anyway — KV sits at 49% | §7.4 |
| `AssertionError: ... only supports the Triton KDA prefill backend, got 'flashkda'` | Remove `--kda-prefill-backend` entirely, at any value | §7.5 |
| No speedup from `torch.compile` | It is inert on K3; `FULL_DECODE_ONLY` is what wins, by freeing ~11.7 GiB into KV | §7.6, §2.3 |
| `--disable-log-requests: unrecognized` | Removed in vLLM 0.27+; use `no-enable-log-requests` (`config.yaml:91`) | §7.7 |
| `python3 -m venv` fails | `apt-get install -y python3.12-venv`; not preinstalled | §7.8 |
| `Op 'grouped_topk' not present…`, Triton JIT messages | Harmless; JIT messages are first-touch compilation, covered by warmup | §7.9 |

All references are to `K3-DEPLOYMENT.md`.

### 8.6 Config changes that silently do nothing

- **`some-flag: false` in `config.yaml`.** The parser emits *nothing* for a
  false boolean (`argparse_utils.py:571-573`), silently giving you the flag's
  default rather than the disabled state. Use the positive form of the negative
  flag, e.g. `no-enable-log-requests: true` (`config.yaml:4-8`).
- **`block-size`.** Deliberately absent. Setting 128 was a no-op: this is a
  hybrid model and vLLM overrides it to 768 to keep the attention page ≥ the
  mamba page (`interface.py:911`), then pads the mamba page by 8.68%
  (`config.yaml:66-72`).
- **Editing `/scratch/deploy/config.yaml` alone.** The container reads
  `/hf/config.yaml`, i.e. `/scratch/hf/config.yaml` (`k3.service:34`, `:38`).
  Copy it over and restart.
- **Editing `tests/gate.py` alone.** `Makefile:7` and `tests/gate.sh:6` run the
  copies at `/scratch/`. Reinstall them.

- **Disabling prefix caching.** Don't. `enable-prefix-caching`
  (`config.yaml:75`) is load-bearing: an 85% shared prefix yields roughly 3×
  throughput, and losing it costs ~3× capacity (`config.yaml:74`,
  `K3-DEPLOYMENT.md` §2.1).

### 8.7 Every request 401s after a key change

Check `VLLM_API_KEY` in `vllm-k3.env` against `api-key.txt` — they must match
exactly (`vllm-k3.env.example:22`). nginx rewrites the customer's header to
whatever `api-key.txt` held **at the time `gen-keys.sh` last ran**
(`gen-keys.sh:20`, `:110`), so changing `api-key.txt` requires re-running
`gen-keys.sh` **and** restarting `k3.service`.

### 8.8 nginx will not reload

`gen-keys.sh:117-121` and `issue-cert.sh:135-138` both refuse to reload on an
invalid config, which means a failed reload leaves the previous config serving.
Run `nginx -t` and read the tail. Recurring causes seen here:

| Error | Cause |
|---|---|
| `could not build map_hash` / unknown `$k3_customer` | `00-k3-keys.conf` missing or not generated — run `./gen-keys.sh` |
| duplicate `map_hash_bucket_size` | It belongs only in `00-k3-keys.conf` (`gen-keys.sh:72`); never redeclare it in the TLS block (`issue-cert.sh:41-42`) |
| cannot load certificate | The TLS block was written before certbot succeeded (`issue-cert.sh:130-134`) |
| `http2` directive rejected | nginx 1.24 needs `listen 443 ssl http2`, not standalone `http2 on;` (`issue-cert.sh:53-55`) |

### 8.9 Testing from the box proves nothing

Curling your own public IP from the box passes through
`-A ufw-before-input -i lo -j ACCEPT`, which accepts everything routed via
loopback — and traffic to a local address takes that path even when the address
is a public one on eth0. **Probe from off-box.** Most third-party fetchers
(codetabs, allorigins) return 522 on non-standard ports; `https://r.jina.ai/`
does work and reports the status code (`K3-DEPLOYMENT.md` §"Testing gotcha, hit
twice now").

### 8.10 Clients report 429

Expected under the per-customer caps in §4.6. nginx answers with a JSON
`rate_limit_error` and `Retry-After: 2` (`issue-cert.sh:94-99`). If a legitimate
portal is hitting it, raise `PER_CUST_CONN` / `PER_CUST_RATE` / `PER_CUST_BURST`
in `gen-keys.sh:38-40` together — raising the connection cap alone does nothing,
because the rate cap then becomes the real limit (`gen-keys.sh:30-33`).

### 8.11 Clients report 413

Request bodies are capped at 256 MB (`issue-cert.sh:66`) and the error message
tells the caller to pass an `image_url` instead of base64
(`issue-cert.sh:101-105`).

### 8.12 Streams cut off at 60 seconds

That is nginx's default `proxy_read_timeout`. The TLS block sets 900 s with
`proxy_buffering off` (`issue-cert.sh:86-91`) precisely because a
12,288-in/512-out request runs ~112 s median. If you see 60 s truncation, the
`/v1/` location being served is not the generated one.
