# RUNBOOK — day-2 operations, Kimi-K3 on 8x MI355X

Task-oriented operations for `https://cflox.store/v1`. If you are on call and
did not build this, start at §9 (incident playbooks) and come back for context.

This document covers **running** the system. It does not cover:

| You want to | Read |
|---|---|
| Rebuild the box, bring up a fresh droplet, roll back a change | [`DEPLOY.md`](DEPLOY.md) |
| Understand why the design is the way it is | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Handle, rotate, or audit a credential | [`SECRETS.md`](SECRETS.md) |
| Measured numbers, tuning history, failure catalog | [`K3-DEPLOYMENT.md`](K3-DEPLOYMENT.md) |

All commands run as root from `/scratch/deploy`.

---

## 1. What is running

| Unit | What it is | Address | Restart policy |
|---|---|---|---|
| `k3.service` | vLLM in docker container `k3`, image pinned by digest `sha256:5aa7e626ff73...` (`k3.service:29-38`) | `127.0.0.1:8001` (`config.yaml:106-107`) | systemd, escalating (§2.2) |
| `k3dash.service` | dashboard, `python3 server.py` from `/opt/k3dash` (`systemd/k3dash.service:13-14`) | `127.0.0.1:8080` (`systemd/k3dash.service:22-23`) | `Restart=always`, `RestartSec=5s` |
| `nginx.service` | the public edge: TLS on 443, auth, quotas, attribution | `0.0.0.0:80`, `0.0.0.0:443` | distro default |
| `certbot.timer` | certificate renewal, twice daily | — | `OnCalendar=*-*-* 00,12:00:00` |

Neither backend is reachable from off-box: both bind loopback, and nginx is the
only listener on a public address. `ss -tln` on a healthy box shows exactly
`80`, `443`, `127.0.0.1:8001`, `127.0.0.1:8080` — nothing on `8000` (that
plaintext edge was removed when the API moved to TLS, `nginx/k3.conf:1-2`).

---

## 2. Service control

### 2.1 Commands

```bash
systemctl status  k3.service k3dash.service
systemctl restart k3.service          # ~6 min before it serves again
systemctl stop    k3.service          # customer-visible outage
systemctl start   k3.service
systemctl restart k3dash.service      # ~instant, monitoring only
systemctl reload  nginx               # config changes; never needs a vLLM restart
journalctl -u k3.service -n 50 --no-pager
```

**systemd owns the container. Do not `docker run`, `docker stop` or
`docker rm` `k3` by hand.** `launch.sh:2-6` states this directly ("Normally you
do NOT run this — systemd owns the container"), and `launch.sh:45-48` execs
`systemctl restart k3.service` whenever the unit is installed. Two concrete
reasons:

- The unit runs the container with `--rm` (`k3.service:29`) and
  `ExecStartPre=-/usr/bin/docker rm -f k3` (`k3.service:28`). A hand-started
  container is not the unit's `MainPID`, so systemd's supervision, backoff and
  `ExecStop=docker stop -t 60` (`k3.service:39`) do not apply to it.
- `docker stop k3` makes the unit's `docker run` exit, which systemd reads as a
  failure and restarts 30 s later (`k3.service:17`, `k3.service:21`) — so the
  stop does not stick and you have paid a 6-minute reload for nothing.

`launch.sh` is still the right tool for one job: verifying or restoring the
pinned image (`launch.sh:17-37`). It fails closed on an image-ID mismatch rather
than serving unknown bits.

### 2.2 Restart cost and backoff

Restarting the engine is not cheap. Measured on the currently running instance:

```
Loading weights took 135.45 seconds       (docker logs k3)
init engine (profile, create kv cache, warmup model) took 79.76 s
```

Budget ~6 min end to end (`deploy.sh:692`: "140s weights + 74s engine init";
`k3.service:19`). `TimeoutStartSec=900` is sized for it (`k3.service:24`).

Backoff is systemd's, deliberately, and the unit explains why
(`k3.service:18-20`): docker's own restart backoff **resets to 100 ms once a
container has run >= 10 s**, and this model takes ~6 min to load, so every
crash clears that threshold and docker would hot-loop a 6-minute startup
forever. systemd escalates instead:

| Setting | Value | Effect |
|---|---|---|
| `Restart=always` | — | any exit is retried (`k3.service:17`) |
| `RestartSec=30s` | 30 s | first retry delay (`k3.service:21`) |
| `RestartSteps=4` | 4 | geometric escalation from 30 s to the max (`k3.service:22`) |
| `RestartMaxDelaySec=10min` | 10 min | ceiling (`k3.service:23`) |
| `StartLimitIntervalSec=1800` / `StartLimitBurst=5` | 5 in 30 min | after that systemd **gives up** (`k3.service:12-13`) |
| `TimeoutStopSec=120` vs `docker stop -t 60` | — | stop has room to finish (`k3.service:25`, `:39`) |

A unit that hit the start limit will not come back on its own. See §9.1.

---

## 3. Health checks

### 3.1 The fast loop

```bash
make health     # HTTP code from :8001/v1/models + the two KV log lines
make cache      # live prefix-cache hit rate
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/health   # 200 = alive
```

Observed on the live box while writing this:

```
$ make health
http 401
GPU KV cache size: 2,295,266 tokens
Maximum concurrency for 262,144 tokens per request: 8.76x

$ make cache
hit_rate=74.65%
```

**`make health` printing `http 401` is correct, not a fault.** `Makefile:11`
curls `/v1/models` with no credentials, and vLLM enforces `VLLM_API_KEY`
itself, so 401 is the right answer — it still proves the engine is listening
and answering. For an unambiguous liveness probe use `/health`, which is
unauthenticated on loopback and returns **200** (verified).

| Probe | Healthy | Why |
|---|---|---|
| `http://127.0.0.1:8001/health` | 200 | liveness, no credential needed |
| `http://127.0.0.1:8001/v1/models` (no key) | 401 | `VLLM_API_KEY` is enforced by vLLM |
| `http://127.0.0.1:8080/api/health` | 200 | dashboard process is alive |
| `https://cflox.store/v1/models` (no key) | 401 | `$k3_customer` map rejects it (`issue-cert.sh:76`) |
| `https://cflox.store/` (no basic auth) | 401 | dashboard behind `auth_basic` (`nginx/k3-dash.inc:4-5`) |
| `https://cflox.store/anything-else` | 403 | allowlist catch-all (`issue-cert.sh:109`) |

All six verified from the box on 2026-09-18. To probe the edge from the box
without depending on DNS, pin the name to loopback the way `verify-auth.sh:36`
does: `curl --resolve cflox.store:443:127.0.0.1 https://cflox.store/...`.

`/health` and `/metrics` are `allow 127.0.0.1; deny all` at the edge
(`issue-cert.sh:72-73`), so these probes work from the box and from nowhere else.

### 3.2 What "healthy" looks like in the engine log

Two lines appear once per launch and are the engine's own statement of its
capacity:

```bash
docker logs k3 2>&1 | grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency"
```

Current instance:

```
Available KV cache memory: 61.06 GiB
GPU KV cache size: 2,295,266 tokens
Maximum concurrency for 262,144 tokens per request: 8.76x
```

The KV pool matches the figure `config.yaml:39` records as measured at
`gpu-memory-utilization: 0.92` (2,295,266 tokens), and `8.76x` is exactly the
`256k -> 8.76x` entry in the worst-case ladder at `config.yaml:40`. If either
number differs after a restart, the engine did not load the config you think it
did — see §9.5 and §10.

**Ignore `Maximum concurrency` as a capacity estimate.** It assumes no prefix
sharing and is pessimistic by ~5x for this workload
(`K3-DEPLOYMENT.md` §9); the production run sustained 427 concurrent requests.
Watch actual KV utilisation instead.

Note that `K3-DEPLOYMENT.md` §9 quotes `61.09 GiB / 1,696,278 tokens /
132.52x` — those are from the old 12,800-token configuration and are the
numbers you should expect *not* to see today.

### 3.3 Prefix cache

`make cache` divides the lifetime counters `vllm:prefix_cache_hits_total` and
`vllm:prefix_cache_queries_total` (`Makefile:14-15`). It is cumulative since
engine start, so a cold engine reads low and recovers slowly. The expectation
is ~81% (`Makefile:13`, `K3-DEPLOYMENT.md` §9 and the production run's
`prefix cache hit rate 81.00%`). Today's 74.65% is a lifetime figure on an
instance ~30 min old; a sustained drop means traffic lost its shared prefix and
capacity falls ~3x (`K3-DEPLOYMENT.md` §2.1). For a *windowed* rate rather than
a lifetime one, read `window.cache_hit_rate` from the dashboard
(`dashboard/server.py:661-664`).

### 3.4 Dashboard

`https://cflox.store/`, basic auth, user `admin`, password in
`dash-password.txt` (`deploy.sh:806-807`; handling in
[`SECRETS.md`](SECRETS.md)). It has **no authentication of its own** — nginx's
basic auth is the only thing protecting it, which is why `DASH_ADDR` must stay
`127.0.0.1` (`systemd/k3dash.service:19-22`).

Three endpoints, all `GET` (`dashboard/server.py:921-933`):

| Path | Returns |
|---|---|
| `/` (or `/index.html`) | the UI |
| `/api/state` | full JSON snapshot: `server`, `capacity`, `window`, `latest`, `latency`, `gpu`, `series`, `access`, `alerts` (`dashboard/server.py:730-752`) |
| `/api/health` | `{"ok": true}` — process liveness only, says nothing about the engine |

The dashboard only `Wants=k3.service` (`systemd/k3dash.service:6-8`), so it
stays up and reports the engine as down during an engine incident. That is
deliberate: a `Requires=` would take monitoring offline exactly when it is
needed.

`alerts` is the part worth reading first (`dashboard/server.py:795-852`). It
fires only on conditions an operator would act on:

| Alert | Threshold |
|---|---|
| engine not answering `/metrics` | `server.state != "up"` |
| requests queued for a decode slot | `latest.waiting > 0` |
| KV cache nearly full | `kv_usage >= 0.90` |
| scheduler preempting work | any preemption in the last minute |
| prefix caching disabled | `enable_prefix_caching` false |
| 5xx returned to clients | any, last 15 min |
| 429s | any, last 15 min — a customer is at its cap |
| 401s | >= 20 in 15 min — misconfigured client, or probing |
| cert expiring | < 21 days (`CERT_WARN_DAYS`, `dashboard/server.py:52`) |
| usage log unreadable | attribution is blind |

Quick read of the whole state without the UI:

```bash
curl -s http://127.0.0.1:8080/api/state | jq -c \
  '{state: .server.state, alerts: [.alerts[].text],
    kv: .capacity.kv_usage, hit: .window.cache_hit_rate,
    waiting: .latest.waiting, tpm: .window.total_tpm}'
```

### 3.5 Full security assertions

```bash
make verify        # = bash ./verify-auth.sh   (Makefile:25-26)
```

`verify-auth.sh` now targets the live TLS edge (`EDGE=https://${DOMAIN}`,
`verify-auth.sh:24`) with `--resolve` pinning to loopback
(`verify-auth.sh:33-36`), and it refuses to run if nothing is listening on 443
(`verify-auth.sh:37-39`). It needs a real **customer** key, which it reads from
`customers.tsv` by default (`verify-auth.sh:19-23`), and it asserts that the
**upstream** key is rejected at the edge with 401 (`verify-auth.sh:73-75`) —
presenting it there would mean the edge accepts a credential no customer holds.
`ALL CHECKS PASSED` is the only acceptable output; the exit code is the number
of failed assertions. See [`DEPLOY.md` §5.1](DEPLOY.md) for what each class of
assertion proves.

This suite drives real inference through the edge (`verify-auth.sh:79-84`,
`:118-123`), so it consumes capacity. Do not loop it.

---

## 4. Customer lifecycle

### 4.1 The two-tier key model — read this before touching any key

This is the most confusing part of the system, and getting it wrong causes a
total outage.

| Tier | Value | Who holds it | Enforced by |
|---|---|---|---|
| Customer key | `sk-k3-<name>-<40 hex>` (`gen-keys.sh:49`) | the customer | nginx, via the `$k3_customer` map |
| Upstream key | contents of `api-key.txt` = `VLLM_API_KEY` | nginx only | vLLM's own `--api-key` |

The request path:

1. The customer sends `Authorization: Bearer sk-k3-<name>-...`.
2. nginx maps that exact header string to the customer's name
   (`gen-keys.sh:75-81`). No match means the map yields `""` and the `/v1/`
   location returns a JSON 401 (`issue-cert.sh:76`).
3. The name is the key for the per-customer `limit_conn` / `limit_req` zones
   (`gen-keys.sh:86-88`) and is logged as `cust=<name>` (`gen-keys.sh:91-93`).
4. nginx **replaces** the `Authorization` header with the single upstream
   credential before proxying (`gen-keys.sh:105-110`). vLLM only ever sees one
   key.

Two consequences that drive every procedure below:

- **Adding or revoking a customer needs an nginx reload only — never a vLLM
  restart** (`gen-keys.sh:107-109`). `gen-keys.sh` does the reload itself.
- **Rotating the upstream key requires `api-key.txt` *and* `vllm-k3.env` *and* a
  re-run of `gen-keys.sh` *and* a vLLM restart**, because `k3-limits.inc` bakes
  the upstream key in at generation time (`gen-keys.sh:20`, `:110`) and the
  container reads `VLLM_API_KEY` only at start. Full procedure in
  [`SECRETS.md`](SECRETS.md).

### 4.2 Onboard

```bash
cd /scratch/deploy
./gen-keys.sh add <name>        # mints the key, regenerates nginx, reloads
```

The key is printed **once**, to stdout (`gen-keys.sh:52-53`). Capture it then;
nothing else will show it to you in a form you should be copying around. Give
the customer:

```
base_url: https://cflox.store/v1
model:    FW-Kimi-K3          (the alias kimi-k3 also resolves)
api_key:  <the key just minted>
```

`<name>` becomes the log identity and the rate-limit zone key, so it must be
unique and contain no whitespace (`customers.tsv.example:10-12`). Never
hand-edit `customers.tsv`: the nginx layer is generated from it and will not
rebuild itself.

### 4.3 Revoke

```bash
./gen-keys.sh revoke <name>
```

Deletes the row (`gen-keys.sh:58`) and takes effect on the reload the script
performs at the end (`gen-keys.sh:117-121`). The revoked key returns 401
immediately after the reload.

### 4.4 List

Print names only — never the key column:

```bash
awk -F'\t' '!/^[[:space:]]*#/ && NF {print $1}' /scratch/deploy/customers.tsv
```

Three customers are configured today. To see who is actually *using* the
endpoint, aggregate the usage log (§7.2):

```bash
awk '{for(i=1;i<=NF;i++) if($i ~ /^cust=/) print substr($i,6)}' \
  /var/log/k3/usage.log | sort | uniq -c | sort -rn
```

### 4.5 Rotate a customer key

`gen-keys.sh` has no `rotate` subcommand, and `add` refuses an existing name
(`gen-keys.sh:48`), so rotation is revoke-then-add. Done naively that is an
outage for the customer between the two reloads. Prefer make-before-break:

1. `./gen-keys.sh add <name>-2` — mints the replacement; both keys now work.
2. Deliver the new key; wait for their traffic to move (watch
   `cust=<name>` vs `cust=<name>-2` in the usage log).
3. `./gen-keys.sh revoke <name>` — old key dies.
4. Optionally repeat in the other direction later to get the original name back.

Note that attribution splits across the two names while both are live, and the
two names get **independent** rate-limit budgets during the overlap.

### 4.6 Rebuild the auth layer from scratch

```bash
./gen-keys.sh            # no arguments: regenerate from customers.tsv as-is
```

Use this after restoring `customers.tsv`, or when nginx complains that
`$k3_customer` is undefined. It rewrites both generated fragments, runs
`nginx -t`, and reloads only if the test passes (`gen-keys.sh:117-121`).

---

## 5. Rate limits and 429s

### 5.1 The ceilings

All four live in one block at `gen-keys.sh:38-41`, with the reasoning at
`gen-keys.sh:24-37`. They are sized for a **portal key shared by many end
users**, not for one developer.

| Knob | Value | Why that number (`gen-keys.sh:25-37`) |
|---|---|---|
| `PER_CUST_CONN` | 200 | Bounds a crowd behind one credential. 200 against `max-num-seqs: 512` is ~39% of engine sequence slots for one key. |
| `PER_CUST_RATE` | 100r/s | Must scale with the connection cap or it becomes the real limit: at 10 r/s a portal allowed 200 concurrent calls would be throttled after ten requests in a second. 200 in-flight at ~3 s each turns over ~67/s. |
| `PER_CUST_BURST` | 200 | Absorbs the login-time stampede when many users reconnect at once. |
| `GLOBAL_CONN` | 512 | Raised to `max-num-seqs` so a 200-connection portal cannot starve other keys; at the old 256 it would have owned 78% of the edge. |

These are mutually constrained, not independent knobs. Raising the connection
cap alone does nothing, because the rate cap then becomes the binding limit.

### 5.2 What a throttled client sees

Both limiters are configured to answer 429 rather than nginx's defaults
(`limit_conn_status 429`, `limit_req_status 429`, `gen-keys.sh:102-103`), and
the TLS block turns that into a JSON body with a retry hint
(`issue-cert.sh:93-99`):

```
HTTP/1.1 429
Retry-After: 2
{"error":{"message":"rate limit exceeded for this api key","type":"rate_limit_error","code":429}}
```

### 5.3 Changing a limit

```bash
$EDITOR gen-keys.sh         # lines 38-41 only
./gen-keys.sh               # regenerate + nginx -t + reload
```

No vLLM restart. Re-read `gen-keys.sh:25-37` before changing one number in
isolation, and record what you changed and why — that comment block is the only
place the sizing argument exists.

---

## 6. The correctness gate

**Run it after every launch, and after any change to `config.yaml` or
`vllm-k3.env`.** This is not ceremony: several configurations start cleanly,
serve at full speed, and emit fluent nonsense
(`K3-DEPLOYMENT.md` §6, §7.3). Large-batch benchmarks do not expose that
failure; only correctness probes do.

```bash
make gate       # full, ~21 s          (Makefile:6-7 -> /scratch/gate.sh)
make quick      # tier 1 only, ~5 s    (Makefile:8-9)
bash /scratch/gate.sh --baseline   # re-record, only after an INTENDED change
```

| Tier | What it checks | Floor |
|---|---|---|
| 1 CORRUPTION | 3 factual probes scored by rank + margin, numeric continuation, semantic stability across 3 samples | rank-1 with margin >= 0.8 over the runner-up (`tests/gate.py:52`) |
| 2 REASONING | 16 arithmetic word problems, exact answers, run 8-way parallel | >= 14/16 (`tests/gate.py:53`) |
| 3 LONGCTX | needle retrieval (`74812`) at ~12,026 prompt tokens | exact match |
| regression guard | tier 2 against the recorded baseline | fails at a drop > 2 even when above the floor (`tests/gate.py:56`) |

Notes that matter in practice:

- The gate talks to vLLM **directly** on `127.0.0.1:8001` and reads the upstream
  key from `api-key.txt` (`tests/gate.py:33-44`), so it bypasses nginx entirely,
  consumes no customer's quota, and stays meaningful when the cert or DNS is
  broken but the engine is fine. `K3_BASE_URL`, `K3_MODEL` and
  `K3_API_KEY_FILE` override those defaults.
- **Each copy scores against its own baseline.** `gate.sh` resolves `gate.py`
  next to itself and `gate.py` resolves `gate-baseline.json` next to itself
  (`tests/gate.sh:6-9`, `tests/gate.py:39-42`), so running `bash tests/gate.sh`
  uses the repo copies and `tests/gate-baseline.json`, while `make gate` runs
  `/scratch/gate.sh` (`Makefile:7`) against `/scratch/gate-baseline.json`. The
  two used to drift and silently test the old copy. If you edit
  `tests/gate.py`, install it with `./deploy.sh install` before trusting
  `make gate` (`deploy.sh:532-535`).
- The committed default baseline is `tests/gate-baseline.json`, recorded
  `2026-09-18T12:25:45Z` at `tier2 16/16`, `tier3_prompt_tokens 12026`.
  `deploy.sh` seeds the deployed copy but never overwrites it, because
  `--baseline` is operator state (`deploy.sh:536-541`).
- It fails **closed**: an unreachable server exits 1 rather than silently
  passing (`K3-DEPLOYMENT.md` §6).

**A gate failure means stop serving traffic you care about, not "retry".** The
output says so directly: `GATE FAIL ... DO NOT BENCHMARK OR SERVE`
(`tests/gate.py:305`). Tier 1 failing with fluent-looking output is the AITER
pairing (§9.6). Tier 3 failing alone points at long-context prefill (vLLM
#51039, NaN logits after long prefill — `tests/gate.py:14-16`). Tier 2 dropping
a couple of problems while tier 1 passes is a genuine regression signal, which
is what the tolerance of 2 exists to catch.

Re-record the baseline **only** when you have deliberately changed the
configuration and the new numbers are ones you are willing to defend. Doing it
to make a red gate go green destroys the only degradation signal there is.

`TODO(operator):` `tests/gate.py:216-220` still sizes tier 3 against
`max_model_len=12800`. The prompt is ~12,026 tokens and passes comfortably
under the live 262,144 window, so the test is valid, but the comment is stale
and understates the headroom.

---

## 7. Logs

### 7.1 Engine

```bash
make logs                       # docker logs -f --tail 50 k3   (Makefile:16-17)
docker logs k3 2>&1 | tail -200
journalctl -u k3.service -n 100 --no-pager      # unit-level: starts, stops, backoff
```

The container's log driver is capped at `--log-opt max-size=100m
--log-opt max-file=5` (`k3.service:33`), so at most ~500 MB is retained. Before
that cap was added the log grew unbounded (`K3-DEPLOYMENT.md`, "Dozzle" section).

**`docker logs k3` only covers the current container.** The unit runs with
`--rm` (`k3.service:29`) and `docker rm -f k3` before each start
(`k3.service:28`), so a restart destroys the previous container's output. If you
are investigating a crash, capture `docker logs k3` **before** restarting
anything; afterwards, `journalctl -u k3.service` is all that is left.

### 7.2 Per-customer usage — the billing record

`/var/log/k3/usage.log`, format `k3usage`, defined at
`gen-keys.sh:91-93` and written only from the `/v1/` location
(`issue-cert.sh:79`), so it is a clean attribution record with no dashboard or
health-probe noise in it.

| Field | nginx variable | Meaning |
|---|---|---|
| (first token) | `$time_iso8601` | request start time, ISO 8601 |
| `cust=` | `$k3_customer` | customer name from the map; empty means the request was rejected before auth |
| `status=` | `$status` | final HTTP status returned to the client (401, 429, 200, ...) |
| `path=` | `$request_uri` | full request URI including query |
| `req_ms=` | `$request_time` | **seconds** (nginx's unit) from first byte read to last byte written — end-to-end, includes the client |
| `up_ms=` | `$upstream_response_time` | **seconds** spent in vLLM. Compare against `req_ms` to separate engine slowness from client slowness |
| `in=` | `$request_length` | request bytes, including headers |
| `out=` | `$body_bytes_sent` | response body bytes |
| `ip=` | `$remote_addr` | client address |

The `req_ms` / `up_ms` names say milliseconds but nginx emits seconds with
millisecond resolution. Read them as seconds.

The dashboard tails this same file (`systemd/k3dash.service:26`) and its
per-customer view and 401/429/5xx alerts come from it; if it is unreadable the
dashboard raises "attribution is unavailable" (`dashboard/server.py:850-852`).

```bash
tail -f /var/log/k3/usage.log
# slowest upstream times, last 1000 requests
tail -1000 /var/log/k3/usage.log | awk '{for(i=1;i<=NF;i++) if($i~/^up_ms=/) print $i, $0}' | sort -rn -t= -k2 | head
```

**Retention — resolved 2026-09-18.** This log previously sat at
`/var/log/k3/usage.log`, where the nginx package's
`/etc/logrotate.d/nginx` glob claimed it with consequences nobody chose: `rotate
14 daily` **deleted the billing record after 14 days**, and `create 0640
www-data adm` widened it from `600 root:root` on every rotation.

A drop-in override is not possible — logrotate rejects a second entry for a path
it already manages (`duplicate log entry`) and has no exclude syntax — so the
log was moved out of that glob to `/var/log/k3/usage.log` and given its own
policy in `nginx/k3-usage.logrotate` (installed to `/etc/logrotate.d/k3-usage`):
`rotate 365 daily`, compressed, `create 0640 root adm`. The worker never reads
this file, only writes it through a descriptor the master opens as root, so
`www-data` needs no access.

The 1,878 pre-migration entries were preserved as
`/var/log/k3/usage.log.pre-migration-20260918` and fall under the new retention.
Three places reference this path and must stay in agreement: the `access_log`
directive (`issue-cert.sh:79`, regenerated into `k3-tls.conf`), the dashboard's
`ACCESS_LOG` (`systemd/k3dash.service:26`), and the logrotate rule. Changing one
alone silently splits or strands the billing record.

`TODO(operator):` nothing ships these logs off-box, so a year of billing data
lives only on an **ephemeral** disk (`ARCHITECTURE.md` §9). Decide whether that
is acceptable or whether rotated files should be shipped to durable storage.

### 7.3 Edge and dashboard

```bash
tail -f /var/log/nginx/error.log         # TLS, upstream, limit_req/limit_conn events
journalctl -fu k3dash.service            # dashboard
journalctl -fu nginx.service
```

`limit_req` and `limit_conn` rejections are logged to `error.log`, which is how
you distinguish "nginx threw the 429" from "the customer's own client did".

---

## 8. Certificates

```bash
openssl x509 -enddate -noout -in /etc/letsencrypt/live/cflox.store/fullchain.pem
systemctl list-timers certbot.timer --no-pager
systemctl is-enabled certbot.timer
```

Current state (2026-09-18): expires **Dec 17 13:56:54 2026 GMT**, issuer Let's
Encrypt, `certbot.timer` enabled and next elapsing within 12 h. The dashboard
surfaces `cert_days_left` and alerts below 21 days
(`dashboard/server.py:52`, `:844-848`).

### 8.1 Renewal

Renewal is `certbot.timer`'s job: it runs `certbot -q renew` twice daily
(`certbot.service`), and certbot renews when the certificate is within 30 days
of expiry. Renewal uses the `webroot` authenticator against
`/var/www/certbot` (`/etc/letsencrypt/renewal/cflox.store.conf`), which the TLS
block serves at `/.well-known/acme-challenge/` (`issue-cert.sh:48`). No
downtime, no operator action in the normal case.

**Gap — nginx is not reloaded after renewal.** `/etc/letsencrypt/renewal-hooks/deploy/`
is empty and the renewal config uses no installer, so certbot writes new files
into `/etc/letsencrypt/archive/` and repoints the `live/` symlinks while nginx
keeps serving the certificate it loaded at startup. The endpoint will serve an
expired certificate until something reloads nginx.

`TODO(operator):` add a deploy hook that reloads nginx (a one-line script in
`/etc/letsencrypt/renewal-hooks/deploy/`), or put a `--deploy-hook` on the
renewal config. Until then, after any renewal:

```bash
nginx -t && systemctl reload nginx
openssl s_client -connect 127.0.0.1:443 -servername cflox.store </dev/null 2>/dev/null \
  | openssl x509 -noout -enddate     # what is actually being SERVED
```

Compare that against the `enddate` of the file on disk. A mismatch is the
symptom of this gap.

### 8.2 Re-issuing

```bash
./issue-cert.sh            # issue once, only if DNS already resolves here
./issue-cert.sh --watch    # poll DNS every 60 s for up to 4 h, then issue
```

`issue-cert.sh` is safe to re-run and does nothing unless a **public** resolver
returns this host's own detected public IP for `cflox.store` (`issue-cert.sh:11-13`,
`:20-34`). That gate exists because **a failed HTTP-01 burns Let's Encrypt rate
limit: 5 failures per hostname per hour** (`issue-cert.sh:4-5`). Do not add a
`--force` path, and do not retry a failure in a loop — you will lock yourself
out of issuance for an hour on a domain that is serving customers.

The script asks for `cflox.store` + `www.cflox.store` first and falls back to
the bare domain if the www name fails HTTP-01, dropping `www` from
`server_name` so nginx does not claim a name it cannot serve
(`issue-cert.sh:115-128`). On success it rewrites
`/etc/nginx/conf.d/k3-tls.conf`, backs the old one up to `.pre-tls.bak`, and
rolls back automatically if `nginx -t` fails (`issue-cert.sh:133-138`). It
finishes with five spot checks against the live edge (`issue-cert.sh:141-153`).

Re-issuing regenerates `k3-tls.conf` from `write_tls_server()`, which
**discards any hand-added comments** in the live file
(`ARCHITECTURE.md` §11 item 1). Edit `issue-cert.sh`, not the generated file.

---

## 9. Incident playbooks

Each one: confirm, cause, fix, verify. Run the confirm step before the fix —
several of these have symptoms that look identical from the client side.

### 9.1 Engine dead or crash-looping

1. **Confirm.**
   ```bash
   systemctl status k3.service
   curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8001/health   # want 200
   docker ps --filter name=k3
   journalctl -u k3.service -n 50 --no-pager
   ```
   `activating (auto-restart)` or repeated `Scheduled restart job` lines mean a
   crash loop. `failed` with `start request repeated too quickly` means the unit
   has given up — go to 9.2.
2. **Capture before you restart.** `docker logs k3 > /scratch/results/k3-crash-$(date +%s).log 2>&1`.
   The container is `--rm`, so this output is gone after the next start (§7.1).
3. **Likely causes, in the order worth checking.**
   - Long-prompt `EngineDeadError` from VRAM headroom, not KV exhaustion. The
     evidence is in the kernel log, not Python: `dmesg -T | grep -i 'svm_range_evict\|amdgpu'`
     showing `svm_range_evict_svm_bo_worker` at the second of the crash
     (`config.yaml:44-47`, `DEPLOY.md` §8.3). A worker dying with no Python
     traceback and only `RuntimeError: cancelled` from `shm_broadcast.py` is
     this signature.
   - Weights missing or `/scratch` unmounted: `HF_HUB_OFFLINE=1`
     (`vllm-k3.env.example:17`) makes a missing weight tree a hard failure, not
     a silent re-download. Check `ls -d /scratch/hf/hub/models--moonshotai--Kimi-K3`.
   - Config change that does not parse, or an image that is not the pinned
     digest (`launch.sh:30-36`).
4. **Fix.** Address the cause, then `systemctl restart k3.service`. If you
   changed `max-model-len` or `gpu-memory-utilization`, remember the engine
   reads `/scratch/hf/config.yaml`, not the repo copy (`k3.service:38`,
   `ARCHITECTURE.md` §9).
5. **Verify.** Wait for `/health` to return 200 (up to ~6 min), confirm the
   expected `GPU KV cache size` line (§3.2), then `make gate`.

### 9.2 Unit gave up — `StartLimitBurst` exhausted

1. **Confirm.** `systemctl status k3.service` shows `failed` and the journal
   says `start request repeated too quickly` / `Start request repeated too
   quickly`. This is 5 failed starts inside 30 min (`k3.service:12-13`).
2. **Cause.** By design: the unit stops retrying rather than looping forever.
   The underlying failure is still there and is in the journal from before the
   limit was hit.
3. **Fix.** Diagnose first (9.1 step 3), then clear the latch:
   ```bash
   journalctl -u k3.service --since '-40min' --no-pager | tail -100
   systemctl reset-failed k3.service
   systemctl start k3.service
   ```
   `reset-failed` clears the rate-limit counter; without it `start` is refused.
4. **Verify.** As 9.1 step 5. If it fails a sixth time, stop restarting and fix
   the cause — you are burning ~6 min per attempt.

### 9.3 Every customer gets 401

1. **Confirm the layer.** The two failures look identical to clients but differ
   on the box:
   ```bash
   curl -s -o /dev/null -w 'edge nokey %{http_code}\n' --resolve cflox.store:443:127.0.0.1 https://cflox.store/v1/models
   grep -c '^    "Bearer ' /etc/nginx/conf.d/00-k3-keys.conf    # customer keys in the map
   curl -s -o /dev/null -w 'engine %{http_code}\n' http://127.0.0.1:8001/health
   ```
   Then look at the usage log: a 401 with a **non-empty** `cust=` means nginx
   authenticated the customer and vLLM rejected the swapped upstream key. A 401
   with an **empty** `cust=` means nginx rejected it.
2. **Cause A — empty `cust=`: the map is gone or empty.** `00-k3-keys.conf`
   missing, or `customers.tsv` emptied. `deploy.sh:660-661` warns about exactly
   this: no keys in the map means every `/v1/` request 401s.
   **Fix:** `./gen-keys.sh` (rebuilds from `customers.tsv` and reloads).
3. **Cause B — populated `cust=`: `VLLM_API_KEY` != `api-key.txt`.** nginx is
   forwarding a credential vLLM rejects, so the 401 happens *after* passing the
   edge (`DEPLOY.md` §8.7, `vllm-k3.env.example:19-23`). Confirm without
   printing either value:
   ```bash
   cmp -s <(sed -n 's/^VLLM_API_KEY=//p' /scratch/deploy/vllm-k3.env | tail -1) \
          <(tr -d '\r\n' < /scratch/deploy/api-key.txt) && echo match || echo MISMATCH
   ```
   **Fix:** the full upstream-key procedure in [`SECRETS.md`](SECRETS.md) —
   reconcile the two files, re-run `./gen-keys.sh` (it bakes the key into
   `k3-limits.inc` at generation time, `gen-keys.sh:20`, `:110`), then
   `systemctl restart k3.service`. Nothing short of the restart takes effect;
   the container reads its environment only at start.
4. **Verify.** `make verify` → `ALL CHECKS PASSED`, and 401s stop appearing in
   the usage log.

### 9.4 401 for exactly one customer

1. **Confirm.** Their name in the map, and their recent log lines:
   ```bash
   grep -c "\"${NAME}\";" /etc/nginx/conf.d/00-k3-keys.conf     # 1 = present
   grep "cust=${NAME} " /var/log/k3/usage.log | tail
   grep 'cust= ' /var/log/k3/usage.log | tail            # rejected-at-edge lines
   ```
2. **Likely causes.** They were revoked; they are sending an old key after a
   rotation; their client mangles the header (a stray `Bearer Bearer`, trailing
   whitespace, or a key with a line break pasted in). The map matches the
   **entire** header string exactly (`gen-keys.sh:80`), so any of these misses.
   Also check that `gen-keys.sh` actually reloaded after their key was added —
   an invalid config aborts the reload (`gen-keys.sh:117-121`) and leaves the
   previous map serving.
3. **Fix.** If they should exist and do not: `./gen-keys.sh add <name>` and
   deliver the new key. If the map looks right, the problem is on their side —
   have them echo the exact bytes they send.
4. **Verify.** Their next request logs `cust=<name> status=200`.

### 9.5 429 storms

1. **Confirm who and how much.**
   ```bash
   awk '$0 ~ /status=429/ {for(i=1;i<=NF;i++) if($i~/^cust=/) print $i}' \
     /var/log/k3/usage.log | sort | uniq -c | sort -rn
   tail -f /var/log/nginx/error.log      # limiting requests / limiting connections
   ```
   The error log distinguishes `limit_req` (rate) from `limit_conn`
   (concurrency), which tells you which ceiling was hit.
2. **Cause.** A single key exceeded `PER_CUST_CONN=200` or `PER_CUST_RATE=100r/s`
   with `PER_CUST_BURST=200`; or the aggregate hit `GLOBAL_CONN=512`
   (`gen-keys.sh:38-41`). A global-limit storm hits everyone at once and is the
   more serious case — check whether one key is consuming the shared budget.
3. **Fix.** First decide whether the limit is doing its job. If a legitimate
   portal has outgrown its cap, raise `PER_CUST_CONN` **and** `PER_CUST_RATE`
   **and** `PER_CUST_BURST` together (§5.3) — raising one alone does nothing
   because the other becomes the binding limit (`gen-keys.sh:30-33`). If the
   engine is the real constraint, raising limits converts 429s into queueing
   and higher latency for everyone; check §10 first.
4. **Verify.** 429 count in the usage log falls; `latest.waiting` on the
   dashboard stays near 0.

### 9.6 Fluent but wrong output — the AITER pairing

This is the most dangerous failure here because nothing looks broken: the
engine starts cleanly, serves at full speed, and returns confident nonsense.

1. **Confirm.** `make quick` (tier 1, ~5 s). The signature is the correct token
   losing its lead in the factual probes, and/or broken numeric continuation.
   A healthy result looks like `logprob(Paris) = -0.242`, counting intact
   (`K3-DEPLOYMENT.md` §7.3).
2. **Cause.** `AITER_SITUV2_A8W4=1` forces an interleaved weight layout *and*
   selects activation dtype by batch size. Below the bf16/fp8 boundary it feeds
   bf16 activations to interleaved A8W4 weights — a silent layout mismatch.
   `AITER_BF16_FP8_MOE_BOUND=0` pins fp8 activations at every batch size and is
   what makes the pairing safe (`vllm-k3.env.example:7-13`).
3. **Fix.** Check both flags are present and paired in `vllm-k3.env`:
   ```bash
   grep -E 'AITER_SITUV2_A8W4|AITER_BF16_FP8_MOE_BOUND' /scratch/deploy/vllm-k3.env
   ```
   Restore from the committed template if they drifted — `./deploy.sh secrets`
   rebuilds `vllm-k3.env` from `vllm-k3.env.example` and re-substitutes
   `VLLM_API_KEY` (`deploy.sh:422-451`) rather than hand-editing. Then
   `systemctl restart k3.service`.
4. **Verify.** `make gate` — full gate, not `--quick`. Large-batch benchmarks do
   **not** expose this failure; only the correctness probes do.

### 9.7 Slow or degraded throughput

1. **Confirm it is the server and not the offered load.**
   ```bash
   curl -s http://127.0.0.1:8080/api/state | jq -c \
     '{waiting: .latest.waiting, running: .latest.running, kv: .capacity.kv_usage,
       preempt: .capacity.window_preempted, hit: .window.cache_hit_rate, gpu: .gpu.mean_use}'
   make cache
   ```
   Low throughput with an empty queue means nobody is asking — the dashboard
   deliberately does not alert on that (`dashboard/server.py:796-800`).
2. **Read the numbers against these expectations** (`K3-DEPLOYMENT.md` §9):

   | Signal | Healthy | If wrong |
   |---|---|---|
   | `num_requests_waiting` | ~0 | sustained >50 means offered load exceeds capacity |
   | KV utilisation | <70% | approaching 100% causes preemption and collapse |
   | prefix cache hit rate | ~81% | a drop means traffic lost its shared prefix; capacity falls ~3x |
   | GPU utilisation | 85-100% | low with a full queue implies a stall, not saturation |

3. **Cause and fix.**
   - **Hit rate collapsed** — a client changed its system prompt or tool
     schemas, or something in the path rewrites requests. Anything that
     perturbs the token prefix destroys the shared-prefix economics
     (`ARCHITECTURE.md` §8). Find the customer via the usage log and compare
     their request shape.
   - **Preemptions >0** — the scheduler is discarding work under KV pressure.
     Alert on this rather than on KV usage; vLLM is designed to run KV near full
     (`K3-DEPLOYMENT.md` §11.5).
   - **`up_ms` high but `req_ms` far higher** — the client, not the engine.
   - **Streams truncating at 60 s** — the `/v1/` location being served is not
     the generated one; the real block sets `proxy_read_timeout 900s` with
     `proxy_buffering off` (`issue-cert.sh:87-91`) because a 12k-in/512-out
     request runs ~112 s.
4. **Verify.** Throughput and `waiting` return to the table above; `make gate`
   still passes (a "slow" engine that is also wrong is 9.6).

### 9.8 GPU problems (card missing, fallen off the bus)

1. **Confirm.**
   ```bash
   ls -l /dev/kfd /dev/dri
   rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l   # want 8
   dmesg -T | grep -i 'amdgpu\|kfd' | tail -40
   ```
   `deploy.sh:235-248` runs exactly these checks: `/dev/kfd` is the compute
   device the container is handed (`k3.service:30`), and fewer than 8 distinct
   GPU indices breaks `tensor-parallel-size: 8` (`config.yaml:19`).
2. **Cause.** If `/dev/kfd` is absent or the count is below 8, vLLM cannot
   start and no amount of restarting helps. Kernel-level amdgpu faults appear
   in `dmesg`, which is also where the one recorded engine death left its
   evidence (`svm_range_evict_svm_bo_worker`, §9.1).
3. **Fix.** This is host/hypervisor territory, not application territory. Stop
   the restart loop so the box is not thrashing:
   `systemctl stop k3.service`, then escalate to the provider with the `dmesg`
   excerpt. A reboot may restore the cards; a reclaimed or faulty host means
   §9.10.
4. **Verify.** All 8 indices visible in `rocm-smi`, then start the unit and run
   `make gate` — a card that came back in a bad state can produce wrong output,
   not just slow output.

`TODO(operator):` no "fell off the bus" event has actually occurred on this box,
so the specific `dmesg` string to alert on is unverified here. Record it if it
ever happens.

### 9.9 Dashboard down

Non-urgent: the dashboard is a monitor. Serving is unaffected.

1. **Confirm.**
   ```bash
   systemctl status k3dash.service
   curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/api/health   # want 200
   curl -s -o /dev/null -w '%{http_code}\n' --resolve cflox.store:443:127.0.0.1 https://cflox.store/   # want 401 unauthenticated
   ```
   A 502 at the edge with a dead unit is the unit; a 401 at the edge with a
   healthy `:8080` is basic auth (expected without credentials).
2. **Causes.** The unit runs `ProtectSystem=strict` and friends
   (`systemd/k3dash.service:32-35`), so anything that tries to write outside
   its allowances fails; `journalctl -u k3dash.service` has the traceback. The
   usage log being unreadable degrades attribution but does not stop the
   process (`dashboard/server.py:850-852`).
3. **Fix.** `systemctl restart k3dash.service` (seconds, no engine impact). If
   the files were edited in the repo, they must be installed to `/opt/k3dash`
   first — `./deploy.sh install` (`deploy.sh:516-521`).
4. **Verify.** `/api/health` 200, and `/api/state` reports
   `server.state == "up"` with `capacity.capacity_tokens > 0` — the two
   assertions `verify-auth.sh:105-112` makes.

### 9.10 Disk pressure

1. **Confirm.** `df -h /scratch /` and `du -sh /scratch/results /scratch/backup`.
   Today: `/scratch` is 40 T with 1.5 T used (4%), `/` is 2 T at 5%.
2. **Cause.** The large consumers are the weights (~1.5 TB), the image tarball
   `/scratch/backup/k3-image.tar` (`launch.sh:13`), benchmark output in
   `/scratch/results`, and the capped container logs (~500 MB max,
   `k3.service:33`). `deploy.sh` requires 1,600 GiB free before a weight
   download so a pull cannot wedge the filesystem (`deploy.sh:70-72`,
   `:272-274`).
3. **Fix.** Reclaim from `/scratch/results` first — it is regenerable benchmark
   output. Do **not** delete `/scratch/backup/k3-image.tar` casually: it is what
   makes a rebuild independent of the upstream `kimi-k3` tag, which is a one-off
   dev tag that may be garbage-collected (`launch.sh:10-12`). Never touch
   `/scratch/hf`.
4. **Verify.** `df -h /scratch` and a successful `make gate`.

### 9.11 Box reclaimed — full rebuild

`/scratch` is ephemeral and is destroyed on reclaim: weights, image tarball,
the repo, and every secret file go with it (`ARCHITECTURE.md` §9).

1. **Before you rebuild**, if the old box is still reachable, extract
   `customers.tsv` and `/var/log/k3/usage.log` (`DEPLOY.md` §7.5).
   Carrying `customers.tsv` over is what keeps existing customer keys working;
   without it every customer must be re-issued.
2. **Rebuild** with [`DEPLOY.md`](DEPLOY.md) — that document owns this
   procedure end to end (droplet creation, `cloud-init.yaml`, `./deploy.sh`).
3. **Credentials**: what to regenerate versus what to carry over is in
   [`SECRETS.md`](SECRETS.md). Note `config.yaml:10-15` keeps the `kimi-k3`
   served-model alias precisely so live integrations survive a rebuild.
4. **Verify.** `make verify` and `make gate`, then confirm a real customer key
   works through the public name from **off-box** — testing from the box proves
   nothing, because loopback-routed traffic bypasses the firewall path
   (`DEPLOY.md` §8.9).

---

## 10. Capacity

The single tradeoff to understand: `max-model-len` does **not** reserve memory.
KV pages are allocated on demand, so a 20k-token request costs 20k of KV
whether the cap is 12.8k or 1M. What `max-model-len` bounds is the **worst
case** — how much of the pool one request may monopolise (`config.yaml:37-39`).

Against the measured KV pool of **2,295,266 tokens** at
`gpu-memory-utilization: 0.92` (`config.yaml:39`, confirmed live in §3.2):

| `max-model-len` | Worst-case concurrency |
|---|---|
| 128k | 17.5x |
| **256k (262,144, current)** | **8.76x** |
| 384k | 5.8x |
| 512k | 4.4x |

`config.yaml:40`. The engine prints the current row itself as
`Maximum concurrency for 262,144 tokens per request: 8.76x`.

That ladder is the floor, not the expectation. Real concurrency is far higher
because requests are nowhere near the cap and share a prefix: the production
run held **427 concurrent** at ~12.8k-token prompts with KV peaking at 49.4%
(`K3-DEPLOYMENT.md` §1, §2.1). The ceiling on sequence slots is
`max-num-seqs: 512` (`config.yaml:63`), and `max-num-batched-tokens: 8192`
keeps chunked prefill from raising the activation peak — a 256k prompt is
processed in 32 chunks of 8,192 (`config.yaml:64-67`).

**Why 256k and not less** is a client-compatibility argument, not a headroom
one (`config.yaml:29-35`): clients send a *fixed* `max_tokens` (Foundry-style
SDKs default to 128,000) and vLLM reserves prompt + `max_tokens` against one
shared window, so usable prompt = `max-model-len` − `max_tokens`. At 131,072
that left 3,072 prompt tokens and 400'd nearly every agentic request. 256k
leaves 134,144.

**Why not more.** 256k is verified at 0.92 (~16.6 GiB/GPU spare) by
concurrency tests with randomised prompts on 2026-09-18 (`config.yaml:49-56`):
single prefill of 249,696 tokens, 4 x 212k concurrent, 6 x 191k concurrent, all
with zero evictions, and a `>262,144` request returning a clean 400 with the
engine surviving. The same shape at `gpu-memory-utilization: 0.96` killed the
engine mid-prefill (`config.yaml:42-47`). Raising either value means repeating
that test **under concurrency with unique prompts** — identical prompts finish
in seconds off the prefix cache and prove nothing — while watching `dmesg` for
`svm_range_evict_svm_bo_worker` (`config.yaml:58-61`).

Note that [`DEPLOY.md` §8.1](DEPLOY.md) still describes 262144-vs-131072 as an
open discrepancy and cites the pre-rewrite line numbers; `config.yaml` has since
been rewritten with the measurements above, and
[`ARCHITECTURE.md` §10](ARCHITECTURE.md) records it as settled. Treat
`config.yaml`'s own comments as authoritative.

For capacity headroom that does **not** cost VRAM: the prefix cache is worth
~3x and is load-bearing (`config.yaml:87-89`). Protect it before considering
any engine-shape change.
