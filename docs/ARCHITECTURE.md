# Architecture — Kimi-K3 on 8x MI355X

**Superseded for the live request path.** `docs/SYSTEM-DESIGN.md` §4 is the
production stack: nginx (Bearer pass-through) → LiteLLM (tenancy) →
backpressure gateway → engine. This file describes the earlier nginx
allowlist + Authorization-swap edge. Do not reinstall that path.

Scope: this file explains **structure and rationale**. It does not repeat
measurements, tuning history, or failure forensics — those are in
[`K3-DEPLOYMENT.md`](K3-DEPLOYMENT.md) (56 KB, authoritative). Operational
procedure lives in `docs/RUNBOOK.md`, first-time bring-up in `docs/DEPLOY.md`,
credential handling in `docs/SECRETS.md`.

Every claim below cites the file it came from. Where something could not be
verified from a file on this box it is marked `TODO(operator):`.

---

## 1. Components

| Component | Listener | Owned by | Config |
|---|---|---|---|
| vLLM (Kimi-K3) | `127.0.0.1:8001` | `k3.service` -> docker container `k3` | `/hf/config.yaml` + `vllm-k3.env` |
| nginx 1.24.0 | `0.0.0.0:80`, `0.0.0.0:443` (+ v6) | `nginx.service` | `/etc/nginx/conf.d/*.conf` |
| dashboard | `127.0.0.1:8080` | `k3dash.service` -> `/opt/k3dash/server.py` | env from the unit |

Verified listeners (`ss -tln`): `0.0.0.0:80`, `0.0.0.0:443`, `[::]:80`,
`[::]:443`, `127.0.0.1:8001`, `127.0.0.1:8080`. Nothing else is bound.
nginx version confirmed with `nginx -v`: `nginx/1.24.0 (Ubuntu)` — this version
number is load-bearing, see §2.1.

The engine runs from a digest-pinned image,
`vllm/vllm-openai-rocm@sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b`
(`k3.service:37`, same digest in `Makefile:2`). A tag would let the underlying
build move; the digest is why "same image" is checkable. The container runs
`--network host` (`k3.service:31`), so vLLM's own `host: 127.0.0.1`
(`config.yaml:106`) is what keeps it off the public interface — there is no
docker NAT layer adding isolation here.

---

## 2. Request path

### 2.1 Inference

```
  customer (TLS client)
        |
        |  https://cflox.store/v1/...   Authorization: Bearer sk-k3-<name>-<40 hex>
        v
+---------------------------------------------------------------+
| nginx :443   server_name cflox.store [www.cflox.store]        |
|                                                               |
|  [1] TLS terminate        ssl_certificate /etc/letsencrypt/.. |
|  [2] server_name match    unmatched -> 00-default-deny.conf   |
|  [3] path allowlist       /v1/ ok; /metrics,/health local;    |
|                           dashboard via k3-dash.inc;          |
|                           everything else -> 403              |
|  [4] identity lookup      map $http_authorization             |
|                             -> $k3_customer   (""  -> 401)    |
|  [5] quota                limit_conn k3_percust 200           |
|                           limit_conn k3_global  512           |
|                           limit_req  k3_rps 100r/s burst 200  |
|                             exceeded -> 429 + Retry-After: 2  |
|  [6] attribution          access_log k3-usage.log k3usage     |
|  [7] CREDENTIAL SWAP      proxy_set_header Authorization      |
|                             "Bearer <upstream key>"           |
|  [8] stream settings      proxy_buffering off,                |
|                           proxy_read/send_timeout 900s        |
+---------------------------------------------------------------+
        |
        |  HTTP/1.1, Connection: "", keepalive pool of 32
        v
  upstream k3_backend = 127.0.0.1:8001
        |
        v
  vLLM  --api-key (VLLM_API_KEY) checked on /v1 -> engine -> SSE/JSON response
        |
        v
  streamed back through nginx unbuffered
```

Steps in prose, with sources:

1. **TLS terminates at nginx.** Cert and key from
   `/etc/letsencrypt/live/<domain>/`, `TLSv1.2 TLSv1.3`, HSTS
   `max-age=31536000` (`issue-cert.sh:58-64`). Port 80 exists only for the ACME
   webroot and a 301 to HTTPS (`issue-cert.sh:47-48`). HTTP/2 is enabled in the
   **combined** form, `listen 443 ssl http2`, because nginx 1.24 rejects the
   standalone `http2 on;` directive added in 1.25.1 (`issue-cert.sh:53-55`) —
   this is why the version in §1 matters.
2. **`server_name` selection.** A request that matches no name is handled by
   `nginx/00-default-deny.conf`, not by the cflox.store block — see §5.
3. **Path allowlist, not blocklist** — see §4.
4. **Identity.** `map $http_authorization $k3_customer` resolves the whole
   `Authorization` header to a customer name; `default ""` means unknown
   (`gen-keys.sh:74-82`). The `/v1/` location refuses an empty result with a
   JSON 401 (`issue-cert.sh:76`).
5. **Quotas** come from `k3-limits.inc`, included *inside* the `/v1/` location
   (`issue-cert.sh:78`, `gen-keys.sh:97-103`). Both limit statuses are
   overridden to 429, and `error_page 429 = @throttled` returns a JSON body with
   `Retry-After: 2` (`issue-cert.sh:93-99`).
6. **Attribution.** `access_log /var/log/k3/usage.log k3usage`
   (`issue-cert.sh:79`). This log is the only place per-customer identity
   exists; the dashboard reads it (§2.2).
7. **Credential swap** — see §3. This is the last thing that happens before
   `proxy_pass`, and it overwrites the customer's header.
8. **Streaming.** `proxy_buffering off`, `proxy_cache off`,
   `proxy_request_buffering off`, `proxy_read_timeout`/`proxy_send_timeout`
   `900s` (`issue-cert.sh:87-91`). The generator's comment gives the reason: a
   12k-in/512-out run takes ~112 s, so nginx's 60 s default would cut streams
   off mid-generation (`issue-cert.sh:86`; measured `median_e2el_ms
   111,689.24` in `K3-DEPLOYMENT.md` §1). With buffering on, SSE would batch
   instead of streaming (`K3-DEPLOYMENT.md` §11.3).
   `client_max_body_size 256m` bounds request bodies, with `error_page 413`
   returning a JSON hint to pass `image_url` rather than base64
   (`issue-cert.sh:66`, `issue-cert.sh:101-105`).

Upstream keepalive is 32 connections with `proxy_http_version 1.1` and
`Connection: ""` (`nginx/k3.conf:3`, `issue-cert.sh:82-83`) — the empty
`Connection` header is required or the keepalive pool is not used.

### 2.2 Dashboard

```
  operator browser
        |  https://cflox.store/  (or /api/state)
        v
  nginx :443  --> location = /   and  location /api/   (k3-dash.inc)
        |         auth_basic "k3 dashboard"
        |         auth_basic_user_file /etc/nginx/k3-dash.htpasswd
        v
  upstream k3_dash = 127.0.0.1:8080   (keepalive 8)
        |
  /opt/k3dash/server.py  (no authentication of its own)
        |
        +--> GET http://127.0.0.1:8001/metrics   every 2.0s   (engine state)
        +--> tail /var/log/k3/usage.log    (per-customer attribution)
        +--> rocm-smi --showuse / --showmemuse   every 5.0s   (per-card)
        +--> openssl x509 on the live cert       every 60s    (expiry)
```

The dashboard serves exactly three paths — `/` (and `/index.html`),
`/api/state`, `/api/health` — and returns JSON 404 for anything else
(`dashboard/server.py:921-935`). It is read-only by design: "it never sends
inference traffic, so it cannot perturb what it measures"
(`dashboard/server.py:12-13`).

Its three data sources exist because no single one answers "is the service
healthy" (`dashboard/server.py:4-10`): vLLM's Prometheus endpoint for
engine-side throughput/queue/KV, the nginx `k3usage` log because **vLLM sees
one upstream credential for everybody** and cannot attribute anything, and
`rocm-smi` for per-card utilisation and VRAM
(`dashboard/server.py:566-568`). Sampling cadence:
`SCRAPE_INTERVAL 2.0`, `GPU_INTERVAL 5.0`, `FACTS_INTERVAL 60.0`,
`ACCESS_WINDOW_S 900.0` (15 min of attribution), `CERT_WARN_DAYS 21` because
certbot renews at 30 (`dashboard/server.py:40-52`).

Every path and URL it uses is injected by the unit rather than defaulted:
`DASH_ADDR`, `DASH_PORT`, `VLLM_BASE_URL`, `PUBLIC_URL`, `ACCESS_LOG`,
`CERT_PATH`, `API_KEY_FILE` (`systemd/k3dash.service:23-29`), which matches
`server.py`'s defaults but makes the unit the source of truth
(`systemd/k3dash.service:18`). `API_KEY_FILE` is optional and used only to read
`/v1/models` for the served name and context limit
(`dashboard/server.py:33-35`).

Location precedence note: `k3-dash.inc` is `include`d *before* the catch-all
`location /` (`issue-cert.sh:107-109`), but that ordering is cosmetic —
`location = /` is an exact match and `location /api/` is a longer prefix, so
both win over `location /` regardless. The dashboard is mounted at `/`, not
`/dash/`, because `index.html` fetches the absolute path `/api/state` and a
prefix would break it (`nginx/k3-dash.inc:1-2`).

---

## 3. The two-tier credential model

There are two distinct credentials in the request path:

| Tier | Secret | Held by | Enforced by | Rotation cost |
|---|---|---|---|---|
| Customer | `sk-k3-<name>-<40 hex>`, one per customer | the customer | nginx `$k3_customer` map | `gen-keys.sh` + `nginx -s reload` |
| Upstream | `VLLM_API_KEY` | nginx only | vLLM `--api-key` | vLLM restart, ~6 min |

Credential locations are listed in `docs/SECRETS.md` and enumerated in
`.gitignore`; no key value appears in this file.

nginx resolves the customer key to an identity, applies that identity's quotas,
logs it, and then **replaces** the `Authorization` header with the single
upstream credential before proxying (`gen-keys.sh:105-110`). The generator's
comment records why the swap is there at all: without it "a customer key passes
nginx and then vLLM returns its own 401 — which is exactly what happened on
first setup."

Why this shape:

- **Revocation without a restart.** Because the swap happens at the edge,
  "adding or revoking a customer never requires a vLLM restart"
  (`gen-keys.sh:109`). `gen-keys.sh revoke <name>` deletes the row and reloads
  nginx (`gen-keys.sh:55-59`, `gen-keys.sh:117-118`). A restart would cost ~6
  min of weight loading (`K3-DEPLOYMENT.md` §3, `k3.service:19`).
- **Attribution.** vLLM sees one credential, so identity only exists in the
  `k3usage` log line (`gen-keys.sh:91-93`), which is what the dashboard's
  per-customer view consumes.
- **Blast radius of a leak.** One key compromises one customer instead of all
  of them, and rotating it does not break every client at once
  (`gen-keys.sh:6-8`).

What it replaced, and why that was unsafe to hand to real customers: a single
shared key with **no identity, no quota and no rate limit**. `gen-keys.sh:4-9`
records the verified consequence — one caller could fire **40 concurrent
requests with zero throttling**, and hold a decode slot **for hours** with
`max_tokens=120000`. vLLM's `--api-key` does accept repeated flags for
rotation, but gives "no per-key identity, budgets, or revocation without
restart" (`K3-DEPLOYMENT.md` §11.3), so the map is not redundant with it.

The caps are deliberately sized for a **portal key shared by many end users**,
not one developer (`gen-keys.sh:25-37`): `PER_CUST_CONN=200` against
`max-num-seqs: 512` is ~39% of engine sequence slots for one key;
`PER_CUST_RATE=100r/s` with `PER_CUST_BURST=200` because at 10 r/s the rate cap
would have become the real limit long before the connection cap
(200 in-flight requests at ~3 s each turn over ~67/s); `GLOBAL_CONN=512` raised
to `max-num-seqs` so a 200-connection portal cannot own 78% of the edge as it
would have at 256. Change one of these numbers and re-read that comment block
first — they are mutually constrained, not independent knobs.

---

## 4. The authorization gap nginx closes

vLLM's `--api-key` middleware only guards the prefixes `("/v1","/v2",
"/inference")` — cited to `entrypoints/serve/utils/server_utils.py:42` in
`verify-auth.sh:4-5`. Every other route reaching the same inference functions
is unauthenticated. On this build, `POST /invocations` **answered 200 with no
credentials** (`verify-auth.sh:5-6`, `issue-cert.sh:69-71`): full inference for
free.

`verify-auth.sh` asserts nginx returns **403** for each of these without
credentials (`verify-auth.sh:31-40`):

| Endpoint | Why it matters |
|---|---|
| `POST /invocations` | full inference, verified 200 unauthenticated on vLLM |
| `POST /scale_elastic_ep` | changes engine parallelism |
| `POST /tokenize`, `POST /detokenize` | tokenizer access |
| `POST /generative_scoring` | model compute |
| `GET /load`, `GET /version` | engine/build disclosure |
| `GET /openapi.json`, `GET /docs` | API surface disclosure |

`K3-DEPLOYMENT.md` §11.3 adds that `/pooling`, `/classify`, `/score`,
`/rerank`, `/pause` and `/update_weights` are open on vLLM for the same reason.

**The config is an allowlist, and that is the whole point.** The TLS block
names only what is permitted — `= /metrics`, `= /health`, `/v1/`, the two
dashboard locations — and ends with `location / { return 403 ... }`
(`issue-cert.sh:109`). None of the endpoints in the table above is mentioned
anywhere in the nginx config; they are refused because they are not allowed.
§11.3 states the rule directly: allowlist paths, "since blocklisting breaks
each time vLLM adds an endpoint." If you add a `location` to this block you are
widening the security boundary — that is the review trigger.

Positive assertions in the same script pin the other side of the behaviour:
`/v1/models` gives 401 with no key and 401 with a wrong key, 200 with a valid
key; `/v1/chat/completions` and `/v1/completions` 200 with a valid key
(`verify-auth.sh:44-54`).

---

## 5. Defence in depth

Four independent layers, each of which still holds if another is
misconfigured.

1. **vLLM is loopback-only and still holds its own key.** `host: 127.0.0.1`
   (`config.yaml:106`); `verify-auth.sh:63-65` asserts both that a direct
   request to `:8001/v1/models` with no key returns **401** and that the socket
   is bound to `127.0.0.1:8001` and nothing wider. So bypassing nginx requires
   already being on the box, and even then `/v1` needs the upstream credential.
2. **Unmatched hosts are dropped, not served.** `nginx/00-default-deny.conf`
   claims `default_server` on both ports: port 80 `return 444` (close without
   response), port 443 `ssl_reject_handshake on` (refuse at TLS, never present
   the cert). Its comment states the purpose — scanners hitting the bare IP
   must not fall through to the cflox.store block
   (`nginx/00-default-deny.conf:1-2`). Without this file, the cflox.store
   server would be nginx's default and would answer for any `Host`.
3. **Diagnostics are local-only.** `location = /metrics` and
   `location = /health` are `allow 127.0.0.1; deny all;` with `access_log off`
   (`issue-cert.sh:72-73`). They are proxied rather than blocked outright
   because the dashboard scrapes `/metrics` over loopback
   (`verify-auth.sh:57-59`, `dashboard/server.py:29`).
4. **The dashboard is confined by systemd and unauthenticated only on
   loopback.** `NoNewPrivileges=true`, `PrivateTmp=true`,
   `ProtectSystem=strict`, `ProtectHome=true`
   (`systemd/k3dash.service:32-35`) — read-only everywhere is sufficient
   because it only scrapes loopback, tails a log, and shells out to `rocm-smi`
   (`systemd/k3dash.service:30-31`). `DASH_ADDR=127.0.0.1` is set explicitly
   with the warning that binding `0.0.0.0` "would re-open the unauthenticated
   hole that was just closed" (`systemd/k3dash.service:19-23`). nginx's basic
   auth against `/etc/nginx/k3-dash.htpasswd` is the only way in
   (`nginx/k3-dash.inc:4-5`).

Not a defence layer but worth knowing: this deployment is **not** exposed to
CVE-2026-48746 (`Host`-header auth bypass, affects vLLM <0.22.0) —
`K3-DEPLOYMENT.md` §11.3.

---

## 6. nginx fragment load order and dependencies

`nginx.conf:59` includes `/etc/nginx/conf.d/*.conf`, in glob (alphabetical)
order. Note that `*.inc` does **not** match that glob: the two `.inc` files are
pulled in only by explicit `include` statements from inside the TLS block.

```
  /etc/nginx/nginx.conf
      |
      +-- include conf.d/*.conf            (alphabetical)
           |
           1. 00-default-deny.conf   [committed: nginx/00-default-deny.conf]
           |     default_server sinks; defines nothing others need
           |
           2. 00-k3-keys.conf        [GENERATED by gen-keys.sh; gitignored]
           |     DEFINES:  map_hash_bucket_size 256
           |               map $http_authorization $k3_customer
           |               limit_conn_zone  k3_percust  (key $k3_customer)
           |               limit_req_zone   k3_rps      (key $k3_customer)
           |               limit_conn_zone  k3_global   (key $server_name)
           |               log_format k3usage
           |
           3. k3-tls.conf            [GENERATED by issue-cert.sh;
           |                          reviewable copy: nginx/k3-tls.conf.template]
           |     CONSUMES: $k3_customer, k3usage, k3_backend, k3_dash
           |     include conf.d/k3-limits.inc  -->  [GENERATED by gen-keys.sh]
           |     |                                   CONSUMES the three zones
           |     |                                   SETS the upstream Authorization
           |     include conf.d/k3-dash.inc    -->  [committed: nginx/k3-dash.inc]
           |                                         CONSUMES k3_dash
           |
           4. k3.conf                [committed: nginx/k3.conf]
                 DEFINES:  upstream k3_backend (keepalive 32)
                           upstream k3_dash    (keepalive 8)
```

The `00-` prefix on the key/limit fragment is not decoration: it makes the
`map`, the zones and the log format load before the server block that uses
them. Inside that one generated file the ordering is also fixed — the map must
precede the `limit_conn_zone $k3_customer` / `limit_req_zone $k3_customer`
lines that key on the variable it defines (`gen-keys.sh:75-88`).

`map_hash_bucket_size 256;` is required because the key material is long: the
generator's own comment says `'Bearer sk-k3-<name>-<40 hex>'` **exceeds the
default 64-byte bucket** (`gen-keys.sh:71-72`). Two consequences:

- Drop it and nginx fails to load the map.
- Declare it a second time in the TLS block and `nginx -t` fails on a
  **duplicate directive** — `write_tls_server()` calls this out explicitly and
  is why that block is kept auth-free (`issue-cert.sh:41-42`).

This ordering is the most fragile thing about a rebuild, because two of the
four `.conf` files and both `.inc` files are generated and gitignored
(`.gitignore`: `nginx/00-k3-keys.conf`, `nginx/k3-limits.inc`, `*.htpasswd`).
Recover them in this order: `gen-keys.sh` (rebuilds `00-k3-keys.conf` and
`k3-limits.inc` from `customers.tsv`), then `issue-cert.sh` (rebuilds
`k3-tls.conf`, but only after a cert exists — `nginx -t` fails on a missing
`ssl_certificate`, `issue-cert.sh:40`). Running them in the other order leaves
a TLS block referencing an undefined `$k3_customer`.

Both generators gate their own reload on `nginx -t` and refuse to reload on
failure (`gen-keys.sh:117-121`), and `issue-cert.sh` additionally rolls back to
`k3-tls.conf.pre-tls.bak` (`issue-cert.sh:133-138`). Preserve that pattern in
anything new.

---

## 7. Supervision model

**systemd owns the container; docker's restart policy is not used.** The
container runs `--rm` with `Restart=always` on the unit (`k3.service:17`,
`k3.service:29`). The reason is in the unit's own comment
(`k3.service:18-20`): docker's backoff **resets to 100 ms once a container has
run >=10 s** (moby `restartmanager.go`), and this model takes ~6 min to load —
so every crash clears that threshold and docker would hot-loop a 6-minute
startup forever, learning nothing.

systemd escalates instead: `RestartSec=30s`, `RestartSteps=4`,
`RestartMaxDelaySec=10min`, with `StartLimitIntervalSec=1800` /
`StartLimitBurst=5` to stop retrying after 5 failures in 30 min rather than
looping forever (`k3.service:11-13`, `k3.service:21-23`). Timeouts are sized to
the real startup: `TimeoutStartSec=900` (load is ~143 s of weights plus ~74 s
engine init per `K3-DEPLOYMENT.md` §3, with headroom), `TimeoutStopSec=120`
against `ExecStop=docker stop -t 60` (`k3.service:24-25`, `k3.service:39`).

`RequiresMountsFor=/scratch` appears on both units. It implies **both**
`Requires=` and `After=` for `scratch.mount`, so a separate `After=` line would
be redundant (`k3.service:3-7`, `systemd/k3dash.service:3-4`); the bind mounts
`-v /scratch/hf:/hf -v /scratch/results:/results` (`k3.service:34`) cannot be
attempted before the disk is there.

**The dashboard only `Wants=` the engine**, never `Requires=`
(`systemd/k3dash.service:5-8`): "the dashboard is a monitor, so it must still
start (and show the engine as down) when k3 is stopped or crash-looping." A
`Requires=` here would take the monitoring offline at exactly the moment it is
needed. `server.py` is built for that case — it has an explicit degrade path
and an alert reading "Engine is not answering /metrics ... Serving is ..."
(`dashboard/server.py:511`, `dashboard/server.py:806`).

---

## 8. Serving-engine shape

Enough to reason about the edge; measurements are in `K3-DEPLOYMENT.md` §2.

- **Hybrid attention is why long context is affordable at all.** Of 93 layers,
  **69 are KDA** (linear attention, constant state per sequence) and only **24
  `full_attn_layers`** hold length-proportional KV (`config.yaml:82-85`). A
  256k context therefore costs KV in 24 layers, not 93.
- **`block-size` is deliberately absent.** Setting 128 was a no-op: vLLM
  overrides it for hybrid models — "Setting attention block size to 768 tokens
  to ensure that attention page size is >= mamba page size"
  (`interface.py:911`, quoted at `config.yaml:79-82`) — then pads the mamba
  page by 8.68% to match. Do not re-add it expecting an effect.
- **Prefix caching is the single largest throughput factor**, not the obvious
  prefill saving. With an ~85% shared prefix, per-sequence cost falls from
  12,800 to 2,355 tokens, a **5.4x** reduction, because the shared KV is stored
  **once** rather than per request; KV utilisation fell from 98.6% (thrashing)
  to 49.4% (`K3-DEPLOYMENT.md` §2.1). `enable-prefix-caching: true` with
  `prefix-match-unit: 128` (`config.yaml:87-89`). Anything that perturbs the
  token prefix destroys this — see `K3-DEPLOYMENT.md` §11.2 on `cache_salt`,
  and §11.4 on proxies that rewrite tool schemas.
- **KV pool versus concurrency is the real tradeoff.** KV pages are allocated
  on demand, so raising `max-model-len` does not reserve anything; it bounds how
  much cache one request may monopolise. Against the measured KV pool at 0.92 of
  **2,295,266 tokens**, the worst-case concurrency ladder is `128k -> 17.5x,
  256k -> 8.76x, 384k -> 5.8x, 512k -> 4.4x` (`config.yaml:36-40`).
  `max-num-seqs: 512` is the
  sequence-slot ceiling (`config.yaml:63`); `max-num-batched-tokens: 8192`
  keeps chunked prefill from raising the activation peak and is the value the
  2.50M TPM result was measured with (`config.yaml:64-67`).
- **`FULL_DECODE_ONLY` is a KV windfall, not a compilation step.** It was
  applied intending to enable `torch.compile`; it did not, because Kimi-K3's
  model class lacks `@support_torch_compile`. The gain came from capturing far
  fewer CUDA graph variants, freeing **11.7 GiB** into KV: 49.41 -> 61.09 GiB,
  **1,169,808 -> 1,696,278 KV tokens** (`config.yaml:91-96`,
  `K3-DEPLOYMENT.md` §2.3). That is why KV — not `--max-num-seqs` — was the
  binding constraint at 512 concurrent.
- **`gpu-memory-utilization: 0.92`** (`config.yaml:78`), reverted from 0.96
  after 0.96 crashed the engine on a long prefill; 0.96 did grow KV to
  2,732,926 tokens but left only ~11.5 GiB/GPU of working room, while 0.92
  leaves **~16.6 GiB/GPU**, measured via `rocm-smi` with the engine loaded and
  serving (`config.yaml:69-77`). That 16.6 GiB is what makes 256k survivable.
  Note the file retracts an earlier ~23 GiB figure as "wrong in the dangerous
  direction" (`config.yaml:73-75`) — if you have that number in your head from
  an older copy, discard it. The file says in capitals: do not raise this
  without testing a >186k-token prompt.
- **Config-file footgun.** Never write `some-flag: false` — the parser emits
  nothing for a false boolean, silently giving the flag's *default* rather than
  the disabled state. Use the positive form of the negative flag, as
  `no-enable-log-requests: true` does (`config.yaml:5-8`, `config.yaml:103-104`).
- **Model naming is a compatibility contract.** `served-model-name` lists
  `FW-Kimi-K3` first (what `/v1/models` and every response reports) and keeps
  `kimi-k3` as an alias because live customer integrations still send it;
  dropping it 404s them (`config.yaml:10-15`).

The AITER environment flags live in `vllm-k3.env` and cannot move into
`config.yaml`, because vLLM reads them from `os.environ` while `--config` only
produces argv (`vllm-k3.env.example:1-2`). One pairing is safety-critical:
`AITER_SITUV2_A8W4` without `AITER_BF16_FP8_MOE_BOUND=0` produces **fluent but
wrong output at full speed** — never change one without the other, and run
`tests/gate.sh` afterwards (`vllm-k3.env.example:7-9`, `K3-DEPLOYMENT.md` §7.3).

---

## 9. State and persistence

Since 2026-09-18 `/scratch` is a directory on the **boot disk** (`/dev/vda1`,
2 TB), not the separate 40 TB volume. Everything below therefore sits on one
device, which is what makes a droplet snapshot self-contained: restoring one
brings the weights, the image, the secrets and the certificate with it. The
40 TB volume is attached at `/mnt/bulk` and nothing depends on it.

The distinction that still matters is **committed in this repo** versus
**machine-local**, since a droplet can be reclaimed at any time.

| Machine-local state (in a snapshot, not in git) | Also machine-local, outside `/scratch` |
|---|---|
| `/scratch/hf/hub/models--moonshotai--Kimi-K3` — weights, 96 safetensors, 1.5 TB (`K3-DEPLOYMENT.md` §3) | `/etc/nginx/` — `nginx.conf` and `conf.d/` fragments |
| `/scratch/hf/config.yaml` — what the container actually reads as `/hf/config.yaml` | `/etc/letsencrypt/live/<domain>/` — cert and key |
| `/scratch/results` — benchmark output (`k3.service:34`) | `/opt/k3dash/` — `server.py`, `index.html` |
| `/scratch/deploy` — this repo, including the live secret files | `/etc/systemd/system/k3.service`, `k3dash.service` |
| `/var/log/k3/usage.log` — the per-customer billing record, which exists nowhere else | `/etc/nginx/k3-dash.htpasswd` |

**This repo is the durable copy; the box is disposable.** A snapshot makes
recovery fast, but the repo is what makes it possible at all. Everything in the
right-hand column is either committed here or regenerable from something here:

- nginx fragments: `nginx/00-default-deny.conf`, `nginx/k3.conf`,
  `nginx/k3-dash.inc` committed and **verified byte-identical to the live
  files** (`diff`, 2026-09-18); `k3-tls.conf` regenerated by `issue-cert.sh`
  with `nginx/k3-tls.conf.template` as the reviewable copy; `00-k3-keys.conf`
  and `k3-limits.inc` regenerated by `gen-keys.sh` from `customers.tsv`.
- systemd units: `k3.service` and `systemd/k3dash.service`, both verified
  identical to `/etc/systemd/system/`.
- dashboard: `dashboard/server.py` and `dashboard/index.html`, both verified
  identical to `/opt/k3dash/`.
- cert: re-issued by `issue-cert.sh`, which will not call certbot unless DNS
  already resolves to `EXPECT_IP` — a failed HTTP-01 burns Let's Encrypt rate
  limit (5 failures/hostname/hour) (`issue-cert.sh:3-6`, `issue-cert.sh:20-34`).
- weights: re-pulled per `K3-DEPLOYMENT.md` §4.2 (~700 s at ~2.23 GB/s, §3).
- secrets: **not** in the repo by design. Every one has a committed `*.example`
  twin (`.gitignore`, `vllm-k3.env.example`, `customers.tsv.example`). Losing
  the box means minting new keys and redistributing them — see
  `docs/SECRETS.md`.

Two gaps worth naming:

- **`/hf/config.yaml` is a copy, not a symlink.** The container mounts
  `/scratch/hf` as `/hf` (`k3.service:34`) and reads `/hf/config.yaml`
  (`k3.service:39`), i.e. `/scratch/hf/config.yaml` — a separate file from the
  repo's `config.yaml`. Their contents are currently identical (`diff`), but
  editing the repo copy alone does **not** change what the engine loads on next
  restart. Copy it across deliberately.
- **Local-only state that nothing regenerates:** `/var/log/k3/usage.log`
  is the sole record of per-customer usage and is not shipped anywhere.
  `TODO(operator):` confirm whether usage attribution needs to survive droplet
  reclaim, and if so, where it is archived.

---

## 10. `max-model-len`: a settled contradiction, and what is still loose

This section was drafted against a `config.yaml` in which line 51 set
`max-model-len: 262144` while the comment block directly above it argued that
262,144 "was tried and is NOT safe on this build" and that 128k was "the
safety property that matters." The file was rewritten on 2026-09-18 while this
document was being written, and the contradiction is now resolved **in the
file, by measurement**. Both states are recorded here because the reasoning is
what protects the next person who wants to raise the value.

**Current state** (`config.yaml:62`): `max-model-len: 262144`, at
`gpu-memory-utilization: 0.92`.

**The crash that caused the doubt** is retained as history
(`config.yaml:42-47`): 262,144 was tried once at `gpu-memory-utilization
0.96` and killed the engine mid-prefill at `num_computed_tokens=186,624` with
`EngineDeadError`. It was **not** KV exhaustion — `kv_cache_usage` was 0.084 at
death. The cause was VRAM headroom: `dmesg` logged
`svm_range_evict_svm_bo_worker` (amdgpu evicting buffer objects) at the exact
second of the crash, because 0.96 left only ~11.5 GiB/GPU spare.

**What settled it** — exactly the test the older comment demanded, concurrency
with unique prompts rather than a single-request probe
(`config.yaml:49-56`), run at 0.92 (~16.6 GiB/GPU spare):

| Shape | Result |
|---|---|
| single prefill 249,696 tok | PASS, 0 new evictions |
| 4 x 212k concurrent, unique prompts (848k) | PASS, 95 s, 0 evictions |
| 6 x 191k concurrent, unique prompts (1.15M) | PASS, 59 s, 0 evictions, 0 preemptions |
| `>262,144` request | clean 400, engine survives |

The file notes the prompts were randomised **specifically to defeat prefix
caching**, because an earlier run with identical prompts finished in 6 s and
proved nothing (`config.yaml:55-56`). That methodological point is the reason
the result is trustworthy, and it is the trap to avoid if anyone re-runs this.

**Why 256k rather than something smaller** is now argued from client behaviour
rather than headroom (`config.yaml:29-34`): clients send a *fixed* `max_tokens`
(Foundry-style SDKs default to 128,000) and vLLM reserves prompt + `max_tokens`
against one shared window, so usable prompt = `max-model-len` − 128,000. At
131,072 that left only 3,072 prompt tokens and 400'd nearly every agentic
request; 256k leaves 134,144. 204,800 was rejected as too tight.

**The safety property is unchanged, and is the thing to preserve:** an
oversized request gets a clean 400 rather than taking the engine down
(`config.yaml:58-59`). Raising `max-model-len` **further** means re-testing the
new shape under concurrency with unique prompts and watching `dmesg` for
`svm_range_evict_svm_bo_worker` — not a single-request probe
(`config.yaml:59-61`).

Still loose:

- **Stale snapshots.** `config.yaml.bak-12800` and
  `config.yaml.bak-131072-1635` are both in the repo root; the latter differs
  from the current `config.yaml` despite its name. `.gitignore:38-40` treats
  `*.bak-*` as machine-local hand-edit debris, so they are not committed, but
  they are live on the box and misleading to read.
  `TODO(operator):` delete them once the current value has held in production.
- **`/scratch/hf/config.yaml` is in sync right now** (`diff`, verified), but it
  is a copy, not a symlink — see §9. Any future edit has to be propagated by
  hand, and a rewrite as substantial as this one is exactly when that gets
  missed.

---

## 11. Discrepancies found while writing this (2026-09-18)

1. **`k3-tls.conf`: live file has comments the generator does not emit.**
   Directives are identical — `diff` of the substituted
   `write_tls_server()` heredoc against `/etc/nginx/conf.d/k3-tls.conf`, with
   comments and blank lines stripped, is empty. But the live file adds a header
   block and several inline comments (on `client_max_body_size`, `/metrics`, the
   401, the catch-all) and **drops** the generator's "Same allowlist as the
   :8000 edge..." comment. The live file was therefore hand-annotated after
   generation, or generated by an earlier revision of the script. No behavioural
   difference today, but the next `issue-cert.sh` run will silently discard those
   comments. `TODO(operator):` fold the wanted comments into
   `write_tls_server()` so they survive regeneration.
2. **`verify-auth.sh` targeted an edge that no longer existed — fixed
   2026-09-18.** It probed `EDGE=http://127.0.0.1:8000` (`verify-auth.sh:9`),
   but `nginx/k3.conf:1-2` records that "the plaintext :8000 edge was removed
   when the API moved to TLS," so the script could not pass on this box. It now
   targets `https://$DOMAIN`, pinning the hostname to loopback with
   `curl --resolve` so it tests the real TLS server block, and distinguishes
   the upstream key (which the edge must reject) from a customer key (which it
   must accept). The suite passes end to end.
3. **`config.yaml` was rewritten mid-review.** All `config.yaml` line citations
   in this document refer to the revision on disk at 2026-09-18 16:35, in which
   `max-model-len` is line 62 and `gpu-memory-utilization` is line 78. The
   earlier revision had them at lines 51 and 64. Re-check line numbers before
   trusting any citation here if the file has moved again; the quoted text is
   the stable part.
4. **Everything else matches.** `nginx/00-default-deny.conf`, `nginx/k3.conf`,
   `nginx/k3-dash.inc`, `k3.service`, `systemd/k3dash.service`,
   `dashboard/server.py`, `dashboard/index.html` are byte-identical to their
   live counterparts, and `/scratch/hf/config.yaml` currently matches
   `config.yaml`.

---

## 12. Blast radius — what breaks if you change X

| Change | What it breaks |
|---|---|
| Add a `location` to the TLS block | Widens the allowlist of §4. Every unlisted vLLM endpoint is currently refused *by omission*. |
| Delete or reorder `00-k3-keys.conf` | `$k3_customer`, the three limit zones and `log_format k3usage` all vanish; the TLS block references all four. |
| Add `map_hash_bucket_size` to the TLS block | `nginx -t` fails: duplicate directive (`issue-cert.sh:42`). |
| Lower `proxy_read_timeout` below ~112 s | Streams cut off mid-generation (`issue-cert.sh:86`). |
| Turn `proxy_buffering` on | SSE batches instead of streaming (`K3-DEPLOYMENT.md` §11.3). |
| Bind vLLM to `0.0.0.0` | Removes layer 1 of §5; only vLLM's own partial `--api-key` remains, and `/invocations` is not covered by it. |
| Bind the dashboard to `0.0.0.0` | Unauthenticated monitoring exposed — the unit warns about this directly (`systemd/k3dash.service:21-22`). |
| Remove `00-default-deny.conf` | cflox.store becomes nginx's default server and answers for any `Host`. |
| Rotate `VLLM_API_KEY` | Requires a vLLM restart, ~6 min of downtime; `api-key.txt` and `vllm-k3.env` must match exactly (`vllm-k3.env.example:19-23`). |
| Rotate one customer key | `gen-keys.sh` + nginx reload. No engine restart, no other customer affected. |
| Edit repo `config.yaml` only | No effect on the engine: it loads `/scratch/hf/config.yaml` (§9). |
| Change `AITER_SITUV2_A8W4` alone | Fluent but wrong output at full speed (`vllm-k3.env.example:7-9`). |
| Change `PER_CUST_CONN` / `PER_CUST_RATE` in isolation | The two are mutually constrained; re-read `gen-keys.sh:25-37`. |
| Raise `gpu-memory-utilization` or `max-model-len` | Costs the VRAM headroom that makes 256k survivable. Requires re-testing the new shape under concurrency with unique prompts, watching `dmesg` (§10). |

---

## See also

- [`K3-DEPLOYMENT.md`](K3-DEPLOYMENT.md) — measurements, tuning history,
  failure catalog. Start at §1 (headline numbers), §2 (what moved the needle),
  §3 (environment), §7 (failure catalog), §11.3 (exposure), §11.5 (verified
  metric names).
- `docs/DEPLOY.md` — bring-up from a bare droplet.
- `docs/RUNBOOK.md` — day-to-day operations and incident response.
- `docs/SECRETS.md` — credential inventory, rotation, and where each one lives.
- `nginx/k3-tls.conf.template` — reviewable copy of the generated TLS edge.
- `verify-auth.sh` — executable specification of the edge's auth behaviour
  (but see §11, item 2).
