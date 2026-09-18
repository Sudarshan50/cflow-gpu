#!/usr/bin/env bash
# One-shot bring-up for the Kimi-K3 endpoint: bare box -> authenticated TLS API.
#
# Everything this repo knows how to do, in the one order that works. The stages
# are separable so an operator can re-run a single piece, and every stage is
# idempotent: re-running converges, it does not duplicate or clobber.
#
#   ./deploy.sh                 full sequence (same as `all`)
#   ./deploy.sh <stage> [...]   run only these stages, in the order given
#   ./deploy.sh --dry-run all   print the plan, touch nothing
#   ./deploy.sh --help          stage list
#
# WHAT THIS SCRIPT DELIBERATELY WILL NOT DO
#   - restart a HEALTHY k3.service. Model load is ~6 min (docs §4.3), so an
#     unattended restart is a 6-minute customer-visible outage. `services` uses
#     `enable --now`, which is a no-op on an active unit; when an installed file
#     changed underneath a running unit the script WARNS and leaves the restart
#     to a human.
#   - run mkfs. docs §4.1 marks it destructive/first-setup-only.
#   - bypass issue-cert.sh's DNS gate. A failed HTTP-01 burns Let's Encrypt
#     rate limit (5 failures/hostname/hour, issue-cert.sh:4-5).
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration. Every value here is overridable from the environment and
# every default is the value that is live on the box today; nothing is invented.
# ---------------------------------------------------------------------------
REPO="${REPO:-/scratch/deploy}"

DOMAIN="${DOMAIN:-cflox.store}"                    # issue-cert.sh:11
EXPECT_IP="${EXPECT_IP:-201.79.29.187}"            # issue-cert.sh:13
MODEL="${MODEL:-moonshotai/Kimi-K3}"               # k3.service:39
SERVED_NAME="${SERVED_NAME:-FW-Kimi-K3}"           # config.yaml served-model-name[0]

# Pinned by BITS, not by name: the `kimi-k3` tag is a one-off dev tag that may
# be garbage-collected upstream (launch.sh:10-11).
IMAGE_TAG="${IMAGE_TAG:-vllm/vllm-openai-rocm:kimi-k3}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b}"
IMAGE_REPO="${IMAGE_REPO:-vllm/vllm-openai-rocm}"
IMAGE_TARBALL="${IMAGE_TARBALL:-/scratch/backup/k3-image.tar}"   # launch.sh:13

VLLM_PORT="${VLLM_PORT:-8001}"                     # config.yaml port
DASH_PORT="${DASH_PORT:-8080}"                     # systemd/k3dash.service:23
HTTPS_PORT="${HTTPS_PORT:-443}"
# The plaintext :8000 edge was removed when the API moved to TLS
# (nginx/k3.conf:1-2). Nothing should listen there; stage_preflight warns if
# something does, since it would be serving inference without the TLS edge's
# auth and path allowlist.
LEGACY_EDGE_PORT="${LEGACY_EDGE_PORT:-8000}"

SCRATCH="${SCRATCH:-/scratch}"
HF_DIR="${HF_DIR:-/scratch/hf}"                    # HF_HOME on the host
RESULTS_DIR="${RESULTS_DIR:-/scratch/results}"     # k3.service:34 bind mount
MODEL_DIR="${MODEL_DIR:-$HF_DIR/hub/models--moonshotai--Kimi-K3}"
HFV="${HFV:-/opt/hfv}"                             # cloud-init.yaml:39 download venv
DASH_DIR="${DASH_DIR:-/opt/k3dash}"                # systemd/k3dash.service:13
# tests/gate.sh execs /scratch/gate.py and gate.py reads
# /scratch/gate-baseline.json (tests/gate.sh:6, tests/gate.py:37). Those paths
# are hardcoded, so the tests are installed FLAT into /scratch, which is also
# what docs §6 and the Makefile (`bash /scratch/gate.sh`) tell operators to run.
TEST_DIR="${TEST_DIR:-/scratch}"

NGINX_DIR="${NGINX_DIR:-/etc/nginx/conf.d}"
HTPASSWD="${HTPASSWD:-/etc/nginx/k3-dash.htpasswd}"   # nginx/k3-dash.inc:5
DASH_USER="${DASH_USER:-admin}"                       # the user in the live htpasswd
CERT_LIVE="${CERT_LIVE:-/etc/letsencrypt/live/$DOMAIN}"
# Scripts here run only after a successful renewal. Empty on the live box until
# the install stage populates it; see nginx/certbot-deploy-hook.sh.
RENEWAL_HOOK_DIR="${RENEWAL_HOOK_DIR:-/etc/letsencrypt/renewal-hooks/deploy}"
# Deliberately not /var/log/nginx - see nginx/k3-usage.logrotate.
USAGE_LOG_DIR="${USAGE_LOG_DIR:-/var/log/k3}"
USAGE_LOG="${USAGE_LOG:-$USAGE_LOG_DIR/usage.log}"

# 900s mirrors TimeoutStartSec in k3.service:24, which is itself sized for the
# ~6 min load (140s weights + 74s engine init + overhead, docs §4.3).
READY_TIMEOUT="${READY_TIMEOUT:-900}"
READY_POLL="${READY_POLL:-10}"
# Weights are ~1.5 TB on disk (`df /scratch` reads 1.5T used with the model
# present). Ask for 1.6 TiB so a download cannot wedge the filesystem.
MIN_FREE_GIB="${MIN_FREE_GIB:-1600}"

DRY_RUN=0
TLS_ARGS=()          # --watch is passed straight through to issue-cert.sh
RESTART_HINTS=()     # units whose on-disk file changed under a running instance
TLS_DEFERRED=0

STAGES_ALL=(preflight host weights secrets install image keys services tls verify)

# ---------------------------------------------------------------------------
# Helpers. Same shape as issue-cert.sh:18 so interleaved output reads as one log.
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN  %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] FATAL %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }
step() { printf '\n[%s] ==== %s ====\n' "$(date -u +%H:%M:%S)" "$*"; }

# Every mutation goes through run(), which is what makes --dry-run trustworthy.
run() {
  if [ "$DRY_RUN" = 1 ]; then printf '        + %s\n' "$*"; return 0; fi
  "$@"
}

# Heredoc writer. Consumes stdin even when dry, so `run_write ... <<EOF` blocks
# do not deadlock or leak into the next command.
run_write() {
  local path="$1" mode="$2"
  if [ "$DRY_RUN" = 1 ]; then
    printf '        + write %s (mode %s)\n' "$path" "$mode"
    cat >/dev/null
    return 0
  fi
  local tmp; tmp="$(mktemp "${path}.XXXXXX")"
  cat > "$tmp"
  chmod "$mode" "$tmp"
  mv -f "$tmp" "$path"
}

have() { command -v "$1" >/dev/null 2>&1; }

# True when the two files differ (or the destination is absent). Used to keep
# install idempotent AND to notice that a running unit is now stale.
differs() { ! cmp -s "$1" "$2"; }

# install(1) with a change report, so the log shows what actually moved.
sync_file() {
  local src="$1" dst="$2" mode="$3"
  [ -f "$src" ] || die "missing repo file: $src"
  if differs "$src" "$dst"; then
    run install -m "$mode" -o root -g root "$src" "$dst"
    log "  installed $dst (mode $mode)"
    return 1          # 1 == "changed", callers use this to raise restart hints
  fi
  # chmod anyway: content can match while modes have drifted.
  run chmod "$mode" "$dst"
  log "  unchanged $dst"
  return 0
}

# `|| true` is load-bearing under `set -e`: curl exits 7 on a refused
# connection, and an unguarded $(http_code ...) assignment would abort the run
# instead of reporting the 000 that printf already emitted.
http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time "${2:-20}" "$1" "${@:3}" || true; }

# nginx -t before every reload, always. A config that does not validate is
# never reloaded; nginx keeps serving the old one.
nginx_ok() { nginx -t >/dev/null 2>&1; }

nginx_reload() {
  local why="$1"
  if [ "$DRY_RUN" = 1 ]; then printf '        + nginx -t && systemctl reload nginx (%s)\n' "$why"; return 0; fi
  if ! nginx_ok; then
    warn "nginx config invalid - NOT reloading ($why):"
    nginx -t 2>&1 | tail -5 >&2
    return 1
  fi
  systemctl reload nginx && log "  nginx reloaded ($why)"
}

usage() {
  cat <<EOF
deploy.sh - bring the Kimi-K3 endpoint up from bare metal, idempotently.

usage: deploy.sh [--dry-run] [--watch] [stage ...]

stages (in dependency order; "all" runs exactly this list):
  preflight   GPUs (/dev/kfd + rocm-smi), docker, nginx, certbot, jq, zstd,
              python3-venv, openssl, curl; /scratch mounted with room for 1.5 TB
  host        sysctl /etc/sysctl.d/99-vllm.conf (values from cloud-init.yaml),
              /scratch mount + fstab entry, directories, ufw allow 22/80/443
  weights     hf download $MODEL into $HF_DIR; skipped if
              $MODEL_DIR exists
  secrets     api-key.txt, dash-password.txt, vllm-k3.env (from the .example,
              VLLM_API_KEY := api-key.txt), $HTPASSWD
  install     config.yaml -> $HF_DIR/config.yaml, dashboard -> $DASH_DIR,
              units -> /etc/systemd/system, nginx fragments -> $NGINX_DIR,
              tests -> $TEST_DIR
  image       verify $IMAGE_DIGEST by BITS; restore from
              $IMAGE_TARBALL, else pull; fail closed on mismatch
  keys        gen-keys.sh: 00-k3-keys.conf + k3-limits.inc from customers.tsv
  services    daemon-reload, enable --now docker/k3/k3dash/nginx/certbot.timer,
              then poll :$VLLM_PORT/v1/models until 200 (up to ${READY_TIMEOUT}s)
  tls         issue-cert.sh (DNS-gated; --watch polls until DNS resolves)
  verify      verify-auth.sh against https://$DOMAIN (needs a cert and at
              least one customer key), then tests/gate.sh

flags:
  --dry-run   print every mutation instead of running it. Read-only probes
              still run, so the plan reflects the box's current state.
  --watch     passed through to issue-cert.sh: poll DNS every 60s for up to 4h.
  --help      this text.

env overrides (current defaults): DOMAIN=$DOMAIN EXPECT_IP=$EXPECT_IP
  MODEL=$MODEL SERVED_NAME=$SERVED_NAME REPO=$REPO
  VLLM_PORT=$VLLM_PORT DASH_PORT=$DASH_PORT HTTPS_PORT=$HTTPS_PORT
  READY_TIMEOUT=$READY_TIMEOUT MIN_FREE_GIB=$MIN_FREE_GIB
EOF
}

# ===========================================================================
# ORDERING CONSTRAINTS ENCODED BELOW. These are the reason this script exists;
# each one was a real failure, and the stage order is the only safe topology.
#
#   host    < everything      /scratch is not mounted by default and is lost on
#                             reclaim (cloud-init.yaml:25); every path below
#                             lives on it.
#   weights < services        vllm-k3.env sets HF_HUB_OFFLINE=1 (line 17), so a
#                             missing weight tree is a hard startup failure, not
#                             a silent re-download (launch.sh:39-41).
#   install < services        k3.service mounts /scratch/hf as /hf and serves
#                             `--config /hf/config.yaml` (k3.service:34,39), so
#                             config.yaml must already be at $HF_DIR/config.yaml.
#   secrets < keys            gen-keys.sh:20 reads api-key.txt at line 20 and
#                             bakes it into k3-limits.inc as the upstream
#                             Authorization header.
#   secrets < services        k3dash.service:28 points API_KEY_FILE at
#                             api-key.txt; the dashboard cannot scrape without it.
#   secrets = one credential  VLLM_API_KEY in vllm-k3.env MUST equal api-key.txt.
#                             A mismatch is a silent 401 at the upstream hop and
#                             nothing in the edge config reveals it.
#   keys    < tls             k3-tls.conf's /v1/ block tests $k3_customer and
#                             includes k3-limits.inc (issue-cert.sh:75,77); both
#                             come from gen-keys.sh, and nginx -t fails on an
#                             undefined variable or a missing include.
#   image   < services        k3.service:37 runs the image BY DIGEST. A tarball
#                             saved from a digest restores with RepoTags None and
#                             no RepoDigests (docs "Image backup" §2), and then
#                             the unit cannot resolve its own image reference.
#   services< tls             issue-cert.sh verifies through the edge at the end
#                             (issue-cert.sh:141-153); with the engine down those
#                             probes fail for the wrong reason.
#   tls     last              k3-tls.conf references
#                             $CERT_LIVE/fullchain.pem and nginx -t FAILS if the
#                             cert is absent, which is why issue-cert.sh only
#                             writes that file after certbot succeeds
#                             (issue-cert.sh:40, 130-131). Never pre-place it.
# ===========================================================================

# ---------------------------------------------------------------------------
stage_preflight() {
  step "preflight"
  local bad=0

  # /dev/kfd is the amdgpu compute device the container is handed
  # (k3.service:30). Its absence means no GPUs, and vLLM will not start.
  [ -e /dev/kfd ] || { warn "/dev/kfd absent - no AMD compute device visible"; bad=1; }
  [ -d /dev/dri ] || { warn "/dev/dri absent - no render nodes"; bad=1; }
  if have rocm-smi; then
    local ngpu
    # Count DISTINCT indices: --showid prints 5 lines per GPU (Device Name, ID,
    # Rev, Subsystem ID, GUID), so a plain line count reads 40 on this box.
    ngpu="$(rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l || true)"
    log "  rocm-smi reports ${ngpu:-0} GPU(s) (tensor-parallel-size 8 needs 8)"
    [ "${ngpu:-0}" -ge 8 ] || warn "fewer than 8 GPUs visible; config.yaml pins tensor-parallel-size 8"
  else
    warn "rocm-smi not on PATH (the dashboard shells out to it for GPU telemetry)"
  fi

  # jq/zstd/python3.12-venv are the cloud-init package set (cloud-init.yaml:8);
  # python3.12-venv in particular is NOT preinstalled and its absence is
  # failure §7.8 in the docs.
  local c
  for c in docker nginx certbot openssl curl python3 jq zstd install; do
    if have "$c"; then printf '  %-10s %s\n' "$c" "$(command -v "$c")"
    else warn "missing command: $c"; bad=1; fi
  done
  python3 -c 'import venv' 2>/dev/null \
    || { warn "python3 venv module missing - apt-get install -y python3.12-venv (docs §7.8)"; bad=1; }
  # htpasswd is NOT installed on this box (apache2-utils absent) and the live
  # $HTPASSWD is an $apr1$ hash, so stage_secrets uses `openssl passwd -apr1`
  # rather than adding a package dependency.
  have htpasswd || log "  htpasswd absent - using openssl passwd -apr1 (matches the live file)"

  docker info >/dev/null 2>&1 || { warn "docker daemon not responding"; bad=1; }

  if mountpoint -q "$SCRATCH"; then
    local free_gib
    free_gib="$(df -BG --output=avail "$SCRATCH" | tail -1 | tr -dc '0-9')"
    log "  $SCRATCH mounted, ${free_gib} GiB free"
    if [ -d "$MODEL_DIR" ]; then
      log "  weights present, skipping the ${MIN_FREE_GIB} GiB free-space requirement"
    elif [ "${free_gib:-0}" -lt "$MIN_FREE_GIB" ]; then
      warn "only ${free_gib} GiB free; the weight pull needs ~${MIN_FREE_GIB} GiB"; bad=1
    fi
  else
    warn "$SCRATCH is not mounted - run the host stage"
  fi

  [ "$bad" = 0 ] || die "preflight failed; fix the warnings above (nothing was changed)"
  log "preflight ok"
}

# ---------------------------------------------------------------------------
stage_host() {
  step "host prep"

  # Verbatim from cloud-init.yaml:12-18 (= docs §4.1 "values currently in
  # effect"). Sized for the 1.5 TB pull; do not tune these by feel, they are
  # the values the measured 2.23 GB/s was achieved with.
  local sysctl_f=/etc/sysctl.d/99-vllm.conf
  local want; want="$(cat <<'EOF'
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456
net.core.default_qdisc = fq_codel
net.ipv4.tcp_congestion_control = cubic
EOF
)"
  # Compare EFFECTIVE values, not file bytes. The live file carries an operator
  # comment header, so a byte comparison would rewrite it (and drop the
  # comment) on every run. The file must still exist for the values to survive
  # a reboot, hence both halves of the condition.
  local drift=0 k v cur line
  while IFS= read -r line; do
    k="${line%% = *}"; v="${line#* = }"
    cur="$(sysctl -n "$k" 2>/dev/null | tr -s '[:space:]' ' ')"
    # sysctl -n tab-separates the tcp_rmem/tcp_wmem triples and leaves a
    # trailing space after the squeeze, so compare on a normalised form.
    [ "${cur% }" = "$(printf '%s' "$v" | tr -s '[:space:]' ' ')" ] || { drift=1; log "  sysctl drift: $k is '${cur% }', want '$v'"; }
  done <<< "$want"

  if [ "$drift" = 1 ] || [ ! -f "$sysctl_f" ]; then
    printf '# vLLM / large model weight transfer tuning\n%s\n' "$want" | run_write "$sysctl_f" 644
    run sysctl --system >/dev/null
    log "  wrote $sysctl_f and applied"
  else
    log "  sysctl tuning already in effect and persisted in $sysctl_f"
  fi

  # /scratch is an ephemeral 40T volume that is NOT mounted by default and is
  # lost when the droplet is reclaimed (cloud-init.yaml:25).
  if ! grep -q DOSCRATCH /etc/fstab; then
    log "  adding DOSCRATCH to /etc/fstab"
    if [ "$DRY_RUN" = 1 ]; then
      printf '        + append LABEL=DOSCRATCH line to /etc/fstab\n'
    else
      printf 'LABEL=DOSCRATCH %s ext4 discard,errors=remount-ro 0 2\n' "$SCRATCH" >> /etc/fstab
    fi
  else
    log "  fstab already has the DOSCRATCH entry"
  fi
  run mkdir -p "$SCRATCH"
  if ! mountpoint -q "$SCRATCH"; then
    # No mkfs here on purpose: docs §4.1 marks it destructive and first-setup
    # only. An unlabelled disk is an operator decision, not a script's.
    if [ "$DRY_RUN" = 1 ] || blkid -L DOSCRATCH >/dev/null 2>&1; then
      run mount "$SCRATCH"
    else
      die "no filesystem labelled DOSCRATCH. TODO(operator): create it once with
       mkfs.ext4 -L DOSCRATCH <device>   # DESTRUCTIVE, docs §4.1
     then re-run: $0 host"
    fi
  fi
  log "  $SCRATCH mounted"

  run mkdir -p "$HF_DIR" "$RESULTS_DIR" "$REPO" "$DASH_DIR" /var/www/certbot
  # Only these three ports are open on the live box; 8000/8080 were closed when
  # the API moved to TLS, so the docs' "Current firewall" section is stale.
  # Rules only - never `ufw enable` from a script, because enabling before 22 is
  # allowed locks out ssh.
  if have ufw && ufw status 2>/dev/null | head -1 | grep -q active; then
    local p
    for p in 22 80 "$HTTPS_PORT"; do run ufw allow "${p}/tcp" >/dev/null; done
    log "  ufw allows 22/80/$HTTPS_PORT (80 is required for the ACME HTTP-01 challenge)"
  fi
}

# ---------------------------------------------------------------------------
stage_weights() {
  step "weights"
  # Presence of this directory is the same test launch.sh:40 makes.
  if [ -d "$MODEL_DIR" ]; then
    # -L is required: everything under models--*/blobs is a symlink into the
    # shared content-addressed store at $HF_DIR/hub/blobs, so a plain `du -sh`
    # on the model directory reads ~1.3 MB and looks like a failed download.
    log "  present: $MODEL_DIR ($(du -shL "$MODEL_DIR" 2>/dev/null | cut -f1 || echo '?'))"
    return 0
  fi

  # Re-downloading IS the optimal recovery: measured 2.23 GB/s (~700s). Ingress
  # is unmetered and uncapped while egress is capped at 10 Gbps AND billed, so
  # every "cache it somewhere" scheme pays >=1249s outbound to build a cache
  # that at best ties a free 700s download (cloud-init.yaml:31-36).
  log "  downloading $MODEL into $HF_DIR (~1.5 TB, ~700s at the measured 2.23 GB/s)"
  [ -n "${HF_TOKEN:-}" ] || warn "HF_TOKEN unset. TODO(operator): export HF_TOKEN if the repo is gated (docs §4.2 passes one)"

  [ -x "$HFV/bin/hf" ] || {
    run python3 -m venv "$HFV"
    run "$HFV/bin/pip" install -q -U "huggingface_hub[cli]"
  }
  # HF_XET_HIGH_PERFORMANCE raises max download concurrency 64 -> 124
  # (cloud-init.yaml:37). Note docs §4.2 records that it was NOT needed to hit
  # 2.23 GB/s; it is kept because cloud-init sets it and it is harmless.
  if [ "$DRY_RUN" = 1 ]; then
    printf '        + HF_HOME=%s HF_XET_HIGH_PERFORMANCE=1 %s/bin/hf download %s\n' "$HF_DIR" "$HFV" "$MODEL"
  else
    HF_HOME="$HF_DIR" HF_XET_HIGH_PERFORMANCE=1 "$HFV/bin/hf" download "$MODEL"
  fi
  [ "$DRY_RUN" = 1 ] || [ -d "$MODEL_DIR" ] || die "download finished but $MODEL_DIR is absent"
}

# ---------------------------------------------------------------------------
stage_secrets() {
  step "secrets"
  # Nothing in this stage is ever printed. Only "generated"/"reusing".
  umask 077

  local keyfile="$REPO/api-key.txt" pwfile="$REPO/dash-password.txt"
  local envfile="$REPO/vllm-k3.env" envex="$REPO/vllm-k3.env.example"

  if [ -s "$keyfile" ]; then
    log "  reusing existing api-key.txt"
  else
    # Same shape as the live key: sk-k3- + 48 hex.
    if [ "$DRY_RUN" = 1 ]; then printf '        + write %s (sk-k3-<48 hex>, mode 600)\n' "$keyfile"
    else printf 'sk-k3-%s\n' "$(openssl rand -hex 24)" > "$keyfile"; fi
    log "  generated api-key.txt"
  fi
  run chmod 600 "$keyfile"

  if [ -s "$pwfile" ]; then
    log "  reusing existing dash-password.txt"
  else
    if [ "$DRY_RUN" = 1 ]; then printf '        + write %s (20 chars base64, mode 600)\n' "$pwfile"
    else openssl rand -base64 15 > "$pwfile"; fi
    log "  generated dash-password.txt"
  fi
  run chmod 600 "$pwfile"

  # vllm-k3.env is built from its committed twin so the AITER flags can never
  # drift. Do not hand-edit it: AITER_SITUV2_A8W4 without
  # AITER_BF16_FP8_MOE_BOUND=0 produces FLUENT BUT WRONG output at full speed
  # (vllm-k3.env.example:7-9, docs §7.3).
  [ -f "$envex" ] || die "missing $envex - cannot build vllm-k3.env without its template"
  local key cur=""
  if [ -s "$keyfile" ]; then key="$(tr -d '\r\n' < "$keyfile")"; else key="sk-k3-DRYRUN"; fi
  if [ -f "$envfile" ]; then cur="$(sed -n 's/^VLLM_API_KEY=//p' "$envfile" | tail -1)"; fi
  if [ "$cur" != "$key" ]; then
    [ -z "$cur" ] || warn "VLLM_API_KEY in vllm-k3.env did not match api-key.txt - rewriting (this WAS a silent 401)"
    # sed the placeholder line rather than appending, so the file keeps exactly
    # one VLLM_API_KEY (docker --env-file takes the last occurrence, which makes
    # a duplicate line an invisible override).
    if [ "$DRY_RUN" = 1 ]; then
      printf '        + write %s from vllm-k3.env.example with VLLM_API_KEY from api-key.txt (mode 600)\n' "$envfile"
    else
      sed "s|^VLLM_API_KEY=.*|VLLM_API_KEY=${key}|" "$envex" > "$envfile"
      grep -q "^VLLM_API_KEY=${key}$" "$envfile" \
        || die "failed to substitute VLLM_API_KEY into $envfile"
    fi
    log "  wrote vllm-k3.env (VLLM_API_KEY := api-key.txt)"
    # Rotating the upstream credential needs a vLLM restart, unlike customer
    # keys (vllm-k3.env.example:19-22). Left to the operator: 6 min of downtime.
    if systemctl is-active --quiet k3.service 2>/dev/null; then
      RESTART_HINTS+=("k3.service (VLLM_API_KEY changed)")
    fi
  else
    log "  vllm-k3.env already carries the api-key.txt credential"
  fi
  run chmod 600 "$envfile"

  # Dashboard basic auth. nginx reads $HTPASSWD as the www-data worker
  # (nginx.conf:1 `user www-data`), hence group www-data and mode 640; the
  # live file is exactly that. It is also an $apr1$ hash, which openssl can
  # produce without pulling in apache2-utils.
  local pw="" need_htpasswd=1
  if [ -s "$pwfile" ]; then pw="$(tr -d '\r\n' < "$pwfile")"; fi
  if [ -s "$HTPASSWD" ] && [ -n "$pw" ]; then
    # Reproduce the stored hash from its own salt; equal means the file already
    # encodes this password, so leave it alone rather than churning the salt.
    local stored salt
    stored="$(sed -n "s/^${DASH_USER}://p" "$HTPASSWD" | head -1)"
    salt="$(printf '%s' "$stored" | cut -d'$' -f3)"
    if [ -n "$stored" ] && [ -n "$salt" ] \
       && [ "$(openssl passwd -apr1 -salt "$salt" "$pw" 2>/dev/null)" = "$stored" ]; then
      need_htpasswd=0
      log "  reusing existing $HTPASSWD (verified against dash-password.txt)"
    else
      warn "$HTPASSWD does not match dash-password.txt for user '$DASH_USER' - regenerating"
    fi
  fi
  if [ "$need_htpasswd" = 1 ]; then
    if [ "$DRY_RUN" = 1 ]; then
      printf '        + write %s for user %s (apr1, mode 640 root:www-data)\n' "$HTPASSWD" "$DASH_USER"
    else
      printf '%s:%s\n' "$DASH_USER" "$(openssl passwd -apr1 "$pw")" > "$HTPASSWD"
      chown root:www-data "$HTPASSWD"
      chmod 640 "$HTPASSWD"
    fi
    log "  generated $HTPASSWD for user '$DASH_USER'"
  fi

  # customers.tsv is seeded here (not in `keys`) so gen-keys.sh always finds a
  # file; it would otherwise create a header-only one itself (gen-keys.sh:43).
  # Comment lines from the example are skipped by the generator's `''|\#*` case,
  # so seeding it does NOT mint a fake customer.
  local tsv="$REPO/customers.tsv"
  if [ -s "$tsv" ]; then
    log "  reusing existing customers.tsv"
  else
    run install -m 600 "$REPO/customers.tsv.example" "$tsv"
    log "  seeded customers.tsv from the example (no live keys yet; use ./gen-keys.sh add <name>)"
  fi
  run chmod 600 "$tsv"
  # Back to the default: umask 077 above must not leak into later stages, or
  # `install`'s directories and /scratch/hf/config.yaml would come out 700/600
  # and the container could not read its own serve config.
  umask 022
}

# ---------------------------------------------------------------------------
stage_install() {
  step "install repo -> system"
  local changed

  # The container sees /scratch/hf as /hf and is started with
  # `--config /hf/config.yaml` (k3.service:34,39), so this path is load-bearing.
  # Installed unconditionally because the repo is the source of truth - but a
  # change only takes effect on restart, which this script will not do for you.
  changed=0; sync_file "$REPO/config.yaml" "$HF_DIR/config.yaml" 644 || changed=1
  if [ "$changed" = 1 ] && systemctl is-active --quiet k3.service 2>/dev/null; then
    RESTART_HINTS+=("k3.service (config.yaml changed)")
  fi

  changed=0
  sync_file "$REPO/dashboard/server.py"  "$DASH_DIR/server.py"  644 || changed=1
  sync_file "$REPO/dashboard/index.html" "$DASH_DIR/index.html" 644 || changed=1
  if [ "$changed" = 1 ] && systemctl is-active --quiet k3dash.service 2>/dev/null; then
    RESTART_HINTS+=("k3dash.service (dashboard files changed)")
  fi

  changed=0
  sync_file "$REPO/k3.service"              /etc/systemd/system/k3.service     644 || changed=1
  sync_file "$REPO/systemd/k3dash.service"  /etc/systemd/system/k3dash.service 644 || changed=1
  if [ "$changed" = 1 ]; then run systemctl daemon-reload; fi

  # Tests land flat in /scratch because gate.sh/gate.py hardcode those paths
  # (see TEST_DIR above). gate-baseline.json is only seeded, never overwritten:
  # `gate.sh --baseline` re-records it on this box and the repo copy is the
  # bring-up default, so clobbering would erase a deliberate re-baseline.
  local t
  for t in gate.py gate.sh agentic-test.py ctx-test.py ide-ready.py; do
    sync_file "$REPO/tests/$t" "$TEST_DIR/$t" 755 || true
  done
  if [ -f "$TEST_DIR/gate-baseline.json" ]; then
    log "  unchanged $TEST_DIR/gate-baseline.json (kept; --baseline is operator state)"
  else
    run install -m 644 "$REPO/tests/gate-baseline.json" "$TEST_DIR/gate-baseline.json"
    log "  seeded $TEST_DIR/gate-baseline.json"
  fi

  # --- nginx fragments, with rollback -------------------------------------
  # Debian's stock site also declares `listen 80 default_server`, which
  # collides with 00-default-deny.conf and makes nginx -t fail with "duplicate
  # default server". The live box has an empty sites-enabled for this reason.
  if [ -L /etc/nginx/sites-enabled/default ]; then
    warn "disabling Debian's default site: it declares a second default_server on 80 and breaks nginx -t"
    run unlink /etc/nginx/sites-enabled/default
  fi

  # Two lists so the rollback is complete: overwritten files are restored from
  # their backup, newly created ones have to be removed again or a fragment
  # that fails nginx -t would stay behind and keep the config broken.
  local f base backups=() added=() ngx_changed=0
  for f in "$REPO"/nginx/*.conf "$REPO"/nginx/*.inc; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    if [ ! -f "$NGINX_DIR/$base" ]; then
      added+=("$base")
    elif differs "$f" "$NGINX_DIR/$base"; then
      # ".pre-deploy.bak" does not end in .conf, so nginx's
      # `include conf.d/*.conf` (nginx.conf:59) never picks these up.
      run cp -p "$NGINX_DIR/$base" "$NGINX_DIR/$base.pre-deploy.bak"
      backups+=("$base")
    fi
    # k3.conf is 640 in the repo and live; the fragments that hold no
    # credentials are 644. gen-keys.sh's generated pair is 640 for real - they
    # embed cleartext keys - and is not installed from here.
    case "$base" in
      k3.conf) sync_file "$f" "$NGINX_DIR/$base" 640 || ngx_changed=1 ;;
      *)       sync_file "$f" "$NGINX_DIR/$base" 644 || ngx_changed=1 ;;
    esac
  done

  # Reload nginx after certbot renews, or renewal succeeds silently while nginx
  # keeps serving the certificate it parsed at its last reload - and clients get
  # TLS errors on expiry day despite a valid cert being on disk. issue-cert.sh
  # uses `certonly --webroot`, so the renewal config has no installer to do this.
  run mkdir -p "$RENEWAL_HOOK_DIR"
  sync_file "$REPO/nginx/certbot-deploy-hook.sh" "$RENEWAL_HOOK_DIR/reload-nginx.sh" 755 || true

  # The usage log is the billing record, so it gets its own directory and its
  # own retention. Under /var/log/nginx/ the package's logrotate glob claimed it
  # at rotate 14 and there is no way to override a path logrotate already
  # manages (nginx/k3-usage.logrotate explains this at length).
  run mkdir -p "$USAGE_LOG_DIR"
  run chown root:adm "$USAGE_LOG_DIR"
  run chmod 750 "$USAGE_LOG_DIR"
  sync_file "$REPO/nginx/k3-usage.logrotate" /etc/logrotate.d/k3-usage 644 || true

  if [ "$DRY_RUN" = 0 ] && [ "$ngx_changed" = 1 ]; then
    if nginx_ok; then
      for base in "${backups[@]+"${backups[@]}"}"; do rm -f "$NGINX_DIR/$base.pre-deploy.bak"; done
      # A fresh box has nginx installed but not started; `services` enables it.
      # `systemctl reload` on a dead unit fails, so do not pretend otherwise.
      if systemctl is-active --quiet nginx.service; then
        nginx_reload "nginx fragments updated" || die "nginx reload failed after a valid -t"
      else
        log "  nginx config valid; nginx is not running yet (the services stage starts it)"
      fi
    else
      warn "nginx -t failed after installing fragments - rolling back"
      for base in "${backups[@]+"${backups[@]}"}"; do mv -f "$NGINX_DIR/$base.pre-deploy.bak" "$NGINX_DIR/$base"; done
      for base in "${added[@]+"${added[@]}"}"; do rm -f "$NGINX_DIR/$base"; done
      nginx -t 2>&1 | tail -5 >&2
      die "nginx fragments rolled back; the running config was never replaced"
    fi
  fi
}

# ---------------------------------------------------------------------------
stage_image() {
  step "image"
  # Straight port of launch.sh:15-37. Verify the BITS: the `kimi-k3` tag is a
  # dev tag that may be garbage-collected upstream, so a matching name proves
  # nothing (launch.sh:10-11).
  local id
  have_id() { docker image inspect "$1" --format '{{.Id}}' 2>/dev/null || true; }
  id="$(have_id "$IMAGE_TAG")"

  if [ "$id" != "$IMAGE_DIGEST" ]; then
    if [ -f "$IMAGE_TARBALL" ]; then
      log "  restoring from $IMAGE_TARBALL ($(du -h "$IMAGE_TARBALL" | cut -f1))"
      run docker load -i "$IMAGE_TARBALL"
    else
      log "  no local tarball; pulling by digest"
      run docker pull "${IMAGE_REPO}@${IMAGE_DIGEST}" \
        || run docker pull "$IMAGE_TAG"
      # A digest pull leaves the image untagged, and k3.service is fine with
      # that (it references the digest) but launch.sh and the tarball workflow
      # both key off the tag.
      run docker tag "${IMAGE_REPO}@${IMAGE_DIGEST}" "$IMAGE_TAG" 2>/dev/null || true
    fi
    id="$(have_id "$IMAGE_TAG")"
  fi

  if [ "$DRY_RUN" = 1 ] && [ -z "$id" ]; then
    log "  (dry-run) would verify $IMAGE_TAG resolves to $IMAGE_DIGEST"
    return 0
  fi
  # Fail closed rather than serving unknown bits (launch.sh:30).
  [ "$id" = "$IMAGE_DIGEST" ] || die "image ID mismatch.
       expected $IMAGE_DIGEST
       got      ${id:-<absent>}"
  log "  image verified by ID: $id"

  # k3.service:37 runs `vllm/vllm-openai-rocm@sha256:...`, so the DIGEST
  # REFERENCE - not just the ID - has to resolve locally. `docker save <digest>`
  # produces an archive whose restored image has RepoTags None and no
  # RepoDigests (docs, "Image backup" mistake 2), and the unit then fails to
  # start on a box where the ID is nonetheless correct. Catch it here, not at
  # 03:00 during a recovery.
  if [ -z "$(have_id "${IMAGE_REPO}@${IMAGE_DIGEST}")" ]; then
    warn "the digest reference ${IMAGE_REPO}@${IMAGE_DIGEST} does not resolve locally"
    warn "  k3.service:37 uses exactly that reference, so the unit will fail to start."
    warn "  fix: docker pull ${IMAGE_REPO}@${IMAGE_DIGEST}   (re-attaches RepoDigests)"
    die "refusing to continue with an image reference k3.service cannot resolve"
  fi
  log "  digest reference resolves (this is what k3.service:37 asks docker for)"
}

# ---------------------------------------------------------------------------
stage_keys() {
  step "keys"
  # 00-k3-keys.conf and k3-limits.inc are GENERATED and never committed: they
  # hold every customer key plus the upstream key in cleartext (.gitignore:22-25).
  # gen-keys.sh runs nginx -t and reloads on its own (gen-keys.sh:117-121).
  [ -x "$REPO/gen-keys.sh" ] || die "missing $REPO/gen-keys.sh"
  [ -s "$REPO/api-key.txt" ] || die "api-key.txt is empty - run the secrets stage first (gen-keys.sh:20 reads it)"
  run "$REPO/gen-keys.sh"
  if [ "$DRY_RUN" = 0 ]; then
    [ -s "$NGINX_DIR/00-k3-keys.conf" ] || die "gen-keys.sh produced no $NGINX_DIR/00-k3-keys.conf"
    [ -s "$NGINX_DIR/k3-limits.inc" ]   || die "gen-keys.sh produced no $NGINX_DIR/k3-limits.inc"
    local n
    n="$(grep -c '^    "Bearer ' "$NGINX_DIR/00-k3-keys.conf" || true)"
    [ "${n:-0}" -gt 0 ] || warn "no customer keys in the map: every /v1/ request will 401 until \`./gen-keys.sh add <name>\`"
  fi
}

# ---------------------------------------------------------------------------
stage_services() {
  step "services"
  run systemctl daemon-reload

  # docker.service is a hard Requires= of k3.service (k3.service:9).
  run systemctl enable --now docker.service >/dev/null 2>&1 || true
  # `enable --now` is intentional over `restart`: on an already-active unit it
  # does nothing, so re-running this script never costs the ~6 min reload.
  run systemctl enable --now k3.service
  run systemctl enable --now k3dash.service
  run systemctl enable --now nginx.service
  # The cert is short-lived; without the timer the endpoint dies in 90 days.
  run systemctl enable --now certbot.timer >/dev/null 2>&1 || true

  if [ "$DRY_RUN" = 1 ]; then
    printf '        + poll http://127.0.0.1:%s/v1/models until 200 (max %ss)\n' "$VLLM_PORT" "$READY_TIMEOUT"
    return 0
  fi

  # /v1/models needs the key even on loopback: vLLM enforces VLLM_API_KEY
  # itself (verify-auth.sh:63 asserts a bare direct call is 401), which makes
  # this poll double as a check that the two copies of the credential agree.
  local key; key="$(tr -d '\r\n' < "$REPO/api-key.txt")"
  local url="http://127.0.0.1:${VLLM_PORT}/v1/models"
  local t0 elapsed code=000
  t0="$(date +%s)"
  log "  waiting for the engine on :$VLLM_PORT (load is ~6 min: 140s weights + 74s engine init)"
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" -H "Authorization: Bearer $key" || true)"
    elapsed=$(( $(date +%s) - t0 ))
    if [ "$code" = "200" ]; then log "  engine ready after ${elapsed}s"; break; fi
    if [ "$code" = "401" ]; then
      die "vLLM answered 401 on loopback: VLLM_API_KEY in vllm-k3.env does not match api-key.txt.
       Re-run \`$0 secrets\`, then \`systemctl restart k3.service\` (the container reads it at start)."
    fi
    # A unit that has given up will not come back on its own: k3.service stops
    # retrying after 5 failures in 30 min (k3.service:12-13).
    if ! systemctl is-active --quiet k3.service; then
      warn "k3.service is $(systemctl is-active k3.service 2>&1) - last 20 log lines:"
      journalctl -u k3.service -n 20 --no-pager >&2 || true
      die "k3.service is not running; the engine will not come up"
    fi
    if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
      die "engine not ready after ${elapsed}s (last http $code). \`docker logs k3\` has the reason."
    fi
    printf '        %4ds  http %s  %s\n' "$elapsed" "$code" \
      "$(docker logs k3 2>&1 | grep -oE 'Loading weights took [0-9.]+ seconds|GPU KV cache size: [0-9,]+ tokens' | tail -1)"
    sleep "$READY_POLL"
  done

  # The dashboard is a monitor, so its unit only Wants= the engine
  # (k3dash.service:7-8) and starts even when k3 is down - check it separately.
  code="$(http_code "http://127.0.0.1:${DASH_PORT}/api/health" 15)"
  if [ "$code" = "200" ]; then log "  dashboard healthy on :$DASH_PORT"
  else warn "dashboard on :$DASH_PORT answered $code (want 200); journalctl -u k3dash.service"; fi
}

# ---------------------------------------------------------------------------
stage_tls() {
  step "tls"
  [ -x "$REPO/issue-cert.sh" ] || die "missing $REPO/issue-cert.sh"

  # issue-cert.sh is safe to re-run and no-ops unless DNS already resolves to
  # $EXPECT_IP, because a failed HTTP-01 burns Let's Encrypt rate limit
  # (5 failures/hostname/hour, issue-cert.sh:4-5). Do not add a --force path.
  if [ -s "$CERT_LIVE/fullchain.pem" ] && [ -f "$NGINX_DIR/k3-tls.conf" ]; then
    log "  cert and k3-tls.conf already in place; expires $(openssl x509 -enddate -noout -in "$CERT_LIVE/fullchain.pem" 2>/dev/null | cut -d= -f2)"
    log "  renewal is certbot.timer's job (is-enabled: $(systemctl is-enabled certbot.timer 2>/dev/null || echo unknown))"
    # Still re-validate: `keys` may have changed the map k3-tls.conf reads.
    nginx_ok || die "nginx -t fails with the current config - fix before reloading"
    return 0
  fi

  if run "$REPO/issue-cert.sh" "${TLS_ARGS[@]+"${TLS_ARGS[@]}"}"; then
    log "  tls in place"
  else
    # Exit 1 here is usually just "DNS is not pointing at us yet", which is a
    # legitimate state on a fresh droplet - but it is NOT success, so the run
    # is marked incomplete rather than reported green.
    TLS_DEFERRED=1
    warn "issue-cert.sh did not complete. Most often DNS for $DOMAIN does not resolve to $EXPECT_IP yet."
    warn "  finish later with: $0 tls --watch   (polls every 60s for up to 4h)"
  fi
}

# ---------------------------------------------------------------------------
stage_verify() {
  step "verify"
  local rc=0

  # verify-auth.sh now targets https://$DOMAIN and pins the hostname to
  # loopback with curl --resolve, so it exercises the real server block before
  # DNS propagates. It needs a cert and a customer key; without either it
  # exits non-zero for environmental reasons rather than finding a fault, so
  # gate on both rather than reporting a spurious failure.
  if [ ! -s "$CERT_LIVE/fullchain.pem" ]; then
    warn "no cert at $CERT_LIVE/fullchain.pem - skipping verify-auth.sh (run the tls stage first)"
  elif ! awk -F'\t' '!/^[[:space:]]*#/ && NF>=2 {found=1} END{exit !found}' \
         "$REPO/customers.tsv" 2>/dev/null; then
    warn "no customer key in customers.tsv - skipping verify-auth.sh; run: ./gen-keys.sh add <name>"
  elif [ "$DRY_RUN" = 1 ]; then
    printf '        + verify-auth.sh against https://%s\n' "$DOMAIN"
  else
    log "  running verify-auth.sh against https://$DOMAIN"
    DOMAIN="$DOMAIN" run "$REPO/verify-auth.sh" || rc=1
  fi

  # The gate is the only check that catches fluent-but-wrong output: several
  # configs start cleanly, serve at full speed and emit nonsense (docs §6, §7.3).
  # Large-batch benchmarks do not expose it.
  log "  running the correctness gate (~21s)"
  run bash "$REPO/tests/gate.sh" || rc=1

  [ "$rc" = 0 ] || die "verification failed (see above)"
  log "verification passed"
}

# ---------------------------------------------------------------------------
summary() {
  local key_hint="./gen-keys.sh add <name>"
  printf '\n'
  printf '=========================================================================\n'
  printf ' kimi-k3 deployment summary\n'
  printf '=========================================================================\n'
  printf '  api endpoint      https://%s/v1        (model: %s)\n' "$DOMAIN" "$SERVED_NAME"
  printf '  dashboard         https://%s/          (basic auth user: %s,\n' "$DOMAIN" "$DASH_USER"
  printf '                                             password in %s/dash-password.txt)\n' "$REPO"
  printf '  add a customer    cd %s && %s\n' "$REPO" "$key_hint"
  printf '                    mints a key, regenerates the nginx map and reloads;\n'
  printf '                    no vLLM restart needed. revoke with: ./gen-keys.sh revoke <name>\n'
  printf '  logs              docker logs -f k3                 engine\n'
  printf '                    journalctl -fu k3dash.service     dashboard\n'
  printf '                    %-33s per-customer access (cust=<name>)\n' "$USAGE_LOG"
  printf '                    /var/log/nginx/error.log          edge errors\n'
  printf '  health            %-33s correctness gate\n' "$TEST_DIR/gate.sh"
  printf '                    %-33s endpoint + KV cache\n' "make health"
  printf '  reference         %s/docs/K3-DEPLOYMENT.md\n' "$REPO"
  if [ "${#RESTART_HINTS[@]}" -gt 0 ]; then
    printf '\n  ACTION REQUIRED - installed files changed under running units:\n'
    local h
    for h in "${RESTART_HINTS[@]}"; do printf '    %s\n' "$h"; done
    printf '    Not restarted automatically: k3.service costs ~6 min of downtime.\n'
    printf '    When you are ready:  systemctl restart k3.service\n'
  fi
  if [ "$TLS_DEFERRED" = 1 ]; then
    printf '\n  INCOMPLETE - no TLS. Run: %s tls --watch\n' "$0"
  fi
  printf '=========================================================================\n'
}

# ---------------------------------------------------------------------------
main() {
  local stages=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --watch)   TLS_ARGS+=(--watch) ;;
      -h|--help) usage; exit 0 ;;
      all)       stages+=("${STAGES_ALL[@]}") ;;
      preflight|host|weights|secrets|install|image|keys|services|tls|verify) stages+=("$1") ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
    shift
  done
  [ "${#stages[@]}" -gt 0 ] || stages=("${STAGES_ALL[@]}")

  # Root is not negotiable: this writes /etc/nginx, /etc/systemd/system and
  # /etc/letsencrypt, and talks to the docker socket.
  [ "$(id -u)" -eq 0 ] || die "must run as root"

  [ -d "$REPO" ] || die "repo not found at $REPO"

  log "deploy.sh starting"
  if [ "$DRY_RUN" = 1 ]; then log "DRY RUN - no changes will be made (read-only probes still run)"; fi
  log "stages: ${stages[*]}"

  local s
  for s in "${stages[@]}"; do "stage_$s"; done

  summary
  [ "$TLS_DEFERRED" = 0 ] || exit 1
}

main "$@"
