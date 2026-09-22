#!/usr/bin/env bash
#
# Production deployer for the capacity-first Kimi-K3 stack.
# Implements docs/SYSTEM-DESIGN.md 4 on a single 8x MI355X node.
#
# Self-contained: this script depends on nothing from the pre-existing vLLM
# deployment. The branch stands up its own stack from a bare host.
#
#   ./deploy.sh                    all stages, baseline profile
#   ./deploy.sh --dry-run          print every mutation, perform none
#   ./deploy.sh engine verify      run named stages only
#   PROFILE=dedup ./deploy.sh engine verify
#
# Stages, in order:
#   preflight  refuse to proceed unless the host can actually serve this model
#   host       directories, kernel tuning, docker
#   weights    fetch the model, resumable, idempotent
#   engine     validate the profile against the image, render unit, start
#   gateway    tenancy + backpressure + trace capture
#   ops        first-boot B0/Z5, D3 distill, E2 scrape, Z3 traffic timer
#   keys       rebuild the per-customer auth layer from the customer table
#   edge       nginx, TLS, default-deny
#   verify     liveness, correctness gate, capacity check against the model
#
# Every stage is idempotent. Re-running a completed stage costs nothing.
#
# GPU COST: this box bills from the moment it boots. Run --dry-run first, and
# read docs/SYSTEM-DESIGN.md 11 "GPU cost discipline" before the first real run.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROFILE_DIR="${SCRIPT_DIR}/profiles"

readonly ALL_STAGES=(preflight host weights engine gateway ops edge keys verify)

# --- Deployment constants ----------------------------------------------------
readonly MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K3}"
readonly SCRATCH="${SCRATCH:-/scratch}"
readonly HF_HOME="${SCRATCH}/hf"
readonly STATE_DIR="${SCRATCH}/deploy-state"
readonly TRACE_DIR="${SCRATCH}/traces"

readonly ENGINE_CONTAINER="k3-engine"
readonly ENGINE_HOST=127.0.0.1
readonly ENGINE_PORT=8001
readonly GATEWAY_PORT=8002

# Pin the image by digest. A floating tag silently changes the AITER contract
# between deploys, which is the documented route to the silent-garbage failure.
readonly ENGINE_IMAGE="${ENGINE_IMAGE:-}"

# Hardware this design assumes. Preflight refuses anything else rather than
# discovering the mismatch after a 1.5 TB download.
readonly REQUIRED_GPUS=8
readonly REQUIRED_RAM_GB=1800
readonly REQUIRED_DISK_GB=1900

PROFILE="${PROFILE:-baseline}"
DRY_RUN=0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

log()   { printf '  %s\n' "$*"; }
info()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()    { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn()  { printf '  \033[33mWARN\033[0m  %s\n' "$*" >&2; }
die()   { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; exit 1; }

# Every mutation goes through this. In dry-run it prints and returns success,
# so the whole plan can be inspected without touching the host.
# Secrets must not reach scrollback. The docs tell operators to run --dry-run
# first, so an un-redacted echo puts HF_TOKEN in every terminal and CI log.
redact() {
  sed -E 's/((HF_TOKEN|[A-Z_]*_?(TOKEN|SECRET|PASSWORD|API_KEY))=)[^ ]*/\1<redacted>/g'
}

run() {
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m %s\n' "$*" | redact
    return 0
  fi
  "$@"
}

# Marks a stage complete so a re-run can skip the expensive parts.
stage_done()    { [[ -f "${STATE_DIR}/$1.done" ]]; }
mark_done()     { run mkdir -p "${STATE_DIR}"; run touch "${STATE_DIR}/$1.done"; }

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

# Settings a caller may override from the environment for a single run. The
# profile is sourced, so without this it would silently clobber them.
readonly OVERRIDABLE=(MAX_RUNNING_REQUESTS MAX_MODEL_LEN LONG_PREFILL_THRESHOLD DP_SIZE MEM_FRACTION_STATIC)

load_profile() {
  local path="${PROFILE_DIR}/${PROFILE}.env"
  [[ -f "$path" ]] || die "no such profile: ${PROFILE} (looked in ${PROFILE_DIR})"

  local key saved=""
  for key in "${OVERRIDABLE[@]}"; do
    [[ -n "${!key:-}" ]] && saved+="${key}=${!key}"$'\n'
  done

  # shellcheck disable=SC1090
  source "$path"

  local entry
  while IFS= read -r entry; do
    [[ -n "$entry" ]] || continue
    export "${entry?}"
    warn "${entry%%=*} overridden from the environment: ${entry#*=}"
  done <<< "$saved"

  log "profile ${PROFILE_NAME} loaded from ${path}"
}

# The admission ceiling is derived, never hardcoded, so it always matches the KV
# pool the chosen profile actually produces. B1 in the register.
derive_admission_ceiling() {
  local ids=""
  (( ENABLE_DP_ATTENTION )) && ids="A1"
  [[ "${KV_CACHE_DTYPE}" == fp8* ]] && ids="${ids:+${ids},}A2"

  python3 - "$ids" <<'PY'
import json, subprocess, sys

wanted = tuple(x for x in sys.argv[1].split(",") if x)
raw = subprocess.check_output(
    [sys.executable, "-m", "redesign.capacity", "--json"], text=True
)
for scenario in json.loads(raw)["scenarios"]:
    if tuple(scenario["register_ids"]) == wanted:
        # recommended_admission is max(zero-sharing floor, observed peak).
        # Using the floor alone was the withdrawn B1 advice: 75 as a live
        # ceiling against a box that had already sustained 427.
        print(scenario["recommended_admission"])
        break
else:
    sys.exit(f"capacity model has no scenario for {wanted}")
PY
}

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

stage_preflight() {
  info "preflight"

  [[ -n "${ENGINE_IMAGE}" ]] || die \
    "ENGINE_IMAGE is unset. Pin the SGLang MI35x image by digest:
         ENGINE_IMAGE=repo/sglang@sha256:... ./deploy.sh
       A floating tag changes the AITER contract between deploys, which is the
       documented route to the silent-garbage failure."

  command -v docker >/dev/null || die "docker not found"
  command -v python3 >/dev/null || die "python3 not found"

  # The host probes below are Linux- and ROCm-specific. A dry run is meant to
  # be inspectable from a laptop, so there it reports what it WOULD check
  # rather than failing on a machine that was never going to serve this model.
  if (( DRY_RUN )); then
    log "would require ${REQUIRED_GPUS} GPUs, >=${REQUIRED_RAM_GB} GB RAM, "\
        ">=${REQUIRED_DISK_GB} GB free on $(dirname "${SCRATCH}")"
  else
    # grep -c prints 0 and exits 1 when nothing matches, so `|| echo 0` would
    # append a second zero and corrupt the arithmetic below.
    local gpus=0
    command -v rocm-smi >/dev/null \
      || die "rocm-smi not found; this is not an Instinct host"
    gpus=$(rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l)
    if (( ${gpus:-0} != REQUIRED_GPUS )); then
      die "found ${gpus:-0} GPUs, this design requires ${REQUIRED_GPUS}.
       1.5 TB of MXFP4 weights do not fit in fewer."
    fi
    ok "${gpus} GPUs visible"

    local ram_gb
    ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    (( ram_gb >= REQUIRED_RAM_GB )) \
      || warn "host RAM ${ram_gb} GB is below ${REQUIRED_RAM_GB} GB; A3 host tiering will be limited"
    ok "host RAM ${ram_gb} GB"

    local disk_gb
    disk_gb=$(df -BG --output=avail "$(dirname "${SCRATCH}")" | tail -1 | tr -dc '0-9')
    local weight_marker="${HF_HOME}/.complete-${MODEL_REPO//\//-}"
    if [[ -f "$weight_marker" ]]; then
      (( ${disk_gb:-0} >= 50 )) \
        || die "only ${disk_gb:-0} GB free; traces and snapshots need headroom"
      ok "${disk_gb} GB free (weights already on disk)"
    else
      (( ${disk_gb:-0} >= REQUIRED_DISK_GB )) \
        || die "only ${disk_gb:-0} GB free; the weights alone are ~1.5 TB"
      ok "${disk_gb} GB free"
    fi
  fi

  # The capacity model is the source of every admission number below. If its
  # own hypothesis test fails, the model does not describe this box and the
  # derived ceiling would be wrong.
  if ( cd "${REPO_ROOT}" && python3 -m redesign.capacity --verify >/dev/null ); then
    ok "capacity model self-check passes"
  else
    die "capacity model hypothesis test failed; do not deploy against it"
  fi

  mark_done preflight
}

# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------

stage_host() {
  info "host"

  # /scratch lives on the boot disk so a snapshot is self-contained and carries
  # the weights. Mounting a volume here would hide them.
  for dir in "${SCRATCH}" "${HF_HOME}" "${STATE_DIR}" "${TRACE_DIR}" /var/log/k3; do
    [[ -d "$dir" ]] && continue
    run mkdir -p "$dir"
    log "created $dir"
  done

  # Sized for a sustained multi-gigabyte-per-second pull of the weights.
  local entry key value current
  for entry in \
    "net.core.rmem_max=536870912" \
    "net.core.wmem_max=536870912" \
    "net.ipv4.tcp_rmem=4096 87380 536870912" \
    "net.ipv4.tcp_wmem=4096 65536 536870912"
  do
    key="${entry%%=*}"
    value="${entry#*=}"
    current=$(sysctl -n "$key" 2>/dev/null || echo "")
    [[ "$current" == "$value" ]] && continue
    run sysctl -qw "${key}=${value}"
  done
  ok "kernel network tuning applied"

  mark_done host
}

# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

stage_weights() {
  info "weights"

  # SGLang resolves a repo id against HF_HOME, so the engine is given
  # MODEL_REPO rather than a hand-built cache path. The previous
  # "/hf/hub/models--<org>--<name>" is the cache *container* directory: it holds
  # blobs/ and snapshots/ and no config.json, so the engine exits instantly and
  # verify then waits out its full timeout -- an hour of download plus fifteen
  # minutes of paid GPU to discover a path string.
  local marker="${HF_HOME}/.complete-${MODEL_REPO//\//-}"
  if [[ -f "$marker" ]]; then
    ok "weights already present, skipping ~1.5 TB pull"
    return 0
  fi

  log "fetching ${MODEL_REPO} into ${HF_HOME}"
  log "this is ~1.5 TB and is the longest step of a cold build"

  run docker run --rm \
    -e HF_HOME=/hf \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
    -v "${HF_HOME}:/hf" \
    "${ENGINE_IMAGE}" \
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_REPO}', max_workers=16, resume_download=True)
"

  run touch "$marker"
  ok "weights fetched"
  mark_done weights
}

# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

# Refuses to launch with a flag this image does not accept. Without this, an
# unknown flag is discovered after the weights load -- several minutes of paid
# GPU per mistake.
validate_profile_flags() {
  local -a required=(tp_size context_length max_running_requests chunked_prefill_size enable_metrics)
  (( ENABLE_DP_ATTENTION )) && required+=(enable_dp_attention dp_size)
  (( ENABLE_PRIORITY_SCHEDULING )) && required+=(
    enable_priority_scheduling
    schedule_policy
    schedule_low_priority_values_first
  )
  [[ "${KV_CACHE_DTYPE}" != auto ]] && required+=(kv_cache_dtype)
  (( ENABLE_HIERARCHICAL_CACHE )) && required+=(enable_hierarchical_cache hicache_ratio)
  [[ "${SGLANG_USE_AITER}" == 1 ]] && required+=(attention_backend)
  [[ -n "${MEM_FRACTION_STATIC:-}" ]] && required+=(mem_fraction_static)
  (( ${LANGUAGE_MODEL_ONLY:-0} )) && required+=(language_model_only)
  [[ -n "${CUDA_GRAPH_MAX_BS_DECODE:-}" ]] && required+=(cuda_graph_max_bs_decode)
  [[ -n "${MAMBA_SSM_DTYPE:-}" ]] && required+=(mamba_ssm_dtype)
  (( ${DISABLE_CUDA_GRAPH:-0} )) && required+=(disable_decode_cuda_graph disable_prefill_cuda_graph)

  log "validating ${#required[@]} flags against the image's own CLI surface"

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m probe image for: %s\n' "${required[*]}"
    return 0
  fi

  local surface
  surface=$(docker run --rm -v "${REPO_ROOT}:/repo" -w /repo "${ENGINE_IMAGE}" \
    python3 -m redesign.probe --engine sglang --json 2>/dev/null) \
    || die "could not probe ${ENGINE_IMAGE}"

  local missing
  missing=$(python3 - "$surface" "${required[*]}" <<'PY'
import json, sys
results = json.loads(sys.argv[1])
wanted = sys.argv[2].split()
present = {
    flag["spec"]["name"]
    for result in results if result["importable"]
    for flag in result["flags"] if flag["present"]
}
print(" ".join(f for f in wanted if f not in present))
PY
)
  [[ -z "$missing" ]] || die \
    "image ${ENGINE_IMAGE} does not accept: ${missing}
       Profile ${PROFILE} cannot run on this build. If enable_dp_attention is
       among these, register item A1 is unavailable and session G2 must not be
       booked -- see redesign/Z2-FINDINGS.md 6."
  ok "every profile flag is accepted by the image"
}

# DP attention floor-divides the server-wide ceiling across attention ranks. Too
# low a per-rank share and the hybrid state pool cannot serve any request, which
# presents as a total outage rather than a config error.
check_admission_floor_division() {
  local ceiling=$1
  (( ENABLE_DP_ATTENTION )) || return 0

  local per_rank=$(( ceiling / DP_SIZE ))
  if (( per_rank < 2 )); then
    die "max-running-requests ${ceiling} over dp_size ${DP_SIZE} leaves ${per_rank}
       per rank. SGLang floor-divides this ceiling, and too small a share means
       'Hybrid state cache is too small to serve any requests' -- a server that
       refuses everything while looking like a migration failure."
  fi
  ok "admission ${ceiling} over dp_size ${DP_SIZE} = ${per_rank} per rank"
}

# Populates the global ENGINE_ARGS. A global rather than a subshell because
# argument arrays do not survive a pipe, and bash 3.2 has no mapfile.
ENGINE_ARGS=()
build_engine_args() {
  local ceiling=$1
  ENGINE_ARGS=(
    --model-path "${MODEL_REPO}"
    --trust-remote-code
    --tp-size "${TP_SIZE}"
    --host "${ENGINE_HOST}"
    --port "${ENGINE_PORT}"
    --context-length "${MAX_MODEL_LEN}"
    --max-running-requests "${ceiling}"
    --chunked-prefill-size "${MAX_NUM_BATCHED_TOKENS}"
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
    --enable-metrics
  )
  if [[ "${SGLANG_USE_AITER}" == 1 ]]; then
    ENGINE_ARGS+=(--attention-backend aiter)
  fi
  if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
    ENGINE_ARGS+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
  fi
  if (( ${LANGUAGE_MODEL_ONLY:-0} )); then
    ENGINE_ARGS+=(--language-model-only)
  fi
  if [[ -n "${CUDA_GRAPH_MAX_BS_DECODE:-}" ]]; then
    ENGINE_ARGS+=(--cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}")
  fi
  if [[ -n "${MAMBA_SSM_DTYPE:-}" ]]; then
    ENGINE_ARGS+=(--mamba-ssm-dtype "${MAMBA_SSM_DTYPE}")
  fi
  if (( ${DISABLE_CUDA_GRAPH:-0} )); then
    ENGINE_ARGS+=(--disable-decode-cuda-graph --disable-prefill-cuda-graph)
  fi
  if (( ENABLE_DP_ATTENTION )); then
    ENGINE_ARGS+=(--enable-dp-attention --dp-size "${DP_SIZE}")
  fi
  # B2. Gateway Priority is 0=P0..3=P3 ("lower runs earlier"). This image
  # defaults to the opposite (higher integer first), so without the low-first
  # flag P3 distillation would preempt interactive keystrokes.
  if (( ENABLE_PRIORITY_SCHEDULING )); then
    ENGINE_ARGS+=(
      --enable-priority-scheduling
      --schedule-policy fcfs
      --schedule-low-priority-values-first
      --retraction-policy priority
      --default-priority-value 0
    )
  fi
  if [[ "${KV_CACHE_DTYPE}" != auto ]]; then
    ENGINE_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
  fi
  if (( ENABLE_HIERARCHICAL_CACHE )); then
    ENGINE_ARGS+=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO}")
  fi
}

stage_engine() {
  info "engine  (profile: ${PROFILE})"

  local ceiling="${MAX_RUNNING_REQUESTS}"
  if [[ -z "$ceiling" ]]; then
    ceiling=$( cd "${REPO_ROOT}" && derive_admission_ceiling )
    log "admission ceiling ${ceiling}, derived from redesign.capacity"
  else
    warn "admission ceiling ${ceiling} overrides the capacity model"
  fi
  check_admission_floor_division "$ceiling"

  validate_profile_flags

  build_engine_args "$ceiling"
  log "engine args: ${ENGINE_ARGS[*]}"

  # Re-running this stage must not cold-restart a healthy engine. The rendered
  # arguments plus the image digest are the identity of a launch; if a running
  # container already matches, there is nothing to do.
  local fingerprint
  fingerprint=$(printf '%s\n' "${ENGINE_IMAGE}" "${ENGINE_ARGS[@]}" | shasum -a 256 | cut -c1-16)
  local fingerprint_file="${STATE_DIR}/engine.fingerprint"

  if [[ -f "${fingerprint_file}" ]] \
     && [[ "$(cat "${fingerprint_file}" 2>/dev/null)" == "${fingerprint}" ]] \
     && [[ "$(docker inspect -f '{{.State.Running}}' "${ENGINE_CONTAINER}" 2>/dev/null)" == "true" ]]; then
    ok "engine already running with these exact arguments; leaving it alone"
    mark_done engine
    return 0
  fi

  if [[ "$(docker inspect -f '{{.State.Running}}' "${ENGINE_CONTAINER}" 2>/dev/null)" == "true" ]]; then
    warn "replacing a RUNNING engine; this is a cold reload of 1.5 TB of weights"
    # SIGKILL does not wait for the HIP context to release VRAM, so the
    # immediate re-launch can fail on a busy device. Stop politely first.
    run docker stop -t 60 "${ENGINE_CONTAINER}" || true
  fi
  run docker rm -f "${ENGINE_CONTAINER}" 2>/dev/null || true

  run docker run -d --name "${ENGINE_CONTAINER}" \
    --restart on-failure:2 \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 32g --network host \
    --security-opt seccomp=unconfined \
    -v "${HF_HOME}:/hf" \
    -e HF_HOME=/hf \
    -e SGLANG_USE_AITER="${SGLANG_USE_AITER}" \
    -e SGLANG_AITER_K3_OPT="${SGLANG_AITER_K3_OPT}" \
    -e AITER_FLYDSL_FORCE="${AITER_FLYDSL_FORCE}" \
    -e AITER_SITUV2_A8W4="${AITER_SITUV2_A8W4}" \
    -e PYTHONHASHSEED="${PYTHONHASHSEED}" \
    -e PYTHONUNBUFFERED=1 \
    -e SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 \
    "${ENGINE_IMAGE}" \
    python3 -m sglang.launch_server "${ENGINE_ARGS[@]}"

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m record engine fingerprint %s\n' "${fingerprint}"
  else
    mkdir -p "${STATE_DIR}"
    printf '%s' "${fingerprint}" > "${fingerprint_file}"
  fi

  log "engine starting; a cold load takes several minutes"
  mark_done engine
}

# ---------------------------------------------------------------------------
# gateway
# ---------------------------------------------------------------------------

stage_gateway() {
  info "gateway"

  # The tenancy layer carries register item D2, the largest customer-visible
  # defect: requests rejected because prompt + max_tokens exceeds the window.
  run install -d -m 0750 "${TRACE_DIR}"

  # The ceiling must match the engine's, or the two layers disagree about how
  # much work is in flight.
  local ceiling="${MAX_RUNNING_REQUESTS}"
  [[ -n "$ceiling" ]] || ceiling=$( cd "${REPO_ROOT}" && derive_admission_ceiling )

  # Only stamp a priority if the engine was actually launched able to honour
  # one. Sending a field the scheduler ignores looks like the class design is
  # working when it is not.
  local send_priority="${ENABLE_PRIORITY_SCHEDULING:-0}"

  # Prepare package and state ownership for the container launcher.
  local install_root=/usr/local/lib/k3
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m install redesign/ -> %s\n' "$install_root"
  else
    mkdir -p "${install_root}"
    rm -rf "${install_root}/redesign"
    cp -a "${REPO_ROOT}/redesign" "${install_root}/redesign"
  fi
  if ! id -u k3 >/dev/null 2>&1; then
    run useradd --system --home /nonexistent --shell /usr/sbin/nologin k3 || true
  fi
  if id -u k3 >/dev/null 2>&1; then
    run chown -R k3:k3 "${TRACE_DIR}" "${install_root}" 2>/dev/null || true
  else
    warn "k3 user missing; verify the container UID and state-directory permissions"
  fi

  local unit=/etc/systemd/system/k3-gateway.service
  local rendered
  rendered=$(cat <<UNIT
[Unit]
Description=Kimi-K3 tenancy and backpressure gateway
After=docker.service k3.service
Requires=docker.service
Wants=k3.service

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=${install_root}
Environment=K3_ENGINE_URL=http://${ENGINE_HOST}:${ENGINE_PORT}
Environment=K3_MAX_MODEL_LEN=${MAX_MODEL_LEN}
Environment=K3_ADMISSION_CEILING=${ceiling}
Environment=K3_SEND_PRIORITY=${send_priority}
Environment=K3_TRACE_PATH=${TRACE_DIR}/requests.jsonl
Environment=K3_OFFBOX_URL=${K3_OFFBOX_URL:-}
Environment=K3_OFFBOX_API_KEY=${K3_OFFBOX_API_KEY:-}
Environment=K3_OFFBOX_MODEL=${K3_OFFBOX_MODEL:-offbox}
Environment=K3_BATCH_CUSTOMERS=${K3_BATCH_CUSTOMERS:-}
EnvironmentFile=-${SCRATCH}/deploy/offbox.env
ExecStartPre=-/usr/bin/docker rm -f k3-gateway
ExecStart=${install_root}/redesign/deploy/containers/run.sh k3-gateway \
  python3 -m redesign.gateway.server --port ${GATEWAY_PORT}
ExecStop=/usr/bin/docker stop -t 10 k3-gateway
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT
)

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m write %s\n' "$unit"
  else
    printf '%s\n' "$rendered" > "$unit"
  fi

  run systemctl daemon-reload
  run systemctl enable --now k3-gateway.service
  ok "gateway on :${GATEWAY_PORT}, traces to ${TRACE_DIR}/requests.jsonl"

  mark_done gateway
}

# ---------------------------------------------------------------------------
# ops -- first-boot checks, distill lane, SLO scrape, trace analysis
# ---------------------------------------------------------------------------

stage_ops() {
  info "ops"
  local unit_dir="${SCRIPT_DIR}/systemd"

  run mkdir -p "${STATE_DIR}/alerts" "${STATE_DIR}/gate" "${STATE_DIR}/traffic" \
    "${SCRATCH}/distill" "${SCRATCH}/deploy"
  if [[ ! -f "${SCRATCH}/deploy/offbox.env" ]]; then
    if (( DRY_RUN )); then
      printf '  \033[36m+\033[0m write %s\n' "${SCRATCH}/deploy/offbox.env"
    else
      cat > "${SCRATCH}/deploy/offbox.env" <<'ENV'
# Optional D1/E1. Drop a real URL here and restart k3-gateway.
# K3_OFFBOX_URL=
# K3_OFFBOX_API_KEY=
# K3_OFFBOX_MODEL=offbox
ENV
      chmod 600 "${SCRATCH}/deploy/offbox.env"
    fi
  fi

  run chmod +x "${SCRIPT_DIR}/first-boot.sh" \
    "${SCRIPT_DIR}/containers/run.sh" \
    "${SCRIPT_DIR}/containers/traffic.sh" \
    "${SCRIPT_DIR}/litellm/start.sh"
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m docker build k3-python:24.04\n'
  else
    docker build -t k3-python:24.04 \
      -f "${SCRIPT_DIR}/containers/Dockerfile.python" \
      "${SCRIPT_DIR}/containers"
  fi
  if id -u k3 >/dev/null 2>&1; then
    run chown -R k3:k3 "${STATE_DIR}" "${SCRATCH}/distill" "${SCRATCH}/traces" 2>/dev/null || true
  fi

  local unit
  for unit in k3-first-boot.service k3-distill.service k3-alerts.service \
              k3-traffic.service k3-traffic.timer k3-litellm-db.service \
              k3-litellm-redis.service k3-litellm.service; do
    if (( DRY_RUN )); then
      printf '  \033[36m+\033[0m install %s\n' "/etc/systemd/system/${unit}"
    else
      cp "${unit_dir}/${unit}" "/etc/systemd/system/${unit}"
    fi
  done

  run systemctl daemon-reload
  run systemctl enable --now k3-alerts.service
  run systemctl enable --now k3-traffic.timer
  run systemctl enable k3-first-boot.service
  run systemctl enable k3-distill.service
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m wipe %s (no Authorization swap)\n' /scratch/deploy/litellm-nginx-auth.conf
  else
    printf '%s\n' '# SYSTEM-DESIGN §4: do not set Authorization. LiteLLM owns the key.' \
      > /scratch/deploy/litellm-nginx-auth.conf
    chmod 644 /scratch/deploy/litellm-nginx-auth.conf
  fi
  if [[ -x /usr/local/lib/k3/venv/bin/litellm ]]; then
    run chmod +x "${SCRIPT_DIR}/litellm/start.sh"
    if [[ -f /scratch/deploy/litellm.env ]]; then
      run systemctl enable --now k3-litellm-db.service
      run systemctl enable --now k3-litellm-redis.service
    fi
    run systemctl enable --now k3-litellm.service
  else
    warn "litellm venv missing; the public API requires LiteLLM"
  fi
  # Start first-boot only if the engine is already answering; otherwise the
  # unit waits on k3.service after the next boot / engine start.
  if curl -fsS "http://${ENGINE_HOST}:${ENGINE_PORT}/health" >/dev/null 2>&1; then
    run systemctl start k3-first-boot.service
  else
    warn "engine not healthy yet; k3-first-boot will run after k3.service"
  fi
  ok "ops: alerts scrape, traffic timer, first-boot B0/Z5, distill enabled"
  mark_done ops
}

# ---------------------------------------------------------------------------
# edge
# ---------------------------------------------------------------------------

# LiteLLM virtual keys are tenancy. nginx only default-denies missing Bearer.
readonly NGINX_CONF_DIR="${NGINX_CONF_DIR:-/etc/nginx/conf.d}"

render_key_map() {
  local keys_conf="${NGINX_CONF_DIR}/00-k3-keys.conf"
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m python3 -m redesign.edge.keys --output %s\n' "$keys_conf"
    return 0
  fi
  ( cd "${REPO_ROOT}" && python3 -m redesign.edge.keys --output "$keys_conf" ) \
    || die "failed to write the pass-through Bearer map"
}

render_bootstrap_block() {
  local conf="${NGINX_CONF_DIR}/k3.conf"
  local body
  body=$(cat <<'NGINX'
# Generated by deploy.sh -- BOOTSTRAP ONLY, replaced once TLS is issued.
server {
    listen 80;
    server_name __DOMAIN__;
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 503; }
}
NGINX
)
  body="${body//__DOMAIN__/${DOMAIN}}"
  printf '%s\n' "$body" > "$conf"
  log "wrote HTTP-only bootstrap block for the ACME challenge"
}

render_server_block() {
  local conf="${NGINX_CONF_DIR}/k3.conf"
  local body
  body=$(cat <<'NGINX'
# Generated by deploy.sh. Implements docs/SYSTEM-DESIGN.md 4 (edge).
#
# The engine's own API key leaves unauthenticated routes open and provides no
# attribution, so this layer is not optional. It terminates TLS, default-denies
# a missing Bearer, applies per-key conn/req caps, and proxies ONLY to LiteLLM
# -- never to the engine, which would bypass tenancy, the clamp, and
# backpressure.

log_format k3usage '$remote_addr $k3_customer [$time_local] "$request" '
                   '$status $body_bytes_sent rt=$request_time '
                   'ttfb=$upstream_header_time cls=$upstream_http_x_k3_class';

# Per-key limits. Identity is the Bearer token; LiteLLM decides if it is valid.
# Empty Authorization is 401'd before these zones apply.
limit_req_zone  $http_authorization zone=k3_req:16m rate=100r/s;
limit_conn_zone $http_authorization zone=k3_conn:16m;

# Per-IP limits. Without these, anyone without a valid key is unthrottled:
# credential stuffing against the key map and connection floods are free, and
# a flood exhausts worker connections for paying customers.
limit_req_zone  $binary_remote_addr zone=k3_ip_req:16m rate=20r/s;
limit_conn_zone $binary_remote_addr zone=k3_ip_conn:16m;

# Every public request must pass LiteLLM authentication and cache policy.
# The unauthenticated gateway cannot serve as a tenancy failover.
upstream k3_tenancy {
    server 127.0.0.1:4000 max_fails=2 fail_timeout=5s;
    keepalive 32;
}
upstream k3_gateway { server 127.0.0.1:__GATEWAY_PORT__; keepalive 32; }

# Anything that matches no server_name is dropped without a response. Default
# server blocks that answer at all are an invitation to scanners.
server {
    listen 80 default_server;
    listen 443 ssl default_server;
    ssl_reject_handshake on;
    server_name _;
    return 444;
}

server {
    listen 80;
    server_name __DOMAIN__;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name __DOMAIN__;

    ssl_certificate     /etc/letsencrypt/live/__DOMAIN__/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/__DOMAIN__/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;

    access_log /var/log/k3/usage.log k3usage;

    # Streaming: buffering here would defeat every TTFT measurement downstream.
    proxy_buffering off;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    client_max_body_size 64m;

    location /v1/ {
        # Per-IP limits for requests that reach nginx's preaccess phase.
        limit_req  zone=k3_ip_req burst=40 nodelay;
        limit_conn k3_ip_conn 96;

        if ($http_authorization = "") { return 401; }

        limit_req  zone=k3_req burst=200 nodelay;
        # Portal-sized: one YYDS/shared key is a pool, not a developer.
        # 96 of admission 427 ≈ 22%. Was 32; that clipped the pool before TPM.
        limit_conn k3_conn 96;

        proxy_set_header X-K3-Customer $k3_customer;
        # S-3: do not forward a client-supplied batch header. Distillation
        # hits the engine on loopback; a public client must not self-declare P3.
        proxy_set_header X-K3-Batch "";
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_pass http://k3_tenancy;
        proxy_next_upstream error timeout http_502 http_503;
    }

    # Scrape surfaces stay on loopback. The gateway has no auth of its own;
    # this block is the only thing protecting it.
    location = /metrics { allow 127.0.0.1; deny all; proxy_pass http://k3_gateway; }
    location = /health  { allow 127.0.0.1; deny all; proxy_pass http://k3_gateway; }

    location /.well-known/acme-challenge/ { root /var/www/certbot; }

    location / {
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_http_version 1.1;
        proxy_pass http://k3_tenancy;
        proxy_next_upstream error timeout http_502 http_503;
    }
}
NGINX
)
  body="${body//__GATEWAY_PORT__/${GATEWAY_PORT}}"
  body="${body//__DOMAIN__/${DOMAIN}}"
  if [[ "${K3_THROUGHPUT_FIRST:-0}" == "1" ]]; then
    # Keep the legacy zones defined for rollback; throughput-mode requests use
    # authenticated quotas and the gateway's shared execution/waiting budgets.
    body="${body/limit_req  zone=k3_ip_req burst=40 nodelay;/# Throughput-first: bounded downstream admission.}"
    body="${body/limit_req  zone=k3_req burst=200 nodelay;/# Authenticated quota enforcement is in LiteLLM.}"
    body="${body/limit_conn k3_ip_conn 96;/limit_conn k3_ip_conn 256;}"
    body="${body/limit_conn k3_conn 96;/limit_conn k3_conn 256;}"
  fi

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m write %s (domain %s)\n' "$conf" "${DOMAIN}"
  else
    printf '%s\n' "$body" > "$conf"
  fi
}

stage_keys() {
  info "keys"
  require_nginx
  render_key_map
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m nginx -t && systemctl reload nginx\n'
  else
    nginx -t && systemctl reload nginx
  fi
  ok "auth layer is pass-through Bearer; LiteLLM owns tenancy"
  mark_done keys
}

require_nginx() {
  if (( DRY_RUN )); then
    log "would require nginx on PATH"
  elif ! command -v nginx >/dev/null; then
    die "nginx not installed. The edge and keys stages write into
       ${NGINX_CONF_DIR}; install nginx before running them."
  fi
}

stage_edge() {
  info "edge"

  [[ -n "${DOMAIN:-}" ]] || die \
    "DOMAIN is unset. The edge needs a hostname for TLS and for the
       default-deny block:  DOMAIN=api.cflowx.in ./deploy.sh edge"


  run mkdir -p /var/log/k3
  render_key_map

  # certbot's HTTP-01 burns Let's Encrypt rate limit at 5 failures per hostname
  # per hour, so refuse to call it until DNS actually points here.
  #
  # Ordering matters: the TLS server block names a certificate file. Writing it
  # before the certificate exists makes `nginx -t` fail, which aborts certbot
  # itself and leaves a broken config behind that breaks every later reload. So
  # the bootstrap path serves plain HTTP until the certificate is issued.
  local cert="/etc/letsencrypt/live/${DOMAIN}/fullchain.pem"
  if [[ -f "$cert" ]]; then
    ok "certificate already present for ${DOMAIN}"
  elif (( DRY_RUN )); then
    printf '  \033[36m+\033[0m write HTTP-only bootstrap block, reload nginx\n'
    printf '  \033[36m+\033[0m verify DNS, then certbot certonly --webroot -d %s\n' "${DOMAIN}"
  else
    render_bootstrap_block
    nginx -t || die "bootstrap nginx config is invalid"
    systemctl reload nginx
    local resolved public
    resolved=$(getent ahostsv4 "${DOMAIN}" | awk '{print $1; exit}')
    public=$(curl -fsS --max-time 10 https://api.ipify.org || echo "")
    if [[ -z "$resolved" || "$resolved" != "$public" ]]; then
      die "${DOMAIN} resolves to '${resolved:-nothing}' but this host is '${public}'.
       Refusing to call certbot: a failed HTTP-01 burns Let's Encrypt rate
       limit at 5 failures per hostname per hour. Point the A record here first."
    fi
    mkdir -p /var/www/certbot
    certbot certonly --webroot -w /var/www/certbot -d "${DOMAIN}" \
      --non-interactive --agree-tos \
      ${CERTBOT_EMAIL:+--email "${CERTBOT_EMAIL}"} \
      ${CERTBOT_EMAIL:---register-unsafely-without-email} \
      || die "certbot failed; the HTTP-only block is still serving"
  fi

  # Only now is it safe to reference the certificate.
  render_server_block

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m nginx -t && systemctl reload nginx\n'
  else
    nginx -t || die "generated nginx config is invalid"
    systemctl reload nginx
  fi

  ok "edge live on https://${DOMAIN} -> gateway :${GATEWAY_PORT}"
  mark_done edge
}

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

stage_verify() {
  info "verify"

  local health="http://${ENGINE_HOST}:${ENGINE_PORT}/health"
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m poll %s until ready\n' "$health"
  else
    local waited=0
    until curl -fsS "$health" >/dev/null 2>&1; do
      (( waited += 10 ))
      (( waited > 2400 )) && die "engine did not become ready within 40 minutes"
      sleep 10
    done
    ok "engine healthy after ${waited}s"
  fi

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m python3 -m redesign.probe.cache_salt --url http://%s:%s\n' \
      "${ENGINE_HOST}" "${ENGINE_PORT}"
  elif ( cd "${REPO_ROOT}" && python3 -m redesign.probe.cache_salt \
           --url "http://${ENGINE_HOST}:${ENGINE_PORT}" \
           --model "${K3_GATE_MODEL:-default}" \
           --json > "${STATE_DIR}/cache-salt.json" ); then
    ok "B0 cache_salt sharing confirmed"
  else
    warn "B0 cache_salt did not see a hit; clients may be salting or the cache is empty"
  fi

  # G2 pass/fail is the AGGREGATE across ranks, never one worker's log line.
  # Under DP attention each rank still reports ~2.3M (AUDIT 1.4).
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m compare AGGREGATE KV pool against the capacity model\n'
  else
    local pools
    pools=$(docker logs "${ENGINE_CONTAINER}" 2>&1 \
      | grep -oE 'KV cache size:? *[0-9]+' | grep -oE '[0-9]+' || echo "")
    if [[ -n "$pools" ]]; then
      log "per-rank pool lines: ${pools}"
      if (( ENABLE_DP_ATTENTION )); then
        ( cd "${REPO_ROOT}" && python3 - "$pools" <<'PY'
import sys
from redesign.capacity import deployment, g2
pools = [int(x) for x in sys.argv[1].split() if x.isdigit()]
base = deployment.production_model()
verdict = g2.interpret_g2_pools(
    pools,
    base.per_rank_resident_tokens(),
    base.with_variant(deduplicated=True).aggregate_unique_tokens(),
)
print(f"  aggregate {verdict.aggregate:,}  expected {verdict.expected_aggregate:,}"
      f"  spread {verdict.occupancy_spread:.0%}  pass={verdict.passed}")
if not verdict.passed:
    sys.exit(1)
PY
        ) || warn "aggregate pool does not match the A1 prediction; see G2 pass/fail"
      fi
    else
      warn "could not read the KV pool from engine logs"
    fi
  fi

  # The gate is the only thing that catches quantised KV's silent-garbage
  # failure, which every throughput benchmark reports as a success.
  local baseline="${STATE_DIR}/gate-baseline-${PROFILE_NAME}.json"
  local gate_args=(--url "http://${ENGINE_HOST}:${ENGINE_PORT}")

  if [[ -f "${STATE_DIR}/gate-baseline-baseline.json" && "${PROFILE_NAME}" != baseline ]]; then
    gate_args+=(--compare "${STATE_DIR}/gate-baseline-baseline.json")
  fi
  gate_args+=(--baseline "${baseline}")

  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m python3 -m redesign.gate %s\n' "${gate_args[*]}"
  elif ( cd "${REPO_ROOT}" && python3 -m redesign.gate "${gate_args[@]}" ); then
    ok "correctness gate passed; baseline at ${baseline}"
  else
    die "CORRECTNESS GATE FAILED.
       Do not route customer traffic to this build and do not accept a
       throughput number from it. If this profile enables fp8 KV, revert to
       the previous profile -- see SYSTEM-DESIGN.md 6 A2."
  fi

  mark_done verify

  info "next"
  log "1. snapshot this box now, before any tuning"
  log "2. capture a week of traces from ${TRACE_DIR} before sizing anything"
  log "3. book G2 only if redesign/probe reported enable_dp_attention present"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
  printf '\nProfiles: '
  ( cd "${PROFILE_DIR}" && ls *.env | sed 's/\.env//' | tr '\n' ' ' )
  printf '\nStages:   %s\n' "${ALL_STAGES[*]}"
}

main() {
  local -a requested=()

  while (( $# )); do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --profile) PROFILE="$2"; shift ;;
      -h|--help) usage; exit 0 ;;
      all) requested=("${ALL_STAGES[@]}") ;;
      -*) die "unknown option: $1" ;;
      *) requested+=("$1") ;;
    esac
    shift
  done

  (( ${#requested[@]} )) || requested=("${ALL_STAGES[@]}")

  for stage in "${requested[@]}"; do
    [[ " ${ALL_STAGES[*]} " == *" ${stage} "* ]] || die "unknown stage: ${stage}"
  done

  (( DRY_RUN )) && info "DRY RUN -- no mutation will be performed"

  load_profile

  for stage in "${requested[@]}"; do
    "stage_${stage}"
  done

  info "done  (profile ${PROFILE}, stages: ${requested[*]})"
}

main "$@"
