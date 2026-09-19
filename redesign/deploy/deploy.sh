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
#   edge       nginx, TLS, default-deny
#   verify     liveness, auth assertions, correctness gate, capacity check
#
# Every stage is idempotent. Re-running a completed stage costs nothing.
#
# GPU COST: this box bills from the moment it boots. Run --dry-run first, and
# read docs/SYSTEM-DESIGN.md 11 "GPU cost discipline" before the first real run.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PROFILE_DIR="${SCRIPT_DIR}/profiles"

readonly ALL_STAGES=(preflight host weights engine gateway edge verify)

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
run() {
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m %s\n' "$*"
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
readonly OVERRIDABLE=(MAX_RUNNING_REQUESTS MAX_MODEL_LEN LONG_PREFILL_THRESHOLD DP_SIZE)

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
        print(scenario["max_concurrency"])
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

  local gpus
  gpus=$(rocm-smi --showid 2>/dev/null | grep -c '^GPU\[' || echo 0)
  if (( gpus != REQUIRED_GPUS )); then
    die "found ${gpus} GPUs, this design requires ${REQUIRED_GPUS}.
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
  (( disk_gb >= REQUIRED_DISK_GB )) \
    || die "only ${disk_gb} GB free; the weights alone are ~1.5 TB"
  ok "${disk_gb} GB free"

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
  local -a required=(tp_size max_model_len max_running_requests)
  (( ENABLE_DP_ATTENTION )) && required+=(enable_dp_attention dp_size)
  [[ "${KV_CACHE_DTYPE}" != auto ]] && required+=(kv_cache_dtype)
  (( ENABLE_HIERARCHICAL_CACHE )) && required+=(enable_hierarchical_cache hicache_ratio)

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
    --model-path "/hf/hub/models--${MODEL_REPO//\//--}"
    --trust-remote-code
    --tp-size "${TP_SIZE}"
    --host "${ENGINE_HOST}"
    --port "${ENGINE_PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-running-requests "${ceiling}"
    --long-prefill-token-threshold "${LONG_PREFILL_THRESHOLD}"
    --reasoning-parser kimi_k3
    --tool-call-parser kimi_k3
  )
  if (( ENABLE_DP_ATTENTION )); then
    ENGINE_ARGS+=(--enable-dp-attention --dp-size "${DP_SIZE}")
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

  run docker rm -f "${ENGINE_CONTAINER}" 2>/dev/null || true

  run docker run -d --name "${ENGINE_CONTAINER}" \
    --restart unless-stopped \
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
    "${ENGINE_IMAGE}" \
    python3 -m sglang.launch_server "${ENGINE_ARGS[@]}"

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

  local unit=/etc/systemd/system/k3-gateway.service
  local rendered
  rendered=$(cat <<UNIT
[Unit]
Description=Kimi-K3 tenancy and backpressure gateway
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=${REPO_ROOT}
Environment=PYTHONUNBUFFERED=1
Environment=K3_ENGINE_URL=http://${ENGINE_HOST}:${ENGINE_PORT}
Environment=K3_MAX_MODEL_LEN=${MAX_MODEL_LEN}
Environment=K3_TRACE_PATH=${TRACE_DIR}/requests.jsonl
ExecStart=/usr/bin/python3 -m redesign.gateway.server --port ${GATEWAY_PORT}
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
# edge
# ---------------------------------------------------------------------------

stage_edge() {
  info "edge"

  # The engine's own API key leaves unauthenticated routes open and gives no
  # per-customer attribution, so the edge is not optional.
  warn "edge stage is a stub on this branch; nginx + TLS not yet implemented"
  log "until it lands, the gateway must not be exposed beyond loopback"
  log "see docs/SYSTEM-DESIGN.md 4"

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
      (( waited > 900 )) && die "engine did not become ready within 15 minutes"
      sleep 10
    done
    ok "engine healthy after ${waited}s"
  fi

  # The reported pool is what every admission number is derived from. If it does
  # not match the profile's prediction, the profile did not do what it claims --
  # most importantly, A1 may have been silently ignored.
  if (( DRY_RUN )); then
    printf '  \033[36m+\033[0m compare reported KV pool against the capacity model\n'
  else
    local pool
    pool=$(docker logs "${ENGINE_CONTAINER}" 2>&1 \
      | grep -oE 'KV cache size:? *[0-9]+' | tail -1 | grep -oE '[0-9]+' || echo "")
    if [[ -n "$pool" ]]; then
      log "engine reports a KV pool of ${pool} tokens"
      ( cd "${REPO_ROOT}" && python3 -m redesign.capacity ) | grep -A3 'HYPOTHESIS TEST' || true
      if (( ENABLE_DP_ATTENTION )) && (( pool < 10000000 )); then
        warn "A1 is enabled but the pool is under 10M tokens.
       De-duplication may not have taken effect -- this is G2's pass/fail."
      fi
    else
      warn "could not read the KV pool from engine logs"
    fi
  fi

  warn "correctness gate not yet implemented on this branch (Z5)"
  log "do NOT route customer traffic, and do NOT enable fp8 KV, until it is"

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
