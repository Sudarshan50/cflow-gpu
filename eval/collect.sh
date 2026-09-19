#!/usr/bin/env bash
# One-shot evidence collection for the Kimi-K3 endpoint evaluation.
#
# Produces a self-contained, zippable run directory under runs/<UTC timestamp>/
# holding everything a reader needs to check the report's claims: the
# environment it was measured on, the raw probe results, the production
# statistics, and the outputs of the repo's own security and correctness gates.
#
#   ./collect.sh                 full run
#   ./collect.sh --quick         skip the long-context and concurrency sections
#   RUN_TAG=name ./collect.sh    label the run directory
#
# Safe to run against the live box: every probe is a normal customer request
# through the public edge, and the load is a few dozen requests. It does NOT
# restart anything.
set -uo pipefail

EVAL_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HARNESS="$EVAL_DIR/harness"
DEPLOY="${DEPLOY:-/scratch/deploy}"
BASE_URL="${BASE_URL:-https://cflox.store/v1}"
MODEL="${MODEL:-FW-Kimi-K3}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$EVAL_DIR/runs/${STAMP}${RUN_TAG:+-$RUN_TAG}"
mkdir -p "$RUN"

QUICK=0
SKIP=""
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1; SKIP="I,K" ;;
    --skip=*) SKIP="${arg#--skip=}" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "run directory: $RUN"

# ---- 1. environment ------------------------------------------------------
log "collecting environment"
python3 "$HARNESS/collect_env.py" "$RUN" 2>&1 | tee "$RUN/run_env.log"

# ---- 2. raw engine surfaces ---------------------------------------------
log "capturing raw engine surfaces"
curl -s --max-time 20 http://127.0.0.1:8001/metrics > "$RUN/metrics_snapshot.prom" \
  || echo "metrics scrape failed" > "$RUN/metrics_snapshot.prom"
# The startup banner carries quantisation, KV sizing and backend selection.
docker logs k3 2>&1 | grep -E \
  "Initializing a V1 LLM engine|quantization=|GPU KV cache size|Maximum concurrency|\
Available KV cache memory|Loading weights took|AITER_MXFP4|non-default args" \
  > "$RUN/engine_startup.log" 2>&1 || true
install -m 644 "$DEPLOY/config.yaml" "$RUN/config.yaml" 2>/dev/null || true

# ---- 3. compatibility health check --------------------------------------
log "running compatibility health check (this sends real inference)"
python3 "$HARNESS/kvv_health.py" \
  --base-url "$BASE_URL" --model "$MODEL" --out-dir "$RUN" \
  ${SKIP:+--skip "$SKIP"} 2>&1 | tee "$RUN/run_health.log"
HEALTH_RC=${PIPESTATUS[0]}

# ---- 4. repo's own gates ------------------------------------------------
log "running the repo security suite (verify-auth.sh)"
( cd "$DEPLOY" && ./verify-auth.sh ) > "$RUN/verify_auth.log" 2>&1
VERIFY_RC=$?

log "running the correctness gate (tests/gate.sh)"
( cd "$DEPLOY" && bash tests/gate.sh ) > "$RUN/gate.log" 2>&1
GATE_RC=$?

# ---- 5. production statistics (after the probes, so they are included) ---
log "collecting production statistics"
python3 "$HARNESS/prod_stats.py" "$RUN" 2>&1 | tee "$RUN/run_prod.log"

# ---- 6. manifest --------------------------------------------------------
log "writing manifest"
python3 - "$RUN" "$HEALTH_RC" "$VERIFY_RC" "$GATE_RC" <<'PY'
import hashlib, json, os, sys, time
run, health_rc, verify_rc, gate_rc = sys.argv[1], *map(int, sys.argv[2:5])
files = []
for name in sorted(os.listdir(run)):
    p = os.path.join(run, name)
    if not os.path.isfile(p) or name == "manifest.json":
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    files.append({"file": name, "bytes": os.path.getsize(p),
                  "sha256": h.hexdigest()})
counts = {}
try:
    with open(os.path.join(run, "health_results.json")) as f:
        counts = json.load(f).get("counts", {})
except Exception:
    pass
json.dump({
    "run": os.path.basename(run),
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "exit_codes": {"health": health_rc, "verify_auth": verify_rc, "gate": gate_rc},
    "health_counts": counts,
    "files": files,
}, open(os.path.join(run, "manifest.json"), "w"), indent=2)
print("manifest: %d files" % len(files))
PY

log "done"
echo
echo "========================================================================="
echo " run      : $RUN"
echo " health   : exit $HEALTH_RC   (see run_health.log / health_results.json)"
echo " verify   : exit $VERIFY_RC   (see verify_auth.log)"
echo " gate     : exit $GATE_RC   (see gate.log)"
echo " zip it   : cd $EVAL_DIR/runs && zip -r ${STAMP}.zip $(basename "$RUN")"
echo "========================================================================="
