#!/usr/bin/env bash
# Cache and correctness checks after engine health becomes available.
set -euo pipefail

ENGINE_URL="${K3_ENGINE_URL:-http://127.0.0.1:8001}"
MODEL="${K3_GATE_MODEL:-FW-Kimi-K3}"
STATE_DIR="${K3_STATE_DIR:-/scratch/deploy-state}"
REPO_ROOT="${K3_REPO_ROOT:-/usr/local/lib/k3}"
TIERS="${K3_GATE_TIERS:-1}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${STATE_DIR}/gate" "${STATE_DIR}/alerts"

until curl -fsS "${ENGINE_URL}/health" >/dev/null 2>&1; do
  sleep 5
done

echo "B0 cache_salt ${ENGINE_URL}"
b0=0
python3 -m redesign.probe.cache_salt \
  --url "${ENGINE_URL}" \
  --model "${MODEL}" \
  --json | tee "${STATE_DIR}/cache-salt.json" || b0=$?

echo "Z5 gate tiers=${TIERS}"
gate_args=(--url "${ENGINE_URL}" --model "${MODEL}" --baseline "${STATE_DIR}/gate/z5.json")
for tier in ${TIERS}; do
  gate_args+=(--tier "${tier}")
done
z5=0
python3 -m redesign.gate "${gate_args[@]}" || z5=$?
exit $(( b0 != 0 || z5 != 0 ? 1 : 0 ))
