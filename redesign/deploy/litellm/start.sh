#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${K3_REPO_ROOT:-/usr/local/lib/k3}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="/usr/local/lib/k3/venv/bin:${PATH}"
OUT="${K3_LITELLM_CONFIG:-/scratch/deploy/litellm.yaml}"
WORKERS="${K3_LITELLM_WORKERS:-32}"
python3 -m redesign.tenancy.render_config --output "$OUT"
exec /usr/local/lib/k3/venv/bin/litellm \
  --config "$OUT" \
  --host 127.0.0.1 \
  --port 4000 \
  --num_workers "$WORKERS"
