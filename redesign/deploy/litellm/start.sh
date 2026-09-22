#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${K3_REPO_ROOT:-/usr/local/lib/k3}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="/usr/local/lib/k3/venv/bin:${PATH}"
OUT="${K3_LITELLM_CONFIG:-/scratch/deploy/litellm.yaml}"
WORKERS="${K3_LITELLM_WORKERS:-32}"
python3 -m redesign.tenancy.render_config --output "$OUT"
LITELLM_WORKDIR="/usr/local/lib/k3/venv/lib/python3.12/site-packages/litellm/proxy"
[[ -f "${LITELLM_WORKDIR}/schema.prisma" ]] || {
  echo "FATAL: LiteLLM Prisma schema is missing from ${LITELLM_WORKDIR}" >&2
  exit 1
}
cd "$LITELLM_WORKDIR"
exec /usr/local/lib/k3/venv/bin/litellm \
  --config "$OUT" \
  --host 127.0.0.1 \
  --port 4000 \
  --num_workers "$WORKERS"
