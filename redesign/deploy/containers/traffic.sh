#!/usr/bin/env bash
set -euo pipefail
mkdir -p /scratch/deploy-state/traffic
exec python3 -m redesign.traffic \
  --traces /scratch/traces/requests.jsonl \
  --window 24h \
  > /scratch/deploy-state/traffic/last.txt
