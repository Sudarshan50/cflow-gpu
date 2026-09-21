#!/usr/bin/env bash
# Dashboard container. Host network so it can scrape 127.0.0.1:8001 and bind
# 0.0.0.0:8080. Extra mounts: usage log, certs, rocm-smi.
set -euo pipefail

IMAGE="${K3_CTL_IMAGE:-k3-python:24.04}"

exec /usr/bin/docker run --rm --name k3-dash --network host --user 0:0 \
  -v /usr/local/lib/k3:/usr/local/lib/k3:ro \
  -v /scratch:/scratch:ro \
  -v /var/log/k3:/var/log/k3:ro \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  -v /usr/bin/rocm-smi:/usr/bin/rocm-smi:ro \
  -v /opt/rocm:/opt/rocm:ro \
  -v /opt/rocm/libexec/rocm_smi:/usr/libexec/rocm_smi:ro \
  -v /etc/alternatives:/etc/alternatives:ro \
  -e PYTHONUNBUFFERED=1 \
  -e DASH_ADDR="${DASH_ADDR:-0.0.0.0}" \
  -e DASH_PORT="${DASH_PORT:-8080}" \
  -e VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8001}" \
  -e PUBLIC_URL="${PUBLIC_URL:-https://api.cflowx.in/v1}" \
  -e ACCESS_LOG="${ACCESS_LOG:-/var/log/k3/usage.log}" \
  -e GPU_METRICS_URL="${GPU_METRICS_URL:-http://127.0.0.1:5000/metrics}" \
  -e CERT_PATH="${CERT_PATH:-/etc/letsencrypt/live/api.cflowx.in/fullchain.pem}" \
  -e API_KEY_FILE="${API_KEY_FILE:-/scratch/deploy/api-key.txt}" \
  -w /usr/local/lib/k3/dashboard \
  "$IMAGE" python3 server.py
