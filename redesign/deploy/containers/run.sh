#!/usr/bin/env bash
# One container per control-plane service. Host network keeps 127.0.0.1
# (engine :8001, gateway :8002, LiteLLM :4000) unchanged.
set -euo pipefail

NAME="${1:?container name}"
shift

IMAGE="${K3_CTL_IMAGE:-k3-python:24.04}"
UIDGID="${K3_CTL_UIDGID:-995:987}"

args=(
  --rm
  --name "$NAME"
  --network host
  --user "$UIDGID"
  -v /usr/local/lib/k3:/usr/local/lib/k3
  -v /scratch:/scratch
  -e PYTHONPATH=/usr/local/lib/k3
  -e PYTHONUNBUFFERED=1
)

while IFS='=' read -r key _; do
  case "$key" in
    K3_*|REDIS_*|UI_*|LITELLM_*|DATABASE_*|HOME|XDG_*|DISABLE_*|POSTGRES_*)
      args+=(-e "$key")
      ;;
  esac
done < <(env)

exec /usr/bin/docker run "${args[@]}" "$IMAGE" "$@"
