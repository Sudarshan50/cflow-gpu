#!/bin/bash
# Reproducible Kimi-K3 launch.
#
# Normally you do NOT run this - systemd owns the container:
#     systemctl start|stop|restart k3.service
# This script exists for a FRESH box, or to verify/restore the image.
set -euo pipefail

TAG="vllm/vllm-openai-rocm:kimi-k3"
# Expected image ID. The `kimi-k3` tag is a one-off dev tag that may be
# garbage-collected upstream, so we verify the BITS, not the name.
EXPECT="sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b"
TARBALL="/scratch/backup/k3-image.tar"

have_id() { docker image inspect "$1" --format '{{.Id}}' 2>/dev/null || true; }

id="$(have_id "$TAG")"
if [ "$id" != "$EXPECT" ]; then
  if [ -f "$TARBALL" ]; then
    echo "Restoring image from local tarball ($(du -h "$TARBALL" | cut -f1))..."
    docker load -i "$TARBALL"
  else
    echo "No local tarball; pulling by digest from registry..."
    docker pull "vllm/vllm-openai-rocm@${EXPECT#sha256:}" 2>/dev/null \
      || docker pull "$TAG"
  fi
  id="$(have_id "$TAG")"
fi

# Fail closed rather than serving unknown bits.
if [ "$id" != "$EXPECT" ]; then
  echo "FATAL: image ID mismatch." >&2
  echo "  expected $EXPECT" >&2
  echo "  got      ${id:-<absent>}" >&2
  exit 1
fi
echo "Image verified: $id"

# Weights must be present; HF_HUB_OFFLINE=1 means no silent re-download.
[ -d /scratch/hf/hub/models--moonshotai--Kimi-K3 ] || {
  echo "FATAL: weights missing at /scratch/hf" >&2; exit 1; }
[ -f /scratch/hf/config.yaml ] || install -m644 /scratch/deploy/config.yaml /scratch/hf/config.yaml

# Prefer systemd if the unit is installed - it owns restart/backoff policy.
if systemctl cat k3.service >/dev/null 2>&1; then
  echo "Starting via systemd (k3.service)..."
  exec systemctl restart k3.service
fi

echo "systemd unit absent; running in foreground..."
docker rm -f k3 >/dev/null 2>&1 || true
exec docker run --rm --name k3 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --network host --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --shm-size 64g -v /scratch/hf:/hf -v /scratch/results:/results \
  --env-file /scratch/deploy/vllm-k3.env \
  --entrypoint vllm "$TAG" serve moonshotai/Kimi-K3 --config /hf/config.yaml
