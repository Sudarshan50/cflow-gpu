#!/usr/bin/env bash
# Switch the pinned image for k3.service between the current dev build and
# v0.29.0, via a systemd drop-in so the original unit file stays untouched and
# rollback is a single file removal.
#
#   ./upgrade.sh to-v0290    point the unit at v0.29.0
#   ./upgrade.sh rollback    remove the override, back to the kimi-k3 digest
set -euo pipefail

DROPIN_DIR=/etc/systemd/system/k3.service.d
DROPIN=$DROPIN_DIR/10-image.conf
V0290=sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1

case ${1:-} in
to-v0290)
  mkdir -p "$DROPIN_DIR"
  # ExecStart must be cleared before being redefined, or systemd appends.
  cat > "$DROPIN" <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/docker run --rm --name k3 \\
  --device /dev/kfd --device /dev/dri --group-add video \\
  --ipc host --network host --security-opt seccomp=unconfined --cap-add SYS_PTRACE \\
  --shm-size 64g \\
  --log-opt max-size=100m --log-opt max-file=5 \\
  -v /scratch/hf:/hf -v /scratch/results:/results \\
  --env-file /scratch/deploy/vllm-k3.env \\
  --entrypoint vllm \\
  vllm/vllm-openai-rocm@$V0290 \\
  serve moonshotai/Kimi-K3 --config /hf/config.yaml
EOF
  systemctl daemon-reload
  echo "pinned to v0.29.0 ($V0290)"
  ;;
rollback)
  rm -f "$DROPIN"
  rmdir "$DROPIN_DIR" 2>/dev/null || true
  systemctl daemon-reload
  echo "override removed; back to the unit's original kimi-k3 digest"
  ;;
*)
  echo "usage: $0 to-v0290|rollback" >&2; exit 2 ;;
esac

systemctl cat k3.service | grep -E "^ *vllm/vllm-openai-rocm@" | sed 's/^ */  image: /'
