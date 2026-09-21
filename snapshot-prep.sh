#!/usr/bin/env bash
# Quiesce the box before taking a DigitalOcean snapshot.
#
#   ./snapshot-prep.sh          stop services, flush, trim, report
#   ./snapshot-prep.sh --resume bring everything back up afterwards
#
# A snapshot of the boot disk carries the weights, the image, the secrets and
# the certificate, so a restored droplet serves without downloading anything.
# Taking it while the engine is mid-write risks capturing a torn state, so this
# stops the writers first. Power the droplet off for the strongest guarantee.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

if [ "${1:-}" = --resume ]; then
  systemctl start k3.service k3-gateway.service k3-litellm.service nginx
  log "services starting; the engine needs ~4 min to load weights"
  log "verify with: ./verify-auth.sh"
  exit 0
fi

[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }

log "stopping writers"
systemctl stop k3.service || true
docker ps -q --filter name=k3 | xargs -r docker stop -t 30 >/dev/null 2>&1 || true

log "flushing to disk"
sync

# Unused blocks the hypervisor still thinks are allocated inflate the snapshot
# and the time it takes to create.
log "trimming free space (shrinks the snapshot)"
fstrim -v / 2>/dev/null || true

used=$(df -h --output=used / | tail -1 | tr -d ' ')
log "boot disk holds $used - expect a snapshot of roughly that size"

cat <<'EOF'

Ready. Now, in the DigitalOcean Control Panel:
  1. Power the droplet off (strongest consistency), then
  2. Create a snapshot of it, then
  3. Power it back on.

If you snapshot live instead, the services are already stopped so the data is
consistent; bring them back with:
    ./snapshot-prep.sh --resume
EOF
