#!/usr/bin/env bash
# SYSTEM-DESIGN §4: nginx is not the tenancy layer.
# LiteLLM virtual keys own who / how much. This script only writes the
# pass-through Bearer map. Do not add customers.tsv rows for portal keys.
set -euo pipefail

echo "tenancy is LiteLLM, not customers.tsv / nginx maps"
echo "writing the pass-through Bearer map only"
python3 -m redesign.edge.keys --output /etc/nginx/conf.d/00-k3-keys.conf
rm -f /etc/nginx/conf.d/k3-limits.inc
nginx -t
systemctl reload nginx
echo "nginx reloaded — Authorization is forwarded to LiteLLM unchanged"
