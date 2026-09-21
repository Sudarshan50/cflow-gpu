#!/usr/bin/env bash
# Retired. This was the v1 nginx allowlist + Authorization-swap deployer.
# Live stack is SYSTEM-DESIGN §4: nginx → LiteLLM → backpressure → engine.
set -euo pipefail
echo "FATAL: repo-root deploy.sh is retired (it rebuilt k3-limits.inc)." >&2
echo "Use:  $(dirname "$(readlink -f "$0")")/redesign/deploy/deploy.sh $*" >&2
exit 1
