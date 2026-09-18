#!/bin/bash
# Entry point for the Kimi-K3 correctness gate. Delegates to gate.py.
#   gate.sh            full gate  (~21s)
#   gate.sh --quick    tier 1 only (~5s)
#   gate.sh --baseline re-record the baseline
# Resolve gate.py next to this script rather than at a fixed /scratch path, so
# the gate runs both from the repo checkout and from wherever deploy.sh
# installs it. The two used to drift and you would silently test the old copy.
exec python3 "$(dirname "$(readlink -f "$0")")/gate.py" "$@"
