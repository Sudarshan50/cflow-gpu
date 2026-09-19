#!/usr/bin/env bash
# Sequential campaign over the current-image variants. The v0.29.0 upgrade is
# deliberately NOT here: it needs the systemd unit's pinned digest changed and
# a different env file, and it is gated on the correctness gate passing.
set -uo pipefail
cd /scratch/ab

for v in baseline kvoffload longprefill async batch16k; do
  ./driver.sh run "$v" "/scratch/ab/configs/$v.yaml"
  rc=$?
  echo "### $v finished rc=$rc status=$(cat "results/$v/STATUS" 2>/dev/null)"
  if [[ $rc -ne 0 ]]; then
    echo "### $v FAILED - continuing to next variant, engine will be reset by it"
  fi
done
echo "### campaign complete"
