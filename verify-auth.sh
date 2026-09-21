#!/usr/bin/env bash
# Public /v1/ follows SYSTEM-DESIGN §4: nginx default-denies a missing
# Bearer; LiteLLM decides if the key is real.
set -euo pipefail

DOMAIN="${DOMAIN:-api.cflowx.in}"
RESOLVE=(--resolve "${DOMAIN}:443:127.0.0.1")
KEY_FILE="${EDGE_KEY_FILE:-/scratch/deploy/edge.key}"
fail=0

chk() {
  local label="$1" want="$2" got="$3"
  if [[ "$got" == "$want" ]]; then
    printf 'OK    %s → %s\n' "$label" "$got"
  else
    printf 'FAIL  %s → %s (want %s)\n' "$label" "$got" "$want"
    fail=1
  fi
}

ss -tln | grep -q ':443 ' || {
  echo "FATAL: nothing listening on :443" >&2
  exit 1
}
[[ -s "$KEY_FILE" ]] || {
  echo "FATAL: no LiteLLM key at ${KEY_FILE} (or set EDGE_KEY_FILE)" >&2
  exit 1
}
KEY=$(tr -d '\n' < "$KEY_FILE")

none=$(curl -sk --http1.1 --max-time 20 -o /dev/null -w '%{http_code}' \
  "${RESOLVE[@]}" "https://${DOMAIN}/v1/models")
junk=$(curl -sk --http1.1 --max-time 20 -o /dev/null -w '%{http_code}' \
  "${RESOLVE[@]}" -H 'Authorization: Bearer sk-not-a-key' \
  "https://${DOMAIN}/v1/models")
ok=$(curl -sk --http1.1 --max-time 20 -o /dev/null -w '%{http_code}' \
  "${RESOLVE[@]}" -H "Authorization: Bearer ${KEY}" \
  "https://${DOMAIN}/v1/models")

chk "no key" 401 "$none"
chk "junk key" 401 "$junk"
chk "LiteLLM virtual key" 200 "$ok"
exit "$fail"
