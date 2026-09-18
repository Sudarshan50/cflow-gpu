#!/usr/bin/env bash
# Proves the public TLS edge (443) is authenticated and path-restricted.
#
# Why this exists: vLLM's --api-key only guards ("/v1","/v2","/inference")
# (entrypoints/serve/utils/server_utils.py:42). On this build POST /invocations
# answered 200 with no credentials -- full inference for free. nginx closes that.

KEY=$(cat /scratch/deploy/api-key.txt)
DOMAIN=${DOMAIN:-cflox.store}

# Two-tier credentials, so the edge and the engine need DIFFERENT keys and
# swapping them silently inverts what this suite proves:
#   KEY      the single upstream credential vLLM enforces (VLLM_API_KEY).
#            nginx injects it; no client should ever possess it. Presenting it
#            AT the edge is correctly a 401, because it is not in the customer
#            map - so it cannot be used to test the happy path.
#   CUST_KEY a real per-customer key from customers.tsv. This is the only
#            thing the public edge accepts.
CUST_KEY=${CUST_KEY:-$(awk -F'\t' '!/^[[:space:]]*#/ && NF>=2 {print $2; exit}' \
           /scratch/deploy/customers.tsv 2>/dev/null)}
[ -n "${CUST_KEY:-}" ] || {
  echo "FATAL: no customer key found in customers.tsv - run ./gen-keys.sh add <name>" >&2
  exit 1; }
EDGE=https://${DOMAIN}            # nginx
DIRECT=http://127.0.0.1:8001      # vLLM itself
fail=0

# The edge used to be plaintext :8000. That listener was REMOVED when the API
# moved to TLS (nginx/k3.conf:1-2), so every assertion below must go through
# 443 or it fails for the wrong reason - a connection refused looks identical
# to a passing deny test if you only compare status codes.
#
# --resolve pins the hostname to loopback so this still exercises the real
# server block (SNI + server_name + the cert) on a box whose DNS has not
# propagated yet, or whose A record points at a different host entirely.
RESOLVE=(--resolve "${DOMAIN}:443:127.0.0.1")
ss -tln | grep -q ':443 ' || {
  echo "FATAL: nothing is listening on :443 - run ./issue-cert.sh first" >&2; exit 1; }

chk() { # label want got
  if [ "$2" = "$3" ]; then printf '  PASS  %-46s %s\n' "$1" "$3"
  else printf '  FAIL  %-46s got %s want %s\n' "$1" "${3:-<empty>}" "$2"; fail=$((fail+1)); fi
}

# Fixed timeout. Do NOT take it as a positional arg -- every remaining arg here
# is a curl flag, and consuming one as --max-time silently voids the request.
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 90 "${RESOLVE[@]}" "$@" 2>/dev/null; }
J='Content-Type: application/json'

echo "== listeners =="
ss -tln 2>/dev/null | awk '$4 ~ /:(80|443|8001|8080)$/ {print $4}' | sort -u | while read -r a; do
  printf '  %-16s %s worker(s)\n' "$a" "$(ss -tlnp 2>/dev/null | grep -c "$a")"
done
printf '  nginx worker_processes: %s\n' "$(pgrep -c -f 'nginx: worker')"

echo
echo "== the hole --api-key does NOT close, now closed by nginx =="
chk "POST /invocations        (unauth)" 403 "$(code -X POST $EDGE/invocations -H "$J" \
      -d '{"model":"FW-Kimi-K3","messages":[{"role":"user","content":"hi"}],"max_tokens":3}')"
chk "POST /scale_elastic_ep   (unauth)" 403 "$(code -X POST $EDGE/scale_elastic_ep -H "$J" -d '{}')"
chk "POST /tokenize           (unauth)" 403 "$(code -X POST $EDGE/tokenize -H "$J" -d '{"prompt":"hi"}')"
chk "POST /detokenize         (unauth)" 403 "$(code -X POST $EDGE/detokenize -H "$J" -d '{"tokens":[1,2]}')"
chk "POST /generative_scoring (unauth)" 403 "$(code -X POST $EDGE/generative_scoring -H "$J" -d '{}')"
chk "GET  /load               (unauth)" 403 "$(code $EDGE/load)"
chk "GET  /version            (unauth)" 403 "$(code $EDGE/version)"
chk "GET  /openapi.json       (unauth)" 403 "$(code $EDGE/openapi.json)"
chk "GET  /docs               (unauth)" 403 "$(code $EDGE/docs)"

echo
echo "== inference requires the key =="
chk "GET  /v1/models   no key"    401 "$(code $EDGE/v1/models)"
chk "GET  /v1/models   wrong key" 401 "$(code $EDGE/v1/models -H 'Authorization: Bearer sk-wrong')"
# The upstream key must NOT work from outside: it is not in the customer map,
# and a 200 here would mean the edge accepts a credential no customer holds.
chk "GET  /v1/models   upstream key" 401 "$(code $EDGE/v1/models -H "Authorization: Bearer $KEY")"
chk "GET  /v1/models   customer key" 200 "$(code $EDGE/v1/models -H "Authorization: Bearer $CUST_KEY")"
chk "POST /v1/chat/completions no key"    401 "$(code -X POST $EDGE/v1/chat/completions -H "$J" \
      -d '{"model":"FW-Kimi-K3","messages":[{"role":"user","content":"hi"}],"max_tokens":3}')"
chk "POST /v1/chat/completions cust key" 200 "$(code -X POST $EDGE/v1/chat/completions \
      -H "Authorization: Bearer $CUST_KEY" -H "$J" \
      -d '{"model":"FW-Kimi-K3","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}')"
chk "POST /v1/completions     cust key" 200 "$(code -X POST $EDGE/v1/completions \
      -H "Authorization: Bearer $CUST_KEY" -H "$J" \
      -d '{"model":"FW-Kimi-K3","prompt":"hi","max_tokens":8,"temperature":0}')"

echo
echo "== diagnostics stay local-only (the :8080 dashboard scrapes /metrics) =="
chk "GET  /metrics from 127.0.0.1" 200 "$(code $EDGE/metrics)"
chk "GET  /health  from 127.0.0.1" 200 "$(code $EDGE/health)"

echo
echo "== vLLM is not directly reachable off-box, and holds the key itself =="
chk "direct :8001 /v1/models no key" 401 "$(code $DIRECT/v1/models)"
chk "vLLM bound to loopback only"    "127.0.0.1:8001" \
      "$(ss -tln | awk '$4 ~ /:8001$/ {print $4}' | head -1)"

echo
echo "== the :8080 dashboard still works =="
chk "GET  :8080/"          200 "$(code http://127.0.0.1:8080/)"
chk "GET  :8080/api/state" 200 "$(code http://127.0.0.1:8080/api/state)"
chk "GET  :8080/api/health" 200 "$(code http://127.0.0.1:8080/api/health)"
# Assert the dashboard's own verdict on its upstream. Do NOT count nulls:
# percentile lists are legitimately empty until traffic flows, and gpu.error
# being null is the healthy case.
chk "dashboard's upstream scrape of vLLM" "up" \
      "$(curl -s --max-time 15 http://127.0.0.1:8080/api/state \
         | python3 -c 'import json,sys; print(json.load(sys.stdin)["server"]["state"])' 2>/dev/null)"
# Field lives under "capacity", not "cache" - server.py renamed it and the old
# path silently yielded <empty>, which read as a failure rather than a bug.
chk "dashboard sees KV capacity (engine live)" "true" \
      "$(curl -s --max-time 15 http://127.0.0.1:8080/api/state \
         | python3 -c 'import json,sys; print(str((json.load(sys.stdin)["capacity"]["capacity_tokens"] or 0)>0).lower())' 2>/dev/null)"

echo
echo "== end-to-end: real answer through the authenticated edge =="
# K3 is a reasoning model and emits its chain before the answer, so give it room
# and assert the digits appear rather than that the reply equals them.
out=$(curl -s --max-time 180 "${RESOLVE[@]}" $EDGE/v1/chat/completions \
  -H "Authorization: Bearer $CUST_KEY" -H "$J" \
  -d '{"model":"FW-Kimi-K3","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":512,"temperature":0}' \
  | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("reasoning_content") or "")+(m.get("content") or ""))' 2>/dev/null)
chk "17*23 = 391 appears in reply" "true" "$(case "$out" in *391*) echo true;; *) echo false;; esac)"
printf '  reply tail: %s\n' "$(printf '%s' "$out" | tr '\n' ' ' | tail -c 110)"

echo
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "$fail CHECK(S) FAILED"; fi
exit $fail
