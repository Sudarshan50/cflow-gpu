#!/usr/bin/env bash
# Serial A/B driver for Kimi-K3 serving-config experiments.
#
# Only one engine can exist at a time (8x MI355X are fully consumed by one
# TP=8 instance), so every variant is an exclusive restart. This script owns
# that sequence: apply config -> restart -> wait ready -> warmup -> bench ->
# correctness gate -> record.
#
# Usage:
#   ./driver.sh run <name> <config-file> [env-file]
#   ./driver.sh restore
set -uo pipefail

AB=/scratch/ab
RESULTS=$AB/results
CONFIG_LIVE=/scratch/hf/config.yaml
ENV_LIVE=/scratch/deploy/vllm-k3.env
API=http://127.0.0.1:8001
KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$ENV_LIVE")
MODEL=FW-Kimi-K3

mkdir -p "$RESULTS"
log() { printf '[%(%H:%M:%S)T] %s\n' -1 "$*"; }

wait_ready() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if curl -sf -m 5 -H "Authorization: Bearer $KEY" "$API/health" >/dev/null 2>&1; then
      log "engine ready after $((SECONDS))s"; return 0
    fi
    if ! systemctl is-active --quiet k3.service; then
      log "FATAL: k3.service died during startup"; return 1
    fi
    sleep 10
  done
  log "FATAL: engine not ready within 900s"; return 1
}

# Benchmark profile: long prompts (the live KV stressor) but with a shared
# prefix large enough to hold the hit rate above 65%, which is the operating
# point we are tuning for. A near-zero-reuse profile would optimise for a
# regime we do not want to run in.
#
#   prefix 18432 = 24 x 768, aligned to the attention block size this hybrid
#                  model forces, so no partial block is wasted
#   mean total   = 18432 + 7168 = 25600  ->  nominal hit rate 72%
#   range 0.5    = unique part in [3584, 10752], so the hit rate stays in
#                  roughly [63%, 84%] per request and ~72% in aggregate
BENCH_PREFIX=18432
BENCH_INPUT=7168
BENCH_RANGE=0.5

bench_live() {
  local out=$1
  docker exec -e OPENAI_API_KEY="$KEY" k3 vllm bench serve \
    --backend openai --endpoint /v1/completions \
    --base-url "$API" --model "$MODEL" --served-model-name "$MODEL" \
    --tokenizer /hf/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06 \
    --trust-remote-code --dataset-name random \
    --random-prefix-len $BENCH_PREFIX --random-input-len $BENCH_INPUT \
    --random-range-ratio $BENCH_RANGE --random-output-len 600 \
    --ignore-eos --request-rate inf --max-concurrency 256 --num-prompts 200 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir /results --result-filename "$(basename "$out")" \
    2>&1 | tee "${out%.json}.log"
}

# Prefix hit rate over the benchmark window only, from the counter delta.
# The cumulative gauge is useless here because warmup pollutes it.
hit_rate() {
  local dir=$1
  python3 - "$dir" <<'PY'
import re, sys, os
d = sys.argv[1]
def grab(f):
    p = os.path.join(d, f)
    if not os.path.exists(p): return None
    t = open(p).read()
    def n(metric):
        m = re.search(r'^vllm:%s\{[^}]*\} ([0-9.e+]+)$' % metric, t, re.M)
        return float(m.group(1)) if m else 0.0
    return n('prefix_cache_queries_total'), n('prefix_cache_hits_total')
a, b = grab('metrics_before.prom'), grab('metrics_after.prom')
if not a or not b:
    print('hit_rate: metrics missing'); sys.exit()
dq, dh = b[0] - a[0], b[1] - a[1]
rate = 100 * dh / dq if dq else float('nan')
print(f'bench-window prefix hit rate: {rate:.1f}%  (queries={dq:.0f} hits={dh:.0f})')
open(os.path.join(d, 'hit_rate.txt'), 'w').write(f'{rate:.2f}\n')
PY
}

warmup() {
  log "warmup (populates cache, triggers AITER/Triton JIT)"
  docker exec -e OPENAI_API_KEY="$KEY" k3 vllm bench serve \
    --backend openai --endpoint /v1/completions \
    --base-url "$API" --model "$MODEL" --served-model-name "$MODEL" \
    --tokenizer /hf/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06 \
    --trust-remote-code --dataset-name random \
    --random-prefix-len 1024 --random-input-len 8000 --random-range-ratio 0.5 \
    --random-output-len 128 --ignore-eos \
    --request-rate 4 --num-prompts 24 >/dev/null 2>&1
  log "warmup done"
}

snapshot_metrics() {
  curl -s -H "Authorization: Bearer $KEY" "$API/metrics" > "$1" 2>/dev/null || true
}

# A `docker exec`ed benchmark outlives a kill of this script, and a survivor
# silently poisons the next variant's numbers. Clear both sides before starting.
kill_strays() {
  pkill -9 -f "vllm bench serve" 2>/dev/null
  docker exec k3 pkill -9 -f "bench serve" 2>/dev/null
  sleep 3
}

cmd_run() {
  local name=$1 cfg=$2 envf=${3:-}
  local dir="$RESULTS/$name"; mkdir -p "$dir"
  log "=== VARIANT: $name ==="
  kill_strays

  install -m644 "$cfg" "$CONFIG_LIVE"
  cp "$cfg" "$dir/config.yaml"
  if [[ -n $envf ]]; then install -m600 "$envf" "$ENV_LIVE"; fi
  cp "$ENV_LIVE" "$dir/env.used"

  log "restarting k3.service"
  systemctl reset-failed k3.service 2>/dev/null
  local t0=$SECONDS
  if ! systemctl restart k3.service; then
    log "FATAL: systemctl restart failed"
    docker logs k3 2>&1 | tail -50 > "$dir/startup-fail.log"
    echo "restart_failed" > "$dir/STATUS"; return 1
  fi
  if ! wait_ready; then
    docker logs k3 2>&1 | tail -120 > "$dir/startup-fail.log"
    echo "not_ready" > "$dir/STATUS"; return 1
  fi
  echo $((SECONDS - t0)) > "$dir/startup_seconds"

  docker logs k3 2>&1 | grep -E \
    "GPU KV cache size|Maximum concurrency|Available KV cache memory|\
non-default args|Initializing a V1 LLM engine|hybrid kv cache|offload|Chunked prefill" \
    > "$dir/startup.log" 2>&1

  warmup
  snapshot_metrics "$dir/metrics_before.prom"
  log "benchmark: live-like profile"
  bench_live "$dir/bench_live.json"
  cp /scratch/results/bench_live.json "$dir/bench_live.json" 2>/dev/null
  snapshot_metrics "$dir/metrics_after.prom"
  hit_rate "$dir" | tee -a "$dir/summary.txt"

  log "correctness gate"
  /scratch/gate.sh > "$dir/gate.log" 2>&1
  echo $? > "$dir/gate_rc"

  echo ok > "$dir/STATUS"
  log "=== $name complete (gate rc=$(cat "$dir/gate_rc")) ==="
}

cmd_restore() {
  log "restoring baseline config + env"
  install -m644 "$AB/configs/baseline.yaml" "$CONFIG_LIVE"
  install -m600 "$AB/env/baseline.env" "$ENV_LIVE"
  systemctl reset-failed k3.service 2>/dev/null
  systemctl restart k3.service && wait_ready
}

case ${1:-} in
  run) shift; cmd_run "$@" ;;
  restore) cmd_restore ;;
  *) echo "usage: $0 run <name> <config> [env] | restore" >&2; exit 2 ;;
esac
