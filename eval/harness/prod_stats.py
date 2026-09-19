#!/usr/bin/env python3
"""Production statistics for the Kimi-K3 endpoint, in the shape of the
reference report's section 6.

Two sources, because neither alone is sufficient:

  nginx k3usage log  - the only place per-customer identity, HTTP status and
                       edge latency exist. The proxy swaps every customer key
                       for one upstream credential, so vLLM cannot attribute
                       anything to a customer.
  vLLM /metrics      - engine-side token counts, prefix-cache reuse, queue
                       depth, preemptions and latency histograms.

Windows mirror the reference report: 1h / 3h / 24h. Rotated and gzipped logs
are included so a 24h window survives a midnight rotation.

Failure causes are classified from HTTP status and request path, which is what
the access log actually carries. The reference report could classify 400s by
upstream error text because it sat in front of a gateway database; that detail
is not available here and the report says so rather than inventing it.
"""

import glob
import gzip
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

LOG_GLOB = "/var/log/k3/usage.log*"
METRICS_URL = "http://127.0.0.1:8001/metrics"
WINDOWS = (("1h", 3600), ("3h", 10800), ("24h", 86400))

LINE_RE = re.compile(
    r"^(?P<ts>\S+)\s+cust=(?P<cust>\S*)\s+status=(?P<status>\d+)\s+"
    r"path=(?P<path>\S+)\s+req_ms=(?P<req>\S+)\s+up_ms=(?P<up>\S+)\s+"
    r"in=(?P<in>\d+)\s+out=(?P<out>\d+)\s+ip=(?P<ip>\S+)\s*$")


def cause_of(status, path):
    """Failure taxonomy the access log can actually support."""
    if status == 401:
        return "401 missing or invalid customer key"
    if status == 403:
        return "403 path not on the edge allowlist"
    if status == 429:
        return "429 per-customer rate or concurrency cap"
    if status == 400:
        return "400 request body rejected by the engine"
    if status == 413:
        return "413 request body over 256 MB"
    if status == 404:
        return "404 unknown route"
    if status == 499:
        return "499 client disconnected before response"
    if 500 <= status < 600:
        return "%d upstream/server error" % status
    return "%d other" % status


def read_rows():
    rows = []
    for path in sorted(glob.glob(LOG_GLOB)):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", errors="replace") as f:
                for line in f:
                    m = LINE_RE.match(line.strip())
                    if not m:
                        continue
                    try:
                        ts = datetime.fromisoformat(m.group("ts")).timestamp()
                    except Exception:
                        continue
                    rows.append({
                        "ts": ts,
                        "cust": m.group("cust") or "(unauthenticated)",
                        "status": int(m.group("status")),
                        "path": m.group("path").split("?")[0],
                        "req_s": float(m.group("req")) if m.group("req") != "-" else None,
                        "up_s": float(m.group("up")) if m.group("up") != "-" else None,
                        "in": int(m.group("in")),
                        "out": int(m.group("out")),
                        "ip": m.group("ip"),
                    })
        except OSError:
            continue
    rows.sort(key=lambda r: r["ts"])
    return rows


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 3)


def summarise(rows, now, window_s):
    sel = [r for r in rows if now - r["ts"] <= window_s]
    total = len(sel)
    fails = [r for r in sel if r["status"] >= 400]
    ok = [r for r in sel if r["status"] < 400]

    causes = {}
    for r in fails:
        c = cause_of(r["status"], r["path"])
        causes[c] = causes.get(c, 0) + 1
    cause_rows = [{"cause": c, "requests": n,
                   "share": round(n / len(fails), 4) if fails else None}
                  for c, n in sorted(causes.items(), key=lambda kv: -kv[1])]

    per_cust = {}
    for r in sel:
        c = per_cust.setdefault(r["cust"], {
            "requests": 0, "failures": 0, "in_bytes": 0, "out_bytes": 0,
            "ips": set(), "lat": [], "throttled": 0})
        c["requests"] += 1
        c["in_bytes"] += r["in"]
        c["out_bytes"] += r["out"]
        c["ips"].add(r["ip"])
        if r["status"] >= 400:
            c["failures"] += 1
        if r["status"] == 429:
            c["throttled"] += 1
        if r["status"] < 400 and r["req_s"] is not None:
            c["lat"].append(r["req_s"])
    cust_rows = []
    for name, c in sorted(per_cust.items(), key=lambda kv: -kv[1]["requests"]):
        cust_rows.append({
            "customer": name,
            "requests": c["requests"],
            "failures": c["failures"],
            "failure_rate": round(c["failures"] / c["requests"], 4) if c["requests"] else None,
            "throttled": c["throttled"],
            "distinct_ips": len(c["ips"]),
            "in_bytes": c["in_bytes"],
            "out_bytes": c["out_bytes"],
            "p50_s": pct(c["lat"], 0.5),
            "p95_s": pct(c["lat"], 0.95),
        })

    per_path = {}
    for r in sel:
        p = per_path.setdefault(r["path"], {"requests": 0, "failures": 0})
        p["requests"] += 1
        if r["status"] >= 400:
            p["failures"] += 1
    path_rows = [{"path": k, "requests": v["requests"], "failures": v["failures"],
                  "failure_rate": round(v["failures"] / v["requests"], 4)}
                 for k, v in sorted(per_path.items(), key=lambda kv: -kv[1]["requests"])]

    status_spread = {}
    for r in sel:
        status_spread[str(r["status"])] = status_spread.get(str(r["status"]), 0) + 1

    lat = [r["req_s"] for r in ok if r["req_s"] is not None]
    return {
        "window": None,
        "window_seconds": window_s,
        "requests": total,
        "successful": len(ok),
        "failures": len(fails),
        "failure_rate": round(len(fails) / total, 4) if total else None,
        "server_error_rate": round(
            sum(1 for r in sel if r["status"] >= 500) / total, 5) if total else None,
        "requests_per_minute": round(total / (window_s / 60.0), 3) if total else 0.0,
        "status_spread": status_spread,
        "failure_causes": cause_rows,
        "per_customer": cust_rows,
        "per_path": path_rows,
        "edge_latency_s": {"p50": pct(lat, 0.5), "p90": pct(lat, 0.9),
                           "p95": pct(lat, 0.95), "p99": pct(lat, 0.99)},
        "bytes_in": sum(r["in"] for r in sel),
        "bytes_out": sum(r["out"] for r in sel),
    }


def hist_percentiles(buckets, quantiles=(0.5, 0.9, 0.95, 0.99)):
    if not buckets:
        return {str(q): None for q in quantiles}
    bounds = sorted(buckets)
    total = buckets[bounds[-1]]
    if total <= 0:
        return {str(q): None for q in quantiles}
    out = {}
    for q in quantiles:
        want = q * total
        lower_b, lower_c = 0.0, 0.0
        chosen = None
        for b in bounds:
            c = buckets[b]
            if c >= want:
                if b == float("inf"):
                    chosen = lower_b
                else:
                    span = c - lower_c
                    frac = (want - lower_c) / span if span > 0 else 1.0
                    chosen = lower_b + (b - lower_b) * frac
                break
            lower_b, lower_c = b, c
        out[str(q)] = round(chosen, 4) if chosen is not None else None
    return out


def engine_metrics():
    try:
        with urllib.request.urlopen(METRICS_URL, timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}

    scalars, buckets = {}, {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+(\S+)$", line)
        if not m:
            continue
        name, labels, raw = m.group(1), m.group(2) or "", m.group(3)
        try:
            val = float(raw)
        except ValueError:
            continue
        if val != val:
            continue
        if name.endswith("_bucket"):
            fam = name[:-7]
            le = re.search(r'le="([^"]+)"', labels)
            if not le:
                continue
            b = float("inf") if le.group(1) in ("+Inf", "Inf") else float(le.group(1))
            buckets.setdefault(fam, {})
            buckets[fam][b] = buckets[fam].get(b, 0.0) + val
        else:
            scalars[name] = scalars.get(name, 0.0) + val

    q = scalars.get("vllm:prefix_cache_queries_total")
    h = scalars.get("vllm:prefix_cache_hits_total")
    prompt = scalars.get("vllm:prompt_tokens_total")
    cached = scalars.get("vllm:prompt_tokens_cached_total")
    return {
        "note": "counters are cumulative since the last engine start, not windowed",
        "engine_start_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(scalars.get("process_start_time_seconds", 0))) if
        scalars.get("process_start_time_seconds") else None,
        "requests_success_total": scalars.get("vllm:request_success_total"),
        "prompt_tokens_total": prompt,
        "generation_tokens_total": scalars.get("vllm:generation_tokens_total"),
        "prompt_tokens_cached_total": cached,
        "cached_token_share": round(cached / prompt, 4) if (prompt and cached) else None,
        "prefix_cache_queries_total": q,
        "prefix_cache_hits_total": h,
        "prefix_cache_hit_rate": round(h / q, 4) if (q and h is not None) else None,
        "preemptions_total": scalars.get("vllm:num_preemptions_total"),
        "requests_running": scalars.get("vllm:num_requests_running"),
        "requests_waiting": scalars.get("vllm:num_requests_waiting"),
        "kv_cache_usage_perc": scalars.get("vllm:kv_cache_usage_perc"),
        "tool_call_parser_invocations_total": scalars.get(
            "vllm:tool_call_parser_invocations_total"),
        "latency_seconds": {
            "ttft": hist_percentiles(buckets.get("vllm:time_to_first_token_seconds", {})),
            "tpot": hist_percentiles(
                buckets.get("vllm:request_time_per_output_token_seconds", {})),
            "itl": hist_percentiles(buckets.get("vllm:inter_token_latency_seconds", {})),
            "e2e": hist_percentiles(buckets.get("vllm:e2e_request_latency_seconds", {})),
            "queue": hist_percentiles(buckets.get("vllm:request_queue_time_seconds", {})),
            "prefill": hist_percentiles(
                buckets.get("vllm:request_prefill_time_seconds", {})),
            "decode": hist_percentiles(
                buckets.get("vllm:request_decode_time_seconds", {})),
        },
        "request_shape": {
            "prompt_tokens": hist_percentiles(
                buckets.get("vllm:request_prompt_tokens", {})),
            "generation_tokens": hist_percentiles(
                buckets.get("vllm:request_generation_tokens", {})),
            "client_max_tokens": hist_percentiles(
                buckets.get("vllm:request_params_max_tokens", {})),
            "iteration_tokens": hist_percentiles(
                buckets.get("vllm:iteration_tokens_total", {})),
        },
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)
    now = time.time()
    rows = read_rows()

    payload = {
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "log_sources": sorted(glob.glob(LOG_GLOB)),
        "rows_parsed": len(rows),
        "oldest_row_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(rows[0]["ts"])) if rows else None,
        "newest_row_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(rows[-1]["ts"])) if rows else None,
        "coverage_seconds": round(rows[-1]["ts"] - rows[0]["ts"], 1) if len(rows) > 1 else 0,
        "windows": {},
        "engine": engine_metrics(),
    }
    for label, secs in WINDOWS:
        w = summarise(rows, now, secs)
        w["window"] = label
        payload["windows"][label] = w

    # Honesty guard: a window wider than the log's own coverage is not a
    # 24-hour measurement, and a reader must not be able to mistake it for one.
    cov = payload["coverage_seconds"]
    payload["caveats"] = [
        "Access-log coverage is %.0f s; any window longer than that is "
        "partial, not a full-period measurement." % cov,
    ]
    for label, secs in WINDOWS:
        if secs > cov:
            payload["windows"][label]["partial"] = True
            payload["windows"][label]["actual_coverage_seconds"] = cov
        else:
            payload["windows"][label]["partial"] = False

    with open(os.path.join(out_dir, "prod_stats.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = ["Production statistics  (collected %s)" % payload["collected_utc"],
             "log coverage: %.0f s  rows: %d" % (cov, payload["rows_parsed"]), ""]
    for label, _ in WINDOWS:
        w = payload["windows"][label]
        lines.append("== window %s%s ==" % (label, "  [PARTIAL]" if w["partial"] else ""))
        lines.append("  requests=%d successful=%d failures=%d failure_rate=%s"
                     % (w["requests"], w["successful"], w["failures"], w["failure_rate"]))
        lines.append("  edge latency p50=%s p95=%s p99=%s"
                     % (w["edge_latency_s"]["p50"], w["edge_latency_s"]["p95"],
                        w["edge_latency_s"]["p99"]))
        for c in w["failure_causes"]:
            lines.append("    cause %-48s %5d  %s" % (c["cause"], c["requests"],
                                                      c["share"]))
        for c in w["per_customer"]:
            lines.append("    cust  %-20s req=%-6d fail=%-5d rate=%s"
                         % (c["customer"], c["requests"], c["failures"],
                            c["failure_rate"]))
        lines.append("")
    lines.append("== engine ==")
    lines.append(json.dumps(payload["engine"], indent=2))
    with open(os.path.join(out_dir, "prod_stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    w24 = payload["windows"]["24h"]
    print("production stats: %d requests in the log, 24h failure_rate=%s "
          "(partial=%s)" % (w24["requests"], w24["failure_rate"], w24["partial"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
