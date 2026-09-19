#!/usr/bin/env python3
"""Production monitor for the Kimi-K3 vLLM endpoint.

Three sources, because no single one answers "is the service healthy":
  - vLLM's Prometheus endpoint: engine-side throughput, queue depth, latency
    histograms, KV and prefix-cache state.
  - nginx's k3usage access log: who is calling, what status they got, and how
    long the edge took. This is the only place per-customer identity exists;
    vLLM sees one upstream credential for everybody.
  - rocm-smi: per-card utilisation and VRAM.

Read-only: it never sends inference traffic, so it cannot perturb what it
measures. Binds loopback only; nginx supplies TLS and basic auth.
"""

import calendar
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001")
METRICS_URL = VLLM_BASE.rstrip("/") + "/metrics"
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://cflox.store/v1")
ACCESS_LOG = os.environ.get("ACCESS_LOG", "/var/log/k3/usage.log")
CERT_PATH = os.environ.get("CERT_PATH", "/etc/letsencrypt/live/cflox.store/fullchain.pem")
# Used only to read /v1/models for the served name and context limit. Optional:
# everything else works without it.
API_KEY_FILE = os.environ.get("API_KEY_FILE", "/scratch/deploy/api-key.txt")
LISTEN_PORT = int(os.environ.get("DASH_PORT", 8080))
# Loopback only: the app has no auth of its own, nginx adds it.
LISTEN_ADDR = os.environ.get("DASH_ADDR", "127.0.0.1")

SCRAPE_INTERVAL = 2.0
GPU_INTERVAL = 5.0
FACTS_INTERVAL = 60.0
HISTORY_LEN = 1800          # 2s cadence -> 60 min of history
RATE_WINDOW_S = 60.0        # trailing window for headline throughput
# Chart points sent to the browser. The raw history is 600 samples, which at
# ~175 bytes per point is a 107 KB response re-fetched every 2s - 54 KB/s, and
# far more resolution than a 200px-tall canvas can draw. Downsampling to 180
# keeps every visible feature and caps the payload near 20 KB.
SERIES_POINTS = 180
ACCESS_WINDOW_S = 900.0     # 15 min of per-customer attribution
ACCESS_MAX_ROWS = 40000
# Longer reporting windows, rebuilt from the rotated logs on a slow timer.
ROLLUP_WINDOWS = (("1h", 3600.0), ("3h", 10800.0), ("24h", 86400.0))
ROLLUP_INTERVAL = 60.0
ROLLUP_MAX_BYTES = 200_000_000   # cap the rescan so a huge log cannot stall it
CERT_WARN_DAYS = 21         # certbot renews at 30; below this is a real problem

_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[-+0-9.eEnaN]+)\s*$"
)

# Metric families we care about. vLLM has renamed some of these across
# versions, so each logical name maps to an ordered list of candidates.
COUNTERS = {
    "prompt_tokens": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
    "generation_tokens": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "requests_success": ["vllm:request_success_total", "vllm:request_success"],
    "preemptions": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    # Prefix cache counters are token counts, not request counts.
    "cache_queries": ["vllm:prefix_cache_queries_total",
                      "vllm:gpu_prefix_cache_queries_total"],
    "cache_hits": ["vllm:prefix_cache_hits_total",
                   "vllm:gpu_prefix_cache_hits_total"],
    "cached_prompt_tokens": ["vllm:prompt_tokens_cached_total"],
    # Tool calling is a first-class feature of this endpoint (kimi_k3 parser),
    # so its volume belongs on the dashboard rather than only in a test report.
    "tool_calls": ["vllm:tool_call_parser_invocations_total"],
}
GAUGES = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    # Pre-0.9 vLLM exposed a ready-made hit rate instead of the counters above.
    "hit_rate_gauge": ["vllm:gpu_prefix_cache_hit_rate"],
}
# Why a request is waiting, not just that it is: "capacity" means the engine is
# genuinely full, which is an operator problem, while "deferred" is ordinary
# scheduling. The single num_requests_waiting gauge cannot tell them apart.
WAIT_REASONS = "vllm:num_requests_waiting_by_reason"
INFO = {
    "cache_config": ["vllm:cache_config_info"],
}
HISTOGRAMS = {
    "ttft": ["vllm:time_to_first_token_seconds"],
    "tpot": ["vllm:request_time_per_output_token_seconds",
             "vllm:time_per_output_token_seconds"],
    "itl": ["vllm:inter_token_latency_seconds"],
    "e2el": ["vllm:e2e_request_latency_seconds",
             "vllm:request_inference_time_seconds"],
    "queue": ["vllm:request_queue_time_seconds"],
    # Prefill and decode split the request: a slow prefill is a prompt-size or
    # cache-miss problem, a slow decode is a batching problem. They need
    # different fixes, so showing only end-to-end hides which one to apply.
    "prefill": ["vllm:request_prefill_time_seconds"],
    "decode": ["vllm:request_decode_time_seconds"],
    "prompt_len": ["vllm:request_prompt_tokens"],
    "gen_len": ["vllm:request_generation_tokens"],
    # What clients ASK for, as opposed to what they get. vLLM reserves
    # prompt+max_tokens against one window, so a fleet defaulting to a huge
    # max_tokens is the documented cause of spurious 400s.
    "client_max_tokens": ["vllm:request_params_max_tokens"],
    "batch_tokens": ["vllm:iteration_tokens_total"],
}


def _to_float(raw):
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _counter_delta(prev, cur, field):
    """Change in a monotonic counter between two samples, or None if unusable."""
    a, b = prev.get(field), cur.get(field)
    if a is None or b is None or b < a:
        return None
    return b - a


def _round(v, digits):
    return None if v is None else round(v, digits)


def downsample(points, limit):
    """Bucket a chart series down to `limit` points.

    Rate fields are averaged, because the mean over the bucket is the honest
    summary of a rate. Concurrency, queue depth and KV usage take the bucket
    MAXIMUM instead: a two-sample queue spike is exactly the thing an operator
    is looking for, and averaging would erase it.
    """
    if len(points) <= limit:
        return points
    mean_fields = ("total_tpm", "prompt_tpm", "gen_tpm", "req_pm", "cache_hit_rate")
    max_fields = ("running", "waiting", "kv_usage")
    n, out = len(points), []
    for i in range(limit):
        lo = i * n // limit
        hi = max(lo + 1, (i + 1) * n // limit)
        chunk = points[lo:hi]
        row = {"ts": chunk[-1]["ts"]}
        for f in mean_fields:
            vals = [p[f] for p in chunk if p.get(f) is not None]
            row[f] = (sum(vals) / len(vals)) if vals else None
        for f in max_fields:
            vals = [p[f] for p in chunk if p.get(f) is not None]
            row[f] = max(vals) if vals else None
        out.append(row)
    return out


def percentile(values, q):
    """Nearest-rank percentile of an unsorted list. None when empty."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def parse_prometheus(text):
    """Return (scalars, buckets, info).

    scalars: metric name -> summed value across all label sets.
    buckets: metric family -> {le_float: summed cumulative count}.
    info:    metric name -> label dict (for *_info metrics, whose payload is
             carried in the labels rather than the value).
    """
    scalars, buckets, info = {}, {}, {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        value = _to_float(m.group("value"))
        if value is None:
            continue
        if name.endswith("_bucket"):
            family = name[: -len("_bucket")]
            labels = m.group("labels") or ""
            le = re.search(r'le="([^"]+)"', labels)
            if not le:
                continue
            bound = float("inf") if le.group(1) in ("+Inf", "Inf") else _to_float(le.group(1))
            if bound is None:
                continue
            buckets.setdefault(family, {})
            buckets[family][bound] = buckets[family].get(bound, 0.0) + value
        elif name.endswith("_info"):
            info[name] = dict(_LABEL_RE.findall(m.group("labels") or ""))
        else:
            scalars[name] = scalars.get(name, 0.0) + value
    return scalars, buckets, info


def pick(mapping, key, store):
    """Resolve the first present candidate name for a logical metric."""
    for candidate in mapping[key]:
        if candidate in store:
            return candidate, store[candidate]
    return None, None


def hist_sum_count(prev, cur):
    """Exact mean over the window from a histogram's own _sum and _count.

    Percentiles interpolated from bucket edges are approximations; the mean is
    not, because sum and count are plain counters. Reported alongside the
    percentiles so a reader can see when the tail is dragging the average.
    """
    if not cur:
        return None
    ds = cur[0] - (prev[0] if prev else 0.0)
    dc = cur[1] - (prev[1] if prev else 0.0)
    if dc <= 0 or ds < 0:
        return None
    return ds / dc


def percentiles_from_bucket_delta(prev, cur, quantiles=(0.5, 0.9, 0.95, 0.99)):
    """Interpolate percentiles from the change in cumulative histogram buckets.

    Returns seconds, or None per quantile when the window holds no samples.
    Prometheus buckets are cumulative in `le`, and remain so after
    differencing two snapshots, so a simple scan is valid.
    """
    if not cur:
        return {q: None for q in quantiles}
    bounds = sorted(cur.keys())
    deltas, running, ok = [], 0.0, False
    for b in bounds:
        d = cur.get(b, 0.0) - (prev.get(b, 0.0) if prev else 0.0)
        if d < 0:                      # counter reset (server restart)
            return {q: None for q in quantiles}
        if d > 0:
            ok = True
        running = max(running, d)      # enforce monotonicity defensively
        deltas.append((b, running))
    total = deltas[-1][1] if deltas else 0.0
    if not ok or total <= 0:
        return {q: None for q in quantiles}

    out = {}
    for q in quantiles:
        want = q * total
        chosen, lower_bound, lower_count = None, 0.0, 0.0
        for bound, cum in deltas:
            if cum >= want:
                if bound == float("inf"):
                    chosen = lower_bound
                else:
                    span = cum - lower_count
                    frac = (want - lower_count) / span if span > 0 else 1.0
                    chosen = lower_bound + (bound - lower_bound) * frac
                break
            lower_bound, lower_count = bound, cum
        out[q] = chosen
    return out


_ACCESS_RE = re.compile(
    r"^(?P<ts>\S+)\s+cust=(?P<cust>\S*)\s+status=(?P<status>\d+)\s+"
    r"path=(?P<path>\S+)\s+req_ms=(?P<req>\S+)\s+up_ms=(?P<up>\S+)\s+"
    r"in=(?P<in>\d+)\s+out=(?P<out>\d+)\s+ip=(?P<ip>\S+)\s*$"
)


class AccessTail:
    """Incremental reader for nginx's k3usage log.

    Per-customer identity, status codes and edge latency only exist here: the
    proxy swaps every customer key for one upstream credential, so vLLM's own
    metrics cannot attribute anything. Survives rotation by watching the inode
    and reseeking to 0 when the file shrinks.

    The field is named req_ms in the log format but nginx's $request_time is
    in SECONDS; it is converted once, here, so the rest of the code can treat
    every latency as seconds.
    """

    def __init__(self, path):
        self.path = path
        self.rows = deque(maxlen=ACCESS_MAX_ROWS)
        self.lock = threading.Lock()
        self.pos = 0
        self.inode = None
        self.error = None
        self.rotations = 0
        # Skip whatever is already on disk at startup, but keep enough tail to
        # fill the first window rather than showing an empty dashboard.
        self._primed = False

    def poll(self):
        try:
            st = os.stat(self.path)
        except OSError as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return
        if self.inode is not None and st.st_ino != self.inode:
            self.rotations += 1
            self.pos = 0
        elif st.st_size < self.pos:          # truncated in place
            self.pos = 0
        self.inode = st.st_ino

        if not self._primed:
            # Start ~4 MB back so a long-lived log does not cost a full parse.
            self.pos = max(0, st.st_size - 4_000_000)
            self._primed = True

        try:
            with open(self.path, "r", errors="replace") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError as e:
            self.error = "%s: %s" % (type(e).__name__, e)
            return

        self.error = None
        parsed = []
        for line in chunk.splitlines():
            m = _ACCESS_RE.match(line.strip())
            if not m:
                continue
            ts = self._parse_ts(m.group("ts"))
            if ts is None:
                continue
            parsed.append({
                "ts": ts,
                "cust": m.group("cust") or "",
                "status": int(m.group("status")),
                "path": m.group("path"),
                "req_s": _to_float(m.group("req")),
                "up_s": _to_float(m.group("up")),
                "in": int(m.group("in")),
                "out": int(m.group("out")),
                "ip": m.group("ip"),
            })
        if parsed:
            with self.lock:
                self.rows.extend(parsed)

    @staticmethod
    def _parse_ts(raw):
        # $time_iso8601 is like 2026-09-18T15:18:59+00:00.
        try:
            from datetime import datetime
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            return None

    def view(self, window_s=ACCESS_WINDOW_S):
        now = time.time()
        with self.lock:
            rows = [r for r in self.rows if now - r["ts"] <= window_s]
            total_rows = len(self.rows)

        by_status, per_cust = {}, {}
        for r in rows:
            cls = "%dxx" % (r["status"] // 100)
            by_status[cls] = by_status.get(cls, 0) + 1
            if r["status"] in (401, 403, 429):
                key = "s%d" % r["status"]
                by_status[key] = by_status.get(key, 0) + 1

            name = r["cust"] or "(unauthenticated)"
            c = per_cust.setdefault(name, {
                "customer": name, "requests": 0, "errors": 0, "in_bytes": 0,
                "out_bytes": 0, "last_seen": 0.0, "ips": set(), "lat": [],
                "throttled": 0,
            })
            c["requests"] += 1
            c["in_bytes"] += r["in"]
            c["out_bytes"] += r["out"]
            c["last_seen"] = max(c["last_seen"], r["ts"])
            c["ips"].add(r["ip"])
            if r["status"] >= 400:
                c["errors"] += 1
            if r["status"] == 429:
                c["throttled"] += 1
            # Only successful, upstream-served calls describe serving latency.
            if r["status"] < 400 and r["req_s"] is not None:
                c["lat"].append(r["req_s"])

        customers = []
        for c in per_cust.values():
            customers.append({
                "customer": c["customer"],
                "requests": c["requests"],
                "errors": c["errors"],
                "throttled": c["throttled"],
                "share": c["requests"] / len(rows) if rows else None,
                "p50_s": percentile(c["lat"], 0.5),
                "p95_s": percentile(c["lat"], 0.95),
                "in_bytes": c["in_bytes"],
                "out_bytes": c["out_bytes"],
                "distinct_ips": len(c["ips"]),
                "last_seen": c["last_seen"] or None,
            })
        customers.sort(key=lambda c: c["requests"], reverse=True)

        errors = [{"ts": r["ts"], "cust": r["cust"] or "(none)", "status": r["status"],
                   "path": r["path"], "ip": r["ip"]}
                  for r in rows if r["status"] >= 400][-14:]
        errors.reverse()

        n = len(rows)
        bad = by_status.get("4xx", 0) + by_status.get("5xx", 0)
        return {
            "available": self.error is None,
            "error": self.error,
            "path": self.path,
            "window_s": window_s,
            "requests": n,
            "req_pm": (n / window_s * 60) if n else 0.0,
            "by_status": by_status,
            "error_rate": (bad / n) if n else None,
            "server_error_rate": (by_status.get("5xx", 0) / n) if n else None,
            "customers": customers,
            "recent_errors": errors,
            "rows_buffered": total_rows,
            "rotations": self.rotations,
        }


def failure_cause(status):
    """Group a rejection by what an operator would actually do about it.

    Derived from HTTP status alone, because that is what the access log
    carries. nginx rejects 401/403/429 itself and never records an upstream
    body, so a finer breakdown of 400s would have to be invented rather than
    measured.
    """
    return {
        400: "400 body rejected by the engine",
        401: "401 missing or invalid customer key",
        403: "403 path not on the edge allowlist",
        404: "404 unknown route",
        413: "413 body over the 256 MB cap",
        429: "429 per-customer rate or concurrency cap",
        499: "499 client disconnected early",
    }.get(status, ("%dxx " % (status // 100)) + ("server error" if status >= 500
                                                 else "client error"))


class LogRollup:
    """Hour/day aggregates over the whole retained log, rotations included.

    AccessTail deliberately keeps only a short in-memory tail, which cannot
    answer "what was the failure rate today". This walks the rotated and
    gzipped files on a slow timer so the longer windows survive a midnight
    rotation, and marks a window partial when the log does not reach back far
    enough to support it -- an under-covered window must not be mistaken for a
    real 24-hour measurement.
    """

    def __init__(self, path, windows=ROLLUP_WINDOWS):
        self.path = path
        self.windows = windows
        self.lock = threading.Lock()
        self.data = {"available": False, "error": "not sampled yet", "windows": {}}

    def _iter_lines(self):
        import glob
        import gzip
        files = sorted(glob.glob(self.path + "*"))
        budget = ROLLUP_MAX_BYTES
        for p in reversed(files):        # newest first, stop once far enough back
            if budget <= 0:
                return
            opener = gzip.open if p.endswith(".gz") else open
            try:
                size = os.path.getsize(p)
                with opener(p, "rt", errors="replace") as f:
                    if not p.endswith(".gz") and size > budget:
                        f.seek(size - budget)
                        f.readline()     # discard the partial line
                    budget -= size
                    for line in f:
                        yield line
            except OSError:
                continue

    def refresh(self):
        now = time.time()
        longest = max(s for _, s in self.windows)
        rows = []
        try:
            for line in self._iter_lines():
                m = _ACCESS_RE.match(line.strip())
                if not m:
                    continue
                ts = AccessTail._parse_ts(m.group("ts"))
                if ts is None or now - ts > longest:
                    continue
                rows.append((ts, m.group("cust") or "(unauthenticated)",
                             int(m.group("status")), m.group("path").split("?")[0],
                             _to_float(m.group("req")), int(m.group("in")),
                             int(m.group("out"))))
        except Exception as e:
            with self.lock:
                self.data = {"available": False,
                             "error": "%s: %s" % (type(e).__name__, e),
                             "windows": {}}
            return

        coverage = (now - min(r[0] for r in rows)) if rows else 0.0
        out = {}
        for label, secs in self.windows:
            sel = [r for r in rows if now - r[0] <= secs]
            total = len(sel)
            fails = [r for r in sel if r[2] >= 400]
            causes = {}
            for r in fails:
                c = failure_cause(r[2])
                causes[c] = causes.get(c, 0) + 1
            paths = {}
            for r in sel:
                p = paths.setdefault(r[3], [0, 0])
                p[0] += 1
                if r[2] >= 400:
                    p[1] += 1
            lat = [r[4] for r in sel if r[2] < 400 and r[4] is not None]
            out[label] = {
                "window_s": secs,
                "partial": secs > coverage,
                "requests": total,
                "failures": len(fails),
                "failure_rate": (len(fails) / total) if total else None,
                "server_error_rate": (sum(1 for r in sel if r[2] >= 500) / total)
                                     if total else None,
                "req_pm": (total / secs * 60) if total else 0.0,
                "tokens_in_bytes": sum(r[5] for r in sel),
                "tokens_out_bytes": sum(r[6] for r in sel),
                "p50_s": percentile(lat, 0.5),
                "p95_s": percentile(lat, 0.95),
                "p99_s": percentile(lat, 0.99),
                "causes": [{"cause": c, "requests": n,
                            "share": n / len(fails) if fails else None}
                           for c, n in sorted(causes.items(), key=lambda kv: -kv[1])],
                "paths": [{"path": k, "requests": v[0], "failures": v[1],
                           "failure_rate": v[1] / v[0]}
                          for k, v in sorted(paths.items(), key=lambda kv: -kv[1][0])][:8],
            }
        with self.lock:
            self.data = {"available": True, "error": None, "coverage_s": coverage,
                         "rows": len(rows), "sampled_at": now, "windows": out}

    def view(self):
        with self.lock:
            return dict(self.data)


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.samples = deque(maxlen=HISTORY_LEN)
        self.gpu = {"per_card": [], "mean_use": None, "mean_vram": None, "ts": None,
                    "error": None}
        self.server_state = "connecting"
        self.server_detail = "no scrape yet"
        self.last_error = None
        self.scrape_count = 0
        self.reset_count = 0
        self.started = time.time()
        self.cache_config = {}
        self.facts = {"model": None, "max_model_len": None, "cert_not_after": None,
                      "cert_days_left": None, "error": None}
        self._prev = None

    # ---------- vLLM metrics ----------

    def scrape_once(self):
        try:
            with urllib.request.urlopen(METRICS_URL, timeout=4) as r:
                body = r.read().decode("utf-8", "replace")
            code = 200
        except urllib.error.HTTPError as e:
            self._degrade("http_error", "metrics returned HTTP %s" % e.code)
            return
        except Exception as e:                       # refused, DNS, timeout
            self._degrade("unreachable", "%s: %s" % (type(e).__name__, e))
            return

        scalars, buckets, info = parse_prometheus(body)
        if not scalars and not buckets:
            self._degrade("no_metrics", "endpoint returned no vllm metrics")
            return

        now = time.time()
        cur = {"ts": now, "counters": {}, "gauges": {}, "buckets": {}, "hsum": {}}
        for key in COUNTERS:
            _, v = pick(COUNTERS, key, scalars)
            cur["counters"][key] = v
        for key in GAUGES:
            _, v = pick(GAUGES, key, scalars)
            cur["gauges"][key] = v
        for key in HISTOGRAMS:
            name, _ = pick(HISTOGRAMS, key, buckets)
            cur["buckets"][key] = buckets.get(name, {}) if name else {}
            # The family that carried buckets is the one whose sum/count to
            # trust; a fallback candidate may exist but be empty.
            s = scalars.get((name or "") + "_sum")
            c = scalars.get((name or "") + "_count")
            cur["hsum"][key] = (s, c) if (s is not None and c is not None) else None
        _, cfg = pick(INFO, "cache_config", info)

        wait_reasons = {}
        for full, v in scalars.items():
            if full == WAIT_REASONS:
                wait_reasons["unlabelled"] = v
        for line in body.splitlines():
            if not line.startswith(WAIT_REASONS + "{"):
                continue
            m = _SAMPLE_RE.match(line)
            if not m:
                continue
            labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
            reason = labels.get("reason") or "unknown"
            val = _to_float(m.group("value"))
            if val is not None:
                wait_reasons[reason] = wait_reasons.get(reason, 0.0) + val
        cur["wait_reasons"] = wait_reasons

        prev = self._prev
        reset = False
        if prev:
            for key, v in cur["counters"].items():
                pv = prev["counters"].get(key)
                if v is not None and pv is not None and v < pv - 1e-9:
                    reset = True
                    break
        if reset:
            self.reset_count += 1
            self.samples.clear()
            prev = None

        rates = {}
        if prev:
            dt = cur["ts"] - prev["ts"]
            if dt > 0:
                for key, v in cur["counters"].items():
                    pv = prev["counters"].get(key)
                    rates[key] = (v - pv) / dt if (v is not None and pv is not None) else None

        row = {
            "ts": now,
            "running": cur["gauges"].get("running"),
            "waiting": cur["gauges"].get("waiting"),
            "kv_usage": cur["gauges"].get("kv_usage"),
            "prompt_total": cur["counters"].get("prompt_tokens"),
            "gen_total": cur["counters"].get("generation_tokens"),
            "success_total": cur["counters"].get("requests_success"),
            "preempt_total": cur["counters"].get("preemptions"),
            "prompt_rate": rates.get("prompt_tokens"),
            "gen_rate": rates.get("generation_tokens"),
            "req_rate": rates.get("requests_success"),
            "cache_query_total": cur["counters"].get("cache_queries"),
            "cache_hit_total": cur["counters"].get("cache_hits"),
            "cached_prompt_total": cur["counters"].get("cached_prompt_tokens"),
            "hit_rate_gauge": cur["gauges"].get("hit_rate_gauge"),
            "tool_calls_total": cur["counters"].get("tool_calls"),
            "tool_call_rate": rates.get("tool_calls"),
            "wait_reasons": cur["wait_reasons"] or None,
        }

        with self.lock:
            self.samples.append(row)
            self._prev = cur
            if cfg:
                self.cache_config = cfg
            self.scrape_count += 1
            self.server_state = "up"
            self.server_detail = "scraped ok (HTTP %d)" % code
            self.last_error = None

    def _degrade(self, state, detail):
        with self.lock:
            self.server_state = state
            self.server_detail = detail
            self.last_error = detail
            self._prev = None      # force a clean baseline on reconnect

    # ---------- endpoint facts ----------

    def sample_facts(self):
        """Served model name, context ceiling, and TLS cert expiry.

        These change rarely but they are the first things asked when a client
        reports a problem ("what model, what context limit, is the cert live").
        """
        facts = {"model": None, "max_model_len": None, "cert_not_after": None,
                 "cert_days_left": None, "error": None}
        key = None
        try:
            with open(API_KEY_FILE) as f:
                key = f.read().strip()
        except OSError:
            pass
        if key:
            try:
                req = urllib.request.Request(
                    VLLM_BASE.rstrip("/") + "/v1/models",
                    headers={"Authorization": "Bearer " + key})
                with urllib.request.urlopen(req, timeout=5) as r:
                    d = json.loads(r.read())
                entry = (d.get("data") or [{}])[0]
                facts["model"] = entry.get("id")
                facts["max_model_len"] = entry.get("max_model_len")
            except Exception as e:
                facts["error"] = "models: %s" % type(e).__name__

        try:
            out = subprocess.run(["openssl", "x509", "-in", CERT_PATH,
                                  "-noout", "-enddate"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and "notAfter=" in out.stdout:
                raw = out.stdout.split("notAfter=", 1)[1].strip()
                # e.g. "Dec 17 13:56:54 2026 GMT" - always GMT from openssl.
                tm = time.strptime(raw.replace(" GMT", ""), "%b %d %H:%M:%S %Y")
                exp = calendar.timegm(tm)
                facts["cert_not_after"] = exp
                facts["cert_days_left"] = (exp - time.time()) / 86400.0
        except Exception as e:
            facts["error"] = (facts["error"] or "") + " cert: %s" % type(e).__name__

        with self.lock:
            self.facts = facts

    # ---------- GPU ----------

    def sample_gpu(self):
        use = self._rocm_csv(["rocm-smi", "--showuse", "--csv"], "GPU use (%)")
        vram = self._rocm_csv(["rocm-smi", "--showmemuse", "--csv"],
                              "GPU Memory Allocated (VRAM%)")
        cards = sorted(set(list(use.keys()) + list(vram.keys())))
        per_card = [{"card": c, "use": use.get(c), "vram": vram.get(c)} for c in cards]
        uses = [v for v in use.values() if v is not None]
        vrams = [v for v in vram.values() if v is not None]
        with self.lock:
            self.gpu = {
                "per_card": per_card,
                "mean_use": sum(uses) / len(uses) if uses else None,
                "mean_vram": sum(vrams) / len(vrams) if vrams else None,
                "ts": time.time(),
                "error": None if per_card else "rocm-smi returned no rows",
            }

    @staticmethod
    def _rocm_csv(cmd, column):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if out.returncode != 0:
                return {}
            lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
            if not lines:
                return {}
            header = [h.strip() for h in lines[0].split(",")]
            if column not in header:
                return {}
            idx = header.index(column)
            res = {}
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) <= idx or not parts[0].startswith("card"):
                    continue
                res[parts[0]] = _to_float(parts[idx])
            return res
        except Exception:
            return {}

    # ---------- derived view ----------

    def snapshot(self):
        with self.lock:
            samples = list(self.samples)
            gpu = dict(self.gpu)
            state = self.server_state
            detail = self.server_detail
            scrapes = self.scrape_count
            resets = self.reset_count
            prev_buckets = self._prev["buckets"] if self._prev else {}
            prev_hsum = self._prev["hsum"] if self._prev else {}
            uptime = time.time() - self.started
            cache_cfg = dict(self.cache_config)
            facts = dict(self.facts)

        # Each sample carries its own histogram snapshot for windowed
        # percentiles; strip those internals from the wire payload.
        latest = {k: v for k, v in samples[-1].items()
                  if not k.startswith("_")} if samples else None

        # Windowed throughput from counter endpoints (robust to scrape jitter).
        window = None
        if len(samples) >= 2:
            newest = samples[-1]
            anchor = samples[0]
            for s in reversed(samples[:-1]):
                if newest["ts"] - s["ts"] >= RATE_WINDOW_S:
                    anchor = s
                    break
            dt = newest["ts"] - anchor["ts"]
            if dt > 0:
                def delta(field):
                    a, b = anchor.get(field), newest.get(field)
                    if a is None or b is None or b < a:
                        return None
                    return b - a

                def rate(field):
                    d = delta(field)
                    return None if d is None else d / dt
                p = rate("prompt_total")
                g = rate("gen_total")
                r = rate("success_total")
                tot = (p + g) if (p is not None and g is not None) else None
                window = {
                    "seconds": dt,
                    "prompt_tps": p,
                    "gen_tps": g,
                    "total_tps": tot,
                    "req_ps": r,
                    "total_tpm": tot * 60 if tot is not None else None,
                    "prompt_tpm": p * 60 if p is not None else None,
                    "gen_tpm": g * 60 if g is not None else None,
                    "req_pm": r * 60 if r is not None else None,
                }
                dq, dh = delta("cache_query_total"), delta("cache_hit_total")
                window["cache_queried_tokens"] = dq
                window["cache_hit_tokens"] = dh
                window["cache_hit_rate"] = (dh / dq) if (dq and dh is not None) else None
                window["cached_tps"] = rate("cached_prompt_total")
                window["completed"] = delta("success_total")
                window["preempted"] = delta("preempt_total")

        # Latency percentiles over the trailing window, from bucket deltas.
        lat = {}
        if samples:
            base_idx = 0
            for i, s in enumerate(samples):
                if samples[-1]["ts"] - s["ts"] <= RATE_WINDOW_S:
                    base_idx = i
                    break
            base = samples[base_idx].get("_buckets_snapshot")
            base_hsum = samples[base_idx].get("_hsum_snapshot")
            for key in HISTOGRAMS:
                cur = prev_buckets.get(key, {})
                got = percentiles_from_bucket_delta(base.get(key) if base else None, cur)
                lat[key] = {
                    "avg": hist_sum_count(
                        (base_hsum or {}).get(key), prev_hsum.get(key)),
                    "p50": got.get(0.5),
                    "p90": got.get(0.9),
                    "p95": got.get(0.95),
                    "p99": got.get(0.99),
                }

        recent = samples[-600:]
        series = []
        for i, s in enumerate(recent):
            # Hit rate is a ratio of two counters, so it only means anything
            # over an interval: diff against the previous sample.
            hit_rate = None
            if i:
                prev_s = recent[i - 1]
                dq = _counter_delta(prev_s, s, "cache_query_total")
                dh = _counter_delta(prev_s, s, "cache_hit_total")
                if dq and dh is not None:
                    hit_rate = dh / dq
            if hit_rate is None and s.get("hit_rate_gauge") is not None:
                hit_rate = s["hit_rate_gauge"]
            series.append({
                "ts": s["ts"],
                "total_tpm": ((s["prompt_rate"] or 0) + (s["gen_rate"] or 0)) * 60
                             if (s["prompt_rate"] is not None or s["gen_rate"] is not None)
                             else None,
                "prompt_tpm": s["prompt_rate"] * 60 if s["prompt_rate"] is not None else None,
                "gen_tpm": s["gen_rate"] * 60 if s["gen_rate"] is not None else None,
                "req_pm": s["req_rate"] * 60 if s["req_rate"] is not None else None,
                "running": s["running"],
                "waiting": s["waiting"],
                "kv_usage": s["kv_usage"],
                "cache_hit_rate": hit_rate,
            })
        # Downsample, then drop the digits no chart can show. Full precision
        # here was two thirds of the response body.
        series = [{
            "ts": round(p["ts"], 1),
            "total_tpm": _round(p["total_tpm"], 0),
            "prompt_tpm": _round(p["prompt_tpm"], 0),
            "gen_tpm": _round(p["gen_tpm"], 0),
            "req_pm": _round(p["req_pm"], 2),
            "running": _round(p["running"], 1),
            "waiting": _round(p["waiting"], 1),
            "kv_usage": _round(p["kv_usage"], 5),
            "cache_hit_rate": _round(p["cache_hit_rate"], 4),
        } for p in downsample(series, SERIES_POINTS)]

        access = ACCESS.view()
        capacity = self.capacity_view(samples[-1] if samples else None, window, cache_cfg)
        return {
            "now": time.time(),
            "server": {"state": state, "detail": detail, "url": VLLM_BASE,
                       "scrapes": scrapes, "counter_resets": resets,
                       "dash_uptime_s": uptime},
            "endpoint": {
                "public_url": PUBLIC_URL,
                "model": facts.get("model"),
                "max_model_len": facts.get("max_model_len"),
                "cert_not_after": facts.get("cert_not_after"),
                "cert_days_left": facts.get("cert_days_left"),
                "cert_warn_days": CERT_WARN_DAYS,
                "facts_error": facts.get("error"),
            },
            "capacity": capacity,
            "window": window,
            "latest": latest,
            "latency": lat,
            "gpu": gpu,
            "series": series,
            "access": access,
            "rollup": ROLLUP.view(),
            "alerts": self.alerts(state, window, latest, capacity, access, facts),
        }

    @staticmethod
    def capacity_view(latest, window, cfg):
        """KV headroom and prefix-cache reuse.

        Both are capacity signals rather than curiosities: KV usage is what
        decides whether the next long request queues, and the hit rate is
        what keeps prefill cost down for agent harnesses that resend a large
        shared prefix on every turn.
        """
        latest = latest or {}
        q, h = latest.get("cache_query_total"), latest.get("cache_hit_total")
        lifetime = (h / q) if (q and h is not None) else None
        if lifetime is None and latest.get("hit_rate_gauge") is not None:
            lifetime = latest["hit_rate_gauge"]

        def cfg_num(key):
            return _to_float(cfg.get(key)) if cfg.get(key) is not None else None

        capacity = cfg_num("kv_cache_size_tokens")
        blocks, block_size = cfg_num("num_gpu_blocks"), cfg_num("block_size")
        if capacity is None and blocks is not None and block_size is not None:
            capacity = blocks * block_size
        usage = latest.get("kv_usage")

        enabled = cfg.get("enable_prefix_caching")
        return {
            "prefix_caching": None if enabled is None else enabled.lower() == "true",
            "supported": q is not None or latest.get("hit_rate_gauge") is not None,
            "lifetime_hit_rate": lifetime,
            "window_hit_rate": (window or {}).get("cache_hit_rate"),
            "cached_prompt_tokens": latest.get("cached_prompt_total"),
            "capacity_tokens": capacity,
            "used_tokens": (capacity * usage)
                           if (capacity is not None and usage is not None) else None,
            "kv_usage": usage,
            "block_size": block_size,
            "preemptions": latest.get("preempt_total"),
            "window_preempted": (window or {}).get("preempted"),
        }

    @staticmethod
    def alerts(state, window, latest, capacity, access, facts):
        """Only conditions an operator would act on. No workload targets.

        Deliberately excludes "throughput is below X": this endpoint serves
        interactive agent traffic, so low throughput means nobody is asking,
        not that anything is wrong.
        """
        out = []
        latest = latest or {}
        if state != "up":
            out.append({"level": "bad",
                        "text": "Engine is not answering /metrics (%s). Serving is "
                                "probably down - check `systemctl status k3`." % state})
        waiting = latest.get("waiting")
        if waiting:
            out.append({"level": "warn",
                        "text": "%d request(s) queued for a decode slot. Sustained "
                                "queueing means offered load exceeds capacity."
                                % int(waiting)})
        kv = capacity.get("kv_usage")
        if kv is not None and kv >= 0.90:
            out.append({"level": "warn",
                        "text": "KV cache %.0f%% full. Long requests will queue or "
                                "be preempted." % (kv * 100)})
        if capacity.get("window_preempted"):
            out.append({"level": "warn",
                        "text": "%d preemption(s) in the last minute - the scheduler "
                                "is evicting work under KV pressure."
                                % int(capacity["window_preempted"])})
        if capacity.get("prefix_caching") is False:
            out.append({"level": "warn",
                        "text": "Prefix caching is disabled, so every agent turn "
                                "re-prefills its whole shared prefix."})
        five = (access.get("by_status") or {}).get("5xx", 0)
        if five:
            out.append({"level": "bad",
                        "text": "%d server error(s) (5xx) returned to clients in the "
                                "last 15 min." % five})
        throttled = (access.get("by_status") or {}).get("s429", 0)
        if throttled:
            out.append({"level": "warn",
                        "text": "%d request(s) rate-limited (429). A customer is "
                                "hitting its per-key cap." % throttled})
        unauth = (access.get("by_status") or {}).get("s401", 0)
        if unauth >= 20:
            out.append({"level": "warn",
                        "text": "%d unauthenticated request(s) (401) in the last "
                                "15 min - a misconfigured client, or probing."
                                % unauth})
        days = facts.get("cert_days_left")
        if days is not None and days < CERT_WARN_DAYS:
            out.append({"level": "bad" if days < 7 else "warn",
                        "text": "TLS certificate expires in %.1f days and certbot "
                                "should have renewed by now." % days})
        if not access.get("available"):
            out.append({"level": "warn",
                        "text": "Cannot read the nginx usage log (%s), so per-customer "
                                "attribution is unavailable." % access.get("error")})
        return out

    # ---------- threads ----------

    def run_metrics_loop(self):
        while True:
            try:
                self.scrape_once()
                # Attach the freshest bucket snapshot to the newest sample so
                # windowed percentiles have a historical baseline to diff.
                with self.lock:
                    if self.samples and self._prev:
                        self.samples[-1]["_buckets_snapshot"] = self._prev["buckets"]
                        self.samples[-1]["_hsum_snapshot"] = self._prev["hsum"]
            except Exception as e:
                self._degrade("monitor_error", "%s: %s" % (type(e).__name__, e))
            time.sleep(SCRAPE_INTERVAL)

    def run_access_loop(self):
        while True:
            try:
                ACCESS.poll()
            except Exception:
                pass
            time.sleep(SCRAPE_INTERVAL)

    def run_rollup_loop(self):
        while True:
            try:
                ROLLUP.refresh()
            except Exception:
                pass
            time.sleep(ROLLUP_INTERVAL)

    def run_gpu_loop(self):
        while True:
            try:
                self.sample_gpu()
            except Exception as e:
                with self.lock:
                    self.gpu["error"] = "%s: %s" % (type(e).__name__, e)
            time.sleep(GPU_INTERVAL)

    def run_facts_loop(self):
        while True:
            try:
                self.sample_facts()
            except Exception:
                pass
            time.sleep(FACTS_INTERVAL)


ACCESS = AccessTail(ACCESS_LOG)
ROLLUP = LogRollup(ACCESS_LOG)
MON = Monitor()

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                with open(HTML_PATH, "r") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/api/state":
                payload = json.dumps(MON.snapshot(), default=lambda o: None)
                self._send(200, payload, "application/json")
            elif path == "/api/health":
                self._send(200, json.dumps({"ok": True}), "application/json")
            else:
                self._send(404, json.dumps({"error": "not found"}), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": "%s: %s" % (type(e).__name__, e)}),
                       "application/json")


def main():
    threading.Thread(target=MON.run_metrics_loop, daemon=True).start()
    threading.Thread(target=MON.run_access_loop, daemon=True).start()
    threading.Thread(target=MON.run_rollup_loop, daemon=True).start()
    threading.Thread(target=MON.run_gpu_loop, daemon=True).start()
    threading.Thread(target=MON.run_facts_loop, daemon=True).start()
    srv = ThreadingHTTPServer((LISTEN_ADDR, LISTEN_PORT), Handler)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
