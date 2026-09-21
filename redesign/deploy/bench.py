"""Edge benchmark for the live Kimi-K3 stack. Uses the dedicated bench key."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import http.client
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOST = "api.cflowx.in"
MODEL = "FW-Kimi-K3"
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


class Edge(http.client.HTTPSConnection):
    def __init__(self, timeout: float = 120) -> None:
        ctx = ssl._create_unverified_context()
        super().__init__("127.0.0.1", 443, timeout=timeout, context=ctx)

    def connect(self) -> None:
        self.sock = self._context.wrap_socket(
            __import__("socket").create_connection((self.host, self.port), self.timeout),
            server_hostname=HOST,
        )


def _key() -> str:
    raw = os.environ.get("BENCH_KEY", "").strip()
    if raw:
        return raw
    path = Path("/scratch/deploy/bench.key")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("key="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("BENCH_KEY or /scratch/deploy/bench.key is required")


def _request(key: str, method: str, path: str, body: dict | None = None, stream: bool = False) -> tuple[int, dict | str, float]:
    started = time.perf_counter()
    conn = Edge()
    payload = json.dumps(body).encode() if body is not None else None
    headers = {
        "Host": HOST,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    if stream:
        first = None
        chunks: list[bytes] = []
        while True:
            line = resp.readline()
            if not line:
                break
            chunks.append(line)
            if first is None and line.startswith(b"data:") and b"[DONE]" not in line:
                first = time.perf_counter() - started
        conn.close()
        return resp.status, {"ttft": first, "bytes": sum(map(len, chunks))}, time.perf_counter() - started
    raw = resp.read()
    conn.close()
    elapsed = time.perf_counter() - started
    try:
        return resp.status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return resp.status, raw.decode("utf-8", "replace")[:400], elapsed


def _chat(text: str, max_tokens: int = 32, **extra) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
    }
    body.update(extra)
    return body


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def functional(key: str) -> dict:
    cases = {}
    status, body, elapsed = _request(key, "GET", "/v1/models")
    cases["models"] = {"status": status, "seconds": round(elapsed, 3), "ok": status == 200}

    status, body, elapsed = _request(key, "POST", "/v1/chat/completions", _chat("Reply with the word pong only.", 8))
    content = ""
    if isinstance(body, dict):
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    cases["short_chat"] = {"status": status, "seconds": round(elapsed, 3), "ok": status == 200 and bool(content), "preview": content[:80]}

    status, body, elapsed = _request(
        key, "POST", "/v1/chat/completions",
        {**_chat("Count to 3.", 16), "stream": True},
        stream=True,
    )
    ttft = body.get("ttft") if isinstance(body, dict) else None
    cases["stream"] = {"status": status, "seconds": round(elapsed, 3), "ttft": round(ttft, 3) if ttft else None, "ok": status == 200 and ttft is not None}

    status, body, elapsed = _request(
        key, "POST", "/v1/chat/completions",
        _chat("Call ping if you can.", 16, tools=[{
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
        }]),
    )
    cases["tools"] = {"status": status, "seconds": round(elapsed, 3), "ok": status == 200}

    status, body, elapsed = _request(
        key, "POST", "/v1/chat/completions",
        {
            "model": MODEL,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "One word: ok"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
            ]}],
        },
    )
    cases["vision"] = {"status": status, "seconds": round(elapsed, 3), "ok": status == 200}

    prefix = "You are a careful assistant. " * 80
    status, body, elapsed = _request(key, "POST", "/v1/chat/completions", _chat(prefix + "Say ready.", 8))
    cases["medium"] = {"status": status, "seconds": round(elapsed, 3), "ok": status == 200}
    status, body, elapsed = _request(key, "POST", "/v1/chat/completions", _chat(prefix + "Say ready.", 8))
    cases["prefix_reuse"] = {"status": status, "seconds": round(elapsed, 3), "ok": status == 200}
    return cases


def load(key: str, name: str, concurrency: int, n: int, prompt: str, max_tokens: int) -> dict:
    latencies: list[float] = []
    errors = 0

    def one(_: int) -> tuple[int, float]:
        status, _, elapsed = _request(key, "POST", "/v1/chat/completions", _chat(prompt, max_tokens))
        return status, elapsed

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, i) for i in range(n)]
        for fut in as_completed(futs):
            status, elapsed = fut.result()
            latencies.append(elapsed)
            if status != 200:
                errors += 1
    wall = time.perf_counter() - started
    return {
        "name": name,
        "concurrency": concurrency,
        "n": n,
        "errors": errors,
        "rps": round(n / wall, 3) if wall else None,
        "p50_s": round(_pct(latencies, 50) or 0, 3),
        "p95_s": round(_pct(latencies, 95) or 0, 3),
        "mean_s": round(statistics.mean(latencies), 3) if latencies else None,
        "ok": errors == 0,
    }


def metrics_snapshot() -> dict:
    import urllib.request
    try:
        raw = urllib.request.urlopen("http://127.0.0.1:8001/metrics", timeout=5).read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    wanted = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:num_preemptions_total",
    )
    out = {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        for name in wanted:
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[name] = line.rsplit(" ", 1)[-1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(prog="redesign.deploy.bench")
    parser.add_argument("--out", type=Path, default=Path("/scratch/deploy-state/bench/last.json"))
    args = parser.parse_args()
    key = _key()
    report = {
        "target": f"https://{HOST}/v1/",
        "model": MODEL,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics_before": metrics_snapshot(),
        "functional": functional(key),
        "load": [
            load(key, "short_x16", 16, 32, "Reply with one word: ok", 16),
            load(key, "medium_x8", 8, 16, ("Context. " * 200) + "Reply with one word: ok", 16),
        ],
    }
    report["metrics_after"] = metrics_snapshot()
    report["functional_pass"] = all(case.get("ok") for case in report["functional"].values())
    report["load_pass"] = all(case.get("ok") for case in report["load"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["functional_pass"] and report["load_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
