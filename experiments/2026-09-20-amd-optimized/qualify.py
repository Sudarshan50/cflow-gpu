#!/usr/bin/env python3
"""Synthetic correctness and bounded sustained decode comparison; no customer data.

Call the engine directly, never the response cache. Benchmark requests deliberately
run to a fixed output budget; their length finishes are expected, unlike smoke
checks. Throughput comparisons require matching image/config/workload and no
unmeasured customer traffic on the engine.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import statistics
import struct
import subprocess
import threading
import time
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

import yaml


def utc():
    return datetime.now(timezone.utc).isoformat()


def provenance(label):
    inspected = json.loads(subprocess.check_output(["docker", "inspect", "k3"], text=True))[0]
    cmd = inspected["Config"]["Cmd"]
    config_name = Path(cmd[cmd.index("--config") + 1]).name
    expected = "config-dspark.yaml" if label == "optimized-dspark" else "config-base.yaml"
    if label.startswith("optimized-") and config_name != expected:
        raise RuntimeError("Running profile does not match qualification label")
    path = Path("/scratch/deploy-state/amd-optimized") / config_name
    if not label.startswith("optimized-"):
        path = Path("/scratch/hf/config.yaml")
    config = yaml.safe_load(path.read_text())
    env = dict(v.split("=", 1) for v in inspected["Config"]["Env"] if "=" in v)
    tuning = {k: v for k,v in env.items() if k.startswith(("VLLM_", "AITER_", "KIMI_K3_", "HIP_", "HSA_", "SAFETENSORS_")) and k != "AITER_JIT_DIR"}
    return {"container_id": inspected["Id"], "image": inspected["Config"]["Image"],
            "target": cmd[cmd.index("serve")+1], "config": config,
            "tuning_env_sha256": hashlib.sha256(json.dumps(tuning, sort_keys=True).encode()).hexdigest(),
            "workload_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workload_version": "balanced-cycles-v2"}


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode()
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        if not name.startswith("vllm:") or name.endswith("_bucket"):
            continue
        try:
            values[name] = values.get(name, 0) + float(parts[1])
        except ValueError:
            pass
    return values


def submit(base, body, timeout=180):
    started = time.monotonic()
    request = urllib.request.Request(base + "/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage, finishes, text, first, tool_calls = None, [], "", None, []
    done = not body.get("stream")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if body.get("stream"):
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("Engine emitted an SSE error")
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if piece and first is None:
                        first = time.monotonic() - started
                    text += delta.get("content") or ""
                    if choice.get("finish_reason"):
                        finishes.append(choice["finish_reason"])
        else:
            event = json.load(response)
            if event.get("error"):
                raise RuntimeError("Engine returned a JSON error")
            usage = event.get("usage")
            for choice in event.get("choices", []):
                message = choice.get("message") or {}
                text += message.get("content") or ""
                tool_calls += message.get("tool_calls") or []
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
    if not done or not finishes or not usage:
        raise RuntimeError("Incomplete response: missing terminal finish/DONE/usage")
    return {"seconds": time.monotonic() - started, "ttft": first,
            "usage": usage, "finishes": finishes,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest()}, text, tool_calls


def png():
    def chunk(name, data):
        return struct.pack("!I", len(data)) + name + data + struct.pack("!I", zlib.crc32(name + data))
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 32, 32, 8, 2, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 32) * 32)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def smoke(base, common):
    rows = []
    cases = [("multiply", "Calculate 17 * 23. Return only the integer.", "391"),
             ("add", "Calculate 149 + 276. Return only the integer.", "425"),
             ("negative", "Calculate 12 - 19. Return only the integer.", "-7"),
             ("prime", "What is the smallest prime larger than 19? Return only the integer.", "23")]
    filler = "\n".join(f"Record {i}: this is unrelated synthetic reference material." for i in range(1800))
    cases.append(("long-retrieval", filler[:len(filler)//2] + "\nThe secret marker is AQUAMARINE-7293.\n" + filler[len(filler)//2:] + "\nReturn only the secret marker.", "AQUAMARINE-7293"))
    for name, prompt, expected in cases:
        row, text, _ = submit(base, {**common, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128})
        row.update(name=name, passed=text.strip().strip("`\n .") == expected and row["finishes"] == ["stop"])
        rows.append(row)
    body = {**common, "messages": [{"role": "user", "content": "Return a JSON object with value equal to 7 plus 8."}],
            "max_tokens": 128, "response_format": {"type": "json_schema", "json_schema": {"name": "sum", "strict": True,
            "schema": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}}}}
    row, text, _ = submit(base, body)
    try:
        valid = json.loads(text) == {"value": 15}
    except ValueError:
        valid = False
    rows.append({**row, "name": "structured-json", "passed": valid and row["finishes"] == ["stop"]})
    body = {**common, "messages": [{"role": "user", "content": "Call record_sum with value equal to 7 plus 8."}],
            "max_tokens": 128, "tools": [{"type": "function", "function": {"name": "record_sum", "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}],
            "tool_choice": {"type": "function", "function": {"name": "record_sum"}}}
    row, _, calls = submit(base, body)
    try:
        valid = len(calls) == 1 and calls[0]["function"]["name"] == "record_sum" and json.loads(calls[0]["function"]["arguments"]) == {"value": 15}
    except (ValueError, KeyError, TypeError):
        valid = False
    rows.append({**row, "name": "tool-call", "passed": valid and "length" not in row["finishes"]})
    row, text, _ = submit(base, {**common, "max_tokens": 128, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is the main colour of this image? Reply with one colour word."},
        {"type": "image_url", "image_url": {"url": png()}}]}]})
    rows.append({**row, "name": "vision", "passed": bool(re.search(r"\bred\b", text.lower())) and row["finishes"] == ["stop"]})
    return rows


def benchmark(base, common, concurrency, seconds):
    # Stable four-task workload shared across arms. Approximate prompt size is
    # reported from usage, never asserted from character count.
    prefix = "\n".join(f"Synthetic source record {i}: buffer, token, stream, cache, schedule." for i in range(96))
    tasks = ["Write a detailed numbered explanation of prefix caching.",
             "Write Python functions to parse a stream of JSON records, with comments.",
             "Explain matrix multiplication and give a worked example.",
             "Write a detailed numbered checklist for testing an HTTP streaming client."]
    def body(i):
        return {**common, "messages": [{"role": "user", "content": prefix + "\n" + tasks[i % len(tasks)]}],
                "max_tokens": 256, "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True}}
    # Warm the actual measured batch shape AND output budget, outside timing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda i: submit(base, body(i)), range(max(concurrency, 4))))
    time.sleep(.25)
    before = metrics(base)
    begin = time.monotonic()
    deadline = begin + seconds
    stop = threading.Event()
    def worker(i):
        rows = []
        while time.monotonic() < deadline and not stop.is_set():
            # Finish balanced four-task cycles, even across the deadline. Each
            # successful arm therefore has equal samples of every task.
            for offset in range(len(tasks)):
                try:
                    task_id = (i + offset) % len(tasks)
                    row, _, _ = submit(base, body(task_id))
                    if row["usage"].get("completion_tokens") != 256:
                        raise RuntimeError("Benchmark did not produce the fixed 256-token budget")
                    rows.append({**row, "task_id": task_id})
                except Exception as exc:
                    stop.set()
                    rows.append({"error": type(exc).__name__ + ": " + str(exc)})
                    return rows
        return rows
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = [row for batch in pool.map(worker, range(concurrency)) for row in batch]
    elapsed = time.monotonic() - begin
    time.sleep(.25)
    after = metrics(base)
    good = [r for r in rows if "error" not in r]
    accounted_output = sum(r["usage"]["completion_tokens"] for r in good)
    engine_output = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)
    engine_finishes = after.get("vllm:request_success_total", 0) - before.get("vllm:request_success_total", 0)
    quiet = all(sample.get(name) == 0 for sample in (before, after) for name in (
        "vllm:num_requests_running", "vllm:num_requests_waiting"))
    def percentile(values, q):
        values = sorted(values)
        return values[min(len(values)-1, int(q*len(values)))] if values else None
    return {"concurrency": concurrency, "offered_seconds": seconds, "wall_seconds": elapsed,
            "completed": len(good), "errors": [r for r in rows if "error" in r],
            "output_tokens_per_second": accounted_output / elapsed,
            "request_output_tokens": accounted_output,
            "engine_output_tokens": engine_output,
            "unaccounted_engine_output_tokens": engine_output - accounted_output,
            "isolated_output_accounting": engine_output == accounted_output,
            "isolated": engine_output == accounted_output and engine_finishes == len(good) and quiet,
            "engine_finishes": engine_finishes, "quiet_boundaries": quiet,
            "task_counts": {str(i): sum(row["task_id"] == i for row in good) for i in range(len(tasks))},
            "prompt_tokens_mean": statistics.mean(r["usage"]["prompt_tokens"] for r in good) if good else None,
            "e2e_p50": percentile([r["seconds"] for r in good], .5),
            "e2e_p95": percentile([r["seconds"] for r in good], .95),
            "ttft_p50": percentile([r["ttft"] for r in good if r["ttft"] is not None], .5),
            "ttft_p95": percentile([r["ttft"] for r in good if r["ttft"] is not None], .95),
            "metric_deltas": {k: v - before.get(k, 0) for k,v in after.items() if "spec_decode" in k or k in (
                "vllm:generation_tokens_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", "vllm:num_preemptions_total")}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8001")
    p.add_argument("--label", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    if not 10 <= args.seconds <= 180:
        p.error("seconds must be in [10,180]")
    common = {"model": "FW-Kimi-K3", "temperature": 0, "seed": 42, "chat_template_kwargs": {"thinking": False}}
    result = {"label": args.label, "started_utc": utc(), "smoke": [], "benchmarks": [],
              "provenance": provenance(args.label)}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    try:
        result["smoke"] = smoke(args.base, common)
        save()
        if not all(row["passed"] for row in result["smoke"]):
            raise RuntimeError("Synthetic correctness check failed")
        if not args.smoke_only:
            for concurrency in (1, 16):
                result["benchmarks"].append(benchmark(args.base, common, concurrency, args.seconds))
                save()
                if result["benchmarks"][-1]["errors"]:
                    raise RuntimeError("Benchmark request failed")
                if not result["benchmarks"][-1]["isolated"]:
                    raise RuntimeError("Benchmark not isolated: engine output/finishes or boundary occupancy differs")
        if provenance(args.label) != result["provenance"]:
            raise RuntimeError("Runtime/configuration changed during qualification")
    except Exception as exc:
        result["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        result["ended_utc"] = utc()
        save()
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
