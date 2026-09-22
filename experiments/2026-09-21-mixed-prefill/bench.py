#!/usr/bin/env python3
"""Isolated mixed-context prefill/decode benchmark, based on the pre-four tree.

Fixed-output performance requests are deliberately distinct from correctness
checks. Synthetic cache salts isolate repetitions without flushing the cache.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "9f951e066358870466ce9875da63b174d5818b95"
IMAGE = "sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8"
VERSION = "mixed-prefill-v1"
WARM_INPUT = 49244
COLD_INPUT = 131164
WARM_OUTPUT = 256
COLD_OUTPUT = 32
CONCURRENCIES = (4, 8, 16)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered)-1)] - ordered[lower]) * (index - lower)


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode()
    values, buckets, cache = {}, {}, {}
    for line in text.splitlines():
        if not line.startswith("vllm:") or " " not in line:
            continue
        head, raw = line.rsplit(" ", 1)
        name = head.split("{", 1)[0]
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', head))
        values[name] = values.get(name, 0) + value
        if name.endswith("_bucket") and "le" in labels:
            group = buckets.setdefault(name[:-7], {})
            group[labels["le"]] = group.get(labels["le"], 0) + value
        if name == "vllm:cache_config_info":
            cache = labels
    return {"values": values, "buckets": buckets, "cache": cache}


def identity(base, variant, blocks):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if variant not in ("baseline", "longprefill3072", "aiter-gemm") and (
        git("rev-parse", "HEAD") != BASELINE
        or git("diff", BASELINE, "--", "redesign")
    ):
        raise RuntimeError("Serving source is not the requested pre-four baseline")
    runtime = json.loads(subprocess.check_output(["docker", "inspect", "k3"], text=True))[0]
    expected = yaml.safe_load(git("show", BASELINE + ":experiments/2026-09-20-amd-optimized/config-base.yaml"))
    if variant == "batch2048":
        expected["max-num-batched-tokens"] = 2048
        # Match the baseline's already allocated pool; this is a control, not
        # an extra memory-capacity treatment.
        expected["num-gpu-blocks-override"] = blocks
    elif variant == "longprefill3072":
        expected["long-prefill-token-threshold"] = 3072
    config_name = (
        "config-longprefill3072.yaml"
        if variant == "longprefill3072"
        else "config-base.yaml"
    )
    actual = yaml.safe_load(
        (Path("/scratch/deploy-state/amd-optimized") / config_name).read_text()
    )
    expected_image = (
        "cflowx/kimi-k3-aiter-gemm:5337bf2c"
        if variant == "aiter-gemm"
        else IMAGE
    )
    if actual != expected or expected_image not in runtime["Config"]["Image"]:
        raise RuntimeError("Unexpected engine image or numerical profile")
    cache = metrics(base)["cache"]
    wanted = {"block_size": "768", "mamba_cache_mode": "align", "prefix_match_unit": "None",
              "num_gpu_blocks": str(blocks), "enable_prefix_caching": "True",
              "num_gpu_blocks_override": str(blocks) if variant == "batch2048" else "None"}
    if any(cache.get(key) != value for key, value in wanted.items()):
        raise RuntimeError("Effective cache settings differ from the experiment")
    since = runtime["State"]["StartedAt"][:19].replace("T", " ") + " UTC"
    startup = subprocess.check_output(["journalctl", "-u", "k3.service", "--since", since,
                "--no-pager", "--grep=max_num_batched_tokens", "-o", "cat"], text=True)
    loaded_budgets = set(re.findall(r"max_num_batched_tokens(?:=|['\"]\s*:\s*)(\d+)", startup))
    if loaded_budgets != {str(expected["max-num-batched-tokens"])}:
        raise RuntimeError("Loaded scheduler token budget does not match the trial")
    env = dict(item.split("=", 1) for item in runtime["Config"]["Env"] if "=" in item)
    tuning = {key: value for key, value in env.items()
              if key.startswith(("VLLM_", "AITER_", "KIMI_K3_", "HIP_", "HSA_", "SAFETENSORS_"))}
    return {"baseline_commit": BASELINE, "branch": git("branch", "--show-current"),
            "container_id": runtime["Id"], "started": runtime["State"]["StartedAt"],
            "image": runtime["Config"]["Image"], "command": runtime["Config"]["Cmd"],
            "config": actual, "tuning_sha256": digest(tuning), "cache_config": cache}


def tokenize(base, payload):
    body = {"model": "FW-Kimi-K3", "return_token_strs": False, **payload}
    request = urllib.request.Request(base + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    if result.get("count") != len(result.get("tokens", [])):
        raise RuntimeError("Invalid tokenization response")
    return result["tokens"]


def fixture(base, name, length):
    topics = ("stream parsing", "scheduler fairness", "cache invalidation", "transaction isolation")
    topic = topics[sum(name.encode()) % len(topics)]
    def payload(count, padding=0):
        modules = ("scheduler", "parser", "cache", "router", "database", "worker", "client", "encoder")
        actions = ("validate", "dispatch", "serialize", "retry", "commit", "inspect", "measure")
        lines = [f"Record {i}: module={modules[i % len(modules)]}; action={actions[i % len(actions)]}; "
                 f"attempt={i % 5}; interval={i % 97}. Preserve ordering and check the result before releasing resources."
                 for i in range(count)]
        return {"messages": [
            {"role": "system", "content": f"Synthetic benchmark {VERSION}, fixture {name}. "
             "Use the reference material to write detailed engineering guidance."},
            {"role": "user", "content": "\n".join(lines) + " x" * padding
             + f"\nWrite a long, detailed technical handbook about {topic}, with examples and failure analysis. "
               "Continue elaborating each section rather than summarizing briefly."},
        ], "chat_template_kwargs": {"thinking": False}}
    count = max(1, length // 45)
    for _ in range(8):
        body = payload(count)
        tokens = tokenize(base, body)
        delta = length - len(tokens)
        if 0 <= delta < 100:
            break
        count = max(1, count + math.floor(delta / max(len(tokens) / count, 1)))
    if len(tokens) > length:
        raise RuntimeError("Could not construct bounded reference material")
    padding = length - len(tokens)
    for _ in range(4):
        body = payload(count, padding)
        tokens = tokenize(base, body)
        if len(tokens) == length:
            return {"name": name, "payload": body, "prompt_tokens": length,
                    "prompt_sha256": digest(tokens), "payload_sha256": digest(body)}
        padding += length - len(tokens)
        if padding < 0:
            break
    raise RuntimeError("Fixture does not have its exact requested token count")


def validate_terminal(usage, finishes, done, expected_prompt, expected_output):
    if not done or usage is None or finishes != ["length"]:
        raise RuntimeError("Incomplete fixed-work stream: require usage, length finish and DONE")
    if usage.get("prompt_tokens") != expected_prompt or usage.get("completion_tokens") != expected_output:
        raise RuntimeError("Fixed prompt/output workload differs from terminal usage")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if type(cached) is not int or not 0 <= cached <= expected_prompt:
        raise RuntimeError("Invalid cached-token usage")


def stream(base, spec, salt, output, label, on_first=None):
    body = {"model": "FW-Kimi-K3", "temperature": 0, "seed": 42,
            "max_tokens": output, "ignore_eos": True, "priority": 2,
            "stream": True, "stream_options": {"include_usage": True},
            "cache_salt": salt, **spec["payload"]}
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.monotonic()
    first = previous = None
    gaps, finishes, text = [], [], []
    usage, done = None, False
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                done = True
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError("Engine returned an SSE error")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    now = time.monotonic()
                    if first is None:
                        first = now
                        if on_first:
                            on_first()
                    if previous is not None:
                        gaps.append(now - previous)
                    previous = now
                    text.append(piece)
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
    end = time.monotonic()
    validate_terminal(usage, finishes, done, spec["prompt_tokens"], output)
    if first is None:
        raise RuntimeError("No visible output in completed benchmark response")
    return {"label": label, "fixture": spec["name"], "prompt_sha256": spec["prompt_sha256"],
            "payload_sha256": spec["payload_sha256"], "usage": usage, "finishes": finishes,
            "seconds": end - start, "ttft": first - start,
            "after_first_tps": output / (end-first) if end > first else None,
            "visible_gap_p50": percentile(gaps, .5), "visible_gap_p95": percentile(gaps, .95),
            "visible_gap_max": max(gaps) if gaps else None,
            "visible_gaps_over_250ms": sum(gap > .25 for gap in gaps), "visible_gaps": len(gaps),
            "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest()}


def idle(snapshot):
    return all(snapshot["values"].get(name) == 0 for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"))


def accounting(before, after, rows):
    names = ("vllm:generation_tokens_total", "vllm:request_success_total", "vllm:num_preemptions_total")
    delta = {name: after["values"].get(name, 0) - before["values"].get(name, 0) for name in names}
    expected = sum(row["usage"]["completion_tokens"] for row in rows)
    if (not idle(before) or not idle(after) or delta[names[0]] != expected or delta[names[1]] != len(rows)):
        raise RuntimeError(f"Non-isolated phase: expected {expected} output tokens/{len(rows)} completions, got {delta}")
    if delta[names[2]] != 0:
        raise RuntimeError("Benchmark exceeded its intended no-preemption working set")
    return delta


def histogram_quantile(buckets, q):
    pairs = sorted((float(bound), count) for bound, count in buckets.items())
    total = pairs[-1][1] if pairs else 0
    if total <= 0:
        return None
    target, lower, last_count = total * q, 0.0, 0.0
    for upper, count in pairs:
        if count >= target:
            if not math.isfinite(upper):
                return lower
            return lower + (upper-lower) * (target-last_count) / (count-last_count) if count > last_count else upper
        lower, last_count = upper, count
    return None


def phase(base, warm, cold, namespace, concurrency, kind, round_id, output=WARM_OUTPUT):
    before = metrics(base)
    if not idle(before):
        raise RuntimeError("Engine has other work before benchmark phase")
    start = time.monotonic()
    events = [threading.Event() for _ in range(concurrency)]
    rows, injection = [], None
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency + len(cold)) as pool:
        futures = [pool.submit(stream, base, warm[i], f"{namespace}-warm-{i}", output,
                               f"warm-{i}", events[i].set) for i in range(concurrency)]
        if cold:
            deadline = start + 60
            while not all(event.is_set() for event in events):
                for future, event in zip(futures, events):
                    if future.done() and not event.is_set():
                        future.result()
                        raise RuntimeError("Warm stream ended without a first-token event")
                if time.monotonic() > deadline:
                    raise RuntimeError("Warm streams did not start before injection deadline")
                time.sleep(.01)
            if any(future.done() for future in futures):
                raise RuntimeError("Warm stream finished before the cold burst")
            injection = time.monotonic() - start
            futures.extend(pool.submit(stream, base, spec, f"{namespace}-{kind}-{round_id}-cold-{i}", COLD_OUTPUT,
                                       f"cold-{i}") for i, spec in enumerate(cold))
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    elapsed = time.monotonic() - start
    time.sleep(.2)
    after = metrics(base)
    delta = accounting(before, after, rows)
    warm_rows = [row for row in rows if row["label"].startswith("warm-")]
    cold_rows = [row for row in rows if row["label"].startswith("cold-")]
    for row in warm_rows:
        if row["usage"]["prompt_tokens_details"]["cached_tokens"] != WARM_INPUT // 768 * 768:
            raise RuntimeError("Intended warm prefix was not available")
    if any(row["usage"]["prompt_tokens_details"]["cached_tokens"] != 0 for row in cold_rows):
        raise RuntimeError("Intended cold burst reused a prefix")
    hist_name = "vllm:inter_token_latency_seconds"
    hist = {bound: count - before["buckets"].get(hist_name, {}).get(bound, 0)
            for bound, count in after["buckets"].get(hist_name, {}).items()}
    summary = {"warm_ttft_p50": percentile([row["ttft"] for row in warm_rows], .5),
               "warm_e2e_p50": percentile([row["seconds"] for row in warm_rows], .5),
               "warm_e2e_p95": percentile([row["seconds"] for row in warm_rows], .95),
               "warm_after_first_tps_p50": percentile([row["after_first_tps"] for row in warm_rows], .5),
               "warm_stream_p95_gap_median": percentile([row["visible_gap_p95"] for row in warm_rows], .5),
               "warm_stream_max_gap": max(row["visible_gap_max"] for row in warm_rows),
               "cold_ttft_p50": percentile([row["ttft"] for row in cold_rows], .5),
               "cold_ttft_max": max((row["ttft"] for row in cold_rows), default=None),
               "aggregate_output_tps": sum(row["usage"]["completion_tokens"] for row in rows) / elapsed,
               "engine_itl_p50_estimate": histogram_quantile(hist, .5),
               "engine_itl_p95_estimate": histogram_quantile(hist, .95)}
    return {"kind": kind, "concurrency": concurrency, "round": round_id, "wall_seconds": elapsed,
            "cold_injection_seconds": injection, "summary": summary, "accounting": delta,
            "requests": sorted(rows, key=lambda row: row["label"])}


def quality(base, salt):
    path = ROOT / "experiments/2026-09-20-amd-optimized/qualify.py"
    spec = importlib.util.spec_from_file_location("baseline_qualify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = {"model": "FW-Kimi-K3", "temperature": 0, "seed": 42,
              "chat_template_kwargs": {"thinking": False}, "cache_salt": salt}
    before = metrics(base)
    if not idle(before):
        raise RuntimeError("Engine has other work before correctness checks")
    rows = module.smoke(base, common)
    material = [f"Record {i}: this is unrelated synthetic reference material." for i in range(11200)]
    material.insert(len(material)//2, "The unique verification marker is QUARTZ-6139.")
    body = {**common, "max_tokens": 64, "messages": [{"role": "user", "content": "\n".join(material)
            + "\nReturn only the unique verification marker."}]}
    for index in range(2):
        row, text, _ = module.submit(base, body)
        rows.append({**row, "name": f"133k-retrieval-{index}",
                     "passed": text.strip().strip("` .\n") == "QUARTZ-6139" and row["finishes"] == ["stop"]})
    time.sleep(.2)
    accounting(before, metrics(base), rows)
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Correctness smoke failed")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--variant",
        choices=("baseline", "batch2048", "longprefill3072", "aiter-gemm"),
        required=True,
    )
    parser.add_argument("--cache-blocks", type=int, default=2177)
    parser.add_argument("--rounds", type=int, default=2, choices=(1, 2))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve prior evidence; choose a new output filename")
    report = {"started_utc": utc(), "variant": args.variant, "workload_version": VERSION,
              "rounds": args.rounds, "identity": identity(args.base, args.variant, args.cache_blocks),
              "quality": [], "phases": [], "fixtures": [], "passed": False}
    namespace = uuid.uuid4().hex
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["quality"] = quality(args.base, namespace + "-quality")
        save()
        print(json.dumps({"quality_passed": len(report["quality"])}), flush=True)
        warm = [fixture(args.base, f"warm-{i:02}", WARM_INPUT) for i in range(max(CONCURRENCIES))]
        cold = [fixture(args.base, f"cold-{i}", COLD_INPUT) for i in range(2)]
        report["fixtures"] = [{key: value for key, value in spec.items() if key != "payload"} for spec in warm+cold]
        save()
        for concurrency in CONCURRENCIES:
            case_namespace = f"{namespace}-c{concurrency}"
            before = metrics(args.base)
            if not idle(before):
                raise RuntimeError("Engine has other work before cache priming")
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                seeded = list(pool.map(lambda i: stream(args.base, warm[i], f"{case_namespace}-warm-{i}", 1,
                                                       f"seed-{i}"), range(concurrency)))
            time.sleep(.2)
            accounting(before, metrics(args.base), seeded)
            # Warm the concurrent decode shape before recording performance.
            warmup = phase(args.base, warm, [], case_namespace, concurrency, "warmup", 0, output=64)
            report.setdefault("warmups", []).append(warmup)
            save()
            for kind in ("warm", "mixed"):
                for round_id in range(args.rounds):
                    result = phase(args.base, warm, cold if kind == "mixed" else [], case_namespace,
                                   concurrency, kind, round_id)
                    report["phases"].append(result)
                    save()
                    print(json.dumps({"concurrency": concurrency, "kind": kind, "round": round_id,
                                      **result["summary"]}), flush=True)
        if identity(args.base, args.variant, args.cache_blocks) != report["identity"]:
            raise RuntimeError("Runtime changed during the experiment")
        report["passed"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["ended_utc"] = utc()
        save()


if __name__ == "__main__":
    main()
