#!/usr/bin/env python3
"""Text-only, sequential KDA prefix-cache experiments against the baseline engine.

The changing-date study uses real Chat rendering and checks the requested date.
Checkpoint probes use valid vocabulary token IDs to control exact boundaries;
their one-token outputs are not model-quality benchmarks. No cache flush, salts
from customers, prompts from traffic, gateway policies, or response replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

BASELINE = "9f951e066358870466ce9875da63b174d5818b95"
IMAGE = "sha256:5f3007aff1bc231eceb9f024e56ee80e44f9ca101a521aa50fe6bfa6c979d6b8"
ROOT = Path(__file__).resolve().parents[2]


def utc():
    return datetime.now(timezone.utc).isoformat()


def lcp(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def post(base, path, body):
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def tokenize(base, model, payload):
    response = post(base, "/tokenize", {"model": model, "return_token_strs": False, **payload})
    if response.get("count") != len(response.get("tokens", [])):
        raise RuntimeError("Invalid tokenization response")
    return response["tokens"]


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
        text = response.read().decode()
    values, cache = {}, {}
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
        values[name] = values.get(name, 0) + value
        if name == "vllm:cache_config_info":
            cache = dict(re.findall(r'(\w+)="([^"\\]*)"', head))
    return values, cache


def checked_cache_config(base, variant, *, cache_blocks=None):
    _, cache = metrics(base)
    expected = {"block_size": "768", "mamba_cache_mode": "align",
                "enable_prefix_caching": "True",
                "prefix_match_unit": "128" if variant == "tail128" else "None",
                "num_gpu_blocks_override": str(cache_blocks) if cache_blocks is not None else "None"}
    if cache_blocks is not None:
        expected["num_gpu_blocks"] = str(cache_blocks)
    if any(cache.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Effective engine cache configuration does not match the study variant")
    return cache


def identity(variant, *, cache_blocks=None):
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("branch", "--show-current") != "redesign/capacity-first":
        raise RuntimeError("Study must run on the restored redesign branch")
    if git("rev-parse", "HEAD") != BASELINE:
        raise RuntimeError("Study checkout differs from the requested checkpoint")
    if git("diff", BASELINE, "--", "redesign"):
        raise RuntimeError("Runtime source differs from the pre-four-optimization checkpoint")
    runtime = json.loads(subprocess.check_output(["docker", "inspect", "k3"], text=True))[0]
    expected = yaml.safe_load(git("show", BASELINE + ":experiments/2026-09-20-amd-optimized/config-base.yaml"))
    actual = yaml.safe_load(Path("/scratch/deploy-state/amd-optimized/config-base.yaml").read_text())
    if variant == "tail128":
        expected["prefix-match-unit"] = 128
    if cache_blocks is not None:
        if cache_blocks <= 0:
            raise ValueError("Cache block control must be positive")
        expected["num-gpu-blocks-override"] = cache_blocks
    if actual != expected or IMAGE not in runtime["Config"]["Image"]:
        raise RuntimeError("Unexpected model image or runtime profile")
    return {"baseline_commit": BASELINE, "branch": git("branch", "--show-current"),
            "container_id": runtime["Id"], "started": runtime["State"]["StartedAt"],
            "image": runtime["Config"]["Image"], "config": actual}


def generate(base, model, payload, token_ids, label, block, *, completion=False, expected_text=None):
    before, _ = metrics(base)
    if before.get("vllm:num_requests_running") != 0 or before.get("vllm:num_requests_waiting") != 0:
        raise RuntimeError("Engine has other work; refusing a contaminated sequential probe")
    body = {"model": model, "temperature": 0, "seed": 42, "max_tokens": 1 if completion else 32,
            "stream": True, "stream_options": {"include_usage": True}, **payload}
    if not completion:
        body["chat_template_kwargs"] = {"thinking": False}
    path = "/v1/completions" if completion else "/v1/chat/completions"
    request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.monotonic()
    ttft = None
    text = ""
    usage = None
    finishes = []
    done = False
    with urllib.request.urlopen(request, timeout=90) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                done = True
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError("Engine emitted an error event")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                piece = choice.get("text") if completion else delta.get("content")
                if piece:
                    ttft = ttft if ttft is not None else time.monotonic() - start
                    text += piece
                if choice.get("finish_reason"):
                    finishes.append(choice["finish_reason"])
    elapsed = time.monotonic() - start
    if not done or not usage or not finishes:
        raise RuntimeError("Incomplete response: require finish reason, usage and DONE")
    if usage["prompt_tokens"] != len(token_ids):
        raise RuntimeError("Rendered and inference prompt counts differ")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    if type(cached) is not int or not 0 <= cached <= len(token_ids):
        raise RuntimeError("Missing/invalid per-request cached-token usage")
    time.sleep(.1)
    after, _ = metrics(base)
    output_delta = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)
    finished_delta = after.get("vllm:request_success_total", 0) - before.get("vllm:request_success_total", 0)
    isolated = output_delta == usage["completion_tokens"] and finished_delta == 1
    row = {"label": label, "utc": utc(), "prompt_tokens": len(token_ids), "cached_tokens": cached,
           "uncached_tokens": len(token_ids)-cached, "cache_fraction": cached / len(token_ids),
           "output_tokens": usage["completion_tokens"], "ttft_seconds": ttft, "elapsed_seconds": elapsed,
           "finish_reasons": finishes, "isolated": isolated,
           "generation_counter_delta": output_delta, "finished_counter_delta": finished_delta,
           "full_block_replay_upper_bound": (len(token_ids)-1)//block*block,
           "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
    if expected_text is not None:
        row["answer_correct"] = text.strip().strip("` .\n") == expected_text
        row["observed_date_answer"] = text[:64]
    if not isolated:
        raise RuntimeError(
            f"Unexpected engine work during isolated probe {label}: "
            f"generated={output_delta}, expected={usage['completion_tokens']}, "
            f"finished={finished_delta}, running={after.get('vllm:num_requests_running')}, "
            f"waiting={after.get('vllm:num_requests_waiting')}"
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="FW-Kimi-K3")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "tail128"), default="baseline")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; use a new filename to preserve earlier evidence")
    report = {"started_utc": utc(), "identity": identity(args.variant), "variant": args.variant, "requests": [], "tokenization_only": {}}
    cache = checked_cache_config(args.base, args.variant)
    block = int(cache["block_size"])
    report["cache_config"] = cache
    run = uuid.uuid4().hex
    report["synthetic_run_id"] = run

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    def record(row):
        report["requests"].append(row)
        save()
        print(json.dumps(row), flush=True)
        if row.get("answer_correct") is False:
            raise RuntimeError("Date-aware answer changed or was truncated")

    try:
        dates = ("2031-04-11", "2031-04-12", "2031-04-13")
        rules = "Study " + run + ". Use only the date in RUNTIME_CONTEXT when asked for today's date.\n"
        rules += "\n".join(f"Stable rule {i}: preserve the given facts and return the requested date without commentary." for i in range(512))
        for layout in ("date_first", "stable_first"):
            seed = previous = None
            for index, date in enumerate(dates):
                dynamic = "RUNTIME_CONTEXT current_date=" + date
                system = dynamic + "\n" + rules if layout == "date_first" else rules + "\n" + dynamic
                payload = {"messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": "What is today's date? Return YYYY-MM-DD only."}],
                           "chat_template_kwargs": {"thinking": False}}
                tokens = tokenize(args.base, args.model, payload)
                row = generate(args.base, args.model, payload, tokens, f"{layout}-{index+1}", block, expected_text=date)
                row["lcp_previous"] = lcp(previous, tokens) if previous is not None else None
                row["lcp_seed"] = lcp(seed, tokens) if seed is not None else None
                record(row)
                seed = tokens if seed is None else seed
                previous = tokens

        pattern = tokenize(args.base, args.model, {"prompt": " alpha beta gamma delta epsilon zeta eta theta", "add_special_tokens": False})
        variants = list(dict.fromkeys(pattern))[:3]
        if len(variants) != 3:
            raise RuntimeError("Need three distinct ordinary vocabulary tokens")

        def token_prompt(tag, length):
            header = tokenize(args.base, args.model, {"prompt": "Synthetic checkpoint " + run + " " + tag + "\n", "add_special_tokens": False})
            return (header + (pattern * (length//len(pattern)+1)))[:length]

        for boundary in (2304, 3840, 4608):
            seed = token_prompt(f"boundary-{boundary}", 7675)
            previous = None
            for index in range(3):
                tokens = seed[:boundary] + [variants[index]] + seed[boundary+1:]
                row = generate(args.base, args.model, {"prompt": tokens}, tokens,
                               f"boundary-{boundary}-{index+1}", block, completion=True)
                row["lcp_previous"] = lcp(previous, tokens) if previous is not None else None
                record(row)
                previous = tokens

        for length in (988, 1024, 7680):
            tokens = token_prompt(f"exact-{length}", length)
            for index in range(3):
                record(generate(args.base, args.model, {"prompt": tokens}, tokens,
                                f"exact-{length}-{index+1}", block, completion=True))

        boundary = 2304
        tokens = token_prompt("primed-2304", 7675)
        record(generate(args.base, args.model, {"prompt": tokens[:boundary+1]}, tokens[:boundary+1],
                        "prime-2304-seed", block, completion=True))
        for index in (1, 2):
            sibling = tokens[:boundary] + [variants[index]] + tokens[boundary+1:]
            row = generate(args.base, args.model, {"prompt": sibling}, sibling,
                           f"prime-2304-sibling-{index}", block, completion=True)
            row["lcp_prime"] = lcp(tokens[:boundary+1], sibling)
            record(row)

        tokens = token_prompt("salt-control", 1587)
        for index, salt in enumerate((run + "-pool-a", run + "-pool-a", run + "-pool-b")):
            record(generate(args.base, args.model, {"prompt": tokens, "cache_salt": salt}, tokens,
                            f"salt-control-{index+1}", block, completion=True))

        sys.path.insert(0, str(ROOT))
        from redesign.gateway.media import normalize_payload
        tools = [{"type": "function", "function": {
            "name": name, "description": ("Stable synthetic tool documentation. " * 60),
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "count": {"type": "integer"}}},
        }} for name in ("alpha", "beta", "gamma")]
        tool_base = {"messages": [{"role": "user", "content": "Tool layout study " + run}],
                     "tools": tools, "chat_template_kwargs": {"thinking": False}}
        def reverse_keys(value):
            if isinstance(value, dict):
                return {key: reverse_keys(value[key]) for key in reversed(value)}
            if isinstance(value, list):
                return [reverse_keys(item) for item in value]
            return value
        reference = tokenize(args.base, args.model, tool_base)
        for name, payload in (
            ("dictionary_order", reverse_keys(tool_base)),
            ("tool_array_order", {**tool_base, "tools": list(reversed(tools))}),
            ("canonical_tool_order", normalize_payload(copy.deepcopy({**tool_base, "tools": list(reversed(tools))}))),
        ):
            candidate = tokenize(args.base, args.model, payload)
            report["tokenization_only"][name] = {
                "reference_tokens": len(reference), "tokens": len(candidate),
                "identical_token_ids": reference == candidate, "lcp_tokens": lcp(reference, candidate),
                "note": "Token-prefix identity, not a GPU cache-hit measurement",
            }
        save()

        if identity(args.variant) != report["identity"]:
            raise RuntimeError("Runtime changed during study")
        if checked_cache_config(args.base, args.variant) != report["cache_config"]:
            raise RuntimeError("Effective cache configuration changed during study")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["ended_utc"] = utc()
        save()


if __name__ == "__main__":
    main()
