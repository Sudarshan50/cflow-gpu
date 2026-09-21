#!/usr/bin/env python3
"""Controlled replay of the observed agentic shape, measured client-side.

The live portal wave is a shared system+tools prefix, a large unique body, and
a very small streamed completion (prompt p50 ~21.7k, completion p50 ~16-86).
Engine histograms give aggregate inter-token latency but cannot hold offered
load constant across a restart, so an A/B needs a generator that does.

Streaming is mandatory here: the complaint this exists to measure is per-stream
token delivery, which a non-streaming request cannot observe.

Talks to the engine directly. The gateway clamps agentic output and sheds under
distress by design, both of which would be measured as engine behaviour.

Compare cold runs only. On this hybrid KDA+MLA model the prefix cache decides
how many sequences fit: measured 2026-09-20 at 96 offered, a 14.5% hit rate
seated 42 and a 34% hit rate seated 66, which is 13,976 vs 22,393 prompt
tokens/s on identical work. Bodies are therefore salted per run by default.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Stable across every request so the prefix cache has something to hit, which
# is what the live traffic does with its system block and tool schema.
SHARED_PREFIX = (
    "You are a coding agent operating in a large repository. Follow the tool "
    "protocol exactly. Never invent file paths. Prefer minimal diffs. "
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository path"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        },
    },
]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))], 4)


def count_tokens(url: str, text: str, model: str) -> int | None:
    """Ask the engine's own tokenizer rather than guessing chars-per-token.

    A guess is how the first run of this harness asked for 26k and sent 59k.
    """
    request = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps({"model": model, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())["count"]
    except Exception:  # noqa: BLE001 -- caller falls back to an estimate
        return None


def _repeat_to_tokens(unit: str, target: int, tokens_per_unit: float) -> str:
    return unit * max(1, int(target / max(tokens_per_unit, 0.001)))


def build_payload(index: int, prefix: str, body_units: int,
                  max_tokens: int, model: str, salt: str) -> dict:
    body = " ".join(f"sym{salt}x{index}_{i}" for i in range(body_units))
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": f"{body}\nName one symbol above. One word."},
        ],
        "tools": TOOLS,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def one_stream(url: str, payload: dict, timeout: float) -> dict:
    started = time.perf_counter()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    ttft = None
    stamps: list[float] = []
    usage = None
    text_chars = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                chunk = raw[5:].strip()
                if chunk == b"[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # An agentic turn streams tool_calls, and K3 streams reasoning
                # before either. Counting only `content` measures nothing.
                piece = (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("reasoning_content")
                    or (json.dumps(delta["tool_calls"]) if delta.get("tool_calls") else None)
                )
                if not piece:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - started
                else:
                    stamps.append(now)
                text_chars += len(piece)
    except Exception as exc:  # noqa: BLE001 -- an error is a data point
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "elapsed": time.perf_counter() - started}

    elapsed = time.perf_counter() - started
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "ok": ttft is not None,
        "ttft": ttft,
        "elapsed": elapsed,
        "chunk_gaps": gaps,
        "chars": text_chars,
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="agentic_load")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="FW-Kimi-K3")
    parser.add_argument("--label", default="agentic")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=26_000)
    parser.add_argument("--prefix-tokens", type=int, default=4_608)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--salt", default=None,
                        help="body salt; defaults to a fresh one per run. "
                             "Reusing a salt replays cached bodies, which is "
                             "how run 2 of an A/B inherits run 1's prefix "
                             "cache and reads 60%% faster on identical work.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.salt is None:
        args.salt = str(int(time.time()))

    # Calibrate both halves of the prompt against the engine tokenizer so the
    # reported shape is the shape that was actually offered.
    prefix_unit_tokens = count_tokens(args.url, SHARED_PREFIX, args.model) or 24
    prefix = _repeat_to_tokens(SHARED_PREFIX, args.prefix_tokens, prefix_unit_tokens)

    probe_units = 500
    probe = " ".join(f"sym0_{i}" for i in range(probe_units))
    probe_tokens = count_tokens(args.url, probe, args.model) or probe_units * 4
    tokens_per_unit = probe_tokens / probe_units
    body_units = max(1, int((args.prompt_tokens - args.prefix_tokens) / tokens_per_unit))

    print(json.dumps({
        "calibration": {
            "prefix_unit_tokens": prefix_unit_tokens,
            "tokens_per_body_unit": round(tokens_per_unit, 3),
            "body_units": body_units,
        }
    }))

    payloads = [
        build_payload(i, prefix, body_units, args.max_tokens, args.model, args.salt)
        for i in range(args.requests)
    ]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda p: one_stream(args.url, p, args.timeout), payloads
        ))
    wall = time.perf_counter() - started

    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    all_gaps = [g for r in ok for g in r.get("chunk_gaps", [])]
    completions = [r["completion_tokens"] for r in ok if r.get("completion_tokens")]
    prompts = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]

    # stream_interval batches several tokens per SSE chunk, so a chunk gap is
    # not an inter-token latency. Derive per-token from the usage count.
    per_token = [
        (r["elapsed"] - r["ttft"]) / (r["completion_tokens"] - 1)
        for r in ok
        if r.get("completion_tokens") and r["completion_tokens"] > 1 and r.get("ttft")
    ]

    report = {
        "label": args.label,
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offered": {
            "salt": args.salt,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "prompt_tokens_target": args.prompt_tokens,
            "shared_prefix_tokens": args.prefix_tokens,
            "max_tokens": args.max_tokens,
        },
        "wall_seconds": round(wall, 2),
        "succeeded": len(ok),
        "failed": len(failed),
        "errors": sorted({r.get("error", "") for r in failed})[:5],
        "ttft_seconds": {
            "p50": _pct([r["ttft"] for r in ok], 0.5),
            "p95": _pct([r["ttft"] for r in ok], 0.95),
        },
        "stream_chunk_gap_seconds": {
            "n": len(all_gaps),
            "p50": _pct(all_gaps, 0.5),
            "p95": _pct(all_gaps, 0.95),
            "mean": round(statistics.mean(all_gaps), 4) if all_gaps else None,
        },
        "per_token_seconds": {
            "p50": _pct(per_token, 0.5),
            "p95": _pct(per_token, 0.95),
        },
        "e2e_seconds": {
            "p50": _pct([r["elapsed"] for r in ok], 0.5),
            "p95": _pct([r["elapsed"] for r in ok], 0.95),
        },
        "throughput": {
            "requests_per_second": round(len(ok) / wall, 3) if wall else None,
            "completion_tokens_per_second": round(sum(completions) / wall, 1)
            if wall and completions else None,
            "prompt_tokens_per_second": round(sum(prompts) / wall, 1)
            if wall and prompts else None,
        },
        "measured_prompt_tokens_mean": round(statistics.mean(prompts))
        if prompts else None,
    }

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if ok and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
