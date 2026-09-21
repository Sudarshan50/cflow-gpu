#!/usr/bin/env python3
"""What actually busts the K3 prefix cache, measured against the live encoder.

56% of live requests reused zero prompt tokens at a ~21k mean (622 requests,
2026-09-20), meaning even the first block missed, so the divergence is at the
very head of the prompt. K3 renders tool declarations *before* any conversation
content, so anything unstable in `tools` moves every later token.

The engine's /tokenize endpoint applies the real encoder, so this compares token
IDs rather than reasoning about `encoding_k3.py`. Reported as the shared prefix
length against a reference payload: that is the number the GPU cache can reuse.
"""

from __future__ import annotations

import argparse
import copy
import json
import urllib.request

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "body": {"type": "string"}},
                "required": ["path", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the suite.",
            "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
        },
    },
]
SYSTEM = "You are a coding agent. Follow the tool protocol. " * 40
USER = "Explain what the repository does, then call a tool. " * 40


def tokenize(url: str, model: str, payload: dict) -> list[int]:
    body = {"model": model, **payload}
    request = urllib.request.Request(
        url.rstrip("/") + "/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["tokens"]


def shared_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def reorder_keys(tools: list[dict]) -> list[dict]:
    """Same schemas, keys emitted in reverse order."""
    def flip(obj):
        if isinstance(obj, dict):
            return {k: flip(obj[k]) for k in reversed(list(obj))}
        if isinstance(obj, list):
            return [flip(i) for i in obj]
        return obj
    return flip(copy.deepcopy(tools))


def main() -> int:
    parser = argparse.ArgumentParser(prog="prefix_probe")
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="FW-Kimi-K3")
    args = parser.parse_args()

    base = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "tools": TOOLS,
    }
    reference = tokenize(args.url, args.model, base)

    cases: dict[str, dict] = {
        "identical payload": base,
        "tool dict keys reversed": {**base, "tools": reorder_keys(TOOLS)},
        "tools array rotated": {**base, "tools": [TOOLS[2], TOOLS[0], TOOLS[1]]},
        "tools array reversed": {**base, "tools": list(reversed(TOOLS))},
        "one extra tool appended": {**base, "tools": TOOLS + [{
            "type": "function",
            "function": {"name": "zzz_extra", "description": "x",
                         "parameters": {"type": "object", "properties": {}}},
        }]},
        "next turn appended": {**base, "messages": base["messages"] + [
            {"role": "assistant", "content": "Reading the file now."},
            {"role": "user", "content": "Continue."},
        ]},
    }

    # Each arm needs its own reference: scoring a normalized variant against a
    # raw reference measures the normalizer's effect on the reference as well,
    # which reads as a loss on payloads that did not change.
    from redesign.gateway.media import normalize_payload

    reference_norm = tokenize(args.url, args.model,
                              normalize_payload(copy.deepcopy(base)))

    print(f"reference prompt = {len(reference)} tokens "
          f"(normalized: {len(reference_norm)})\n")
    print(f"{'variation':28s} {'tokens':>7s} {'raw':>13s} {'normalized':>13s}")
    results = {}
    for label, payload in cases.items():
        raw = tokenize(args.url, args.model, payload)
        fixed = tokenize(args.url, args.model,
                         normalize_payload(copy.deepcopy(payload)))
        raw_shared = shared_prefix(reference, raw)
        fixed_shared = shared_prefix(reference_norm, fixed)
        results[label] = {
            "tokens": len(raw),
            "raw_shared_prefix": raw_shared,
            "normalized_shared_prefix": fixed_shared,
            "raw_pct": round(raw_shared / len(raw) * 100, 1) if raw else 0.0,
            "normalized_pct": round(fixed_shared / len(fixed) * 100, 1) if fixed else 0.0,
        }
        print(f"{label:28s} {len(raw):7d} "
              f"{raw_shared:6d} {results[label]['raw_pct']:5.1f}% "
              f"{fixed_shared:6d} {results[label]['normalized_pct']:5.1f}%")

    print("\nA variation whose shared prefix collapses to near zero is a total"
          "\ncache miss for the whole prompt, not just for the part that moved.")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
