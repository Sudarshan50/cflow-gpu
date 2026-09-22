#!/usr/bin/env python3
"""Matched c=1 cache microbenchmark with fixed tokens and explicit warmups.

These are cache-specific first-token timings, not an aggregate-throughput test.
Only synthetic cache salts vary between runs. An identical fixture revision and
prompt hashes permit checking that both engine variants saw the same inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import uuid
from pathlib import Path

from study import checked_cache_config, generate, identity, tokenize, utc
from validate import fixture

REVISION = "prefix-latency-v1"
WARMUP = 3
SAMPLES = 8
CASES = (("cold-7675", 7675), ("sibling-2304", 7675),
         ("exact-988", 988), ("exact-1024", 1024),
         ("exact-7680", 7680), ("stable-chat-8796", 8796))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="FW-Kimi-K3")
    parser.add_argument("--variant", choices=("baseline", "tail128"), required=True)
    parser.add_argument("--cache-blocks", type=int,
                        help="Common fixed cache capacity for both benchmark arms")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; use a new filename to preserve earlier evidence")
    if args.cache_blocks is not None and args.cache_blocks <= 0:
        parser.error("--cache-blocks must be positive")
    report = {"started_utc": utc(), "identity": identity(args.variant, cache_blocks=args.cache_blocks),
              "cache_config": checked_cache_config(args.base, args.variant, cache_blocks=args.cache_blocks),
              "variant": args.variant, "fixture_revision": REVISION,
              "cache_block_control": args.cache_blocks,
              "warmups_per_case": WARMUP, "samples_per_case": SAMPLES,
              "requests": [], "summary": {}, "passed": False}
    run = uuid.uuid4().hex

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    try:
        pattern = tokenize(args.base, args.model, {"prompt": " alpha beta gamma delta epsilon zeta eta theta",
                                                 "add_special_tokens": False})
        choices = tokenize(args.base, args.model, {"prompt": " apple banana cherry dragonfruit elderberry fig grape "
                           "hazelnut iris jasmine kiwi lemon mango nectarine orange peach quince raspberry strawberry",
                           "add_special_tokens": False})
        branches = list(dict.fromkeys(choices))[:WARMUP + SAMPLES]
        if len(branches) != WARMUP + SAMPLES:
            raise RuntimeError("Insufficient distinct ordinary tokens for sibling requests")

        def raw_tokens(case, length):
            header = tokenize(args.base, args.model, {"prompt": REVISION + " " + case + "\n",
                                                      "add_special_tokens": False})
            return (header + pattern * (length // len(pattern) + 1))[:length]

        chat, _ = fixture(args.base, args.model, REVISION, 8796)
        codes = tuple(chat)
        expected_hits = {"cold-7675": 0, "sibling-2304": 2304,
                         "exact-988": 896 if args.variant == "tail128" else 768,
                         "exact-1024": 768, "exact-7680": 6912,
                         "stable-chat-8796": 8704 if args.variant == "tail128" else 8448}
        for case, length in CASES:
            seed = raw_tokens(case, length)
            for index in range(WARMUP + SAMPLES):
                expected_text = None
                if case == "stable-chat-8796":
                    code = codes[index % len(codes)]
                    payload, tokens = chat[code]
                    expected_text = "QUARTZ-6143|" + code
                else:
                    tokens = seed[:]
                    if case == "sibling-2304":
                        tokens[2304] = branches[index]
                    payload = {"prompt": tokens}
                salt = f"{run}-{case}" + (f"-{index}" if case.startswith("cold-") else "")
                row = generate(args.base, args.model, {**payload, "cache_salt": salt}, tokens,
                               f"{case}-{index+1}", 768, completion=expected_text is None,
                               expected_text=expected_text)
                row.update({"case": case, "phase": "warmup" if index < WARMUP else "measure",
                            "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
                report["requests"].append(row)
                save()
                if row["ttft_seconds"] is None or row.get("answer_correct") is False:
                    raise RuntimeError("Missing content or incorrect semantic answer")
                if expected_text is not None and row["finish_reasons"] != ["stop"]:
                    raise RuntimeError("Semantic answer was truncated")
                if expected_text is None and (row["output_tokens"] != 1 or row["finish_reasons"] != ["length"]):
                    raise RuntimeError("Raw probe did not complete its fixed one-token workload")
                if index >= WARMUP and row["cached_tokens"] != expected_hits[case]:
                    raise RuntimeError("Measured cache state differs from the intended workload")
            measured = [row for row in report["requests"] if row["case"] == case and row["phase"] == "measure"]
            times = [row["ttft_seconds"] for row in measured]
            summary = {"samples": len(measured), "cached_tokens": expected_hits[case],
                       "ttft_median_seconds": statistics.median(times),
                       "ttft_min_seconds": min(times), "ttft_max_seconds": max(times)}
            report["summary"][case] = summary
            print(json.dumps({"case": case, **summary}), flush=True)
            save()
        if identity(args.variant, cache_blocks=args.cache_blocks) != report["identity"]:
            raise RuntimeError("Runtime changed during benchmark")
        if checked_cache_config(args.base, args.variant, cache_blocks=args.cache_blocks) != report["cache_config"]:
            raise RuntimeError("Effective cache configuration changed during benchmark")
        report["passed"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["ended_utc"] = utc()
        save()


if __name__ == "__main__":
    main()
