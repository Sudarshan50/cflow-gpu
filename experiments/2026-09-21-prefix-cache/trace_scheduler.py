#!/usr/bin/env python3
"""Trace the installed text-prefill splitting method without importing vLLM.

Only the reviewed _mamba_block_aligned_split function is compiled from its AST.
This explains scheduling boundaries; it does not simulate GPU state or timing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS


def trace(split, length, hit, shared_boundary, unit):
    scheduler = NS(cache_config=NS(block_size=768), use_eagle=False,
                   hash_block_size=unit, mamba_partial_cache_hit=unit < 768,
                   max_num_scheduled_tokens=4096,
                   scheduler_config=NS(long_prefill_token_threshold=0))
    request = NS(num_computed_tokens=hit, num_prompt_tokens=length,
                 num_tokens=length, shared_prefix_boundary=shared_boundary)
    ends = []
    while request.num_computed_tokens < length:
        budget = min(4096, length - request.num_computed_tokens)
        scheduled = split(scheduler, request, budget)
        if not 0 < scheduled <= budget:
            raise RuntimeError("Text-only trace stopped making bounded progress")
        request.num_computed_tokens += scheduled
        ends.append(request.num_computed_tokens)
    return {"initial_cache_hit": hit, "shared_boundary": shared_boundary,
            "chunk_ends": ends, "prefill_steps": len(ends)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; preserve earlier evidence")
    source = args.source.read_text()
    tree = ast.parse(source)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name == "_mamba_block_aligned_split"]
    if len(methods) != 1:
        raise RuntimeError("Expected exactly one reviewed prefill-splitting method")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              methods[0]], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(args.source), "exec"), namespace)
    split = namespace["_mamba_block_aligned_split"]
    cases = (
        ("cold-7675", 7675, 0, 0, 0),
        ("second-sibling-2304", 7675, 0, 0, 2304),
        ("warm-sibling-2304", 7675, 2304, 2304, 0),
        ("cold-988", 988, 0, 0, 0),
        ("warm-988", 988, 768, 896, 0),
        ("cold-1024", 1024, 0, 0, 0),
        ("warm-1024", 1024, 768, 768, 0),
        ("cold-7680", 7680, 0, 0, 0),
        ("first-replay-7680", 7680, 3840, 3840, 6912),
        ("warm-replay-7680", 7680, 6912, 6912, 0),
        ("cold-stable-chat", 8796, 0, 0, 0),
        ("warm-stable-chat", 8796, 8448, 8704, 0),
    )
    report = {"source": str(args.source), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "scope": "Text-only scheduling boundaries, not GPU execution", "cases": {}}
    for name, length, baseline_hit, tail_hit, shared in cases:
        report["cases"][name] = {"baseline": trace(split, length, baseline_hit, shared, 768),
                                 "tail128": trace(split, length, tail_hit, shared, 128)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
