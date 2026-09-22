#!/usr/bin/env python3
"""Exercise accepted GEMM rows through the production AITER dispatch path."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from aiter.tuned_gemm import tgemm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    with args.overlay.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    torch.manual_seed(42)
    results = []
    for row in rows:
        m, n, k = (int(row[name]) for name in ("M", "N", "K"))
        inp = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(inp, weight)
        actual = tgemm.mm(inp, weight, otype=torch.bfloat16)
        bad = (~torch.isclose(actual, expected, atol=0.05, rtol=0.05)).float()
        err_ratio = bad.mean().item()
        maximum_delta = (actual.float() - expected.float()).abs().max().item()
        for _ in range(10):
            tgemm.mm(inp, weight, otype=torch.bfloat16)
        timings = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            tgemm.mm(inp, weight, otype=torch.bfloat16)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000)
        result = {
            "m": m,
            "n": n,
            "k": k,
            "configured_libtype": row["libtype"],
            "configured_kernel": row["kernelName"],
            "err_ratio": err_ratio,
            "max_abs_delta": maximum_delta,
            "median_us": statistics.median(timings),
            "passes": err_ratio <= 0.05,
        }
        results.append(result)
        del inp, weight, expected, actual
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if not all(result["passes"] for result in results):
        raise SystemExit("production dispatch accuracy gate failed")


if __name__ == "__main__":
    main()
