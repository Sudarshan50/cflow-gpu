#!/usr/bin/env python3
"""Finish only c=16 after an unrelated request invalidated its warmup.

The original failed report remains intact. Previously completed c=4/c=8 phases
are retained with a source-file hash; all included phases are revalidated by the
normal comparison contract before this composite report can pass.
"""
import argparse
import concurrent.futures
import hashlib
import json
import time
import uuid
from pathlib import Path

import bench
from compare import index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Preserve earlier evidence")
    raw = args.source.read_bytes()
    original = json.loads(raw)
    expected = {(c, kind, r) for c in (4, 8) for kind in ("warm", "mixed") for r in range(2)}
    recorded = {(p["concurrency"], p["kind"], p["round"]) for p in original["phases"]}
    if (original.get("passed") is not False or original.get("variant") != "batch2048"
            or original.get("rounds") != 2 or recorded != expected or len(original["phases"]) != len(expected)
            or "Non-isolated phase: expected 1024 output tokens/16 completions" not in original.get("failure", "")):
        raise RuntimeError("Source is not the specific interrupted c=16 warmup")
    blocks = int(original["identity"]["cache_config"]["num_gpu_blocks"])
    if bench.identity(args.base, "batch2048", blocks) != original["identity"]:
        raise RuntimeError("Runtime changed since the retained measured phases")
    report = {**original, "resumed_utc": bench.utc(), "passed": False,
              "resumed_from": {"path": str(args.source), "sha256": hashlib.sha256(raw).hexdigest(),
                               "discarded_warmup_failure": original["failure"]}}
    report.pop("failure", None)

    def save():
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    try:
        warm = [bench.fixture(args.base, f"warm-{i:02}", bench.WARM_INPUT) for i in range(16)]
        cold = [bench.fixture(args.base, f"cold-{i}", bench.COLD_INPUT) for i in range(2)]
        signatures = [{key: value for key, value in spec.items() if key != "payload"} for spec in warm+cold]
        if signatures != original["fixtures"]:
            raise RuntimeError("Reconstructed fixtures differ from the earlier phases")
        namespace = uuid.uuid4().hex
        before = bench.metrics(args.base)
        if not bench.idle(before):
            raise RuntimeError("Engine is busy before resumed cache priming")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            seeds = list(pool.map(lambda i: bench.stream(args.base, warm[i], f"{namespace}-warm-{i}",
                                                        1, f"seed-{i}"), range(16)))
        time.sleep(.2)
        bench.accounting(before, bench.metrics(args.base), seeds)
        report["warmups"].append(bench.phase(args.base, warm, [], namespace, 16, "warmup", 0, output=64))
        save()
        for kind in ("warm", "mixed"):
            for round_id in range(2):
                phase = bench.phase(args.base, warm, cold if kind == "mixed" else [], namespace,
                                    16, kind, round_id)
                report["phases"].append(phase)
                save()
                print(json.dumps({"concurrency": 16, "kind": kind, "round": round_id,
                                  **phase["summary"]}), flush=True)
        if bench.identity(args.base, "batch2048", blocks) != report["identity"]:
            raise RuntimeError("Runtime changed during the resumed phases")
        report["passed"] = True
        index(report)
    except Exception as exc:
        report["passed"] = False
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["ended_utc"] = bench.utc()
        save()


if __name__ == "__main__":
    main()
