#!/usr/bin/env python3
"""Two synthetic ~128k retrieval/cache checks after throughput qualification."""
import argparse
import json
from pathlib import Path

from qualify import submit, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lines = [f"Record {i}: this is unrelated synthetic reference material." for i in range(11200)]
    lines.insert(len(lines)//2, "The unique verification marker is QUARTZ-6139.")
    prompt = "\n".join(lines) + "\nReturn only the unique verification marker."
    body = {"model": "FW-Kimi-K3", "temperature": 0, "seed": 42,
            "chat_template_kwargs": {"thinking": False}, "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}]}
    report = {"label": args.label, "started_utc": utc(), "requests": []}
    try:
        for _ in range(2):
            row, text, _ = submit(args.base, body, timeout=300)
            row["passed"] = text.strip().strip("`\n .") == "QUARTZ-6139" and row["finishes"] == ["stop"]
            report["requests"].append(row)
            if not row["passed"]:
                raise RuntimeError("Long-context marker retrieval failed")
        report["cache_reuse_observed"] = report["requests"][1]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) > 0
        report["passed"] = report["cache_reuse_observed"]
        if not report["passed"]:
            raise RuntimeError("Repeated long-context prefix reuse was not demonstrated")
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["ended_utc"] = utc()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
