#!/usr/bin/env python3
"""Compare the baseline heuristic rejection with actual rendered token counts.

Only synthetic /tokenize requests are sent. No generation or caller content.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from bench import ROOT, digest, idle, metrics, tokenize, utc

sys.path.insert(0, str(ROOT))
from redesign.tenancy.policy import TenancyPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output path")
    texts = {
        "english": "Explain safe stream parsing.\n" + "Check ordering, validate input, and record the result.\n" * 1000,
        "indented_python": "Read this synthetic code without executing it.\n" + "                pass\n" * 50000,
        "json_lines": "Read these synthetic records.\n" + "\n".join(json.dumps({"module": "scheduler", "status": "ready", "values": [1, 2, 3, 4, 5]}) for _ in range(2500)),
        "cjk": "请检查以下合成记录。\n" + "这是用于验证分词计数的合成数据。" * 3000,
        "emoji": "Synthetic display symbols:\n" + "👩🏽‍💻" * 20000,
    }
    cases = {name: {"messages": [{"role": "user", "content": text}],
                    "chat_template_kwargs": {"thinking": False}, "max_tokens": 512}
             for name, text in texts.items()}
    cases["unicode_tool_documentation"] = {
        "messages": [{"role": "user", "content": "Describe the available synthetic lookup tool."}],
        "tools": [{"type": "function", "function": {"name": "lookup_record",
                   "description": "返回指定模块的合成测试记录。" * 12000,
                   "parameters": {"type": "object", "properties": {}}}}],
        "chat_template_kwargs": {"thinking": False}, "max_tokens": 512,
    }
    policy = TenancyPolicy(max_model_len=262144)
    before = metrics(args.base)
    if not idle(before):
        raise RuntimeError("Run token-count audit between performance phases")
    report = {"started_utc": utc(), "scope": "Synthetic rendered-token checks; no inference", "cases": []}
    for name, payload in cases.items():
        rendered_payload = {key: value for key, value in payload.items() if key != "max_tokens"}
        started = time.monotonic()
        tokens = tokenize(args.base, rendered_payload)
        elapsed = time.monotonic() - started
        estimated = policy.estimator.estimate(payload)
        decision = policy.decide(payload, "FW-Kimi-K3")
        fits = len(tokens) + 512 + 256 <= 262144
        row = {"case": name, "payload_sha256": digest(payload), "rendered_tokens": len(tokens),
               "heuristic_tokens": estimated, "heuristic_relative_error_percent": 100 * (estimated / len(tokens) - 1),
               "tokenization_seconds": elapsed, "baseline_policy_rejects": bool(decision.reject),
               "fits_with_512_output_and_256_reserve": fits,
               "false_positive_rejection": bool(decision.reject) and fits}
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
    after = metrics(args.base)
    if not idle(after) or any(after["values"].get(key, 0) != before["values"].get(key, 0)
                             for key in ("vllm:generation_tokens_total", "vllm:request_success_total")):
        raise RuntimeError("Unrelated inference overlapped tokenizer audit")
    report["ended_utc"] = utc()
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
