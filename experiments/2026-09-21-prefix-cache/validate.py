#!/usr/bin/env python3
"""Cold-versus-cached semantic checks, including partial-tail copy-on-write.

Each fixture retrieves an earlier archive code and a changing runtime code.
Three independently salted cold references are compared with five shared-cache
branches (A, B, A, C, B). Text is synthetic; customer namespaces are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from study import checked_cache_config, generate, identity, lcp, tokenize, utc


def fixture(base, model, run, length):
    lines = [f"Reference {i}: ordinary synthetic reference material."
             for i in range(24 if length < 2000 else min(1024, length // 20))]
    lines.insert(len(lines) // 2, "ARCHIVE_CODE=QUARTZ-6143")
    prefix = "Synthetic parity fixture " + run + "\n" + "\n".join(lines)

    def payload(code, padding):
        return {"messages": [
            {"role": "system", "content": "Read the reference material. Return exactly "
             "ARCHIVE_CODE|RUNTIME_CODE using the two labeled values. No commentary."},
            {"role": "user", "content": prefix + " x" * padding
             + "\nRUNTIME_CODE=" + code + "\nReturn ARCHIVE_CODE|RUNTIME_CODE only."},
        ], "chat_template_kwargs": {"thinking": False}}

    codes = ("RUBY-1736", "RUBY-2847", "RUBY-3958")
    padding = 0
    for _ in range(4):
        tokens = tokenize(base, model, payload(codes[0], padding))
        if len(tokens) == length:
            break
        padding += length - len(tokens)
        if padding < 0:
            raise RuntimeError("Fixture exceeds its requested prompt length")
    fixtures = {code: (payload(code, padding), tokenize(base, model, payload(code, padding)))
                for code in codes}
    boundary = length // 128 * 128
    if any(len(tokens) != length for _, tokens in fixtures.values()):
        raise RuntimeError("Failed to construct equal-length semantic fixtures")
    common = min(lcp(fixtures[codes[0]][1], fixtures[code][1]) for code in codes[1:])
    if not (boundary % 768 and boundary <= common < length):
        raise RuntimeError("Fixture does not exercise a reusable partial-tail checkpoint")
    return fixtures, common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="FW-Kimi-K3")
    parser.add_argument("--variant", choices=("baseline", "tail128"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; use a new filename to preserve earlier evidence")
    report = {"started_utc": utc(), "identity": identity(args.variant),
              "cache_config": checked_cache_config(args.base, args.variant),
              "variant": args.variant, "requests": [], "passed": False}
    run = uuid.uuid4().hex

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    def probe(payload, tokens, code, label, salt):
        row = generate(args.base, args.model, {**payload, "cache_salt": salt}, tokens,
                       label, 768, expected_text="QUARTZ-6143|" + code)
        row["observed_answer"] = row.pop("observed_date_answer")
        row["prompt_sha256"] = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        report["requests"].append(row)
        save()
        print(json.dumps(row), flush=True)
        if not row["answer_correct"] or row["finish_reasons"] != ["stop"]:
            raise RuntimeError("Archive/runtime retrieval failed")
        return row

    try:
        for length in (988, 133468):
            fixtures, common = fixture(args.base, args.model, run, length)
            references = {}
            expected_hit = length // (128 if args.variant == "tail128" else 768)
            expected_hit *= 128 if args.variant == "tail128" else 768
            for index, code in enumerate(("RUBY-1736", "RUBY-2847", "RUBY-1736", "RUBY-3958", "RUBY-2847")):
                payload, tokens = fixtures[code]
                if code not in references:
                    cold = probe(payload, tokens, code, f"{length}-cold-{code}", f"{run}-{length}-cold-{code}")
                    if cold["cached_tokens"] != 0:
                        raise RuntimeError("Salted reference was not cold")
                    references[code] = cold
                warm = probe(payload, tokens, code, f"{length}-shared-{index+1}", f"{run}-{length}-shared")
                warm["lcp_across_variants"] = common
                warm["exact_cold_output_match"] = warm["output_sha256"] == references[code]["output_sha256"]
                save()
                if not warm["exact_cold_output_match"]:
                    raise RuntimeError("Cached output differs from its identical cold reference")
                if warm["cached_tokens"] != (expected_hit if index else 0):
                    raise RuntimeError("Expected shared checkpoint was not reused")
        if identity(args.variant) != report["identity"]:
            raise RuntimeError("Runtime changed during validation")
        if checked_cache_config(args.base, args.variant) != report["cache_config"]:
            raise RuntimeError("Effective cache configuration changed during validation")
        report["passed"] = True
    except Exception as exc:
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        report["ended_utc"] = utc()
        save()


if __name__ == "__main__":
    main()
