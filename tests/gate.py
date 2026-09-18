#!/usr/bin/env python3
"""
Kimi-K3 correctness gate.

Catches the failure mode this deployment actually hit: a config that starts
cleanly, serves at full speed, and emits fluent nonsense. Large-batch
benchmarks do NOT expose it; only correctness probes do.

Three tiers, cheapest first:
  1 CORRUPTION  logprob confidence, counting, determinism   (~5s)
  2 REASONING   16 arithmetic word problems, exact answers  (~30s)
  3 LONGCTX     ~11.5k-token needle retrieval               (~15s)

Tier 3 matters specifically: vLLM #51039 reports NaN logits after long-context
prefill, and this deployment's production shape is 12,288-token prompts. A gate
that only tests short prompts would miss it entirely.

Usage:
  gate.py              all tiers, compare against baseline if present
  gate.py --quick      tier 1 only
  gate.py --baseline   run all tiers and record results as the new baseline

Exit 0 = PASS, 1 = FAIL.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

# Talk to vLLM directly on loopback, bypassing the nginx TLS edge. The edge
# only exposes /v1 and would add an auth hop for no benefit; vLLM itself still
# requires the key because VLLM_API_KEY is set in the container env. This also
# keeps the gate meaningful when the cert or DNS is broken but the engine is fine.
BASE = os.environ.get("K3_BASE_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("K3_MODEL", "FW-Kimi-K3")
# Baseline sits next to this file so the repo copy and the installed copy do
# not silently score against each other's recorded run.
BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gate-baseline.json")
API_KEY = open(os.environ.get("K3_API_KEY_FILE",
                              "/scratch/deploy/api-key.txt")).read().strip()
TIMEOUT = 180

# Factual probes are scored by RANK plus MARGIN, not an absolute logprob.
# Absolute floors are not portable across prompts: " Paris" sits near -0.24
# while " Celsius" sits near -0.70 simply because it competes with " cent"
# and " Cent". Both are correct and both are rank-1. What actually indicates
# corruption is the right token losing its lead over the runner-up.
MARGIN_FLOOR = 0.8
REASONING_FLOOR = 14  # of 16
# Degradation guard: fail if accuracy drops more than this below the baseline,
# even when still above the absolute floor.
REGRESSION_TOLERANCE = 2


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def complete(prompt, max_tokens=16, logprobs=None):
    p = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
         "temperature": 0}
    if logprobs:
        p["logprobs"] = logprobs
    return post("/v1/completions", p)


def chat(content, max_tokens=700):
    return post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0,
    })


# ---------------------------------------------------------------- tier 1
FACTUAL = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Water freezes at zero degrees", " Celsius"),
]


def tier1():
    results, ok = [], True

    for prompt, expect in FACTUAL:
        try:
            r = complete(prompt, 1, logprobs=5)
            top = r["choices"][0]["logprobs"]["top_logprobs"][0]
            ranked = sorted(top.items(), key=lambda kv: -kv[1])
            lp = top.get(expect)
            is_top = bool(ranked) and ranked[0][0] == expect
            runner_up = ranked[1][1] if len(ranked) > 1 else float("-inf")
            margin = (lp - runner_up) if lp is not None else 0.0
            good = is_top and margin >= MARGIN_FLOOR
            detail = (f"{lp:.3f}  margin {margin:+.2f} over "
                      f"{ranked[1][0]!r}" if lp is not None and len(ranked) > 1
                      else "MISSING")
            results.append((f"top1({expect.strip()})", detail, good))
        except Exception as e:
            results.append((f"top1({expect.strip()})", f"ERR {e}", False))
            good = False
        ok &= good

    # Numeric continuation - catches tokenizer / quantization corruption.
    try:
        txt = complete("1 2 3 4 5 6 7 8 9", 12)["choices"][0]["text"]
        good = all(str(n) in txt for n in (10, 11, 12))
        results.append(("counting", txt.strip()[:40], good))
    except Exception as e:
        results.append(("counting", f"ERR {e}", False)); good = False
    ok &= good

    # Semantic stability, NOT bitwise determinism. vLLM is not bitwise
    # deterministic at temperature 0: continuous batching varies batch
    # composition and floating-point reduction is not associative, so
    # identical requests legitimately differ (measured here: 5 requests,
    # 3 distinct outputs). Asserting byte equality would fail on a healthy
    # server. What must hold is that every sample carries the same facts.
    try:
        q = "List the first five prime numbers."
        outs = [chat(q, 200)["choices"][0]["message"]["content"]
                for _ in range(3)]
        need = ("2", "3", "5", "7", "11")
        good = all(all(n in o for n in need) for o in outs)
        distinct = len({o.strip() for o in outs})
        results.append(("semantic stability",
                        f"facts consistent across 3 samples "
                        f"({distinct} distinct texts)", good))
    except Exception as e:
        results.append(("semantic stability", f"ERR {e}", False)); good = False
    ok &= good

    return ok, results


# ---------------------------------------------------------------- tier 2
PROBLEMS = [
    ("Janet has 3 boxes with 7 apples each. She eats 4. How many apples remain?", 17),
    ("A train travels 60 km in 45 minutes. What is its speed in km/h?", 80),
    ("If 5 shirts cost $75, how much do 8 shirts cost in dollars?", 120),
    ("What is the sum of all positive integers from 1 to 20?", 210),
    ("What is 15% of 240?", 36),
    ("A rectangle is 12 cm by 8 cm. What is its perimeter in cm?", 40),
    ("If x + 7 = 22, what is 3x?", 45),
    ("Compute 144 divided by 12, then add 5 times 3.", 27),
    ("What is half of 86, plus 19?", 62),
    ("A book has 300 pages. Sam read 2/5 of it. How many pages are left?", 180),
    ("3 workers finish a job in 6 hours. How many hours would 9 workers "
     "at the same rate take?", 2),
    ("A $200 item is discounted 25%, then a further 10% off the new price. "
     "What is the final price in dollars?", 135),
    ("What is 7 squared minus 4 squared?", 33),
    ("A car drives at 50 mph for 2.5 hours. How many miles does it travel?", 125),
    ("There are 24 students and one third are absent. How many are present?", 16),
    ("Compute 2 + 2 * 3 squared.", 20),
]


def extract_number(text):
    """Last integer in the response, tolerating commas and trailing prose."""
    import re
    nums = re.findall(r"-?\d[\d,]*", text.replace("$", ""))
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None


def solve(item):
    q, expect = item
    try:
        r = chat(q + " Reply with only the final number.", 700)
        c = r["choices"][0]
        txt = (c["message"].get("content") or "").strip()
        got = extract_number(txt)
        return (q[:46], expect, got, got == expect, c.get("finish_reason"))
    except Exception as e:
        return (q[:46], expect, f"ERR {e}", False, "error")


def tier2():
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(solve, PROBLEMS))
    correct = sum(1 for r in rows if r[3])
    return correct, rows


# ---------------------------------------------------------------- tier 3
def tier3():
    """Needle retrieval at production prompt length (~11.5k tokens).

    Also exercises the prefix-cache path, since the filler is identical across
    runs and will be served from cache after the first call.
    """
    filler = ("The quarterly logistics report notes routine warehouse "
              "throughput and standard inventory rotation across regional "
              "depots. ")
    needle = ("IMPORTANT: The authorization code for the Meridian project "
              "is 74812. ")
    # ~14 tokens per filler sentence; ~780 repeats lands near 11.5k tokens.
    # Measured: ~17 tokens per filler repeat. 700 repeats ~= 11.9k tokens,
    # leaving headroom for the needle, question and 120 output tokens inside
    # max_model_len=12800. Sized deliberately close to the 12,288-token
    # production shape.
    body = filler * 350 + needle + filler * 350
    prompt = (body + "\n\nBased only on the text above, what is the "
              "authorization code for the Meridian project? "
              "Reply with only the number.")
    try:
        r = chat(prompt, 120)
        c = r["choices"][0]
        txt = (c["message"].get("content") or "").strip()
        got = extract_number(txt)
        ptoks = r.get("usage", {}).get("prompt_tokens", 0)
        return got == 74812, {"prompt_tokens": ptoks, "answer": txt[:60],
                              "expected": 74812, "got": got}
    except Exception as e:
        return False, {"error": str(e)}


# ---------------------------------------------------------------- driver
def main():
    quick = "--quick" in sys.argv
    write_baseline = "--baseline" in sys.argv
    t0 = time.time()
    summary = {}
    failed = []

    print("=" * 64)
    print("TIER 1  corruption probes")
    print("-" * 64)
    ok1, rows = tier1()
    for name, val, good in rows:
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<22} {val}")
    summary["tier1"] = ok1
    if not ok1:
        failed.append("tier1: corruption probes")

    if not quick:
        print()
        print("TIER 2  reasoning correctness")
        print("-" * 64)
        correct, rows = tier2()
        for q, exp, got, good, finish in rows:
            flag = "PASS" if good else "FAIL"
            extra = "" if finish in ("stop", None) else f"  [{finish}]"
            print(f"  [{flag}] {q:<48} want={exp:<5} got={got}{extra}")
        print(f"  -> {correct}/{len(PROBLEMS)} correct "
              f"(floor {REASONING_FLOOR})")
        summary["tier2_correct"] = correct
        summary["tier2_total"] = len(PROBLEMS)
        if correct < REASONING_FLOOR:
            failed.append(f"tier2: {correct}/{len(PROBLEMS)} below "
                          f"floor {REASONING_FLOOR}")

        print()
        print("TIER 3  long-context integrity (production prompt shape)")
        print("-" * 64)
        ok3, info = tier3()
        print(f"  [{'PASS' if ok3 else 'FAIL'}] needle retrieval  {info}")
        summary["tier3"] = ok3
        summary["tier3_prompt_tokens"] = info.get("prompt_tokens", 0)
        if not ok3:
            failed.append("tier3: long-context needle retrieval")

    # Degradation check against a recorded baseline.
    try:
        with open(BASELINE_PATH) as f:
            base = json.load(f)
        if not quick and "tier2_correct" in base:
            drop = base["tier2_correct"] - summary.get("tier2_correct", 0)
            if drop > REGRESSION_TOLERANCE:
                failed.append(
                    f"regression: reasoning {base['tier2_correct']} -> "
                    f"{summary['tier2_correct']} (drop {drop} > "
                    f"{REGRESSION_TOLERANCE})")
            print()
            print(f"baseline {base.get('recorded','?')}: "
                  f"tier2 {base['tier2_correct']}/{base.get('tier2_total','?')}"
                  f"  (now {summary.get('tier2_correct')})")
    except FileNotFoundError:
        print()
        print("no baseline recorded (run with --baseline to create one)")

    elapsed = time.time() - t0
    print()
    print("=" * 64)
    if failed:
        print(f"GATE FAIL in {elapsed:.1f}s - DO NOT BENCHMARK OR SERVE")
        for f_ in failed:
            print(f"  - {f_}")
        print("=" * 64)
        return 1

    print(f"GATE PASS in {elapsed:.1f}s")
    print("=" * 64)

    if write_baseline:
        summary["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(BASELINE_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"baseline written to {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
