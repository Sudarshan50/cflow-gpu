"""Z5 -- correctness gate. Run after every launch, before any customer traffic.

    python3 -m redesign.gate --url http://127.0.0.1:8001
    python3 -m redesign.gate --tier 1              # fast smoke
    python3 -m redesign.gate --baseline out.json   # record
    python3 -m redesign.gate --compare out.json    # compare against a record

Exit 0 only if every selected check passes. A non-zero exit means DO NOT route
customer traffic to this build, whatever the throughput numbers say.

Comparison is on pass/fail per check, never on token-exact output: kernel and
layout changes move numerics legitimately, so exact comparison would fail
correct builds while passing subtly broken ones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .checks import CheckResult, build_registry
from .client import GateClient

WIDTH = 78
TIER_NAMES = {1: "corruption", 2: "reasoning", 3: "long context"}


def run_checks(client: GateClient, tiers: set[int], long_context: tuple[int, ...]):
    results = []
    for check in build_registry(long_context):
        if check.tier not in tiers:
            continue
        print(f"  {check.name:42s} ", end="", flush=True)
        result = check.run(client)
        print(f"{result.status}  {result.latency_seconds:6.2f}s")
        if not result.passed:
            print(f"      {result.detail}")
            if result.observed:
                print(f"      observed: {result.observed}")
        results.append(result)
    return results


def render(results: list[CheckResult], elapsed: float) -> str:
    lines = ["", "=" * WIDTH]
    for tier in sorted({r.tier for r in results}):
        tier_results = [r for r in results if r.tier == tier]
        failed = [r for r in tier_results if not r.passed]
        status = "PASS" if not failed else f"FAIL ({len(failed)})"
        lines.append(
            f"  tier {tier} {TIER_NAMES[tier]:<14s} "
            f"{len(tier_results) - len(failed)}/{len(tier_results)}  {status}"
        )

    failures = [r for r in results if not r.passed]
    lines.append("-" * WIDTH)
    if not results:
        lines.append("  GATE FAIL   no checks ran")
        lines.append("")
        lines.append("  A gate that runs nothing proves nothing. Exiting non-zero")
        lines.append("  rather than licensing a deploy on an empty result set.")
    elif failures:
        lines.append(f"  GATE FAIL   {len(failures)}/{len(results)} checks failed "
                     f"in {elapsed:.1f}s")
        lines.append("")
        lines.append("  Do NOT route customer traffic to this build, and do NOT")
        lines.append("  accept a throughput result from it. Quantised KV fails as")
        lines.append("  fluent, confident, wrong output -- which is what this")
        lines.append("  catches and what a benchmark cannot.")
    else:
        lines.append(f"  GATE PASS   {len(results)} checks in {elapsed:.1f}s")
    lines.append("=" * WIDTH)
    return "\n".join(lines)


def compare(results: list[CheckResult], baseline_path: Path) -> int:
    baseline = {
        entry["name"]: entry["passed"]
        for entry in json.loads(baseline_path.read_text())["results"]
    }
    regressions, fixes, new = [], [], []
    for result in results:
        if result.name not in baseline:
            new.append(result.name)
        elif baseline[result.name] and not result.passed:
            regressions.append(result.name)
        elif not baseline[result.name] and result.passed:
            fixes.append(result.name)

    print("\nCOMPARED TO BASELINE")
    print(f"  baseline      {baseline_path}")
    for label, names in (("regressed", regressions), ("fixed", fixes), ("new", new)):
        if names:
            print(f"  {label:<13s} {', '.join(names)}")
    if not (regressions or fixes or new):
        print("  identical pass/fail profile")

    if regressions:
        print("\n  REGRESSION. This build loses correctness the baseline had.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.gate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="default")
    parser.add_argument("--api-key")
    parser.add_argument("--tier", type=int, action="append", choices=[1, 2, 3],
                        help="repeatable; default is all tiers")
    parser.add_argument("--long-context", type=int, nargs="+", default=[32_000, 128_000],
                        help="context sizes for tier 3, in tokens")
    parser.add_argument("--baseline", type=Path, help="write results here")
    parser.add_argument("--rebaseline", action="store_true",
                        help="overwrite an existing baseline (refused otherwise)")
    parser.add_argument("--compare", type=Path, help="compare against a recorded run")
    args = parser.parse_args(argv)

    tiers = set(args.tier) if args.tier else {1, 2, 3}
    client = GateClient(args.url, model=args.model, api_key=args.api_key)

    print(f"gate -> {args.url}  (tiers {sorted(tiers)})")
    started = time.monotonic()
    results = run_checks(client, tiers, tuple(args.long_context))
    elapsed = time.monotonic() - started

    print(render(results, elapsed))

    if args.baseline and args.baseline.exists() and not args.rebaseline:
        # Overwriting the golden record with results from a different engine
        # makes every later --compare pass against garbage, silently. That
        # defeats the only mechanism that catches quantised-KV silent garbage.
        print(f"\n  baseline {args.baseline} already exists; not overwriting."
              f"\n  Pass --rebaseline to replace it deliberately.")
    elif args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps({
            "url": args.url,
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": [asdict(r) for r in results],
        }, indent=2))
        print(f"\n  baseline written to {args.baseline}")

    # `all([])` is True -- an empty selection must not report success.
    exit_code = 0 if results and all(r.passed for r in results) else 1
    if args.compare:
        exit_code = max(exit_code, compare(results, args.compare))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
