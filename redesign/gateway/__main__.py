"""Z4 -- dry-run the gateway policy against a synthetic traffic mix.

    python3 -m redesign.gateway
    python3 -m redesign.gateway --max-model-len 262144 --ceiling 96
    python3 -m redesign.gateway --json

Exercises the decision path with no engine and no GPU. The mix follows the
prompt-length distribution observed before the teardown and the fixed
max_tokens that Foundry-style SDKs send, which is what produced the 11.5%
rejection rate in docs/SYSTEM-DESIGN.md 3.4.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .capture import MemorySink, TraceRecorder
from .clamping import PROMPT_TOO_LONG
from .models import RequestEnvelope
from .policy import build_default

FOUNDRY_DEFAULT_MAX_TOKENS = 128_000

# (prompt tokens, share, sends a fixed max_tokens, has tools)
SYNTHETIC_MIX = (
    (5_000, 0.53, True, True),
    (15_000, 0.16, True, True),
    (35_000, 0.12, True, True),
    (75_000, 0.12, True, True),
    (150_000, 0.04, True, False),
    (250_000, 0.03, True, False),
)

SAMPLE_SIZE = 1_000

# 75 of 651 requests returned 400 in the surviving window (SYSTEM-DESIGN 3.4).
OBSERVED_REJECTION_SHARE = 75 / 651


def _envelopes() -> list[RequestEnvelope]:
    out = []
    for prompt_tokens, share, fixed_max, has_tools in SYNTHETIC_MIX:
        for _ in range(round(SAMPLE_SIZE * share)):
            out.append(
                RequestEnvelope(
                    customer="synthetic",
                    prompt_tokens=prompt_tokens,
                    requested_max_tokens=FOUNDRY_DEFAULT_MAX_TOKENS if fixed_max else None,
                    streaming=True,
                    has_tools=has_tools,
                )
            )
    return out


def _simulate(max_model_len: int, ceiling: int) -> dict:
    policy = build_default(max_model_len=max_model_len, concurrency_ceiling=ceiling)
    recorder = TraceRecorder(MemorySink())

    classes: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    clamps: Counter[str] = Counter()
    baseline_rejections = 0
    rescued = 0
    unservable = 0

    for envelope in _envelopes():
        decision = policy.decide(envelope)
        recorder.record(decision)

        classes[decision.traffic_class] += 1
        outcomes[decision.outcome.name] += 1
        clamps[decision.clamp.reason] += 1
        baseline_rejections += decision.clamp.exceeded_window
        rescued += decision.rescued_from_rejection
        unservable += decision.clamp.reason == PROMPT_TOO_LONG

        policy.release(decision)

    total = sum(outcomes.values())
    return {
        "max_model_len": max_model_len,
        "concurrency_ceiling": ceiling,
        "requests": total,
        "classes": dict(classes),
        "outcomes": dict(outcomes),
        "clamp_reasons": dict(clamps),
        "baseline_rejections": baseline_rejections,
        "baseline_rejection_share": baseline_rejections / total if total else 0.0,
        "rescued_from_400": rescued,
        "rescued_share": rescued / total if total else 0.0,
        "unservable_prompts": unservable,
        "observed_rejection_share": OBSERVED_REJECTION_SHARE,
        "unexplained_share": max(
            0.0, OBSERVED_REJECTION_SHARE - (baseline_rejections / total if total else 0.0)
        ),
    }


def _render(result: dict) -> str:
    width = 74
    lines = [
        "=" * width,
        "GATEWAY POLICY DRY RUN".center(width),
        "docs/SYSTEM-DESIGN.md 5, register item D2".center(width),
        "=" * width,
        f"\nmax-model-len       {result['max_model_len']:,}",
        f"concurrency ceiling {result['concurrency_ceiling']}",
        f"requests simulated  {result['requests']:,}",
        "\nCLASS ASSIGNMENT",
    ]
    for name, count in sorted(result["classes"].items()):
        lines.append(f"  {name:22s} {count:>6,}  {count / result['requests']:>6.1%}")

    lines.append("\nOUTCOMES")
    for name, count in sorted(result["outcomes"].items()):
        lines.append(f"  {name:22s} {count:>6,}  {count / result['requests']:>6.1%}")

    lines.append("\nCLAMP REASONS")
    for name, count in sorted(result["clamp_reasons"].items()):
        lines.append(f"  {name:22s} {count:>6,}  {count / result['requests']:>6.1%}")

    lines += [
        "\nREGISTER ITEM D2",
        f"  engine rejects today, unguarded      {result['baseline_rejections']:>6,}"
        f"  {result['baseline_rejection_share']:>6.1%}",
        f"  rescued by the clamp                 {result['rescued_from_400']:>6,}"
        f"  {result['rescued_share']:>6.1%}",
        f"  prompts unservable at any max_tokens {result['unservable_prompts']:>6,}",
        "",
        f"  observed production rejection rate   "
        f"       {result['observed_rejection_share']:>6.1%}",
        f"  NOT explained by this cause          "
        f"       {result['unexplained_share']:>6.1%}",
        "",
        "  The fixed-max_tokens reservation accounts for part of the observed",
        "  400s, not all of them. The remainder has another cause that the",
        "  surviving logs do not identify -- the gateway's error taxonomy is",
        "  what will name it on first boot. Do not claim D2 recovers 11.5%.",
        "=" * width,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.gateway", description=__doc__)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--ceiling", type=int, default=96)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = _simulate(args.max_model_len, args.ceiling)
    print(json.dumps(result, indent=2) if args.json else _render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
