"""Pre-registered GPU sessions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Session:
    id: str
    profile: str
    question: str
    pass_fail: str
    rollback: str
    estimate: str
    blocked_by: tuple[str, ...] = ()


REGISTRY: tuple[Session, ...] = (
    Session(
        id="G-build",
        profile="baseline",
        question="Does the fresh SGLang stack stand up and serve?",
        pass_fail=(
            "Engine loads; `python3 -m redesign.gate` records a baseline; "
            "snapshot taken before any profile change."
        ),
        rollback="Destroy the instance. There is no prior snapshot.",
        estimate="~2 h + 1.5 TB weight pull",
    ),
    Session(
        id="G2",
        profile="dedup",
        question="Does --enable-dp-attention de-duplicate the KV pool for K3?",
        pass_fail=(
            "AGGREGATE served tokens across all DP ranks grows toward ~19M "
            "(never the per-rank pool log line, which stays ~2.3M). "
            "Per-rank occupancy spread is reported. Preemptions 0. Gate passes. "
            "P0 TTFT p95 does not regress against the G-build baseline."
        ),
        rollback="Revert to the G-build snapshot. Do not book G3.",
        estimate="~4 h",
        blocked_by=("G-build", "redesign.probe reports enable_dp_attention present"),
    ),
    Session(
        id="G3",
        profile="dedup-fp8",
        question="Does fp8 KV double the pool again without corrupting output?",
        pass_fail=(
            "Aggregate unique tokens roughly double vs G2, AND every Z5 gate "
            "check still passes against the G-build baseline. Either both, or revert."
        ),
        rollback="Revert to the G2 snapshot. Leave A2 off.",
        estimate="~3 h",
        blocked_by=("G2 pass", "Z5 gate green on G2"),
    ),
    Session(
        id="G4",
        profile="dedup-hicache",
        question="Does native host L2 serve prefixes after GPU eviction?",
        pass_fail=(
            "FAIL recorded 2026-09-20 (a3-readpath). Write path worked "
            "(~806 GB GPU→CPU/wave). Read path silent: CPU→GPU 0, "
            "external_prefix_cache_hits 0, replay TTFT identical to fill. "
            "A3 disabled. Do not raise 64 GB. Do not re-enable without a "
            "proven CPU→GPU hit."
        ),
        rollback="A3 is already off. Leave it off.",
        estimate="~6 h",
        blocked_by=("A1 unavailable or G2/G3 leave KV binding",),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.sessions", description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--id", choices=[s.id for s in REGISTRY])
    args = parser.parse_args(argv)

    sessions = [s for s in REGISTRY if args.id is None or s.id == args.id]
    if args.json:
        print(json.dumps([asdict(s) for s in sessions], indent=2))
        return 0

    width = 78
    print("=" * width)
    print("PRE-REGISTERED GPU SESSIONS".center(width))
    print("docs/SYSTEM-DESIGN.md 11".center(width))
    print("=" * width)
    for session in sessions:
        print(f"\n{session.id}  profile={session.profile}  {session.estimate}")
        print(f"  Q      {session.question}")
        print(f"  PASS   {session.pass_fail}")
        print(f"  UNDO   {session.rollback}")
        if session.blocked_by:
            print(f"  BLOCK  {'; '.join(session.blocked_by)}")
    print("=" * width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
