"""Z1/Z3 -- what the surviving edge logs say about production traffic.

    python3 -m redesign.traffic
    python3 -m redesign.traffic --window 24h
    python3 -m redesign.traffic --json

Exit 1 when any check is CRITICAL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import textwrap
from pathlib import Path

from .analysis import CRITICAL, analyse
from .records import TrafficWindow
from .sources import ProdStatsSource

DEFAULT_STATS = Path("eval/runs/20260919T145955Z-baseline/prod_stats.json")
WIDTH = 78


def _render_text(window: TrafficWindow, findings) -> str:
    lines = [
        "=" * WIDTH,
        "EDGE TRAFFIC ANALYSIS".center(WIDTH),
        "docs/SYSTEM-DESIGN.md 3".center(WIDTH),
        "=" * WIDTH,
        f"\nsource        {window.source}",
        f"coverage      {window.coverage_seconds / 60:,.0f} min"
        f"  ({window.requests:,} requests, {window.requests_per_minute:,.2f}/min)",
        f"outcome       {window.successful:,} ok, {window.failures:,} failed"
        f"  ({window.failure_rate:.1%})",
        f"status spread {json.dumps(window.status_spread)}",
        "",
    ]

    for finding in findings:
        lines.append(f"[{finding.severity:8s}] {finding.check}")
        lines.append(f"            {finding.headline}")
        if finding.detail:
            lines.append(f"            {finding.detail}")
        if finding.implication:
            lines.append("")
            for line in textwrap.wrap(finding.implication, WIDTH - 14):
                lines.append(f"            {line}")
        lines.append("")

    lines.append("=" * WIDTH)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.traffic", description=__doc__)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--window", default="1h")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.stats.exists():
        parser.error(f"no such file: {args.stats}")

    window = ProdStatsSource(args.stats, args.window).load()
    findings = analyse(window)

    if args.json:
        print(json.dumps([dataclasses.asdict(f) for f in findings], indent=2))
    else:
        print(_render_text(window, findings))

    return 1 if any(f.severity == CRITICAL for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
