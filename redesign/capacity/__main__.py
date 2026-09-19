"""CLI entry point. Wires a report to a renderer and prints it.

    python3 -m redesign.capacity
    python3 -m redesign.capacity --sweep
    python3 -m redesign.capacity --json
    python3 -m redesign.capacity --verify

--verify exits 0 when the replication hypothesis holds, 1 when it does not.
"""

from __future__ import annotations

import argparse
import sys

from . import report as report_module
from .renderers import JsonRenderer, Renderer, TextRenderer


def _select_renderer(as_json: bool) -> Renderer:
    return JsonRenderer() if as_json else TextRenderer()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.capacity", description=__doc__)
    parser.add_argument("--sweep", action="store_true",
                        help="include concurrency vs prompt length")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--verify", action="store_true",
                        help="hypothesis test only; exit 0 if replicated")
    args = parser.parse_args(argv)

    report = report_module.build(include_sweep=args.sweep)

    if args.verify:
        result = report.hypothesis
        for name, value in vars(result).items():
            print(f"{name:32s} {value}")
        return 0 if result.confirmed else 1

    print(_select_renderer(args.json).render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
