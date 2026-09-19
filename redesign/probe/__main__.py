"""Z2 -- static capability probe for MLA KV de-duplication.

    python3 -m redesign.probe
    python3 -m redesign.probe --engine sglang
    python3 -m redesign.probe --json

Exit 0 when at least one probed engine exposes an A1 flag, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from .engines import REGISTRY
from .renderers import JsonRenderer, Renderer, TextRenderer


def _select_renderer(as_json: bool) -> Renderer:
    return JsonRenderer() if as_json else TextRenderer()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.probe", description=__doc__)
    parser.add_argument("--engine", choices=[*REGISTRY, "both"], default="both")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    names = list(REGISTRY) if args.engine == "both" else [args.engine]
    results = [REGISTRY[name]().run() for name in names]

    print(_select_renderer(args.json).render(results))
    return 0 if any(r.a1_available for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
