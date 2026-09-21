"""Run a P3 distillation campaign against the local engine."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from redesign.gateway.backpressure import CircuitBreaker
from redesign.gateway.engine import EngineClient

from .runner import Checkpoint, DistillRunner, load_prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.distill", description=__doc__)
    parser.add_argument("--engine-url", default=os.environ.get(
        "K3_ENGINE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--model", default=os.environ.get("K3_GATE_MODEL", "FW-Kimi-K3"))
    args = parser.parse_args(argv)

    engine = EngineClient(args.engine_url, snapshot_ttl=2.0)
    runner = DistillRunner(
        engine=engine,
        breaker=CircuitBreaker(engine),
        checkpoint=Checkpoint(args.checkpoint or args.output.with_suffix(".ckpt")),
        output=args.output,
        poll_seconds=args.poll_seconds,
        model=args.model,
    )
    written = runner.run(load_prompts(args.prompts))
    print(f"wrote {written} new samples to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
