"""Scrape the gateway and write the E2 alert file Prometheus would have written.

    python3 -m redesign.alerts --once
    python3 -m redesign.alerts --interval 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from .evaluate import evaluate, load_rules, render


DEFAULT_RULES = Path(__file__).resolve().parents[1] / "deploy" / "alerts" / "k3-gateway.rules.yml"
DEFAULT_STATE = Path("/scratch/deploy-state/alerts/last.json")


def _scrape(url: str, timeout: float) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def _run_once(
    metrics_url: str,
    rules_path: Path,
    state_path: Path,
    timeout: float,
) -> int:
    rules = load_rules(rules_path)
    previous = {}
    if state_path.is_file():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    now = time.time()
    text = _scrape(metrics_url, timeout)
    elapsed = now - float(previous.get("scraped_at", now))
    pending_since = dict(previous.get("pending_since") or {})
    verdicts = evaluate(
        rules,
        text,
        previous.get("exposition"),
        elapsed_seconds=elapsed if previous.get("exposition") else 0.0,
        pending_since=pending_since,
        now=now,
    )
    payload = {
        "scraped_at": now,
        "metrics_url": metrics_url,
        "exposition": text,
        "pending_since": pending_since,
        **render(verdicts),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(state_path)

    firing = payload["firing"]
    for alert in firing:
        print(f"FIRING {alert['name']}: {alert['detail']}", file=sys.stderr)
    if not firing:
        print(f"ok  {len(payload['ok'])} rules quiet", file=sys.stderr)
    return 1 if firing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.alerts", description=__doc__)
    parser.add_argument(
        "--metrics-url",
        default=os.environ.get("K3_ALERTS_METRICS_URL", "http://127.0.0.1:8002/metrics"),
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path(os.environ.get("K3_ALERTS_RULES", str(DEFAULT_RULES))),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(os.environ.get("K3_ALERTS_STATE", str(DEFAULT_STATE))),
    )
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    while True:
        try:
            code = _run_once(args.metrics_url, args.rules, args.state, args.timeout)
        except (URLError, TimeoutError, OSError) as exc:
            print(f"scrape failed: {exc}", file=sys.stderr)
            code = 0
        if args.once or args.interval <= 0:
            return code
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
