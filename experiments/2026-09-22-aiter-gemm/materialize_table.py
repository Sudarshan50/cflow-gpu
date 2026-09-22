#!/usr/bin/env python3
"""Materialize a unique complete AITER BF16 table plus accepted overlay rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEY = (
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "bias",
    "dtype",
    "outdtype",
    "scaleAB",
    "bpreshuffle",
)


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or ()), list(reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream", type=Path)
    parser.add_argument("overlay", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fields, upstream = load(args.upstream)
    overlay_fields, overlay = load(args.overlay)
    if fields != overlay_fields or not fields:
        raise SystemExit("upstream and overlay schemas differ")
    rows = {tuple(row[name] for name in KEY): row for row in upstream}
    if len(rows) != len(upstream):
        raise SystemExit("upstream dispatch table contains duplicate keys")
    for row in overlay:
        if float(row["err_ratio"]) > 0.05:
            raise SystemExit(f"unsafe err_ratio in overlay: {row}")
        rows[tuple(row[name] for name in KEY)] = row
    ordered = sorted(
        rows.values(),
        key=lambda row: tuple(
            int(row[name]) if name in ("cu_num", "M", "N", "K") else row[name]
            for name in KEY
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    print(
        f"materialized {len(ordered)} unique rows "
        f"({len(overlay)} accepted overlay rows)"
    )


if __name__ == "__main__":
    main()
