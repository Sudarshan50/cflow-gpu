#!/usr/bin/env python3
"""Select observed Kimi-K3 decode/chunked-prefill shapes for bounded tuning."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

TARGET_N = 6288
TARGET_K = 7168
# Decode occupancy, transition, and observed chunked-prefill representatives.
REPRESENTATIVE_M = (8, 32, 64, 128, 768, 1536, 3840, 4096)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captured", type=Path)
    parser.add_argument("selected", type=Path)
    args = parser.parse_args()
    with args.captured.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [
            row
            for row in reader
            if int(row["N"]) == TARGET_N
            and int(row["K"]) == TARGET_K
            and int(row["M"]) in REPRESENTATIVE_M
        ]
        fields = reader.fieldnames
    found = {int(row["M"]) for row in rows}
    missing = set(REPRESENTATIVE_M) - found
    if missing:
        raise SystemExit(f"representative shapes were not observed: {sorted(missing)}")
    args.selected.parent.mkdir(parents=True, exist_ok=True)
    with args.selected.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["M"])))
    print(f"selected {len(rows)} observed shapes: {sorted(found)}")


if __name__ == "__main__":
    main()
