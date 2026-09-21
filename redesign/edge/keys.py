"""Render the nginx identity map. LiteLLM owns who the key is.

    python3 -m redesign.edge.keys --output /etc/nginx/conf.d/00-k3-keys.conf

SYSTEM-DESIGN §4: nginx is transport + attribution, not tenancy. Any
Authorization: Bearer <token> is a conn/req identity and is forwarded
unchanged. Missing or non-Bearer requests are denied here. Spend, allowlists
and user names are LiteLLM's.
"""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path

# customers.tsv is not the tenancy layer. Kept only so a leftover table
# cannot be interpolated into nginx (quote-injection).
UNSAFE = frozenset('"\\\n\r;{}')


class KeyTableError(ValueError):
    pass


def escape_or_reject(value: str, field: str) -> str:
    bad = sorted({c for c in value if c in UNSAFE})
    if bad:
        raise KeyTableError(
            f"{field} contains {bad!r}; refusing to interpolate into an nginx map"
        )
    return value


def load_customers(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise KeyTableError(f"no customer table at {path}")
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 2:
            raise KeyTableError(f"{path}:{line_no} is not name<TAB>key")
        name, key = parts[0].strip(), parts[1].strip()
        if not name or not key:
            raise KeyTableError(f"{path}:{line_no} has an empty field")
        rows.append((escape_or_reject(name, "name"), escape_or_reject(key, "key")))
    if not rows:
        raise KeyTableError(f"{path} has no customers")
    return rows


def render_map(_rows: list[tuple[str, str]] | None = None) -> str:
    return (
        "# SYSTEM-DESIGN §4. Nginx is not the tenancy layer.\n"
        "# Any non-empty Authorization is admitted to LiteLLM.\n"
        "# Do not list keys here. Do not swap Authorization for a master key.\n"
        "map_hash_bucket_size 256;\n"
        "map $http_authorization $k3_customer {\n"
        '    default "bearer";\n'
        '    ""      "";\n'
        "}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.edge.keys")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, help="ignored; customers.tsv is not tenancy")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_map(), encoding="utf-8")
    args.output.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)
    print(f"OK    pass-through Bearer map -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
