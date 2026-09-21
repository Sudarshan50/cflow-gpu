"""Labeled Prometheus exposition parse and a tiny PromQL subset.

The gateway's request-path parser flattens labels on purpose. Alerts cannot:
P0 TTFT and shed counters are only meaningful with their class/outcome labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[-+0-9.eE]+)\s*$"
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
_EXPR = re.compile(
    r"^(?:increase\((?P<inc_name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<inc_labels>[^}]*)\})?\[(?P<window>[^\]]+)\]\)"
    r"|(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?)"
    r"\s*(?P<op>>|>=|<|<=|==)\s*(?P<threshold>[-+0-9.eE]+)\s*$"
)
_WINDOW = re.compile(r"^(?P<n>\d+)(?P<unit>[smhd])$")


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float

    def matches(self, name: str, labels: dict[str, str]) -> bool:
        if self.name != name:
            return False
        have = dict(self.labels)
        return all(have.get(key) == value for key, value in labels.items())


@dataclass(frozen=True)
class Expression:
    name: str
    labels: dict[str, str]
    op: str
    threshold: float
    increase_window_seconds: float | None = None


def parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return {match.group(1): match.group(2) for match in _LABEL.finditer(raw)}


def parse_exposition(text: str) -> list[Sample]:
    samples: list[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        samples.append(
            Sample(
                name=match.group("name"),
                labels=tuple(sorted(parse_labels(match.group("labels")).items())),
                value=float(match.group("value")),
            )
        )
    return samples


def parse_window(raw: str) -> float:
    match = _WINDOW.match(raw.strip())
    if not match:
        raise ValueError(f"unsupported window: {raw}")
    n = int(match.group("n"))
    unit = match.group("unit")
    return float({"s": 1, "m": 60, "h": 3600, "d": 86400}[unit] * n)


def parse_expression(expr: str) -> Expression:
    match = _EXPR.match(expr.strip())
    if not match:
        raise ValueError(f"unsupported alert expression: {expr}")
    if match.group("inc_name"):
        return Expression(
            name=match.group("inc_name"),
            labels=parse_labels(match.group("inc_labels")),
            op=match.group("op"),
            threshold=float(match.group("threshold")),
            increase_window_seconds=parse_window(match.group("window")),
        )
    return Expression(
        name=match.group("name"),
        labels=parse_labels(match.group("labels")),
        op=match.group("op"),
        threshold=float(match.group("threshold")),
    )


def sample_value(samples: list[Sample], name: str, labels: dict[str, str]) -> float:
    return sum(s.value for s in samples if s.matches(name, labels))


def compare(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    return value == threshold
