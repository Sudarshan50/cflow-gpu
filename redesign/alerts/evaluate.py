"""Evaluate the shipped E2 rules against a scrape (and optional previous scrape)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .parser import (
    Expression,
    Sample,
    compare,
    parse_expression,
    parse_exposition,
    parse_window,
    sample_value,
)


@dataclass(frozen=True)
class AlertRule:
    name: str
    expr: str
    pending_for_seconds: float
    severity: str
    summary: str


@dataclass(frozen=True)
class AlertVerdict:
    name: str
    firing: bool
    pending: bool
    value: float | None
    detail: str
    severity: str
    summary: str


def load_rules(path: Path) -> list[AlertRule]:
    try:
        import yaml
    except ImportError:
        yaml = None
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
        return _from_payload(payload)
    return _from_payload(_minimal_rules_yaml(text))


def _from_payload(payload: Any) -> list[AlertRule]:
    rules: list[AlertRule] = []
    for group in payload.get("groups", []):
        for raw in group.get("rules", []):
            rules.append(
                AlertRule(
                    name=raw["alert"],
                    expr=raw["expr"],
                    pending_for_seconds=parse_window(str(raw.get("for", "0s"))),
                    severity=raw.get("labels", {}).get("severity", "ticket"),
                    summary=raw.get("annotations", {}).get("summary", ""),
                )
            )
    return rules


def _minimal_rules_yaml(text: str) -> dict[str, Any]:
    """Enough of the shipped rules file to load without PyYAML."""
    groups: list[dict[str, Any]] = [{"rules": []}]
    current: dict[str, Any] | None = None
    section = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("- alert:"):
            current = {
                "alert": stripped.split(":", 1)[1].strip(),
                "labels": {},
                "annotations": {},
            }
            groups[0]["rules"].append(current)
            section = ""
            continue
        if current is None or not stripped or stripped.startswith("#"):
            continue
        if stripped == "labels:":
            section = "labels"
            continue
        if stripped == "annotations:":
            section = "annotations"
            continue
        key, _, value = stripped.partition(":")
        value = value.strip()
        if key == "expr":
            current["expr"] = value
        elif key == "for":
            current["for"] = value
        elif section in ("labels", "annotations") and value:
            current[section][key] = value
    return {"groups": groups}


def evaluate(
    rules: list[AlertRule],
    current_text: str,
    previous_text: str | None = None,
    elapsed_seconds: float = 0.0,
    pending_since: dict[str, float] | None = None,
    now: float = 0.0,
) -> list[AlertVerdict]:
    current = parse_exposition(current_text)
    previous = parse_exposition(previous_text) if previous_text else []
    if pending_since is None:
        pending_since = {}
    verdicts = []
    for rule in rules:
        expression = parse_expression(rule.expr)
        value = _value(expression, current, previous, elapsed_seconds)
        breached = value is not None and compare(value, expression.op, expression.threshold)
        started = pending_since.get(rule.name)
        if breached:
            started = now if started is None else started
            pending = (now - started) < rule.pending_for_seconds
            firing = not pending
        else:
            started = None
            pending = False
            firing = False
        if started is None:
            pending_since.pop(rule.name, None)
        else:
            pending_since[rule.name] = started
        verdicts.append(
            AlertVerdict(
                name=rule.name,
                firing=firing,
                pending=pending,
                value=value,
                detail=_detail(expression, value, breached, firing, pending),
                severity=rule.severity,
                summary=rule.summary,
            )
        )
    return verdicts


def _value(
    expression: Expression,
    current: list[Sample],
    previous: list[Sample],
    elapsed_seconds: float,
) -> float | None:
    now = sample_value(current, expression.name, expression.labels)
    if expression.increase_window_seconds is None:
        present = any(s.matches(expression.name, expression.labels) for s in current)
        return now if present else None
    if not previous or elapsed_seconds <= 0:
        return 0.0
    then = sample_value(previous, expression.name, expression.labels)
    return max(0.0, now - then) * (expression.increase_window_seconds / elapsed_seconds)


def _detail(
    expression: Expression,
    value: float | None,
    breached: bool,
    firing: bool,
    pending: bool,
) -> str:
    if value is None:
        return "no such metric"
    state = "firing" if firing else "pending" if pending else "ok"
    return f"{value:g} {expression.op} {expression.threshold:g} ({state})"


def render(verdicts: list[AlertVerdict]) -> dict[str, Any]:
    return {
        "firing": [asdict(v) for v in verdicts if v.firing],
        "pending": [asdict(v) for v in verdicts if v.pending],
        "ok": [asdict(v) for v in verdicts if not v.firing and not v.pending],
    }
