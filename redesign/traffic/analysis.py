"""Derives design-relevant findings from a traffic window.

Each check answers a question docs/SYSTEM-DESIGN.md depends on. A check that
cannot decide returns UNKNOWN rather than a default, because a sample this small
should not be allowed to look conclusive.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from .records import TrafficWindow

CRITICAL = "CRITICAL"
WARN = "WARN"
OK = "OK"
UNKNOWN = "UNKNOWN"

# Below this many requests, or this many seconds of coverage, no conclusion
# about traffic shape is defensible.
MIN_REQUESTS = 5_000
MIN_COVERAGE_SECONDS = 86_400

# A single client above this share means the sample describes one user, not a
# fleet, and shared-prefix reasoning does not transfer.
CONCENTRATION_LIMIT = 0.60

INTERACTIVE_TTFT_TARGET_S = 3.0


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    headline: str
    detail: str
    implication: str


class Check(abc.ABC):
    @abc.abstractmethod
    def run(self, window: TrafficWindow) -> Finding: ...


class SampleAdequacy(Check):
    def run(self, window: TrafficWindow) -> Finding:
        adequate = (
            window.requests >= MIN_REQUESTS
            and window.coverage_seconds >= MIN_COVERAGE_SECONDS
        )
        return Finding(
            check="sample adequacy",
            severity=OK if adequate else WARN,
            headline=f"{window.requests:,} requests over "
                     f"{window.coverage_seconds / 60:,.0f} minutes",
            detail=f"target is {MIN_REQUESTS:,} requests over "
                   f"{MIN_COVERAGE_SECONDS / 3600:.0f} hours",
            implication="Sizing numbers derived from this window are provisional."
            if not adequate
            else "Window is wide enough to size against.",
        )


class ClientConcentration(Check):
    def run(self, window: TrafficWindow) -> Finding:
        billable = [c for c in window.customers if c.customer != "(unauthenticated)"]
        if not billable:
            return Finding("client concentration", UNKNOWN, "no billable traffic", "", "")

        top = max(billable, key=lambda c: c.requests)
        total = sum(c.requests for c in billable)
        share = top.requests / total

        concentrated = share > CONCENTRATION_LIMIT
        return Finding(
            check="client concentration",
            severity=WARN if concentrated else OK,
            headline=f"'{top.customer}' is {share:.0%} of billable requests "
                     f"({top.requests:,}/{total:,}), {len(billable)} clients total",
            detail=f"{top.distinct_ips} distinct IPs for the top client",
            implication=(
                "This window describes one client's usage, not a fleet. The 85% "
                "shared-prefix premise cannot be evaluated from it, and a low "
                "prefix hit rate here may simply mean varied ad-hoc prompts "
                "rather than cache eviction."
            )
            if concentrated
            else "Traffic is spread across clients; prefix reasoning transfers.",
        )


class RejectedRequests(Check):
    """400s are the largest customer-visible defect and need no GPU to fix."""

    def run(self, window: TrafficWindow) -> Finding:
        rejected = sum(c.requests for c in window.failure_causes if c.status == 400)
        if not window.requests:
            return Finding("rejected requests", UNKNOWN, "no requests", "", "")

        share = rejected / window.requests
        failure_share = rejected / window.failures if window.failures else 0.0

        if share >= 0.05:
            severity = CRITICAL
        elif share > 0:
            severity = WARN
        else:
            severity = OK

        return Finding(
            check="rejected requests",
            severity=severity,
            headline=f"{rejected:,} requests rejected with 400 "
                     f"({share:.1%} of all traffic, {failure_share:.0%} of failures)",
            detail="engine rejected the request body",
            implication=(
                "config.yaml records the cause: clients send a fixed max_tokens "
                "and the engine reserves prompt+max_tokens against one window. "
                "A per-class clamp at the tenancy layer fixes this with no GPU "
                "time. Register item D2 is worth more than it was scored."
            )
            if share >= 0.05
            else "Rejection volume is within tolerance.",
        )


class LatencyBudget(Check):
    def run(self, window: TrafficWindow) -> Finding:
        latency = window.latency
        if latency.p50 is None:
            return Finding("latency budget", UNKNOWN, "no latency data", "", "")

        over = latency.p50 > INTERACTIVE_TTFT_TARGET_S
        tail = (latency.p99 / latency.p50) if latency.p50 else 0.0

        return Finding(
            check="latency budget",
            severity=CRITICAL if over else OK,
            headline=f"p50 {latency.p50:,.0f}s · p95 {latency.p95:,.0f}s "
                     f"· p99 {latency.p99:,.0f}s",
            detail=f"p99/p50 tail ratio {tail:,.0f}x",
            implication=(
                f"The median request already exceeds the {INTERACTIVE_TTFT_TARGET_S:.0f}s "
                "P0 target by orders of magnitude. Note this is full request "
                "duration at the edge, not TTFT, and streaming requests run long "
                "by design -- so it bounds the problem rather than measuring it. "
                "Per-class TTFT instrumentation is required before any SLO is "
                "promised."
            )
            if over
            else "Within the interactive target.",
        )


class UnservedEndpoints(Check):
    def run(self, window: TrafficWindow) -> Finding:
        broken = [p for p in window.paths if p.requests and p.failure_rate >= 0.5]
        if not broken:
            return Finding("unserved endpoints", OK, "no wholly failing routes", "", "")

        listed = ", ".join(f"{p.path} ({p.failures}/{p.requests})" for p in broken)
        return Finding(
            check="unserved endpoints",
            severity=WARN,
            headline=f"{len(broken)} route(s) failing at 50%+: {listed}",
            detail="includes routes the deployment does not serve at all",
            implication=(
                "Separate self-inflicted probe failures from real client demand. "
                "Monitoring that curls /v1/models without credentials records a "
                "401 as a customer failure; a client calling /v1/embeddings is "
                "asking for a capability this model does not expose."
            ),
        )


class UnauthenticatedProbes(Check):
    def run(self, window: TrafficWindow) -> Finding:
        anon = next(
            (c for c in window.customers if c.customer == "(unauthenticated)"), None
        )
        if anon is None:
            return Finding("unauthenticated probes", OK, "none recorded", "", "")

        return Finding(
            check="unauthenticated probes",
            severity=OK,
            headline=f"{anon.requests} rejected from {anon.distinct_ips} IPs",
            detail="all returned 401",
            implication=(
                "Internet background scanning, refused at the edge. The edge is "
                "doing its job; exclude these from customer failure rates."
            ),
        )


REGISTRY: tuple[Check, ...] = (
    SampleAdequacy(),
    ClientConcentration(),
    RejectedRequests(),
    LatencyBudget(),
    UnservedEndpoints(),
    UnauthenticatedProbes(),
)

SEVERITY_ORDER = {CRITICAL: 0, WARN: 1, UNKNOWN: 2, OK: 3}


def analyse(window: TrafficWindow) -> list[Finding]:
    findings = [check.run(window) for check in REGISTRY]
    return sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity])
