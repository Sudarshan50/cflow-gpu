"""First-boot prefix-cache sharing audit.

    python3 -m redesign.probe.cache_salt --url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import sys
from dataclasses import asdict, dataclass

from redesign.gateway.engine import EngineClient, parse_prometheus


HIT_COUNTERS = (
    "sglang:prefix_cache_hits_total",
    "vllm:prefix_cache_hits_total",
    "sglang:prefix_hit_tokens_total",
)

# prefix-match-unit on this vLLM is 128 tokens. These 160 space-separated
# numbers exceed that unit with the Kimi tokenizer, in under 700 characters.
# Keep decoding tiny; even a length-limited thinking response can report usage.
_PROBE = {
    "messages": [{
        "role": "user",
        "content": "Cache sharing probe. Reply OK.\n" + " ".join(f"{n:03d}" for n in range(160)),
    }],
    "stream": False,
    "max_tokens": 8,
    "temperature": 0,
}


@dataclass(frozen=True)
class SaltVerdict:
    first_hits: float | None
    second_hits: float | None
    sharing: bool
    detail: str
    first_cached_tokens: int | float | None = None
    second_cached_tokens: int | float | None = None

    @property
    def passed(self) -> bool:
        return self.sharing


def _counter(metrics: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in metrics and math.isfinite(metrics[name]):
            return metrics[name]
    return None


def compare_counters(first_hits: float, second_hits: float) -> SaltVerdict:
    """Legacy diagnostic helper, NOT evidence of this probe's prefix reuse.

    Retains the historical boolean API for callers examining global activity.
    Concurrent traffic and warm caches make these counters unattributable.
    """
    sharing = second_hits > 0
    if second_hits > first_hits:
        detail = "global hit activity increased"
    elif sharing:
        detail = "global hit activity is positive"
    else:
        detail = "no positive global hit activity"
    return SaltVerdict(
        first_hits=first_hits,
        second_hits=second_hits,
        sharing=sharing,
        detail=f"diagnostic only: {detail}; cannot attribute hits to this probe",
    )


def _request_cached_tokens(engine: EngineClient, payload: dict) -> int | float:
    response = engine.proxy("/v1/chat/completions", payload, stream=False)
    try:
        # Drain even an HTTP error response. Closing an unstarted body is not
        # evidence that prefill or generation completed.
        raw = b"".join(response.body)
        if not 200 <= response.status < 300:
            raise ValueError(f"HTTP {response.status}")
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("malformed JSON response") from exc
        if not isinstance(body, dict) or body.get("error") is not None:
            raise ValueError("invalid completion response")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing or invalid choices")
        for choice in choices:
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise ValueError("invalid choices: expected a chat message")
            # This is a cache probe, not a semantic gate. Empty content is OK
            # when a thinking model uses the eight-token budget on reasoning.
            if choice.get("finish_reason") not in (
                "stop", "length", "tool_calls", "function_call", "content_filter",
            ):
                raise ValueError("missing or invalid finish_reason; incomplete completion")
        usage = body.get("usage")
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if (
            type(cached) not in (int, float)
            or cached < 0
            or (isinstance(cached, float) and not math.isfinite(cached))
        ):
            raise ValueError(
                "missing or invalid usage.prompt_tokens_details.cached_tokens; "
                "need a finite nonnegative numeric per-response cache count"
            )
        return cached
    finally:
        response.close()


def _diagnostic_hits(scrape) -> float | None:
    try:
        return _counter(parse_prometheus(scrape()), HIT_COUNTERS)
    except Exception:  # A failed diagnostic scrape cannot decide the audit.
        return None


def _delta(before: float | None, after: float | None) -> float | None:
    return after - before if before is not None and after is not None else None


def audit(engine: EngineClient, scrape, model: str = "default") -> SaltVerdict:
    payload = {**_PROBE, "model": model}
    hits = [_diagnostic_hits(scrape)]
    cached = []
    failures = []
    for label in ("first", "second"):
        try:
            cached.append(_request_cached_tokens(engine, payload))
        except (OSError, http.client.HTTPException, ValueError, TypeError) as exc:
            cached.append(None)
            failures.append(f"{label} request failed: {exc}")
        hits.append(_diagnostic_hits(scrape))

    sharing = not failures and cached[1] > 0
    if failures:
        detail = "; ".join(failures)
    elif sharing:
        detail = f"second response reports cached_tokens={cached[1]}; prefix reuse observed"
    else:
        detail = "second response reports cached_tokens=0; prefix sharing not demonstrated"
    detail += "; global hit counters are diagnostic only"
    if any(hit is None for hit in hits):
        detail += " (unavailable for some samples)"
    return SaltVerdict(
        first_hits=_delta(hits[0], hits[1]),
        second_hits=_delta(hits[1], hits[2]),
        sharing=sharing,
        detail=detail,
        first_cached_tokens=cached[0],
        second_cached_tokens=cached[1],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="redesign.probe.cache_salt", description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="default")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    engine = EngineClient(args.url)
    verdict = audit(engine, engine.metrics_text, model=args.model)
    if args.json:
        print(json.dumps(asdict(verdict), indent=2))
    else:
        print(f"{'PASS' if verdict.passed else 'FAIL'}  {verdict.detail}")
        print(f"  global hit deltas (diagnostic only): "
              f"first_hits={verdict.first_hits}  second_hits={verdict.second_hits}")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
