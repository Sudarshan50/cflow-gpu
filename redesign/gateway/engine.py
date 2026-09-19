"""Client for the upstream engine: health scraping and request proxying.

Stdlib only. The policy core is transport-agnostic, so if SSE relay throughput
ever becomes the bottleneck this module is the only thing that has to change.

Metric names are candidate lists rather than constants. SGLang and vLLM expose
different names for the same quantity, and a name that does not resolve must
degrade to "no signal" rather than to a false reading -- a breaker that trips on
a missing metric would shed traffic for no reason.
"""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from typing import Iterator
from urllib.parse import urlparse

from .backpressure import EngineHealthSource
from .models import EngineSnapshot

RUNNING_METRICS = (
    "sglang:num_running_reqs",
    "vllm:num_requests_running",
)
WAITING_METRICS = (
    "sglang:num_queue_reqs",
    "vllm:num_requests_waiting",
)
KV_USAGE_METRICS = (
    "sglang:token_usage",
    "sglang:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)
PREEMPTION_METRICS = (
    "sglang:num_preemptions_total",
    "vllm:num_preemptions_total",
)

DEFAULT_TIMEOUT_SECONDS = 600
HEALTH_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class ProxyResponse:
    status: int
    headers: list[tuple[str, str]]
    body: Iterator[bytes]


def parse_prometheus(text: str) -> dict[str, float]:
    """Flattens exposition text to name -> value, ignoring labels.

    Label-summing is deliberate: every metric read here is a single-instance
    gauge or counter, and a label dimension appearing later should not silently
    change the reading.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        if not rest:
            continue
        name = name.split("{", 1)[0]
        try:
            value = float(rest.strip())
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def _first(metrics: dict[str, float], candidates: tuple[str, ...]) -> float | None:
    for name in candidates:
        if name in metrics:
            return metrics[name]
    return None


class EngineClient(EngineHealthSource):
    def __init__(self, base_url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        parsed = urlparse(base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._timeout = timeout
        self._last_preemptions: float | None = None

    def _connect(self, timeout: int) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def healthy(self) -> bool:
        try:
            conn = self._connect(HEALTH_TIMEOUT_SECONDS)
            conn.request("GET", "/health")
            return 200 <= conn.getresponse().status < 300
        except OSError:
            return False
        finally:
            try:
                conn.close()
            except (OSError, UnboundLocalError):
                pass

    def snapshot(self) -> EngineSnapshot:
        """Reads engine health. Raises on failure so the breaker fails open."""
        conn = self._connect(HEALTH_TIMEOUT_SECONDS)
        try:
            conn.request("GET", "/metrics")
            response = conn.getresponse()
            metrics = parse_prometheus(response.read().decode("utf-8", "replace"))
        finally:
            conn.close()

        preemptions_total = _first(metrics, PREEMPTION_METRICS)
        rate = 0.0
        if preemptions_total is not None:
            if self._last_preemptions is not None:
                rate = max(0.0, preemptions_total - self._last_preemptions)
            self._last_preemptions = preemptions_total

        kv = _first(metrics, KV_USAGE_METRICS)
        return EngineSnapshot(
            kv_usage=kv if kv is not None else 0.0,
            running=int(_first(metrics, RUNNING_METRICS) or 0),
            waiting=int(_first(metrics, WAITING_METRICS) or 0),
            preemptions_per_minute=rate,
        )

    def proxy(self, path: str, payload: dict, stream: bool) -> ProxyResponse:
        body = json.dumps(payload).encode("utf-8")
        conn = self._connect(self._timeout)
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json", "Accept": "*/*"},
        )
        response = conn.getresponse()
        headers = [
            (key, value)
            for key, value in response.getheaders()
            if key.lower() not in ("transfer-encoding", "connection", "content-length")
        ]
        return ProxyResponse(
            status=response.status,
            headers=headers,
            body=_drain(conn, response, stream),
        )


def _drain(
    conn: http.client.HTTPConnection,
    response: http.client.HTTPResponse,
    stream: bool,
) -> Iterator[bytes]:
    try:
        if stream:
            while chunk := response.read(8192):
                yield chunk
        else:
            yield response.read()
    finally:
        conn.close()
