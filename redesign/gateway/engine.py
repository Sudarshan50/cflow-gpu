"""Client for the upstream engine: health scraping and request proxying.

Metric names are candidate lists. A name that does not resolve must degrade to
"no signal" rather than a false reading.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator
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
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)
PREEMPTION_METRICS = (
    "sglang:num_preemptions_total",
    "vllm:num_preemptions_total",
)

# Headers the proxy generates itself or that are connection-scoped. Echoing
# these produces duplicate Date/Server, which RFC 9110 forbids.
HOP_BY_HOP = frozenset({
    "transfer-encoding", "connection", "content-length", "date", "server",
    "keep-alive", "upgrade", "te", "trailer", "proxy-authenticate",
    "proxy-authorization",
})

DEFAULT_TIMEOUT_SECONDS = 600
HEALTH_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class ProxyResponse:
    status: int
    headers: list[tuple[str, str]]
    body: Iterator[bytes]
    _close_upstream: Callable[[], None] | None = field(default=None, repr=False)

    def close(self) -> None:
        """Abandons the upstream response, releasing its connection."""
        try:
            self.body.close()
        finally:
            # Closing an unstarted generator does not execute its finally.
            if self._close_upstream is not None:
                self._close_upstream()


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
    def __init__(
        self,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        extra_headers: dict[str, str] | None = None,
        snapshot_ttl: float = 2.0,
    ) -> None:
        parsed = urlparse(base_url)
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._timeout = timeout
        self._extra_headers = extra_headers or {}
        self._snapshot_ttl = snapshot_ttl
        self._lock = threading.Lock()
        self._last_preemptions: float | None = None
        self._last_preemption_at: float | None = None
        self._cached_snapshot: EngineSnapshot | None = None
        self._cached_at: float = 0.0

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

    def metrics_text(self) -> str:
        conn = self._connect(HEALTH_TIMEOUT_SECONDS)
        try:
            conn.request("GET", "/metrics")
            return conn.getresponse().read().decode("utf-8", "replace")
        finally:
            conn.close()

    def invalidate_snapshot(self) -> None:
        with self._lock:
            self._cached_snapshot = None
            self._cached_at = 0.0

    def snapshot(self) -> EngineSnapshot:
        """Reads engine health. Raises on failure so the breaker fails open.

        Cached briefly so the hot path does not scrape /metrics on every
        request. The preemption figure is a true per-minute rate, not a
        per-request delta.
        """
        now = time.monotonic()
        with self._lock:
            if (
                self._cached_snapshot is not None
                and now - self._cached_at < self._snapshot_ttl
            ):
                return self._cached_snapshot

        conn = self._connect(HEALTH_TIMEOUT_SECONDS)
        try:
            conn.request("GET", "/metrics")
            response = conn.getresponse()
            metrics = parse_prometheus(response.read().decode("utf-8", "replace"))
        finally:
            conn.close()

        preemptions_total = _first(metrics, PREEMPTION_METRICS)
        rate = 0.0
        with self._lock:
            if preemptions_total is not None:
                if self._last_preemptions is not None and self._last_preemption_at:
                    elapsed_min = max(1e-6, (now - self._last_preemption_at) / 60.0)
                    rate = max(0.0, preemptions_total - self._last_preemptions) / elapsed_min
                self._last_preemptions = preemptions_total
                self._last_preemption_at = now

            kv = _first(metrics, KV_USAGE_METRICS)
            snapshot = EngineSnapshot(
                kv_usage=kv if kv is not None else 0.0,
                running=int(_first(metrics, RUNNING_METRICS) or 0),
                waiting=int(_first(metrics, WAITING_METRICS) or 0),
                preemptions_per_minute=rate,
            )
            self._cached_snapshot = snapshot
            self._cached_at = now
        return snapshot

    def proxy(self, path: str, payload: dict, stream: bool) -> ProxyResponse:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            **self._extra_headers,
        }
        conn = self._connect(self._timeout)
        response = None
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                # HTTPConnection may have relinquished a Connection: close
                # response's socket. The response must also be closed explicitly.
                if response is not None:
                    response.close()
            finally:
                conn.close()

        try:
            conn.request("POST", path, body=body, headers=headers)
            response = conn.getresponse()
            headers = [
                (key, value)
                for key, value in response.getheaders()
                if key.lower() not in HOP_BY_HOP
            ]
            return ProxyResponse(
                status=response.status,
                headers=headers,
                body=_drain(response, stream, close),
                _close_upstream=close,
            )
        except BaseException:
            close()
            raise


def _drain(
    response: http.client.HTTPResponse,
    stream: bool,
    close: Callable[[], None],
) -> Iterator[bytes]:
    """Yields the response body.

    `read(n)` blocks until it has all n bytes, which on a token stream means
    withholding the first byte until ~8 KB of SSE has accumulated -- seconds of
    invented TTFT. `read1(n)` returns as soon as any data is available.
    """
    try:
        if stream:
            while chunk := response.read1(8192):
                yield chunk
        else:
            yield response.read()
    finally:
        close()
