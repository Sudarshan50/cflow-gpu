"""Client for the upstream engine: health scraping and request proxying.

Metric names are candidate lists. A name that does not resolve must degrade to
"no signal" rather than a false reading.
"""

from __future__ import annotations

import http.client
import json
import re
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
ITL_SUM_METRICS = (
    "vllm:inter_token_latency_seconds_sum",
)
ITL_COUNT_METRICS = (
    "vllm:inter_token_latency_seconds_count",
)
TTFT_SUM_METRICS = (
    "vllm:time_to_first_token_seconds_sum",
)
TTFT_COUNT_METRICS = (
    "vllm:time_to_first_token_seconds_count",
)
PREFILL_SUM_METRICS = (
    "vllm:request_prefill_time_seconds_sum",
)
PREFILL_COUNT_METRICS = (
    "vllm:request_prefill_time_seconds_count",
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
        self._refresh_lock = threading.Lock()
        self._last_preemptions: float | None = None
        self._last_preemption_at: float | None = None
        self._latency_counters: dict[
            str, tuple[float, float, float | None, float]
        ] = {}
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
        return self._snapshot_with_ttl(self._snapshot_ttl)

    def refresh_snapshot(self) -> EngineSnapshot:
        """Reconcile a stale gauge without a scrape per waiting request."""
        return self._snapshot_with_ttl(min(self._snapshot_ttl, .25))

    def _rolling_mean(
        self,
        name: str,
        metrics: dict[str, float],
        sum_metrics: tuple[str, ...],
        count_metrics: tuple[str, ...],
        now: float,
    ) -> float | None:
        total = _first(metrics, sum_metrics)
        count = _first(metrics, count_metrics)
        previous = self._latency_counters.get(name)
        mean = previous[2] if previous is not None else None
        if total is None or count is None:
            return (
                mean if previous is not None and now - previous[3] <= 10 else None
            )
        sampled_at = previous[3] if previous is not None else now
        if previous is not None and count > previous[1]:
            sample = max(0.0, total - previous[0]) / (count - previous[1])
            mean = sample if mean is None else .5 * sample + .5 * mean
            sampled_at = now
        elif previous is not None and now - sampled_at > 10:
            mean = None
        self._latency_counters[name] = (total, count, mean, sampled_at)
        return mean

    def _snapshot_with_ttl(self, ttl: float) -> EngineSnapshot:
        now = time.monotonic()
        with self._lock:
            if (
                self._cached_snapshot is not None
                and now - self._cached_at < ttl
            ):
                return self._cached_snapshot

        # Ordinary cache hits do not wait for a reconciliation scrape. Cache
        # misses are single-flight, including expiry bursts and queued retries.
        with self._refresh_lock:
            now = time.monotonic()
            with self._lock:
                if self._cached_snapshot is not None and now - self._cached_at < ttl:
                    return self._cached_snapshot
            return self._read_snapshot(now)

    def _read_snapshot(self, now: float) -> EngineSnapshot:
        conn = self._connect(HEALTH_TIMEOUT_SECONDS)
        try:
            conn.request("GET", "/metrics")
            response = conn.getresponse()
            if not 200 <= response.status < 300:
                raise OSError(f"engine metrics returned HTTP {response.status}")
            text = response.read().decode("utf-8", "replace")
            metrics = parse_prometheus(text)
            capacities = [int(value) for value in re.findall(r'kv_cache_size_tokens="(\d+)"', text)]
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

            mean_itl = self._rolling_mean(
                "itl", metrics, ITL_SUM_METRICS, ITL_COUNT_METRICS, now
            )
            mean_ttft = self._rolling_mean(
                "ttft", metrics, TTFT_SUM_METRICS, TTFT_COUNT_METRICS, now
            )
            mean_prefill = self._rolling_mean(
                "prefill", metrics, PREFILL_SUM_METRICS, PREFILL_COUNT_METRICS, now
            )

            kv = _first(metrics, KV_USAGE_METRICS)
            snapshot = EngineSnapshot(
                kv_usage=kv if kv is not None else 0.0,
                running=int(_first(metrics, RUNNING_METRICS) or 0),
                waiting=int(_first(metrics, WAITING_METRICS) or 0),
                preemptions_per_minute=rate,
                kv_capacity_tokens=min(capacities) if capacities else None,
                cache_epoch=metrics.get("process_start_time_seconds", metrics.get("vllm:request_success_created")),
                sampled_at=time.monotonic(),
                mean_itl_seconds=mean_itl,
                mean_ttft_seconds=mean_ttft,
                mean_prefill_seconds=mean_prefill,
            )
            self._cached_snapshot = snapshot
            self._cached_at = snapshot.sampled_at
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
