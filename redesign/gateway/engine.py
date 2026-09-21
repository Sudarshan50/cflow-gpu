"""Client for the upstream engine: health scraping and request proxying.

Metric names are candidate lists. A name that does not resolve must degrade to
"no signal" rather than a false reading.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator
from urllib.parse import urlparse

from .backpressure import EngineHealthSource
from .cancellation import (
    DEFAULT_TIMEOUT_SECONDS,
    RequestCancellation,
    RequestCancelled,
    UpstreamResources,
    current_cancellation,
)
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
            # Shutdown precedes closing any buffered reader. EngineClient's
            # body iterator also supports close concurrently with a blocked read.
            if self._close_upstream is not None:
                self._close_upstream()
        finally:
            close_body = getattr(self.body, "close", None)
            if close_body is not None:
                close_body()


_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+.*)?$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^"\\])*)"')


def _samples(text: str) -> Iterator[tuple[str, dict[str, str], float]]:
    for line in text.splitlines():
        match = _SAMPLE.fullmatch(line.strip())
        if match is None:
            continue
        name, labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        yield name, dict(_LABEL.findall(labels or "")), value


def parse_prometheus(text: str) -> dict[str, float]:
    """Flattens exposition text to name -> value, ignoring labels.

    Label-summing is deliberate: every metric read here is a single-instance
    gauge or counter, and a label dimension appearing later should not silently
    change the reading.
    """
    out: dict[str, float] = {}
    for name, _, value in _samples(text):
        out[name] = out.get(name, 0.0) + value
    return out


def _first(metrics: dict[str, float], candidates: tuple[str, ...]) -> float | None:
    for name in candidates:
        if name in metrics and math.isfinite(metrics[name]):
            return metrics[name]
    return None


def _positive_int(value: str | None) -> int | None:
    try:
        parsed = int(value)
    except (ValueError, TypeError):
        return None
    return parsed if parsed > 0 else None


def _kv_capacity(text: str) -> int | None:
    capacities = []
    for name, labels, value in _samples(text):
        if name != "vllm:cache_config_info" or value <= 0:
            continue
        # Current vLLM publishes the usable token capacity directly. It can
        # differ from num_gpu_blocks * block_size (e.g. hybrid/Mamba caches).
        capacity = _positive_int(labels.get("kv_cache_size_tokens"))
        if capacity is None:
            blocks = _positive_int(labels.get("num_gpu_blocks"))
            block_size = _positive_int(labels.get("block_size"))
            if blocks is not None and block_size is not None:
                capacity = blocks * block_size
        if capacity is not None:
            capacities.append(capacity)
    # This client targets one engine. Never sum info labels repeated per rank
    # or model and accidentally multiply the capacity of its shared KV pool.
    return min(capacities) if capacities else None


class EngineClient(EngineHealthSource):
    # The handler advertises unsupported legacy transports separately. Wrappers
    # may opt in only if they propagate current_cancellation to all their I/O.
    supports_request_cancellation = True

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        extra_headers: dict[str, str] | None = None,
        snapshot_ttl: float = 2.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("engine timeout must be finite and positive")
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

    def _connect(self, timeout: float) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def healthy(self) -> bool:
        response = None
        try:
            response = self._request("GET", "/health", timeout=HEALTH_TIMEOUT_SECONDS)
            return 200 <= response.status < 300
        except RequestCancelled:
            # A standalone health deadline means unhealthy. An enclosing
            # request's cancellation must still reach that request's handler.
            if current_cancellation() is not None:
                raise
            return False
        except (OSError, http.client.HTTPException):
            return False
        finally:
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass

    def metrics_text(self) -> str:
        response = self._request("GET", "/metrics", timeout=HEALTH_TIMEOUT_SECONDS)
        try:
            if not 200 <= response.status < 300:
                raise http.client.HTTPException(f"metrics HTTP {response.status}")
            return b"".join(response.body).decode("utf-8", "replace")
        finally:
            response.close()

    def invalidate_snapshot(self) -> None:
        with self._lock:
            self._cached_snapshot = None
            self._cached_at = 0.0

    def snapshot(self) -> EngineSnapshot:
        """Reads engine health. Raises on failure so the breaker fails open.

        Cached briefly so the hot path does not scrape /metrics on every
        request. The preemption figure is a true per-minute rate, not a
        per-request delta. Missing/unusable health signals, including an
        unestablished preemption rate, make known=False; numeric placeholders
        in that snapshot are not measured zeroes. KV capacity is optional.
        """
        now = time.monotonic()
        with self._lock:
            if (
                self._cached_snapshot is not None
                and now - self._cached_at < self._snapshot_ttl
            ):
                return self._cached_snapshot

        text = self.metrics_text()
        metrics = parse_prometheus(text)

        preemptions_total = _first(metrics, PREEMPTION_METRICS)
        rate = 0.0
        rate_known = False
        with self._lock:
            if preemptions_total is not None and preemptions_total >= 0:
                # A zero counter establishes zero preemptions. A first positive
                # counter (or a reset to a positive count) needs a second sample
                # before it can establish a rate suitable for borrowing.
                rate_known = preemptions_total == 0
                if (
                    self._last_preemptions is not None
                    and self._last_preemption_at is not None
                    and now > self._last_preemption_at
                    and preemptions_total >= self._last_preemptions
                ):
                    elapsed_min = max(1e-6, (now - self._last_preemption_at) / 60.0)
                    rate = (preemptions_total - self._last_preemptions) / elapsed_min
                    rate_known = True
                self._last_preemptions = preemptions_total
                self._last_preemption_at = now
            else:
                self._last_preemptions = None
                self._last_preemption_at = None

            kv = _first(metrics, KV_USAGE_METRICS)
            running = _first(metrics, RUNNING_METRICS)
            waiting = _first(metrics, WAITING_METRICS)
            known = (
                kv is not None and 0 <= kv <= 1
                and running is not None and running >= 0
                and waiting is not None and waiting >= 0
                and rate_known
                and time.monotonic() - now <= max(2.0, self._snapshot_ttl)
            )
            snapshot = EngineSnapshot(
                kv_usage=kv if kv is not None else 0.0,
                running=int(running) if running is not None else 0,
                waiting=int(waiting) if waiting is not None else 0,
                preemptions_per_minute=rate,
                known=known,
                kv_capacity_tokens=_kv_capacity(text),
            )
            self._cached_snapshot = snapshot
            self._cached_at = now
        return snapshot

    def proxy(self, path: str, payload: dict, stream: bool) -> ProxyResponse:
        """Relay under the request context, or a standalone total deadline.

        The public signature intentionally remains compatible with legacy fake
        engines and OffBoxClient. Cancellation is explicit through the context
        interface in cancellation.py, not a speculative keyword/retry.
        """
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            **self._extra_headers,
        }
        return self._request(
            "POST", path, body, headers, stream, self._timeout, limit_deadline=True
        )

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        stream: bool = False,
        timeout: float = HEALTH_TIMEOUT_SECONDS,
        limit_deadline: bool = False,
    ) -> ProxyResponse:
        cancellation = current_cancellation()
        owns_cancellation = cancellation is None
        if cancellation is None:
            cancellation = RequestCancellation(timeout=timeout).start()
        elif limit_deadline:
            cancellation.limit_timeout(timeout)
        resources = None
        try:
            cancellation.check()
            conn = self._connect(min(timeout, cancellation.remaining))
            resources = UpstreamResources(conn, cancellation, timeout, owns_cancellation)
            with resources.io_lock:
                resources.check()
                conn.request(method, path, body=body, headers=headers or {})
                resources.check()
                response = conn.getresponse()
                resources.response = response
                resources.check()
                forwarded_headers = [
                    (key, value)
                    for key, value in response.getheaders()
                    if key.lower() not in HOP_BY_HOP
                ]
            return ProxyResponse(
                status=response.status,
                headers=forwarded_headers,
                body=_ResponseBody(resources, stream),
                _close_upstream=resources.close,
            )
        except BaseException:
            try:
                cancellation.check()
            finally:
                if resources is not None:
                    resources.close()
                elif owns_cancellation:
                    cancellation.finish()
            raise


class _ResponseBody(Iterator[bytes]):
    """A single-consumer iterator whose close can interrupt another thread.

    `read(n)` blocks until it has all n bytes, which on a token stream means
    withholding the first byte until ~8 KB of SSE has accumulated -- seconds of
    invented TTFT. `read1(n)` returns as soon as any data is available.
    """

    def __init__(self, resources: UpstreamResources, stream: bool) -> None:
        self._resources = resources
        self._stream = stream
        self._done = False

    def __next__(self) -> bytes:
        resources = self._resources
        try:
            with resources.io_lock:
                resources.cancellation.check()
                if self._done or resources.closed:
                    raise StopIteration
                resources.check()
                response = resources.response
                chunk = response.read1(8192) if self._stream else response.read()
                resources.check()
                if not self._stream:
                    self._done = True
                elif not chunk:
                    # read1, unlike read(), can silently return EOF before a
                    # declared Content-Length. Do not frame that as completed.
                    remaining = getattr(response, "length", None)
                    if isinstance(remaining, int) and remaining > 0:
                        raise http.client.IncompleteRead(b"", remaining)
                    raise StopIteration
                return chunk
        except BaseException:
            try:
                resources.cancellation.check()
            finally:
                resources.close()
            raise

    def close(self) -> None:
        self._resources.close()
