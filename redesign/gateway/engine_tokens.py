"""Bounded, cancellation-aware K3 prompt counting through vLLM /tokenize.

``max_pending`` bounds ALL distinct outstanding jobs (running plus queued).
Duplicate callers share a job without taking another slot. Fixed executor
workers drain a removable queue: cancelling queued Futures cannot accumulate
tombstones in ThreadPoolExecutor's otherwise unbounded private work queue.

Each job has its own deadline, including queue time, independent of every
waiter's request context. A timeout/unavailable count is an explicit 503-level
failure, never a fallback to the incomplete text/image heuristic.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Literal, NoReturn
from urllib.parse import urlparse

from .cancellation import CancelReason, RequestCancellation, RequestCancelled, current_cancellation
from .engine import EngineClient
from .prompt_protocol import PROTOCOL_REVISION, TokenCountInputError, build_tokenize_request


MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_WAIT_INTERVAL = 0.025


class TokenCountUnavailable(RuntimeError):
    """A transient/overload/count-deadline failure; return HTTP 503 to callers."""

    status_code = 503

    def __init__(
        self, message: str = "engine token count is unavailable", *,
        reason: str = "unavailable", upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.upstream_status = upstream_status


@dataclass(frozen=True)
class TokenCountResult:
    count: int
    max_model_len: int
    source: Literal["engine_rendered"] = field(default="engine_rendered", init=False)
    cache_hit: bool = False


@dataclass(frozen=True)
class _Failure:
    """Cross-thread errors without exceptions/tracebacks retaining prompt data."""

    kind: Literal["input", "unavailable"]
    message: str
    reason: str
    param: str | None = None
    upstream_status: int | None = None

    def raise_error(self) -> NoReturn:
        if self.kind == "input":
            raise TokenCountInputError(
                self.message, param=self.param, reason=self.reason,
                upstream_status=self.upstream_status,
            )
        raise TokenCountUnavailable(
            self.message, reason=self.reason, upstream_status=self.upstream_status,
        )


@dataclass(frozen=True)
class _Outcome:
    result: TokenCountResult | None = None
    failure: _Failure | None = None


@dataclass(eq=False)
class _Job:
    key: bytes
    body: bytes | None
    cancellation: RequestCancellation
    future: Future[_Outcome] = field(default_factory=Future)
    waiters: int = 1
    running: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class _CacheEntry:
    result: TokenCountResult
    expires: float


def _stop_failure(reason: str) -> _Failure:
    return _Failure("unavailable", {
        "timeout": "engine token count deadline exceeded",
        "closed": "engine token estimator is closed",
        "abandoned": "engine token count has no remaining waiters",
    }[reason], reason)


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")


def _seconds(name: str, value: float, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")


class EngineTokenEstimator:
    """Count the normalized engine-effective prompt, including expanded images.

    Use normalize_engine_prompt(payload, path) for the inference body too; the
    estimator also applies it defensively to its counting copy. ``cache_hit``
    means a completed LRU hit, not a singleflight join. ``revision`` partitions
    counts for a checkpoint/renderer/media-policy profile; recreate this object
    when that profile changes. No network call is made during construction.

    snapshot() returns only numeric counters/gauges. requests/successes/errors
    are per caller; leaders/jobs_* are per distinct job. request_bytes_total
    counts attempted POST bodies, response_bytes_total counts received chunks.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 2.0,
        max_workers: int = 2,
        max_pending: int = 8,
        cache_entries: int = 2048,
        cache_ttl: float = 60.0,
        revision: str | None = None,
    ) -> None:
        _seconds("timeout", timeout)
        _seconds("cache_ttl", cache_ttl, allow_zero=True)
        _positive_int("max_workers", max_workers)
        _positive_int("max_pending", max_pending)
        _positive_int("cache_entries", cache_entries, allow_zero=True)
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http" or not parsed.hostname
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.username is not None or parsed.password is not None
        ):
            raise ValueError("base_url must be an HTTP engine origin without credentials or a path")
        if revision is not None and not isinstance(revision, str):
            raise ValueError("revision must be a string or None")

        self._timeout = float(timeout)
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._cache_entries = cache_entries
        self._cache_ttl = float(cache_ttl)
        self._revision = hashlib.sha256((PROTOCOL_REVISION + "\0" + (revision or "")).encode()).digest()
        self._engine = EngineClient(base_url, timeout=timeout)
        self._condition = threading.Condition(threading.Lock())
        self._close_lock = threading.Lock()
        self._jobs: dict[bytes, _Job] = {}
        self._pending: set[_Job] = set()
        self._queue: deque[_Job] = deque()
        self._cache: OrderedDict[bytes, _CacheEntry] = OrderedDict()
        self._running = 0
        self._waiters = 0
        self._calls = 0
        self._closed = False
        self._stats: dict[str, int | float] = dict.fromkeys((
            "requests_total", "successes_total", "input_errors_total", "unavailable_total",
            "waiter_cancellations_total", "overloaded_total", "count_timeouts_total",
            "cache_hits_total", "cache_misses_total", "cache_evictions_total", "cache_expirations_total",
            "leaders_total", "singleflight_joins_total", "jobs_started_total", "jobs_finished_total",
            "jobs_succeeded_total", "jobs_input_errors_total", "jobs_unavailable_total",
            "jobs_timed_out_total", "jobs_aborted_total", "jobs_cancelled_queued_total",
            "request_bytes_total", "response_bytes_total", "request_seconds_sum", "last_max_model_len",
        ), 0)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"engine-token-{id(self):x}",
        )
        try:
            self._workers = [self._executor.submit(self._worker) for _ in range(max_workers)]
        except Exception:
            self.close()
            raise

    def __enter__(self) -> EngineTokenEstimator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def estimate(self, payload: dict) -> int:
        return self.count(payload).count

    def count(self, payload: dict) -> TokenCountResult:
        parent = current_cancellation()
        started = time.monotonic()
        job = None
        projection = body = None
        with self._condition:
            self._stats["requests_total"] += 1
            self._calls += 1
        try:
            if parent is not None:
                parent.check()
            with self._condition:
                if self._closed:
                    _stop_failure("closed").raise_error()
            projection = build_tokenize_request(payload)
            body = self._encode_request(projection, parent)
            digest = hashlib.sha256(self._revision)
            digest.update(body)
            key = digest.digest()
            if parent is not None:
                parent.check()
            with self._condition:
                if self._closed:
                    _stop_failure("closed").raise_error()
                result = self._cached_locked(key)
                if result is None:
                    self._stats["cache_misses_total"] += 1
                    job = self._jobs.get(key)
                    if job is None:
                        if len(self._pending) >= self._max_pending:
                            raise TokenCountUnavailable("engine token count queue is full", reason="overloaded")
                        job = _Job(key, body, RequestCancellation(timeout=self._timeout))
                        self._jobs[key] = job
                        self._pending.add(job)
                        self._queue.append(job)
                        self._stats["leaders_total"] += 1
                        self._condition.notify()
                    else:
                        job.waiters += 1
                        self._stats["singleflight_joins_total"] += 1
                    self._waiters += 1
            # The queue owns the sole retained counting body. Duplicate waiters
            # need only the digest/job, not a second copy of a large prompt.
            payload = projection = body = None
            if result is None:
                result = self._wait(job, parent)
            if parent is not None:
                parent.check()
            with self._condition:
                self._stats["successes_total"] += 1
            return result
        except TokenCountInputError:
            with self._condition:
                self._stats["input_errors_total"] += 1
            raise
        except TokenCountUnavailable as exc:
            with self._condition:
                self._stats["unavailable_total"] += 1
                if exc.reason == "overloaded":
                    self._stats["overloaded_total"] += 1
                if exc.reason == "timeout":
                    self._stats["count_timeouts_total"] += 1
            raise
        except RequestCancelled:
            with self._condition:
                self._stats["waiter_cancellations_total"] += 1
            raise
        finally:
            payload = projection = body = None
            if job is not None:
                self._detach(job)
            with self._condition:
                self._calls -= 1
                self._stats["request_seconds_sum"] += time.monotonic() - started

    @staticmethod
    def _encode_request(projection: dict, parent: RequestCancellation | None) -> bytes:
        encoded = bytearray()
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        try:
            # No sort_keys: tool arguments and other nested object insertion
            # order can affect rendering. Arrays always retain their order.
            for chunk in encoder.iterencode(projection):
                if parent is not None:
                    parent.check()
                if len(chunk) > MAX_REQUEST_BYTES:
                    raise TokenCountInputError("tokenizer request exceeds 64 MiB", reason="request_too_large")
                data = chunk.encode("utf-8")
                if len(encoded) + len(data) > MAX_REQUEST_BYTES:
                    raise TokenCountInputError("tokenizer request exceeds 64 MiB", reason="request_too_large")
                encoded.extend(data)
        except TokenCountInputError:
            raise
        except (TypeError, ValueError, RecursionError, UnicodeError):
            raise TokenCountInputError("prompt must be finite JSON data", param="body") from None
        return bytes(encoded)

    def _cached_locked(self, key: bytes) -> TokenCountResult | None:
        cached = self._cache.get(key)
        if cached is None:
            return None
        if cached.expires <= time.monotonic():
            del self._cache[key]
            self._stats["cache_expirations_total"] += 1
            return None
        self._cache.move_to_end(key)
        self._stats["cache_hits_total"] += 1
        return replace(cached.result, cache_hit=True)

    def _wait(self, job: _Job, parent: RequestCancellation | None) -> TokenCountResult:
        while True:
            if parent is not None:
                parent.check()
            if not job.future.done() and job.cancellation.remaining <= 0:
                self._stop(job, "timeout")
            try:
                outcome = job.future.result(timeout=_WAIT_INTERVAL)
            except TimeoutError:
                continue
            except CancelledError:
                _stop_failure(job.stop_reason or "abandoned").raise_error()
            if parent is not None:
                parent.check()
            if outcome.failure is not None:
                outcome.failure.raise_error()
            if outcome.result is None:
                raise TokenCountUnavailable("engine token counting failed", reason="internal_error")
            return outcome.result

    def _detach(self, job: _Job) -> None:
        cancellation = None
        with self._condition:
            job.waiters -= 1
            self._waiters -= 1
            if job.waiters == 0 and job in self._pending and job.stop_reason is None:
                cancellation = self._stop_locked(job, "abandoned")
        if cancellation is not None:
            cancellation.cancel(CancelReason.CLIENT_DISCONNECT)

    def _stop(self, job: _Job, reason: str) -> None:
        with self._condition:
            cancellation = self._stop_locked(job, reason)
        if cancellation is not None:
            cancellation.cancel(CancelReason.DEADLINE if reason == "timeout" else CancelReason.CLIENT_DISCONNECT)

    def _stop_locked(self, job: _Job, reason: str) -> RequestCancellation | None:
        if job not in self._pending or job.stop_reason is not None:
            return None
        job.stop_reason = reason
        job.body = None
        if self._jobs.get(job.key) is job:
            del self._jobs[job.key]
        failure = _stop_failure(reason)
        if not job.running:
            self._queue.remove(job)
            self._pending.remove(job)
            self._stats["jobs_cancelled_queued_total"] += 1
            self._record_job_locked(None, failure)
            if reason == "abandoned":
                job.future.cancel()
        if not job.future.done():
            job.future.set_result(_Outcome(failure=failure))
        # Running jobs keep their slot until actual transport/worker cleanup.
        return job.cancellation

    def _worker(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._queue)
                if self._closed:
                    return
                job = self._queue.popleft()
                job.future.set_running_or_notify_cancel()
                job.running = True
                self._running += 1
                self._stats["jobs_started_total"] += 1
            self._run(job)

    def _run(self, job: _Job) -> None:
        result = failure = None
        try:
            # ContextVars are deliberately NOT copied from any waiter. All
            # EngineClient I/O sees this job-local deadline, never parent 600s.
            with job.cancellation:
                job.cancellation.check()
                result = self._fetch_count(job.body)
                job.cancellation.check()
        except RequestCancelled as exc:
            failure = _stop_failure("timeout" if exc.reason is CancelReason.DEADLINE else "abandoned")
        except TokenCountInputError as exc:
            failure = _Failure("input", str(exc), exc.reason, exc.param, exc.upstream_status)
        except TokenCountUnavailable as exc:
            failure = _Failure("unavailable", str(exc), exc.reason, upstream_status=exc.upstream_status)
        except Exception:
            # Do not put an exception traceback (which can retain a prompt or
            # returned token IDs) into a Future or long-lived estimator state.
            failure = _Failure("unavailable", "engine token counting failed", "internal_error")
        finally:
            job.body = None
            with self._condition:
                if job.stop_reason is not None:
                    result, failure = None, _stop_failure(job.stop_reason)
                elif failure is not None:
                    result = None
                elif result is None:
                    failure = _Failure("unavailable", "engine token counting was interrupted", "internal_error")
                self._running -= 1
                self._pending.remove(job)
                if self._jobs.get(job.key) is job:
                    del self._jobs[job.key]
                if result is not None and not self._closed and job.waiters:
                    self._put_cache_locked(job.key, result)
                self._record_job_locked(result, failure)
                if not job.future.done():
                    job.future.set_result(_Outcome(result=result, failure=failure))

    def _record_job_locked(self, result: TokenCountResult | None, failure: _Failure | None) -> None:
        self._stats["jobs_finished_total"] += 1
        if result is not None:
            self._stats["jobs_succeeded_total"] += 1
            self._stats["last_max_model_len"] = result.max_model_len
        elif failure is not None:
            name = "jobs_input_errors_total" if failure.kind == "input" else "jobs_unavailable_total"
            self._stats[name] += 1
            if failure.reason == "timeout":
                self._stats["jobs_timed_out_total"] += 1
            elif failure.reason in ("closed", "abandoned"):
                self._stats["jobs_aborted_total"] += 1

    def _put_cache_locked(self, key: bytes, result: TokenCountResult) -> None:
        if not self._cache_entries or not self._cache_ttl:
            return
        self._cache[key] = _CacheEntry(result, time.monotonic() + self._cache_ttl)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_entries:
            self._cache.popitem(last=False)
            self._stats["cache_evictions_total"] += 1

    def _fetch_count(self, body: bytes) -> TokenCountResult:
        response = None
        try:
            with self._condition:
                self._stats["request_bytes_total"] += len(body)
            # Streaming here only selects EngineClient's bounded read1(8192)
            # transport. /tokenize itself returns one ordinary JSON document.
            response = self._engine._request(
                "POST", "/tokenize", body,
                {"Content-Type": "application/json", "Accept": "application/json"},
                stream=True, timeout=self._timeout,
            )
            if response.status in (400, 404, 413, 422):
                raise TokenCountInputError(
                    "tokenizer rejected the prompt", reason="http_4xx", upstream_status=response.status,
                )
            if response.status != 200:
                raise TokenCountUnavailable(
                    "tokenizer returned an unsuccessful status", reason="http_status", upstream_status=response.status,
                )
            raw = bytearray()
            for chunk in response.body:
                with self._condition:
                    self._stats["response_bytes_total"] += len(chunk)
                if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise TokenCountUnavailable("tokenizer response exceeds 8 MiB", reason="response_too_large")
                raw.extend(chunk)
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError):
                raise TokenCountUnavailable("tokenizer returned invalid JSON", reason="invalid_response") from None
            if not isinstance(result, dict):
                raise TokenCountUnavailable("tokenizer returned an invalid count", reason="invalid_response")
            count, tokens, maximum = result.get("count"), result.get("tokens"), result.get("max_model_len")
            if (
                type(count) is not int or count < 0
                or not isinstance(tokens, list) or count != len(tokens)
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(maximum) is not int or maximum <= 0
            ):
                raise TokenCountUnavailable("tokenizer returned an invalid count", reason="invalid_response")
            # count may exceed max_model_len: the gateway needs the FULL input
            # count in order to reject/clamp appropriately, not a truncated one.
            return TokenCountResult(count=count, max_model_len=maximum)
        except TimeoutError:
            raise TokenCountUnavailable("engine token count deadline exceeded", reason="timeout") from None
        except (OSError, http.client.HTTPException):
            raise TokenCountUnavailable("tokenizer transport failed", reason="transport_error") from None
        finally:
            if response is not None:
                response.close()

    def snapshot(self) -> dict[str, int | float]:
        """Detached aggregate metrics only: no keys, URLs, prompts or tokens."""
        with self._condition:
            now = time.monotonic()
            for key, entry in list(self._cache.items()):
                if entry.expires <= now:
                    del self._cache[key]
                    self._stats["cache_expirations_total"] += 1
            return {
                **self._stats,
                "calls_in_progress": self._calls,
                "pending": len(self._pending),
                "running": self._running,
                "queued": len(self._queue),
                "waiters": self._waiters,
                "cache_entries": len(self._cache),
                "max_workers": self._max_workers,
                "max_pending": self._max_pending,
                "closed": int(self._closed),
            }

    def close(self) -> None:
        """Abort shared work and join all owned workers/deadline watchers."""
        with self._close_lock:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                cancellations = [self._stop_locked(job, "closed") for job in list(self._pending)]
                self._cache.clear()
                self._condition.notify_all()
            for cancellation in cancellations:
                if cancellation is not None:
                    cancellation.cancel(CancelReason.CLIENT_DISCONNECT)
            self._executor.shutdown(wait=True, cancel_futures=True)
