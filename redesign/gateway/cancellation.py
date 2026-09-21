"""Request-scoped cancellation for the synchronous HTTP gateway.

The context is an opt-in interface: transports read ``current_cancellation()``
and register a non-blocking abort callback. Merely running a legacy ``proxy``
inside the context does NOT make it interruptible. No proxy call is retried to
discover whether it accepts a new keyword argument.

Start downstream monitoring only after the request body has been read. On Linux
POLLRDHUP detects a FIN even behind unread pipelined data, without consuming any
bytes or contending with BaseHTTPRequestHandler's buffered reader. A TCP FIN
cannot distinguish abandonment from shutdown(SHUT_WR) by a client still waiting
for its response: this gateway treats BOTH as cancellation during a request.
Clients must leave their write half open until the response finishes. HTTP's
``Connection: close`` header alone is fine. The portable peek fallback never
consumes data, but unread pipelined bytes can hide a FIN until the deadline.

Cancellation shuts down the downstream socket too, including on a deadline.
This bounds blocked writes; a deadline therefore closes/truncates the response
rather than promising an HTTP error response. The timeout is wall-clock time
after body receipt, not an idle timeout renewed by each token/read.
"""

from __future__ import annotations

import contextvars
import errno
import http.client
import math
import os
import select
import socket
import threading
import time
from collections.abc import Callable
from enum import Enum

DEFAULT_TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 0.05


class CancelReason(str, Enum):
    CLIENT_DISCONNECT = "client_cancel"
    DEADLINE = "deadline"


class RequestCancelled(Exception):
    """Separate from OSError so cancellation cannot trigger upstream fallback."""

    def __init__(self, reason: CancelReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


_CURRENT: contextvars.ContextVar[RequestCancellation | None] = contextvars.ContextVar(
    "gateway_cancellation", default=None
)


def current_cancellation() -> RequestCancellation | None:
    return _CURRENT.get()


def shutdown_socket(sock: socket.socket) -> None:
    """Wake blocked I/O before anybody tries to close a buffered reader."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        # Already closed, reset, or not connected yet. Socket publication and
        # post-connect checks below handle cancellation before connect finishes.
        pass


class RequestCancellation:
    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        downstream: socket.socket | None = None,
        started: float | None = None,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        self.started = time.monotonic() if started is None else started
        self._deadline = self.started + timeout
        self._downstream = downstream
        self._lock = threading.Lock()
        self._reason: CancelReason | None = None
        self._finished = False
        self._callbacks: dict[object, Callable[[], None]] = {}
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._context_token = None

    @property
    def reason(self) -> CancelReason | None:
        with self._lock:
            return self._reason

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self._deadline - time.monotonic())

    def limit_timeout(self, timeout: float) -> None:
        """A transport may shorten, never restart, the request's total budget."""
        with self._lock:
            self._deadline = min(self._deadline, self.started + timeout)
        self._wake.set()

    def check(self) -> None:
        with self._lock:
            expired = not self._finished and time.monotonic() >= self._deadline
        if expired:
            self.cancel(CancelReason.DEADLINE)
        reason = self.reason
        if reason is not None:
            raise RequestCancelled(reason)

    def on_cancel(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register before acquiring a resource; late registration aborts now.

        Callbacks must not wait for an I/O lock or close an active buffered
        reader. They run outside the guard lock, at most once per registration.
        """
        key = object()
        with self._lock:
            abort_now = self._reason is not None or self._finished
            if not abort_now:
                self._callbacks[key] = callback
        if abort_now:
            callback()

        def unregister() -> None:
            with self._lock:
                self._callbacks.pop(key, None)

        return unregister

    def cancel(self, reason: CancelReason) -> None:
        with self._lock:
            if self._reason is not None or self._finished:
                return
            self._reason = reason
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        self._wake.set()
        if self._downstream is not None:
            shutdown_socket(self._downstream)
        for callback in callbacks:
            try:
                callback()
            except OSError:
                # Abort/close still runs all other owners if one socket has
                # already failed. Normal worker cleanup remains idempotent.
                pass

    def start(self) -> RequestCancellation:
        with self._lock:
            if self._thread is not None or self._finished:
                return self
            self._thread = threading.Thread(
                target=self._watch, name="gateway-cancellation", daemon=True
            )
            self._thread.start()
        return self

    def finish(self) -> CancelReason | None:
        """Disarm and join before HTTP keep-alive can start the next request."""
        with self._lock:
            self._finished = True
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        return self.reason

    def __enter__(self) -> RequestCancellation:
        self._context_token = _CURRENT.set(self)
        try:
            return self.start()
        except BaseException:
            _CURRENT.reset(self._context_token)
            raise

    def __exit__(self, *exc) -> None:
        try:
            self.finish()
        finally:
            _CURRENT.reset(self._context_token)

    def _watch(self) -> None:
        poller = None
        if self._downstream is not None and hasattr(select, "POLLRDHUP"):
            poller = select.poll()
            # Deliberately omit POLLIN: pipelined requests are not cancellation
            # and must not turn this watcher into a busy loop.
            try:
                poller.register(
                    self._downstream,
                    select.POLLRDHUP | select.POLLHUP | select.POLLERR | select.POLLNVAL,
                )
            except (OSError, ValueError):
                self.cancel(CancelReason.CLIENT_DISCONNECT)
                return
        while True:
            with self._lock:
                if self._finished or self._reason is not None:
                    return
                remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                self.cancel(CancelReason.DEADLINE)
                return
            if self._disconnected(poller):
                self.cancel(CancelReason.CLIENT_DISCONNECT)
                return
            # Always wait, including when a portable MSG_PEEK sees queued
            # pipeline bytes. Readability alone must never cause a spin.
            self._wake.wait(min(POLL_INTERVAL_SECONDS, remaining))
            self._wake.clear()

    def _disconnected(self, poller) -> bool:
        if self._downstream is None:
            return False
        try:
            if poller is not None:
                return bool(poller.poll(0))
            readable, _, errors = select.select(
                [self._downstream], [], [self._downstream], 0
            )
            if errors:
                return True
            if readable:
                flags = socket.MSG_PEEK | getattr(socket, "MSG_DONTWAIT", 0)
                return self._downstream.recv(1, flags) == b""
        except (BlockingIOError, InterruptedError):
            pass
        except (OSError, ValueError):
            return True
        return False


# getaddrinfo cannot be interrupted by Python. Bound both the caller's wait
# and the number of abandoned OS lookups; resolver threads never open sockets.
_DNS_SLOTS = threading.BoundedSemaphore(4)


class UpstreamResources:
    """One HTTP connection, its response, and the actual sockets it created.

    The raw socket is retained even when HTTPConnection relinquishes it for a
    Connection: close response. Publication is locked and rejects late sockets.
    ``io_lock`` serializes HTTP/buffered-reader access against close; shutdown
    ALWAYS precedes waiting for that lock. Cancellation itself never waits on
    the lock: an active reader performs final cleanup after shutdown wakes it.
    """

    def __init__(
        self,
        connection: http.client.HTTPConnection,
        cancellation: RequestCancellation,
        timeout: float,
        owns_cancellation: bool = False,
    ) -> None:
        self.connection = connection
        self.response: http.client.HTTPResponse | None = None
        self.cancellation = cancellation
        self.io_lock = threading.RLock()
        self._lock = threading.Lock()
        self._sockets: set[socket.socket] = set()
        self._closed = False
        self._aborted = False
        self._timeout = timeout
        self._owns_cancellation = owns_cancellation
        self._unregister: Callable[[], None] = lambda: None
        # HTTPConnection calls this factory *before* assigning conn.sock. It
        # publishes sockets before connect, closing the close-before-connect
        # hole in simply registering conn.close as a cancellation callback.
        connection._create_connection = self._create_connection
        existing = getattr(connection, "sock", None)
        if isinstance(existing, socket.socket):
            self._sockets.add(existing)
        self._unregister = cancellation.on_cancel(self.abort)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def check(self) -> None:
        self.cancellation.check()
        with self._lock:
            if self._closed or self._aborted:
                raise OSError("upstream connection closed")

    def publish_socket(self, sock: socket.socket) -> None:
        with self._lock:
            reject = self._closed or self._aborted
            if not reject:
                self._sockets.add(sock)
        if reject:
            shutdown_socket(sock)
            sock.close()
            self.check()
        self.check()

    def abort(self) -> None:
        with self._lock:
            self._aborted = True
            sockets = tuple(self._sockets)
        for sock in sockets:
            shutdown_socket(sock)
        # An unstarted/paused response has no worker blocked in I/O to clean it
        # up. Close it here only when doing so cannot wait on a buffered reader.
        if self.io_lock.acquire(blocking=False):
            try:
                self.close()
            finally:
                self.io_lock.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sockets = tuple(self._sockets)
        for sock in sockets:
            shutdown_socket(sock)
        try:
            with self.io_lock:
                try:
                    if self.response is not None:
                        self.response.close()
                finally:
                    self.connection.close()
        finally:
            for sock in sockets:
                sock.close()
            self._unregister()
            if self._owns_cancellation:
                self.cancellation.finish()

    def _wait_budget(self, deadline: float) -> float:
        self.check()
        remaining = min(deadline - time.monotonic(), self.cancellation.remaining)
        if remaining <= 0:
            self.check()
            raise TimeoutError("upstream connection timed out")
        return min(POLL_INTERVAL_SECONDS, remaining)

    def _resolve(self, host: str, port: int, deadline: float):
        for family in (socket.AF_INET, socket.AF_INET6):
            try:
                socket.inet_pton(family, host)
            except OSError:
                continue
            address = (host, port) if family == socket.AF_INET else (host, port, 0, 0)
            return [(family, socket.SOCK_STREAM, 0, "", address)]

        while not _DNS_SLOTS.acquire(timeout=self._wait_budget(deadline)):
            pass
        done = threading.Event()
        result: list = []

        def resolve() -> None:
            try:
                result.append(socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM))
            except Exception as exc:
                result.append(exc)
            finally:
                _DNS_SLOTS.release()
                done.set()

        try:
            threading.Thread(target=resolve, name="gateway-dns", daemon=True).start()
        except BaseException:
            _DNS_SLOTS.release()
            raise
        while not done.wait(self._wait_budget(deadline)):
            pass
        self.check()
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def _create_connection(self, address, timeout, source_address=None):
        deadline = time.monotonic() + min(timeout, self.cancellation.remaining)
        addresses = self._resolve(*address, deadline)
        last_error: OSError | None = None
        for family, socktype, protocol, _, sockaddr in addresses:
            self.check()
            sock = socket.socket(family, socktype, protocol)
            connected = False
            try:
                self.publish_socket(sock)
                sock.setblocking(False)
                if source_address:
                    sock.bind(source_address)
                self.check()
                error = sock.connect_ex(sockaddr)
                pending = (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY, errno.EINTR)
                while error in pending:
                    _, writable, exceptional = select.select(
                        [], [sock], [sock], self._wait_budget(deadline)
                    )
                    if writable or exceptional:
                        error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                self.check()
                if error:
                    raise OSError(error, os.strerror(error))
                sock.settimeout(min(self._timeout, self.cancellation.remaining))
                self.check()
                connected = True
                return sock
            except OSError as exc:
                last_error = exc
                self.check()
            finally:
                if not connected:
                    with self._lock:
                        self._sockets.discard(sock)
                    shutdown_socket(sock)
                    sock.close()
        if last_error is not None:
            raise last_error
        raise OSError("getaddrinfo returned no addresses")
