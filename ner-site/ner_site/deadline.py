"""A Neon executor whose every call ends by a wall-clock deadline, for the record of runs
(site analytics design, section 9).

Why not the engine's ``NeonHTTPExecutor``. Its timeout is the HTTP library's: a limit
on each step of an exchange (connecting, sending, each read), not on the exchange. A
server that sends one byte now and then never lets a read time out, and a resolver that
does not answer is not timed at all, so a call can block for as long as the other side
likes. A blocked call in a thread cannot be stopped from outside either. For the
indexes that is a slow request; for recording a run it is a worker that holds one of
the instance's few places (``http.USAGE_MAX_IN_FLIGHT``) with no end in sight.

How this one keeps its deadline. The exchange is a coroutine on one event loop that a
daemon thread runs for the whole process; the calling thread hands it over and waits.
``asyncio.wait_for`` ends the coroutine when ``timeout`` seconds are up, wherever it
is: cancelling a task closes the connection it was reading from, so nothing is left
running. Name lookups, the one thing a loop cannot cancel, run in a few daemon threads
of their own (``_Loop.getaddrinfo``): the coroutine stops waiting at its deadline, the
thread ends when the system's resolver gives up, and while every one of them is stuck
a new lookup fails at once instead of adding a thread. So after ``timeout`` seconds
the caller has an answer or a ``TimeoutError``, and whatever was started for the call
is gone or is one of at most ``RESOLVER_THREADS`` daemon threads.

The protocol is the engine's (``xwalk.retrieval.postgres.NeonHTTPExecutor``): ``POST
https://<host>/sql`` with the connection string in ``Neon-Connection-String``, one
statement as ``{"query", "params"}``, a transaction as ``{"queries": [...]}``, rows as
arrays beside ``fields``, a SQL error as a JSON body with ``message`` and ``code``,
bytes as bytea hex. Nothing is ever retried: a recorded run is not idempotent.

What leaves this module when a call fails is one of a few exceptions made here, with
fixed words: a ``PostgresError`` (a status, a SQLSTATE that looks like one, the name of
an error's class) or a ``TimeoutError``. Each is raised in the calling thread, outside
any handler, so it has no cause and no context. The exchange itself never raises: it
hands back what happened as plain values. So nothing the server sent (it may echo the
connection string), nothing the HTTP library said about a request (it quotes header
values) and no decoder's error (it holds the body) can travel with a failure. A
connection string that could not be sent as a header is refused when the executor is
made, in fixed words.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import re
import socket
import threading
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx
from xwalk.retrieval.postgres import PostgresError, Row, Statement

__all__ = ["RESOLVER_THREADS", "DeadlineExecutor"]

# Name lookups under way at once, in daemon threads. A lookup the system's resolver
# does not answer keeps its thread until the resolver gives up; with all of them stuck
# a new lookup fails at once.
RESOLVER_THREADS = 4
# How long past its deadline a call may wait for the loop to say that the exchange was
# cancelled, before it stops waiting anyway (a loop that is itself held up).
_GRACE_S = 0.5
MAX_CONNECTIONS = 4
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")
_UNUSABLE = "a postgres:// or postgresql:// connection string with a host"
_LATE = "Neon did not answer in time"
# What an exchange hands back instead of raising: its first item says what happened.
_OK, _TIMED_OUT, _UNREACHABLE, _REFUSED, _NOT_JSON = "ok", "late", "unreachable", "status", "text"


class _Loop(asyncio.SelectorEventLoop):
    """The event loop of this module, with name lookups it can walk away from."""

    def __init__(self) -> None:
        super().__init__()
        self._lookups = 0
        self._lookups_lock = threading.Lock()

    async def getaddrinfo(self, host: Any, port: Any, **options: Any) -> Any:  # type: ignore[override]
        with self._lookups_lock:
            if self._lookups >= RESOLVER_THREADS:
                raise OSError("too many name lookups are still waiting for the resolver")
            self._lookups += 1
        answer: asyncio.Future[Any] = self.create_future()

        def settle(result: Any, error: BaseException | None) -> None:
            if answer.done():  # the caller's deadline came first
                return
            if error is not None:
                answer.set_exception(error)
            else:
                answer.set_result(result)

        def look_up() -> None:
            result, error = None, None
            try:
                result = socket.getaddrinfo(host, port, **options)
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting coroutine
                error = exc
            finally:
                self._lookup_over()
            try:
                self.call_soon_threadsafe(settle, result, error)
            except RuntimeError:  # the loop is closed: nobody is waiting
                return

        started = False
        try:
            threading.Thread(target=look_up, name="ner-site-usage-dns", daemon=True).start()
            started = True
        finally:
            if not started:  # no thread, so nothing will ever give the place back
                self._lookup_over()
        return await answer

    def _lookup_over(self) -> None:
        with self._lookups_lock:
            self._lookups -= 1


_LOOP_LOCK = threading.Lock()
_LOOP: _Loop | None = None


def _loop() -> _Loop:
    """The one loop, started on first use in a daemon thread that never holds the
    process."""
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            loop = _Loop()
            threading.Thread(
                target=loop.run_forever, name="ner-site-usage-loop", daemon=True
            ).start()
            _LOOP = loop
        return _LOOP


def _wire(value: Any) -> Any:
    """A parameter as the engine's executor sends it: bytes as bytea hex (as Neon's own
    driver does), a tuple as an array."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "\\x" + bytes(value).hex()
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    return value


def _rows(result: Any) -> list[Row] | None:
    """The rows of one result, or None for a result that is not one."""
    try:
        fields = result.get("fields") or []
        names = [str(f.get("name")) for f in fields if isinstance(f, dict)]
        out: list[Row] = []
        for row in result.get("rows") or []:
            # A dict is a server that ignored Neon-Array-Mode.
            out.append(dict(row) if isinstance(row, dict) else dict(zip(names, row, strict=False)))
    except (AttributeError, TypeError, ValueError):
        return None
    return out


def _results(payload: Any, count: int) -> list[list[Row]] | None:
    """The rows of a transaction's `count` results; None when the answer does not have
    one result for each statement, an empty list when one of them is not a result."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != count:
        return None
    found = [_rows(result) for result in results]
    return [rows for rows in found if rows is not None] if None not in found else []


def _usable(connection_string: str) -> tuple[bool, str]:
    """Whether the string can be used and sent, and the host as a URL writes it."""
    # A header value is plain ASCII without spaces or control characters; the HTTP
    # library's own refusal of anything else quotes the value.
    if not connection_string.isascii() or not all(
        0x21 <= ord(c) <= 0x7E for c in connection_string
    ):
        return False, ""
    try:
        parts = urlsplit(connection_string)
        host = parts.hostname or ""
        parts.port  # noqa: B018 - raises for a port that is not a number in range
    except ValueError:
        return False, ""
    usable = parts.scheme in {"postgres", "postgresql"} and bool(host)
    return usable, f"[{host}]" if ":" in host else host


class DeadlineExecutor:
    """An ``SQLExecutor`` over Neon's HTTPS SQL endpoint. ``execute`` and
    ``transaction`` each end within ``timeout`` seconds of wall-clock time, whatever
    the resolver, the network or the server does: with rows, a ``PostgresError`` (its
    ``code`` the SQLSTATE when the server gave one) or a ``TimeoutError``, each in
    fixed words (module docstring). Thread-safe.

    ``within(seconds)`` is the same executor with less time for a call, for a caller
    that has already used some of its own deadline. ``transport`` (an
    ``httpx.AsyncBaseTransport``) and ``endpoint`` exist for tests.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        endpoint: str | None = None,
    ) -> None:
        usable, host = _usable(connection_string)
        if not usable or not timeout > 0:
            raise ValueError(_UNUSABLE)  # fixed words: the connection string is a secret
        self._connection_string = connection_string
        self._endpoint = endpoint or f"https://{host}/sql"
        self._timeout = float(timeout)
        self._transport = transport
        # One HTTP client, and one queue for it, for the executor and every view of
        # it (``within``).
        self._shared: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"DeadlineExecutor(timeout={self._timeout:g})"

    def within(self, seconds: float) -> DeadlineExecutor:
        """This executor with at most `seconds` for a call (never more than its own
        timeout). With no time left a call fails at once and sends nothing."""
        view = copy.copy(self)
        view._timeout = min(self._timeout, float(seconds))
        return view

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        rows = _rows(self._call({"query": sql, "params": [_wire(p) for p in params]}))
        if rows is None:  # nothing of the answer is in this frame
            raise PostgresError("Neon answered with a result that is not an object")
        return rows

    def transaction(
        self, statements: Sequence[Statement], *, read_only: bool = False
    ) -> list[list[Row]]:
        body = {
            "queries": [
                {"query": sql, "params": [_wire(p) for p in params]} for sql, params in statements
            ]
        }
        found = _results(
            self._call(body, {"Neon-Batch-Read-Only": "true"} if read_only else {}),
            len(statements),
        )
        # Raised with nothing of the answer in this frame (``_results`` has returned).
        if found is None:
            raise PostgresError("Neon answered a transaction without one result per statement")
        if not found and statements:
            raise PostgresError("Neon answered with a result that is not an object")
        return found

    def _call(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        """One exchange, and what it came to: the payload, or one of this module's
        failures, made and raised here and nowhere inside a handler."""
        deadline = time.monotonic() + self._timeout
        outcome: tuple[Any, ...]
        try:
            content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            outcome = ("unwritten", type(exc).__name__)
        else:
            outcome = self._exchange_from_here(content, headers or {}, deadline)
        kind = outcome[0]
        if kind == _OK:
            return outcome[1]
        if kind == _TIMED_OUT:
            raise TimeoutError(_LATE)
        if kind == _REFUSED:
            _, status, code, retryable = outcome
            raise PostgresError(
                f"Neon answered HTTP {status}" + (f" with SQLSTATE {code}" if code else ""),
                code=code,
                status=status,
                retryable=retryable,
            )
        if kind == _NOT_JSON:
            raise PostgresError("Neon answered HTTP 200 with a body that is not JSON", status=200)
        if kind == "unwritten":
            raise PostgresError(f"a statement could not be written as JSON ({outcome[1]})")
        raise PostgresError(f"could not reach Neon ({outcome[1]})", retryable=True)

    def _exchange_from_here(
        self, content: bytes, headers: dict[str, str], deadline: float
    ) -> tuple[Any, ...]:
        """Hand the exchange to the loop and wait for what it hands back, no longer
        than to the deadline (and a moment, for a loop that is held up)."""
        if deadline - time.monotonic() <= 0:
            return (_TIMED_OUT,)
        future = asyncio.run_coroutine_threadsafe(
            self._exchange(content, headers, deadline), _loop()
        )
        try:
            return future.result(max(0.0, deadline - time.monotonic()) + _GRACE_S)
        except (TimeoutError, concurrent.futures.TimeoutError, concurrent.futures.CancelledError):
            # Cancelled where it is, or before it began: a write whose deadline has
            # passed is not sent when the loop gets to it.
            future.cancel()
            return (_TIMED_OUT,)
        except Exception as exc:  # noqa: BLE001 - the exchange does not raise; if it does
            return (_UNREACHABLE, type(exc).__name__)

    async def _exchange(
        self, content: bytes, headers: dict[str, str], deadline: float
    ) -> tuple[Any, ...]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:  # the loop got to it too late
            return (_TIMED_OUT,)
        try:
            return await asyncio.wait_for(self._post(content, headers), remaining)
        except (TimeoutError, asyncio.TimeoutError):
            return (_TIMED_OUT,)
        except Exception as exc:  # noqa: BLE001 - named by its class, never quoted
            return (_UNREACHABLE, type(exc).__name__)

    async def _post(self, content: bytes, headers: dict[str, str]) -> tuple[Any, ...]:
        client = self._shared.get("client")
        if client is None:
            # Made on the loop, on first use: part of the first call's deadline. It
            # has no timeouts of its own: the deadline of the call is the only one, so
            # a call that runs out of time always ends the same way. Nor a limit on
            # its connections: calls wait their turn at the semaphore below, not in
            # the library's pool, which does not get over waiters that were cancelled
            # (httpcore 1.0.9: after more cancelled calls at once than it had
            # connections, every later call waited for a connection that never came).
            client = self._shared["client"] = httpx.AsyncClient(
                timeout=None,
                transport=self._transport,
                limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=MAX_CONNECTIONS
                ),
            )
            self._shared["turns"] = asyncio.Semaphore(MAX_CONNECTIONS)
        async with self._shared["turns"]:
            response = await client.post(
                self._endpoint,
                content=content,
                headers={
                    "Neon-Connection-String": self._connection_string,
                    "Neon-Array-Mode": "true",
                    "Content-Type": "application/json",
                    **headers,
                },
            )
        status = response.status_code
        try:
            data: Any = response.json()
            readable = True
        except ValueError:  # the decoder's error holds the body: it stays in here
            data, readable = None, False
        if status == 200:
            return (_OK, data) if readable else (_NOT_JSON,)
        # As the engine reads an error: a JSON object with a message is Neon's, and
        # may carry a SQLSTATE and Neon's mark that a retry could succeed. The words
        # themselves are left out (they may name the host or quote a value), and a
        # code is kept only when it has the form of a SQLSTATE.
        code, marked = None, False
        if isinstance(data, dict) and isinstance(data.get("message"), str):
            found = data.get("code")
            code = found if isinstance(found, str) and _SQLSTATE.fullmatch(found) else None
            marked = bool(data.get("neon:retryable"))
        return (_REFUSED, status, code, marked or status == 429 or status >= 500)

    def close(self) -> None:
        """Close the HTTP client, if one was made (tests)."""
        client = self._shared.pop("client", None)
        self._shared.pop("turns", None)
        if client is not None:
            asyncio.run_coroutine_threadsafe(client.aclose(), _loop()).result(5)
