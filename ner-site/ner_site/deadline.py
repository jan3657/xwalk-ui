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
arrays beside ``fields``, a SQL error as a JSON body with ``message`` and ``code``.
Nothing is ever retried: a recorded run is not idempotent. No message of an error
raised here holds the connection string, the host or anything the server wrote.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import socket
import threading
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
                with self._lookups_lock:
                    self._lookups -= 1
            with contextlib.suppress(RuntimeError):  # the loop is closed: nobody waits
                self.call_soon_threadsafe(settle, result, error)

        threading.Thread(target=look_up, name="ner-site-usage-dns", daemon=True).start()
        return await answer


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


def _rows(result: Any) -> list[Row]:
    if not isinstance(result, dict):
        raise PostgresError("Neon answered with a result that is not an object")
    fields = result.get("fields") or []
    names = [str(f.get("name")) for f in fields if isinstance(f, dict)]
    out: list[Row] = []
    for row in result.get("rows") or []:
        # A dict is a server that ignored Neon-Array-Mode.
        out.append(dict(row) if isinstance(row, dict) else dict(zip(names, row, strict=False)))
    return out


class DeadlineExecutor:
    """An ``SQLExecutor`` over Neon's HTTPS SQL endpoint. ``execute`` and
    ``transaction`` each end within ``timeout`` seconds of wall-clock time, whatever
    the resolver, the network or the server does: with rows, a ``PostgresError`` (its
    ``code`` the SQLSTATE when the server gave one) or a ``TimeoutError``. Thread-safe.

    ``transport`` (an ``httpx.AsyncBaseTransport``) and ``endpoint`` exist for tests.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
        endpoint: str | None = None,
    ) -> None:
        try:
            parts = urlsplit(connection_string)
            usable = parts.scheme in {"postgres", "postgresql"} and bool(parts.hostname)
        except ValueError:
            usable = False
        if not usable or timeout <= 0:
            # Fixed words: the connection string is a secret.
            raise ValueError("a postgres:// or postgresql:// connection string with a host")
        self._connection_string = connection_string
        self._endpoint = endpoint or f"https://{parts.hostname}/sql"
        self._timeout = float(timeout)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def __repr__(self) -> str:
        return f"DeadlineExecutor(timeout={self._timeout:g})"

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        return _rows(self._call({"query": sql, "params": list(params)}))

    def transaction(
        self, statements: Sequence[Statement], *, read_only: bool = False
    ) -> list[list[Row]]:
        body = {"queries": [{"query": sql, "params": list(params)} for sql, params in statements]}
        payload = self._call(body, {"Neon-Batch-Read-Only": "true"} if read_only else {})
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or len(results) != len(statements):
            raise PostgresError("Neon answered a transaction without one result per statement")
        return [_rows(result) for result in results]

    def _call(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        exchange = asyncio.wait_for(self._post(content, headers or {}), self._timeout)
        future = asyncio.run_coroutine_threadsafe(exchange, _loop())
        try:
            return future.result(self._timeout + _GRACE_S)
        except (TimeoutError, concurrent.futures.TimeoutError, asyncio.TimeoutError):
            future.cancel()  # nothing to cancel when the loop already ended it
            failure: Exception = TimeoutError(f"Neon did not answer within {self._timeout:g} s")
        except concurrent.futures.CancelledError:
            failure = TimeoutError("the request to Neon was cancelled")
        # Raised out here: the original may quote what the server sent.
        raise failure

    async def _post(self, content: bytes, headers: dict[str, str]) -> Any:
        if self._client is None:
            # Made on the loop, on first use: part of the first call's deadline. It
            # has no timeouts of its own: the deadline of the call is the only one, so
            # a call that runs out of time always ends the same way.
            self._client = httpx.AsyncClient(
                timeout=None,
                transport=self._transport,
                limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
            )
        try:
            response = await self._client.post(
                self._endpoint,
                content=content,
                headers={
                    "Neon-Connection-String": self._connection_string,
                    "Neon-Array-Mode": "true",
                    "Content-Type": "application/json",
                    **headers,
                },
            )
        except (httpx.HTTPError, httpx.InvalidURL, OSError) as exc:
            failure = PostgresError(f"could not reach Neon ({type(exc).__name__})", retryable=True)
        else:
            return self._read(response)
        raise failure

    @staticmethod
    def _read(response: httpx.Response) -> Any:
        status = response.status_code
        data: Any = None
        try:
            data = response.json()
        except ValueError:
            data = None
            if status == 200:
                raise PostgresError(
                    f"Neon answered HTTP 200 with a body that is not JSON "
                    f"({len(response.content)} bytes)",
                    status=200,
                ) from None
        if status == 200:
            return data
        code = data.get("code") if isinstance(data, dict) else None
        code = code if isinstance(code, str) and code else None
        return _refused(status, code)

    def close(self) -> None:
        """Close the HTTP client, if one was made (tests)."""
        client, self._client = self._client, None
        if client is not None:
            asyncio.run_coroutine_threadsafe(client.aclose(), _loop()).result(5)


def _refused(status: int, code: str | None) -> Any:
    # The server's own words are left out: they may name the host or quote a value.
    raise PostgresError(
        f"Neon answered HTTP {status}" + (f" with SQLSTATE {code}" if code else ""),
        code=code,
        status=status,
        retryable=status == 429 or status >= 500,
    )
