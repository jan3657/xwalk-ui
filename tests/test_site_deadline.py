"""ner_site.deadline: a Neon executor whose calls end by a wall-clock deadline, and the
record of runs through it (site analytics design, section 9).

Offline. "Neon" here is an ``httpx.MockTransport``, or a socket on 127.0.0.1 that
misbehaves the way a re-review's probes did: it answers one byte at a time for longer
than the deadline, in the body or in the headers, or it accepts the connection and says
nothing; and a resolver that never answers. The engine's own executor
(``NeonHTTPExecutor``) is run against the same probes first, to show what they do to
a client whose timeout is per read.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from ner_site import deadline
from ner_site import http as http_mod
from ner_site.deadline import DeadlineExecutor
from ner_site.service import Settings
from xwalk.retrieval.postgres import NeonHTTPExecutor, PostgresError, SQLExecutor

from tests.test_site_http import (  # noqa: F401 - cache_dir is a fixture
    DB_URL,
    SENTINEL_KEY,
    STATS_PASSCODE,
    _fake,
    _post,
    _settings,
    cache_dir,
)

URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"
SECRETS = (URL, "npg_SENTINELpw0123", "ep-x-pooler")
# The deadline of the calls here, and how long after it everything must be over.
TIMEOUT = 0.4
BOUND = 1.0


def _wait_for(condition: Callable[[], bool], seconds: float) -> bool:
    end = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


def _threads(prefix: str) -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith(prefix) and t.is_alive())


# --- the protocol: the engine's, statement for statement --------------------------------

STATS_ROW = {"fields": [{"name": "stats"}, {"name": "n"}], "rows": [['{"a": 1}', 7]]}
SCRIPT: list[tuple[str, int, Any]] = [
    ("one row", 200, STATS_ROW),
    ("no rows", 200, {"fields": [], "rows": []}),
    ("rows as objects", 200, {"fields": [], "rows": [{"stats": "x"}]}),
    ("a SQL error", 400, {"message": f"relation missing at {URL}", "code": "42P01"}),
    ("a server error", 503, {"message": "compute is starting", "code": None}),
    ("an error that is not JSON", 502, "<html>bad gateway</html>"),
    ("a 200 that is not JSON", 200, "<html>ok</html>"),
    ("a result that is no object", 200, [1, 2]),
]


def _answer(status: int, body: Any) -> httpx.Response:
    if isinstance(body, str):
        return httpx.Response(status, text=body)
    return httpx.Response(status, json=body)


def _outcome(call: Callable[[], Any]) -> tuple[str, Any, Any]:
    try:
        return ("rows", call(), None)
    except PostgresError as exc:
        return ("PostgresError", exc.code, exc.status)


@pytest.mark.parametrize(("name", "status", "body"), SCRIPT, ids=[s[0] for s in SCRIPT])
def test_one_statement_is_sent_and_read_as_the_engines_executor_does(
    name: str, status: int, body: Any
) -> None:
    seen: dict[str, list[tuple[str, dict[str, str], Any]]] = {"engine": [], "ours": []}

    def record(kind: str, request: httpx.Request) -> httpx.Response:
        headers = {
            k: v for k, v in request.headers.items() if k.startswith("neon-") or k == "content-type"
        }
        seen[kind].append((str(request.url), headers, json.loads(request.content)))
        return _answer(status, body)

    async def ours_handler(request: httpx.Request) -> httpx.Response:
        return record("ours", request)

    engine = NeonHTTPExecutor(
        URL, max_retries=0, transport=httpx.MockTransport(lambda r: record("engine", r))
    )
    ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(ours_handler))
    sql, params = "/* xwalk:site_run */ INSERT ... $1::jsonb", ['{"at": "x", "n": null}']
    assert _outcome(lambda: ours.execute(sql, params)) == _outcome(
        lambda: engine.execute(sql, params)
    )
    assert seen["ours"] == seen["engine"]
    (url, headers, sent) = seen["ours"][0]
    assert url == "https://ep-x-pooler.c-1.eu.aws.neon.tech/sql"
    assert headers["neon-connection-string"] == URL and headers["neon-array-mode"] == "true"
    assert sent == {"query": sql, "params": params}
    ours.close()


@pytest.mark.parametrize("read_only", [False, True])
def test_a_transaction_is_one_request_with_one_result_per_statement(read_only: bool) -> None:
    seen: dict[str, list[Any]] = {"engine": [], "ours": []}
    results = {"results": [{"fields": [], "rows": []}, STATS_ROW]}

    def record(kind: str, request: httpx.Request) -> httpx.Response:
        seen[kind].append(
            (request.headers.get("neon-batch-read-only"), json.loads(request.content))
        )
        return httpx.Response(200, json=results)

    async def ours_handler(request: httpx.Request) -> httpx.Response:
        return record("ours", request)

    engine = NeonHTTPExecutor(
        URL, max_retries=0, transport=httpx.MockTransport(lambda r: record("engine", r))
    )
    ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(ours_handler))
    statements = [("CREATE TABLE IF NOT EXISTS t ()", ()), ("SELECT $1::date", ("2026-11-18",))]
    expected = [[], [{"stats": '{"a": 1}', "n": 7}]]
    assert ours.transaction(statements, read_only=read_only) == expected
    assert engine.transaction(statements, read_only=read_only) == expected
    assert seen["ours"] == seen["engine"] and len(seen["ours"]) == 1
    assert seen["ours"][0][0] == ("true" if read_only else None)
    ours.close()


def test_a_transaction_answered_with_too_few_results_is_an_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"fields": [], "rows": []}]})

    ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(handler))
    with pytest.raises(PostgresError, match="one result per statement"):
        ours.transaction([("SELECT 1", ()), ("SELECT 2", ())])
    ours.close()


def test_nothing_is_retried_and_no_error_holds_the_connection_string_or_the_servers_words() -> None:
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        if hits == 1:
            raise httpx.ConnectError(f"cannot connect to {URL}", request=request)
        return httpx.Response(500, json={"message": f"failed at {URL}", "code": "08006"})

    ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(handler))
    assert isinstance(ours, SQLExecutor)
    messages = []
    for code in (None, "08006"):
        with pytest.raises(PostgresError) as caught:
            ours.execute("SELECT 1")
        assert caught.value.code == code and caught.value.retryable
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        messages.append(str(caught.value))
    assert hits == 2
    assert messages == [
        "could not reach Neon (ConnectError)",
        "Neon answered HTTP 500 with SQLSTATE 08006",
    ]
    for secret in SECRETS:
        assert secret not in "".join(messages) + repr(ours)
    ours.close()


@pytest.mark.parametrize(
    "url",
    ["mysql://u:p@h/db", "postgresql:///nohost", "postgresql://[half/db", "", "ep-x.neon.tech"],
)
def test_a_connection_string_it_cannot_use_is_refused_in_fixed_words(url: str) -> None:
    with pytest.raises(ValueError) as caught:
        DeadlineExecutor(url, timeout=2)
    assert str(caught.value) == "a postgres:// or postgresql:// connection string with a host"


def test_calls_from_several_threads_share_the_loop_and_each_get_their_own_answer() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        (n,) = json.loads(request.content)["params"]
        return httpx.Response(200, json={"fields": [{"name": "n"}], "rows": [[n]]})

    ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(handler))
    answers: dict[int, Any] = {}
    threads = [
        threading.Thread(target=lambda n=n: answers.update({n: ours.execute("SELECT $1", [n])}))
        for n in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert answers == {n: [{"n": n}] for n in range(12)}
    assert _threads("ner-site-usage-loop") == 1
    ours.close()


# --- a server that does not finish, and a resolver that does not answer -----------------


class Misbehaving:
    """A server on 127.0.0.1 that takes a request and then, by ``mode``: ``body`` sends
    the headers of a long answer and its body one byte at a time; ``headers`` sends the
    header block itself one byte at a time; ``silent`` sends nothing. Until stopped,
    or until the client goes away, which it counts (``gone``)."""

    def __init__(self, mode: str, every: float = 0.02) -> None:
        self.mode, self.every = mode, every
        self.accepted = 0
        self.gone = 0
        self._stop = threading.Event()
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(32)
        self.port = self._server.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{self.port}/sql"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(0.5)
            data = b""
            while b"\r\n\r\n" not in data:
                data += connection.recv(65536)
            if self.mode == "body":
                head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                connection.sendall(head + b"Content-Length: 1000000\r\n\r\n")
            elif self.mode == "headers":
                connection.sendall(b"HTTP/1.1 200 OK\r\nX-Pad: ")
            connection.settimeout(self.every)
            while not self._stop.is_set():
                if self.mode != "silent":
                    connection.sendall(b" ")
                    time.sleep(self.every)
                    continue
                try:  # the rest of the request, then nothing until the client hangs up
                    if connection.recv(65536) == b"":
                        break
                except TimeoutError:
                    continue
        except OSError:
            pass
        finally:
            if not self._stop.is_set():
                self.gone += 1
            connection.close()

    def close(self) -> None:
        self._stop.set()
        self._server.close()


@pytest.fixture(params=["body", "headers", "silent"])
def misbehaving(request: pytest.FixtureRequest) -> Iterator[Misbehaving]:
    server = Misbehaving(request.param)
    try:
        yield server
    finally:
        server.close()


@pytest.mark.parametrize("mode", ["body", "headers"])
def test_the_engines_executor_has_no_deadline_against_a_trickling_answer(mode: str) -> None:
    """The probe, against the client recording used before: its timeout is per read, and
    a byte every 20 ms means no read ever times out. Six timeouts later it still waits."""
    server = Misbehaving(mode)
    engine = NeonHTTPExecutor(
        URL, timeout=TIMEOUT, max_retries=0, endpoint=server.endpoint, trust_env=False
    )
    ended = threading.Event()

    def call() -> None:
        with contextlib.suppress(Exception):  # only whether it ends matters
            engine.execute("SELECT 1")
        ended.set()

    try:
        threading.Thread(target=call, daemon=True).start()
        assert not ended.wait(TIMEOUT * 6)
    finally:
        server.close()
    assert ended.wait(10)  # the server went away: the read fails and the thread ends


def test_a_call_ends_at_its_deadline_whatever_the_server_does(misbehaving: Misbehaving) -> None:
    ours = DeadlineExecutor(URL, timeout=TIMEOUT, endpoint=misbehaving.endpoint)
    for call in (
        lambda: ours.execute("SELECT 1"),
        lambda: ours.transaction([("SELECT 1", ()), ("SELECT 2", ())]),
    ):
        started = time.monotonic()
        with pytest.raises(TimeoutError) as caught:
            call()
        waited = time.monotonic() - started
        assert TIMEOUT <= waited < TIMEOUT + BOUND
        assert str(caught.value) == "Neon did not answer within 0.4 s"
        assert caught.value.__cause__ is None and caught.value.__context__ is None
    # Cancelled, not abandoned: the connections are closed, so the server sees the
    # client go, and nothing but the loop's one thread is left.
    assert _wait_for(lambda: misbehaving.gone == misbehaving.accepted == 2, BOUND)
    assert _threads("ner-site-usage-loop") == 1 and _threads("ner-site-usage-dns") == 0
    ours.close()


@pytest.fixture
def stalled_resolver(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Name lookups for ``*.stalled.test`` never return until ``release`` is set."""
    release = threading.Event()
    real = socket.getaddrinfo
    asked: list[str] = []

    def lookup(host: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode("ascii", "replace") if isinstance(host, bytes) else str(host)
        if name.endswith(".stalled.test"):
            asked.append(name)
            release.wait(60)
            raise socket.gaierror("the resolver gave up")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    try:
        yield asked
    finally:
        release.set()
        assert _wait_for(lambda: _threads("ner-site-usage-dns") == 0, 5)


def test_a_resolver_that_never_answers_costs_a_call_its_deadline_and_a_few_threads(
    stalled_resolver: list[str],
) -> None:
    ours = DeadlineExecutor(
        URL, timeout=TIMEOUT, endpoint="http://db.stalled.test/sql", transport=None
    )
    outcomes = []
    for _ in range(deadline.RESOLVER_THREADS + 3):
        started = time.monotonic()
        with pytest.raises((TimeoutError, PostgresError)) as caught:
            ours.execute("SELECT 1")
        assert time.monotonic() - started < TIMEOUT + BOUND
        outcomes.append(type(caught.value).__name__)
    # Each of the first lookups holds a daemon thread until the resolver gives up; once
    # they are all held, a call fails at once instead of adding one.
    stuck = deadline.RESOLVER_THREADS
    assert outcomes == ["TimeoutError"] * stuck + ["PostgresError"] * 3
    assert len(stalled_resolver) == stuck == _threads("ner-site-usage-dns")
    assert all(t.daemon for t in threading.enumerate() if t.name.startswith("ner-site-usage"))


# --- the record of runs through it: no place is held past the deadline ------------------


def _recording_site(cache: Path, tmp_path: Path) -> Settings:
    return _settings(
        cache,
        tmp_path,
        OPENROUTER_API_KEY=SENTINEL_KEY,
        NER_SITE_STATS_PASSCODE=STATS_PASSCODE,
        DATABASE_URL=DB_URL,
    )


def _runs(settings: Settings, count: int, capsys: pytest.CaptureFixture[str]) -> list[Any]:
    capsys.readouterr()
    for _ in range(count):
        started = time.monotonic()
        result = _post(settings, {"text": "x"}, llm_factory=_fake('{"entities": []}'))
        assert result[0] == 200
        # The request itself (a busy machine makes it slow), and no more than one wait.
        assert time.monotonic() - started < TIMEOUT + 3
    lines = capsys.readouterr().err.strip().splitlines()
    return [json.loads(line).get("usage") for line in lines]


def _places_free(bound: float = BOUND) -> bool:
    return _wait_for(
        lambda: (
            http_mod._USAGE_FLIGHTS.count == 0
            and not any(t.name == "ner-site-usage" for t in threading.enumerate())
        ),
        bound,
    )


def test_every_place_is_free_again_after_the_deadline_against_a_trickling_database(
    cache_dir: Path,  # noqa: F811
    tmp_path: Path,
    misbehaving: Misbehaving,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The first probe of the re-review: with the engine's executor all four places
    were still taken twelve timeouts later. Nothing here releases the server."""
    monkeypatch.setattr(http_mod, "USAGE_TIMEOUT_S", TIMEOUT)
    monkeypatch.setattr(
        http_mod,
        "make_usage_executor",
        lambda url, timeout: DeadlineExecutor(url, timeout=timeout, endpoint=misbehaving.endpoint),
    )
    s = _recording_site(cache_dir, tmp_path)
    cap = http_mod.USAGE_MAX_IN_FLIGHT
    notes = _runs(s, cap + 2, capsys)
    # Sequential requests never find the places taken: each write is over, by its own
    # deadline, when its request stops waiting for it.
    assert notes == ["TimeoutError"] * (cap + 2)
    assert _places_free()
    assert _wait_for(lambda: misbehaving.gone == misbehaving.accepted == cap + 2, BOUND)
    # At once, three times as many as there are places.
    answers: list[int] = []
    threads = [
        threading.Thread(
            target=lambda: answers.append(
                _post(s, {"text": "x"}, llm_factory=_fake('{"entities": []}'))[0]
            )
        )
        for _ in range(cap * 3)
    ]
    capsys.readouterr()
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert answers == [200] * (cap * 3) and time.monotonic() - started < TIMEOUT + 5
    lines = capsys.readouterr().err.strip().splitlines()
    notes = sorted(json.loads(line).get("usage") for line in lines)
    assert notes == ["TimeoutError"] * cap + ["saturated"] * (cap * 2)
    assert _places_free()


def test_every_place_is_free_again_after_the_deadline_with_a_stalled_resolver(
    cache_dir: Path,  # noqa: F811
    tmp_path: Path,
    stalled_resolver: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The second probe: the real executor as the site builds it, for a host whose
    name never resolves."""
    monkeypatch.setattr(http_mod, "USAGE_TIMEOUT_S", TIMEOUT)
    monkeypatch.setattr(
        http_mod,
        "make_usage_executor",
        lambda url, timeout: DeadlineExecutor(
            url, timeout=timeout, endpoint="https://ep-x.stalled.test/sql"
        ),
    )
    s = _recording_site(cache_dir, tmp_path)
    count = deadline.RESOLVER_THREADS + 3
    notes = _runs(s, count, capsys)
    assert notes[: deadline.RESOLVER_THREADS] == ["TimeoutError"] * deadline.RESOLVER_THREADS
    assert notes[deadline.RESOLVER_THREADS :] == ["PostgresError"] * 3
    assert _places_free()
    assert _threads("ner-site-usage-dns") == deadline.RESOLVER_THREADS
