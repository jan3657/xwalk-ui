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
        self.behave = False  # set: connections accepted from now on are answered well
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
            while self.behave:  # one good answer for each request, as long as they come
                body = b'{"fields": [{"name": "n"}], "rows": [[1]]}'
                head = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                connection.sendall(head + b"Content-Length: %d\r\n\r\n" % len(body) + body)
                connection.settimeout(5)
                if connection.recv(65536) == b"":
                    return
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
        assert str(caught.value) == "Neon did not answer in time"
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


# --- round 3: errors that carry nothing, one deadline, the engine's parameters ----------

SENTINEL_URL = "postgresql://owner:npg_URLSENTINELpw9@ep-sentinel-host.neon.tech/neondb"
URL_SECRETS = (SENTINEL_URL, "npg_URLSENTINELpw9", "URLSENTINEL", "ep-sentinel-host")


def _chain(exc: BaseException) -> list[BaseException]:
    """`exc` and every exception reachable from it through causes and contexts."""
    seen: list[BaseException] = []
    stack: list[BaseException | None] = [exc]
    while stack:
        item = stack.pop()
        if item is None or any(item is other for other in seen):
            continue
        seen.append(item)
        stack += [item.__cause__, item.__context__]
    return seen


def _assert_sealed(exc: BaseException, *extra: str) -> None:
    """Nothing of the URL, the server's words or `extra` in the exception: its message,
    arguments, attributes, repr, or anything chained to it."""
    assert exc.__cause__ is None and exc.__context__ is None
    for item in _chain(exc):
        shown = " ".join([str(item), repr(item), repr(item.args), repr(vars(item))])
        for secret in (*URL_SECRETS, *extra):
            assert secret not in shown, (secret, type(item).__name__)


ECHOES: dict[str, tuple[int, bytes, dict[str, str]]] = {
    # The three cases of the second re-review.
    "a code that echoes the URL": (
        400,
        json.dumps({"message": "bad", "code": f"failed for {SENTINEL_URL}"}).encode(),
        {"Content-Type": "application/json"},
    ),
    "a 200 that is not JSON and echoes the URL": (
        200,
        f'{{"rows": [ oops {SENTINEL_URL}'.encode(),
        {"Content-Type": "application/json"},
    ),
    "an error that is not JSON and echoes the URL": (
        500,
        f"<html>upstream {SENTINEL_URL}</html>".encode(),
        {"Content-Type": "text/html"},
    ),
    # And what else a server could try.
    "a message that echoes the URL": (
        400,
        json.dumps({"message": f"no relation at {SENTINEL_URL}", "code": "42P01"}).encode(),
        {"Content-Type": "application/json"},
    ),
    "a code that is no string": (
        400,
        json.dumps({"message": "bad", "code": [SENTINEL_URL]}).encode(),
        {"Content-Type": "application/json"},
    ),
    "a code of five characters that are not a SQLSTATE": (
        400,
        json.dumps({"message": "bad", "code": "ab cd", "detail": SENTINEL_URL}).encode(),
        {"Content-Type": "application/json"},
    ),
    "a redirect to a URL with the secret": (
        302,
        b"",
        {"Location": f"https://elsewhere.example/?u={SENTINEL_URL}"},
    ),
    "a body that is not UTF-8": (200, b"\xff\xfe" + SENTINEL_URL.encode(), {}),
    "rows that are no rows": (
        200,
        json.dumps({"fields": SENTINEL_URL, "rows": 7}).encode(),
        {"Content-Type": "application/json"},
    ),
}


@pytest.mark.parametrize("case", list(ECHOES), ids=list(ECHOES))
@pytest.mark.parametrize("call", ["execute", "transaction"])
def test_no_failure_carries_what_the_server_sent_or_the_connection_string(
    case: str, call: str
) -> None:
    status, content, headers = ECHOES[case]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=content, headers=headers)

    ours = DeadlineExecutor(SENTINEL_URL, timeout=5, transport=httpx.MockTransport(handler))
    with pytest.raises(PostgresError) as caught:
        if call == "execute":
            ours.execute("SELECT 1")
        else:
            ours.transaction([("SELECT 1", ()), ("SELECT 2", ())])
    exc = caught.value
    _assert_sealed(exc, "oops", "upstream", "elsewhere", "no relation")
    assert type(exc) is PostgresError
    assert exc.code in (None, "42P01") and (exc.code is None or len(exc.code) == 5)
    assert str(exc) in {
        f"Neon answered HTTP {status}",
        "Neon answered HTTP 400 with SQLSTATE 42P01",
        "Neon answered HTTP 200 with a body that is not JSON",
        "Neon answered with a result that is not an object",
        "Neon answered a transaction without one result per statement",
        "Neon answered with rows that cannot be read",
    }
    ours.close()


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://owner:päss-URLSENTINEL@ep-sentinel-host.neon.tech/db",
        "postgresql://owner:URLSENTINEL☃@ep-sentinel-host.neon.tech/db",
        "postgresql://owner:URLSENTINEL pw@ep-sentinel-host.neon.tech/db",
        "postgresql://owner:URLSENTINEL\npw@ep-sentinel-host.neon.tech/db",
        "postgresql://owner:URLSENTINEL\x7f@ep-sentinel-host.neon.tech/db",
        "postgresql://owner:URLSENTINEL@ep-sentinel-host.neon.tech:port/db",
    ],
)
def test_a_connection_string_that_cannot_be_sent_is_refused_before_any_request(url: str) -> None:
    """A header takes plain ASCII. The HTTP library's own refusal quotes the value."""
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, json={"fields": [], "rows": []})

    with pytest.raises(ValueError) as caught:
        DeadlineExecutor(url, timeout=5, transport=httpx.MockTransport(handler))
    assert str(caught.value) == "a postgres:// or postgresql:// connection string with a host"
    assert type(caught.value) is ValueError
    _assert_sealed(caught.value, "päss")
    assert hits == 0


def test_whatever_fails_while_sending_is_one_fixed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should the library refuse a request for a reason of its own, quoting what it was
    given (a header it cannot encode, a URL it cannot build), that text stays inside."""
    for error in (
        UnicodeEncodeError("ascii", SENTINEL_URL, 0, 1, "ordinal not in range"),
        httpx.InvalidURL(f"cannot use {SENTINEL_URL}"),
        httpx.ConnectError(f"cannot connect for {SENTINEL_URL}"),
        RuntimeError(f"something else with {SENTINEL_URL}"),
        OSError(f"socket for {SENTINEL_URL}"),
    ):

        async def handler(request: httpx.Request, error: Exception = error) -> httpx.Response:
            raise error

        ours = DeadlineExecutor(SENTINEL_URL, timeout=5, transport=httpx.MockTransport(handler))
        with pytest.raises(PostgresError) as caught:
            ours.execute("SELECT 1")
        _assert_sealed(caught.value)
        assert str(caught.value) == f"could not reach Neon ({type(error).__name__})"
        assert caught.value.code is None
        ours.close()
    # A parameter that cannot be written as JSON fails before anything is sent.
    ours = DeadlineExecutor(SENTINEL_URL, timeout=5, transport=httpx.MockTransport(handler))
    with pytest.raises(PostgresError) as caught:
        ours.execute("SELECT $1", [object()])
    assert str(caught.value) == "a statement could not be written as JSON (TypeError)"
    with pytest.raises(PostgresError):
        ours.execute("SELECT $1", [float("nan")])
    _assert_sealed(caught.value)


def test_the_log_line_of_a_run_holds_nothing_of_a_failed_write(
    cache_dir: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The three cases through the handler: the run is answered, the log line names a
    class, and neither holds the URL, its password, its host or the server's words."""
    for case in list(ECHOES)[:3]:
        status, content, headers = ECHOES[case]

        async def handler(
            request: httpx.Request,
            status: int = status,
            content: bytes = content,
            headers: dict[str, str] = headers,
        ) -> httpx.Response:
            return httpx.Response(status, content=content, headers=headers)

        monkeypatch.setattr(
            http_mod,
            "make_usage_executor",
            lambda url, timeout, handler=handler: DeadlineExecutor(
                url, timeout=timeout, transport=httpx.MockTransport(handler)
            ),
        )
        http_mod.reset()
        s = _settings(
            cache_dir,
            tmp_path,
            OPENROUTER_API_KEY=SENTINEL_KEY,
            NER_SITE_STATS_PASSCODE=STATS_PASSCODE,
            DATABASE_URL=SENTINEL_URL,
            NER_SITE_VISITOR_SALT="a-salt-of-its-own",
        )
        capsys.readouterr()
        result = _post(s, {"text": "x"}, llm_factory=_fake('{"entities": []}'))
        stats = http_mod.handle(
            "GET", "/api/stats", {"x-stats-passcode": STATS_PASSCODE}, b"", settings=s
        )
        err = capsys.readouterr().err
        run_line, stats_line = (json.loads(line) for line in err.strip().splitlines())
        assert result[0] == 200 and run_line["usage"] == "PostgresError"
        assert stats[0] == 503 and stats_line["stats"] == (
            "site runs could not be queried (PostgresError)"
        )
        for secret in (*URL_SECRETS, "oops", "upstream"):
            assert secret not in err, (case, secret)
            assert secret.encode() not in result[2] + stats[2]


def test_a_sqlstate_is_five_characters_of_its_alphabet_or_none() -> None:
    codes = ["42P01", "08006", "42p01", "42P011", "42P0", " 42P01", "42P01\n", "", None, 42601]
    kept = []
    for code in codes:

        async def handler(request: httpx.Request, code: Any = code) -> httpx.Response:
            return httpx.Response(400, json={"message": "m", "code": code})

        ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(handler))
        with pytest.raises(PostgresError) as caught:
            ours.execute("SELECT 1")
        kept.append(caught.value.code)
        ours.close()
    assert kept == ["42P01", "08006", *[None] * 8]


def test_binary_parameters_and_the_retryable_mark_are_the_engines() -> None:
    """Parity with ``NeonHTTPExecutor`` beyond what analytics sends: bytes travel as
    bytea hex and tuples as arrays, and an error the server marks retryable says so
    (nothing here acts on it: there are no retries)."""
    seen: dict[str, list[Any]] = {"engine": [], "ours": []}
    answers = [
        (200, {"fields": [{"name": "n"}], "rows": [[1]]}),
        (400, {"message": "serialization failure", "code": "40001", "neon:retryable": True}),
        (400, {"message": "syntax", "code": "42601"}),
        (400, {"code": "40001", "neon:retryable": True}),  # no message: not Neon's shape
        (429, {"message": "slow down"}),
        (503, {"message": "starting", "neon:retryable": False}),
    ]

    def record(kind: str, request: httpx.Request) -> httpx.Response:
        seen[kind].append(json.loads(request.content))
        status, body = answers[len(seen[kind]) - 1]
        return httpx.Response(status, json=body)

    async def ours_handler(request: httpx.Request) -> httpx.Response:
        return record("ours", request)

    engine = NeonHTTPExecutor(
        URL, max_retries=0, transport=httpx.MockTransport(lambda r: record("engine", r))
    )
    ours = DeadlineExecutor(URL, timeout=5, transport=httpx.MockTransport(ours_handler))
    params = [b"\x00\xffbytes", bytearray(b"ab"), memoryview(b"cd"), (1, (b"\x01", "x")), None]

    def outcome(executor: Any) -> Any:
        try:
            return executor.execute("SELECT $1, $2, $3, $4, $5", params)
        except PostgresError as exc:
            return (exc.code, exc.status, exc.retryable)

    results = [(outcome(ours), outcome(engine)) for _ in answers]
    assert all(mine == theirs for mine, theirs in results), results
    assert [r[0] for r in results[1:]] == [
        ("40001", 400, True),
        ("42601", 400, False),
        (None, 400, False),
        (None, 429, True),
        (None, 503, True),
    ]
    assert seen["ours"] == seen["engine"]
    assert seen["ours"][0]["params"] == [
        "\\x00ff6279746573",
        "\\x6162",
        "\\x6364",
        [1, ["\\x01", "x"]],
        None,
    ]
    statements = [("SELECT $1", (b"\x01",)), ("SELECT $1", ((b"\x02",),))]
    seen["ours"].clear()
    seen["engine"].clear()
    answers[:] = [(200, {"results": [{"fields": [], "rows": []}] * 2})] * 2
    assert ours.transaction(statements) == engine.transaction(statements)
    assert seen["ours"] == seen["engine"]
    ours.close()


def test_a_bracketed_address_keeps_its_brackets_in_the_endpoint() -> None:
    urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json={"fields": [], "rows": []})

    for url in ("postgresql://u:p@[2001:db8::7]:5432/db", "postgresql://u:p@db.example.:5432/db"):
        ours = DeadlineExecutor(url, timeout=5, transport=httpx.MockTransport(handler))
        ours.execute("SELECT 1")
        ours.close()
    assert urls == ["https://[2001:db8::7]/sql", "https://db.example./sql"]


def test_a_view_with_less_time_ends_sooner_and_shares_the_connection_pool() -> None:
    server = Misbehaving("body")
    ours = DeadlineExecutor(URL, timeout=30, endpoint=server.endpoint)
    try:
        short = ours.within(TIMEOUT)
        assert isinstance(short, SQLExecutor) and repr(short) == "DeadlineExecutor(timeout=0.4)"
        started = time.monotonic()
        with pytest.raises(TimeoutError) as caught:
            short.execute("SELECT 1")
        assert TIMEOUT <= time.monotonic() - started < TIMEOUT + BOUND
        assert str(caught.value) == "Neon did not answer in time"
        _assert_sealed(caught.value)
        # No more than the executor's own time, and never a wait that is not positive.
        assert repr(ours.within(99)) == "DeadlineExecutor(timeout=30)"
        for none in (0, -1, 1e-9):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                ours.within(none).execute("SELECT 1")
            assert time.monotonic() - started < 0.2
        assert _wait_for(lambda: server.gone == server.accepted == 1, BOUND)
    finally:
        server.close()
        ours.close()


def test_a_write_whose_deadline_passed_while_the_loop_was_held_up_is_never_sent() -> None:
    """The loop itself delayed (it is one thread): the caller still gets its answer
    soon after the deadline, and the write does not go out when the loop is back."""
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, json={"fields": [], "rows": []})

    ours = DeadlineExecutor(URL, timeout=TIMEOUT, transport=httpx.MockTransport(handler))
    ours.execute("SELECT 1")
    assert hits == 1
    deadline._loop().call_soon_threadsafe(time.sleep, TIMEOUT * 4)
    time.sleep(0.05)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        ours.execute("SELECT 1")
    assert time.monotonic() - started < TIMEOUT + BOUND
    time.sleep(TIMEOUT * 4)
    assert hits == 1
    ours.execute("SELECT 1")
    assert hits == 2
    ours.close()


def test_a_lookup_thread_that_cannot_start_gives_its_place_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four threads that failed to start used to use up the lookups for good."""
    real_thread = threading.Thread
    refuse = True

    class Thread(real_thread):
        def start(self) -> None:
            if refuse and self.name == "ner-site-usage-dns":
                raise RuntimeError("can't start new thread")
            super().start()

    resolved = socket.getaddrinfo("127.0.0.1", 80)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: resolved,
    )
    monkeypatch.setattr(deadline.threading, "Thread", Thread)
    ours = DeadlineExecutor(URL, timeout=TIMEOUT, endpoint="http://name.lookup.test:9/sql")
    for _ in range(deadline.RESOLVER_THREADS + 2):
        with pytest.raises(PostgresError) as caught:
            ours.execute("SELECT 1")
        assert str(caught.value) == "could not reach Neon (RuntimeError)"
    assert deadline._loop()._lookups == 0
    refuse = False
    server = Misbehaving("silent")
    try:
        reached = DeadlineExecutor(
            URL, timeout=TIMEOUT, endpoint=f"http://name.lookup.test:{server.port}/sql"
        )
        with pytest.raises(TimeoutError):  # the lookup ran: the silent server was reached
            reached.execute("SELECT 1")
        assert server.accepted == 1
        reached.close()
    finally:
        server.close()
        ours.close()


def test_calls_cancelled_while_they_waited_their_turn_do_not_use_the_executor_up() -> None:
    """Found by probing: with more calls at once than the HTTP library's pool had
    connections, the ones cancelled while they waited in that pool were never got
    over, and every later call timed out for good. Calls wait at a semaphore of this
    module instead."""
    server = Misbehaving("body")
    ours = DeadlineExecutor(URL, timeout=TIMEOUT, endpoint=server.endpoint)
    outcomes: list[str] = []

    def call() -> None:
        try:
            ours.execute("SELECT 1")
            outcomes.append("rows")
        except Exception as exc:  # noqa: BLE001 - collected by name
            outcomes.append(type(exc).__name__)

    try:
        threads = [threading.Thread(target=call) for _ in range(deadline.MAX_CONNECTIONS * 6)]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        assert time.monotonic() - started < TIMEOUT + BOUND
        assert outcomes == ["TimeoutError"] * len(threads)
        # No more than the executor's share of connections was ever opened.
        assert server.accepted == deadline.MAX_CONNECTIONS
        server.behave = True
        assert [ours.execute("SELECT 1") for _ in range(6)] == [[{"n": 1}]] * 6
    finally:
        server.close()
        ours.close()
