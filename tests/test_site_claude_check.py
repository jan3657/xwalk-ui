"""``ner-site/tools/claude_check.py`` against a stub site on a local socket.

Offline: the stub is a ``ThreadingHTTPServer`` on 127.0.0.1 that answers ``/api/health``
and ``/api/ner`` from a script and records what it was sent. Nothing here reaches a
deployment, a provider or a database.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TOOL = Path(__file__).resolve().parents[1] / "ner-site" / "tools" / "claude_check.py"
OPUS, SONNET, GEMINI = "claude-opus-5-5", "claude-sonnet-5-5", "google/gemini-3.1-flash-lite"

HAIKU, LUNA = "claude-haiku-5-5", "gpt-6-luna"
# Sentinels in the shape the site accepts (20 to 512 printable ASCII, no spaces).
OWN_KEY = "sk-stub-own-key-0123456789abcdef"
KEY_ENV = "CHECK_TEST_OWN_KEY"

DEADLINE = "cut off by the request deadline"  # ner_site.select.DEADLINE_NOTE

Reply = tuple[int, dict[str, Any]]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ner_site_claude_check", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up by name
    spec.loader.exec_module(module)
    return module


check = _load()


def _pick(selector: str, status: str = "match") -> dict[str, Any]:
    return {"selector": selector, "status": status, "id": "X:1", "label": "x"}


def ok_response(
    *,
    model: str = OPUS,
    requested: str = OPUS,
    picks: tuple[dict[str, Any], ...] = (),
    unpicked: int = 0,
    fallbacks: tuple[dict[str, str], ...] = (),
    usage: dict[str, Any] | None = None,
    total_ms: int = 1500,
    provider: str = "site",
    credential: str = "site",
) -> Reply:
    entities = [{"id": f"e{i}", "pick": pick} for i, pick in enumerate(picks)]
    entities += [{"id": f"u{i}"} for i in range(unpicked)]
    return 200, {
        "ok": True,
        "model": model,
        "requested_model": requested,
        "selector": requested,
        "requested_selector": requested,
        "entities": entities,
        "fallbacks": list(fallbacks),
        "notices": [],
        "provider": provider,
        "credential_source": credential,
        "usage": usage
        or {
            "llm_calls": 1,
            "prompt_tokens": 900,
            "completion_tokens": 300,
            "embedding_requests": 1,
            "selector_calls": len(picks),
            "selector_prompt_tokens": 2000 if picks else None,
            "selector_completion_tokens": 100 if picks else None,
        },
        "timing_ms": {"extract": 1000, "map": 200, "select": 300, "total": total_ms},
    }


def error_response(status: int, code: str, message: str = "no") -> Reply:
    return status, {"ok": False, "error": {"code": code, "message": message}}


ADMIT_SONNET = {"stage": "admit", "from": OPUS, "to": SONNET, "reason": "opus_allowance_used"}


class Stub:
    """What the stub site answers and what it was sent."""

    def __init__(self) -> None:
        self.health: dict[str, Any] = {
            "ok": True,
            "mode": "live",
            "default_model": OPUS,
            "default_selector": OPUS,
            "passcode_required": False,
            "claude": {"status": "ok"},
            "visitor": {
                "limits": "on",
                "tier": "opus",
                "opus_left": 7,
                "sonnet_left": 20,
                "resets_in_s": 1380,
                "opus_left_today": 17,
                "sonnet_left_today": 40,
                "own_left": 30,
                "own_left_today": 60,
                "day_resets_in_s": 30180,
                "window": "hour",
            },
            "own_keys": {
                "enabled": True,
                "providers": [
                    {"id": "anthropic", "name": "Anthropic", "models": [OPUS, HAIKU]},
                    {"id": "openai", "name": "OpenAI", "models": [LUNA]},
                ],
            },
        }
        self.answer: Callable[[dict[str, Any]], Reply] = lambda body: ok_response()
        # What the limiter counted: merged into ``visitor`` once a POST has arrived, so
        # that the health read after a burst shows the balances the burst left.
        self.spent: dict[str, Any] = {}
        self.health_status = 200
        self.reply_headers: dict[str, str] = {}
        self.requests: list[tuple[str, str, dict[str, str], dict[str, Any] | None]] = []
        self.lock = threading.Lock()
        self.url = ""


@pytest.fixture
def stub() -> Iterator[Stub]:
    state = Stub()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _reply(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for name, value in state.reply_headers.items() if self.command == "POST" else ():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        def _record(self, body: dict[str, Any] | None) -> None:
            headers = {name.lower(): value for name, value in self.headers.items()}
            with state.lock:
                state.requests.append((self.command, self.path, headers, body))

        def do_GET(self) -> None:
            self._record(None)
            if self.path == "/api/health":
                health = dict(state.health)
                posted = any(method == "POST" for method, _, _, _ in state.requests)
                if posted and state.spent and isinstance(health.get("visitor"), dict):
                    health["visitor"] = {**health["visitor"], **state.spent}
                self._reply(state.health_status, health)
            else:
                self._reply(404, {"ok": False, "error": {"code": "not_found", "message": "no"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self._record(body)
            with state.lock:
                status, payload = state.answer(body)
            self._reply(status, payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def write(tmp_path: Path, name: str, text: str) -> str:
    file = tmp_path / name
    file.write_text(text, encoding="utf-8")
    return str(file)


def posts(stub: Stub) -> list[tuple[dict[str, str], dict[str, Any]]]:
    return [(h, b) for method, _, h, b in stub.requests if method == "POST" and b is not None]


def test_health_blocks_come_first_and_each_file_gets_one_line(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.answer = lambda body: ok_response(
        model=SONNET,
        picks=(
            _pick(SONNET),
            _pick(SONNET, "review"),
            _pick(GEMINI),
            {"selector": OPUS, "status": "failed", "explanation": DEADLINE},
        ),
        unpicked=1,
        fallbacks=(
            {"stage": "extract", "from": OPUS, "to": SONNET, "reason": "rate_limited"},
            {"stage": "select", "from": SONNET, "to": GEMINI, "reason": "overloaded"},
        ),
    )
    code = check.main(["--url", stub.url + "/", write(tmp_path, "a.txt", "Aspirin and fever.")])
    out = capsys.readouterr().out.splitlines()

    assert code == 0
    assert out[0].startswith("health: mode live, default model claude-opus-5-5")
    assert out[1] == '  claude: {"status": "ok"}'
    assert out[2].startswith('  visitor: {"day_resets_in_s": 30180, "limits": "on", "opus_left": 7')
    assert out[3].startswith('  own_keys: {"enabled": true, "providers": [{"id": "anthropic"')
    assert out[4] == "running as provider site: the site's keys"
    (line,) = [row for row in out if row.startswith("a.txt")]
    assert "  200  " in line
    assert "provider site (credential site)" in line
    assert f"model {OPUS} -> {SONNET}" in line
    assert f"selector {OPUS}" in line
    assert "entities 5" in line
    assert f"picks 4 (match 2, review 1, failed 1; by {SONNET} 2, {GEMINI} 1, {OPUS} 1)" in line
    assert f"failed picks: {DEADLINE} 1" in line
    assert (
        f"fallbacks extract:{OPUS}>{SONNET}:rate_limited select:{SONNET}>{GEMINI}:overloaded"
        in line
    )
    assert "timing_ms extract=1000 map=200 select=300 total=1500" in line
    assert "tokens extract 900+300 in 1 calls, select 2000+100 in 4 calls" in line

    # Only the two paths, on the URL given; the POST is what the page would send.
    assert [(m, p) for m, p, _, _ in stub.requests] == [
        ("GET", "/api/health"),
        ("POST", "/api/ner"),
        ("GET", "/api/health"),
    ]
    ((headers, body),) = posts(stub)
    assert body == {"text": "Aspirin and fever."}
    assert headers["x-requested-with"] == "xwalk-ner"
    assert headers["content-type"] == "application/json"
    assert "x-passcode" not in headers
    assert "authorization" not in headers


def test_totals_sum_the_requests_and_name_the_slowest_and_the_largest(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def answer(body: dict[str, Any]) -> Reply:
        if "big" in body["text"]:
            return ok_response(picks=tuple(_pick(OPUS) for _ in range(5)), total_ms=42000)
        return ok_response(
            model=SONNET,
            fallbacks=(ADMIT_SONNET,),
            usage={
                "llm_calls": 2,
                "prompt_tokens": None,
                "completion_tokens": 50,
                "embedding_requests": 0,
            },
        )

    stub.answer = answer
    code = check.main(
        [
            "--url",
            stub.url,
            "--model",
            OPUS,
            "--selector",
            "jev",
            write(tmp_path, "small.txt", "small"),
            write(tmp_path, "big.txt", "big"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert (
        f"totals: 2 requests, 2 ok, 0 failed; 5 entities; extracted by {SONNET} 1, {OPUS} 1" in out
    )
    # An unreported count is left out of the sum and said to be missing, never added as 0.
    assert "tokens: extract 900 (+1 unreported) in + 350 out over 3 calls" in out
    assert "select 2000 in + 100 out over 5 calls" in out
    assert "step-downs by reason: opus_allowance_used 1" in out
    assert "most entities: big.txt (5 entities, " in out
    assert "42000 ms on the server" in out
    assert "slowest: " in out and "the site's deadline is 100 s" in out
    assert [body for _, body in posts(stub)] == [
        {"text": "small", "model": OPUS, "selector": "jev"},
        {"text": "big", "model": OPUS, "selector": "jev"},
    ]


def test_a_failed_request_is_named_and_gives_exit_code_1(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.answer = lambda body: (
        error_response(502, "provider_error", "the provider refused the request")
        if "bad" in body["text"]
        else ok_response()
    )
    code = check.main(
        [
            "--url",
            stub.url,
            write(tmp_path, "good.txt", "good"),
            write(tmp_path, "bad.txt", "bad"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 1
    (line,) = [row for row in out.splitlines() if row.startswith("bad.txt")]
    assert "  502  " in line
    assert "FAILED provider_error: the provider refused the request (sent as provider site)" in line
    assert "totals: 2 requests, 1 ok, 1 failed" in out


def test_a_site_that_does_not_answer_is_a_failure_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with socket.socket() as sock:  # a port nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    code = check.main(
        ["--url", f"http://127.0.0.1:{port}", "--timeout", "2", write(tmp_path, "a.txt", "a")]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "health: FAILED no response" in out
    assert "a.txt  ---  " in out and "FAILED no response" in out


@pytest.mark.parametrize(
    ("reply", "named"),
    [
        (b"this is not HTTP\r\n\r\n", "BadStatusLine"),
        (b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n{", "IncompleteRead"),
    ],
)
def test_a_broken_http_answer_is_a_failed_request_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], reply: bytes, named: str
) -> None:
    """``http.client`` raises these outside ``OSError``; they must not end the run."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    stop = threading.Event()

    def serve() -> None:
        listener.settimeout(0.2)
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except OSError:
                continue
            with connection:
                try:
                    connection.settimeout(2)
                    connection.recv(65536)
                    connection.settimeout(0.1)  # a POST's body may come separately
                    try:
                        while connection.recv(65536):
                            pass
                    except TimeoutError:
                        pass
                    connection.sendall(reply)
                    connection.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        files = [write(tmp_path, "a.txt", "a"), write(tmp_path, "b.txt", "b")]
        code = check.main(["--url", url, "--timeout", "5", *files])
    finally:
        stop.set()
        thread.join(timeout=5)
        listener.close()
    out = capsys.readouterr().out

    assert code == 1
    assert f"health: FAILED no response ({named}" in out
    # Each file is still asked and gets its own line.
    for name in ("a.txt", "b.txt"):
        (line,) = [row for row in out.splitlines() if row.startswith(name)]
        assert f"FAILED no response ({named}" in line
    assert "totals: 2 requests, 0 ok, 2 failed" in out


def test_a_passcode_the_site_echoes_is_not_printed(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "stub-passcode-value"
    monkeypatch.setenv("CHECK_TEST_PASSCODE", secret)
    stub.health["claude"] = {"status": "unavailable", "reason": f"saw {secret}"}
    echoing = ok_response(
        picks=({"selector": OPUS, "status": "failed", "explanation": f"upstream said {secret}"},),
        fallbacks=({"stage": "extract", "from": OPUS, "to": SONNET, "reason": secret},),
    )
    stub.answer = lambda body: (
        error_response(401, "passcode_invalid", f"{secret} is not correct")
        if "bad" in body["text"]
        else echoing
    )
    argv = ["--url", stub.url, "--passcode-env", "CHECK_TEST_PASSCODE"]
    files = [write(tmp_path, "good.txt", "good"), write(tmp_path, "bad.txt", "bad")]
    assert check.main([*argv, *files]) == 1
    assert check.main([*argv, "--burst", "2", files[1]]) == 1
    captured = capsys.readouterr()

    assert secret not in captured.out + captured.err
    assert "passcode_invalid: [redacted] is not correct" in captured.out
    assert "failed picks: upstream said [redacted] 1" in captured.out
    assert '"reason": "saw [redacted]"' in captured.out
    # The next run in this process, without a passcode, redacts nothing.
    stub.answer = lambda body: error_response(502, "provider_error", f"{secret} again")
    check.main(["--url", stub.url, files[0]])
    assert f"{secret} again" in capsys.readouterr().out


def test_the_passcode_comes_from_the_named_environment_variable(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHECK_TEST_PASSCODE", "stub-passcode-value")
    file = write(tmp_path, "a.txt", "a")
    code = check.main(["--url", stub.url, "--passcode-env", "CHECK_TEST_PASSCODE", file])
    captured = capsys.readouterr()

    assert code == 0
    ((headers, _),) = posts(stub)
    assert headers["x-passcode"] == "stub-passcode-value"
    assert "stub-passcode-value" not in captured.out + captured.err
    # The health check needs no passcode and is not sent one.
    assert all("x-passcode" not in h for method, _, h, _ in stub.requests if method == "GET")

    # A variable that is not set is a usage error before any request is made.
    stub.requests.clear()
    monkeypatch.delenv("CHECK_TEST_PASSCODE")
    with pytest.raises(SystemExit) as raised:
        check.main(["--url", stub.url, "--passcode-env", "CHECK_TEST_PASSCODE", file])
    assert raised.value.code == 2
    assert stub.requests == []


def test_sequential_delay_waits_between_requests_only(
    stub: Stub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(check.time, "sleep", slept.append)
    files = [write(tmp_path, f"{n}.txt", n) for n in ("a", "b", "c")]
    assert check.main(["--url", stub.url, "--sequential-delay", "7.5", *files]) == 0
    assert slept == [7.5, 7.5]


def _scripted(replies: list[Reply]) -> Callable[[dict[str, Any]], Reply]:
    remaining = list(replies)
    return lambda body: remaining.pop(0)


def test_burst_counts_where_the_parallel_requests_started(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.health["visitor"].update(opus_left=2, sonnet_left=3)
    stub.spent = {"opus_left": 0, "sonnet_left": 1, "opus_left_today": 15, "sonnet_left_today": 38}
    stub.answer = _scripted(
        [
            ok_response(),
            ok_response(),
            ok_response(model=SONNET, fallbacks=(ADMIT_SONNET,)),
            ok_response(model=SONNET, fallbacks=(ADMIT_SONNET,)),
            error_response(429, "visitor_limit_reached", HOUR_USED),
            error_response(429, "visitor_limit_reached", HOUR_USED),
            error_response(429, "visitor_limit_reached", DAY_USED),
        ]
    )
    code = check.main(
        [
            "--url",
            stub.url,
            "--burst",
            "7",
            write(tmp_path, "first.txt", "first"),
            write(tmp_path, "second.txt", "second"),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert (
        "burst: 7 parallel requests of first.txt as provider site: started on Opus 2, "
        "on Sonnet 2, on another model 0, on the own key 0, 429 hour used 2, "
        "429 day used 1, 429 own-key limit 0, 429 too many at once 0, "
        "503 limits_unavailable 0, not counted 0, failed 0"
    ) in out
    assert (
        "within the allowance health reported before the burst "
        "(opus_left 2, sonnet_left 3, opus_left_today 17, sonnet_left_today 40)"
    ) in out
    assert "and counted: opus_left 2 -> 0, sonnet_left 3 -> 1" in out
    assert [body for _, body in posts(stub)] == [{"text": "first"}] * 7


# The messages of ner_site.http._admit, with the own-key clause of a site that offers them.
HOUR_USED = "you have used this hour's requests; try again in 23 minutes, or use your own API key"
DAY_USED = "you have used today's requests; they reset at midnight UTC, or use your own API key"
OWN_USED = "too many requests with your own key; try again in 23 minutes"
CANNOT_COUNT = (
    "the site cannot count requests right now; try again shortly, or use your own API key"
)
UNCOUNTED = {"stage": "admit", "from": OPUS, "to": SONNET, "reason": "limits_unavailable"}
TOO_MANY = "too many requests at once; try again in a few seconds"  # ner_site.http._admit


def test_burst_counts_a_contended_lock_apart_and_does_not_fail_on_it(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tail of a large burst waits too long for the visitor's lock and is refused
    with 429: the limiter doing its work, not the hour being used and not a failure."""
    stub.health["visitor"].update(opus_left=10, sonnet_left=20)
    stub.spent = {"opus_left": 7}
    stub.answer = _scripted(
        [ok_response()] * 3
        + [error_response(429, "visitor_limit_reached", TOO_MANY)] * 4
        + [error_response(429, "visitor_limit_reached", HOUR_USED)]
    )
    code = check.main(["--url", stub.url, "--burst", "8", write(tmp_path, "a.txt", "a")])
    out = capsys.readouterr().out

    assert code == 0
    assert "started on Opus 3, on Sonnet 0" in out
    assert "429 hour used 1, 429 day used 0, 429 own-key limit 0, 429 too many at once 4" in out
    assert "503 limits_unavailable 0, not counted 0, failed 0" in out
    assert "within the allowance health reported before the burst" in out
    assert "refused because too many arrived at once" in out


def test_burst_fails_when_the_site_could_not_count_requests(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What a limiter that fails under concurrency looks like (Neon's batch endpoint not
    at READ COMMITTED, SQLSTATE 40001): one start on Opus, the rest refused with 503.
    Nothing is over the allowance, and that is exactly why it must not pass. One 503 is
    enough: the site ran nothing uncounted, but it did not serve the visitor either."""
    stub.health["visitor"].update(opus_left=10, sonnet_left=20)
    stub.answer = _scripted(
        [ok_response()] * 3 + [error_response(503, "limits_unavailable", CANNOT_COUNT)]
    )
    code = check.main(["--url", stub.url, "--burst", "4", write(tmp_path, "a.txt", "a")])
    out = capsys.readouterr().out

    assert code == 1
    assert "started on Opus 3, on Sonnet 0" in out
    assert "503 limits_unavailable 1, not counted 0, failed 0" in out
    assert "within the allowance" not in out
    assert "THE LIMITER IS FAILING: 1 of 4 requests were refused with 503" in out
    assert "limits_unavailable: the site cannot count requests right now" in out
    assert "40001" in out and "`limits`" in out and "DATABASE_URL" in out


def test_burst_fails_on_a_site_that_still_runs_requests_uncounted(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The code before 18 November 2026 ran such a request on Sonnet with a step-down
    note instead of refusing it. It is not a start on Sonnet, and one is a failure."""
    stub.health["visitor"].update(opus_left=10, sonnet_left=20)
    stub.answer = _scripted(
        [ok_response()] * 6 + [ok_response(model=SONNET, fallbacks=(UNCOUNTED,))]
    )
    code = check.main(["--url", stub.url, "--burst", "7", write(tmp_path, "a.txt", "a")])
    out = capsys.readouterr().out

    assert code == 1
    assert "started on Opus 6, on Sonnet 0" in out and "not counted 1, failed 0" in out
    assert "within the allowance" not in out
    assert "THE LIMITER IS FAILING: 1 of 7 requests ran uncounted" in out


@pytest.mark.parametrize(
    "visitor",
    [
        None,
        {"limits": "unavailable", "tier": "sonnet", "opus_left": 0, "sonnet_left": 20},
        {"limits": "off", "tier": "opus", "opus_left": 10, "sonnet_left": 20},
    ],
    ids=["absent", "unavailable", "off"],
)
def test_burst_fails_when_there_is_no_allowance_to_compare_with(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    visitor: dict[str, Any] | None,
) -> None:
    if visitor is None:
        del stub.health["visitor"]
    else:
        stub.health["visitor"] = visitor
    code = check.main(["--url", stub.url, "--burst", "2", write(tmp_path, "a.txt", "a")])
    out = capsys.readouterr().out

    assert code == 1
    # Nothing counted them, so they are starts on no tier.
    assert "started on Opus 0, on Sonnet 0" in out and "not counted 2, failed 0" in out
    assert "INCONCLUSIVE" in out and "within the allowance" not in out
    assert "the 2 requests that ran were not counted" in out


@pytest.mark.parametrize("limits", ["unavailable", "off"])
def test_a_sonnet_burst_nothing_counted_is_not_tallied_as_sonnet_starts(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str], limits: str
) -> None:
    """Asked for Sonnet, a request carries no step-down to tell a counted run from an
    uncounted one: only the limiter's state before the burst does."""
    stub.health["visitor"].update(limits=limits, tier="blocked", opus_left=0, sonnet_left=20)
    stub.answer = lambda body: ok_response(model=SONNET, requested=SONNET)
    code = check.main(
        ["--url", stub.url, "--model", SONNET, "--burst", "3", write(tmp_path, "a.txt", "a")]
    )
    out = capsys.readouterr().out

    assert code == 1
    assert "on Sonnet 0" in out and "not counted 3" in out
    assert "OVER THE ALLOWANCE" not in out and "within the allowance" not in out


def test_started_on_falls_back_to_the_selector_when_the_model_is_not_claude() -> None:
    """A request for an OpenRouter extraction with a Claude selector is admitted on the
    selector's tier (``ner_site.http._admit``)."""

    def data(model: str, selector: str, *fallbacks: dict[str, str]) -> dict[str, Any]:
        return {
            "model": model,
            "requested_model": model,
            "requested_selector": selector,
            "fallbacks": list(fallbacks),
        }

    assert check.started_on(data(GEMINI, OPUS)) == "opus"
    assert check.started_on(data(GEMINI, SONNET)) == "sonnet"
    assert check.started_on(data(GEMINI, OPUS, ADMIT_SONNET)) == "sonnet"
    assert check.started_on(data(GEMINI, "jev")) == "other"
    assert check.started_on(data(GEMINI, "retrieval")) == "other"
    assert check.started_on(data(OPUS, SONNET)) == "opus"  # the model decides when it is Claude
    assert check.started_on(data(SONNET, OPUS)) == "sonnet"
    # A response from before `requested_*`: what it has.
    assert check.started_on({"model": GEMINI, "selector": OPUS}) == "opus"
    assert check.started_on({"model": OPUS}) == "opus"


def test_burst_fails_when_more_started_on_opus_than_the_allowance_had_left(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.health["visitor"].update(opus_left=2, sonnet_left=20)
    stub.spent = {"opus_left": 0}
    # Opus admitted three times; one of them stepped down later, which is not an
    # admission on Sonnet.
    mid_request = {"stage": "extract", "from": OPUS, "to": SONNET, "reason": "rate_limited"}
    stub.answer = _scripted(
        [ok_response(), ok_response(), ok_response(model=SONNET, fallbacks=(mid_request,))]
    )
    code = check.main(["--url", stub.url, "--burst", "3", write(tmp_path, "a.txt", "a")])
    out = capsys.readouterr().out

    assert code == 1
    assert "started on Opus 3, on Sonnet 0" in out
    assert (
        "OVER THE ALLOWANCE: 3 requests started on opus, but opus_left was 2 before the burst"
    ) in out


def test_burst_fails_on_an_answer_that_is_neither_a_run_nor_the_visitor_limit(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.answer = _scripted([ok_response(), error_response(502, "provider_error", "upstream")])
    code = check.main(["--url", stub.url, "--burst", "2", write(tmp_path, "a.txt", "a")])
    out = capsys.readouterr().out

    assert code == 1
    assert "FAILED provider_error: upstream" in out
    assert "429 too many at once 0, 503 limits_unavailable 0, not counted 0, failed 1" in out


# --- a visitor's own key -----------------------------------------------------------------


def own_ok(provider: str = "anthropic", model: str = HAIKU, **kwargs: Any) -> Reply:
    return ok_response(
        model=model, requested=model, provider=provider, credential="visitor", **kwargs
    )


def own_argv(stub: Stub, provider: str = "anthropic") -> list[str]:
    return ["--url", stub.url, "--provider", provider, "--key-env", KEY_ENV]


def test_an_own_key_goes_out_as_bearer_on_the_post_only(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    stub.answer = lambda body: own_ok()
    code = check.main([*own_argv(stub), "--model", HAIKU, write(tmp_path, "a.txt", "a")])
    captured = capsys.readouterr()

    assert code == 0
    ((headers, body),) = posts(stub)
    assert headers["authorization"] == f"Bearer {OWN_KEY}"
    assert body == {"text": "a", "model": HAIKU, "provider": "anthropic"}
    # Health is asked before and after, and never sees the key.
    gets = [h for method, _, h, _ in stub.requests if method == "GET"]
    assert len(gets) == 2 and all("authorization" not in h for h in gets)
    assert OWN_KEY not in captured.out + captured.err
    assert (
        f"running as provider anthropic: your own key from the environment variable {KEY_ENV}, "
        "which is billed for the model calls"
    ) in captured.out
    (line,) = [row for row in captured.out.splitlines() if row.startswith("a.txt")]
    assert "provider anthropic (credential visitor)" in line
    assert "totals: 1 requests, 1 ok, 0 failed" in captured.out


def test_an_own_key_the_site_echoes_is_not_printed(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    monkeypatch.setenv("CHECK_TEST_PASSCODE", "stub-passcode-value")
    stub.health["claude"] = {"status": "unavailable", "reason": f"saw {OWN_KEY}"}
    echoing = own_ok(
        model=f"model-{OWN_KEY}",
        picks=({"selector": HAIKU, "status": "failed", "explanation": f"said {OWN_KEY}"},),
        fallbacks=({"stage": "extract", "from": HAIKU, "to": OWN_KEY, "reason": OWN_KEY},),
    )
    stub.answer = lambda body: (
        error_response(401, "provider_key_refused", f"Anthropic refused {OWN_KEY}")
        if "bad" in body["text"]
        else echoing
    )
    argv = [*own_argv(stub), "--passcode-env", "CHECK_TEST_PASSCODE"]
    files = [write(tmp_path, "good.txt", "good"), write(tmp_path, "bad.txt", "bad")]
    assert check.main([*argv, *files]) == 1
    assert check.main([*argv, "--burst", "2", files[1]]) == 1
    captured = capsys.readouterr()

    assert OWN_KEY not in captured.out + captured.err
    assert "stub-passcode-value" not in captured.out + captured.err
    assert "provider_key_refused: Anthropic refused [redacted] (sent as provider anthropic)" in (
        captured.out
    )
    assert "failed picks: said [redacted] 1" in captured.out
    assert '"reason": "saw [redacted]"' in captured.out
    assert all(h["x-passcode"] == "stub-passcode-value" for h, _ in posts(stub))
    # The next run in this process, on the site's keys, sends no key and redacts nothing.
    stub.requests.clear()
    stub.answer = lambda body: error_response(502, "provider_error", f"{OWN_KEY} again")
    check.main(["--url", stub.url, files[0]])
    assert f"{OWN_KEY} again" in capsys.readouterr().out
    assert all("authorization" not in h for h, _ in posts(stub))


@pytest.mark.parametrize(
    ("argv", "value", "said"),
    [
        (["--provider", "openai"], OWN_KEY, "--provider openai needs --key-env"),
        (["--key-env", KEY_ENV], OWN_KEY, "--key-env needs --provider"),
        (["--provider", "site", "--key-env", KEY_ENV], OWN_KEY, "--key-env needs --provider"),
        (["--provider", "openai", "--key-env", KEY_ENV], None, f"{KEY_ENV} is not set"),
        (["--provider", "openai", "--key-env", KEY_ENV], "", f"{KEY_ENV} is not set"),
        (["--provider", "openai", "--key-env", KEY_ENV], "too-short", "does not hold an API key"),
        (
            ["--provider", "openai", "--key-env", KEY_ENV],
            "sk-with a space-0123456789abcdef",
            "does not hold an API key",
        ),
        (
            ["--provider", "openai", "--key-env", KEY_ENV],
            "sk-not-ascii-\u017e-0123456789abcdef",
            "does not hold an API key",
        ),
        (["--provider", "chatgpt", "--key-env", KEY_ENV], OWN_KEY, "invalid choice"),
    ],
)
def test_own_key_usage_errors_exit_2_before_any_request(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    value: str | None,
    said: str,
) -> None:
    if value is None:
        monkeypatch.delenv(KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(KEY_ENV, value)
    with pytest.raises(SystemExit) as raised:
        check.main(["--url", stub.url, *argv, write(tmp_path, "a.txt", "a")])
    captured = capsys.readouterr()

    assert raised.value.code == 2
    assert said in " ".join(captured.err.split())
    assert stub.requests == []
    if value:  # a value that was refused is not shown either
        assert value not in captured.out + captured.err


@pytest.mark.parametrize("url", ["http://example.invalid", "http://192.0.2.1:8765"])
def test_an_own_key_is_not_sent_over_plain_http_to_another_machine(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    opened: list[object] = []
    monkeypatch.setattr(check._OPENER, "open", lambda *a, **k: opened.append(a))
    with pytest.raises(SystemExit) as raised:
        check.main(
            ["--url", url, "--provider", "openai", "--key-env", KEY_ENV, write(tmp_path, "a", "a")]
        )
    assert raised.value.code == 2
    assert "--url must be https" in capsys.readouterr().err
    assert opened == []


@pytest.mark.parametrize("own_keys", [None, {"enabled": False, "providers": []}, "listed"])
def test_an_own_key_is_not_sent_to_a_site_that_does_not_offer_the_provider(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    own_keys: object,
) -> None:
    """Own keys off (``NER_SITE_OWN_PROVIDERS=``), an API from before them, or this
    provider not enabled: the key stays here."""
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    if own_keys is None:
        del stub.health["own_keys"]
    elif own_keys != "listed":  # the stub lists anthropic and openai, not openrouter
        stub.health["own_keys"] = own_keys
    file = write(tmp_path, "a.txt", "a")
    assert check.main([*own_argv(stub, "openrouter"), file]) == 1
    assert check.main([*own_argv(stub, "openrouter"), "--burst", "3", file]) == 1
    captured = capsys.readouterr()

    assert "does not offer own keys for openrouter" in captured.out
    assert "nothing was sent" in captured.out
    assert [(m, p) for m, p, _, _ in stub.requests] == [("GET", "/api/health")] * 2
    assert OWN_KEY not in captured.out + captured.err


@pytest.mark.parametrize(
    ("provider", "reply", "said"),
    [
        # An own-key request that the site ran on its own keys: the operator paid.
        ("openai", ok_response(), "the response says provider site, credential site"),
        ("openai", own_ok("anthropic"), "the response says provider anthropic, credential visitor"),
        # The reverse: a site request reported as a visitor's.
        ("site", own_ok("openai", LUNA), "the response says provider openai, credential visitor"),
    ],
)
def test_a_run_on_another_credential_than_asked_for_fails(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    reply: Reply,
    said: str,
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    stub.answer = lambda body: reply
    argv = ["--url", stub.url] if provider == "site" else own_argv(stub, provider)
    file = write(tmp_path, "a.txt", "a")
    assert check.main([*argv, file]) == 1
    out = capsys.readouterr().out
    assert f"FAILED wrong credential: asked for provider {provider}" in out and said in out
    assert "totals: 1 requests, 0 ok, 1 failed" in out

    assert check.main([*argv, "--burst", "2", file]) == 1
    out = capsys.readouterr().out
    assert "on the own key 0" in out and "started on Opus 0" in out and "failed 2" in out


def test_a_site_response_from_before_the_credential_fields_is_not_a_wrong_credential(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, old = ok_response()
    del old["provider"], old["credential_source"]
    stub.answer = lambda body: (status, old)
    assert check.main(["--url", stub.url, write(tmp_path, "a.txt", "a")]) == 0
    assert "provider not reported (credential not reported)" in capsys.readouterr().out


def test_an_own_key_burst_is_compared_with_the_own_key_balance(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    # No site allowance left at all: an own key is outside it. The day binds the own one.
    stub.health["visitor"].update(
        tier="blocked", opus_left=0, sonnet_left=0, own_left=3, own_left_today=3, window="day"
    )
    stub.spent = {"own_left": 0, "own_left_today": 0}
    stub.answer = _scripted(
        [own_ok()] * 3 + [error_response(429, "visitor_limit_reached", OWN_USED)] * 2
    )
    file = write(tmp_path, "a.txt", "a")
    assert check.main([*own_argv(stub), "--burst", "5", file]) == 0
    out = capsys.readouterr().out

    assert (
        "burst: 5 parallel requests of a.txt as provider anthropic: started on Opus 0, "
        "on Sonnet 0, on another model 0, on the own key 3, 429 hour used 0, "
        "429 day used 0, 429 own-key limit 2, 429 too many at once 0, "
        "503 limits_unavailable 0, not counted 0, failed 0"
    ) in out
    assert (
        "within the allowance health reported before the burst (own_left 3, own_left_today 3)"
    ) in out
    assert "and counted: own_left 3 -> 0" in out
    assert all(h["authorization"] == f"Bearer {OWN_KEY}" for h, _ in posts(stub))
    assert [b for _, b in posts(stub)] == [{"text": "a", "provider": "anthropic"}] * 5
    assert OWN_KEY not in out

    # One more start than the balance had left.
    stub.requests.clear()
    stub.answer = lambda body: own_ok()
    assert check.main([*own_argv(stub), "--burst", "4", file]) == 1
    out = capsys.readouterr().out
    assert "OVER THE ALLOWANCE: 4 requests started on own, but own_left was 3" in out
    assert "within the allowance" not in out


def test_an_own_key_burst_the_limiter_cannot_count_is_not_counted(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An own-key request runs when the limiter cannot be asked, uncounted."""
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    stub.health["visitor"] = {"limits": "unavailable", "tier": "blocked"}
    stub.answer = lambda body: own_ok()
    assert check.main([*own_argv(stub), "--burst", "2", write(tmp_path, "a.txt", "a")]) == 1
    out = capsys.readouterr().out

    assert "on the own key 0" in out and "not counted 2, failed 0" in out
    assert "INCONCLUSIVE" in out and "the 2 requests that ran were not counted" in out


def test_an_own_key_burst_without_a_visitor_block_is_inconclusive_not_uncounted(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health has a ``visitor`` block only where Claude is offered on the site's keys.
    An OpenRouter-only deployment with a database counts own-key requests all the same:
    the tool cannot see it, and must neither call the starts uncounted nor pass."""
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    del stub.health["visitor"]
    stub.answer = lambda body: own_ok()
    assert check.main([*own_argv(stub), "--burst", "2", write(tmp_path, "a.txt", "a")]) != 0
    out = capsys.readouterr().out

    assert "on the own key 2" in out and "not counted 0, failed 0" in out
    assert "INCONCLUSIVE" in out and "no `visitor` block" in out
    assert "only where Claude is offered on the site's keys" in out
    assert "were not counted" not in out and "within the allowance" not in out


@pytest.mark.parametrize("via", ["site", "own"])
def test_a_burst_whose_balances_did_not_move_was_not_counted(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    via: str,
) -> None:
    """Every request ran and nothing is over the allowance health reported: exactly what
    a limiter that counts nothing looks like. The balances after the burst decide."""
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    stub.answer = lambda body: own_ok() if via == "own" else ok_response()
    argv = own_argv(stub) if via == "own" else ["--url", stub.url]
    assert check.main([*argv, "--burst", "2", write(tmp_path, "a.txt", "a")]) != 0
    out = capsys.readouterr().out

    tier, key = ("own", "own_left") if via == "own" else ("opus", "opus_left")
    left = stub.health["visitor"][key]
    assert (
        f"NOT COUNTED: 2 requests started on {tier}, but {key} fell by 0 (from {left} to {left})"
    ) in out
    assert "within the allowance" not in out


def test_one_own_key_run_with_unchanged_balances_does_not_pass_as_a_burst(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    stub.answer = lambda body: own_ok()
    assert check.main([*own_argv(stub), "--burst", "1", write(tmp_path, "a.txt", "a")]) != 0
    assert "NOT COUNTED: 1 requests started on own" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("spent", "said"),
    [
        # More went than this burst started: it was not alone.
        ({"opus_left": 2}, "opus_left fell by 5 for 2 starts"),
        # The clock hour ended between the two health reads.
        ({"opus_left": 5, "resets_in_s": 3590}, "the hour or the day ended during the burst"),
        ({"opus_left": 5, "day_resets_in_s": 86390}, "the hour or the day ended during the burst"),
        # The limiter was not there to be asked afterwards.
        ({"limits": "unavailable", "tier": "blocked"}, "after the burst"),
        ({"opus_left": None}, "after the burst"),
        ({"opus_left": True}, "after the burst"),
    ],
)
def test_a_burst_health_cannot_account_for_is_inconclusive(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    spent: dict[str, Any],
    said: str,
) -> None:
    stub.spent = spent
    assert check.main(["--url", stub.url, "--burst", "2", write(tmp_path, "a.txt", "a")]) != 0
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out and said in out
    assert "within the allowance" not in out and "NOT COUNTED" not in out


def test_a_burst_on_no_claude_model_has_no_balance_to_show_it(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.answer = lambda body: ok_response(model=GEMINI, requested=GEMINI)
    argv = ["--url", stub.url, "--model", GEMINI, "--selector", "retrieval", "--burst", "2"]
    assert check.main([*argv, write(tmp_path, "a.txt", "a")]) != 0
    out = capsys.readouterr().out
    assert "on another model 2" in out and "INCONCLUSIVE" in out
    assert "no balance for requests that use no Claude model" in out


def test_a_claude_unavailable_step_down_does_not_change_the_tier_that_was_charged(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """While Claude is blocked a request starts on the OpenRouter fallback, but the
    limiter had already counted it on the tier it asked for (``ner_site.http._admit``)."""
    blocked = {"stage": "admit", "from": OPUS, "to": GEMINI, "reason": "claude_unavailable"}
    after_used = {"stage": "admit", "from": SONNET, "to": GEMINI, "reason": "claude_unavailable"}
    data = {"model": GEMINI, "requested_model": OPUS, "requested_selector": "retrieval"}
    assert check.started_on({**data, "fallbacks": [blocked]}) == "opus"
    assert check.started_on({**data, "fallbacks": [ADMIT_SONNET, after_used]}) == "sonnet"

    stub.health["visitor"].update(opus_left=2, sonnet_left=20)
    stub.spent = {"opus_left": 0}
    stub.answer = lambda body: ok_response(model=GEMINI, fallbacks=(blocked,))
    assert check.main(["--url", stub.url, "--burst", "2", write(tmp_path, "a.txt", "a")]) == 0
    out = capsys.readouterr().out
    assert "started on Opus 2, on Sonnet 0, on another model 0" in out
    assert "and counted: opus_left 2 -> 0" in out


def test_an_own_key_burst_fails_when_the_limiter_stopped_answering_during_it(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Such requests run uncounted and their responses do not say so."""
    monkeypatch.setenv(KEY_ENV, OWN_KEY)

    def answer(body: dict[str, Any]) -> Reply:
        stub.health["visitor"] = {"limits": "unavailable", "tier": "blocked"}
        return own_ok()

    stub.answer = answer
    assert check.main([*own_argv(stub), "--burst", "2", write(tmp_path, "a.txt", "a")]) == 1
    out = capsys.readouterr().out

    assert "on the own key 2" in out
    assert "INCONCLUSIVE" in out and "within the allowance" not in out
    assert "An own-key request runs uncounted when the limiter cannot be asked" in out


def test_the_tool_follows_no_redirect(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(308)
            self.send_header("Location", stub.url + self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        code = check.main(["--url", url, write(tmp_path, "a.txt", "a")])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    out = capsys.readouterr().out

    assert code == 1
    assert "a.txt  308  " in out
    assert stub.requests == []  # the redirect's target was never asked


@pytest.mark.parametrize(
    "argv",
    [
        ["--url", "ftp://example.invalid", "FILE"],
        ["--url", "https://user:secret@example.invalid", "FILE"],
        ["--url", "http://127.0.0.1:1", "--burst", "101", "FILE"],
        ["--url", "http://127.0.0.1:1", "--sequential-delay", "-1", "FILE"],
        ["--url", "http://127.0.0.1:1", "MISSING"],
    ],
)
def test_usage_errors_exit_2_before_any_request(argv: list[str], tmp_path: Path) -> None:
    file = write(tmp_path, "a.txt", "a")
    resolved = [
        file if arg == "FILE" else str(tmp_path / "none.txt") if arg == "MISSING" else arg
        for arg in argv
    ]
    with pytest.raises(SystemExit) as raised:
        check.main(resolved)
    assert raised.value.code == 2


def test_help_says_what_a_run_costs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        check.main(["--help"])
    assert raised.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "counts against your hourly visitor allowance" in out
    assert "spends the balance of the operator's key" in out
    assert "--sequential-delay SECONDS" in out and "--burst N" in out
    assert "a run with an own key bills that key" in out
    assert "--provider {site,anthropic,openai,openrouter}" in out and "--key-env NAME" in out


# --- the tool's own handling of the secrets (final review, B4) ---------------------------

# What a secret may look like: a key with a quote and a backslash is 20 to 512 printable
# ASCII characters; a passcode may also hold spaces.
QUOTED_KEY = 'sk-own-"quoted"-\\back\\slash-0123456789'
QUOTED_PASSCODE = 'pass "word" with \\ in it'


def spellings(secret: str) -> list[str]:
    """The secret as it is, as JSON writes it, as ``repr`` writes it, percent-encoded."""
    from urllib.parse import quote

    return [secret, json.dumps(secret)[1:-1], repr(secret)[1:-1], quote(secret, safe="")]


def assert_unprinted(captured: Any, *secrets: str) -> None:
    for secret in secrets:
        for spelling in spellings(secret):
            assert spelling not in captured.out + captured.err


def test_a_loopback_url_is_asked_directly_whatever_proxy_is_configured(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``urllib`` sends an http request for 127.0.0.1 to the proxy the environment
    names: the key and the passcode would go to another machine in clear text although
    the URL says this one."""
    seen: list[str] = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _any(self) -> None:
            seen.append(f"{self.command} {self.path} {dict(self.headers)}")
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            raw = json.dumps(stub.health).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = _any

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{proxy.server_address[1]}"
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY", "https_proxy"):
        monkeypatch.setenv(name, address)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    monkeypatch.setenv("CHECK_TEST_PASSCODE", "stub-passcode-value")
    stub.answer = lambda body: own_ok()
    try:
        fresh = _load()  # the tool reads the environment's proxies when it is loaded
        argv = [*own_argv(stub), "--passcode-env", "CHECK_TEST_PASSCODE"]
        code = fresh.main([*argv, write(tmp_path, "a.txt", "a")])
    finally:
        proxy.shutdown()
        proxy.server_close()
        thread.join(timeout=5)
        sys.modules[check.__name__] = check
    capsys.readouterr()

    assert seen == []  # nothing went to the proxy, with or without a secret
    assert code == 0
    assert [(m, p) for m, p, _, _ in stub.requests] == [
        ("GET", "/api/health"),
        ("POST", "/api/ner"),
        ("GET", "/api/health"),
    ]


@pytest.mark.parametrize("url", ["http://example.invalid", "http://192.0.2.1:8765"])
def test_a_passcode_is_not_sent_over_plain_http_to_another_machine(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    monkeypatch.setenv("CHECK_TEST_PASSCODE", "stub-passcode-value")
    opened: list[object] = []
    for opener in (check._OPENER, check._DIRECT):
        monkeypatch.setattr(opener, "open", lambda *a, **k: opened.append(a))
    with pytest.raises(SystemExit) as raised:
        check.main(
            ["--url", url, "--passcode-env", "CHECK_TEST_PASSCODE", write(tmp_path, "a", "a")]
        )
    assert raised.value.code == 2
    assert "--url must be https" in capsys.readouterr().err
    assert opened == []


@pytest.mark.parametrize(
    ("status", "own_keys"),
    [
        # A 503 (a misconfigured site, an error page of a proxy) whose body lists it.
        (503, {"enabled": True, "providers": [{"id": "anthropic"}]}),
        (500, {"enabled": True, "providers": [{"id": "anthropic"}]}),
        # Listed, but own keys are not on.
        (200, {"enabled": False, "providers": [{"id": "anthropic"}]}),
        (200, {"providers": [{"id": "anthropic"}]}),
        (200, {"enabled": "yes", "providers": [{"id": "anthropic"}]}),
    ],
)
def test_an_own_key_is_sent_only_after_a_good_health_that_says_own_keys_are_on(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    own_keys: dict[str, Any],
) -> None:
    monkeypatch.setenv(KEY_ENV, OWN_KEY)
    stub.health_status = status
    stub.health["own_keys"] = own_keys
    file = write(tmp_path, "a.txt", "a")
    assert check.main([*own_argv(stub), file]) == 1
    assert check.main([*own_argv(stub), "--burst", "2", file]) == 1
    captured = capsys.readouterr()

    assert "nothing was sent" in captured.out
    assert [(m, p) for m, p, _, _ in stub.requests] == [("GET", "/api/health")] * 2
    assert_unprinted(captured, OWN_KEY)


def test_secrets_with_quotes_and_backslashes_are_not_printed_in_any_spelling(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The health blocks are printed as JSON, which writes a quote as ``\\"``: a search
    for the secret in the encoded text finds nothing to replace."""
    assert check._KEY.fullmatch(QUOTED_KEY)
    monkeypatch.setenv(KEY_ENV, QUOTED_KEY)
    monkeypatch.setenv("CHECK_TEST_PASSCODE", QUOTED_PASSCODE)
    both = f"{QUOTED_KEY} and {QUOTED_PASSCODE}"
    stub.health["claude"] = {"status": "unavailable", "reason": f"saw {both}", QUOTED_KEY: 1}
    stub.health["visitor"]["note"] = [both, {"nested": QUOTED_PASSCODE}]
    stub.health["own_keys"]["providers"][0]["name"] = QUOTED_KEY
    echoing = own_ok(
        model=f"model-{QUOTED_KEY}",
        picks=({"selector": QUOTED_KEY, "status": "failed", "explanation": f"said {both}"},),
        fallbacks=({"stage": "extract", "from": HAIKU, "to": QUOTED_KEY, "reason": both},),
    )
    echoing[1]["timing_ms"][QUOTED_PASSCODE] = 1
    stub.answer = lambda body: (
        error_response(401, "provider_key_refused", f"refused {both}")
        if "bad" in body["text"]
        else echoing
    )
    argv = [*own_argv(stub), "--passcode-env", "CHECK_TEST_PASSCODE"]
    files = [write(tmp_path, "good.txt", "good"), write(tmp_path, "bad.txt", "bad")]
    assert check.main([*argv, *files]) == 1
    assert check.main([*argv, "--burst", "2", files[1]]) == 1
    captured = capsys.readouterr()

    assert_unprinted(captured, QUOTED_KEY, QUOTED_PASSCODE)
    assert '"reason": "saw [redacted] and [redacted]"' in captured.out
    assert "provider_key_refused: refused [redacted] and [redacted]" in captured.out
    ((headers, _), *_) = posts(stub)
    assert headers["authorization"] == f"Bearer {QUOTED_KEY}"
    assert headers["x-passcode"] == QUOTED_PASSCODE


@pytest.mark.parametrize(
    "passcode",
    [
        "first line\nSECOND-LINE-of-the-passcode",
        "carriage\rRETURN-in-the-passcode",
        "tab\tin-the-passcode-VALUE",
        "not-ascii-\u017e-in-the-passcode-VALUE",
        " leading-space-in-the-passcode-VALUE",
        "trailing-space-in-the-passcode-VALUE ",
    ],
)
def test_a_passcode_that_cannot_be_a_header_is_a_usage_error_that_shows_no_part_of_it(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    passcode: str,
) -> None:
    """``http.client`` refuses such a value with an error that quotes it, which used to
    end the run in a traceback with the passcode in it."""
    monkeypatch.setenv("CHECK_TEST_PASSCODE", passcode)
    with pytest.raises(SystemExit) as raised:
        check.main(
            ["--url", stub.url, "--passcode-env", "CHECK_TEST_PASSCODE", write(tmp_path, "a", "a")]
        )
    captured = capsys.readouterr()

    assert raised.value.code == 2
    assert "CHECK_TEST_PASSCODE" in captured.err and "printable ASCII" in captured.err
    assert stub.requests == []
    for part in (passcode, "SECOND-LINE", "RETURN", "VALUE", "in-the-passcode"):
        assert part not in captured.out + captured.err


def test_a_request_that_cannot_be_built_fails_in_fixed_words(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    """Should a value that cannot be a header ever get as far as the request, the
    library's error (which quotes it) is not what is printed."""
    answer = check.post_text(
        stub.url,
        "a",
        model=None,
        selector=None,
        passcode="first\nSECOND-LINE",
        timeout=5,
    )
    assert answer.status == 0 and answer.error == "no request was sent (ValueError)"
    assert "SECOND-LINE" not in check.line("a.txt", answer)
    assert stub.requests == []


ECHO_SAID = "CREDENTIAL ECHO"


def echo_cases() -> list[tuple[str, Any]]:
    """Where and how a site that mishandles a credential might send it back."""
    from urllib.parse import quote

    def body(text: str) -> Callable[[Stub], Reply]:
        return lambda stub: error_response(401, "provider_key_refused", text)

    def header(stub: Stub) -> Reply:
        stub.reply_headers = {"X-Upstream-Detail": f"key={QUOTED_KEY}"}
        return error_response(401, "provider_key_refused", "Anthropic refused this key.")

    def in_a_run(stub: Stub) -> Reply:
        reply = own_ok()
        reply[1]["warnings"] = [f"upstream said {QUOTED_KEY}"]
        return reply

    def as_a_field_name(stub: Stub) -> Reply:
        reply = own_ok()
        reply[1]["timing_ms"][QUOTED_KEY] = 1
        return reply

    return [
        ("in the message", body(f"Anthropic refused {QUOTED_KEY}")),
        ("percent-encoded", body("refused " + quote(QUOTED_KEY, safe=""))),
        ("escaped twice", body("refused " + json.dumps(QUOTED_KEY))),
        ("in a response header", header),
        ("in a field the tool does not print", in_a_run),
        ("as a field name", as_a_field_name),
    ]


@pytest.mark.parametrize("case", echo_cases(), ids=lambda case: case[0])
def test_a_response_that_echoes_the_key_fails_the_run_and_says_so(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[str, Any],
) -> None:
    """The wrong-key rollout check: what the operator reads is redacted, so a clean
    report cannot show that the response was clean. The tool has to say."""
    monkeypatch.setenv(KEY_ENV, QUOTED_KEY)
    reply = case[1](stub)
    stub.answer = lambda body: reply
    file = write(tmp_path, "a.txt", "a")
    assert check.main([*own_argv(stub), file]) == 1
    captured = capsys.readouterr()
    assert_unprinted(captured, QUOTED_KEY)
    (line,) = [row for row in captured.out.splitlines() if row.startswith("a.txt")]
    assert ECHO_SAID in line
    assert "totals: 1 requests, 0 ok, 1 failed" in captured.out
    assert "credential echo check: FAILED, 1 of 3 responses" in captured.out

    assert check.main([*own_argv(stub), "--burst", "2", file]) == 1
    captured = capsys.readouterr()
    assert_unprinted(captured, QUOTED_KEY)
    assert captured.out.count(ECHO_SAID) >= 2 and "failed 2" in captured.out


def test_a_refused_key_that_is_not_echoed_is_reported_as_clean(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, QUOTED_KEY)
    stub.answer = lambda body: error_response(
        401, "provider_key_refused", "Anthropic refused this key."
    )
    assert check.main([*own_argv(stub), write(tmp_path, "a.txt", "a")]) == 1  # the POST failed
    out = capsys.readouterr().out
    assert ECHO_SAID not in out
    assert "FAILED provider_key_refused: Anthropic refused this key." in out
    assert "credential echo check: none of the 3 responses contained" in out


def test_a_health_answer_that_echoes_the_passcode_fails_the_run(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHECK_TEST_PASSCODE", QUOTED_PASSCODE)
    stub.health["claude"] = {"status": "ok", "reason": f"saw {QUOTED_PASSCODE}"}
    argv = ["--url", stub.url, "--passcode-env", "CHECK_TEST_PASSCODE"]
    assert check.main([*argv, write(tmp_path, "a.txt", "a")]) == 1
    captured = capsys.readouterr()
    assert_unprinted(captured, QUOTED_PASSCODE)
    assert f"health: {ECHO_SAID}" in captured.out
    assert "credential echo check: FAILED, 2 of 3 responses" in captured.out


def test_a_run_without_a_secret_checks_for_no_echo(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check.main(["--url", stub.url, write(tmp_path, "a.txt", "a")]) == 0
    assert "credential echo check" not in capsys.readouterr().out


def test_a_short_passcode_is_not_mistaken_for_an_echo(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every answer holds ``"ok"``; a passcode that short cannot be looked for."""
    monkeypatch.setenv("CHECK_TEST_PASSCODE", "ok")
    argv = ["--url", stub.url, "--passcode-env", "CHECK_TEST_PASSCODE"]
    assert check.main([*argv, write(tmp_path, "a.txt", "a")]) == 0
    out = capsys.readouterr().out
    assert ECHO_SAID not in out and "credential echo check: not made" in out


def test_a_balance_that_rose_during_a_burst_is_inconclusive(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.spent = {"opus_left": 5, "sonnet_left": 25}
    assert check.main(["--url", stub.url, "--burst", "2", write(tmp_path, "a.txt", "a")]) != 0
    out = capsys.readouterr().out
    assert "INCONCLUSIVE: sonnet_left rose from 20 to 25" in out and "NOT COUNTED" not in out


# --- second review round ------------------------------------------------------------------


def lower_percent(text: str) -> str:
    """Every character percent-escaped with lower-case hex digits: valid, and not what
    ``urllib.parse.quote`` writes."""
    return "".join(f"%{ord(char):02x}" for char in text)


@pytest.mark.parametrize(
    "spelling",
    [
        lower_percent,
        lambda key: lower_percent(key[:8]) + key[8:],
        lambda key: lower_percent(lower_percent(key)),
        lambda key: key.replace("-", "%2d"),
    ],
    ids=["lowercase", "partly", "twice", "dashes-only"],
)
@pytest.mark.parametrize("where", ["error", "run", "health"])
def test_a_percent_escaped_echo_in_any_case_is_caught_and_not_printed(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    spelling: Callable[[str], str],
    where: str,
) -> None:
    monkeypatch.setenv(KEY_ENV, QUOTED_KEY)
    echo = spelling(QUOTED_KEY)
    reply = error_response(401, "provider_key_refused", f"refused {echo} by upstream")
    if where == "run":
        reply = own_ok(
            picks=({"selector": HAIKU, "status": "failed", "explanation": f"said {echo}"},)
        )
    elif where == "health":
        stub.health["claude"] = {"status": "ok", "reason": f"saw {echo}"}
        reply = own_ok()
    stub.answer = lambda body: reply
    assert check.main([*own_argv(stub), write(tmp_path, "a.txt", "a")]) == 1
    captured = capsys.readouterr()
    shown = captured.out + captured.err
    assert echo not in shown and QUOTED_KEY not in shown
    assert lower_percent(QUOTED_KEY[:8]) not in shown and "%2d" not in shown
    assert ECHO_SAID in shown and "credential echo check: FAILED" in shown


def test_a_burst_that_was_refused_entirely_certifies_nothing(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Five slots left, every request turned away at the lock, the balances unchanged:
    zero starts match a fall of zero, which shows nothing about counting."""
    stub.health["visitor"].update(opus_left=5, sonnet_left=20)
    stub.answer = lambda body: error_response(429, "visitor_limit_reached", TOO_MANY)
    assert check.main(["--url", stub.url, "--burst", "5", write(tmp_path, "a.txt", "a")]) != 0
    out = capsys.readouterr().out
    assert "started on Opus 0, on Sonnet 0" in out and "429 too many at once 5" in out
    assert "INCONCLUSIVE: no request of the burst started" in out
    assert "and counted" not in out and "within the allowance" not in out


@pytest.mark.parametrize("missing", ["day_resets_in_s", "resets_in_s"])
def test_health_snapshots_with_different_reset_fields_are_inconclusive_not_a_crash(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    def answer(body: dict[str, Any]) -> Reply:
        stub.health["visitor"].pop(missing, None)
        stub.health["visitor"]["opus_left"] = 5
        return ok_response()

    stub.answer = answer
    assert check.main(["--url", stub.url, "--burst", "2", write(tmp_path, "a.txt", "a")]) == 1
    out = capsys.readouterr().out
    assert "INCONCLUSIVE" in out and "and counted" not in out


# --- third review round: every layer of percent decoding ---------------------------------

# A key may hold a percent sign and what looks like an escape: all printable ASCII.
PERCENT_KEY = "sk-own%20key%2Dwith-escapes-0123456789"


def layers(text: str, times: int) -> str:
    for _ in range(times):
        text = lower_percent(text)
    return text


@pytest.mark.parametrize(
    ("key", "spelling"),
    [
        # Escaped once: the key shows after one decoding and is gone again after two,
        # because its own "%20" is then decoded as well.
        (PERCENT_KEY, lower_percent),
        (PERCENT_KEY, lambda key: key.replace("-", "%2d")),
        (PERCENT_KEY, lambda key: layers(key, 2)),
        # Deeper than the tool is willing to decode: no verdict is not a pass.
        (QUOTED_KEY, lambda key: layers(key, 5)),
        (QUOTED_KEY, lambda key: layers(key, 12)),
        (PERCENT_KEY, lambda key: layers(key, 9)),
    ],
    ids=["once", "dashes", "twice", "five-layers", "twelve-layers", "nine-layers"],
)
@pytest.mark.parametrize("where", ["error", "run", "health"])
def test_an_echo_is_found_at_every_layer_of_percent_decoding(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    spelling: Callable[[str], str],
    where: str,
) -> None:
    assert check._KEY.fullmatch(key)
    monkeypatch.setenv(KEY_ENV, key)
    echo = spelling(key)
    reply = error_response(401, "provider_key_refused", f"refused {echo} by upstream")
    if where == "run":
        reply = own_ok(
            picks=({"selector": HAIKU, "status": "failed", "explanation": f"said {echo}"},)
        )
    elif where == "health":
        stub.health["claude"] = {"status": "ok", "reason": f"saw {echo}"}
        reply = own_ok()
    stub.answer = lambda body: reply
    assert check.main([*own_argv(stub), write(tmp_path, "a.txt", "a")]) == 1
    captured = capsys.readouterr()
    shown = captured.out + captured.err
    assert echo not in shown and key not in shown
    assert echo[:24] not in shown and echo[-24:] not in shown
    assert ECHO_SAID in shown and "credential echo check: FAILED" in shown


def test_ordinary_percent_signs_are_not_an_echo(
    stub: Stub,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, PERCENT_KEY)
    stub.health["claude"] = {"status": "ok", "reason": "100% sure, 5%20 of the time, %zz"}
    stub.answer = lambda body: own_ok()
    assert check.main([*own_argv(stub), write(tmp_path, "a.txt", "a")]) == 0
    out = capsys.readouterr().out
    assert ECHO_SAID not in out and "100% sure, 5%20 of the time, %zz" in out
