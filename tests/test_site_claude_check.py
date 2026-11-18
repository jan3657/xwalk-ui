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
            },
        }
        self.answer: Callable[[dict[str, Any]], Reply] = lambda body: ok_response()
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
            self.end_headers()
            self.wfile.write(raw)

        def _record(self, body: dict[str, Any] | None) -> None:
            headers = {name.lower(): value for name, value in self.headers.items()}
            with state.lock:
                state.requests.append((self.command, self.path, headers, body))

        def do_GET(self) -> None:
            self._record(None)
            if self.path == "/api/health":
                self._reply(200, state.health)
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
    assert out[2].startswith('  visitor: {"limits": "on", "opus_left": 7')
    (line,) = [row for row in out if row.startswith("a.txt")]
    assert "  200  " in line
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
    assert "FAILED provider_error: the provider refused the request" in line
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
    stub.answer = _scripted(
        [
            ok_response(),
            ok_response(),
            ok_response(model=SONNET, fallbacks=(ADMIT_SONNET,)),
            ok_response(model=SONNET, fallbacks=(ADMIT_SONNET,)),
            # Not counted by the limiter: reported apart, outside both tiers.
            ok_response(
                model=SONNET,
                fallbacks=(
                    {
                        "stage": "admit",
                        "from": OPUS,
                        "to": SONNET,
                        "reason": "limits_unavailable",
                    },
                ),
            ),
            error_response(429, "visitor_limit_reached", "try again in 23 minutes"),
            error_response(429, "visitor_limit_reached", "try again in 23 minutes"),
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
        "burst: 7 parallel requests of first.txt: started on Opus 2, on Sonnet 2, "
        "on another model 0, 429 visitor_limit_reached 2, uncounted (limits_unavailable) 1, "
        "failed 0"
    ) in out
    assert "within the allowance health reported before the burst" in out
    assert "uncounted requests ran on Sonnet outside the allowance" in out
    assert [body for _, body in posts(stub)] == [{"text": "first"}] * 7


def test_burst_fails_when_more_started_on_opus_than_the_allowance_had_left(
    stub: Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.health["visitor"].update(opus_left=2, sonnet_left=20)
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
    assert "429 visitor_limit_reached 0, uncounted (limits_unavailable) 0, failed 1" in out


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
