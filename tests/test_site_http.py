"""ner_site.http.handle and the api/*.py Vercel shims (docs/ner-site/PLAN.md 3.1, 7).

Offline: FakeLLM, the offline stand-in and httpx.MockTransport only. BM25 is built in
tmp_path; dense always gets an explicit directory (tmp or a missing one), never the
committed ``ner-site/indexes``.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest
from ner_site import http as http_mod
from ner_site import indexes as indexes_mod
from ner_site.schema import HealthResponse, NerResponse
from ner_site.service import Settings, make_llm
from xwalk.llm.base import LLMFatalError, LLMRetryableError
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.llm.fake import FakeLLM
from xwalk.retrieval import embeddings_api

SENTINEL_KEY = "sk-or-v1-SENTINEL-c0ffee-4242"
PASSCODE = "correct horse battery"
SECRET_TEXT = "Quokkaline therapy for haloperidol"
SECRET_PROMPT = "PROMPTSENTINEL extract chemicals only"
PROJECT = Path(__file__).resolve().parents[1] / "ner-site"
EXAMPLE_TEXT = (
    "Clinic note 🩺: the patient has Alkaptonuria and was screened for "
    "Hydroxykynureninuria. She took haloperidol, later ketamine, and aspirin for a fever. "
    "Genotyping covered NAT2 and ADA. Her diet: hummus, garlic and cumin."
)
POST_HEADERS = {"content-type": "application/json", "x-requested-with": "xwalk-ner"}

Handled = tuple[int, dict[str, str], bytes]
# The real one: the conftest fixture swaps it for a refusal in every site test.
REAL_LIMITS_EXECUTOR = http_mod.make_limits_executor


@pytest.fixture(autouse=True)
def _no_key_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health asks OpenRouter and Anthropic about their keys; offline tests never reach
    the network (and never build an SDK client)."""
    monkeypatch.setattr(http_mod, "check_key", lambda settings: None)
    monkeypatch.setattr(http_mod, "check_claude_key", lambda settings: None)


def test_health_reports_what_the_key_check_found(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site.schema import RetrieverStatus

    refused = RetrieverStatus(status="unavailable", reason="OPENROUTER_API_KEY was refused")
    monkeypatch.setattr(http_mod, "check_key", lambda settings: refused)
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    _, _, body = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert json.loads(body)["llm"] == {
        "status": "unavailable",
        "reason": "OPENROUTER_API_KEY was refused",
    }


@pytest.fixture(autouse=True)
def _fresh_memo() -> Iterator[None]:
    indexes_mod.reset()
    http_mod.reset()
    yield
    indexes_mod.reset()
    http_mod.reset()


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One BM25 cache for the module: built once, reused warm (as on Vercel)."""
    return tmp_path_factory.mktemp("bm25-cache")


def _settings(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    # The default selector is the first that can run (Jev with only an OpenRouter key);
    # these tests are about requests without a selection step unless they say so.
    return Settings.from_env(
        {
            "NER_SITE_CACHE_DIR": str(cache_dir),
            "NER_SITE_DENSE_DIR": str(tmp_path / "no-dense"),
            "NER_SITE_SELECTOR": "retrieval",
            **env,
        }
    )


def _answer(*items: tuple[str, str, int, int]) -> str:
    return json.dumps(
        {"entities": [{"text": t, "type": ty, "start": s, "end": e} for t, ty, s, e in items]}
    )


def _post(
    settings: Settings,
    payload: object,
    *,
    headers: dict[str, str] | None = None,
    llm_factory: Callable[[Settings, str], Any] | None = None,
    raw: bytes | None = None,
) -> Handled:
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return http_mod.handle(
        "POST",
        "/api/ner",
        dict(POST_HEADERS if headers is None else headers),
        body,
        settings=settings,
        llm_factory=llm_factory,
    )


def _json(result: Handled) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result[2])
    return data


def _error(result: Handled) -> tuple[int, str]:
    data = _json(result)
    assert data["ok"] is False and set(data) == {"ok", "error"}
    assert set(data["error"]) == {"code", "message"} and data["error"]["message"]
    return result[0], data["error"]["code"]


def _fake(*script: Any) -> Callable[[Settings, str], FakeLLM]:
    llm = FakeLLM(list(script))
    return lambda settings, model: llm


def _never(settings: Settings, model: str) -> Any:
    pytest.fail("the LLM was created for a request that should have been refused")


def _assert_common_headers(headers: dict[str, str]) -> None:
    lowered = {k.lower(): v for k, v in headers.items()}
    assert lowered["cache-control"] == "no-store"
    assert lowered["content-type"].startswith("application/json")
    assert not any(k.startswith("access-control-") for k in lowered)


# --- health -------------------------------------------------------------------------


def test_health_shape(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    status, headers, body = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert status == 200
    _assert_common_headers(headers)
    health = HealthResponse.model_validate_json(body)
    assert health.mode == "live" and health.default_model == "google/gemini-3.1-flash-lite"
    assert health.models == ["google/gemini-3.1-flash-lite", "openai/gpt-6-luna"]
    assert health.limits == {
        "max_chars": 5000,
        "max_prompt_chars": 4000,
        "max_entities": 25,
        "top_k_max": 10,
    }
    assert health.passcode_required is False
    assert "chemical" in health.default_prompt
    assert [o.slug for o in health.ontologies] == [
        "chebi-sample",
        "ctd-disease-sample",
        "ncbi-gene-sample",
        "foodon-sample",
    ]
    assert all(o.count == 200 for o in health.ontologies)
    assert health.retrieval["bm25"].status == "ok"
    dense = health.retrieval["dense"]
    assert dense.status == "unavailable" and dense.reason and "no index at" in dense.reason
    assert _json((status, headers, body))["llm"] == {"status": "ok"}
    assert SENTINEL_KEY.encode() not in body


def test_health_without_a_key_still_answers_and_says_what_is_missing(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _settings(cache_dir, tmp_path)
    result = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert result[0] == 200
    data = _json(result)
    assert data["mode"] == "live"
    assert data["llm"]["status"] == "unavailable"
    assert "OPENROUTER_API_KEY" in data["llm"]["reason"]


def test_health_offline_reports_the_offline_model(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, NER_SITE_OFFLINE="1")
    health = HealthResponse.model_validate_json(
        http_mod.handle("GET", "/api/health", {}, b"", settings=s)[2]
    )
    assert health.mode == "offline"
    assert health.default_model == "offline" and health.models == ["offline"]


def test_health_offline_with_committed_indexes_and_no_key_is_dense_unavailable(
    cache_dir: Path, tmp_path: Path
) -> None:
    """A dense index exists (a fake one here), but no key: never `ok`, and the reason
    names the key."""
    index = tmp_path / "dense" / "chebi-sample" / "dense"
    index.mkdir(parents=True)
    (index / "xwalk_meta.json").write_text("{}")
    s = _settings(
        cache_dir, tmp_path, NER_SITE_OFFLINE="1", NER_SITE_DENSE_DIR=str(tmp_path / "dense")
    )
    dense = _json(http_mod.handle("GET", "/api/health", {}, b"", settings=s))["retrieval"]["dense"]
    assert dense["status"] == "unavailable" and "OPENROUTER_API_KEY is not set" in dense["reason"]


def test_health_reports_the_passcode_requirement_without_needing_it(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    result = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert result[0] == 200 and _json(result)["passcode_required"] is True
    assert PASSCODE.encode() not in result[2]


def test_head_on_health_is_allowed(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path)
    assert http_mod.handle("HEAD", "/api/health", {}, b"", settings=s)[0] == 200


# --- routing, methods, headers ------------------------------------------------------


@pytest.mark.parametrize("path", ["/api/nope", "/api", "/", "/api/ner/extra"])
def test_unknown_path_is_404(cache_dir: Path, tmp_path: Path, path: str) -> None:
    s = _settings(cache_dir, tmp_path)
    result = http_mod.handle("GET", path, {}, b"", settings=s)
    assert _error(result) == (404, "not_found")
    _assert_common_headers(result[1])


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/ner"),
        ("PUT", "/api/ner"),
        ("DELETE", "/api/ner"),
        ("POST", "/api/health"),
        ("OPTIONS", "/api/ner"),
        ("OPTIONS", "/api/health"),
    ],
)
def test_wrong_method_is_405(cache_dir: Path, tmp_path: Path, method: str, path: str) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = http_mod.handle(
        method,
        path,
        {"origin": "https://evil.example", "access-control-request-method": "POST"},
        b"",
        settings=s,
        llm_factory=_never,
    )
    assert _error(result) == (405, "method_not_allowed")
    _assert_common_headers(result[1])
    assert result[1]["Allow"] in {"POST", "GET, HEAD"}


@pytest.mark.parametrize(
    "headers",
    [
        {"content-type": "application/json"},
        {"content-type": "application/json", "x-requested-with": "XMLHttpRequest"},
        {"content-type": "application/json", "x-requested-with": ""},
    ],
)
def test_the_requested_with_header_is_required(
    cache_dir: Path, tmp_path: Path, headers: dict[str, str]
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, headers=headers, llm_factory=_never)
    assert _error(result) == (400, "bad_header")


def test_header_names_are_case_insensitive(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(
        s,
        {"text": "nothing"},
        headers={"Content-Type": "application/json", "X-Requested-With": "xwalk-ner"},
        llm_factory=_fake('{"entities": []}'),
    )
    assert result[0] == 200


# --- passcode -----------------------------------------------------------------------


def test_passcode_required_and_invalid(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    assert _error(_post(s, {"text": "a"}, llm_factory=_never)) == (401, "passcode_required")
    empty = {**POST_HEADERS, "x-passcode": ""}
    assert _error(_post(s, {"text": "a"}, headers=empty, llm_factory=_never)) == (
        401,
        "passcode_required",
    )
    wrong = {**POST_HEADERS, "x-passcode": "correct horse"}
    result = _post(s, {"text": "a"}, headers=wrong, llm_factory=_never)
    assert _error(result) == (401, "passcode_invalid")
    assert PASSCODE.encode() not in result[2]
    right = {**POST_HEADERS, "x-passcode": PASSCODE}
    ok = _post(s, {"text": "nothing"}, headers=right, llm_factory=_fake('{"entities": []}'))
    assert ok[0] == 200


def test_passcode_is_compared_in_constant_time(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = http_mod.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(http_mod.hmac, "compare_digest", spy)
    # ASCII only (a non-ASCII passcode is refused at settings load; see test_site_service).
    s = _settings(
        cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE="passe-x1"
    )
    headers = {**POST_HEADERS, "x-passcode": "passe-x1"}
    assert _post(s, {"text": "x"}, headers=headers, llm_factory=_fake('{"entities": []}'))[0] == 200
    assert seen == [(b"passe-x1", b"passe-x1")]


# --- request validation codes -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"{not json", "bad_json"),
        (b"", "bad_json"),
        (b"\xff\xfe{}", "bad_json"),
        (b"[1, 2]", "bad_json"),
        (b'"text"', "bad_json"),
        (b'{"text": "a", "extra": 1}', "bad_json"),
        (b'{"text": "a", "prompt": 5}', "bad_json"),
        (b'{"text": "a", "model": 5}', "bad_json"),
        (b"{}", "missing_text"),
        (b'{"text": null}', "missing_text"),
        (b'{"text": 42}', "missing_text"),
        (b'{"text": ""}', "missing_text"),
        (b'{"text": "  \\n "}', "missing_text"),
        (b'{"text": "a", "top_k": 0}', "bad_top_k"),
        (b'{"text": "a", "top_k": 11}', "bad_top_k"),
        (b'{"text": "a", "top_k": "5"}', "bad_top_k"),
        (b'{"text": "a", "top_k": true}', "bad_top_k"),
        (b'{"text": "a", "top_k": 2.5}', "bad_top_k"),
        (b'{"text": "a", "model": "anthropic/claude-opus"}', "unknown_model"),
    ],
)
def test_bad_requests_are_400_with_their_code(
    cache_dir: Path, tmp_path: Path, raw: bytes, code: str
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, None, raw=raw, llm_factory=_never)
    assert _error(result) == (400, code)
    _assert_common_headers(result[1])


def test_text_and_prompt_limits(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    long_text = _post(s, {"text": "a" * 5001}, llm_factory=_never)
    assert _error(long_text) == (400, "text_too_long")
    assert _json(long_text)["error"]["message"] == "text is 5001 characters; the limit is 5000"
    long_prompt = _post(s, {"text": "a", "prompt": "p" * 4001}, llm_factory=_never)
    assert _error(long_prompt) == (400, "prompt_too_long")


def test_body_over_64_kb_is_413(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    raw = json.dumps({"text": "a" * 70000}).encode()
    assert _error(_post(s, None, raw=raw, llm_factory=_never)) == (413, "body_too_large")


def test_no_key_and_not_offline_is_503(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path)
    result = _post(s, {"text": "aspirin"}, llm_factory=_never)
    assert _error(result) == (503, "not_configured")
    assert "OPENROUTER_API_KEY" in _json(result)["error"]["message"]


# --- errors from the extraction and the provider ------------------------------------


def test_unparseable_answer_is_422(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, llm_factory=_fake("Sorry, here is prose."))
    assert _error(result) == (422, "extraction_unparseable")


def test_answer_without_entities_is_422(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, llm_factory=_fake('{"items": []}'))
    assert _error(result) == (422, "extraction_schema")


def test_exhausted_call_budget_is_429(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    budgeted = BudgetedLLM(FakeLLM(['{"entities": []}']), CallBudget(limit=0))
    result = _post(s, {"text": "abc"}, llm_factory=lambda settings, model: budgeted)
    assert _error(result) == (429, "call_budget_exhausted")


@pytest.mark.parametrize(
    "failure",
    [
        LLMFatalError(f"HTTP 401: No auth credentials for {SENTINEL_KEY}"),
        LLMRetryableError("exhausted 2 retries: ConnectError at https://u:hunter2@gw.example/v1"),
    ],
)
def test_provider_failure_is_502_without_secrets(
    cache_dir: Path, tmp_path: Path, failure: Exception
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": "abc"}, llm_factory=_fake(failure))
    assert _error(result) == (502, "provider_error")
    assert SENTINEL_KEY.encode() not in result[2] and b"hunter2" not in result[2]


def test_a_real_client_401_through_mock_transport_is_502_without_the_key(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def chat(request: httpx.Request) -> httpx.Response:
        echoed = request.headers["authorization"]
        return httpx.Response(401, json={"error": {"message": f"bad credentials: {echoed}"}})

    s = _settings(
        cache_dir,
        tmp_path,
        OPENROUTER_API_KEY=SENTINEL_KEY,
        NER_SITE_BASE_URL="https://user:hunter2@gw.example/v1",
    )
    result = _post(
        s,
        {"text": "abc"},
        llm_factory=lambda st, m: make_llm(st, m, transport=httpx.MockTransport(chat)),
    )
    assert _error(result) == (502, "provider_error")
    assert SENTINEL_KEY.encode() not in result[2] and b"hunter2" not in result[2]
    assert SENTINEL_KEY not in capsys.readouterr().err


def test_an_unexpected_exception_is_a_generic_500(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(settings: Settings, model: str) -> Any:
        raise RuntimeError(f"kaboom {SENTINEL_KEY} {SECRET_TEXT}")

    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(s, {"text": SECRET_TEXT}, llm_factory=boom)
    assert _error(result) == (500, "internal")
    assert b"kaboom" not in result[2] and SENTINEL_KEY.encode() not in result[2]
    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert SENTINEL_KEY not in err and SECRET_TEXT not in err and "kaboom" not in err


# --- full POSTs ---------------------------------------------------------------------


def test_full_post_offline_through_real_bm25(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, NER_SITE_OFFLINE="1")
    result = _post(s, {"text": EXAMPLE_TEXT, "prompt": None, "model": "offline", "top_k": 3})
    assert result[0] == 200
    _assert_common_headers(result[1])
    out = NerResponse.model_validate_json(result[2])
    assert out.mode == "offline" and out.model == "offline"
    by_text = {e.text: e for e in out.entities}
    for e in out.entities:
        assert EXAMPLE_TEXT[e.start : e.end] == e.text  # code-point offsets
    assert by_text["haloperidol"].matches[0].id == "CHEBI:5613"
    assert by_text["Alkaptonuria"].matches[0].id == "MESH:D000474"
    assert by_text["NAT2"].ontology == "ncbi-gene-sample"
    assert by_text["fever"].ontology is None and by_text["fever"].matches == []
    assert all(len(e.matches) <= 3 for e in out.entities)
    assert out.retrieval["bm25"].status == "ok"
    dense = out.retrieval["dense"]
    assert dense.status == "unavailable" and dense.reason
    assert any("dense retrieval unavailable" in w for w in out.warnings)
    assert out.usage.llm_calls == 0 and out.usage.prompt_tokens is None
    assert out.usage.embedding_requests == 0


def test_full_post_live_through_mock_transport(cache_dir: Path, tmp_path: Path) -> None:
    text = "She took haloperidol and ketamine."
    bodies: list[dict[str, Any]] = []

    def chat(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        answer = _answer(("haloperidol", "chemical", 9, 20), ("ketamine", "chemical", 0, 8))
        return httpx.Response(
            200,
            json={
                "model": "google/gemini-3.1-flash-lite",
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 412, "completion_tokens": 88},
            },
        )

    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(
        s,
        {"text": text, "prompt": SECRET_PROMPT, "model": "openai/gpt-6-luna"},
        llm_factory=lambda st, m: make_llm(st, m, transport=httpx.MockTransport(chat)),
    )
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    assert out.model == "openai/gpt-6-luna" and out.mode == "live"
    assert [(e.text, e.start, e.end) for e in out.entities] == [
        ("haloperidol", 9, 20),
        ("ketamine", 25, 33),
    ]
    assert out.usage.model_dump() == {
        "llm_calls": 1,
        "prompt_tokens": 412,
        "completion_tokens": 88,
        "embedding_requests": 0,
        "selector_calls": 0,
        "selector_prompt_tokens": None,
        "selector_completion_tokens": None,
        "selector_cost_usd": None,
    }
    assert bodies[0]["model"] == "openai/gpt-6-luna"
    assert SECRET_PROMPT in bodies[0]["messages"][-1]["content"]
    assert SENTINEL_KEY.encode() not in result[2]


def _histogram_gateway(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        data = []
        for i, text in enumerate(body["input"]):
            vector = [0.0] * embeddings_api.DEFAULT_DIMENSIONS
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            vector[26] = 0.01  # never an all-zero vector
            data.append({"index": i, "embedding": vector})
        return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 1}})

    return gateway


def _build_tool() -> ModuleType:
    tool = PROJECT / "tools" / "build_indexes.py"
    spec = importlib.util.spec_from_file_location("ner_site_build_indexes_http", tool)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_post_with_dense_through_the_default_encoder(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dense index is built with the site's own default encoder (Qwen3 settings),
    its HTTP calls answered by a 26-letter histogram gateway, then opened by
    `get_indexes` exactly as on Vercel."""
    seen: list[httpx.Request] = []
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(_histogram_gateway(seen))
        return real_client(**kwargs)

    monkeypatch.setattr("xwalk.retrieval.embeddings_api.httpx.Client", client)
    dense_dir = tmp_path / "indexes"
    encoder = embeddings_api.default_embeddings_encoder(api_key=SENTINEL_KEY)
    _build_tool().build_all(dense_dir, encoder)
    built = len(seen)
    assert built > 0

    s = _settings(
        cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_DENSE_DIR=str(dense_dir)
    )
    health = _json(http_mod.handle("GET", "/api/health", {}, b"", settings=s))
    assert health["retrieval"] == {"bm25": {"status": "ok"}, "dense": {"status": "ok"}}

    text = "haloperidol and Alkaptonuria, and fever"
    answer = _answer(
        ("haloperidol", "chemical", 0, 11),
        ("Alkaptonuria", "disease", 16, 28),
        ("fever", "other", 34, 39),
    )
    result = _post(s, {"text": text, "top_k": 5}, llm_factory=_fake(answer))
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    assert out.retrieval["dense"].status == "ok"
    halo = out.entities[0]
    # The histogram also embeds the Qwen3 query instruction, so dense ranks are noise;
    # what matters is that both retrievers ran and were fused.
    by_id = {m.id: m for m in halo.matches}
    assert by_id["CHEBI:5613"].retrievers["bm25"] == 1
    assert any("dense-qwen3" in m.retrievers for m in halo.matches)
    assert all(m.id.startswith("CHEBI:") for m in halo.matches)
    assert out.usage.embedding_requests == 1  # one batch for both routed entities
    assert not any("dense" in w for w in out.warnings)
    query_calls = seen[built:]
    assert len(query_calls) == 1
    assert all(r.headers["authorization"] == f"Bearer {SENTINEL_KEY}" for r in query_calls)
    assert SENTINEL_KEY.encode() not in result[2]
    assert SENTINEL_KEY not in capsys.readouterr().err


# --- logging ------------------------------------------------------------------------


def test_one_log_line_per_request_without_text_prompt_or_key(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    capsys.readouterr()
    headers = {**POST_HEADERS, "x-passcode": PASSCODE}
    answer = _answer(("haloperidol", "chemical", 23, 34))
    ok = _post(
        s,
        {"text": SECRET_TEXT, "prompt": SECRET_PROMPT, "top_k": 2},
        headers=headers,
        llm_factory=_fake(answer),
    )
    assert ok[0] == 200
    bad = _post(s, {"text": SECRET_TEXT}, headers={**headers, "x-passcode": "nope"})
    assert bad[0] == 401
    http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    first = records[0]
    assert first["status"] == 200 and first["path"] == "/api/ner"
    assert first["model"] == "google/gemini-3.1-flash-lite" and first["mode"] == "live"
    assert first["text_chars"] == len(SECRET_TEXT) and first["custom_prompt"] is True
    assert first["entities"] == 1 and first["llm_calls"] == 1
    assert {"extract", "map", "total"} <= set(first["timing_ms"])
    assert records[1]["status"] == 401 and records[1]["code"] == "passcode_invalid"
    assert records[2]["path"] == "/api/health" and records[2]["status"] == 200
    for line in lines:
        for secret in (SENTINEL_KEY, PASSCODE, SECRET_TEXT, SECRET_PROMPT, "Quokka", "PROMPTSENT"):
            assert secret not in line


# --- api/*.py shims through a real BaseHTTPRequestHandler on a socket --------------


def _shim(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ner_site_api_{name}", PROJECT / "api" / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["ner.py", "health.py", "stats.py"])
def shim_server(
    request: pytest.FixtureRequest,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[int]:
    for name in ("OPENROUTER_API_KEY", "NER_SITE_PASSCODE", "NER_SITE_MODELS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NER_SITE_OFFLINE", "1")
    monkeypatch.setenv("NER_SITE_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("NER_SITE_DENSE_DIR", str(tmp_path / "no-dense"))
    module = _shim(request.param)
    handler = module.handler
    assert handler.__name__ == "handler"  # Vercel looks up exactly this name
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _request(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    conn = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        conn.close()


def test_shim_serves_health_and_ner(shim_server: int, capsys: pytest.CaptureFixture[str]) -> None:
    status, headers, body = _request(shim_server, "GET", "/api/health?x=1")
    assert status == 200 and headers["cache-control"] == "no-store"
    assert not any(k.startswith("access-control-") for k in headers)
    assert HealthResponse.model_validate_json(body).mode == "offline"
    payload = json.dumps({"text": EXAMPLE_TEXT, "model": "offline"}).encode()
    status, headers, body = _request(shim_server, "POST", "/api/ner", payload, {**POST_HEADERS})
    assert status == 200, body
    assert int(headers["content-length"]) == len(body)
    assert any(e["text"] == "haloperidol" for e in json.loads(body)["entities"])
    status, _, body = _request(shim_server, "HEAD", "/api/health")
    assert status == 200 and body == b""
    status, headers, _ = _request(shim_server, "OPTIONS", "/api/ner")
    assert status == 405 and not any(k.startswith("access-control-") for k in headers)
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 4  # one line per request, no access-log lines on top
    assert all(json.loads(line)["status"] in {200, 405} for line in err)


def _raw_exchange(port: int, request: bytes) -> tuple[bytes, bytes]:
    """Send `request` as is and read until the server closes; (status line, body)."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    data = b""
    try:
        sock.sendall(request)
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()
    head, _, body = data.partition(b"\r\n\r\n")
    return head.split(b"\r\n")[0], body


def test_shim_refuses_a_large_body_before_reading_it(shim_server: int) -> None:
    """Content-Length says 10 MB and no byte of it is sent: the answer comes anyway, so
    the shim did not try to read the body."""
    status_line, body = _raw_exchange(
        shim_server,
        b"POST /api/ner HTTP/1.1\r\nHost: x\r\nX-Requested-With: xwalk-ner\r\n"
        b"Content-Type: application/json\r\nContent-Length: 10000000\r\n\r\n",
    )
    assert b" 413 " in status_line
    assert json.loads(body)["error"]["code"] == "body_too_large"


@pytest.mark.parametrize(
    ("length_header", "status", "code"),
    [(b"", b" 411 ", "length_required"), (b"Content-Length: -3\r\n", b" 400 ", "bad_header")],
)
def test_shim_needs_a_valid_content_length_on_post(
    shim_server: int, length_header: bytes, status: bytes, code: str
) -> None:
    status_line, body = _raw_exchange(
        shim_server,
        b"POST /api/ner HTTP/1.1\r\nHost: x\r\nX-Requested-With: xwalk-ner\r\n"
        + length_header
        + b"Connection: close\r\n\r\n",
    )
    assert status in status_line
    assert json.loads(body)["error"]["code"] == code


def test_shim_with_a_bad_setting_answers_503_not_a_traceback(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NER_SITE_MAX_CHARS", "lots")
    monkeypatch.setenv("NER_SITE_CACHE_DIR", str(cache_dir))
    module = _shim("health.py")
    server = ThreadingHTTPServer(("127.0.0.1", 0), module.handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _request(server.server_address[1], "GET", "/api/health")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 503
    error = json.loads(body)["error"]
    assert error["code"] == "not_configured" and "NER_SITE_MAX_CHARS" in error["message"]
    assert "Traceback" not in error["message"]


def test_a_short_passcode_does_not_corrupt_responses(cache_dir: Path, tmp_path: Path) -> None:
    """The secret scrubber must not rewrite ordinary JSON ("ok", "true") in a body."""
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE="ok")
    health = http_mod.handle("GET", "/api/health", {}, b"", settings=s)
    assert _json(health)["ok"] is True
    result = _post(
        s,
        {"text": "nothing"},
        headers={**POST_HEADERS, "x-passcode": "ok"},
        llm_factory=_fake('{"entities": []}'),
    )
    assert result[0] == 200 and _json(result)["ok"] is True


# --- review fixes: wall-time bound, dense status per request, scrubbing, parsing ---

SIX = "haloperidol ketamine cedrol Alkaptonuria NAT2 cumin"
SIX_ANSWER = _answer(
    ("haloperidol", "chemical", 0, 11),
    ("ketamine", "chemical", 12, 20),
    ("cedrol", "chemical", 21, 27),
    ("Alkaptonuria", "disease", 28, 40),
    ("NAT2", "gene", 41, 45),
    ("cumin", "food", 46, 51),
)


@pytest.fixture(scope="module")
def qwen_dense_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dense indexes built once by the site's default Qwen3 encoder against the
    histogram gateway; each test then decides how the query embeddings answer."""
    out = tmp_path_factory.mktemp("qwen-dense") / "indexes"
    encoder = embeddings_api.default_embeddings_encoder(
        api_key=SENTINEL_KEY, transport=httpx.MockTransport(_histogram_gateway([]))
    )
    _build_tool().build_all(out, encoder)
    return out


def _query_gateway(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Every embeddings client created from now on (the site's query encoder) answers
    through `handler`."""
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr("xwalk.retrieval.embeddings_api.httpx.Client", client)


def _dense_settings(cache_dir: Path, tmp_path: Path, dense_dir: Path) -> Settings:
    return _settings(
        cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_DENSE_DIR=str(dense_dir)
    )


@pytest.fixture
def stalled() -> Iterator[threading.Event]:
    """A query gateway that hangs for up to 6 s; released at teardown so no thread
    outlives the test by long."""
    release = threading.Event()
    yield release
    release.set()


def test_abandoned_embedding_threads_do_not_hold_the_response(
    cache_dir: Path,
    tmp_path: Path,
    qwen_dense_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    stalled: threading.Event,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reviewer's repro: an encoder stuck for 6 s, a 1 s retriever timeout and six
    entities. The response must not wait for the stuck threads to finish."""
    import time

    def stuck(request: httpx.Request) -> httpx.Response:
        stalled.wait(6.0)
        return _histogram_gateway([])(request)

    _query_gateway(monkeypatch, stuck)
    monkeypatch.setattr("ner_site.service.MAP_RETRIEVER_TIMEOUT_S", 1.0)
    s = _dense_settings(cache_dir, tmp_path, qwen_dense_dir)
    started = time.monotonic()
    result = _post(s, {"text": SIX, "top_k": 3}, llm_factory=_fake(SIX_ANSWER))
    elapsed = time.monotonic() - started
    assert result[0] == 200, result[2]
    assert elapsed < 4.0, f"handle took {elapsed:.1f} s; it waited for abandoned threads"
    out = NerResponse.model_validate_json(result[2])
    assert out.timing_ms["total"] <= 4000
    assert all(e.matches and e.matches[0].retrievers.get("bm25") for e in out.entities)

    # The stuck threads finish after the response; a later response is its own.
    stalled.set()
    _query_gateway(monkeypatch, _histogram_gateway([]))
    indexes_mod.reset()
    again = _post(s, {"text": SIX, "top_k": 3}, llm_factory=_fake(SIX_ANSWER))
    later = NerResponse.model_validate_json(again[2])
    assert later.retrieval["dense"].status == "ok"
    assert later.usage.embedding_requests == 1  # one batch
    assert not any("timed out" in n for e in later.entities for n in e.notes)
    # The first response is unchanged: the stuck batch, then six single requests.
    assert out.usage.embedding_requests == 7
    assert "Traceback" not in capsys.readouterr().err


def test_one_deadline_bounds_the_whole_mapping_step(
    cache_dir: Path,
    tmp_path: Path,
    qwen_dense_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    stalled: threading.Event,
) -> None:
    """Six entities, four at a time, each dense search stuck: without one overall
    deadline the per-call timeouts add up. With it the request ends at the deadline,
    every entity keeps its BM25 matches and says its dense search was cut off."""
    import time

    def stuck(request: httpx.Request) -> httpx.Response:
        stalled.wait(6.0)
        return _histogram_gateway([])(request)

    _query_gateway(monkeypatch, stuck)
    monkeypatch.setattr("ner_site.service.REQUEST_DEADLINE_S", 2.0)
    monkeypatch.setattr("ner_site.service.MIN_MAP_BUDGET_S", 1.0)
    s = _dense_settings(cache_dir, tmp_path, qwen_dense_dir)
    started = time.monotonic()
    result = _post(s, {"text": SIX, "top_k": 3}, llm_factory=_fake(SIX_ANSWER))
    elapsed = time.monotonic() - started
    assert result[0] == 200, result[2]
    assert elapsed < 3.5, f"handle took {elapsed:.1f} s past a 2 s deadline"
    out = NerResponse.model_validate_json(result[2])
    assert out.timing_ms["total"] <= 3000
    for entity in out.entities:
        assert entity.matches and entity.matches[0].retrievers.get("bm25") == 1, entity
        assert any("cut off by the request deadline" in n for n in entity.notes), entity.notes
    assert any("cut off by the request deadline" in w for w in out.warnings)
    dense = out.retrieval["dense"]
    assert dense.status == "unavailable"
    assert dense.reason and dense.reason.startswith("every dense search failed in this request")
    assert "dense retrieval unavailable; matches use BM25 only" in out.warnings


def test_the_deadline_leaves_room_inside_max_duration() -> None:
    """Worst case: the extraction uses its whole budget, then mapping gets the rest of
    the request deadline (at least its minimum), plus the grace for local BM25."""
    from ner_site import mapping, service

    max_duration = json.loads((PROJECT / "vercel.json").read_text())["functions"]["api/*.py"][
        "maxDuration"
    ]
    worst = (
        max(service.REQUEST_DEADLINE_S, service.EXTRACT_TIMEOUT_S + service.MIN_MAP_BUDGET_S)
        + mapping.LOCAL_GRACE_S
    )
    assert worst <= 105 < max_duration


def test_the_query_encoder_never_waits_out_a_retry_after(
    cache_dir: Path, qwen_dense_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 with Retry-After: 120 must fail the one query at once (no retry, no sleep):
    a sleeping encoder thread outlives the function's duration."""
    seen: list[httpx.Request] = []

    def limited(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            429, headers={"retry-after": "120"}, json={"error": {"message": "slow down"}}
        )

    _query_gateway(monkeypatch, limited)
    opened = indexes_mod.open_indexes(
        cache_dir=cache_dir, dense_dir=qwen_dense_dir, api_key=SENTINEL_KEY
    )
    dense = opened["chebi-sample"].retrievers[-1]
    encoder = dense._encoder  # type: ignore[attr-defined]
    outcome: list[BaseException] = []

    def query() -> None:
        try:
            encoder.encode(["aspirin"], is_query=True)
        except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
            outcome.append(exc)

    worker = threading.Thread(target=query, daemon=True)
    worker.start()
    worker.join(5.0)
    assert not worker.is_alive(), "the query encoder is sleeping on Retry-After"
    assert len(seen) == 1 and isinstance(outcome[0], embeddings_api.EmbeddingAPIError)


def _failing_for(words: tuple[str, ...]) -> Callable[[httpx.Request], httpx.Response]:
    good = _histogram_gateway([])

    def gateway(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        if any(w in t for t in texts for w in words):
            return httpx.Response(500, json={"error": {"message": "upstream exploded"}})
        return good(request)

    return gateway


def test_dense_failing_for_every_entity_is_reported_unavailable(
    cache_dir: Path, tmp_path: Path, qwen_dense_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _query_gateway(monkeypatch, _failing_for(("Query:",)))  # every query
    s = _dense_settings(cache_dir, tmp_path, qwen_dense_dir)
    health = _json(http_mod.handle("GET", "/api/health", {}, b"", settings=s))
    assert health["retrieval"]["dense"] == {"status": "ok"}  # open-time status
    result = _post(s, {"text": SIX, "top_k": 3}, llm_factory=_fake(SIX_ANSWER))
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    dense = out.retrieval["dense"]
    assert dense.status == "unavailable"
    assert dense.reason is not None
    assert dense.reason.startswith("every dense search failed in this request: ")
    assert "HTTP 500" in dense.reason and "upstream exploded" in dense.reason
    assert "dense retrieval unavailable; matches use BM25 only" in out.warnings
    assert all(e.matches for e in out.entities)
    assert _json(result)["retrieval"]["dense"]["status"] == "unavailable"


def test_dense_failing_for_some_entities_names_them(
    cache_dir: Path, tmp_path: Path, qwen_dense_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _query_gateway(monkeypatch, _failing_for(("ketamine", "NAT2")))
    s = _dense_settings(cache_dir, tmp_path, qwen_dense_dir)
    result = _post(s, {"text": SIX, "top_k": 3}, llm_factory=_fake(SIX_ANSWER))
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    assert out.retrieval["dense"].status == "ok"
    assert "dense retrieval unavailable; matches use BM25 only" not in out.warnings
    named = [w for w in out.warnings if w.startswith("dense search failed for 2 of 6 entities")]
    assert named and "(e2, e5)" in named[0], out.warnings


def test_an_embeddings_401_echoing_the_key_leaks_it_nowhere(
    cache_dir: Path,
    tmp_path: Path,
    qwen_dense_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"]
        return httpx.Response(401, json={"error": {"message": f"No auth credentials: {auth}"}})

    _query_gateway(monkeypatch, echo)
    s = _dense_settings(cache_dir, tmp_path, qwen_dense_dir)
    capsys.readouterr()
    result = _post(s, {"text": SIX, "top_k": 3}, llm_factory=_fake(SIX_ANSWER))
    assert result[0] == 200, result[2]
    assert b"No auth credentials" in result[2]  # the error itself is reported
    assert SENTINEL_KEY.encode() not in result[2]
    assert SENTINEL_KEY[8:].encode() not in result[2]
    assert SENTINEL_KEY not in capsys.readouterr().err


def test_a_chat_401_echoing_the_key_leaks_it_nowhere(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def chat(request: httpx.Request) -> httpx.Response:
        auth = request.headers["authorization"]
        return httpx.Response(401, json={"error": {"message": f"User not found: {auth}"}})

    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    capsys.readouterr()
    result = _post(
        s,
        {"text": "abc"},
        llm_factory=lambda st, m: make_llm(st, m, transport=httpx.MockTransport(chat)),
    )
    assert _error(result) == (502, "provider_error")
    assert b"User not found" in result[2]
    assert SENTINEL_KEY.encode() not in result[2]
    assert SENTINEL_KEY not in capsys.readouterr().err


def test_the_body_and_log_filters_remove_the_key(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The last line of defence on its own: whatever reaches the body or the log line."""
    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_PASSCODE=PASSCODE)
    body = http_mod._json_body({"leak": f"x {SENTINEL_KEY} {PASSCODE} y"}, s)
    assert SENTINEL_KEY.encode() not in body and PASSCODE.encode() not in body
    capsys.readouterr()
    http_mod._log({"leak": f"x {SENTINEL_KEY} {PASSCODE} y"}, s)
    err = capsys.readouterr().err
    assert "[redacted]" in err and SENTINEL_KEY not in err and PASSCODE not in err


@pytest.mark.parametrize(
    "payload",
    [b'{"text": "aspirin \\ud800 here"}', b'{"text": "aspirin", "prompt": "find \\udfff"}'],
)
def test_a_lone_surrogate_is_a_400_not_a_500(
    cache_dir: Path, tmp_path: Path, payload: bytes
) -> None:
    def chat(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"entities": []}'}}]})

    s = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    result = _post(
        s,
        None,
        raw=payload,
        llm_factory=lambda st, m: make_llm(st, m, transport=httpx.MockTransport(chat)),
    )
    assert _error(result) == (400, "bad_json")
    assert b"UTF-8" in result[2]


@pytest.mark.parametrize("value", [b"\xb2", b"\xb9", b"9" * 5000])
def test_shim_refuses_a_content_length_that_only_looks_numeric(
    shim_server: int, value: bytes
) -> None:
    """U+00B2 is str.isdigit() but not int()-able; 5000 digits exceed int()'s limit."""
    status_line, body = _raw_exchange(
        shim_server,
        b"POST /api/ner HTTP/1.1\r\nHost: x\r\nX-Requested-With: xwalk-ner\r\n"
        b"Content-Length: " + value + b"\r\nConnection: close\r\n\r\n",
    )
    assert b" 400 " in status_line, status_line
    assert json.loads(body)["error"]["code"] == "bad_header"


def test_the_default_prompt_follows_the_entity_limit(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path, NER_SITE_OFFLINE="1", NER_SITE_MAX_ENTITIES="7")
    health = _json(http_mod.handle("GET", "/api/health", {}, b"", settings=s))
    assert "at most 7 entities" in health["default_prompt"]
    assert "at most 25" not in health["default_prompt"]
    llm = FakeLLM(['{"entities": []}'])
    s_live = _settings(
        cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_MAX_ENTITIES="7"
    )
    result = _post(s_live, {"text": "nothing", "prompt": None}, llm_factory=lambda st, m: llm)
    assert result[0] == 200, result[2]
    assert "at most 7 entities" in llm.requests[0].user


# --- open access: admission, the chain and the visitor allowance --------------------
# (docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md, 5, 6, 8)

ANTHROPIC_KEY = "sk-ant-SENTINEL-feedbeef-9191"
DB_URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"
RAW_IP = "203.0.113.77"
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
GEMINI = "google/gemini-3.1-flash-lite"
VISITOR_HEADERS = {**POST_HEADERS, "x-real-ip": RAW_IP, "x-forwarded-for": f"{RAW_IP}, 10.0.0.1"}
HALO_TEXT = "She took haloperidol for a fever."
HALO = _answer(("haloperidol", "chemical", 9, 20))
PICK = '{"chosen_key": "C01", "confidence_score": 0.7, "explanation": "first"}'


def _claude_error(reason: str) -> Exception:
    error = LLMRetryableError(f"Anthropic said {reason} to {ANTHROPIC_KEY}")
    error.reason = reason  # type: ignore[attr-defined]
    return error


class _Claude:
    """The Claude seam of ``handle``: scripted clients, one per model and purpose."""

    def __init__(self, extracted: str = '{"entities": []}', **scripts: Any) -> None:
        self.extracted = extracted
        self.scripts = scripts  # opus_extract=[...], sonnet_select=[...]
        self.made: dict[tuple[str, str], FakeLLM] = {}

    def __call__(self, settings: Settings, model: str, purpose: str) -> FakeLLM:
        tier = "opus" if model == OPUS else "sonnet"
        script = self.scripts.get(f"{tier}_{purpose}")
        answer = self.extracted if purpose == "extract" else PICK
        llm = (
            FakeLLM(script, model=model)
            if script is not None
            else FakeLLM(handler=lambda request: answer, model=model)
        )
        self.made[(model, purpose)] = llm
        return llm

    def calls(self, model: str, purpose: str = "extract") -> int:
        made = self.made.get((model, purpose))
        return 0 if made is None else len(made.requests)


def _no_claude(settings: Settings, model: str, purpose: str) -> Any:
    pytest.fail("a Claude client was built for a request that should not use Claude")


@pytest.fixture
def visits(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The limiter's table, in memory (tests/test_site_limits.py), behind the seam the
    real Neon executor is made by."""
    from tests.test_site_limits import FakeVisits

    fake = FakeVisits()
    opened: list[str] = []

    def executor(database_url: str) -> Any:
        opened.append(database_url)
        return fake

    monkeypatch.setattr(http_mod, "make_limits_executor", executor)
    fake.opened = opened
    return fake


def _open_site(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    """An open deployment: both keys, the database, no passcode, the new defaults."""
    base = {
        "OPENROUTER_API_KEY": SENTINEL_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "DATABASE_URL": DB_URL,
        "NER_SITE_SELECTOR": "",
    }
    return _settings(cache_dir, tmp_path, **{**base, **env})


# Own keys are off unless the operator lists providers; these tests turn them on.
ALL_OWN = "anthropic,openai,openrouter"


def _own_only(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    """No site key at all, and visitors' own keys turned on for every provider."""
    return _settings(cache_dir, tmp_path, **{"NER_SITE_OWN_PROVIDERS": ALL_OWN, **env})


def _own_site(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    """The open deployment with visitors' own keys turned on for every provider."""
    return _open_site(cache_dir, tmp_path, **{"NER_SITE_OWN_PROVIDERS": ALL_OWN, **env})


def _ask(
    settings: Settings,
    payload: dict[str, Any] | None = None,
    *,
    claude: Any = None,
    llm_factory: Any = None,
    headers: dict[str, str] | None = None,
) -> Handled:
    body = json.dumps({"text": "nothing", "selector": "retrieval", **(payload or {})})
    return http_mod.handle(
        "POST",
        "/api/ner",
        dict(VISITOR_HEADERS if headers is None else headers),
        body.encode("utf-8"),
        settings=settings,
        llm_factory=llm_factory or _never,
        claude_factory=claude or _Claude(),
    )


def _tiers(fake: Any) -> dict[str, int]:
    found: dict[str, int] = {}
    for (_, _, tier), count in fake.rows.items():
        found[tier] = found.get(tier, 0) + count
    return found


def _last_log(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    return record


def test_a_request_runs_on_opus_by_default(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _open_site(cache_dir, tmp_path)
    claude = _Claude(HALO)
    result = http_mod.handle(
        "POST",
        "/api/ner",
        dict(VISITOR_HEADERS),
        json.dumps({"text": HALO_TEXT}).encode(),
        settings=s,
        llm_factory=_never,
        claude_factory=claude,
    )
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    assert out.model == out.requested_model == OPUS
    assert out.selector == out.requested_selector == OPUS
    assert out.fallbacks == [] and out.notices == []
    (entity,) = out.entities
    assert entity.pick is not None and entity.pick.selector == OPUS and entity.pick.id
    assert claude.calls(OPUS, "extract") == 1 and claude.calls(OPUS, "select") == 1
    assert claude.calls(SONNET, "extract") == claude.calls(SONNET, "select") == 0
    assert _tiers(visits) == {"opus": 1} and visits.opened == [DB_URL]
    record = _last_log(capsys)
    assert record["tier"] == "opus" and record["fallbacks"] == []
    assert record["model"] == OPUS and record["ran"] == OPUS and record["status"] == 200


def test_the_eleventh_request_runs_on_sonnet_and_the_thirty_first_is_refused(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _own_site(cache_dir, tmp_path)
    for _ in range(10):
        tenth = _ask(s)
    assert _json(tenth)["model"] == OPUS and _json(tenth)["notices"] == []
    eleventh = _json(_ask(s))
    assert eleventh["model"] == SONNET and eleventh["requested_model"] == OPUS
    assert eleventh["fallbacks"] == [
        {"stage": "admit", "from": OPUS, "to": SONNET, "reason": "opus_allowance_used"}
    ]
    assert eleventh["notices"] == [
        "Your Opus allowance for this hour is used; this ran on Claude Sonnet 5.5."
    ]
    record = _last_log(capsys)
    assert record["tier"] == "sonnet" and record["fallbacks"] == eleventh["fallbacks"]
    for _ in range(19):
        thirtieth = _ask(s)
    assert thirtieth[0] == 200 and _tiers(visits) == {"opus": 10, "sonnet": 20}

    claude = _Claude()
    refused = _ask(s, claude=claude)
    assert _error(refused) == (429, "visitor_limit_reached")
    retry_after = int(refused[1]["Retry-After"])
    assert 1 <= retry_after <= 3600
    minutes = -(-retry_after // 60)
    unit = "minute" if minutes == 1 else "minutes"
    assert _json(refused)["error"]["message"] == (
        f"you have used this hour's requests; try again in {minutes} {unit}"
        ", or use your own API key"
    )
    _assert_common_headers(refused[1])
    assert claude.made == {} and _tiers(visits) == {"opus": 10, "sonnet": 20}  # not recorded
    record = _last_log(capsys)
    assert record["status"] == 429 and record["code"] == "visitor_limit_reached"
    # Another visitor is not affected.
    other = _ask(s, headers={**POST_HEADERS, "x-real-ip": "198.51.100.9"})
    assert other[0] == 200 and _json(other)["model"] == OPUS


def test_the_allowance_follows_the_settings(cache_dir: Path, tmp_path: Path, visits: Any) -> None:
    s = _open_site(cache_dir, tmp_path, NER_SITE_OPUS_PER_HOUR="1", NER_SITE_SONNET_PER_HOUR="1")
    assert [_json(_ask(s)).get("model") for _ in range(2)] == [OPUS, SONNET]
    assert _ask(s)[0] == 429


def test_a_request_for_sonnet_is_counted_as_sonnet(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path)
    claude = _Claude()
    out = _json(_ask(s, {"model": SONNET}, claude=claude))
    assert out["model"] == SONNET and out["fallbacks"] == [] and _tiers(visits) == {"sonnet": 1}
    assert (OPUS, "extract") not in claude.made  # the chain starts at the wanted model


def test_an_unreachable_limiter_refuses_every_site_model(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from xwalk.retrieval.postgres import PostgresError

    visits.fail = PostgresError(f"could not reach Neon at ep-x-pooler ({DB_URL})", code="08006")
    s = _own_site(cache_dir, tmp_path)
    for model in (OPUS, SONNET, GEMINI):
        result = _ask(s, {"model": model}, claude=_no_claude)
        assert _error(result) == (503, "limits_unavailable")
        assert _json(result)["error"]["message"] == (
            "the site cannot count requests right now; try again shortly, or use your own API key"
        )
        assert result[1]["Retry-After"] == "30"
    assert visits.rows == {}
    err = capsys.readouterr().err
    assert "PostgresError" in err and "ep-x-pooler" not in err and "SENTINELpw" not in err


def test_a_contended_visitor_lock_is_a_429_and_no_chain_is_built(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """One visitor's parallel requests queue on the visitor's lock; one that waited past
    the lock timeout (SQLSTATE 55P03) is refused, not let through uncounted on Sonnet."""
    from xwalk.retrieval.postgres import PostgresError

    visits.fail = PostgresError(f"canceling statement due to lock timeout ({DB_URL})", code="55P03")
    s = _open_site(cache_dir, tmp_path)
    for payload in ({}, {"model": SONNET}, {"model": GEMINI}):
        claude = _Claude()
        refused = _ask(s, payload, claude=claude)
        assert _error(refused) == (429, "visitor_limit_reached")
        assert _json(refused)["error"]["message"] == (
            "too many requests at once; try again in a few seconds"
        )
        assert refused[1]["Retry-After"] == "5"
        _assert_common_headers(refused[1])
        assert claude.made == {} and visits.rows == {}
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[-1])
    assert record["status"] == 429 and record["code"] == "visitor_limit_reached"
    assert "55P03" in record["limits"] and "tier" not in record
    assert "ep-x-pooler" not in err and "SENTINELpw" not in err
    # Other database failures use the temporary outage response.
    visits.fail = PostgresError("could not serialize access", code="40001")
    assert _error(_ask(s, claude=_no_claude)) == (503, "limits_unavailable")
    assert "40001" in _last_log(capsys)["limits"]


def test_the_block_reason_in_a_notice_is_the_one_read_at_admission(
    cache_dir: Path, tmp_path: Path, visits: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The block is process-wide and can change or expire while a request runs. The
    request's notice says what was true when its start was decided, and the block is
    read once."""
    from ner_site import fallback

    reads: list[str | None] = []
    answers = iter(["balance_exhausted", "key_refused", None])

    def block(**kw: Any) -> str | None:
        reads.append(next(answers))
        return reads[-1]

    monkeypatch.setattr(fallback, "claude_block", block)
    s = _open_site(cache_dir, tmp_path)
    out = _json(_ask(s, claude=_no_claude_calls(), llm_factory=_fake('{"entities": []}')))
    assert reads == ["balance_exhausted"]
    assert out["fallbacks"] == [
        {"stage": "admit", "from": OPUS, "to": GEMINI, "reason": "claude_unavailable"}
    ]
    assert out["notices"] == [
        "Claude is unavailable (balance exhausted); this ran on Gemini 3.1 Flash Lite "
        "through OpenRouter."
    ]


def test_an_index_failure_is_not_a_counted_visit(
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The indexes are opened before admission: a request that cannot run for that
    reason has cost the visitor nothing."""
    s = _open_site(cache_dir, tmp_path)
    opened: list[int] = []
    real = http_mod.get_indexes

    def broken(view: Any) -> Any:
        raise RuntimeError("the index store cannot be opened")

    monkeypatch.setattr(http_mod, "get_indexes", broken)
    failed = _ask(s, claude=_no_claude)
    assert _error(failed) == (500, "internal")
    assert visits.sent == [] and visits.rows == {}
    assert _last_log(capsys)["exception"] == "RuntimeError"

    def counting(view: Any) -> Any:
        opened.append(1)
        assert visits.rows == {}, "the indexes are opened before the visit is counted"
        return real(view)

    monkeypatch.setattr(http_mod, "get_indexes", counting)
    assert _ask(s)[0] == 200
    assert opened == [1] and _tiers(visits) == {"opus": 1}  # opened once, then passed on


def test_without_a_database_an_open_site_is_blocked_and_a_private_one_unmetered(
    cache_dir: Path, tmp_path: Path
) -> None:
    open_site = _open_site(cache_dir, tmp_path, DATABASE_URL="")
    refused = _ask(open_site, claude=_no_claude)
    assert _error(refused) == (503, "limits_unavailable")
    assert _json(refused)["error"]["message"] == "visitor limits are not configured"
    private = _open_site(cache_dir, tmp_path, DATABASE_URL="", NER_SITE_PASSCODE=PASSCODE)
    headers = {**VISITOR_HEADERS, "x-passcode": PASSCODE}
    for _ in range(12):  # unmetered: Opus every time
        out = _json(_ask(private, headers=headers))
        assert out["model"] == OPUS and out["fallbacks"] == [] and out["notices"] == []


def test_a_private_site_with_a_database_is_metered_after_the_passcode(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path, NER_SITE_PASSCODE=PASSCODE)
    assert _error(_ask(s, claude=_no_claude)) == (401, "passcode_required")
    assert visits.sent == []  # refused before the limiter is asked
    assert _ask(s, headers={**VISITOR_HEADERS, "x-passcode": PASSCODE})[0] == 200
    assert _tiers(visits) == {"opus": 1}


def test_a_blocked_claude_starts_on_the_fallback_model_with_the_notice(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from ner_site import fallback

    fallback.block_claude("balance_exhausted")
    s = _open_site(cache_dir, tmp_path)
    claude = _Claude()
    result = _ask(s, claude=claude, llm_factory=_fake('{"entities": []}'))
    assert result[0] == 200, result[2]
    out = _json(result)
    assert out["requested_model"] == OPUS and out["model"] == "fake"  # what the answer names
    assert out["fallbacks"] == [
        {"stage": "admit", "from": OPUS, "to": GEMINI, "reason": "claude_unavailable"}
    ]
    assert out["notices"] == [
        "Claude is unavailable (balance exhausted); this ran on Gemini 3.1 Flash Lite "
        "through OpenRouter."
    ]
    assert all(len(fake.requests) == 0 for fake in claude.made.values())
    assert _tiers(visits) == {"opus": 1}  # the visit is still the visitor's
    assert _last_log(capsys)["fallbacks"] == out["fallbacks"]
    # After the allowance step-down too: both departures, in order.
    for _ in range(10):
        last = _json(_ask(s, llm_factory=_fake('{"entities": []}')))
    assert [f["reason"] for f in last["fallbacks"]] == ["opus_allowance_used", "claude_unavailable"]
    assert [(f["from"], f["to"]) for f in last["fallbacks"]] == [(OPUS, SONNET), (SONNET, GEMINI)]
    # Without an OpenRouter key there is no step to move to: Claude is tried.
    alone = _open_site(cache_dir, tmp_path, OPENROUTER_API_KEY="")
    tried = _Claude()
    fresh = {**POST_HEADERS, "x-real-ip": "198.51.100.9"}  # a visitor with Opus left
    assert _json(_ask(alone, claude=tried, headers=fresh))["fallbacks"] == []
    assert tried.calls(OPUS) == 1


def test_a_balance_error_in_one_request_moves_the_next_one_off_claude(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path)
    dead = _claude_error("balance_exhausted")
    first = _ask(
        s,
        claude=_Claude(opus_extract=[dead], sonnet_extract=[dead]),
        llm_factory=_fake('{"entities": []}'),
    )
    assert first[0] == 200 and ANTHROPIC_KEY.encode() not in first[2]
    # One move, past Sonnet: it runs on the same balance.
    assert _json(first)["fallbacks"] == [
        {"stage": "extract", "from": OPUS, "to": GEMINI, "reason": "balance_exhausted"}
    ]
    second = _json(_ask(s, claude=_no_claude_calls(), llm_factory=_fake('{"entities": []}')))
    assert [f["reason"] for f in second["fallbacks"]] == ["claude_unavailable"]


def _no_claude_calls() -> Any:
    def build(settings: Settings, model: str, purpose: str) -> FakeLLM:
        return FakeLLM([AssertionError("Claude was called while it is blocked")], model=model)

    return build


def test_a_request_for_an_openrouter_model_has_no_chain_and_counts_as_other(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _open_site(cache_dir, tmp_path)
    seen: list[str] = []

    def factory(settings: Settings, model: str) -> FakeLLM:
        seen.append(model)
        return FakeLLM(['{"entities": []}'])

    result = _ask(s, {"model": GEMINI}, llm_factory=factory, claude=_no_claude)
    assert result[0] == 200, result[2]
    out = _json(result)
    assert out["model"] == out["requested_model"] == GEMINI and seen == [GEMINI]
    assert out["fallbacks"] == [] and out["notices"] == []
    assert _tiers(visits) == {"other": 1} and _last_log(capsys)["tier"] == "other"
    # Jev as the selector does not make it a Claude request either.
    monkey_decider: list[int] = []

    def decider(settings: Settings) -> Any:
        from xwalk.decide.fake import FakeDecider

        monkey_decider.append(1)
        return FakeDecider()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(http_mod, "make_decider", decider)
        jev = _ask(s, {"model": GEMINI, "selector": "jev"}, llm_factory=factory, claude=_no_claude)
    assert jev[0] == 200 and monkey_decider == [1] and _tiers(visits) == {"other": 2}


def test_a_claude_selector_decides_the_tier_of_an_openrouter_request(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path, NER_SITE_OPUS_PER_HOUR="1")
    claude = _Claude()
    payload = {"text": HALO_TEXT, "model": GEMINI, "selector": OPUS}
    first = _json(_ask(s, payload, claude=claude, llm_factory=_fake(HALO)))
    assert first["model"] == GEMINI and first["entities"][0]["pick"]["selector"] == OPUS
    assert first["fallbacks"] == [] and _tiers(visits) == {"opus": 1}
    assert (OPUS, "extract") not in claude.made  # a chain for selection only
    second = _json(_ask(s, payload, claude=_Claude(), llm_factory=_fake(HALO)))
    assert second["entities"][0]["pick"]["selector"] == SONNET
    assert second["fallbacks"] == [
        {"stage": "admit", "from": OPUS, "to": SONNET, "reason": "opus_allowance_used"}
    ]


def test_opus_extraction_with_jev_or_an_openrouter_selector_keeps_those_selectors(
    cache_dir: Path, tmp_path: Path, visits: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _open_site(cache_dir, tmp_path)
    picker = FakeLLM(handler=lambda request: PICK, model=GEMINI)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    claude = _Claude(HALO)
    out = _json(_ask(s, {"text": HALO_TEXT, "selector": GEMINI}, claude=claude))
    assert out["model"] == OPUS and out["entities"][0]["pick"]["selector"] == GEMINI
    assert (OPUS, "select") not in claude.made and len(picker.requests) == 1


def test_a_refused_request_is_not_counted(cache_dir: Path, tmp_path: Path, visits: Any) -> None:
    s = _open_site(cache_dir, tmp_path)
    for payload in ({"top_k": 0}, {"text": " "}, {"model": "nobody/knows"}, {"selector": "x"}):
        assert _ask(s, payload, claude=_no_claude)[0] == 400
    missing = http_mod.handle(
        "POST", "/api/ner", {"x-real-ip": RAW_IP}, b'{"text": "a"}', settings=s
    )
    assert _error(missing) == (400, "bad_header")
    assert visits.sent == [] and visits.rows == {}


def test_no_address_visitor_id_or_key_reaches_a_response_or_a_log_line(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from ner_site import limits

    s = _open_site(cache_dir, tmp_path)
    visitor = limits.visitor_id({"x-real-ip": RAW_IP}, limits.default_salt(DB_URL))
    capsys.readouterr()
    limited = _claude_error("rate_limited")
    results = [
        _ask(s, {"text": HALO_TEXT, "selector": OPUS}, claude=_Claude(HALO)),
        _ask(s, claude=_Claude(opus_extract=[limited])),
        _ask(s, {"top_k": 99}),
        http_mod.handle("GET", "/api/health", dict(VISITOR_HEADERS), b"", settings=s),
    ]
    for _ in range(30):
        results.append(_ask(s))
    assert results[-1][0] == 429
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == len(results)
    for secret in (RAW_IP, visitor, ANTHROPIC_KEY, SENTINEL_KEY, DB_URL, "SENTINELpw", "10.0.0.1"):
        assert secret not in err, secret
        for _, headers, body in results:
            assert secret.encode() not in body and secret not in json.dumps(headers)
    # The visitor is the keyed hash of the address in every statement, never the address.
    params = [value for _, sent in visits.sent for value in sent]
    assert visitor in params and RAW_IP not in json.dumps(params)


def test_the_salt_setting_changes_the_visitor_id(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    from ner_site import limits

    s = _open_site(cache_dir, tmp_path, NER_SITE_VISITOR_SALT="salt-SENTINEL-0001")
    assert _ask(s)[0] == 200
    ((visitor, _, _),) = visits.rows
    assert visitor == limits.visitor_id({"x-real-ip": RAW_IP}, "salt-SENTINEL-0001")


def test_offline_mode_never_consults_the_limiter_or_claude(cache_dir: Path, tmp_path: Path) -> None:
    """The conftest fixture fails any attempt to open the limiter's database."""
    s = _open_site(cache_dir, tmp_path, NER_SITE_OFFLINE="1")
    result = http_mod.handle(
        "POST",
        "/api/ner",
        dict(VISITOR_HEADERS),
        json.dumps({"text": EXAMPLE_TEXT}).encode(),
        settings=s,
        claude_factory=_no_claude,
    )
    assert result[0] == 200, result[2]
    out = _json(result)
    assert out["model"] == "offline" and out["fallbacks"] == [] and out["notices"] == []


def test_a_deployment_without_the_anthropic_key_behaves_as_before(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """OpenRouter key, passcode and a Neon database: no limiter, no chain, no tier. (The
    conftest fixture fails any attempt to open the limiter's database.)"""
    s = _settings(
        cache_dir,
        tmp_path,
        OPENROUTER_API_KEY=SENTINEL_KEY,
        NER_SITE_PASSCODE=PASSCODE,
        DATABASE_URL=DB_URL,
    )
    headers = {**VISITOR_HEADERS, "x-passcode": PASSCODE}
    for _ in range(35):  # more than the hourly total
        result = _post(s, {"text": "nothing"}, headers=headers, llm_factory=_fake("{}"))
        assert _error(result) == (422, "extraction_schema")
    ok = _post(s, {"text": "nothing"}, headers=headers, llm_factory=_fake('{"entities": []}'))
    out = _json(ok)
    assert ok[0] == 200 and out["model"] == out["requested_model"] == GEMINI
    assert out["requested_selector"] == "retrieval" and out["fallbacks"] == []
    record = _last_log(capsys)
    assert "tier" not in record and record["model"] == GEMINI
    refused = _post(s, {"text": "a", "model": OPUS}, headers=headers, llm_factory=_never)
    assert _error(refused) == (400, "unknown_model")


def test_every_step_failing_is_a_502_whose_log_line_says_how_far_it_went(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _open_site(cache_dir, tmp_path)
    limited = _claude_error("rate_limited")
    result = _ask(
        s,
        claude=_Claude(opus_extract=[limited], sonnet_extract=[limited]),
        llm_factory=_fake(LLMFatalError(f"HTTP 401 for {SENTINEL_KEY}")),
    )
    assert _error(result) == (502, "provider_error")
    assert b"HTTP 401" in result[2] and SENTINEL_KEY.encode() not in result[2]
    record = _last_log(capsys)
    assert record["status"] == 502 and record["tier"] == "opus"
    assert [(f["stage"], f["from"], f["to"]) for f in record["fallbacks"]] == [
        ("extract", OPUS, SONNET),
        ("extract", SONNET, GEMINI),
    ]
    assert _tiers(visits) == {"opus": 1}  # admitted, so counted, though it failed later


def test_only_the_anthropic_key_is_a_configured_site(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path, OPENROUTER_API_KEY="")
    limited = _claude_error("rate_limited")
    out = _json(_ask(s, {"model": SONNET}, claude=_Claude(sonnet_extract=[limited])))
    # The chain ends at Sonnet: its error is the provider error, scrubbed.
    assert out["error"]["code"] == "provider_error"
    assert "rate_limited" in out["error"]["message"] and ANTHROPIC_KEY not in json.dumps(out)
    bare = _settings(cache_dir, tmp_path)
    message = _json(_ask(bare, claude=_no_claude))["error"]["message"]
    assert "ANTHROPIC_API_KEY" in message and "OPENROUTER_API_KEY" in message


# --- health: claude and visitor (spec section 6) -----------------------------------


def _health_of(settings: Settings, headers: dict[str, str] | None = None) -> dict[str, Any]:
    result = http_mod.handle(
        "GET",
        "/api/health",
        dict(VISITOR_HEADERS if headers is None else headers),
        b"",
        settings=settings,
    )
    assert result[0] == 200, result[2]
    assert RAW_IP.encode() not in result[2]
    return _json(result)


def test_health_reports_claude_and_the_calling_visitors_allowance(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path)
    health = _health_of(s)
    assert health["default_model"] == OPUS and health["default_selector"] == OPUS
    assert health["models"][:2] == [OPUS, SONNET] and health["passcode_required"] is False
    assert health["claude"] == {"status": "ok"} and health["llm"] == {"status": "ok"}
    visitor = health["visitor"]
    assert visitor["window"] == "hour" and 1 <= visitor["day_resets_in_s"] <= 86400
    assert (visitor["opus_left_today"], visitor["sonnet_left_today"]) == (20, 40)
    assert (visitor["own_left"], visitor["own_left_today"]) == (30, 60)
    assert (visitor["limits"], visitor["tier"]) == ("on", "opus")
    assert (visitor["opus_left"], visitor["sonnet_left"]) == (10, 20)
    assert 1 <= visitor["resets_in_s"] <= 3600
    assert visits.rows == {}  # looking records nothing

    for _ in range(3):
        _ask(s)
    assert _health_of(s)["visitor"]["opus_left"] == 7
    for _ in range(7):
        _ask(s)
    used = _health_of(s)["visitor"]
    assert (used["tier"], used["opus_left"], used["sonnet_left"]) == ("sonnet", 0, 20)
    for _ in range(20):
        _ask(s)
    blocked = _health_of(s)["visitor"]
    assert (blocked["limits"], blocked["tier"]) == ("on", "blocked")
    assert (blocked["opus_left"], blocked["sonnet_left"]) == (0, 0)
    assert _tiers(visits) == {"opus": 10, "sonnet": 20}
    # The allowance is the caller's own.
    other = _health_of(s, {"x-real-ip": "198.51.100.9"})["visitor"]
    assert (other["tier"], other["opus_left"]) == ("opus", 10)


def test_health_says_why_claude_is_unavailable(
    cache_dir: Path, tmp_path: Path, visits: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site import fallback
    from ner_site.schema import RetrieverStatus

    s = _open_site(cache_dir, tmp_path)
    refused = RetrieverStatus(status="unavailable", reason="key refused")
    monkeypatch.setattr(http_mod, "check_claude_key", lambda settings: refused)
    assert _health_of(s)["claude"] == {"status": "unavailable", "reason": "key refused"}

    # A check that could not be completed is not an outage, and says so.
    unchecked = RetrieverStatus(status="ok", reason="key not checked (Anthropic unreachable: X)")
    monkeypatch.setattr(http_mod, "check_claude_key", lambda settings: unchecked)
    assert _health_of(s)["claude"] == {"status": "ok", "reason": unchecked.reason}

    # What a request met is known without asking again, and wins while it is remembered.
    def never(settings: Settings) -> Any:
        pytest.fail("the key was checked while Claude is known to be unusable")

    monkeypatch.setattr(http_mod, "check_claude_key", never)
    fallback.block_claude("balance_exhausted")
    assert _health_of(s)["claude"] == {"status": "unavailable", "reason": "balance exhausted"}
    fallback.block_claude("key_refused")
    assert _health_of(s)["claude"] == {"status": "unavailable", "reason": "key refused"}


def test_health_without_the_anthropic_key_offers_no_claude_and_no_allowance(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(settings: Settings) -> Any:
        pytest.fail("no Anthropic key, nothing to check")

    monkeypatch.setattr(http_mod, "check_claude_key", never)
    s = _settings(
        cache_dir,
        tmp_path,
        OPENROUTER_API_KEY=SENTINEL_KEY,
        NER_SITE_PASSCODE=PASSCODE,
        DATABASE_URL=DB_URL,
        NER_SITE_SELECTOR="",
    )
    health = _health_of(s)
    assert health["claude"] == {"status": "unavailable", "reason": "key not set"}
    assert "visitor" not in health and health["llm"] == {"status": "ok"}
    assert health["models"] == [GEMINI, "openai/gpt-6-luna"] and health["default_model"] == GEMINI
    assert health["selectors"] == ["retrieval", "jev", GEMINI, "openai/gpt-6-luna"]
    assert health["default_selector"] == "jev" and health["passcode_required"] is True
    offline = _health_of(_open_site(cache_dir, tmp_path, NER_SITE_OFFLINE="1"))
    assert "claude" not in offline and "visitor" not in offline


def test_health_with_the_limiter_unreachable_or_off(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    from xwalk.retrieval.postgres import PostgresError

    visits.fail = PostgresError("could not reach Neon at ep-x-pooler", code="08006")
    down = _health_of(_open_site(cache_dir, tmp_path))["visitor"]
    assert (down["limits"], down["tier"]) == ("unavailable", "blocked")
    assert (down["opus_left"], down["sonnet_left"]) == (0, 0) and down["resets_in_s"] >= 1
    off = _health_of(_open_site(cache_dir, tmp_path, DATABASE_URL=""))["visitor"]
    assert (off["limits"], off["tier"], off["opus_left"]) == ("unavailable", "blocked", 0)
    private = _open_site(cache_dir, tmp_path, DATABASE_URL="", NER_SITE_PASSCODE=PASSCODE)
    unmetered = _health_of(private)["visitor"]
    assert (unmetered["limits"], unmetered["tier"]) == ("off", "opus")
    assert (unmetered["opus_left"], unmetered["sonnet_left"]) == (10, 20)


def test_health_with_only_the_anthropic_key(cache_dir: Path, tmp_path: Path) -> None:
    health = _health_of(_open_site(cache_dir, tmp_path, OPENROUTER_API_KEY="", DATABASE_URL=""))
    assert health["models"] == [OPUS, SONNET] and health["selectors"] == ["retrieval", OPUS, SONNET]
    assert health["claude"] == {"status": "ok"}
    assert health["llm"]["status"] == "unavailable"
    assert "OPENROUTER_API_KEY is not set" in health["llm"]["reason"]
    assert "503" not in health["llm"]["reason"]  # the site runs, on Claude alone


def test_the_limiter_has_its_own_executor_that_never_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counting a visit is not idempotent and the indexes' executor retries; a slow
    database must become "limits unavailable" within a few seconds, not a slow request."""
    built: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(http_mod, "NeonHTTPExecutor", lambda url, **kw: built.append((url, kw)))
    REAL_LIMITS_EXECUTOR(DB_URL)
    assert built == [(DB_URL, {"timeout": http_mod.LIMITS_TIMEOUT_S, "max_retries": 0})]
    assert http_mod.LIMITS_TIMEOUT_S <= 5 < indexes_mod.NEON_TIMEOUT_S


def test_the_limiter_is_kept_per_warm_instance_and_reset_forgets_it(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path)
    assert _ask(s)[0] == 200 and _ask(s)[0] == 200
    assert visits.opened == [DB_URL] and visits.tags().count("site_visits_table") == 1
    http_mod.reset()
    assert _ask(s)[0] == 200 and visits.opened == [DB_URL, DB_URL]
    # A connection string the executor cannot use is "limits unavailable", not a 500.
    bad = _open_site(cache_dir, tmp_path, DATABASE_URL="postgresql:///nohost")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(http_mod, "make_limits_executor", REAL_LIMITS_EXECUTOR)
        result = _ask(bad, claude=_no_claude)
    assert _error(result) == (503, "limits_unavailable")


OWN_KEY = 'visitor-sentinel-"quoted"-\\tail'
OWN_HEADERS = {**VISITOR_HEADERS, "Authorization": "Bearer " + OWN_KEY}


@pytest.fixture
def own_clients(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    from ner_site import providers

    made = []

    class Client(FakeLLM):
        async def aclose(self) -> None:
            made.append(("closed", self.model, ""))

    def build(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        assert credential.token() == OWN_KEY
        made.append((credential.provider, model, purpose))
        return Client([HALO if purpose == "extract" else PICK], model=model)

    def never(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("an own-key run reached a site client, key probe or Claude block")

    monkeypatch.setattr(providers, "own_llm", build)
    for name in ("make_chain", "make_select_llm", "make_decider", "check_key", "check_claude_key"):
        monkeypatch.setattr(http_mod, name, never)
    monkeypatch.setattr(http_mod.fallback, "claude_block", never)
    return made


@pytest.mark.parametrize(
    "provider,model", [("anthropic", OPUS), ("openai", "gpt-6-luna"), ("openrouter", GEMINI)]
)
def test_own_stages_use_only_the_chosen_provider(
    provider: str,
    model: str,
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    own_clients: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = _settings(cache_dir, tmp_path, DATABASE_URL=DB_URL, NER_SITE_OWN_PROVIDERS=ALL_OWN)
    result = _ask(
        s, {"provider": provider, "text": HALO_TEXT, "selector": model}, headers=OWN_HEADERS
    )
    assert result[0] == 200, result[2]
    out = _json(result)
    assert (out["provider"], out["credential_source"], out["model"]) == (provider, "visitor", model)
    assert out["fallbacks"] == out["notices"] == []
    assert out["entities"][0]["pick"]["selector"] == model
    assert own_clients[:2] == [(provider, model, "extract"), (provider, model, "select")]
    assert own_clients.count(("closed", model, "")) == 2
    assert _tiers(visits) == {"own": 1}
    assert OWN_KEY not in str(visits.sent)
    record = _last_log(capsys)
    assert (record["provider"], record["credential"], record["tier"]) == (
        provider,
        "visitor",
        "own",
    )


@pytest.mark.parametrize("database", ["", DB_URL])
def test_own_runs_when_site_limits_unavailable(
    database: str, cache_dir: Path, tmp_path: Path, visits: Any, own_clients: Any
) -> None:
    visits.fail = RuntimeError("database unavailable")
    s = _own_site(cache_dir, tmp_path, DATABASE_URL=database)
    result = _post(
        s, {"text": HALO_TEXT, "provider": "anthropic"}, headers=OWN_HEADERS, llm_factory=_never
    )
    assert result[0] == 200 and _json(result)["selector"] == "retrieval"
    assert own_clients == [("anthropic", OPUS, "extract"), ("closed", OPUS, "")]
    assert visits.rows == {}


def test_own_refusals_precede_admission_and_construction(
    cache_dir: Path, tmp_path: Path, visits: Any, own_clients: Any
) -> None:
    s = _own_site(cache_dir, tmp_path, NER_SITE_PASSCODE=PASSCODE)
    payload = {"provider": "openai"}
    headers = {**OWN_HEADERS, "X-Passcode": PASSCODE}
    duplicate = {"authorization": OWN_HEADERS["Authorization"]}
    for body, sent, error in [
        (payload, {**OWN_HEADERS, **duplicate}, (401, "passcode_required")),
        (payload, {**headers, **duplicate}, (400, "bad_credential")),
        (payload, {**headers, "Authorization": ["Bearer " + OWN_KEY] * 2}, (400, "bad_credential")),
        ({**payload, "prompt": 3}, headers, (400, "bad_json")),
        ({**payload, "text": OWN_KEY}, headers, (400, "bad_credential")),
        ({**payload, "model": OWN_KEY}, headers, (400, "unknown_model")),
        ({**payload, "selector": "jev"}, headers, (400, "unknown_selector")),
        ({"provider": "site"}, headers, (400, "bad_credential")),
    ]:
        result = _ask(s, body, headers=sent)
        assert _error(result) == error
        assert OWN_KEY not in _json(result)["error"]["message"]
    assert not visits.sent and not own_clients


def test_duplicate_authorization_survives_shim(
    cache_dir: Path, tmp_path: Path, own_clients: Any
) -> None:
    from email.message import Message
    from io import BytesIO
    from types import SimpleNamespace

    body = b'{"text":"x","provider":"openai"}'
    headers = Message()
    for k, v in {**OWN_HEADERS, "content-length": str(len(body))}.items():
        headers[k] = v
    headers["authorization"] = "Bearer " + OWN_KEY
    request = SimpleNamespace(headers=headers, command="POST", path="/api/ner", rfile=BytesIO(body))
    result = http_mod._shim_response(request, _own_only(cache_dir, tmp_path))
    assert _error(result) == (400, "bad_credential")
    assert "exactly one" in _json(result)["error"]["message"]
    assert not own_clients


@pytest.mark.parametrize("window,own", [("day", False), ("hour", True), ("day", True)])
def test_daily_and_own_flood_limit_wording(
    window: str,
    own: bool,
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_mod, "_now", lambda: 43200)
    env = (
        {"NER_SITE_OWN_PER_" + window.upper(): "0"}
        if own
        else {"NER_SITE_SONNET_PER_DAY": "0", "NER_SITE_OPUS_PER_DAY": "0"}
    )
    s = _own_site(cache_dir, tmp_path, **env)
    result = _ask(
        s, {"provider": "openai"} if own else {}, headers=OWN_HEADERS if own else VISITOR_HEADERS
    )
    assert _error(result) == (429, "visitor_limit_reached")
    seconds = 3600 if window == "hour" else 43200
    assert result[1]["Retry-After"] == str(seconds)
    expected = (
        f"too many requests with your own key; try again in {seconds // 60} minutes"
        if own
        else "you have used today's requests; they reset at midnight UTC, or use your own API key"
    )
    assert _json(result)["error"]["message"] == expected


def test_health_lists_own_models_even_without_site_credentials(
    cache_dir: Path, tmp_path: Path
) -> None:
    fixture = json.loads((PROJECT / "tools/fixtures/health.json").read_text())
    listed = _own_only(cache_dir, tmp_path)
    assert _health_of(listed)["own_keys"] == fixture["own_keys"]
    health = _health_of(
        _settings(
            cache_dir,
            tmp_path,
            NER_SITE_OWN_PROVIDERS="openai",
            NER_SITE_OWN_MODELS_OPENAI="custom,second",
        )
    )
    assert health["own_keys"] == {
        "enabled": True,
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "models": ["custom", "second"],
                "default_model": "custom",
            }
        ],
    }
    for off in ({}, {"NER_SITE_OWN_PROVIDERS": ""}):
        assert _health_of(_settings(cache_dir, tmp_path, **off))["own_keys"] == {
            "enabled": False,
            "providers": [],
        }


def test_concurrent_own_requests_keep_keys_local(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from ner_site import providers

    barrier = threading.Barrier(2)
    keys = [OWN_KEY, 'other-sentinel-"-key-\\0002']
    s = _own_site(cache_dir, tmp_path, DATABASE_URL="")

    def build(credential: Any, model: str, **kwargs: Any) -> Any:
        key = credential.token()

        def reply(request: Any) -> str:
            barrier.wait(timeout=5)
            label = "first" if key == keys[0] else "second"
            return _answer((label, "other", 0, len(label)), (key, "other", 0, 0))

        class Client(FakeLLM):
            async def aclose(self) -> None:
                pass

        return Client(handler=reply, model=model)

    real_log = http_mod._log

    def log(record: dict[str, Any], *args: Any) -> None:
        record["nested"] = [keys[0 if record["text_chars"] == 5 else 1]]
        real_log(record, *args)

    monkeypatch.setattr(http_mod, "_log", log)
    monkeypatch.setattr(providers, "own_llm", build)

    def send(pair: tuple[str, str]) -> Handled:
        key, label = pair
        return _ask(
            s,
            {"provider": "openai", "text": label},
            headers={**VISITOR_HEADERS, "Authorization": "Bearer " + key},
            claude=_no_claude,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, zip(keys, ("first", "second"), strict=True)))
    for result, label in zip(results, ("first", "second"), strict=True):
        assert result[0] == 200, result[2]
        out = _json(result)
        assert [e["text"] for e in out["entities"]] == [label]
        assert "[redacted]" in str(out["warnings"])
        for key in keys:
            assert json.dumps(key)[1:-1] not in result[2].decode() and key not in str(result[1])
            assert all(key not in w for w in out["warnings"])
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    for line in lines:
        assert json.loads(line)["credential"] == "visitor"
        assert json.loads(line)["nested"] == ["[redacted]"]
        assert all(key not in line and json.dumps(key)[1:-1] not in line for key in keys)


def test_own_fixture_errors_match_the_real_credential_reader(
    cache_dir: Path, tmp_path: Path
) -> None:
    directory = PROJECT / "tools/fixtures"
    s = _own_only(cache_dir, tmp_path)
    spec = importlib.util.spec_from_file_location(
        "serve_local_own_test", PROJECT / "tools/serve_local.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.canned_backend(directory, "")
    for provider, value in [
        ("openai", ""),
        ("openai", "Bearer short"),
        ("openai", "Token abc"),
        ("site", "Bearer " + OWN_KEY),
        ("disabled", "Bearer " + OWN_KEY),
    ]:
        headers = {**POST_HEADERS, **({"authorization": value} if value else {})}
        body = {"provider": provider, "text": "sample"}
        real = _post(s, body, headers=headers, llm_factory=_never)
        canned = backend("POST", "/api/ner", headers, json.dumps(body).encode())
        assert _json(canned) == _json(real)
        if value == "Bearer short":
            assert _json(real) == json.loads((directory / "error-own-bad.json").read_text())


def test_disabled_own_health_fixture_matches_live_balances(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, visits: Any
) -> None:
    fixture = json.loads((PROJECT / "tools/fixtures/health-own-off.json").read_text())
    now = 56220 - 3600
    monkeypatch.setattr(http_mod, "_now", lambda: now)
    s = _open_site(
        cache_dir, tmp_path, NER_SITE_OWN_PROVIDERS="", NER_SITE_DENSE_DIR=str(PROJECT / "indexes")
    )
    for _ in range(2):
        assert _ask(s)[0] == 200
    now += 3600
    for _ in range(3):
        assert _ask(s)[0] == 200
    health = _health_of(s)
    assert fixture == health


def test_own_selector_construction_failure_closes_extractor(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site import providers

    closed = []

    class Client:
        async def aclose(self) -> None:
            closed.append(True)

    def build(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        if purpose == "extract":
            return Client()
        raise RuntimeError("upstream echoed " + credential.token())

    monkeypatch.setattr(providers, "own_llm", build)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "openai", "selector": "gpt-6-luna"},
        headers=OWN_HEADERS,
    )
    assert _error(result) == (502, "provider_error")
    assert _json(result)["error"]["message"] == "OpenAI could not complete the request (error)."
    assert closed == [True]


def test_bad_route_does_not_echo_a_bearer_in_the_path(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _settings(cache_dir, tmp_path)
    result = http_mod.handle("POST", "/" + OWN_KEY, OWN_HEADERS, b"", settings=s)
    assert _error(result) == (404, "not_found")
    assert OWN_KEY not in _json(result)["error"]["message"]
    assert OWN_KEY not in json.loads(capsys.readouterr().err)["path"]


def test_daily_opus_stepdown_tells_the_visitor_which_allowance_was_used(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _open_site(cache_dir, tmp_path, NER_SITE_OPUS_PER_DAY="0")
    assert _json(_ask(s))["notices"] == [
        "Your Opus allowance for today is used; this ran on Claude Sonnet 5.5."
    ]


def test_invalid_provider_type_is_bad_credential(cache_dir: Path, tmp_path: Path) -> None:
    result = _post(
        _own_only(cache_dir, tmp_path), {"text": "ok", "provider": ["openai"]}, headers=OWN_HEADERS
    )
    assert _error(result) == (400, "bad_credential")


def test_unexpected_own_mapping_failure_keeps_internal_envelope(
    cache_dir: Path,
    tmp_path: Path,
    own_clients: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ner_site import service

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(OWN_KEY)

    monkeypatch.setattr(service, "map_entities", broken)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "openai", "text": HALO_TEXT},
        headers=OWN_HEADERS,
    )
    assert _error(result) == (500, "internal")
    record = _last_log(capsys)
    assert record["exception"] == "RuntimeError"
    assert OWN_KEY not in json.dumps(record) and "locals" not in record


def test_an_own_key_is_redacted_before_pick_explanations_are_capped(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site import providers

    class Client(FakeLLM):
        async def aclose(self) -> None:
            pass

    pick = json.loads(PICK)
    pick["explanation"] = "x" * 280 + OWN_KEY

    def build(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        return Client([HALO if purpose == "extract" else json.dumps(pick)], model=model)

    monkeypatch.setattr(providers, "own_llm", build)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "openai", "selector": "gpt-6-luna", "text": HALO_TEXT},
        headers=OWN_HEADERS,
    )
    assert _json(result)["entities"][0]["pick"]["explanation"] == "x" * 280 + "[redacted]"


def test_ignored_database_url_is_not_read_into_settings_cache() -> None:
    class Env(dict):
        def get(self, key: str, default: Any = None) -> Any:
            assert key != "DATABASE_URL"
            return super().get(key, default)

    assert http_mod.env_settings(Env()).database_url is None
    assert http_mod.env_settings(Env(NER_SITE_OWN_PROVIDERS="")).database_url is None
    assert http_mod.env_settings(Env(NER_SITE_OFFLINE="1")).database_url is None


def test_settings_error_cache_does_not_retain_the_first_request_frame() -> None:
    error = http_mod.env_settings({"NER_SITE_OWN_PROVIDERS": "unknown"})
    assert isinstance(error, Exception)
    assert error.__traceback__ is None and error.__context__ is None


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
def test_provider_diagnostics_cannot_log_an_echoed_visitor_key(
    provider: str,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import logging

    import httpx2
    from ner_site import providers

    module = httpx2 if provider == "anthropic" else httpx

    def refuse(request: Any) -> Any:
        assert request.headers["Authorization"] == "Bearer " + OWN_KEY
        return module.Response(
            401,
            headers={"request-id": OWN_KEY},
            extensions={"reason_phrase": OWN_KEY.encode()},
            json={"error": {"message": OWN_KEY}},
        )

    real = providers.own_llm

    def build(*args: Any, **kwargs: Any) -> Any:
        return real(*args, transport=module.MockTransport(refuse), backoff_base=0.0, **kwargs)

    monkeypatch.setattr(providers, "own_llm", build)
    caplog.set_level(logging.DEBUG)
    result = _ask(_own_only(cache_dir, tmp_path), {"provider": provider}, headers=OWN_HEADERS)
    assert _error(result) == (401, "provider_key_refused")
    assert OWN_KEY not in caplog.text
    assert (
        OWN_KEY not in _json(result)["error"]["message"] and OWN_KEY not in capsys.readouterr().err
    )
    logging.getLogger("httpx").info("diagnostics outside the own-key request")
    assert "diagnostics outside the own-key request" in caplog.text


def test_a_key_duplicated_in_address_headers_is_never_hashed(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, own_clients: Any, visits: Any
) -> None:
    original = http_mod.limits.visitor_id

    def identify(headers: Any, salt: str) -> str:
        assert all(OWN_KEY not in headers.get(k, "") for k in http_mod.limits.ADDRESS_HEADERS)
        return original(headers, salt)

    monkeypatch.setattr(http_mod.limits, "visitor_id", identify)
    result = _ask(
        _own_site(cache_dir, tmp_path),
        {"provider": "openai"},
        headers={**OWN_HEADERS, "x-real-ip": OWN_KEY},
    )
    assert result[0] == 200 and _tiers(visits) == {"own": 1}


def test_canned_backend_never_copies_a_key_from_request_fields() -> None:
    spec = importlib.util.spec_from_file_location(
        "serve_local_fields", PROJECT / "tools/serve_local.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.canned_backend(PROJECT / "tools/fixtures", "")
    for field, code in [
        ("text", "bad_credential"),
        ("prompt", "bad_credential"),
        ("model", "unknown_model"),
        ("selector", "unknown_selector"),
    ]:
        body = json.dumps({"text": "sample", "provider": "openai", field: OWN_KEY}).encode()
        result = backend(
            "POST", "/api/ner", {**POST_HEADERS, "authorization": "Bearer " + OWN_KEY}, body
        )
        assert _error(result) == (400, code)
        assert OWN_KEY not in _json(result)["error"]["message"]


def test_duplicate_authorization_over_http(
    shim_server: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NER_SITE_OFFLINE", "0")
    monkeypatch.setenv("NER_SITE_OWN_PROVIDERS", "openai")
    body = b'{"text":"sample","provider":"openai"}'
    first = b"Authorization: Bearer first-synthetic-key-0001\r\n"
    second = b"aUThorization: Bearer second-synthetic-key-0002\r\n"
    status, wire = _raw_exchange(
        shim_server,
        b"POST /api/ner HTTP/1.0\r\n"
        + first
        + second
        + b"X-Requested-With: xwalk-ner\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body,
    )
    assert b"400" in status and json.loads(wire)["error"]["code"] == "bad_credential"
    assert "exactly one" in json.loads(wire)["error"]["message"]


def test_health_never_reads_authorization(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site import providers

    class Headers(dict):
        def __getitem__(self, key: str) -> Any:
            assert key.lower() != "authorization"
            return super().__getitem__(key)

        def items(self) -> Any:
            pytest.fail("health must not enumerate header values")

    monkeypatch.setattr(
        providers, "read_credential", lambda *a, **kw: pytest.fail("health read a credential")
    )
    result = http_mod.handle(
        "GET", "/api/health", Headers(OWN_HEADERS), b"", settings=_settings(cache_dir, tmp_path)
    )
    assert result[0] == 200


def test_offline_settings_never_offer_chat_on_visitor_keys(cache_dir: Path, tmp_path: Path) -> None:
    from dataclasses import replace

    offline = replace(_own_site(cache_dir, tmp_path), offline=True)
    assert _health_of(offline)["own_keys"] == {"enabled": False, "providers": []}
    assert _error(_ask(offline, {"provider": "openai"}, headers=OWN_HEADERS)) == (
        400,
        "bad_credential",
    )


@pytest.mark.parametrize("off", [{}, {"NER_SITE_OWN_PROVIDERS": ""}], ids=["unset", "empty"])
@pytest.mark.parametrize("window", ["hour", "day"])
def test_used_site_allowance_with_own_keys_off(
    window: str, off: dict[str, str], cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    env = {f"NER_SITE_{model}_PER_{window.upper()}": "0" for model in ("OPUS", "SONNET")}
    s = _open_site(cache_dir, tmp_path, **off, **env)
    result = _ask(s, claude=_no_claude)
    assert _error(result) == (429, "visitor_limit_reached")
    assert "use your own" not in _json(result)["error"]["message"]


def test_an_unreachable_limiter_names_no_own_key_while_own_keys_are_off(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    visits.fail = RuntimeError("database unavailable")
    result = _ask(_open_site(cache_dir, tmp_path), claude=_no_claude)
    assert _error(result) == (503, "limits_unavailable")
    assert _json(result)["error"]["message"] == (
        "the site cannot count requests right now; try again shortly"
    )


def test_outage_health_fixture_reports_uncounted_own_runs(
    cache_dir: Path, tmp_path: Path, visits: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_mod, "_now", lambda: 56220)
    visits.fail = RuntimeError("unavailable")
    fixture = json.loads((PROJECT / "tools/fixtures/health-unavailable.json").read_text())
    assert _health_of(_open_site(cache_dir, tmp_path))["visitor"] == fixture["visitor"]


def test_own_error_scrubber_covers_response_headers_and_body(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, own_clients: Any
) -> None:
    from ner_site.schema import ServiceError

    async def failed(*args: Any, **kwargs: Any) -> Any:
        raise ServiceError("provider_error", OWN_KEY, status=502, headers={"X-Echo": OWN_KEY})

    monkeypatch.setattr(http_mod, "run", failed)
    result = _ask(_own_only(cache_dir, tmp_path), {"provider": "openai"}, headers=OWN_HEADERS)
    assert _json(result)["error"]["message"] == result[1]["X-Echo"] == "[redacted]"


def test_log_identifies_requested_provider_even_on_a_missing_credential(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = _own_only(cache_dir, tmp_path)
    assert _error(_ask(s, {"provider": "openai"})) == (400, "bad_credential")
    record = _last_log(capsys)
    assert (record["provider"], record["credential"]) == ("openai", "visitor")
    assert _error(_ask(s)) == (503, "not_configured")
    record = _last_log(capsys)
    assert (record["provider"], record["credential"]) == ("site", "site")


def test_json_escaped_provider_echo_cannot_enter_the_mention_regex_cache(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site import extract as extraction
    from ner_site import providers

    original = extraction.locate
    seen = []

    def locate(text: str, mention: str, *args: Any, **kwargs: Any) -> Any:
        seen.append(mention)
        return original(text, mention, *args, **kwargs)

    monkeypatch.setattr(extraction, "locate", locate)

    class Client(FakeLLM):
        async def aclose(self) -> None:
            pass

    def build(credential: Any, model: str, **options: Any) -> Any:
        return Client([_answer((credential.token(), "other", 0, 1))], model=model)

    monkeypatch.setattr(providers, "own_llm", build)
    result = _ask(_own_only(cache_dir, tmp_path), {"provider": "openai"}, headers=OWN_HEADERS)
    assert result[0] == 200
    assert seen and set(seen) == {"[redacted]"}


# --- own keys are off unless providers are listed (final review, A1 and A2) ------------

BASIC = "Basic dXNlcjpwYXNzd29yZA=="


def _before_own_keys(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    """The deployment from before this feature: the OpenRouter key, a passcode, explicit
    model lists, the bundled backend, and a ``DATABASE_URL`` some integration left."""
    return _settings(
        cache_dir,
        tmp_path,
        OPENROUTER_API_KEY=SENTINEL_KEY,
        NER_SITE_PASSCODE=PASSCODE,
        NER_SITE_MODELS="google/gemini-3.1-flash-lite,openai/gpt-6-luna",
        NER_SITE_SELECTORS="jev,google/gemini-3.1-flash-lite",
        **{"DATABASE_URL": "file:./dev.db", **env},
    )


def test_own_keys_are_off_by_default(cache_dir: Path, tmp_path: Path) -> None:
    s = _settings(cache_dir, tmp_path)
    assert _health_of(s)["own_keys"] == {"enabled": False, "providers": []}
    for provider in ("anthropic", "openai", "openrouter"):
        result = _post(s, {"text": "x", "provider": provider}, headers=OWN_HEADERS)
        assert _error(result) == (400, "bad_credential")
        assert OWN_KEY not in result[2].decode()


def test_the_deployment_from_before_own_keys_loads_without_a_database(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _before_own_keys(cache_dir, tmp_path)
    assert s.database_url is None and s.own_providers == ()
    health = _health_of(s)
    assert health["own_keys"] == {"enabled": False, "providers": []} and "visitor" not in health


@pytest.mark.parametrize("provider", [None, "site"])
@pytest.mark.parametrize(
    "authorization", [BASIC, "Bearer proxy-token-0123456789abcdef", "", [BASIC, BASIC]]
)
def test_authorization_on_a_site_request_is_ignored_while_own_keys_are_off(
    provider: str | None,
    authorization: Any,
    cache_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP basic auth, ``curl -u`` and an auth proxy in front of the site all add the
    header; before own keys nothing looked at it."""
    s = _before_own_keys(cache_dir, tmp_path)
    headers: dict[str, Any] = {**POST_HEADERS, "x-passcode": PASSCODE}
    headers["Authorization"] = authorization
    payload = {"text": "nothing", **({} if provider is None else {"provider": provider})}
    result = http_mod.handle(
        "POST",
        "/api/ner",
        headers,
        json.dumps(payload).encode(),
        settings=s,
        llm_factory=_fake('{"entities": []}'),
    )
    assert result[0] == 200, result[2]
    out = _json(result)
    assert (out["provider"], out["credential_source"]) == ("site", "site")
    line = capsys.readouterr().err
    assert "dXNlcjpwYXNzd29yZA" not in line and "proxy-token" not in line
    assert "dXNlcjpwYXNzd29yZA" not in result[2].decode()


def test_naming_an_own_provider_is_refused_while_own_keys_are_off(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _before_own_keys(cache_dir, tmp_path)
    headers = {**OWN_HEADERS, "x-passcode": PASSCODE}
    result = _post(s, {"text": "x", "provider": "openai"}, headers=headers, llm_factory=_never)
    assert _error(result) == (400, "bad_credential")


def test_authorization_on_a_site_request_is_refused_while_own_keys_are_on(
    cache_dir: Path, tmp_path: Path
) -> None:
    s = _before_own_keys(cache_dir, tmp_path, NER_SITE_OWN_PROVIDERS="openai", DATABASE_URL="")
    headers = {**POST_HEADERS, "x-passcode": PASSCODE, "Authorization": BASIC}
    result = _post(s, {"text": "x"}, headers=headers, llm_factory=_never)
    assert _error(result) == (400, "bad_credential")


# --- own-key failures in the log, and a close that fails (final review, C1 and C2) -----


def _own_build(monkeypatch: pytest.MonkeyPatch, complete: Any, close: Any = None) -> None:
    """Own-key clients whose ``complete`` and ``aclose`` do what the test says."""
    from ner_site import providers

    class Client(FakeLLM):
        async def complete(self, request: Any) -> Any:
            return await complete(self, request)

        async def aclose(self) -> None:
            if close is not None:
                await close(self)

    def build(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        return Client([HALO if purpose == "extract" else PICK], model=model)

    monkeypatch.setattr(providers, "own_llm", build)


def test_a_failure_building_an_own_client_is_named_in_the_log(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ner_site import providers

    def build(credential: Any, model: str, **options: Any) -> Any:
        raise RuntimeError("the client refused to be built for " + credential.token())

    monkeypatch.setattr(providers, "own_llm", build)
    result = _ask(_own_only(cache_dir, tmp_path), {"provider": "openai"}, headers=OWN_HEADERS)
    assert _error(result) == (502, "provider_error")
    assert _json(result)["error"]["message"] == "OpenAI could not complete the request (error)."
    record = _last_log(capsys)
    assert record["own_error"] == "RuntimeError" and record["code"] == "provider_error"
    assert "exception" not in record and "where" not in record
    for text in (json.dumps(record), result[2].decode()):
        assert OWN_KEY not in text and json.dumps(OWN_KEY)[1:-1] not in text


@pytest.mark.parametrize(
    ("failure", "status", "code", "named"),
    [
        (LLMFatalError("OpenAI request failed (http_5xx)"), 502, "provider_error", "LLMFatalError"),
        (
            LLMRetryableError("HTTP 429: slow down"),
            429,
            "provider_rate_limited",
            "LLMRetryableError",
        ),
        (TimeoutError("took too long"), 502, "provider_error", "TimeoutError"),
    ],
)
def test_a_provider_failure_on_an_own_key_is_named_in_the_log(
    failure: Exception,
    status: int,
    code: str,
    named: str,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import asyncio

    async def fail(client: Any, request: Any) -> Any:
        if isinstance(failure, TimeoutError):
            raise asyncio.TimeoutError()
        raise failure

    _own_build(monkeypatch, fail)
    result = _ask(_own_only(cache_dir, tmp_path), {"provider": "openai"}, headers=OWN_HEADERS)
    assert _error(result) == (status, code)
    record = _last_log(capsys)
    assert record["own_error"] == named and "exception" not in record


@pytest.mark.parametrize("where", ["client", "stage"])
def test_a_bug_on_the_own_path_is_a_500_with_its_type_and_frames_in_the_log(
    where: str,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``KeyError`` is not the provider failing. It used to leave as 502
    ``provider_error`` with nothing in the log to find it by."""
    from ner_site import service

    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise KeyError(OWN_KEY)

    async def fine(client: Any, request: Any) -> Any:
        return await FakeLLM.complete(client, request)

    if where == "client":
        _own_build(monkeypatch, broken)
    else:
        _own_build(monkeypatch, fine)
        monkeypatch.setattr(service, "extract", broken)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "openai", "text": HALO_TEXT},
        headers=OWN_HEADERS,
    )
    assert _error(result) == (500, "internal")
    record = _last_log(capsys)
    assert record["exception"] == "KeyError" and record["code"] == "internal"
    assert record["where"] and any(frame.startswith("service.py:") for frame in record["where"])
    assert "own_error" not in record and "locals" not in record
    assert (record["provider"], record["credential"]) == ("openai", "visitor")
    for text in (json.dumps(record), result[2].decode()):
        assert OWN_KEY not in text and json.dumps(OWN_KEY)[1:-1] not in text


@pytest.mark.parametrize("selector", ["retrieval", "gpt-6-luna"])
def test_a_close_that_fails_does_not_replace_the_result(
    selector: str,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []

    async def answer(client: Any, request: Any) -> Any:
        return await FakeLLM.complete(client, request)

    async def close(client: Any) -> None:
        closed.append(client.model)
        raise OSError("connection reset while closing, said " + OWN_KEY)

    _own_build(monkeypatch, answer, close)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "openai", "text": HALO_TEXT, "selector": selector},
        headers=OWN_HEADERS,
    )
    assert result[0] == 200, result[2]
    out = _json(result)
    assert out["credential_source"] == "visitor" and out["entities"][0]["text"] == "haloperidol"
    assert len(closed) == (1 if selector == "retrieval" else 2)  # every stage was still closed
    record = _last_log(capsys)
    assert record["status"] == 200 and record["own_close_error"] == "OSError"
    assert "own_error" not in record and "code" not in record
    assert OWN_KEY not in json.dumps(record) and json.dumps(OWN_KEY)[1:-1] not in json.dumps(record)


def test_a_close_that_fails_does_not_replace_the_runs_own_error(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(client: Any, request: Any) -> Any:
        raise LLMFatalError("HTTP 401: no")

    async def close(client: Any) -> None:
        raise RuntimeError("close failed")

    _own_build(monkeypatch, fail, close)
    result = _ask(_own_only(cache_dir, tmp_path), {"provider": "openai"}, headers=OWN_HEADERS)
    assert _error(result) == (401, "provider_key_refused")
    record = _last_log(capsys)
    assert (record["own_error"], record["own_close_error"]) == ("LLMFatalError", "RuntimeError")


# --- second review round: cookies, a refusal during selection, a bug in an adapter -----

PICKS_TEXT = "She took haloperidol, later ketamine, and aspirin."


def _real_own_clients(monkeypatch: pytest.MonkeyPatch, respond: Any) -> list[Any]:
    """The real own-key clients (``providers.own_llm``) over a mock Anthropic: `respond`
    gets each request and whether it is a pick, and returns an ``httpx2.Response``."""
    import httpx2
    from ner_site import providers

    seen: list[Any] = []
    real = providers.own_llm

    def handler(request: Any) -> Any:
        seen.append(request)
        return respond(request, "chosen_key" in request.content.decode())

    def build(*args: Any, **kwargs: Any) -> Any:
        return real(*args, transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(providers, "own_llm", build)
    return seen


def _claude_message(text: str, **headers: str) -> Any:
    import httpx2

    return httpx2.Response(
        200,
        headers=headers,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    )


def _three_drugs() -> str:
    return _answer(
        *(
            (name, "chemical", PICKS_TEXT.index(name), PICKS_TEXT.index(name) + len(name))
            for name in ("haloperidol", "ketamine", "aspirin")
        )
    )


def test_a_provider_that_sets_a_cookie_does_not_break_the_picks_after_the_first(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Anthropic answers with ``Set-Cookie`` like any site behind a CDN. The HTTP client
    replayed it on the next request of the same client, and the wire refused that."""
    cookie = {"Set-Cookie": "__cf_bm=abc123; Path=/; HttpOnly"}
    seen = _real_own_clients(
        monkeypatch,
        lambda request, pick: _claude_message(PICK if pick else _three_drugs(), **cookie),
    )
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "anthropic", "text": PICKS_TEXT, "selector": "claude-haiku-5-5"},
        headers=OWN_HEADERS,
    )
    assert result[0] == 200, result[2]
    picks = [e["pick"] for e in _json(result)["entities"] if e.get("pick")]
    assert len(picks) >= 2 and all(p["status"] != "failed" for p in picks), picks
    assert len(seen) == 1 + len(picks)
    assert all("cookie" not in request.headers for request in seen)
    assert "own_error" not in _last_log(capsys)


def test_a_request_refused_during_selection_is_named_in_the_log(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed pick is part of a 200. Without a word in the log nobody would learn that
    the site itself stopped the request."""
    from ner_site import providers

    seen = _real_own_clients(
        monkeypatch, lambda request, pick: _claude_message(PICK if pick else _three_drugs())
    )
    real = providers._Wire._allowed

    def no_picks(self: Any, request: Any) -> bool:
        return "candidates" not in request.content.decode().lower() and real(self, request)

    monkeypatch.setattr(providers._Wire, "_allowed", no_picks)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "anthropic", "text": PICKS_TEXT, "selector": "claude-haiku-5-5"},
        headers=OWN_HEADERS,
    )
    assert result[0] == 200, result[2]
    picks = [e["pick"] for e in _json(result)["entities"] if e.get("pick")]
    assert picks and {(p["status"], p["explanation"]) for p in picks} == {("failed", "error")}
    assert len(seen) == 1  # the extraction; no pick left the process
    record = _last_log(capsys)
    assert record["status"] == 200 and record["own_error"] == "WireRefused"
    assert OWN_KEY not in json.dumps(record)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
@pytest.mark.parametrize("stage", ["extract", "select"])
def test_a_bug_inside_a_real_adapter_is_not_reported_as_the_providers_failure(
    provider: str,
    stage: str,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exception's message holds the key. During extraction it is a 500 whose log
    line has the type and the frames; during selection it is a failed pick, named in
    the log. The message goes nowhere."""
    import httpx2
    from ner_site import providers
    from ner_site.claude import ClaudeClient

    lib: Any = httpx2 if provider == "anthropic" else httpx
    model = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna", "openrouter": GEMINI}[
        provider
    ]

    def answer(request: Any) -> Any:
        if provider == "anthropic":
            return _claude_message(_three_drugs())
        body = {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": _three_drugs()}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }
        return lib.Response(200, json=body)

    real = providers.own_llm

    def build(*args: Any, purpose: str, **kwargs: Any) -> Any:
        llm = real(*args, purpose=purpose, transport=lib.MockTransport(answer), **kwargs)
        if purpose == stage:
            adapter = llm.inner._inner
            name = "_arguments" if isinstance(adapter, ClaudeClient) else "_headers"

            def broken(*a: Any, **k: Any) -> Any:
                raise KeyError(OWN_KEY)

            monkeypatch.setattr(adapter, name, broken)
        return llm

    monkeypatch.setattr(providers, "own_llm", build)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": provider, "text": PICKS_TEXT, "selector": model},
        headers=OWN_HEADERS,
    )
    record = _last_log(capsys)
    if stage == "extract":
        assert _error(result) == (500, "internal")
        assert record["exception"] == "KeyError" and "own_error" not in record
        assert any(frame.startswith("providers.py:") for frame in record["where"])
    else:
        assert result[0] == 200, result[2]
        picks = [e["pick"] for e in _json(result)["entities"] if e.get("pick")]
        assert picks and {(p["status"], p["explanation"]) for p in picks} == {("failed", "error")}
        assert record["own_error"] == "KeyError"
    for text in (json.dumps(record), result[2].decode(), str(result[1])):
        assert OWN_KEY not in text and json.dumps(OWN_KEY)[1:-1] not in text


# --- third review round ------------------------------------------------------------------


@pytest.mark.parametrize("refused", ["haloperidol", "ketamine"])
def test_a_refusal_among_other_failed_picks_is_what_the_log_names(
    refused: str,
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One pick is stopped by the site's own guard, the others fail at the provider,
    in whatever order. The provider failing is ordinary; the guard refusing is what the
    operator has to see, so it is not hidden behind whichever failure came first."""
    import httpx2
    from ner_site import providers

    def respond(request: Any, pick: bool) -> Any:
        if pick:
            return httpx2.Response(
                500, json={"type": "error", "error": {"type": "api_error", "message": "no"}}
            )
        return _claude_message(_three_drugs())

    seen = _real_own_clients(monkeypatch, respond)
    real = providers._Wire._allowed

    def one_refused(self: Any, request: Any) -> bool:
        body = json.loads(request.content)
        is_pick = "chosen_key" in request.content.decode()
        mention = f"Mention: {refused}" in json.dumps(body) or f"[{refused}]" in json.dumps(body)
        return not (is_pick and mention) and real(self, request)

    monkeypatch.setattr(providers._Wire, "_allowed", one_refused)
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "anthropic", "text": PICKS_TEXT, "selector": "claude-haiku-5-5"},
        headers=OWN_HEADERS,
    )
    assert result[0] == 200, result[2]
    picks = [e["pick"] for e in _json(result)["entities"] if e.get("pick")]
    assert len(picks) == 2 and all(p["status"] == "failed" for p in picks)
    assert sorted(p["explanation"] for p in picks) == ["error", "http_5xx"]
    assert len(seen) == 2  # the extraction and the one pick that was sent
    record = _last_log(capsys)
    assert record["own_error"] == "WireRefused"
    assert record["own_errors"] == ["LLMRetryableError", "WireRefused"]


def test_a_malformed_200_during_extraction_is_the_providers_bad_answer_not_a_500(
    cache_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import httpx2

    _real_own_clients(
        monkeypatch,
        lambda request, pick: httpx2.Response(
            200, headers={"Content-Type": "application/json"}, content=b'{"content": ['
        ),
    )
    result = _ask(
        _own_only(cache_dir, tmp_path),
        {"provider": "anthropic", "text": PICKS_TEXT},
        headers=OWN_HEADERS,
    )
    assert _error(result) == (502, "provider_error")
    assert _json(result)["error"]["message"] == (
        "Anthropic could not complete the request (bad_answer)."
    )
    record = _last_log(capsys)
    assert record["own_error"] == "LLMRetryableError" and "exception" not in record


# --- an unreadable 200 on the site's own Anthropic key (after the third review round) ---


def _site_claude(script: dict[str, list[bytes | None]], seen: list[str]) -> Any:
    """The real adapter and SDK on the operator's key over a mock Anthropic. `script`
    gives each model its answers in order: bytes are sent as the 200 body as they are,
    None is a good answer; the last one repeats."""
    import anthropic
    import httpx2
    from ner_site.claude import ClaudeClient

    def handler(request: Any) -> Any:
        model = json.loads(request.content)["model"]
        seen.append(model)
        steps = script[model]
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if step is not None:
            return httpx2.Response(200, headers={"Content-Type": "application/json"}, content=step)
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": HALO}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        return httpx2.Response(200, json=body)

    def factory(settings: Settings, model: str, purpose: str) -> Any:
        sdk = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_KEY,
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
        )
        return ClaudeClient(model, api_key=ANTHROPIC_KEY, max_tokens=1000, client=sdk)

    return factory


@pytest.mark.parametrize(
    "body", [b'{"content": [', b"", b"\xff\xfe{", b'{"id":"msg_1","content":[{"type":"te']
)
def test_an_unreadable_200_from_opus_twice_ends_on_sonnet_with_a_notice(
    body: bytes,
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 200 whose body cannot be decoded is the provider failing, as a 5xx is: the
    chain tries the model once more and then moves on. It used to be a 500."""
    seen: list[str] = []
    result = _ask(
        _open_site(cache_dir, tmp_path),
        {"text": HALO_TEXT},
        claude=_site_claude({OPUS: [body], SONNET: [None]}, seen),
    )
    assert result[0] == 200, result[2]
    out = _json(result)
    assert seen == [OPUS, OPUS, SONNET]
    assert (out["model"], out["requested_model"]) == (SONNET, OPUS)
    assert out["fallbacks"] == [
        {"stage": "extract", "from": OPUS, "to": SONNET, "reason": "transient"}
    ]
    assert out["notices"] == ["Claude Opus 5.5 did not answer; this ran on Claude Sonnet 5.5."]
    assert out["entities"][0]["text"] == "haloperidol"
    assert out["usage"]["llm_calls"] == 3 and out["usage"]["prompt_tokens"] is None
    record = _last_log(capsys)
    assert record["status"] == 200 and "exception" not in record


def test_an_unreadable_200_from_opus_once_is_retried_and_stays_on_opus(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    seen: list[str] = []
    result = _ask(
        _open_site(cache_dir, tmp_path),
        {"text": HALO_TEXT},
        claude=_site_claude({OPUS: [b'{"content": [', None], SONNET: [None]}, seen),
    )
    assert result[0] == 200, result[2]
    out = _json(result)
    assert seen == [OPUS, OPUS]
    assert out["model"] == OPUS and out["fallbacks"] == [] and out["notices"] == []


def test_a_fault_in_the_site_adapter_is_still_a_500(
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only what the SDK raises while decoding the answer is translated. The same
    class of error from the adapter's own code is a bug and stays one."""
    from ner_site.claude import ClaudeClient

    def broken(self: Any, request: Any) -> Any:
        return json.loads("{not json")

    monkeypatch.setattr(ClaudeClient, "_arguments", broken)
    seen: list[str] = []
    result = _ask(
        _open_site(cache_dir, tmp_path),
        {"text": HALO_TEXT},
        claude=_site_claude({OPUS: [None], SONNET: [None]}, seen),
    )
    assert _error(result) == (500, "internal")
    assert seen == [] and _last_log(capsys)["exception"] == "JSONDecodeError"


# --- site analytics: one row per run, and GET /api/stats (design 2026-11-18) ----------

STATS_PASSCODE = "stats-SENTINEL-passcode-7781"
STATS_HEADERS = {"x-stats-passcode": STATS_PASSCODE}
# 18 November 2026, 14:20:00 UTC.
STATS_NOW = 1795011600.0
STATS_KEYS = [
    "ok",
    "days",
    "generated_at",
    "totals",
    "by_day",
    "by_model",
    "by_code",
    "by_fallback",
    "by_country",
]
REAL_USAGE_EXECUTOR = http_mod.make_usage_executor


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The ``site_runs`` table, in memory (tests/test_site_usage.py), behind the seam
    its real Neon executor is made by, and the clock at ``STATS_NOW``."""
    from tests.test_site_usage import FakeRuns

    fake = FakeRuns()
    opened: list[tuple[str, float]] = []

    def executor(database_url: str, timeout: float) -> Any:
        opened.append((database_url, timeout))
        return fake

    monkeypatch.setattr(http_mod, "make_usage_executor", executor)
    monkeypatch.setattr(http_mod, "_now", lambda: STATS_NOW)
    fake.opened = opened
    return fake


def _stats_site(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    """The open deployment with the record of runs and the stats page turned on."""
    return _open_site(cache_dir, tmp_path, **{"NER_SITE_STATS_PASSCODE": STATS_PASSCODE, **env})


def _stats(settings: Settings, query: str = "", headers: dict[str, str] | None = None) -> Handled:
    return http_mod.handle(
        "GET",
        "/api/stats" + query,
        dict(STATS_HEADERS if headers is None else headers),
        b"",
        settings=settings,
    )


def _recorded(fake: Any) -> list[dict[str, Any]]:
    """The rows, and that each took one statement of its own."""
    assert fake.tags().count("site_run") == len(fake.rows)
    return list(fake.rows)


def test_a_run_writes_one_row_from_what_the_handler_already_logs(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from ner_site import limits, usage

    s = _stats_site(cache_dir, tmp_path)
    result = _ask(
        s,
        {"text": HALO_TEXT, "selector": OPUS},
        claude=_Claude(HALO),
        headers={**VISITOR_HEADERS, "X-Vercel-IP-Country": "SI"},
    )
    assert result[0] == 200, result[2]
    out = NerResponse.model_validate_json(result[2])
    (row,) = _recorded(runs)
    assert runs.round_trips == 1 and runs.opened == [(DB_URL, http_mod.USAGE_TIMEOUT_S)]
    salt = limits.default_salt(DB_URL)
    visitor = limits.visitor_id({"x-real-ip": RAW_IP}, salt)
    assert row["visitor_day"] == usage.visitor_day(visitor, salt, STATS_NOW)
    assert row["at"].timestamp() == STATS_NOW
    assert {k: row[k] for k in ("status", "code", "credential", "provider", "tier", "country")} == {
        "status": 200,
        "code": None,
        "credential": "site",
        "provider": "site",
        "tier": "opus",
        "country": "SI",
    }
    assert (row["requested_model"], row["model"], row["requested_selector"]) == (OPUS, OPUS, OPUS)
    assert row["fallback_reasons"] == "" and row["custom_prompt"] is False
    assert row["text_chars"] == len(HALO_TEXT)
    assert (row["entities"], row["picks"]) == (1, 1)
    assert row["matches"] == len(out.entities[0].matches) > 0
    wire = out.usage
    assert row["llm_calls"] == wire.llm_calls == 1
    assert row["selector_calls"] == wire.selector_calls == 1
    assert (row["prompt_tokens"], row["completion_tokens"]) == (
        wire.prompt_tokens,
        wire.completion_tokens,
    )
    assert (row["selector_prompt_tokens"], row["selector_completion_tokens"]) == (
        wire.selector_prompt_tokens,
        wire.selector_completion_tokens,
    )
    assert row["selector_cost_usd"] == wire.selector_cost_usd
    assert row["embedding_requests"] == wire.embedding_requests
    assert {k: row[k + "_ms"] for k in ("extract", "map", "select", "total")} == {
        k: out.timing_ms.get(k) for k in ("extract", "map", "select", "total")
    }
    record = _last_log(capsys)
    # The log line is what it was: no trace of recording unless recording failed.
    assert "usage" not in record and "selector_prompt_tokens" not in record
    assert not any("visitor" in key for key in record)


def test_every_status_class_after_parsing_writes_exactly_one_row(
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    runs: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xwalk.retrieval.postgres import PostgresError

    s = _stats_site(cache_dir, tmp_path)
    seen: list[tuple[int, str | None]] = []

    def check(result: Handled, **expected: Any) -> dict[str, Any]:
        rows = _recorded(runs)
        seen.append((result[0], None if result[0] == 200 else _error(result)[1]))
        assert len(rows) == len(seen)
        assert [(row["status"], row["code"]) for row in rows] == seen
        for name, value in expected.items():
            assert rows[-1][name] == value, name
        return rows[-1]

    # 200 after a step-down: asked for Opus, answered by Sonnet.
    limited = _claude_error("rate_limited")
    stepped = _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO, opus_extract=[limited]))
    assert stepped[0] == 200
    check(
        stepped,
        requested_model=OPUS,
        model=SONNET,
        tier="opus",
        fallback_reasons="rate_limited",
        country=None,
    )
    # 400 from validation, after the body was parsed.
    refused = _ask(s, {"top_k": 99})
    assert refused[0] == 400
    check(refused, requested_model=None, model=None, tier=None, text_chars=len("nothing"))
    # 429 from admission.
    used = _stats_site(
        cache_dir, tmp_path, NER_SITE_OPUS_PER_HOUR="0", NER_SITE_SONNET_PER_HOUR="0"
    )
    limit = _ask(used, claude=_no_claude)
    assert _error(limit) == (429, "visitor_limit_reached")
    check(limit, requested_model=OPUS, model=None, tier=None, llm_calls=None)
    # 503 from admission: the counters cannot be asked.
    visits.fail = PostgresError("down", code="08006")
    down = _ask(s, claude=_no_claude)
    assert _error(down) == (503, "limits_unavailable")
    check(down, requested_model=OPUS, tier=None)
    visits.fail = None
    # A provider failure after admission.
    fatal = LLMFatalError("the provider said no")
    failed = _ask(s, claude=_Claude(opus_extract=[fatal], sonnet_extract=[fatal]))
    assert failed[0] >= 500
    check(failed, requested_model=OPUS, model=None, tier="opus", entities=None)
    # A 500.
    with monkeypatch.context() as patch:

        async def broken(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

        patch.setattr(http_mod, "run", broken)
        crashed = _ask(s)
    assert _error(crashed) == (500, "internal")
    check(crashed, requested_model=OPUS, tier="opus", total_ms=None)
    assert runs.round_trips == len(seen) == 6


def test_requests_that_never_became_a_run_write_nothing(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any
) -> None:
    s = _stats_site(cache_dir, tmp_path, NER_SITE_PASSCODE=PASSCODE)
    good = {**VISITOR_HEADERS, "x-passcode": PASSCODE}
    body = json.dumps({"text": "nothing"}).encode()

    def ner(headers: dict[str, str], raw: bytes = body, method: str = "POST") -> Handled:
        return http_mod.handle(method, "/api/ner", headers, raw, settings=s, llm_factory=_never)

    refusals = [
        (ner({k: v for k, v in good.items() if k != "x-requested-with"}), 400, "bad_header"),
        (ner({**good, "x-passcode": "nope"}), 401, "passcode_invalid"),
        (ner(VISITOR_HEADERS), 401, "passcode_required"),
        (ner(good, b"{not json"), 400, "bad_json"),
        (ner(good, b"[]"), 400, "bad_json"),
        (ner(good, b'{"prompt": "no text"}'), 400, "missing_text"),
        (ner(good, b" " * (s.limits.max_body_bytes + 1)), 413, "body_too_large"),
        (ner(good, method="GET"), 405, "method_not_allowed"),
        (http_mod.handle("POST", "/api/nope", good, body, settings=s), 404, "not_found"),
        (_stats(s, headers={}), 401, "stats_passcode_required"),
    ]
    for result, status, code in refusals:
        assert _error(result) == (status, code)
    assert http_mod.handle("GET", "/api/health", good, b"", settings=s)[0] == 200
    assert _stats(s)[0] == 200
    assert runs.rows == [] and "site_run" not in runs.tags()
    assert visits.rows == {}


def test_a_run_row_holds_no_text_prompt_address_visitor_passcode_or_key(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from ner_site import limits

    s = _stats_site(cache_dir, tmp_path, NER_SITE_PASSCODE=PASSCODE)
    headers = {
        **VISITOR_HEADERS,
        "x-passcode": PASSCODE,
        "user-agent": "AGENTSENTINEL/1.0",
        "x-vercel-ip-country": "SI",
        "x-vercel-ip-city": "CITYSENTINEL",
    }
    capsys.readouterr()
    results = [
        _ask(
            s,
            {"text": SECRET_TEXT, "prompt": SECRET_PROMPT, "selector": OPUS},
            claude=_Claude(_answer(("haloperidol", "chemical", 23, 34))),
            headers=headers,
        ),
        _ask(s, {"text": SECRET_TEXT, "prompt": SECRET_PROMPT, "top_k": 99}, headers=headers),
        _ask(
            s,
            {"text": SECRET_TEXT, "prompt": SECRET_PROMPT},
            claude=_Claude(opus_extract=[_claude_error("rate_limited")] * 3, sonnet_extract=[]),
            headers={**headers, "x-vercel-ip-country": SECRET_TEXT},
        ),
    ]
    assert results[0][0] == 200 and results[1][0] == 400
    assert len(_recorded(runs)) == 3 and runs.rows[2]["country"] is None
    results.append(_stats(s))
    assert results[-1][0] == 200 and _json(results[-1])["totals"]["runs"] == 3
    visitor = limits.visitor_id({"x-real-ip": RAW_IP}, limits.default_salt(DB_URL))
    stored = json.dumps([runs.sent, [str(row) for row in runs.rows]], default=str)
    err = capsys.readouterr().err
    for secret in (
        SECRET_TEXT,
        "Quokka",
        SECRET_PROMPT,
        "PROMPTSENT",
        RAW_IP,
        "10.0.0.1",
        visitor,
        PASSCODE,
        STATS_PASSCODE,
        ANTHROPIC_KEY,
        SENTINEL_KEY,
        DB_URL,
        "SENTINELpw",
        "AGENTSENTINEL",
        "CITYSENTINEL",
    ):
        assert secret not in stored, secret
        assert secret not in err, secret
        for _, response_headers, body in results:
            assert secret.encode() not in body and secret not in json.dumps(response_headers)


def test_an_own_key_run_is_recorded_without_the_key(
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    runs: Any,
    own_clients: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ner_site import limits, usage

    s = _stats_site(cache_dir, tmp_path, NER_SITE_OWN_PROVIDERS=ALL_OWN)
    capsys.readouterr()
    ran = _ask(
        s,
        {"provider": "openai", "text": HALO_TEXT, "selector": "gpt-6-luna"},
        # The key pasted where an address belongs must not be hashed into the row either.
        headers={**OWN_HEADERS, "x-real-ip": OWN_KEY, "x-forwarded-for": OWN_KEY},
    )
    assert ran[0] == 200, ran[2]
    # Refused before a credential was read: recorded, with the address left out.
    missing = _ask(s, {"provider": "openai"}, headers={**VISITOR_HEADERS, "x-real-ip": OWN_KEY})
    assert missing[0] in (400, 401)
    first, second = _recorded(runs)
    assert (first["credential"], first["provider"], first["tier"], first["status"]) == (
        "visitor",
        "openai",
        "own",
        200,
    )
    assert (first["requested_model"], first["model"]) == ("gpt-6-luna", "gpt-6-luna")
    assert (second["credential"], second["provider"], second["status"]) == (
        "visitor",
        "openai",
        missing[0],
    )
    salt = limits.default_salt(DB_URL)
    nobody = usage.visitor_day(limits.visitor_id({}, salt), salt, STATS_NOW)
    assert first["visitor_day"] == second["visitor_day"] == nobody
    stats = _stats(s)
    assert stats[0] == 200
    assert _json(stats)["by_model"] == [
        {
            "model": "gpt-6-luna",
            "credential": "visitor",
            "runs": 1,
            "prompt_tokens": first["prompt_tokens"] + first["selector_prompt_tokens"],
            "completion_tokens": first["completion_tokens"] + first["selector_completion_tokens"],
            "estimated_usd": 0.0,
        },
        {
            "model": None,
            "credential": "visitor",
            "runs": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
        },
    ]
    stored = json.dumps([runs.sent, [str(row) for row in runs.rows]], default=str)
    err = capsys.readouterr().err
    for form in (OWN_KEY, json.dumps(OWN_KEY)[1:-1], "visitor-sentinel"):
        assert form not in stored and form not in str(runs.sent), form
        assert form not in err, form
        for result in (ran, missing, stats):
            assert form.encode() not in result[2] and form not in json.dumps(result[1])


def test_a_failing_recorder_leaves_the_answer_untouched_and_is_named_in_the_log(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from xwalk.retrieval.postgres import PostgresError

    s = _stats_site(cache_dir, tmp_path)

    def ask() -> tuple[int, dict[str, str], dict[str, Any]]:
        status, headers, body = _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO))
        data = json.loads(body)
        data.pop("timing_ms")
        return status, headers, data

    healthy = ask()
    assert healthy[0] == 200 and "usage" not in _last_log(capsys)
    runs.fail = PostgresError(f"could not reach Neon at ep-x-pooler ({DB_URL})", code="08006")
    assert ask() == healthy
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[-1])
    assert record["usage"] == "PostgresError" and record["status"] == 200
    assert "ep-x-pooler" not in err and "SENTINELpw" not in err
    # A refusal keeps its status, code and headers too.
    used = _stats_site(
        cache_dir, tmp_path, NER_SITE_OPUS_PER_HOUR="0", NER_SITE_SONNET_PER_HOUR="0"
    )
    refused = _ask(used, claude=_no_claude)
    assert _error(refused) == (429, "visitor_limit_reached") and "Retry-After" in refused[1]
    assert _last_log(capsys)["usage"] == "PostgresError"
    assert len(runs.rows) == 1


def test_a_recorder_that_cannot_be_built_or_breaks_is_swallowed(
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = _stats_site(cache_dir, tmp_path)

    def unusable(database_url: str, timeout: float) -> Any:
        raise ValueError(f"cannot use {database_url}")

    monkeypatch.setattr(http_mod, "make_usage_executor", unusable)
    assert _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO))[0] == 200
    err = capsys.readouterr().err
    assert json.loads(err.strip().splitlines()[-1])["usage"] == "ValueError"
    assert "SENTINELpw" not in err

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(DB_URL)

    monkeypatch.setattr(http_mod.usage, "run_of", broken)
    monkeypatch.setattr(http_mod, "make_usage_executor", lambda url, timeout: object())
    http_mod.reset()
    assert _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO))[0] == 200
    err = capsys.readouterr().err
    assert json.loads(err.strip().splitlines()[-1])["usage"] == "RuntimeError"
    assert "SENTINELpw" not in err


def test_a_slow_recorder_delays_the_answer_by_no_more_than_its_timeout(
    cache_dir: Path,
    tmp_path: Path,
    visits: Any,
    runs: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import time

    release = threading.Event()
    entered = threading.Event()
    real = runs.transaction

    def stuck(statements: Any, **kwargs: Any) -> Any:
        entered.set()
        release.wait(30)
        return real(statements, **kwargs)

    monkeypatch.setattr(runs, "transaction", stuck)
    monkeypatch.setattr(http_mod, "USAGE_TIMEOUT_S", 0.2)
    s = _stats_site(cache_dir, tmp_path)
    try:
        started = time.monotonic()
        result = _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO))
        waited = time.monotonic() - started
        assert result[0] == 200 and entered.is_set()
        assert NerResponse.model_validate_json(result[2]).model == OPUS
        assert waited < 10
        assert _last_log(capsys)["usage"] == "TimeoutError"
    finally:
        release.set()
    assert http_mod.USAGE_TIMEOUT_S == 0.2 and REAL_USAGE_TIMEOUT_S == 2.0


REAL_USAGE_TIMEOUT_S = http_mod.USAGE_TIMEOUT_S


def test_without_the_stats_passcode_nothing_is_recorded_and_stats_do_not_exist(
    cache_dir: Path, tmp_path: Path, visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ``runs`` fixture: building a usage executor here fails the test (conftest)."""
    s = _open_site(cache_dir, tmp_path)
    assert s.stats_passcode is None
    assert _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO))[0] == 200
    assert "usage" not in _last_log(capsys)
    unknown = http_mod.handle("GET", "/api/nope", dict(STATS_HEADERS), b"", settings=s)
    unknown_log = _last_log(capsys)
    for method in ("GET", "HEAD", "POST", "OPTIONS"):
        for headers in ({}, STATS_HEADERS, {"x-stats-passcode": "wrong"}, POST_HEADERS):
            result = http_mod.handle(method, "/api/stats?days=7", dict(headers), b"", settings=s)
            assert result == unknown
            assert _error(result) == (404, "not_found")
            record = _last_log(capsys)
            assert record["path"] == "(unknown)"
            record["method"] = "GET"
            record["timing_ms"] = unknown_log["timing_ms"]
            assert record == unknown_log
    assert "stats" not in json.dumps(_health_of(s))


def test_the_usage_executor_is_closed_off_unless_a_test_gives_one(
    cache_dir: Path, tmp_path: Path, visits: Any
) -> None:
    s = _stats_site(cache_dir, tmp_path)
    with pytest.raises(pytest.fail.Exception, match="site_runs"):
        _ask(s)


def test_with_the_passcode_but_no_database_runs_are_skipped_and_stats_are_503(
    cache_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No ``runs`` fixture again: nothing may be built for a database that is not there."""
    s = _settings(
        cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_STATS_PASSCODE=STATS_PASSCODE
    )
    assert s.database_url is None
    assert _post(s, {"text": "x"}, llm_factory=_fake('{"entities": []}'))[0] == 200
    assert "usage" not in _last_log(capsys)
    result = _stats(s)
    assert _error(result) == (503, "stats_unavailable")
    record = _last_log(capsys)
    assert (record["path"], record["status"], record["code"]) == (
        "/api/stats",
        503,
        "stats_unavailable",
    )
    assert _error(_stats(s, headers={})) == (401, "stats_passcode_required")


def test_the_settings_cache_reads_the_database_url_once_the_stats_passcode_is_set() -> None:
    env = {"NER_SITE_STATS_PASSCODE": STATS_PASSCODE, "DATABASE_URL": DB_URL}
    found = http_mod.env_settings(env)
    assert isinstance(found, Settings) and found.database_url == DB_URL
    off = http_mod.env_settings({"DATABASE_URL": DB_URL})
    assert isinstance(off, Settings) and off.database_url is None and off is not found


def test_stats_need_their_own_passcode_and_nothing_else(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[bytes, bytes]] = []
    real = http_mod.hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(http_mod.hmac, "compare_digest", spy)
    s = _stats_site(cache_dir, tmp_path, NER_SITE_PASSCODE=PASSCODE)
    secret = STATS_PASSCODE.encode()
    for headers in (
        {},
        {"x-stats-passcode": ""},
        # Neither the visitor passcode nor a bearer token opens it.
        {**POST_HEADERS, "x-passcode": PASSCODE},
        {"Authorization": "Bearer " + STATS_PASSCODE},
        {"x-passcode": STATS_PASSCODE},
    ):
        result = _stats(s, headers=headers)
        assert _error(result) == (401, "stats_passcode_required")
        _assert_common_headers(result[1])
    assert seen == []
    for wrong in (PASSCODE, STATS_PASSCODE + "x", STATS_PASSCODE[:-1], "x"):
        result = _stats(s, headers={"x-stats-passcode": wrong})
        assert _error(result) == (401, "stats_passcode_invalid")
        assert seen[-1] == (wrong.encode(), secret)
    assert runs.round_trips == 0 and runs.opened == []
    # The stats passcode alone: no visitor passcode, no X-Requested-With.
    result = _stats(s, headers={"X-Stats-Passcode": STATS_PASSCODE})
    assert result[0] == 200 and seen[-1] == (secret, secret)
    _assert_common_headers(result[1])
    assert result[1]["Cache-Control"] == "no-store"
    assert list(_json(result)) == STATS_KEYS
    # It is not a way around the visitor passcode, and the limiter never hears of it.
    assert _error(_ask(s, headers={**VISITOR_HEADERS, **STATS_HEADERS})) == (
        401,
        "passcode_required",
    )
    assert visits.round_trips == 0 and visits.rows == {}


@pytest.mark.parametrize("method", ["POST", "HEAD", "PUT", "DELETE", "OPTIONS"])
def test_stats_allow_only_get(cache_dir: Path, tmp_path: Path, runs: Any, method: str) -> None:
    s = _stats_site(cache_dir, tmp_path)
    result = http_mod.handle(method, "/api/stats", dict(STATS_HEADERS), b"", settings=s)
    assert _error(result) == (405, "method_not_allowed")
    assert result[1]["Allow"] == "GET" and runs.round_trips == 0
    _assert_common_headers(result[1])


@pytest.mark.parametrize(
    ("query", "days"),
    [("", 30), ("?days=1", 1), ("?days=7", 7), ("?days=365", 365), ("?other=1", 30)],
)
def test_stats_days_default_to_thirty_and_run_from_one_to_a_year(
    cache_dir: Path,
    tmp_path: Path,
    runs: Any,
    capsys: pytest.CaptureFixture[str],
    query: str,
    days: int,
) -> None:
    result = _stats(_stats_site(cache_dir, tmp_path), query)
    assert result[0] == 200, result[2]
    data = _json(result)
    assert data["ok"] is True and data["days"] == days and len(data["by_day"]) == days
    assert data["by_day"][-1]["day"] == "2026-11-18"
    assert data["generated_at"] == "2026-11-18T14:20:00Z"
    assert runs.round_trips == 1 and runs.opened == [(DB_URL, http_mod.STATS_TIMEOUT_S)]
    record = _last_log(capsys)
    assert (record["path"], record["status"], record["days"]) == ("/api/stats", 200, days)


@pytest.mark.parametrize(
    "query",
    [
        "?days=0",
        "?days=366",
        "?days=-1",
        "?days=abc",
        "?days=",
        "?days=1.5",
        "?days=7&days=30",
        "?days=%C2%B2",
        "?days=0007",
        "?days=1e2",
        "?days=+7",
        "?days=%207",
    ],
)
def test_stats_days_out_of_bounds_are_refused_before_the_database(
    cache_dir: Path, tmp_path: Path, runs: Any, query: str
) -> None:
    s = _stats_site(cache_dir, tmp_path)
    assert _error(_stats(s, query)) == (400, "bad_days")
    # The passcode comes first: a stranger learns nothing about the query rules.
    assert _error(_stats(s, query, headers={})) == (401, "stats_passcode_required")
    assert runs.round_trips == 0


def test_stats_summarise_the_recorded_runs(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any
) -> None:
    s = _stats_site(cache_dir, tmp_path, NER_SITE_OPUS_PER_HOUR="1", NER_SITE_SONNET_PER_HOUR="1")
    si = {**VISITOR_HEADERS, "x-vercel-ip-country": "SI"}
    statuses = [
        _ask(s, {"text": HALO_TEXT, "selector": OPUS}, claude=_Claude(HALO), headers=si)[0],
        _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO), headers=si)[0],
        _ask(s, claude=_Claude(HALO), headers={**si, "x-real-ip": "198.51.100.9"})[0],
        _ask(s, claude=_no_claude, headers=si)[0],
    ]
    assert statuses == [200, 200, 200, 429]
    data = _json(_stats(s, "?days=2"))
    assert list(data) == STATS_KEYS
    assert [day["day"] for day in data["by_day"]] == ["2026-11-17", "2026-11-18"]
    today = data["by_day"][1]
    assert (today["runs"], today["ok"], today["visitors"]) == (4, 3, 2)
    assert (today["opus"], today["sonnet"], today["stepped_down"]) == (2, 1, 1)
    assert data["totals"]["runs"] == 4 and data["totals"]["errors"] == 1
    assert data["totals"]["visitors"] == 2 and data["totals"]["unpriced_runs"] == 0
    assert data["by_code"] == [{"code": "visitor_limit_reached", "runs": 1}]
    assert data["by_fallback"] == [{"reason": "opus_allowance_used", "runs": 1}]
    assert data["by_country"] == [{"country": "SI", "runs": 4}]
    assert {(row["model"], row["credential"], row["runs"]) for row in data["by_model"]} == {
        (OPUS, "site", 2),
        (SONNET, "site", 1),
        (None, "site", 1),
    }
    # One store for recording and one for reading, each kept for the warm instance.
    assert runs.opened == [(DB_URL, http_mod.USAGE_TIMEOUT_S), (DB_URL, http_mod.STATS_TIMEOUT_S)]
    assert _stats(s)[0] == 200 and len(runs.opened) == 2
    http_mod.reset()
    assert _stats(s)[0] == 200 and len(runs.opened) == 3


def test_stats_with_the_database_down_are_a_503_without_its_address(
    cache_dir: Path, tmp_path: Path, runs: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from xwalk.retrieval.postgres import PostgresError

    runs.fail = PostgresError(f"could not reach Neon at ep-x-pooler ({DB_URL})", code="08006")
    result = _stats(_stats_site(cache_dir, tmp_path))
    assert _error(result) == (503, "stats_unavailable")
    _assert_common_headers(result[1])
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[-1])
    assert record["stats"] == "site runs could not be queried (PostgresError 08006)"
    for secret in ("ep-x-pooler", "SENTINELpw", STATS_PASSCODE):
        assert secret not in err and secret.encode() not in result[2]


def test_a_stats_answer_that_echoed_a_secret_would_still_be_scrubbed(
    cache_dir: Path, tmp_path: Path, runs: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing puts one there; the last line of defence covers this route like any other."""
    s = _stats_site(cache_dir, tmp_path)
    assert _stats(s)[0] == 200
    store = next(iter(http_mod._USAGE.values()))
    leaky = {"ok": True, "note": f"{STATS_PASSCODE} {ANTHROPIC_KEY} {DB_URL}"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store, "stats", lambda days: leaky)
        result = _stats(s)
    assert _json(result) == {"ok": True, "note": "[redacted] [redacted] [redacted]"}
    assert STATS_PASSCODE not in capsys.readouterr().err


def test_the_usage_executor_is_the_limiters_kind_with_its_own_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a run is not idempotent either, and it runs while the visitor waits."""
    built: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(http_mod, "NeonHTTPExecutor", lambda url, **kw: built.append((url, kw)))
    monkeypatch.setattr(http_mod, "make_limits_executor", REAL_LIMITS_EXECUTOR)
    REAL_USAGE_EXECUTOR(DB_URL, http_mod.USAGE_TIMEOUT_S)
    REAL_USAGE_EXECUTOR(DB_URL, http_mod.STATS_TIMEOUT_S)
    assert built == [
        (DB_URL, {"timeout": 2.0, "max_retries": 0}),
        (DB_URL, {"timeout": http_mod.STATS_TIMEOUT_S, "max_retries": 0}),
    ]
    assert http_mod.USAGE_TIMEOUT_S == 2.0 < http_mod.STATS_TIMEOUT_S <= 15
    assert "make_usage_executor" in http_mod.__all__


def test_the_stats_shim_serves_api_stats(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api/stats.py`` is the function Vercel runs for ``/api/stats``; like the other
    two it hands every request to ``handle``."""
    for name in ("OPENROUTER_API_KEY", "NER_SITE_PASSCODE", "NER_SITE_MODELS", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("NER_SITE_STATS_PASSCODE", raising=False)
    monkeypatch.setenv("NER_SITE_OFFLINE", "1")
    monkeypatch.setenv("NER_SITE_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("NER_SITE_DENSE_DIR", str(tmp_path / "no-dense"))
    handler = _shim("stats.py").handler
    assert handler.__name__ == "handler"
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        status, headers, body = _request(port, "GET", "/api/stats?days=7", headers=STATS_HEADERS)
        assert status == 404 and json.loads(body)["error"]["code"] == "not_found"
        monkeypatch.setenv("NER_SITE_STATS_PASSCODE", STATS_PASSCODE)
        status, headers, body = _request(port, "GET", "/api/stats?days=7")
        assert status == 401 and json.loads(body)["error"]["code"] == "stats_passcode_required"
        assert headers["cache-control"] == "no-store"
        status, _, body = _request(port, "GET", "/api/stats", headers={"X-Stats-Passcode": "no"})
        assert status == 401 and json.loads(body)["error"]["code"] == "stats_passcode_invalid"
        # Offline with no database: the passcode is right and there is nothing to read.
        status, _, body = _request(port, "GET", "/api/stats?days=7", headers=STATS_HEADERS)
        assert status == 503 and json.loads(body)["error"]["code"] == "stats_unavailable"
        assert STATS_PASSCODE.encode() not in body
        status, _, body = _request(port, "GET", "/api/health")
        assert status == 200 and b"stats" not in body
    finally:
        server.shutdown()
        server.server_close()


def _json_shape(value: Any) -> Any:
    """Names and JSON types, without the values: a list is the set of its items' shapes."""
    if isinstance(value, dict):
        return {key: _json_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return sorted({json.dumps(_json_shape(item), sort_keys=True) for item in value})
    if isinstance(value, bool):
        return "boolean"
    return "number" if isinstance(value, (int, float)) else type(value).__name__


def test_the_stats_fixtures_have_the_shape_and_the_errors_the_server_sends(
    cache_dir: Path, tmp_path: Path, visits: Any, runs: Any
) -> None:
    """The stats page is developed against ``tools/fixtures/stats*.json``: they must be
    answers this server could give, field for field, and the refusals word for word."""
    directory = PROJECT / "tools/fixtures"

    def fixture(name: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((directory / name).read_text(encoding="utf-8"))
        return data

    s = _stats_site(cache_dir, tmp_path)
    assert _json(_stats(s, headers={})) == fixture("stats-error-required.json")
    assert _json(_stats(s, headers={"x-stats-passcode": "x"})) == fixture(
        "stats-error-invalid.json"
    )
    off = _open_site(cache_dir, tmp_path)
    assert _json(_stats(off)) == fixture("stats-nostats.json")
    without = _stats_site(cache_dir, tmp_path, DATABASE_URL="", NER_SITE_PASSCODE=PASSCODE)
    assert _json(_stats(without)) == fixture("stats-unavailable.json")

    empty = _json(_stats(s, "?days=2"))
    si = {**VISITOR_HEADERS, "x-vercel-ip-country": "SI"}
    limited = _claude_error("rate_limited")
    assert _ask(s, {"text": HALO_TEXT}, claude=_Claude(HALO, opus_extract=[limited]), headers=si)
    assert _ask(s, {"top_k": 99})[0] == 400  # no model, and no country
    populated = _json(_stats(s, "?days=2"))
    assert all(populated[name] for name in STATS_KEYS[4:])
    for name, real in (
        ("stats.json", populated),
        ("stats-7.json", populated),
        ("stats-90.json", populated),
        ("stats-empty.json", empty),
    ):
        canned = fixture(name)
        assert list(canned) == STATS_KEYS, name
        assert _json_shape(canned) == _json_shape(real), name
        assert len(canned["by_day"]) == canned["days"], name
        days = [day["day"] for day in canned["by_day"]]
        assert days == sorted(set(days)), name
        for day in canned["by_day"]:
            assert sum(day[tier] for tier in ("opus", "sonnet", "other", "own")) <= day["runs"]
            assert day["ok"] <= day["runs"]
        totals = canned["totals"]
        assert totals["runs"] == sum(day["runs"] for day in canned["by_day"]), name
        assert totals["errors"] == totals["runs"] - totals["ok"], name
        assert totals["visitors"] == sum(day["visitors"] for day in canned["by_day"]), name


# --- analytics fix wave: recording must not harm the live site ------------------------


def _wait_for(condition: Callable[[], bool], seconds: float = 5.0) -> bool:
    import time

    end = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > end:
            return False
        time.sleep(0.01)
    return True


def _usage_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "ner-site-usage" and t.is_alive())


@pytest.fixture
def blocked(runs: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The ``runs`` table behind a database that takes every statement and does not
    answer until ``release`` is set; ``peak`` is the most statements it held at once."""
    release = threading.Event()
    lock = threading.Lock()
    state = SimpleNamespace(release=release, held=0, peak=0, entered=0)

    def holding(real: Any) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            with lock:
                state.held += 1
                state.entered += 1
                state.peak = max(state.peak, state.held)
            try:
                release.wait(30)
                with lock:
                    return real(*args, **kwargs)
            finally:
                with lock:
                    state.held -= 1

        return call

    monkeypatch.setattr(runs, "transaction", holding(runs.transaction))
    monkeypatch.setattr(runs, "execute", holding(runs.execute))
    monkeypatch.setattr(http_mod, "USAGE_TIMEOUT_S", 0.15)
    try:
        yield state
    finally:
        release.set()
        assert _wait_for(lambda: http_mod._USAGE_FLIGHTS.count == 0)


def _plain_stats_site(cache_dir: Path, tmp_path: Path, **env: str) -> Settings:
    """A deployment without a limiter (OpenRouter only) that records its runs."""
    return _settings(
        cache_dir,
        tmp_path,
        **{
            "OPENROUTER_API_KEY": SENTINEL_KEY,
            "NER_SITE_STATS_PASSCODE": STATS_PASSCODE,
            "DATABASE_URL": DB_URL,
            **env,
        },
    )


def _plain_run(settings: Settings) -> tuple[int, dict[str, str], dict[str, Any]]:
    status, headers, body = _post(settings, {"text": "x"}, llm_factory=_fake('{"entities": []}'))
    data = json.loads(body)
    data.pop("timing_ms", None)
    return status, headers, data


def _usage_notes(err: str) -> list[str | None]:
    return [json.loads(line).get("usage") for line in err.strip().splitlines()]


def test_a_hanging_database_holds_no_more_than_the_cap_and_later_runs_skip_at_once(
    cache_dir: Path, tmp_path: Path, runs: Any, blocked: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sequential requests against a database that never answers. Each abandoned write
    keeps a thread and a connection: before the cap there was one of each per request."""
    import time

    cap = http_mod.USAGE_MAX_IN_FLIGHT
    assert cap == 4
    off = _settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    healthy = _plain_run(off)
    assert healthy[0] == 200
    s = _plain_stats_site(cache_dir, tmp_path)
    capsys.readouterr()
    for _ in range(cap):
        assert _plain_run(s) == healthy
    assert _usage_notes(capsys.readouterr().err) == ["TimeoutError"] * cap
    assert http_mod._USAGE_FLIGHTS.count == cap == _usage_threads()
    started = time.monotonic()
    for _ in range(3):
        assert _plain_run(s) == healthy
    waited = time.monotonic() - started
    assert _usage_notes(capsys.readouterr().err) == ["saturated"] * 3
    # Skipped without waiting, and without another thread, statement or executor.
    assert waited < 3 * http_mod.USAGE_TIMEOUT_S
    assert blocked.entered == cap == blocked.peak == _usage_threads()
    assert http_mod._USAGE_FLIGHTS.count == cap and len(runs.opened) == 1
    # The database answers again: the abandoned writes end, and recording resumes.
    blocked.release.set()
    assert _wait_for(lambda: http_mod._USAGE_FLIGHTS.count == 0 and _usage_threads() == 0)
    assert _plain_run(s) == healthy
    assert _usage_notes(capsys.readouterr().err) == [None]
    assert len(runs.rows) == cap + 1


def test_concurrent_requests_against_a_hanging_database_are_capped_too(
    cache_dir: Path, tmp_path: Path, runs: Any, blocked: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    cap = http_mod.USAGE_MAX_IN_FLIGHT
    s = _plain_stats_site(cache_dir, tmp_path)
    healthy = _plain_run(_settings(cache_dir, tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY))
    capsys.readouterr()
    answers: list[Any] = []
    threads = [
        threading.Thread(target=lambda: answers.append(_plain_run(s))) for _ in range(cap * 3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert len(answers) == cap * 3 and all(answer == healthy for answer in answers)
    notes = _usage_notes(capsys.readouterr().err)
    assert sorted(notes) == ["TimeoutError"] * cap + ["saturated"] * (cap * 2)
    assert blocked.peak == cap == blocked.entered == http_mod._USAGE_FLIGHTS.count
    blocked.release.set()
    assert _wait_for(lambda: http_mod._USAGE_FLIGHTS.count == 0)
    assert len(runs.rows) == cap


def test_setting_up_the_store_is_under_the_same_deadline_as_the_write(
    cache_dir: Path,
    tmp_path: Path,
    runs: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The store is built behind a lock every request of the instance shares. A request
    that finds it held (another one is building the executor) waits no longer for it
    than for the write itself."""
    import time

    monkeypatch.setattr(http_mod, "USAGE_TIMEOUT_S", 0.15)
    s = _plain_stats_site(cache_dir, tmp_path)
    capsys.readouterr()
    with http_mod._USAGE_LOCK:
        started = time.monotonic()
        result = _plain_run(s)
        waited = time.monotonic() - started
    assert result[0] == 200 and waited < 5
    assert _usage_notes(capsys.readouterr().err) == ["TimeoutError"]
    # Once the lock is free the abandoned write goes through and the slot is given back.
    assert _wait_for(lambda: http_mod._USAGE_FLIGHTS.count == 0)
    assert len(runs.rows) == 1


def test_the_executors_own_timeout_ends_an_abandoned_write(
    cache_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real executor against a server that accepts the connection and never
    answers: nothing releases the write but its own timeout, and the slot comes back."""
    import socket

    from xwalk.retrieval.postgres import NeonHTTPExecutor

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    port = server.getsockname()[1]
    held: list[socket.socket] = []

    def accept() -> None:
        try:
            while True:
                held.append(server.accept()[0])
        except OSError:
            return

    threading.Thread(target=accept, daemon=True).start()
    built: list[float] = []

    def executor(database_url: str, timeout: float) -> Any:
        built.append(timeout)
        return NeonHTTPExecutor(
            database_url,
            timeout=timeout,
            max_retries=0,
            endpoint=f"http://127.0.0.1:{port}/sql",
            trust_env=False,
        )

    monkeypatch.setattr(http_mod, "make_usage_executor", executor)
    monkeypatch.setattr(http_mod, "USAGE_TIMEOUT_S", 0.3)
    try:
        s = _plain_stats_site(cache_dir, tmp_path)
        for _ in range(2):
            assert _plain_run(s)[0] == 200
        assert built == [0.3]  # the write's timeout is the wait's
        assert _wait_for(lambda: http_mod._USAGE_FLIGHTS.count == 0 and _usage_threads() == 0)
        assert len(held) == 2
    finally:
        server.close()
        for connection in held:
            connection.close()
