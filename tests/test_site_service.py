"""ner_site.service and ner_site.fake (docs/ner-site/PLAN.md sections 3.2, 3.3, 7).

Offline: FakeLLM, an offline stand-in, and httpx.MockTransport; BM25 is built in
tmp_path and dense is always given an explicit directory (never the committed one).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from ner_site import fake, service
from ner_site import indexes as indexes_mod
from ner_site.extract import DEFAULT_PROMPT, ExtractionError, extract
from ner_site.schema import Limits, NerRequest, NerResponse, ServiceError
from ner_site.service import Settings, SettingsError, make_llm, run, validate_request
from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
)
from xwalk.llm.budget import BudgetedLLM, CallBudget
from xwalk.llm.fake import FakeLLM
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.records import Usage

SENTINEL_KEY = "sk-or-v1-SENTINEL-c0ffee-4242"
PROJECT = Path(__file__).resolve().parents[1] / "ner-site"
# public/app.js EXAMPLE_TEXT, verbatim.
EXAMPLE_TEXT = (
    "Clinic note 🩺: the patient has Alkaptonuria and was screened for "
    "Hydroxykynureninuria. She took haloperidol, later ketamine, and aspirin for a fever. "
    "Genotyping covered NAT2 and ADA. Her diet: hummus, garlic and cumin."
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    # The default selector is the first that can run (Jev with only an OpenRouter key);
    # these tests are about the request without a selection step unless they say so.
    env = {
        "NER_SITE_CACHE_DIR": str(tmp_path / "cache"),
        "NER_SITE_DENSE_DIR": str(tmp_path / "no-dense"),
        "NER_SITE_SELECTOR": "retrieval",
        **overrides,
    }
    return Settings.from_env(env)


@pytest.fixture(autouse=True)
def _fresh_memo() -> Any:
    indexes_mod.reset()
    yield
    indexes_mod.reset()


@pytest.fixture(scope="module")
def bm25_only(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return indexes_mod.open_indexes(
        cache_dir=tmp_path_factory.mktemp("bm25"), dense_dir=None, api_key=None
    )


def _answer(*items: tuple[str, str, int, int]) -> str:
    return json.dumps(
        {"entities": [{"text": t, "type": ty, "start": s, "end": e} for t, ty, s, e in items]}
    )


# --- Settings.from_env ------------------------------------------------------------


def test_from_env_defaults() -> None:
    s = Settings.from_env({})
    assert s.api_key is None and s.passcode is None and s.offline is False
    assert s.base_url == "https://openrouter.ai/api/v1"
    assert s.models == (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "google/gemini-3.1-flash-lite",
        "openai/gpt-6-luna",
    )
    # No Anthropic key: the Claude ids are configured but not offered.
    assert s.offered_models == s.models[2:]
    assert s.default_model == "google/gemini-3.1-flash-lite"
    assert s.limits == Limits()
    assert s.dense_dir == PROJECT / "indexes"
    assert s.cache_dir == Path("/tmp/xwalk-ner-site")
    assert s.mode == "live"


def test_from_env_reads_every_variable(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "OPENROUTER_API_KEY": f"  {SENTINEL_KEY} ",
            "NER_SITE_BASE_URL": "https://gateway.example/v1/",
            "NER_SITE_MODELS": " a/one , b/two,,a/one ",
            "NER_SITE_PASSCODE": "open sesame",
            "NER_SITE_MAX_CHARS": "1200",
            "NER_SITE_MAX_ENTITIES": "7",
            "NER_SITE_MAX_PROMPT_CHARS": "300",
            "NER_SITE_DENSE_DIR": str(tmp_path / "d"),
            "NER_SITE_CACHE_DIR": str(tmp_path / "c"),
        }
    )
    assert s.api_key == SENTINEL_KEY
    assert s.base_url == "https://gateway.example/v1"
    assert s.models == ("a/one", "b/two") and s.default_model == "a/one"
    assert s.passcode == "open sesame"
    assert (s.limits.max_chars, s.limits.max_entities, s.limits.max_prompt_chars) == (1200, 7, 300)
    assert s.dense_dir == tmp_path / "d" and s.cache_dir == tmp_path / "c"


def test_empty_values_mean_unset() -> None:
    s = Settings.from_env(
        {"OPENROUTER_API_KEY": " ", "NER_SITE_PASSCODE": "", "NER_SITE_MAX_CHARS": ""}
    )
    assert s.api_key is None and s.passcode is None and s.limits.max_chars == 5000


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_offline_mode(value: str) -> None:
    s = Settings.from_env({"NER_SITE_OFFLINE": value})
    assert s.offline and s.mode == "offline"
    assert s.default_model == "offline" and s.offered_models == ("offline",)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NER_SITE_MAX_CHARS", "lots"),
        ("NER_SITE_MAX_CHARS", "0"),
        ("NER_SITE_MAX_CHARS", "-5"),
        ("NER_SITE_MAX_CHARS", "1000000"),
        ("NER_SITE_MAX_CHARS", "5.5"),
        ("NER_SITE_MAX_ENTITIES", "0"),
        ("NER_SITE_MAX_ENTITIES", "101"),
        ("NER_SITE_MAX_PROMPT_CHARS", "99999"),
        ("NER_SITE_OFFLINE", "maybe"),
        ("NER_SITE_MODELS", " , "),
        ("NER_SITE_BASE_URL", "ftp://example/v1"),
        ("NER_SITE_BASE_URL", "not a url"),
    ],
)
def test_bad_values_are_a_settings_error_naming_the_variable(name: str, value: str) -> None:
    with pytest.raises(SettingsError) as exc:
        Settings.from_env({name: value})
    assert name in str(exc.value)


def test_secrets_stay_out_of_repr() -> None:
    s = Settings.from_env({"OPENROUTER_API_KEY": SENTINEL_KEY, "NER_SITE_PASSCODE": "pass-XYZ"})
    assert SENTINEL_KEY not in repr(s) and "pass-XYZ" not in repr(s)


def test_offline_index_view_never_carries_the_key(tmp_path: Path) -> None:
    live = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    offline = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_OFFLINE="1")
    assert live.index_settings().api_key == SENTINEL_KEY
    assert offline.index_settings().api_key is None
    assert offline.index_settings().cache_dir == offline.cache_dir


# --- make_llm -----------------------------------------------------------------------


def test_make_llm_wraps_a_budgeted_openai_client(tmp_path: Path) -> None:
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    llm = make_llm(s, "google/gemini-2.5-flash-lite")
    assert isinstance(llm, BudgetedLLM)
    assert llm.budget.limit == Limits().llm_call_cap == 3
    inner = llm.inner
    assert isinstance(inner, OpenAICompatClient)
    assert inner.model == "google/gemini-2.5-flash-lite"
    assert inner.capabilities.json_schema and inner.capabilities.strict_schema  # "openai"
    asyncio.run(llm.aclose())


def test_make_llm_sends_the_key_headers_and_generation_settings(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def chat(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-4o-mini",
                "choices": [{"message": {"content": '{"entities": []}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    llm = make_llm(s, "openai/gpt-4o-mini", transport=httpx.MockTransport(chat))

    async def go() -> tuple[Any, ...]:
        try:
            return await extract(llm, "nothing here", DEFAULT_PROMPT, max_entities=25)
        finally:
            await llm.aclose()

    entities, _, usage = asyncio.run(go())
    assert entities == [] and usage.prompt_tokens == 50
    (request,) = seen
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {SENTINEL_KEY}"
    assert request.headers["x-title"] == "xwalk NER demo"
    assert request.headers["http-referer"].startswith("https://")
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["temperature"] == 0 and body["max_tokens"] == 2048
    assert body["response_format"]["type"] == "json_schema"


def test_make_llm_retries_at_most_twice(tmp_path: Path) -> None:
    calls: list[int] = []

    def down(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    llm = make_llm(s, s.default_model, transport=httpx.MockTransport(down), backoff_base=0.0)
    with pytest.raises(Exception, match="exhausted 2 retries"):
        asyncio.run(llm.complete(LLMRequest(system="s", user="u")))
    assert len(calls) == 3


# --- validate_request ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_", "code"),
    [
        (NerRequest(text=""), "missing_text"),
        (NerRequest(text="   \n"), "missing_text"),
        (NerRequest(text="x" * 5001), "text_too_long"),
        (NerRequest(text="ok", prompt="p" * 4001), "prompt_too_long"),
        (NerRequest(text="ok", top_k=0), "bad_top_k"),
        (NerRequest(text="ok", top_k=11), "bad_top_k"),
        (NerRequest(text="ok", model="evil/model"), "unknown_model"),
        (NerRequest(text="ok", model="offline"), "unknown_model"),
    ],
)
def test_validate_request_codes(tmp_path: Path, request_: NerRequest, code: str) -> None:
    with pytest.raises(ServiceError) as exc:
        validate_request(request_, _settings(tmp_path))
    assert exc.value.code == code and exc.value.status == 400


def test_validate_request_resolves_defaults(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    got = validate_request(NerRequest(text="x" * 5000, prompt="  ", top_k=10), s)
    assert got.prompt == DEFAULT_PROMPT and got.model == s.default_model and got.top_k == 10
    second = s.offered_models[1]
    custom = validate_request(NerRequest(text="ok", prompt="only genes", model=second), s)
    assert custom.prompt == "only genes" and custom.model == second


def test_text_length_counts_code_points(tmp_path: Path) -> None:
    s = _settings(tmp_path, NER_SITE_MAX_CHARS="3")
    validate_request(NerRequest(text="🩺🩺🩺"), s)  # 3 code points, 6 UTF-16 units
    with pytest.raises(ServiceError, match="text is 4 characters; the limit is 3"):
        validate_request(NerRequest(text="🩺🩺🩺a"), s)


def test_offline_accepts_only_the_offline_model(tmp_path: Path) -> None:
    s = _settings(tmp_path, NER_SITE_OFFLINE="1")
    assert validate_request(NerRequest(text="ok", model="offline"), s).model == "offline"
    with pytest.raises(ServiceError) as exc:
        validate_request(NerRequest(text="ok", model="google/gemini-3.1-flash-lite"), s)
    assert exc.value.code == "unknown_model"


# --- run --------------------------------------------------------------------------


def test_run_end_to_end_with_fake_llm(tmp_path: Path, bm25_only: Any) -> None:
    text = "She took haloperidol for a fever."
    llm = FakeLLM(
        [
            _answer(
                ("haloperidol", "chemical", 0, 3), ("fever", "other", 27, 32), ("x", "gene", 0, 1)
            )
        ]
    )
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    out = asyncio.run(run(NerRequest(text=text, top_k=2), settings=s, llm=llm, indexes=bm25_only))
    assert isinstance(out, NerResponse)
    assert out.model == "google/gemini-3.1-flash-lite" and out.mode == "live"
    assert out.requested_model == out.model and out.requested_selector == "retrieval"
    assert out.fallbacks == [] and out.notices == []
    first, second = out.entities
    assert (first.text, first.start, first.end) == ("haloperidol", 9, 20)  # relocated
    assert text[first.start : first.end] == first.text
    assert first.ontology == "chebi-sample" and first.matches[0].id == "CHEBI:5613"
    assert len(first.matches) <= 2 and first.matches[0].retrievers == {"bm25": 1}
    assert second.ontology is None and second.notes == ["no ontology for type 'other'"]
    assert "entity 'x' dropped: not found in text" in out.warnings
    assert any("dense retrieval unavailable" in w for w in out.warnings)
    assert out.retrieval["dense"].status == "unavailable"
    assert out.usage.model_dump() == {
        "llm_calls": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "embedding_requests": 0,
        "selector_calls": 0,
        "selector_prompt_tokens": None,
        "selector_completion_tokens": None,
        "selector_cost_usd": None,
    }
    assert out.selector == "retrieval" and all(e.pick is None for e in out.entities)
    assert set(out.timing_ms) == {"extract", "map", "total"}
    assert llm.requests[0].user.endswith(f"TEXT:\n{text}")
    assert DEFAULT_PROMPT in llm.requests[0].user


class _UnreportedLLM:
    """Answers once and reports no token counts (as a provider omitting `usage`)."""

    model = "unreported"
    fingerprint = "unreported"
    capabilities = LLMCapabilities()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text='{"entities": []}', usage=Usage.unreported(), model=self.model)


def test_unknown_tokens_are_null_not_zero(tmp_path: Path, bm25_only: Any) -> None:
    llm: LLMClient = _UnreportedLLM()
    out = asyncio.run(
        run(NerRequest(text="nothing"), settings=_settings(tmp_path), llm=llm, indexes=bm25_only)
    )
    assert out.usage.llm_calls == 1
    assert out.usage.prompt_tokens is None and out.usage.completion_tokens is None


def test_run_passes_extraction_errors_through(tmp_path: Path, bm25_only: Any) -> None:
    llm = FakeLLM(["I could not find anything, sorry."])
    with pytest.raises(ExtractionError) as exc:
        asyncio.run(
            run(NerRequest(text="abc"), settings=_settings(tmp_path), llm=llm, indexes=bm25_only)
        )
    assert exc.value.code == "extraction_unparseable" and exc.value.status == 422


def test_run_maps_a_provider_failure_to_502_without_secrets(tmp_path: Path, bm25_only: Any) -> None:
    llm = FakeLLM(
        [
            LLMFatalError(
                f"HTTP 401: invalid key {SENTINEL_KEY} at https://user:hunter2@gw.example/v1?key=abc"
            )
        ]
    )
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    with pytest.raises(ServiceError) as exc:
        asyncio.run(run(NerRequest(text="abc"), settings=s, llm=llm, indexes=bm25_only))
    assert exc.value.status == 502 and exc.value.code == "provider_error"
    message = exc.value.message
    assert SENTINEL_KEY not in message and "hunter2" not in message and "abc" not in message
    assert "HTTP 401" in message and "gw.example" in message


def test_run_maps_an_exhausted_budget_to_429(tmp_path: Path, bm25_only: Any) -> None:
    llm = BudgetedLLM(FakeLLM(['{"entities": []}']), CallBudget(limit=0))
    with pytest.raises(ServiceError) as exc:
        asyncio.run(
            run(NerRequest(text="abc"), settings=_settings(tmp_path), llm=llm, indexes=bm25_only)
        )
    assert exc.value.status == 429 and exc.value.code == "call_budget_exhausted"
    assert "--max-calls" not in exc.value.message  # the library's CLI advice is not ours


def test_run_bounds_the_extraction_time(
    tmp_path: Path, bm25_only: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Slow:
        model = "slow"
        fingerprint = "slow"
        capabilities = LLMCapabilities()

        async def complete(self, request: LLMRequest) -> LLMResponse:
            await asyncio.sleep(5)
            raise AssertionError("not reached")

    monkeypatch.setattr(service, "EXTRACT_TIMEOUT_S", 0.05)
    with pytest.raises(ServiceError) as exc:
        asyncio.run(
            run(NerRequest(text="abc"), settings=_settings(tmp_path), llm=Slow(), indexes=bm25_only)
        )
    assert exc.value.status == 502 and "did not answer" in exc.value.message


# --- fake.offline_llm --------------------------------------------------------------


def test_offline_llm_is_an_llm_client() -> None:
    llm = fake.offline_llm()
    assert isinstance(llm, LLMClient)
    assert llm.model == "offline"


def test_offline_llm_answers_like_a_model_and_types_by_ontology() -> None:
    llm = fake.offline_llm()
    entities, warnings, usage = asyncio.run(
        extract(llm, EXAMPLE_TEXT, DEFAULT_PROMPT, max_entities=25)
    )
    found = {(e.text, e.type) for e in entities}
    assert {
        ("Alkaptonuria", "disease"),
        ("Hydroxykynureninuria", "disease"),
        ("haloperidol", "chemical"),
        ("ketamine", "chemical"),
        ("aspirin", "chemical"),
        ("fever", "other"),
        ("NAT2", "gene"),
        ("ADA", "gene"),
        ("hummus", "food"),
        ("garlic", "food"),
        ("cumin", "food"),
    } <= found
    for e in entities:
        assert EXAMPLE_TEXT[e.start : e.end] == e.text
    assert not any(e.text.lower() in {"the", "and", "her", "she", "was"} for e in entities)
    assert warnings == []
    assert usage.calls == 0  # a stand-in, not a provider call


def test_offline_llm_is_deterministic_and_returns_model_shaped_json() -> None:
    llm = fake.offline_llm()
    request = LLMRequest(system="s", user=f"INSTRUCTIONS:\nanything\n\nTEXT:\n{EXAMPLE_TEXT}")
    a = asyncio.run(llm.complete(request)).text
    b = asyncio.run(fake.offline_llm().complete(request)).text
    assert a == b
    payload = json.loads(a)
    assert set(payload) == {"entities"}
    for item in payload["entities"]:
        assert set(item) == {"text", "type", "start", "end"}
        assert EXAMPLE_TEXT[item["start"] : item["end"]] == item["text"]


def test_offline_llm_finds_nothing_in_plain_prose() -> None:
    llm = fake.offline_llm()
    request = LLMRequest(system="s", user="INSTRUCTIONS:\nx\n\nTEXT:\nWe met on a rainy day.")
    assert json.loads(asyncio.run(llm.complete(request)).text) == {"entities": []}


def test_one_reason_shared_by_every_ontology_is_reported_once(tmp_path: Path) -> None:
    from ner_site.service import retrieval_status

    index = tmp_path / "dense" / "chebi-sample" / "dense"
    index.mkdir(parents=True)
    (index / "xwalk_meta.json").write_text("{}")
    all_missing_key = indexes_mod.open_indexes(
        cache_dir=tmp_path / "c", dense_dir=tmp_path / "dense", api_key=None
    )
    reason = retrieval_status(all_missing_key, _settings(tmp_path))["dense"].reason
    # chebi has an index but no key; the other three have no index: two distinct reasons.
    assert reason is not None and reason.count("|") == 3
    same = indexes_mod.open_indexes(cache_dir=tmp_path / "c", dense_dir=None, api_key=None)
    once = retrieval_status(same, _settings(tmp_path))["dense"].reason
    assert once == "no dense index directory configured (NER_SITE_DENSE_DIR) (all 4 ontologies)"


# --- review fixes -------------------------------------------------------------------


def test_a_non_ascii_passcode_is_a_settings_error() -> None:
    """http.server reads header values as Latin-1, so a non-ASCII passcode could never
    match what a visitor sends; it is refused at settings load, naming the variable."""
    with pytest.raises(SettingsError) as exc:
        Settings.from_env({"NER_SITE_PASSCODE": "pässwort-geheim"})
    assert "NER_SITE_PASSCODE" in str(exc.value) and "ASCII" in str(exc.value)
    assert "pässwort" not in str(exc.value)
    assert Settings.from_env({"NER_SITE_PASSCODE": "plain-ascii-ok"}).passcode


def test_the_default_prompt_follows_the_configured_entity_limit(tmp_path: Path) -> None:
    s = _settings(tmp_path, NER_SITE_MAX_ENTITIES="7")
    assert "at most 7 entities" in s.default_prompt
    got = validate_request(NerRequest(text="abc"), s)
    assert got.prompt == s.default_prompt
    assert _settings(tmp_path).default_prompt == DEFAULT_PROMPT


class _LeakyRetriever:
    """Fails with the key, a URL with userinfo and a very long message."""

    name = "dense-qwen3"
    fingerprint = "leaky"
    default_limit = 5

    async def search(self, request: Any) -> Any:
        raise RuntimeError(
            f"HTTP 401: No auth credentials: Bearer {SENTINEL_KEY} "
            f"at https://user:hunter2@emb.example/v1/embeddings " + "x" * 2000
        )


def test_run_scrubs_and_caps_entity_notes_and_warnings(tmp_path: Path, bm25_only: Any) -> None:
    chebi = bm25_only["chebi-sample"]
    leaky = {
        "chebi-sample": indexes_mod.OntologyIndex(
            "chebi-sample", chebi.store, [*chebi.retrievers, _LeakyRetriever()], chebi.status
        )
    }
    llm = FakeLLM([_answer(("haloperidol", "chemical", 0, 11))])
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    out = asyncio.run(run(NerRequest(text="haloperidol"), settings=s, llm=llm, indexes=leaky))
    (entity,) = out.entities
    assert entity.matches and entity.matches[0].id == "CHEBI:5613"  # BM25 still serves
    leaked = [n for n in entity.notes if "No auth credentials" in n]
    assert leaked, entity.notes
    for text in [*entity.notes, *out.warnings]:
        assert SENTINEL_KEY not in text and "hunter2" not in text
        assert len(text) <= service.MAX_ERROR_DETAIL + 40
    assert "emb.example" in leaked[0] and "[redacted]" in leaked[0]


# --- selection --------------------------------------------------------------------


def test_selectors_are_offered_and_validated(tmp_path: Path) -> None:
    from ner_site.select import JEV, RETRIEVAL_ONLY

    plain = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    assert plain.offered_selectors[0] == RETRIEVAL_ONLY and JEV in plain.offered_selectors
    assert plain.default_selector == RETRIEVAL_ONLY  # the helper's NER_SITE_SELECTOR
    assert validate_request(NerRequest(text="ok"), plain).selector == RETRIEVAL_ONLY
    jev = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_SELECTOR="jev")
    assert jev.default_selector == JEV
    assert validate_request(NerRequest(text="ok"), jev).selector == JEV
    chosen = validate_request(NerRequest(text="ok", selector=RETRIEVAL_ONLY), jev)
    assert chosen.selector == RETRIEVAL_ONLY
    with pytest.raises(ServiceError) as exc:
        validate_request(NerRequest(text="ok", selector="openai/gpt-1"), jev)
    assert exc.value.code == "unknown_selector" and exc.value.status == 400
    own = _settings(tmp_path, NER_SITE_SELECTORS="jev, qwen/qwen3.7-flash")
    assert own.offered_selectors == (RETRIEVAL_ONLY, JEV, "qwen/qwen3.7-flash")
    with pytest.raises(SettingsError):
        _settings(tmp_path, NER_SITE_SELECTOR="nobody")
    offline = _settings(tmp_path, NER_SITE_OFFLINE="1")
    assert offline.offered_selectors == (RETRIEVAL_ONLY, JEV)


def test_run_with_jev_picks_among_the_retrieved_terms(tmp_path: Path, bm25_only: Any) -> None:
    from xwalk.decide.fake import FakeDecider

    text = "She took haloperidol for a fever."
    llm = FakeLLM([_answer(("haloperidol", "chemical", 9, 20))])
    decider = FakeDecider()
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_SELECTOR="jev")
    out = asyncio.run(
        run(NerRequest(text=text), settings=s, llm=llm, indexes=bm25_only, decider=decider)
    )
    (entity,) = out.entities
    assert out.selector == "jev"
    assert entity.pick is not None and entity.pick.selector == "jev"
    assert entity.pick.status in {"match", "review", "no_match"}
    assert entity.pick.id is None or entity.pick.id.startswith("CHEBI:")
    assert out.usage.selector_calls == len(decider.calls) > 0
    assert out.usage.selector_prompt_tokens == 100 * len(decider.calls)
    assert "select" in out.timing_ms
    # Jev saw the mention in its sentence.
    assert any("took" in json.dumps(state) for state, _ in decider.calls)


def test_run_with_an_llm_selector_needs_its_client(tmp_path: Path, bm25_only: Any) -> None:
    text = "She took haloperidol for a fever."
    s = _settings(
        tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, NER_SITE_SELECTORS="qwen/qwen3.7-flash"
    )
    picker = FakeLLM(['{"chosen_key": "C01", "confidence_score": 0.7, "explanation": "first"}'])
    out = asyncio.run(
        run(
            NerRequest(text=text, selector="qwen/qwen3.7-flash"),
            settings=s,
            llm=FakeLLM([_answer(("haloperidol", "chemical", 9, 20))]),
            indexes=bm25_only,
            select_llm=picker,
        )
    )
    (entity,) = out.entities
    assert entity.pick is not None and entity.pick.status == "match"
    assert entity.pick.id == entity.matches[0].id == "CHEBI:5613"
    assert out.usage.selector_calls == 1 and out.usage.selector_cost_usd is None


# --- real-world fixes (4 November 2026) -------------------------------------------


def _capture(tmp_path: Path, model: str, maker: Any, **env: str) -> dict[str, Any]:
    """The JSON body the client posts for one extraction call to `model`."""
    seen: list[httpx.Request] = []

    def chat(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [{"message": {"content": '{"entities": []}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY, **env)
    llm = maker(s, model, transport=httpx.MockTransport(chat))

    async def go() -> None:
        try:
            await extract(llm, "nothing here", DEFAULT_PROMPT, max_entities=25)
        finally:
            await llm.aclose()

    asyncio.run(go())
    (request,) = seen
    body: dict[str, Any] = json.loads(request.content)
    return body


@pytest.mark.parametrize("maker", [make_llm, service.make_select_llm])
def test_reasoning_is_switched_off_on_openrouter(tmp_path: Path, maker: Any) -> None:
    """GPT-6 Luna and Claude Haiku 5.5 spent all 2048 output tokens on hidden reasoning
    and returned nothing on real abstracts; with reasoning off they answer."""
    for model in (
        "openai/gpt-6-luna",
        "anthropic/claude-haiku-5.5",
        "google/gemini-3.1-flash-lite",
    ):
        assert _capture(tmp_path, model, maker)["reasoning"] == {"enabled": False}
    other = _capture(tmp_path, "gpt-x", maker, NER_SITE_BASE_URL="https://llm.example/v1")
    assert "reasoning" not in other  # an unknown field for other OpenAI-compatible APIs


@pytest.mark.parametrize("maker", [make_llm, service.make_select_llm])
def test_anthropic_models_are_asked_for_json_in_the_prompt_not_a_schema(
    tmp_path: Path, maker: Any
) -> None:
    """With reasoning off, Claude Haiku 5.5 under a strict json_schema returned one
    entity with zero offsets; asked in the prompt it returned them all."""
    assert "response_format" not in _capture(tmp_path, "anthropic/claude-haiku-5.5", maker)
    assert "response_format" in _capture(tmp_path, "openai/gpt-6-luna", maker)
    assert "response_format" in _capture(tmp_path, "google/gemini-3.1-flash-lite", maker)


def test_the_full_catalog_neither_asks_for_nor_keeps_genes(tmp_path: Path) -> None:
    from ner_site.extract import build_request, parse_entities

    full = ("chemical", "disease", "food", "other")
    request = build_request("x", "find things", types=full)
    assert '"gene"' not in request.system
    assert request.schema is not None
    items = request.schema["properties"]["entities"]["items"]
    assert items["properties"]["type"]["enum"] == list(full)
    raw = json.dumps({"entities": [{"text": "BRCA1", "type": "gene", "start": 0, "end": 5}]})
    (entity,), _ = parse_entities(raw, "BRCA1 is a gene.", max_entities=5, types=full)
    assert entity.type == "other"
    every = build_request("x", "find things")
    assert '"gene"' in every.system


def _key_answer(status: int, payload: dict[str, Any]) -> httpx.MockTransport:
    def answer(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/v1/key"
        assert request.headers["authorization"] == f"Bearer {SENTINEL_KEY}"
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(answer)


def test_health_reports_a_key_openrouter_refuses(tmp_path: Path) -> None:
    """An expired key left /api/health at llm ok while every POST failed with 502."""
    service.reset_key_check()
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    expired = _key_answer(401, {"error": {"message": "API key expired.", "code": 401}})
    status = service.check_key(s, transport=expired)
    assert status is not None and status.status == "unavailable"
    assert status.reason is not None and "API key expired." in status.reason
    assert SENTINEL_KEY not in status.reason
    # Remembered for a while: no second request.
    assert service.check_key(s, transport=_key_answer(500, {})) == status


def test_health_reports_a_spent_key_and_passes_a_good_one(tmp_path: Path) -> None:
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    service.reset_key_check()
    spent = _key_answer(200, {"data": {"limit": 5.0, "limit_remaining": 0.0, "usage": 5.0}})
    status = service.check_key(s, transport=spent)
    assert status is not None and status.status == "unavailable"
    assert status.reason is not None and "limit" in status.reason
    service.reset_key_check()
    good = _key_answer(200, {"data": {"limit": 5.0, "limit_remaining": 4.2, "usage": 0.8}})
    assert service.check_key(s, transport=good) == service.RetrieverStatus(status="ok")


def test_a_key_check_that_cannot_reach_openrouter_is_not_an_outage(tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    service.reset_key_check()
    s = _settings(tmp_path, OPENROUTER_API_KEY=SENTINEL_KEY)
    status = service.check_key(s, transport=httpx.MockTransport(down))
    assert status is not None and status.status == "ok"
    assert status.reason is not None and "not checked" in status.reason
    assert service.check_key(_settings(tmp_path), transport=httpx.MockTransport(down)) is None


# --- Claude as the default provider (spec 2026-11-18, section 4) ----------------------

ANTHROPIC_KEY = "sk-ant-SENTINEL-feedbeef-9191"
OPUS, SONNET = "claude-opus-5-5", "claude-sonnet-5-5"
GEMINI, LUNA = "google/gemini-3.1-flash-lite", "openai/gpt-6-luna"
DB_URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"


def test_claude_leads_the_defaults_when_both_keys_are_set() -> None:
    from ner_site.limits import Allowance

    s = Settings.from_env({"OPENROUTER_API_KEY": SENTINEL_KEY, "ANTHROPIC_API_KEY": ANTHROPIC_KEY})
    assert s.anthropic_api_key == ANTHROPIC_KEY and s.configured
    assert s.models == service.DEFAULT_MODELS == (OPUS, SONNET, GEMINI, LUNA)
    assert s.offered_models == (OPUS, SONNET, GEMINI, LUNA) and s.default_model == OPUS
    assert s.selectors == service.DEFAULT_SELECTORS == (OPUS, SONNET, "jev", GEMINI, LUNA)
    assert s.offered_selectors == ("retrieval", OPUS, SONNET, "jev", GEMINI, LUNA)
    assert s.default_selector == OPUS
    assert s.claude_effort == "low" and s.fallback_model == GEMINI
    assert s.allowance == Allowance(opus_per_hour=10, sonnet_per_hour=20)
    assert s.visitor_salt is None and s.database_url is None


def test_without_the_anthropic_key_nothing_about_claude_is_offered() -> None:
    s = Settings.from_env({"OPENROUTER_API_KEY": SENTINEL_KEY})
    assert s.offered_models == (GEMINI, LUNA) and s.default_model == GEMINI
    assert s.offered_selectors == ("retrieval", "jev", GEMINI, LUNA)
    assert s.default_selector == "jev" and s.configured
    with pytest.raises(ServiceError) as exc:
        validate_request(NerRequest(text="ok", model=OPUS, selector="retrieval"), s)
    assert exc.value.code == "unknown_model"
    with pytest.raises(ServiceError) as exc:
        validate_request(NerRequest(text="ok", selector=SONNET), s)
    assert exc.value.code == "unknown_selector"
    # No key at all: not configured, and what is listed is what an OpenRouter key would
    # offer, as before.
    bare = Settings.from_env({})
    assert not bare.configured and bare.offered_models == (GEMINI, LUNA)
    assert bare.default_selector == "jev"


def test_without_the_openrouter_key_only_claude_is_offered() -> None:
    s = Settings.from_env({"ANTHROPIC_API_KEY": ANTHROPIC_KEY})
    assert s.configured and s.api_key is None
    assert s.offered_models == (OPUS, SONNET) and s.default_model == OPUS
    assert s.offered_selectors == ("retrieval", OPUS, SONNET) and s.default_selector == OPUS
    with pytest.raises(ServiceError) as exc:
        validate_request(NerRequest(text="ok", selector="jev"), s)
    assert exc.value.code == "unknown_selector"


def test_offline_offers_no_claude_even_with_the_key() -> None:
    s = Settings.from_env({"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "NER_SITE_OFFLINE": "1"})
    assert s.offered_models == ("offline",) and s.offered_selectors == ("retrieval", "jev")
    assert s.default_selector == "jev"


def test_a_configured_selector_is_the_default_while_it_is_offered() -> None:
    both = {"OPENROUTER_API_KEY": SENTINEL_KEY, "ANTHROPIC_API_KEY": ANTHROPIC_KEY}
    assert Settings.from_env({**both, "NER_SITE_SELECTOR": "jev"}).default_selector == "jev"
    kept = Settings.from_env({**both, "NER_SITE_SELECTOR": "retrieval"})
    assert kept.default_selector == "retrieval"
    # Named, but its key is missing: the first selector that can run, not a 503.
    lost = Settings.from_env({"OPENROUTER_API_KEY": SENTINEL_KEY, "NER_SITE_SELECTOR": OPUS})
    assert lost.default_selector == "jev"
    with pytest.raises(SettingsError, match="NER_SITE_SELECTOR"):
        Settings.from_env({**both, "NER_SITE_SELECTOR": "nobody"})


def test_the_claude_effort_and_the_fallback_model_are_read_and_checked() -> None:
    s = Settings.from_env({"NER_SITE_CLAUDE_EFFORT": " High ", "NER_SITE_FALLBACK_MODEL": LUNA})
    assert s.claude_effort == "high" and s.fallback_model == LUNA
    for name, value in (
        ("NER_SITE_CLAUDE_EFFORT", "max"),
        ("NER_SITE_CLAUDE_EFFORT", "0"),
        ("NER_SITE_FALLBACK_MODEL", SONNET),
        ("NER_SITE_FALLBACK_MODEL", "retrieval"),
        ("NER_SITE_FALLBACK_MODEL", "jev"),
    ):
        with pytest.raises(SettingsError) as exc:
            Settings.from_env({name: value})
        assert name in str(exc.value)


@pytest.mark.parametrize("name", ["NER_SITE_OPUS_PER_HOUR", "NER_SITE_SONNET_PER_HOUR"])
def test_the_allowance_is_read_and_refused_out_of_range(name: str) -> None:
    from ner_site.limits import Allowance

    low = Settings.from_env({name: "0"}).allowance
    high = Settings.from_env({name: "1000"}).allowance
    field = "opus_per_hour" if "OPUS" in name else "sonnet_per_hour"
    assert getattr(low, field) == 0 and getattr(high, field) == 1000
    assert Settings.from_env({name: " "}).allowance == Allowance()
    for bad in ("1001", "-1", "ten", "2.5"):
        with pytest.raises(SettingsError) as exc:
            Settings.from_env({name: bad})
        assert name in str(exc.value) and "0 to 1000" in str(exc.value)


@pytest.mark.parametrize("name", ["NER_SITE_MODELS", "NER_SITE_SELECTORS"])
def test_an_unknown_claude_id_is_a_settings_error(name: str) -> None:
    with pytest.raises(SettingsError) as exc:
        Settings.from_env({name: f"{OPUS}, claude-haiku-5-5"})
    assert name in str(exc.value) and "claude-haiku-5-5" in str(exc.value)
    # An OpenRouter id of an Anthropic model is not a Claude id of this site.
    ok = Settings.from_env({name: f"anthropic/claude-haiku-5.5, {SONNET}"})
    assert SONNET in (ok.models if name == "NER_SITE_MODELS" else ok.selectors)


def test_a_model_list_that_no_key_can_run_is_a_settings_error() -> None:
    with pytest.raises(SettingsError, match="NER_SITE_MODELS"):
        Settings.from_env({"OPENROUTER_API_KEY": SENTINEL_KEY, "NER_SITE_MODELS": OPUS})
    with pytest.raises(SettingsError, match="NER_SITE_MODELS"):
        Settings.from_env({"ANTHROPIC_API_KEY": ANTHROPIC_KEY, "NER_SITE_MODELS": GEMINI})
    assert Settings.from_env({"NER_SITE_MODELS": OPUS, "NER_SITE_OFFLINE": "1"}).offline


def test_both_keys_and_the_salt_are_scrubbed_and_stay_out_of_repr() -> None:
    s = Settings.from_env(
        {
            "OPENROUTER_API_KEY": SENTINEL_KEY,
            "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
            "NER_SITE_VISITOR_SALT": "salt-SENTINEL-0001",
        }
    )
    assert s.visitor_salt == "salt-SENTINEL-0001"
    assert {SENTINEL_KEY, ANTHROPIC_KEY, "salt-SENTINEL-0001"} <= set(s.secrets())
    for secret in (SENTINEL_KEY, ANTHROPIC_KEY, "salt-SENTINEL-0001"):
        assert secret not in repr(s)
    scrubbed = service.scrub(f"401 for {ANTHROPIC_KEY} and {SENTINEL_KEY}", s)
    assert scrubbed == "401 for [redacted] and [redacted]"


def test_the_database_url_is_read_for_the_limiter_on_any_backend() -> None:
    s = Settings.from_env({"DATABASE_URL": DB_URL})
    assert s.backend == "bundled" and s.database_url == DB_URL
    assert DB_URL in s.secrets() and "npg_SENTINELpw0123" in s.secrets()
    assert "SENTINELpw" not in repr(s)
    # The index backend choice is unchanged: bundled indexes never see the database.
    assert s.index_settings().database_url is None and s.index_settings().backend == "bundled"
    neon = Settings.from_env({"DATABASE_URL": DB_URL, "NER_SITE_BACKEND": "neon"})
    assert neon.index_settings().database_url == DB_URL
    with pytest.raises(SettingsError, match="DATABASE_URL"):
        Settings.from_env({"DATABASE_URL": "mysql://u:p@h/db"})


def test_every_new_variable_is_part_of_the_settings_cache_key() -> None:
    assert {
        "ANTHROPIC_API_KEY",
        "NER_SITE_CLAUDE_EFFORT",
        "NER_SITE_FALLBACK_MODEL",
        "NER_SITE_OPUS_PER_HOUR",
        "NER_SITE_SONNET_PER_HOUR",
        "NER_SITE_VISITOR_SALT",
    } <= set(service.ENV_NAMES)
