"""A visitor's own API key (spec 2026-11-18 own keys, section 5): offline, no real key.

Every key here is a sentinel, and the tests assert its absence wherever a string leaves
the module: error messages, reprs, the call budget's memory, the answer. The clients are
the real ones (the ``anthropic`` SDK and the engine's OpenAI-compatible client) over mock
transports, so what is asserted is the HTTP request each would send; nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import pickle
from email.message import Message
from typing import Any

import httpx
import httpx2
import pytest
from ner_site import claude, fallback, providers, service
from ner_site.claude import ClaudeClient
from ner_site.providers import (
    BASE_URLS,
    KEY_MAX,
    KEY_MIN,
    OWN_PROVIDERS,
    PROVIDER_NAMES,
    SITE,
    Credential,
    failure_word,
    own_llm,
    provider_error,
    read_credential,
)
from ner_site.schema import ServiceError
from xwalk.llm.base import (
    LLMClient,
    LLMError,
    LLMFatalError,
    LLMRequest,
    LLMRetryableError,
    ParseError,
)
from xwalk.llm.budget import BudgetedLLM, CallLimitExceeded
from xwalk.records import Usage

KEY = "sk-SENTINEL-visitor-A-0123456789abcdef"
OTHER_KEY = "sk-SENTINEL-visitor-B-fedcba9876543210"
SITE_OPENROUTER_KEY = "sk-or-OPERATOR-do-not-use-0123456789"
SITE_ANTHROPIC_KEY = "sk-ant-OPERATOR-do-not-use-0123456789"
UPSTREAM = "UPSTREAM-WORDS-that-must-not-travel"
# What a provider may say about a bad key: the key itself, and a masked form of it.
ECHO = f"Incorrect API key provided: {KEY} ({KEY[:6]}...{KEY[-4:]}). {UPSTREAM}"
# The operator's environment may carry headers for the site's own Anthropic client. The
# SDK applies them after the credential, so each of these would replace the visitor's.
OPERATOR_HEADERS = "\n".join(
    [
        f"Authorization: Bearer {SITE_ANTHROPIC_KEY}",
        f"X-Api-Key: {SITE_ANTHROPIC_KEY}",
        "anthropic-beta: server-side-fallback-2026-07-01",
        "anthropic-version: 1999-01-01",
        "User-Agent: OPERATOR-agent",
        "X-Operator-Org: OPERATOR-org",
    ]
)
# The names an own-key Anthropic request may carry: the SDK's own, and httpx's.
ANTHROPIC_HEADER_NAMES = {
    "accept",
    "accept-encoding",
    "connection",
    "content-length",
    "content-type",
    "host",
    "user-agent",
    "authorization",
    "anthropic-version",
}
ALL = list(OWN_PROVIDERS)
MODELS = {
    "anthropic": claude.HAIKU,
    "openai": "gpt-6-luna",
    "openrouter": "google/gemini-3.1-flash-lite",
}
HOSTS = {
    "anthropic": "api.anthropic.com",
    "openai": "api.openai.com",
    "openrouter": "openrouter.ai",
}
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}
REQUEST = LLMRequest(system="You extract entities.", user="Aspirin reduced fever.", schema=SCHEMA)


def bearer(key: str = KEY) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def read(
    provider: str | None,
    headers: Any,
    *,
    enabled: Any = ALL,
    text: str = "Aspirin reduced fever.",
    prompt: str | None = None,
) -> Credential | None:
    return read_credential(provider, headers, enabled=enabled, text=text, prompt=prompt)


def strings(error: ServiceError) -> str:
    """Everything of an error that could reach a response or a log."""
    return " ".join(
        [str(error), repr(error), error.message, json.dumps(error.envelope()), repr(error.args)]
    )


def refused(provider: Any, headers: Any, **kw: Any) -> ServiceError:
    with pytest.raises(ServiceError) as caught:
        read(provider, headers, **kw)
    error = caught.value
    assert (error.status, error.code) == (400, "bad_credential")
    for secret in (KEY, OTHER_KEY, KEY[:12], "SENTINEL"):
        assert secret not in strings(error)
    assert error.headers == {}
    return error


# --- constants -------------------------------------------------------------------------


def test_constants() -> None:
    assert SITE == "site"
    assert OWN_PROVIDERS == ("anthropic", "openai", "openrouter")
    assert PROVIDER_NAMES == {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
    }
    assert BASE_URLS == {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    assert (KEY_MIN, KEY_MAX) == (20, 512)


def test_provider_info_is_plain_frozen_data() -> None:
    info = providers.ProviderInfo(
        id="openai",
        name=PROVIDER_NAMES["openai"],
        models=("gpt-6-luna",),
        default_model="gpt-6-luna",
    )
    assert dataclasses.asdict(info) == {
        "id": "openai",
        "name": "OpenAI",
        "models": ("gpt-6-luna",),
        "default_model": "gpt-6-luna",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.id = "x"  # type: ignore[misc]


# --- read_credential -------------------------------------------------------------------


@pytest.mark.parametrize("provider", [None, "site"])
def test_the_sites_keys_need_no_credential(provider: str | None) -> None:
    assert read(provider, {}) is None
    assert read(provider, {"X-Passcode": "open sesame", "Content-Type": "application/json"}) is None
    # Also with own keys switched off.
    assert read(provider, {}, enabled=()) is None


@pytest.mark.parametrize("provider", [None, "site"])
@pytest.mark.parametrize(
    "headers",
    [
        bearer(),
        {"authorization": f"Bearer {KEY}"},
        {"AUTHORIZATION": [f"Bearer {KEY}"]},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": ""},
    ],
)
def test_a_site_request_must_not_carry_a_key(provider: str | None, headers: Any) -> None:
    error = refused(provider, headers)
    assert "Authorization" in error.message
    assert refused(provider, headers, enabled=()).message == error.message


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
def test_a_good_bearer_key_gives_a_credential_that_never_shows_it(provider: str) -> None:
    credential = read(provider, bearer())
    assert isinstance(credential, Credential)
    assert credential.provider == provider
    assert credential.source == "visitor"
    assert credential.token() == KEY
    shown = [repr(credential), str(credential), f"{credential}", f"{credential!r:>80}"]
    shown += [repr(getattr(credential, f.name)) for f in dataclasses.fields(credential)]
    shown += [str(getattr(credential, f.name)) for f in dataclasses.fields(credential)]
    shown += [repr(dataclasses.asdict(credential)), repr(copy.deepcopy(credential))]
    shown += [repr(dataclasses.replace(credential, provider="openai"))]
    for text in shown:
        assert KEY not in text and "SENTINEL" not in text
    with pytest.raises(dataclasses.FrozenInstanceError):
        credential.provider = "openai"  # type: ignore[misc]
    with pytest.raises(TypeError):
        pickle.dumps(credential)


@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": f"Bearer {KEY}"},
        {"AUTHORIZATION": f"bearer {KEY}"},
        {"Authorization": f"BEARER {KEY}"},
        {"Authorization": [f"Bearer {KEY}"]},
        {"Authorization": (f"Bearer {KEY}",)},
        {"Authorization": f"  Bearer {KEY}\t"},
        {"Authorization": f"Bearer {KEY}", "X-Requested-With": "xwalk-ner", "Cookie": "a=b"},
    ],
)
def test_the_header_is_found_in_any_case_and_as_a_one_value_sequence(headers: Any) -> None:
    credential = read("openai", headers)
    assert credential is not None and credential.token() == KEY


def test_a_message_object_with_one_header_is_read_and_with_two_refused() -> None:
    """``http.server`` hands over an ``email.message.Message``, whose ``items()`` repeats a
    repeated header; a dict built from it would silently keep one of the two."""
    message = Message()
    message["Authorization"] = f"Bearer {KEY}"
    credential = read("anthropic", message)
    assert credential is not None and credential.token() == KEY
    message["authorization"] = f"Bearer {OTHER_KEY}"
    assert "exactly one" in refused("anthropic", message).message


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Api-Key": KEY},
        {"Authorization": []},
        {"Authorization": [f"Bearer {KEY}", f"Bearer {OTHER_KEY}"]},
        {"Authorization": [f"Bearer {KEY}", f"Bearer {KEY}"]},
        {"Authorization": f"Bearer {KEY}", "authorization": f"Bearer {KEY}"},
        {"Authorization": KEY},
        {"Authorization": f"Basic {KEY}"},
        {"Authorization": f"Token {KEY}"},
        {"Authorization": f"Bearer{KEY}"},
        {"Authorization": f"Bearer\t{KEY}"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer "},
        {"Authorization": ""},
        {"Authorization": [None]},
        {"Authorization": None},
        {"Authorization": f"Bearer {KEY}".encode()},
    ],
)
def test_a_missing_repeated_or_malformed_header_is_refused(headers: Any) -> None:
    error = refused("openai", headers)
    assert "exactly one Authorization header" in error.message


@pytest.mark.parametrize(
    "key",
    [
        KEY[: KEY_MIN - 1],
        "x" * (KEY_MAX + 1),
        KEY + "x" * 100_000,
        f"{KEY} {OTHER_KEY}",
        f" {KEY}",
        f"{KEY},Bearer {OTHER_KEY}",
        f"{KEY}\t{OTHER_KEY}",
        f"{KEY}\n",
        f"{KEY}\r\nX-Injected: 1",
        f"{KEY}\x00",
        f"{KEY}\x7f",
        f"{KEY}é",
        f"{KEY} ",
        f"{KEY} ",
        "ｋｅｙ" * 10,
    ],
)
def test_a_key_of_the_wrong_shape_is_refused(key: str) -> None:
    error = refused("anthropic", {"Authorization": f"Bearer {key}"})
    assert error.message == (
        "the API key must be 20 to 512 printable ASCII characters without spaces"
    )


@pytest.mark.parametrize("length", [KEY_MIN, KEY_MAX])
def test_the_length_bounds_are_inclusive(length: int) -> None:
    key = ("sk-" + "a~!" * 200)[:length]
    credential = read("openrouter", bearer(key))
    assert credential is not None and credential.token() == key


@pytest.mark.parametrize(
    ("provider", "enabled"),
    [
        ("openai", ("anthropic", "openrouter")),
        ("anthropic", ()),
        ("chatgpt", ALL),
        ("chatgpt", ("chatgpt",)),
        ("Anthropic", ALL),
        ("", ALL),
        ("site ", ALL),
        # A key pasted into the wrong field does not come back in the refusal.
        (KEY, ALL),
        (KEY, (KEY,)),
        (7, ALL),
        (["openai"], ALL),
    ],
)
def test_a_provider_that_is_not_enabled_is_refused(provider: Any, enabled: Any) -> None:
    error = refused(provider, bearer(), enabled=enabled)
    assert error.message == "own API keys are not available for that provider on this site"
    # The same answer with no header at all: the provider is checked first.
    assert refused(provider, {}, enabled=enabled).message == error.message


def test_a_key_that_is_also_in_the_text_or_the_prompt_is_refused() -> None:
    in_text = refused("openai", bearer(), text=f"my key is {KEY}, annotate this")
    in_prompt = refused("openai", bearer(), prompt=f"Use {KEY}.")
    assert in_text.message == in_prompt.message
    assert "text or the prompt" in in_text.message
    # Another key in the text is the visitor's business; no prompt is fine.
    assert read("openai", bearer(), text=f"see {OTHER_KEY}", prompt=None) is not None
    assert read("openai", bearer(), text="plain", prompt="plain") is not None


# --- own_llm: the request each provider's client sends ---------------------------------


def chat_body(model: str, content: str = '{"entities": []}') -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


def message_body(model: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": '{"entities": []}'}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 7, "output_tokens": 3},
        **extra,
    }


class Wire:
    """A mock transport for one provider: records requests, answers from a script of
    ``(status, json)`` pairs, exceptions or callables (the last one repeats)."""

    def __init__(self, provider: str, *script: Any) -> None:
        self.provider = provider
        self.requests: list[Any] = []
        self._script = list(script) or [None]
        lib: Any = httpx2 if provider == "anthropic" else httpx
        self._lib = lib
        self.transport = lib.MockTransport(self._handle)

    async def _handle(self, request: Any) -> Any:
        self.requests.append(request)
        await asyncio.sleep(0)  # let concurrent calls interleave
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if callable(step) and not isinstance(step, type):
            step = step(request)
        if isinstance(step, type) and issubclass(step, Exception):
            raise step(f"failed with {ECHO}", request=request)
        if isinstance(step, self._lib.Response):
            return step
        if step is None:
            model = json.loads(request.content)["model"]
            body = message_body(model) if self.provider == "anthropic" else chat_body(model)
            return self._lib.Response(200, json=body)
        status, body = step
        if isinstance(body, str):
            return self._lib.Response(status, text=body)
        return self._lib.Response(status, json=body)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def credential_for(provider: str, key: str = KEY) -> Credential:
    credential = read_credential(provider, bearer(key), enabled=ALL, text="", prompt=None)
    assert credential is not None
    return credential


def build(wire: Wire, *, key: str = KEY, purpose: Any = "extract", **kw: Any) -> LLMClient:
    options: dict[str, Any] = {"effort": "low", "max_entities": 30, "call_cap": 3, **kw}
    return own_llm(
        credential_for(wire.provider, key),
        options.pop("model", MODELS[wire.provider]),
        purpose=purpose,
        transport=wire.transport,
        backoff_base=0.0,
        **options,
    )


def headers_text(request: Any) -> str:
    return " ".join(f"{name}: {value}" for name, value in request.headers.multi_items())


@pytest.fixture(autouse=True)
def operator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator's own keys and a hostile environment are present in every test; no
    request built for a visitor may show any trace of them."""
    monkeypatch.setenv("OPENROUTER_API_KEY", SITE_OPENROUTER_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", SITE_ANTHROPIC_KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SITE_ANTHROPIC_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", SITE_OPENROUTER_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.invalid")
    # Everything else the SDK (2.10.0 ``_client.py``, ``_tracing.py``) reads from the
    # environment that could add a header, a credential or a host to a client.
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", OPERATOR_HEADERS)
    monkeypatch.setenv("ANTHROPIC_PROFILE", "OPERATOR-profile")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent/OPERATOR-config")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", SITE_ANTHROPIC_KEY)
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "OPERATOR-rule")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "OPERATOR-org")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "OPERATOR-workspace")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "OPERATOR-account")
    monkeypatch.setenv("ANTHROPIC_WEBHOOK_SIGNING_KEY", SITE_ANTHROPIC_KEY)
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY", "true")
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_PROPAGATION", "true")
    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "content")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.invalid/v1")
    monkeypatch.setenv("NER_SITE_BASE_URL", "https://elsewhere.invalid/v1")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_the_client_is_a_budgeted_llm_client_that_shows_no_key(provider: str) -> None:
    llm = build(Wire(provider))
    assert isinstance(llm, BudgetedLLM) and isinstance(llm, LLMClient)
    assert llm.model == MODELS[provider]
    assert llm.capabilities.usage_reporting is True
    for text in (repr(llm), repr(llm.inner), llm.fingerprint, repr(vars(llm.inner))):
        assert KEY not in text
    await llm.aclose()


async def test_openai_request() -> None:
    wire = Wire("openai")
    llm = build(wire)
    response = await llm.complete(REQUEST)
    await llm.aclose()

    (request,) = wire.requests
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    (body,) = wire.bodies
    assert body["model"] == "gpt-6-luna"
    # The limit under the name OpenAI's reasoning models take, with room for hidden
    # reasoning; no sampling parameter, and nothing OpenRouter-specific.
    assert body["max_completion_tokens"] == claude.EXTRACT_MAX_TOKENS == 16000
    for absent in ("max_tokens", "temperature", "top_p", "reasoning", "reasoning_effort", "seed"):
        assert absent not in body
    assert body["messages"] == [
        {"role": "system", "content": "You extract entities."},
        {"role": "user", "content": "Aspirin reduced fever."},
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": SCHEMA, "strict": True},
    }
    assert "http-referer" not in request.headers
    assert response.text == '{"entities": []}'
    assert response.usage == Usage(prompt_tokens=7, completion_tokens=3, calls=1)


async def test_openai_ignores_a_stage_token_limit_and_selects_with_its_own() -> None:
    """The engine's stages ask for 2048 / 600 tokens per request; a reasoning model
    would spend them before answering."""
    wire = Wire("openai")
    llm = build(wire, purpose="select")
    await llm.complete(
        LLMRequest(system="s", user="u", max_tokens=600, temperature=0.0, extra={"temperature": 0})
    )
    await llm.aclose()
    (body,) = wire.bodies
    assert body["max_completion_tokens"] == claude.SELECT_MAX_TOKENS == 8000
    assert "max_tokens" not in body and "temperature" not in body


@pytest.mark.parametrize(("purpose", "max_tokens"), [("extract", 2048), ("select", 600)])
async def test_openrouter_request_is_the_sites_own_with_the_visitors_key(
    purpose: str, max_tokens: int
) -> None:
    wire = Wire("openrouter")
    llm = build(wire, purpose=purpose)
    assert isinstance(llm.inner._inner, service.OpenRouterClient)  # type: ignore[attr-defined]  # noqa: SLF001
    await llm.complete(REQUEST)
    await llm.aclose()

    (request,) = wire.requests
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.headers["http-referer"] == service.SITE_URL
    assert request.headers["x-title"] == service.SITE_TITLE
    (body,) = wire.bodies
    assert body["model"] == "google/gemini-3.1-flash-lite"
    assert body["reasoning"] == {"enabled": False}
    assert (body["max_tokens"], body["temperature"]) == (max_tokens, 0.0)
    assert body["response_format"]["json_schema"]["strict"] is True


async def test_openrouter_asks_an_anthropic_model_for_json_in_the_prompt_as_the_site_does() -> None:
    wire = Wire("openrouter")
    llm = build(wire, model="anthropic/claude-haiku-5.5")
    await llm.complete(REQUEST)
    await llm.aclose()
    assert "response_format" not in wire.bodies[0]


@pytest.mark.parametrize(("purpose", "max_tokens"), [("extract", 16000), ("select", 8000)])
async def test_anthropic_request(purpose: str, max_tokens: int) -> None:
    wire = Wire("anthropic")
    llm = build(wire, purpose=purpose, effort="medium")
    assert isinstance(llm.inner._inner, ClaudeClient)  # type: ignore[attr-defined]  # noqa: SLF001
    response = await llm.complete(REQUEST)
    await llm.aclose()

    (request,) = wire.requests
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    # The SDK (2.10.0) sends an API key as a bearer token; the visitor's, and only once.
    assert request.headers.get_list("authorization") == [f"Bearer {KEY}"]
    assert "x-api-key" not in request.headers
    # No substitute model billed to the visitor: no fallback, and no beta flag at all.
    assert "anthropic-beta" not in request.headers
    (body,) = wire.bodies
    assert "fallbacks" not in body and "betas" not in body
    assert body["model"] == claude.HAIKU
    assert body["max_tokens"] == max_tokens
    assert body["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert not {"temperature", "top_p", "top_k", "thinking", "seed"} & set(body)
    assert response.text == '{"entities": []}'
    assert response.model == claude.HAIKU
    assert getattr(response, "served_by_fallback", None) is False


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
@pytest.mark.parametrize("purpose", ["extract", "select"])
async def test_only_the_visitors_key_and_the_pinned_host(provider: str, purpose: str) -> None:
    wire = Wire(provider)
    llm = build(wire, purpose=purpose)
    await llm.complete(REQUEST)
    await llm.aclose()
    (request,) = wire.requests
    assert (request.url.scheme, request.url.host) == ("https", HOSTS[provider])
    assert request.url.port is None and not request.url.userinfo
    assert str(request.url).startswith(BASE_URLS.get(provider, "https://api.anthropic.com") + "/")
    sent = headers_text(request) + " " + request.content.decode()
    assert sent.count(KEY) == 1
    for foreign in (SITE_OPENROUTER_KEY, SITE_ANTHROPIC_KEY, OTHER_KEY, "OPERATOR", "elsewhere"):
        assert foreign not in sent
    assert request.headers.get_list("authorization") == [f"Bearer {KEY}"]
    for absent in ("anthropic-beta", "x-api-key", "x-operator-org", "traceparent", "tracestate"):
        assert absent not in request.headers
    if provider == "anthropic":
        # Nothing from the operator's environment: no header the SDK would not send on
        # its own, and its own values for the ones it does.
        names = {name for name in request.headers if not name.startswith("x-stainless-")}
        assert names <= ANTHROPIC_HEADER_NAMES
        assert request.headers["anthropic-version"] != "1999-01-01"
        assert request.headers["user-agent"].startswith("AsyncAnthropic/Python")


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_redirect_is_not_followed(provider: str) -> None:
    """A redirect would carry the key to wherever ``Location`` points."""
    wire = Wire(provider)
    moved = wire._lib.Response(  # noqa: SLF001
        307, headers={"Location": "https://elsewhere.invalid/v1/collect"}
    )
    wire._script = [moved]  # noqa: SLF001
    llm = build(wire, call_cap=1)
    with pytest.raises(LLMError) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    # The provider's own 307 is the failure; nothing was sent after it.
    assert failure_word(caught.value) in ("bad_request", "error")
    assert str(caught.value).startswith(f"{PROVIDER_NAMES[provider]} request failed (")
    assert [request.url.host for request in wire.requests] == [HOSTS[provider]]


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_without_a_test_transport_no_proxy_and_no_redirect_is_configured(
    provider: str,
) -> None:
    """Built as in production (no transport passed), with proxies in the environment."""
    llm = own_llm(
        credential_for(provider),
        MODELS[provider],
        purpose="extract",
        effort="low",
        max_entities=30,
        call_cap=3,
    )
    inner = llm.inner._inner  # type: ignore[attr-defined]  # noqa: SLF001
    if provider == "anthropic":
        sdk = inner._client  # noqa: SLF001
        assert str(sdk.base_url).rstrip("/") == "https://api.anthropic.com"
        assert sdk.max_retries == 0
        assert sdk.api_key == KEY and sdk.auth_token is None
        http = sdk._client  # noqa: SLF001
    else:
        http = inner._client  # noqa: SLF001
        assert inner._base_url == BASE_URLS[provider]  # noqa: SLF001
    assert http.follow_redirects is False
    assert http.trust_env is False
    assert http._mounts == {}  # noqa: SLF001 - no proxy transport from the environment
    assert "proxy" not in repr(vars(http._transport)).lower()  # noqa: SLF001
    await llm.aclose()


# --- own_llm: the call budget ----------------------------------------------------------


def assert_fixed_call_limit(exc: BaseException, provider: str) -> None:
    """The limit error of an own-key client is one of the fixed ones, whichever layer
    refused the call: not the engine's own text, which quotes the last provider error
    and advises a command-line flag."""
    assert type(exc) is CallLimitExceeded
    assert str(exc) == f"{PROVIDER_NAMES[provider]} call limit for this request reached"
    assert exc.reason == "call_limit"  # type: ignore[attr-defined]
    assert exc.__cause__ is None and exc.__context__ is None
    assert failure_word(exc) == "call_limit"
    assert provider_error(exc, provider).message == (
        f"{PROVIDER_NAMES[provider]} could not complete the request (call_limit)."
    )


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_the_extraction_budget_counts_every_http_request(provider: str) -> None:
    wire = Wire(provider, (500, {"error": {"message": ECHO}}))
    llm = build(wire, call_cap=2)
    with pytest.raises(CallLimitExceeded) as caught:
        await llm.complete(REQUEST)  # two retries allowed, but only two requests
    assert len(wire.requests) == 2
    assert_fixed_call_limit(caught.value, provider)
    assert llm.budget.dispatched == 2  # type: ignore[attr-defined]
    with pytest.raises(CallLimitExceeded) as caught:
        await llm.complete(REQUEST)
    assert_fixed_call_limit(caught.value, provider)
    assert len(wire.requests) == 2
    await llm.aclose()


async def test_the_anthropic_budget_is_one_request_per_call_with_no_hidden_retry() -> None:
    wire = Wire(
        "anthropic",
        (529, {"type": "error", "error": {"type": "overloaded_error", "message": ECHO}}),
    )
    llm = build(wire, call_cap=2)
    for _ in range(2):
        with pytest.raises(LLMRetryableError):
            await llm.complete(REQUEST)
    with pytest.raises(CallLimitExceeded) as caught:
        await llm.complete(REQUEST)
    assert len(wire.requests) == 2
    assert_fixed_call_limit(caught.value, "anthropic")
    await llm.aclose()


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_the_selection_budget_is_two_calls_per_entity(provider: str) -> None:
    wire = Wire(provider)
    llm = build(wire, purpose="select", max_entities=2, call_cap=1)
    for _ in range(4):
        await llm.complete(REQUEST)
    with pytest.raises(CallLimitExceeded) as caught:
        await llm.complete(REQUEST)
    assert_fixed_call_limit(caught.value, provider)
    assert len(wire.requests) == 4
    await llm.aclose()


def test_an_unknown_provider_or_purpose_is_a_programming_error_that_echoes_nothing() -> None:
    odd = Credential(provider=KEY, token=lambda: OTHER_KEY)
    with pytest.raises(ValueError, match="provider") as caught:
        own_llm(odd, "m", purpose="extract", effort="low", max_entities=1, call_cap=1)
    assert KEY not in str(caught.value) and OTHER_KEY not in str(caught.value)
    with pytest.raises(ValueError, match="purpose"):
        own_llm(
            credential_for("openai"),
            "m",
            purpose=KEY,  # type: ignore[arg-type]
            effort="low",
            max_entities=1,
            call_cap=1,
        )


# --- errors: fixed words, nothing from upstream ----------------------------------------


def anthropic_error(type_: str, text: str = ECHO) -> dict[str, Any]:
    return {"type": "error", "error": {"type": type_, "message": text}}


def compat_error(text: str = ECHO, **extra: Any) -> dict[str, Any]:
    return {"error": {"message": text, "type": "invalid_request_error", **extra}}


REFUSED_KEY = (401, "provider_key_refused", "key_refused")
LIMITED = (429, "provider_rate_limited", "rate_limited")


def failed(word: str) -> tuple[int, str, str]:
    return (502, "provider_error", word)


QUOTA = "You exceeded your current quota, please check your plan and billing details. " + ECHO
BALANCE = "Your credit balance is too low to access the Anthropic API. " + ECHO

COMPAT_FAILURES: list[tuple[Any, tuple[int, str, str]]] = [
    ((401, compat_error()), REFUSED_KEY),
    ((403, compat_error()), REFUSED_KEY),
    ((402, compat_error()), REFUSED_KEY),
    ((429, compat_error()), LIMITED),
    ((429, compat_error(QUOTA, code="insufficient_quota")), REFUSED_KEY),
    ((408, compat_error()), failed("timeout")),
    ((500, compat_error()), failed("http_5xx")),
    ((502, f"<html>{ECHO}</html>"), failed("http_5xx")),
    ((503, compat_error()), failed("http_5xx")),
    ((529, compat_error()), failed("overloaded")),
    ((400, compat_error()), failed("bad_request")),
    # A key without access to the model gets a 404 (OpenAI: model_not_found).
    ((404, compat_error()), REFUSED_KEY),
    ((200, {"error": {"code": 404, "message": ECHO}}), REFUSED_KEY),
    ((422, compat_error()), failed("bad_request")),
    ((200, f"<html>{ECHO}</html>"), failed("bad_answer")),
    ((200, {"choices": []}), failed("bad_answer")),
    ((200, {"choices": [{"error": {"message": ECHO}}]}), failed("bad_answer")),
    ((200, {"error": {"code": 401, "message": ECHO}}), REFUSED_KEY),
    ((200, {"error": {"code": 429, "message": ECHO}}), LIMITED),
    ((200, {"error": {"code": 502, "message": ECHO}}), failed("http_5xx")),
    ((200, {"error": {"message": ECHO}}), failed("error")),
    (httpx.ConnectError, failed("unreachable")),
    (httpx.ReadError, failed("unreachable")),
    (httpx.RemoteProtocolError, failed("unreachable")),
    (httpx.ReadTimeout, failed("timeout")),
    (httpx.ConnectTimeout, failed("timeout")),
]

ANTHROPIC_FAILURES: list[tuple[Any, tuple[int, str, str]]] = [
    ((401, anthropic_error("authentication_error")), REFUSED_KEY),
    ((403, anthropic_error("permission_error")), REFUSED_KEY),
    ((402, anthropic_error("billing_error")), REFUSED_KEY),
    ((400, anthropic_error("invalid_request_error", BALANCE)), REFUSED_KEY),
    ((429, anthropic_error("rate_limit_error")), LIMITED),
    ((529, anthropic_error("overloaded_error")), failed("overloaded")),
    ((500, anthropic_error("api_error")), failed("http_5xx")),
    ((503, anthropic_error("api_error")), failed("http_5xx")),
    ((408, anthropic_error("timeout_error")), failed("timeout")),
    ((400, anthropic_error("invalid_request_error")), failed("bad_request")),
    ((404, anthropic_error("not_found_error")), REFUSED_KEY),
    ((413, anthropic_error("request_too_large")), failed("bad_request")),
    (
        (200, message_body(claude.HAIKU, stop_reason="refusal", content=[])),
        failed("refused"),
    ),
    (httpx2.ConnectError, failed("unreachable")),
    (httpx2.ReadTimeout, failed("timeout")),
]

SENTENCES = {
    "provider_key_refused": (
        "{name} refused this key. It may be mistyped, revoked, out of credit, or not "
        "allowed to use this model."
    ),
    "provider_rate_limited": "{name} is rate limiting this key; try again shortly.",
    "provider_error": "{name} could not complete the request ({word}).",
}


def leaks(text: str) -> list[str]:
    """Which of the things that must not travel are in `text`."""
    hostile = (KEY, KEY[:6], KEY[-4:], "SENTINEL", UPSTREAM, "Incorrect API key")
    return [secret for secret in hostile if secret in text]


async def check_failure(provider: str, step: Any, expected: tuple[int, str, str]) -> None:
    status, code, word = expected
    wire = Wire(provider, step)
    llm = build(wire, call_cap=3)
    with pytest.raises(Exception) as caught:  # noqa: PT011 - the class is asserted below
        await llm.complete(REQUEST)
    exc = caught.value
    await llm.aclose()
    assert wire.requests, "the failure must come from the provider, not from the test"

    # The error a client hands on: an engine error in fixed words, with its machine word.
    assert isinstance(exc, LLMError)
    assert exc.reason == word  # type: ignore[attr-defined]
    assert str(exc) == f"{PROVIDER_NAMES[provider]} request failed ({word})"
    assert exc.__cause__ is None and exc.__context__ is None
    assert isinstance(exc.usage, Usage) and exc.usage.calls == len(wire.requests)
    frames = []
    trace = exc.__traceback__
    while trace is not None:
        frames.append(repr(trace.tb_frame.f_locals))
        trace = trace.tb_next
    remembered = llm.budget.provider_error  # type: ignore[attr-defined]
    for text in (str(exc), repr(exc), repr(exc.args), repr(vars(exc)), str(remembered)):
        assert leaks(text) == []
    # The key is in no frame the traceback keeps alive (the test's own locals aside).
    assert all(KEY not in frame for frame in frames[1:])

    assert failure_word(exc) == word
    error = provider_error(exc, provider)
    assert (error.status, error.code) == (status, code)
    assert error.message == SENTENCES[code].format(name=PROVIDER_NAMES[provider], word=word)
    assert leaks(strings(error)) == []
    assert error.headers == {}


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
@pytest.mark.parametrize(("step", "expected"), COMPAT_FAILURES)
async def test_openai_compatible_failures(
    provider: str, step: Any, expected: tuple[int, str, str]
) -> None:
    await check_failure(provider, step, expected)


@pytest.mark.parametrize(("step", "expected"), ANTHROPIC_FAILURES)
async def test_anthropic_failures(step: Any, expected: tuple[int, str, str]) -> None:
    await check_failure("anthropic", step, expected)


def with_reason(error: LLMError, reason: str) -> LLMError:
    error.reason = reason  # type: ignore[attr-defined]
    return error


UNSEALED: list[tuple[BaseException, str]] = [
    # As ner_site.claude raises them for the site's own key.
    (with_reason(LLMRetryableError(f"Anthropic HTTP 429: {ECHO}"), "rate_limited"), "rate_limited"),
    (with_reason(LLMRetryableError(f"Anthropic HTTP 529: {ECHO}"), "overloaded"), "overloaded"),
    (with_reason(LLMFatalError(f"Anthropic HTTP 402: {ECHO}"), "balance_exhausted"), "key_refused"),
    (with_reason(LLMFatalError(f"Anthropic HTTP 401: {ECHO}"), "key_refused"), "key_refused"),
    (with_reason(LLMRetryableError(f"Anthropic APITimeoutError: {ECHO}"), "timeout"), "timeout"),
    (
        with_reason(LLMRetryableError(f"Anthropic HTTP 500 api_error: {ECHO}"), "transient"),
        "http_5xx",
    ),
    (
        with_reason(LLMRetryableError(f"Anthropic APIConnectionError: {ECHO}"), "transient"),
        "unreachable",
    ),
    (
        with_reason(
            LLMRetryableError(f"Anthropic APIResponseValidationError: {ECHO}"), "transient"
        ),
        "bad_answer",
    ),
    (with_reason(LLMFatalError(f"Anthropic HTTP 400: {ECHO}"), "bad_request"), "bad_request"),
    (claude.ClaudeRefused(f"claude-opus-5-5 declined the request {ECHO}"), "refused"),
    # As the engine's OpenAI-compatible client raises them.
    (LLMFatalError(f"HTTP 401: {ECHO}"), "key_refused"),
    (LLMFatalError(f"HTTP 422: HTTP 401: {ECHO}"), "bad_request"),
    (LLMFatalError(f"HTTP 404: {ECHO}"), "key_refused"),
    (
        with_reason(LLMFatalError(f"Anthropic HTTP 404 not_found_error: {ECHO}"), "bad_request"),
        "key_refused",
    ),
    (with_reason(LLMFatalError(f"{ECHO} Anthropic HTTP 404"), "bad_request"), "bad_request"),
    (LLMFatalError(f"HTTP 200 with error 402: {ECHO}"), "key_refused"),
    (LLMRetryableError(f"exhausted 2 retries: HTTP 429: {ECHO}"), "rate_limited"),
    (LLMRetryableError(f"exhausted 2 retries: HTTP 503: {ECHO}"), "http_5xx"),
    (LLMRetryableError(f"exhausted 2 retries: ReadTimeout: {ECHO}"), "timeout"),
    (LLMRetryableError(f"exhausted 2 retries: ConnectError: {ECHO}"), "unreachable"),
    (LLMRetryableError(f"exhausted 2 retries: HTTP 200 with error None: {ECHO}"), "error"),
    (LLMRetryableError(f"malformed response: {ECHO}"), "bad_answer"),
    # A status or a word further into the text is the upstream's, and is not read.
    (LLMFatalError(f"{ECHO} HTTP 401: rate_limited"), "error"),
    (with_reason(LLMFatalError(ECHO), KEY), "error"),
    # Everything else a stage can meet.
    (CallLimitExceeded(f"call limit of 3 reached: {ECHO}"), "call_limit"),
    (asyncio.TimeoutError(), "timeout"),
    (TimeoutError(ECHO), "timeout"),
    (ParseError(f"not JSON: {ECHO}", raw=ECHO), "bad_answer"),
    (httpx.ReadTimeout(ECHO), "timeout"),
    (httpx.ConnectError(ECHO), "unreachable"),
    (httpx2.ConnectError(ECHO), "unreachable"),
    (RuntimeError(ECHO), "error"),
    (KeyError(KEY), "error"),
    (asyncio.CancelledError(ECHO), "error"),
]


@pytest.mark.parametrize(("exc", "word"), UNSEALED, ids=lambda value: type(value).__name__)
@pytest.mark.parametrize("provider", [*OWN_PROVIDERS, "site", KEY])
def test_any_exception_is_named_in_fixed_words(
    provider: str, exc: BaseException, word: str
) -> None:
    assert failure_word(exc) == word
    assert word in providers.WORDS
    error = provider_error(exc, provider)
    name = PROVIDER_NAMES.get(provider, "The provider")
    assert error.message == SENTENCES[error.code].format(name=name, word=word)
    assert error.status == {"key_refused": 401, "rate_limited": 429}.get(word, 502)
    assert leaks(strings(error)) == []


def test_a_client_with_foreign_default_headers_is_refused() -> None:
    """The last check on the Anthropic client: only the SDK's own headers, and the
    visitor's key as the one credential."""
    own = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "AsyncAnthropic/Python 2.10.0",
        "X-Stainless-Lang": "python",
        "X-Stainless-Async": "false",
        "anthropic-version": "2023-06-01",
        "Authorization": f"Bearer {KEY}",
    }
    providers._own_headers_only(own, KEY)  # noqa: SLF001
    for wrong in (
        {**own, "Authorization": f"Bearer {SITE_ANTHROPIC_KEY}"},
        {**own, "authorization": f"Bearer {SITE_ANTHROPIC_KEY}"},
        {**own, "anthropic-beta": "server-side-fallback-2026-07-01"},
        {**own, "X-Api-Key": SITE_ANTHROPIC_KEY},
        {**own, "X-Operator-Org": "OPERATOR-org"},
        {name: value for name, value in own.items() if name != "Authorization"},
    ):
        with pytest.raises(RuntimeError) as caught:
            providers._own_headers_only(wrong, KEY)  # noqa: SLF001
        assert KEY not in str(caught.value) and "OPERATOR" not in str(caught.value)


class Failing:
    """An inner client that raises what it is given."""

    model = "m"
    fingerprint = "f"
    capabilities = None

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def complete(self, request: LLMRequest) -> Any:
        raise self.exc


@pytest.mark.parametrize("retry_after", [10**400, float("nan"), float("inf"), -1, True, "7", KEY])
async def test_an_odd_retry_after_is_dropped_not_fatal_to_the_sealing(retry_after: Any) -> None:
    inner = Failing(LLMRetryableError(f"HTTP 429: {ECHO}", retry_after=retry_after))
    sealed = providers._Sealed(inner, credential_for("openai"))  # type: ignore[arg-type]  # noqa: SLF001
    with pytest.raises(LLMRetryableError) as caught:
        await sealed.complete(REQUEST)
    assert str(caught.value) == "OpenAI request failed (rate_limited)"
    assert caught.value.retry_after is None
    assert caught.value.__context__ is None


async def test_a_usable_retry_after_is_kept() -> None:
    inner = Failing(LLMRetryableError(f"HTTP 429: {ECHO}", retry_after=7))
    sealed = providers._Sealed(inner, credential_for("openai"))  # type: ignore[arg-type]  # noqa: SLF001
    with pytest.raises(LLMRetryableError) as caught:
        await sealed.complete(REQUEST)
    assert caught.value.retry_after == 7.0


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_if_the_sealing_itself_fails_a_fixed_error_leaves_and_nothing_is_chained(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever goes wrong while an error is being replaced, the original must not ride
    out as the context of something else."""

    def broken(exc: BaseException, name: str) -> Exception:
        raise RuntimeError(f"sealing failed on {exc}")

    monkeypatch.setattr(providers, "_sealed_error", broken)
    refusal = (
        (401, anthropic_error("authentication_error"))
        if provider == "anthropic"
        else (401, compat_error())
    )
    llm = build(Wire(provider, refusal))
    with pytest.raises(LLMFatalError) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    exc = caught.value
    assert type(exc) is LLMFatalError
    assert str(exc) == "request failed (error)"
    assert exc.reason == "error"  # type: ignore[attr-defined]
    assert exc.__cause__ is None and exc.__context__ is None
    assert leaks(str(exc) + repr(exc) + repr(vars(exc))) == []
    assert provider_error(exc, provider).status == 502


async def test_a_cancelled_call_is_not_turned_into_a_provider_failure() -> None:
    wire = Wire("openai", lambda request: (_ for _ in ()).throw(asyncio.CancelledError()))
    llm = build(wire)
    with pytest.raises(asyncio.CancelledError):
        await llm.complete(REQUEST)
    await llm.aclose()


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_key_echoed_in_an_answer_is_removed(provider: str) -> None:
    echoed = f"model-{KEY}"
    body = (
        message_body(echoed, content=[{"type": "text", "text": f'{{"entities": ["{KEY}"]}}'}])
        if provider == "anthropic"
        else chat_body(echoed, f'{{"entities": ["{KEY}"]}}')
    )
    wire = Wire(provider, (200, body))
    llm = build(wire)
    response = await llm.complete(REQUEST)
    await llm.aclose()
    assert KEY not in response.text and KEY not in response.model
    assert response.text == '{"entities": ["[redacted]"]}'
    assert response.usage == Usage(prompt_tokens=7, completion_tokens=3, calls=1)
    assert response.structured is True


# --- isolation -------------------------------------------------------------------------


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_two_credentials_used_concurrently_never_cross(provider: str) -> None:
    wires = {KEY: Wire(provider), OTHER_KEY: Wire(provider)}
    clients = {key: build(wire, key=key, call_cap=20) for key, wire in wires.items()}
    order = [KEY, OTHER_KEY] * 8
    responses = await asyncio.gather(*(clients[key].complete(REQUEST) for key in order))
    assert len(responses) == 16
    for key, wire in wires.items():
        other = OTHER_KEY if key == KEY else KEY
        assert len(wire.requests) == 8
        for request in wire.requests:
            sent = headers_text(request)
            assert key in sent and other not in sent
    for client in clients.values():
        await client.aclose()


def module_state() -> dict[str, str]:
    """A picture of everything a request could leave behind at module level."""
    seen: dict[str, str] = {}
    for module in (providers, claude, fallback, service):
        for name, value in vars(module).items():
            if name.startswith("__") or callable(value) or isinstance(value, type(asyncio)):
                continue
            seen[f"{module.__name__}.{name}"] = repr(value)
    return seen


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_request_leaves_no_module_level_state(provider: str) -> None:
    """No cache, no probe result, no Claude block, also after a refused key (which on
    the site's own key blocks Claude for every later request)."""
    fallback.reset()
    service.reset_key_check()
    before = module_state()
    refusal = (
        (401, anthropic_error("authentication_error"))
        if provider == "anthropic"
        else (401, compat_error())
    )
    for step in (None, refusal):
        llm = build(Wire(provider, step))
        try:
            await llm.complete(REQUEST)
        except LLMError as exc:
            provider_error(exc, provider)
            failure_word(exc)
        await llm.aclose()
    after = module_state()
    assert after == before
    assert fallback.claude_block() is None
    assert service._KEY_CACHE == {}  # noqa: SLF001
    for text in after.values():
        assert KEY not in text


# --- the transport boundary: redirects, echoing headers, the wire guard (final review) --


def in_parse_caches(needle: str) -> list[str]:
    """Strings containing `needle` in any cache reachable from ``urllib.parse``: the
    ``lru_cache`` of ``urlsplit`` and friends (3.11 and later) or ``_parse_cache`` (3.10)."""
    import gc
    import types
    import urllib.parse

    stack: list[Any] = [
        value
        for name, value in vars(urllib.parse).items()
        if hasattr(value, "cache_info")
        or (isinstance(value, (dict, list, set)) and not name.startswith("__"))
    ]
    seen: set[int] = set()
    hits: list[str] = []
    skipped = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, skipped):
            continue
        seen.add(id(item))
        if isinstance(item, (str, bytes)):
            if (needle if isinstance(item, str) else needle.encode()) in item:
                hits.append(repr(item))
            continue
        stack.extend(gc.get_referents(item))
    return hits


def held_anywhere(needle: str, mine: tuple[Any, ...]) -> list[str]:
    """Every string containing `needle` that a live container still refers to, other
    than the test's own (`mine`, by identity)."""
    import gc
    import types

    gc.collect()
    own = {id(item) for item in mine}
    hits: list[str] = []
    for holder in gc.get_objects():
        if isinstance(holder, (types.FrameType, types.CodeType)) or id(holder) in own:
            continue
        for item in gc.get_referents(holder):
            if id(item) in own or not isinstance(item, (str, bytes)):
                continue
            if (needle if isinstance(item, str) else needle.encode()) in item:
                hits.append(f"{type(holder).__name__}: {item!r}")
    return hits


async def fail_once(provider: str, key: str, respond: Any) -> tuple[str, int]:
    """One failing call on `key` against `respond`, the client closed and dropped: the
    error's text and how many requests went out. Nothing of the call is returned."""
    lib: Any = httpx2 if provider == "anthropic" else httpx
    sent = 0

    def handler(request: Any) -> Any:
        nonlocal sent
        sent += 1
        return respond(lib, request.headers["authorization"].split(" ", 1)[1])

    llm = own_llm(
        credential_for(provider, key),
        MODELS[provider],
        purpose="extract",
        effort="low",
        max_entities=30,
        call_cap=1,
        transport=lib.MockTransport(handler),
        backoff_base=0.0,
    )
    try:
        await llm.complete(REQUEST)
    except LLMError as exc:
        said = f"{type(exc).__name__}: {exc} ({failure_word(exc)})"
    else:
        said = "answered"
    await llm.aclose()
    return said, sent


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_redirect_that_echoes_the_key_leaves_it_in_no_cache(
    provider: str, status: int
) -> None:
    """Even with redirects not followed, the HTTP library works out where a redirect
    points, and a relative ``Location`` goes through ``urllib.parse.urljoin``, whose
    ``urlsplit`` cache outlives the client on a warm instance. A provider that echoes
    the key there must leave nothing behind."""
    import urllib.parse

    key = f"sk-REDIRECT-{provider}-{status}-0123456789abcdef"
    needle = key[3:]  # also found inside a longer string
    urllib.parse.clear_cache()

    def respond(lib: Any, echoed: str) -> Any:
        return lib.Response(status, headers={"Location": f"/v1/moved?echo={echoed}"})

    said, sent = await fail_once(provider, key, respond)
    assert sent == 1  # the provider's redirect is the failure; nothing followed it
    assert said == f"LLMFatalError: {PROVIDER_NAMES[provider]} request failed (error) (error)"
    assert in_parse_caches(needle) == []
    assert held_anywhere(needle, (key, needle)) == []


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_an_absolute_redirect_is_stripped_too(provider: str) -> None:
    key = f"sk-ABSOLUTE-{provider}-0123456789abcdef"

    def respond(lib: Any, echoed: str) -> Any:
        return lib.Response(
            302, headers={"Location": f"https://elsewhere.invalid/collect?k={echoed}"}, text=echoed
        )

    needle = key[3:]
    said, sent = await fail_once(provider, key, respond)
    assert (sent, failure_word_of(said)) == (1, "error")
    assert held_anywhere(needle, (key, needle)) == []


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_redirect_location_is_never_parsed(provider: str) -> None:
    """Not only a literal echo: the key reversed, encoded or cut in pieces would pass a
    search for it, so no ``Location`` reaches the HTTP library at all."""
    import urllib.parse

    key = f"sk-MIRRORED-{provider}-0123456789abcdef"
    mirrored = key[::-1]
    urllib.parse.clear_cache()

    def respond(lib: Any, echoed: str) -> Any:
        return lib.Response(307, headers={"Location": f"/v1/moved?echo={echoed[::-1]}"})

    said, sent = await fail_once(provider, key, respond)
    assert (sent, failure_word_of(said)) == (1, "error")
    assert in_parse_caches(mirrored) == []
    assert held_anywhere(mirrored, (mirrored,)) == []


def failure_word_of(said: str) -> str:
    return said.rsplit("(", 1)[1].rstrip(")")


@pytest.mark.parametrize("status", [200, 401])
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_key_echoed_in_a_response_header_reaches_no_process_cache(
    provider: str, status: int
) -> None:
    """A charset the response claims is looked up through ``codecs``, and the
    ``encodings`` package remembers every name it did not find, for the life of the
    process. A header that carries the key is dropped before the HTTP library reads it."""
    import encodings

    key = f"sk-CHARSET.{provider}.{status}.0123456789abcdef"

    def respond(lib: Any, echoed: str) -> Any:
        return lib.Response(
            status,
            headers={
                "Content-Type": f"text/plain; charset={echoed}",
                "Set-Cookie": f"session={echoed}",
                "X-Request-Id": echoed,
            },
            content=b"not json",
        )

    needle = key[3:]
    said, sent = await fail_once(provider, key, respond)
    assert sent == 1 and needle not in said
    marker = f"charset.{provider}.{status}"
    assert [name for name in encodings._cache if marker in name.lower()] == []  # noqa: SLF001
    assert held_anywhere(needle, (key, needle)) == []


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_an_answer_whose_header_echoes_the_key_is_still_read(provider: str) -> None:
    def echoing(request: Any) -> Any:
        lib: Any = httpx2 if provider == "anthropic" else httpx
        model = json.loads(request.content)["model"]
        body = message_body(model) if provider == "anthropic" else chat_body(model)
        return lib.Response(200, headers={"X-Echo": KEY, "X-Kept": "kept"}, json=body)

    llm = build(Wire(provider, echoing))
    response = await llm.complete(REQUEST)
    await llm.aclose()
    assert response.text == '{"entities": []}'


# What an SDK that applied the operator's environment in another layer than the one
# ``providers._claude`` empties would put on the wire: each under a name the SDK itself
# sends (so a check of names alone passes it), or beside them.
OPERATOR_ON_THE_WIRE = [
    {"Authorization": f"Bearer {SITE_ANTHROPIC_KEY}"},
    {"X-Api-Key": SITE_ANTHROPIC_KEY},
    {"User-Agent": "OPERATOR-agent"},
    {"anthropic-version": "1999-01-01"},
    {"X-Stainless-Lang": "OPERATOR-lang"},
    {"X-Stainless-Package-Version": "OPERATOR"},
    {"X-Stainless-OS": "OPERATOR-os"},
    {"x-stainless-retry-count": "OPERATOR"},
    {"X-Stainless-Helper": "OPERATOR-helper"},
    {"Accept": "OPERATOR/accept"},
    {"Accept-Encoding": "OPERATOR"},
    {"anthropic-beta": "server-side-fallback-2026-07-01"},
    {"X-Operator-Org": "OPERATOR-org"},
    {"traceparent": "00-OPERATOR-01"},
]


@pytest.mark.parametrize("extra", OPERATOR_ON_THE_WIRE, ids=lambda extra: next(iter(extra)))
async def test_an_operator_header_on_the_wire_stops_the_anthropic_request(
    extra: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check is of the request about to be sent, not of the client's defaults: a
    header an upgraded SDK adds after them is seen, and so is the operator's value
    under a name the SDK uses."""
    import anthropic

    real = anthropic.AsyncAnthropic._build_headers  # noqa: SLF001

    def late(self: Any, *args: Any, **kwargs: Any) -> Any:
        headers = real(self, *args, **kwargs)
        for name, value in extra.items():
            headers[name] = value
        return headers

    monkeypatch.setattr(anthropic.AsyncAnthropic, "_build_headers", late)
    wire = Wire("anthropic")
    llm = build(wire)
    with pytest.raises(providers.WireRefused) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    assert wire.requests == []  # nothing left the process
    assert str(caught.value) == "Anthropic request failed (error)"
    assert failure_word(caught.value) == "error"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert leaks(str(caught.value) + repr(vars(caught.value))) == []
    error = provider_error(caught.value, "anthropic")
    assert (error.status, error.code) == (502, "provider_error")
    assert "OPERATOR" not in strings(error) and SITE_ANTHROPIC_KEY not in strings(error)


async def test_a_second_authorization_header_stops_the_anthropic_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    real = anthropic.AsyncAnthropic._build_request  # noqa: SLF001

    def doubled(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = real(self, *args, **kwargs)
        raw = [*request.headers.raw, (b"Authorization", f"Bearer {SITE_ANTHROPIC_KEY}".encode())]
        request.headers = httpx2.Headers(raw)
        return request

    monkeypatch.setattr(anthropic.AsyncAnthropic, "_build_request", doubled)
    wire = Wire("anthropic")
    llm = build(wire)
    with pytest.raises(providers.WireRefused):
        await llm.complete(REQUEST)
    await llm.aclose()
    assert wire.requests == []


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
@pytest.mark.parametrize(
    "elsewhere",
    [
        "https://elsewhere.invalid/v1/",
        "http://{host}/v1/",
        "https://{host}:8443/v1/",
        "https://user:pw@{host}/v1/",
        "https://{host}.elsewhere.invalid/v1/",
    ],
)
async def test_a_request_to_another_address_is_not_sent(provider: str, elsewhere: str) -> None:
    """The base URLs are constants; should anything ever move a client off its host,
    the key does not follow."""
    wire = Wire(provider)
    llm = build(wire)
    inner = llm.inner._inner  # type: ignore[attr-defined]  # noqa: SLF001
    target = elsewhere.format(host=HOSTS[provider])
    if provider == "anthropic":
        inner._client.base_url = target  # noqa: SLF001
    else:
        inner._base_url = target.rstrip("/")  # noqa: SLF001
    with pytest.raises(providers.WireRefused) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    assert wire.requests == []
    assert str(caught.value) == f"{PROVIDER_NAMES[provider]} request failed (error)"


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_another_bearer_on_an_openai_compatible_request_is_not_sent(provider: str) -> None:
    wire = Wire(provider)
    llm = build(wire)
    llm.inner._inner._api_key = SITE_OPENROUTER_KEY  # type: ignore[attr-defined]  # noqa: SLF001
    with pytest.raises(providers.WireRefused):
        await llm.complete(REQUEST)
    await llm.aclose()
    assert wire.requests == []


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_the_wire_guard_shows_no_key(provider: str) -> None:
    llm = build(Wire(provider))
    inner = llm.inner._inner  # type: ignore[attr-defined]  # noqa: SLF001
    http = inner._client._client if provider == "anthropic" else inner._client  # noqa: SLF001
    guard = http._transport  # noqa: SLF001
    assert isinstance(guard, providers._Wire)  # noqa: SLF001
    assert KEY not in repr(guard) and KEY not in repr(vars(guard))
    await llm.aclose()


def test_the_anthropic_sdk_is_the_audited_version() -> None:
    """``providers._claude`` keeps the operator's environment out of a visitor's request
    by knowing exactly what SDK 2.10.0 reads and sends. A different version has not been
    looked at."""
    import anthropic

    assert anthropic.__version__ == providers.AUDITED_SDK == "2.10.0", (
        f"anthropic {anthropic.__version__} is installed but providers._claude was audited "
        f"against {providers.AUDITED_SDK}. Before changing the pin, re-audit for the new "
        "version: (1) every environment variable the SDK reads (_client.py, _tracing.py) "
        "and whether an explicit api_key, base_url and open_telemetry=False still keep it "
        "out; (2) where ANTHROPIC_CUSTOM_HEADERS is stored and applied, and that emptying "
        "it still works; (3) the headers a request carries on the wire and their values "
        "(providers._claude's expected set, ANTHROPIC_VERSION); (4) that the default HTTP "
        "client still takes a transport and reads no proxy with one. Then update "
        "AUDITED_SDK, both pins (ner-site/requirements.txt, pyproject.toml) and run the "
        "own-key Anthropic rollout check in ner-site/README.md."
    )


def test_the_anthropic_sdk_is_pinned_where_it_is_installed_from() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pin = f"anthropic=={providers.AUDITED_SDK}"
    requirements = (root / "ner-site" / "requirements.txt").read_text().splitlines()
    assert [line for line in requirements if line.startswith("anthropic")] == [pin]
    assert f'"{pin}"' in (root / "pyproject.toml").read_text()
    assert providers.ANTHROPIC_VERSION == "2026-10-21"


# --- second review round: what a response may bring back, per-call refusals, bugs -------


def percent(text: str) -> str:
    return "".join(f"%{ord(char):02X}" for char in text)


@pytest.mark.parametrize("status", [200, 401])
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_key_echoed_as_an_encoded_charset_reaches_no_process_cache(
    provider: str, status: int
) -> None:
    """RFC 2231: ``charset*=utf-8''%73%6B...`` is decoded by the HTTP library when it
    reads the content type, after any search of the raw header for the key."""
    import encodings

    key = f"sk-ENCODED.{provider}.{status}.0123456789abcdef"

    def respond(lib: Any, echoed: str) -> Any:
        return lib.Response(
            status,
            headers={"Content-Type": f"text/plain; charset*=utf-8''{percent(echoed)}"},
            content=b"not json",
        )

    needle = key[3:]
    said, sent = await fail_once(provider, key, respond)
    assert sent == 1 and needle not in said
    marker = f"encoded.{provider}.{status}"
    assert [name for name in encodings._cache if marker in name.lower()] == []  # noqa: SLF001
    assert held_anywhere(needle, (key, needle)) == []


@pytest.mark.parametrize("lib", [httpx, httpx2], ids=["httpx", "httpx2"])
@pytest.mark.parametrize("status", [200, 429, 500])
async def test_only_a_fixed_set_of_response_headers_reaches_the_http_library(
    lib: Any, status: int
) -> None:
    provider = "openai" if lib is httpx else "anthropic"
    credential = credential_for(provider)
    sent = {
        "Content-Type": "application/json; charset=latin-1; boundary=x",
        "Content-Length": "2",
        "Content-Encoding": "identity",
        "Retry-After": "7",
        "Retry-After-Ms": "7000",
        "Set-Cookie": "session=abc; Path=/",
        "Request-Id": "req_0123",
        "X-Request-Id": KEY,
        "Link": "</next>; rel=next",
        "Location": "/elsewhere",
        "X-Anything": "at all",
    }
    wire_class = providers._HttpxWire if lib is httpx else providers._Httpx2Wire  # noqa: SLF001
    wire = wire_class(
        lib.MockTransport(lambda request: lib.Response(status, headers=sent, content=b"{}")),
        credential,
    )
    async with lib.AsyncClient(transport=wire) as client:
        response = await client.post(f"https://{HOSTS[provider]}/v1/x", headers=bearer(), json={})
        assert dict(response.headers) == {
            "content-type": "application/json; charset=utf-8",
            "content-length": "2",
            "content-encoding": "identity",
            "retry-after": "7",
            "retry-after-ms": "7000",
        }
        assert len(client.cookies.jar) == 0
        assert response.json() == {}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("Retry-After", "Wed, 21 Oct 2026 07:28:00 GMT"),
        ("Retry-After", f"7; {KEY}"),
        ("Retry-After", "1e9"),
        ("Retry-After-Ms", "-1"),
        ("Content-Length", "2, 2"),
        ("Content-Type", f"text/html; charset={KEY}"),
    ],
)
async def test_a_response_header_outside_its_strict_form_is_dropped(name: str, value: str) -> None:
    wire = providers._HttpxWire(  # noqa: SLF001
        httpx.MockTransport(
            lambda request: httpx.Response(200, headers={name: value}, content=b"{}")
        ),
        credential_for("openai"),
    )
    async with httpx.AsyncClient(transport=wire) as client:
        response = await client.post("https://api.openai.com/v1/x", headers=bearer(), json={})
    kept = dict(response.headers)
    assert KEY not in repr(kept)
    if name == "Content-Type":
        assert kept["content-type"] == "text/plain; charset=utf-8"  # not JSON: said so, fixed
    else:
        assert name.lower() not in kept


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_cookie_from_the_provider_is_never_stored_or_sent_back(provider: str) -> None:
    """The HTTP client keeps cookies and sends them with the next request of the same
    client. For Anthropic the wire then saw a header the SDK does not send and refused
    every call after the first."""

    def with_cookie(request: Any) -> Any:
        lib: Any = httpx2 if provider == "anthropic" else httpx
        model = json.loads(request.content)["model"]
        body = message_body(model) if provider == "anthropic" else chat_body(model)
        return lib.Response(200, headers={"Set-Cookie": "__cf_bm=abc123; Path=/"}, json=body)

    wire = Wire(provider, with_cookie)
    llm = build(wire, call_cap=8)
    first = await llm.complete(REQUEST)
    rest = await asyncio.gather(*(llm.complete(REQUEST) for _ in range(4)))
    await llm.aclose()
    assert [r.text for r in (first, *rest)] == ['{"entities": []}'] * 5
    assert len(wire.requests) == 5
    assert all("cookie" not in request.headers for request in wire.requests)


async def test_a_refusal_belongs_to_the_call_that_was_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several picks share one client. One refused request must not turn the provider
    failures of the calls beside it, or after it, into refusals."""
    calls = 0
    real = providers._Wire._allowed  # noqa: SLF001

    def first_refused(self: Any, request: Any) -> bool:
        nonlocal calls
        calls += 1
        return calls != 1 and real(self, request)

    monkeypatch.setattr(providers._Wire, "_allowed", first_refused)  # noqa: SLF001
    wire = Wire("anthropic", (500, anthropic_error("api_error")), None)
    llm = build(wire, call_cap=8)
    results = await asyncio.gather(
        *(llm.complete(REQUEST) for _ in range(3)), return_exceptions=True
    )
    late = await asyncio.gather(llm.complete(REQUEST), return_exceptions=True)
    await llm.aclose()
    kinds = [type(result).__name__ for result in (*results, *late)]
    assert kinds == ["WireRefused", "LLMRetryableError", "ClaudeResponse", "ClaudeResponse"]
    assert len(wire.requests) == 3


async def test_a_content_length_that_is_not_the_bodys_stops_the_anthropic_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    real = anthropic.AsyncAnthropic._build_request  # noqa: SLF001

    def padded(self: Any, *args: Any, **kwargs: Any) -> Any:
        request = real(self, *args, **kwargs)
        request.headers["Content-Length"] = str(len(request.content) + 64)
        return request

    monkeypatch.setattr(anthropic.AsyncAnthropic, "_build_request", padded)
    wire = Wire("anthropic")
    llm = build(wire)
    with pytest.raises(providers.WireRefused):
        await llm.complete(REQUEST)
    await llm.aclose()
    assert wire.requests == []


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_fault_inside_an_adapter_is_not_dressed_up_as_the_providers(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only what a provider, the SDK or the HTTP library can do to a call is sealed. A
    ``KeyError`` in the adapter's own code is a bug of the site and leaves as it is, to
    be logged as one (``http.handle`` logs its type and frames, never its message)."""

    def broken(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise KeyError(KEY)

    if provider == "anthropic":
        monkeypatch.setattr(ClaudeClient, "_arguments", broken)
    else:
        monkeypatch.setattr(providers.OpenAICompatClient, "_headers", broken)
    wire = Wire(provider)
    llm = build(wire)
    with pytest.raises(KeyError):
        await llm.complete(REQUEST)
    await llm.aclose()
    assert wire.requests == []


# --- third review round: compressed answers, unreadable answers, the budget's memory ----


def answer_body(provider: str) -> bytes:
    model = MODELS[provider]
    body = message_body(model) if provider == "anthropic" else chat_body(model)
    return json.dumps(body).encode()


def raw_answer(provider: str, content: bytes, **headers: str) -> Any:
    """An answer as a real transport hands it over: an unread stream, still encoded."""
    lib: Any = httpx2 if provider == "anthropic" else httpx
    return lambda request: lib.Response(
        200,
        headers={"Content-Type": "application/json", **headers},
        stream=lib.ByteStream(content),
    )


@pytest.mark.parametrize(
    "coding", ["gzip", "GZip", "GZIP", " gzip ", "gzip, identity", "Identity,GZip"]
)
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_compressed_answer_is_read_whatever_the_case_of_its_coding(
    provider: str, coding: str
) -> None:
    """Coding names are case-insensitive (RFC 9110). A header kept only in one spelling
    left the body compressed, and the bytes went to the JSON parser as they were."""
    import gzip

    wire = Wire(
        provider,
        raw_answer(provider, gzip.compress(answer_body(provider)), **{"Content-Encoding": coding}),
    )
    llm = build(wire)
    response = await llm.complete(REQUEST)
    await llm.aclose()
    assert response.text == '{"entities": []}'


@pytest.mark.parametrize("coding", ["deflate", "Deflate"])
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_deflated_answer_is_read(provider: str, coding: str) -> None:
    import zlib

    wire = Wire(
        provider,
        raw_answer(provider, zlib.compress(answer_body(provider)), **{"Content-Encoding": coding}),
    )
    llm = build(wire)
    response = await llm.complete(REQUEST)
    await llm.aclose()
    assert response.text == '{"entities": []}'


@pytest.mark.parametrize(
    "coding", ["x-made-up", "gzip, x-made-up", "x-gzip", "compress", f"gzip, {KEY}", "gzip;q=1"]
)
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_an_answer_in_a_coding_that_cannot_be_decoded_is_a_fixed_failure(
    provider: str, coding: str
) -> None:
    """Not dropped in silence: the body would reach the parser still encoded."""
    import gzip

    wire = Wire(
        provider,
        raw_answer(provider, gzip.compress(answer_body(provider)), **{"Content-Encoding": coding}),
    )
    llm = build(wire, call_cap=1)
    with pytest.raises(LLMError) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    assert str(caught.value) == f"{PROVIDER_NAMES[provider]} request failed (bad_answer)"
    assert failure_word(caught.value) == "bad_answer"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert provider_error(caught.value, provider).status == 502
    assert len(wire.requests) == 1
    # The dispatch is settled as what it was, on every provider (re-review 3: the two
    # OpenAI-compatible ones left it open, so the budget said the provider was well).
    budget = llm.budget
    assert budget.dispatched == 1 and budget.provider_error is not None
    assert KEY not in budget.provider_error and "x-made-up" not in budget.provider_error
    if provider != "anthropic":
        assert budget.provider_error == "bad_answer"
    with pytest.raises(LLMError) as limit:
        await llm.complete(REQUEST)
    assert failure_word(limit.value) == "call_limit" and len(wire.requests) == 1


UNREADABLE = {
    "malformed": b'{"id": "msg_1", "content": [',
    "truncated": b'{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"te',
    "empty": b"",
    "not utf-8": b'\xff\xfe{"a":1}',
    "echoing": ('{"error": "' + KEY + '", ').encode(),
    # Longer than Python converts: the decoder raises a plain ValueError, not its own.
    "huge integer": b'{"id":"msg_1","n":' + b"9" * 4301 + b',"content":[{"type":"te',
}
WRONG_SHAPE = {
    "a list": b"[1, 2]",
    "a string": b'"hello"',
    "null": b"null",
    "content is a string": b'{"content": "x", "choices": "no"}',
    "content of numbers": b'{"content": [1, 2], "choices": [1]}',
}


@pytest.mark.parametrize("body", list(UNREADABLE), ids=list(UNREADABLE))
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_200_that_is_not_json_is_the_providers_bad_answer(provider: str, body: str) -> None:
    """Narrowing what is sealed (round 2) let the SDK's ``JSONDecodeError`` through for
    Anthropic, and an extraction answered 500. What a provider sends that cannot be
    decoded is the provider's failure; a bug in this code still is not."""
    wire = Wire(provider, raw_answer(provider, UNREADABLE[body]))
    llm = build(wire, call_cap=1)
    with pytest.raises(LLMError) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    assert str(caught.value) == f"{PROVIDER_NAMES[provider]} request failed (bad_answer)"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    error = provider_error(caught.value, provider)
    assert (error.status, error.code) == (502, "provider_error")
    assert leaks(strings(error) + str(caught.value)) == []


NOT_JSON_TYPED = {
    "html": (b"<html><body>busy</body></html>", {"Content-Type": "text/html"}),
    "no content type, malformed": (b'{"id": "msg_1", "content": [', {}),
    "no content type, empty": (b"", {}),
    "no content type, huge integer": (UNREADABLE["huge integer"], {}),
    "plain text echoing": (f"upstream says {KEY}".encode(), {"Content-Type": "text/plain"}),
}


@pytest.mark.parametrize("case", list(NOT_JSON_TYPED), ids=list(NOT_JSON_TYPED))
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_200_without_a_json_content_type_that_is_not_json_is_a_bad_answer(
    provider: str, case: str
) -> None:
    """Re-review 3: for Anthropic the SDK hands such a body back as a string, which
    passed for an empty answer (422). It is the provider's ``bad_answer`` like the
    rest, and the call is settled as that in the budget."""
    content, headers = NOT_JSON_TYPED[case]
    lib: Any = httpx2 if provider == "anthropic" else httpx
    wire = Wire(
        provider,
        lambda request: lib.Response(200, headers=headers, stream=lib.ByteStream(content)),
    )
    llm = build(wire, call_cap=1)
    with pytest.raises(LLMError) as caught:
        await llm.complete(REQUEST)
    await llm.aclose()
    assert str(caught.value) == f"{PROVIDER_NAMES[provider]} request failed (bad_answer)"
    assert failure_word(caught.value) == "bad_answer"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    error = provider_error(caught.value, provider)
    assert (error.status, error.code) == (502, "provider_error")
    assert leaks(strings(error) + str(caught.value) + repr(vars(llm.budget))) == []
    assert len(wire.requests) == 1


@pytest.mark.parametrize("body", list(WRONG_SHAPE), ids=list(WRONG_SHAPE))
@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_200_of_the_wrong_shape_is_never_an_unexpected_exception(
    provider: str, body: str
) -> None:
    """JSON, but not an answer: a fixed failure, or an empty answer the stage refuses
    (422), as before. Never something that leaves as a 500."""
    wire = Wire(provider, raw_answer(provider, WRONG_SHAPE[body]))
    llm = build(wire, call_cap=1)
    try:
        response = await llm.complete(REQUEST)
    except LLMError as exc:
        assert str(exc) == f"{PROVIDER_NAMES[provider]} request failed (bad_answer)"
    else:
        assert response.text == ""
    await llm.aclose()


@pytest.mark.parametrize("provider", OWN_PROVIDERS)
async def test_a_fault_in_an_adapter_leaves_no_message_in_the_call_budget(
    provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's budget remembers how its last call ended, as ``Class: message``, and
    quotes it later. A fault that is not sealed still must not leave its message there."""
    import gc

    def broken(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise KeyError(f"diagnostic with {KEY}")

    if provider == "anthropic":
        monkeypatch.setattr(ClaudeClient, "_arguments", broken)
    else:
        monkeypatch.setattr(providers.OpenAICompatClient, "_headers", broken)
    llm = build(Wire(provider), call_cap=2)
    for _ in range(2):
        with pytest.raises(KeyError):
            await llm.complete(REQUEST)
    budget = llm.budget
    seen: set[int] = set()
    stack: list[Any] = [budget]
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, type):
            continue
        seen.add(id(item))
        if isinstance(item, (str, bytes)):
            assert (KEY if isinstance(item, str) else KEY.encode()) not in item, item
        elif isinstance(item, (dict, list, tuple, set)) or item is budget:
            stack.extend(gc.get_referents(item))
    assert KEY not in repr(vars(budget)) and KEY not in budget.limit_message()
    assert "KeyError" in repr(vars(budget)) if provider == "anthropic" else True
    with pytest.raises(LLMError) as caught:  # the limit, in fixed words
        await llm.complete(REQUEST)
    assert KEY not in str(caught.value)
    await llm.aclose()
