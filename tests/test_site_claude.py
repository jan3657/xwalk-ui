"""The Anthropic adapter (spec 2026-11-18, section 3.1): offline, no key, no network.

Two kinds of double. The request and response cases use a fake with the attribute path the
adapter calls (``beta.messages.create``), which records keyword arguments. The error table
and one wire case use the real SDK client over an ``httpx2.MockTransport``, so the SDK
itself builds the exceptions and the HTTP request; nothing leaves the process.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from ner_site import claude
from ner_site.claude import ClaudeClient, ClaudeRefused, ClaudeResponse, key_status
from ner_site.extract import DEFAULT_PROMPT, build_request, entity_schema
from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMFatalError,
    LLMRequest,
    LLMRetryableError,
)
from xwalk.prompts.contract import SELECT_SCHEMA
from xwalk.records import Usage

KEY = "sk-ant-SENTINEL-do-not-leak-0123456789"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}


def message(
    *blocks: Any,
    stop_reason: str = "end_turn",
    model: str = claude.OPUS,
    usage: Any = None,
    stop_details: Any = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
        usage=usage
        or SimpleNamespace(
            input_tokens=100,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            output_tokens=20,
        ),
    )


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


class FakeSDK:
    """Records ``beta.messages.create`` keyword arguments; answers with a script."""

    def __init__(self, *script: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._script = list(script) or [message(text("{}"))]
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(step, BaseException):
            raise step
        return step

    async def close(self) -> None:
        self.closed = True


def make(sdk: Any, *, model: str = claude.OPUS, max_tokens: int = 16000, **kw: Any) -> ClaudeClient:
    return ClaudeClient(model, api_key=KEY, max_tokens=max_tokens, client=sdk, **kw)


def real_sdk(handler: Any) -> anthropic.AsyncAnthropic:
    """The real async client, answered by `handler` instead of the network."""
    return anthropic.AsyncAnthropic(
        api_key=KEY,
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def wire_message(model: str, **extra: Any) -> dict[str, Any]:
    """A Messages response body as the API sends it."""
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


def error_body(type_: str, text_: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": type_, "message": text_}}


# --- constants -------------------------------------------------------------------------


def test_constants_and_is_claude() -> None:
    assert claude.OPUS == "claude-opus-5-5"
    assert claude.SONNET == "claude-sonnet-5-5"
    assert claude.CLAUDE_MODELS == (claude.OPUS, claude.SONNET)
    # Haiku is an id the adapter knows and names, for a visitor's own key; it is not one
    # of the site's own models.
    assert claude.HAIKU == "claude-haiku-5-5"
    assert claude.HAIKU not in claude.CLAUDE_MODELS
    assert claude.DISPLAY_NAMES == {
        "claude-opus-5-5": "Claude Opus 5.5",
        "claude-sonnet-5-5": "Claude Sonnet 5.5",
        "claude-haiku-5-5": "Claude Haiku 5.5",
    }
    assert claude.EXTRACT_MAX_TOKENS == 16000
    assert claude.SELECT_MAX_TOKENS == 8000
    assert claude.EFFORTS == ("low", "medium", "high")
    assert claude.is_claude("claude-opus-5-5")
    assert claude.is_claude("claude-something-else")
    assert not claude.is_claude("anthropic/claude-haiku-5.5")
    assert not claude.is_claude("openai/gpt-6-luna")


def test_client_contract() -> None:
    client = make(FakeSDK(), max_tokens=8000)
    assert isinstance(client, LLMClient)
    assert client.model == claude.OPUS
    assert client.capabilities == LLMCapabilities(
        json_schema=True, strict_schema=True, usage_reporting=True, seed=False
    )
    assert client.generation_defaults == {"temperature": None, "max_tokens": 8000}


def test_fingerprint_follows_what_is_sent_and_never_the_key() -> None:
    base = make(FakeSDK()).fingerprint
    assert base == make(FakeSDK()).fingerprint
    assert (
        base
        == ClaudeClient(
            claude.OPUS, api_key="another-key", max_tokens=16000, client=FakeSDK()
        ).fingerprint
    )
    assert base != make(FakeSDK(), model=claude.SONNET).fingerprint
    assert base != make(FakeSDK(), max_tokens=8000).fingerprint
    assert base != make(FakeSDK(), effort="high").fingerprint
    # Asking for no server-side fallback is another request; asking for it is the default.
    assert base == make(FakeSDK(), refusal_fallback=True).fingerprint
    assert base != make(FakeSDK(), refusal_fallback=False).fingerprint


def test_unknown_effort_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="effort"):
        make(FakeSDK(), effort="max")


# --- the request -----------------------------------------------------------------------


async def test_request_carries_the_adapters_settings_and_no_sampling_parameters() -> None:
    sdk = FakeSDK()
    await make(sdk, effort="medium").complete(
        LLMRequest(
            system="You extract entities.",
            user="Aspirin reduced fever.",
            temperature=0.0,
            seed=1,
            max_tokens=600,
            extra={"temperature": 0.2, "top_p": 0.9},
        )
    )
    (sent,) = sdk.calls
    assert sent["model"] == claude.OPUS
    assert sent["system"] == "You extract entities."
    assert sent["messages"] == [{"role": "user", "content": "Aspirin reduced fever."}]
    assert sent["max_tokens"] == 16000
    assert sent["output_config"] == {"effort": "medium"}
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "seed", "extra_body"):
        assert forbidden not in sent
    assert "budget_tokens" not in json.dumps(sent)


async def test_effort_defaults_to_low() -> None:
    sdk = FakeSDK()
    await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert sdk.calls[0]["output_config"]["effort"] == "low"


async def test_schema_request_carries_a_json_schema_format() -> None:
    sdk = FakeSDK()
    response = await make(sdk).complete(LLMRequest(system="s", user="u", schema=SCHEMA))
    assert sdk.calls[0]["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert response.structured is True


async def test_request_without_schema_carries_no_format() -> None:
    sdk = FakeSDK()
    response = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert "format" not in sdk.calls[0]["output_config"]
    assert response.structured is False


async def test_loose_schema_is_sent_normalised_and_the_callers_is_untouched() -> None:
    loose: dict[str, Any] = {
        "type": "object",
        "properties": {
            "choice": {"type": "string"},
            "confidence": {"type": "number"},
            "detail": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
            },
        },
        "required": ["choice"],
    }
    before = copy.deepcopy(loose)
    sdk = FakeSDK()
    await make(sdk).complete(LLMRequest(system="s", user="u", schema=loose))

    assert loose == before
    sent = sdk.calls[0]["output_config"]["format"]["schema"]
    assert sent["additionalProperties"] is False
    assert sent["required"] == ["choice", "confidence", "detail"]
    assert sent["properties"]["choice"] == {"type": "string"}
    # An optional property becomes required but nullable.
    assert sent["properties"]["confidence"] == {"anyOf": [{"type": "number"}, {"type": "null"}]}
    nested = sent["properties"]["detail"]["anyOf"][0]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["note"]


async def test_the_engines_selection_schema_is_sent_in_the_supported_subset() -> None:
    """SELECT_SCHEMA has a list-valued ``type`` and numeric bounds, which the API's
    schema-enforced output does not take; a 400 here would fail every pick."""
    before = copy.deepcopy(dict(SELECT_SCHEMA))
    sdk = FakeSDK()
    await make(sdk).complete(LLMRequest(system="s", user="u", schema=SELECT_SCHEMA))

    assert dict(SELECT_SCHEMA) == before
    assert sdk.calls[0]["output_config"]["format"]["schema"] == {
        "type": "object",
        "properties": {
            "chosen_key": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "confidence_score": {"type": "number", "description": "{minimum: 0, maximum: 1}"},
            "explanation": {"type": "string"},
        },
        "required": ["chosen_key", "confidence_score", "explanation"],
        "additionalProperties": False,
    }


async def test_the_sites_extraction_schema_is_sent_unchanged() -> None:
    """What ``extract.build_request`` sends is already in the supported subset."""
    request = build_request("Aspirin reduced fever.", DEFAULT_PROMPT)
    assert request.schema is not None
    sdk = FakeSDK()
    await make(sdk).complete(request)
    assert sdk.calls[0]["output_config"]["format"]["schema"] == request.schema
    assert request.schema == entity_schema()


def test_unsupported_keywords_move_into_the_description() -> None:
    sent = claude.normalise_schema(
        {
            "type": "object",
            "description": "An answer.",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 9, "pattern": "^a"},
                "when": {"type": "string", "format": "date"},
                "code": {"type": "string", "format": "ontology-id"},
                "step": {"type": "integer", "multipleOf": 5, "description": "A step."},
                "few": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "many": {
                    "type": "array",
                    "items": {"type": "integer", "maximum": 3},
                    "minItems": 2,
                    "maxItems": 4,
                },
            },
            "required": ["name", "when", "code", "step", "few", "many"],
            "additionalProperties": False,
        }
    )
    props = sent["properties"]
    assert props["name"] == {
        "type": "string",
        "description": "{minLength: 1, maxLength: 9, pattern: ^a}",
    }
    assert props["when"] == {"type": "string", "format": "date"}
    assert props["code"] == {"type": "string", "description": "{format: ontology-id}"}
    assert props["step"] == {"type": "integer", "description": "A step.\n\n{multipleOf: 5}"}
    assert props["few"] == {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert props["many"] == {
        "type": "array",
        "items": {"type": "integer", "description": "{maximum: 3}"},
        "description": "{minItems: 2, maxItems: 4}",
    }
    assert sent["description"] == "An answer."


def test_only_schema_nodes_are_treated_as_schemas() -> None:
    """A property named like a keyword is a schema; enum and const values are data."""
    literal = {"properties": {"x": 1}, "type": "object"}
    sent = claude.normalise_schema(
        {
            "type": "object",
            "properties": {
                "properties": {"type": "string"},
                "required": {"type": "integer"},
                "minimum": {"type": "number"},
                "kind": {"enum": ["a", "b", None]},
                "fixed": {"const": "x"},
                "odd": {"enum": [literal]},
            },
            "required": ["properties", "required", "minimum", "kind", "fixed", "odd"],
        }
    )
    assert sent["required"] == ["properties", "required", "minimum", "kind", "fixed", "odd"]
    assert sent["additionalProperties"] is False
    assert sent["properties"]["properties"] == {"type": "string"}
    assert sent["properties"]["required"] == {"type": "integer"}
    assert sent["properties"]["minimum"] == {"type": "number"}
    assert sent["properties"]["kind"] == {"enum": ["a", "b", None]}
    assert sent["properties"]["fixed"] == {"const": "x"}
    # An enum of objects is outside the supported subset, so it becomes a note; the
    # object in it is data and was not rewritten as if it were a schema.
    assert sent["properties"]["odd"] == {"description": f"{{enum: [{literal}]}}"}
    assert "additionalProperties" not in sent["properties"]["odd"]["description"]


def test_list_types_and_combinators_are_rewritten() -> None:
    sent = claude.normalise_schema(
        {
            "type": "object",
            "properties": {
                "size": {"type": ["integer", "null"], "minimum": 0, "description": "A size."},
                "only": {"type": ["string"]},
                "either": {"oneOf": [{"type": "string", "minLength": 2}, {"type": "null"}]},
                "maybe": {"type": ["string", "null"]},
                "ref": {"$ref": "#/$defs/thing"},
            },
            "required": ["size", "only", "either", "ref"],
            "$defs": {"thing": {"type": "object", "properties": {"a": {"type": "string"}}}},
        }
    )
    props = sent["properties"]
    assert props["size"] == {
        "anyOf": [{"type": "integer", "description": "{minimum: 0}"}, {"type": "null"}],
        "description": "A size.",
    }
    assert props["only"] == {"type": "string"}
    assert props["either"] == {
        "anyOf": [{"type": "string", "description": "{minLength: 2}"}, {"type": "null"}]
    }
    # Optional and already nullable: required, and not wrapped a second time.
    assert props["maybe"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert props["ref"] == {"$ref": "#/$defs/thing"}
    assert sent["required"] == ["size", "only", "either", "maybe", "ref"]
    assert sent["$defs"]["thing"] == {
        "type": "object",
        "properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["a"],
        "additionalProperties": False,
    }


async def test_refusal_fallback_is_requested() -> None:
    sdk = FakeSDK()
    await make(sdk).complete(LLMRequest(system="s", user="u", schema=SCHEMA))
    await make(sdk).complete(LLMRequest(system="s", user="u"))
    for sent in sdk.calls:
        assert sent["fallbacks"] == "default"
        assert sent["betas"] == ["server-side-fallback-2026-07-01"]


async def test_without_refusal_fallback_neither_the_parameter_nor_the_beta_is_sent() -> None:
    """A visitor's own key is never asked for a substitute model (spec 2026-11-18 own
    keys, section 5): no ``fallbacks``, and no beta flag, in the arguments or on the wire."""
    sdk = FakeSDK()
    await make(sdk, refusal_fallback=False).complete(
        LLMRequest(system="s", user="u", schema=SCHEMA)
    )
    (sent,) = sdk.calls
    assert "fallbacks" not in sent and "betas" not in sent
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}

    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=wire_message(claude.HAIKU))

    client = make(real_sdk(handler), model=claude.HAIKU, refusal_fallback=False)
    response = await client.complete(LLMRequest(system="s", user="u", schema=SCHEMA))
    await client.aclose()
    (request,) = seen
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert "anthropic-beta" not in request.headers
    assert "fallbacks" not in body and "betas" not in body
    assert body["model"] == claude.HAIKU
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert response.model == claude.HAIKU
    assert isinstance(response, ClaudeResponse) and response.served_by_fallback is False


async def test_the_real_sdk_accepts_and_sends_the_request() -> None:
    """The keyword arguments are real SDK parameters: the body and the beta header the SDK
    builds from them carry structured output and the refusal fallback together."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": claude.OPUS,
                "content": [{"type": "text", "text": '{"entities": []}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    client = make(real_sdk(handler))
    response = await client.complete(
        LLMRequest(system="s", user="u", schema=SCHEMA, temperature=0.0, seed=1)
    )
    await client.aclose()

    (request,) = seen
    body = json.loads(request.content)
    assert request.url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert body["output_config"]["effort"] == "low"
    assert body["max_tokens"] == 16000
    assert not {"temperature", "top_p", "top_k", "seed", "thinking", "betas"} & set(body)
    assert response.text == '{"entities": []}'
    assert response.usage == Usage(prompt_tokens=7, completion_tokens=3, calls=1)


# --- the response ----------------------------------------------------------------------


async def test_text_blocks_are_joined_thinking_ignored_and_usage_counts_cache_tokens() -> None:
    sdk = FakeSDK(
        message(
            SimpleNamespace(type="thinking", thinking="hidden reasoning", signature="x"),
            text('{"entities": '),
            SimpleNamespace(type="thinking", thinking="more", signature="y"),
            text("[]}"),
            model=claude.SONNET,
            usage=SimpleNamespace(
                input_tokens=100,
                cache_creation_input_tokens=30,
                cache_read_input_tokens=5,
                output_tokens=42,
            ),
        )
    )
    response = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert response.text == '{"entities": []}'
    assert response.usage == Usage(prompt_tokens=135, completion_tokens=42, calls=1)
    assert response.model == claude.SONNET
    assert response.finish_reason == "end_turn"


def fallback_usage(served: str) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=100,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        output_tokens=20,
        iterations=[
            SimpleNamespace(type="message", model=claude.OPUS),
            SimpleNamespace(type="fallback_message", model=served),
        ],
    )


async def test_text_before_the_last_fallback_block_is_dropped() -> None:
    """A ``fallback`` content block marks where a model that declined gives way to the
    next one. What the declined model had written is not part of the answer: joined to
    the serving model's text it would corrupt the JSON (spec own keys, section 8)."""
    sdk = FakeSDK(
        message(
            text('{"entities": [{"text": "Asp'),
            SimpleNamespace(type="fallback"),
            text("I would rather"),
            SimpleNamespace(type="fallback"),
            SimpleNamespace(type="thinking", thinking="hidden", signature="x"),
            text('{"entities": '),
            text("[]}"),
            usage=fallback_usage("claude-opus-4-8"),
        )
    )
    response = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert response.text == '{"entities": []}'
    assert isinstance(response, ClaudeResponse) and response.served_by_fallback is True
    assert response.model == "claude-opus-4-8"


async def test_a_fallback_block_with_nothing_after_it_leaves_no_text() -> None:
    sdk = FakeSDK(message(text("partial"), SimpleNamespace(type="fallback")))
    assert (await make(sdk).complete(LLMRequest(system="s", user="u"))).text == ""


async def test_a_fallback_served_answer_names_the_serving_model() -> None:
    """The SDK's signal that a server-side fallback model served the response is a
    ``fallback_message`` entry in ``usage.iterations``; its model is who answered, also
    should the top-level ``model`` still echo the requested id."""
    served = "claude-opus-4-9"
    usage = SimpleNamespace(
        input_tokens=100,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        output_tokens=20,
        iterations=[
            SimpleNamespace(type="message", model=claude.OPUS),
            SimpleNamespace(type="fallback_message", model=served),
        ],
    )
    sdk = FakeSDK(message(text("{}"), model=claude.OPUS, usage=usage))
    response = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert response.model == served
    # The flag travels with the answer, so concurrent picks cannot mix theirs up.
    assert isinstance(response, ClaudeResponse) and response.served_by_fallback is True
    # A fallback onto a model the caller also knows is flagged all the same.
    sdk = FakeSDK(message(text("{}"), model=claude.OPUS, usage=fallback_usage(claude.SONNET)))
    again = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert (again.model, again.served_by_fallback) == (claude.SONNET, True)  # type: ignore[attr-defined]
    # Without such an entry the answer's own ``model`` stands.
    plain = SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        iterations=[SimpleNamespace(type="message", model="claude-other")],
    )
    sdk = FakeSDK(message(text("{}"), model=claude.OPUS, usage=plain))
    unflagged = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert unflagged.model == claude.OPUS
    # A differently spelt echo of the requested id is not a fallback.
    sdk = FakeSDK(message(text("{}"), model="claude-opus-5-5-20260901", usage=plain))
    echoed = await make(sdk).complete(LLMRequest(system="s", user="u"))
    for answer in (unflagged, echoed):
        assert isinstance(answer, ClaudeResponse) and answer.served_by_fallback is False


async def test_the_real_sdk_reports_a_fallback_served_answer() -> None:
    """The names are the SDK's own (2.10.0): a ``fallback`` content block and a
    ``fallback_message`` entry in ``usage.iterations``, parsed by the real client."""
    served = "claude-opus-4-8"
    hop = {"from": {"model": claude.OPUS}, "to": {"model": served}}
    body = wire_message(
        claude.OPUS,
        content=[
            {"type": "text", "text": '{"entities": [{"te'},
            {"type": "fallback", **hop, "trigger": {"type": "refusal", "category": "bio"}},
            {"type": "text", "text": '{"entities": []}'},
        ],
        usage={
            "input_tokens": 7,
            "output_tokens": 3,
            "iterations": [
                {"type": "message", "model": claude.OPUS, "input_tokens": 7, "output_tokens": 1},
                {
                    "type": "fallback_message",
                    "model": served,
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ],
        },
    )
    client = make(real_sdk(lambda request: httpx2.Response(200, json=body)))
    response = await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert response.text == '{"entities": []}'
    assert response.model == served
    assert isinstance(response, ClaudeResponse) and response.served_by_fallback is True


async def test_missing_usage_is_unknown_not_zero() -> None:
    sdk = FakeSDK(message(text("ok"), usage=SimpleNamespace(input_tokens=None, output_tokens=3)))
    response = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert response.text == "ok"
    assert response.usage == Usage.unreported()


async def test_refusal_raises_claude_refused_even_with_empty_content() -> None:
    sdk = FakeSDK(
        message(
            stop_reason="refusal",
            stop_details=SimpleNamespace(type="refusal", category="bio", explanation="declined"),
        )
    )
    with pytest.raises(ClaudeRefused) as caught:
        await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert isinstance(caught.value, LLMFatalError)
    assert caught.value.reason == "refused"
    assert "bio" in str(caught.value)
    # The refused request was still dispatched and may have been billed.
    assert caught.value.usage == Usage(prompt_tokens=100, completion_tokens=20, calls=1)


async def test_refusal_is_checked_before_content() -> None:
    sdk = FakeSDK(message(text("partial answer"), stop_reason="refusal"))
    with pytest.raises(ClaudeRefused):
        await make(sdk).complete(LLMRequest(system="s", user="u"))


async def test_max_tokens_returns_the_text_with_finish_reason_length() -> None:
    sdk = FakeSDK(message(text('{"entities": ['), stop_reason="max_tokens"))
    response = await make(sdk).complete(LLMRequest(system="s", user="u"))
    assert response.text == '{"entities": ['
    assert response.finish_reason == "length"


# --- errors ----------------------------------------------------------------------------

BALANCE = "Your credit balance is too low to access the Anthropic API."

ERROR_TABLE = [
    (429, "rate_limit_error", "slow down", LLMRetryableError, "rate_limited"),
    (529, "overloaded_error", "Overloaded", LLMRetryableError, "overloaded"),
    (402, "billing_error", "payment required", LLMFatalError, "balance_exhausted"),
    (400, "invalid_request_error", BALANCE, LLMFatalError, "balance_exhausted"),
    (401, "authentication_error", "invalid x-api-key", LLMFatalError, "key_refused"),
    (403, "permission_error", "not allowed", LLMFatalError, "key_refused"),
    (500, "api_error", "internal", LLMRetryableError, "transient"),
    (503, "api_error", "unavailable", LLMRetryableError, "transient"),
    (504, "timeout_error", "gateway timeout", LLMRetryableError, "transient"),
    # The engine's contract (xwalk.llm.base.LLMRetryableError) names 408 and 409 retryable.
    (408, "timeout_error", "request timeout", LLMRetryableError, "transient"),
    (409, "invalid_request_error", "conflict", LLMRetryableError, "transient"),
    (400, "invalid_request_error", "max_tokens: too large", LLMFatalError, "bad_request"),
    (404, "not_found_error", "model: nope", LLMFatalError, "bad_request"),
    (413, "request_too_large", "too large", LLMFatalError, "bad_request"),
]


@pytest.mark.parametrize(("status", "type_", "text_", "kind", "reason"), ERROR_TABLE)
async def test_error_table(
    status: int, type_: str, text_: str, kind: type[Exception], reason: str
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=error_body(type_, text_))

    client = make(real_sdk(handler))
    with pytest.raises(kind) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()

    assert type(caught.value) is kind
    assert caught.value.reason == reason  # type: ignore[attr-defined]
    assert str(status) in str(caught.value)
    assert KEY not in str(caught.value)
    # One request was dispatched; its tokens are unknown, never free.
    assert caught.value.usage == Usage.unreported()  # type: ignore[attr-defined]


async def test_rate_limit_keeps_retry_after() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429, headers={"retry-after": "7"}, json=error_body("rate_limit_error", "slow down")
        )

    client = make(real_sdk(handler))
    with pytest.raises(LLMRetryableError) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert caught.value.retry_after == 7.0


async def test_connection_failure_is_transient() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client = make(real_sdk(handler))
    with pytest.raises(LLMRetryableError) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert caught.value.reason == "transient"  # type: ignore[attr-defined]
    assert KEY not in str(caught.value)


async def test_timeout_has_its_own_reason() -> None:
    """A timeout is not ``transient``: the chain must not wait out the same model again."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout(f"timed out with {KEY}", request=request)

    client = make(real_sdk(handler))
    with pytest.raises(LLMRetryableError) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert caught.value.reason == "timeout"  # type: ignore[attr-defined]
    assert "APITimeoutError" in str(caught.value) and KEY not in str(caught.value)
    assert caught.value.usage == Usage.unreported()


async def test_one_complete_is_one_request_with_no_hidden_retry() -> None:
    hits = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal hits
        hits += 1
        return httpx2.Response(529, json=error_body("overloaded_error", "Overloaded"))

    # The adapter builds its own SDK client here; only the transport is replaced.
    client = ClaudeClient(claude.OPUS, api_key=KEY, max_tokens=16000)
    client._client = client._client.with_options(  # noqa: SLF001
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    )
    with pytest.raises(LLMRetryableError):
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert hits == 1


async def test_a_message_that_echoes_the_key_is_scrubbed() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401, json=error_body("authentication_error", f"invalid x-api-key: {KEY}")
        )

    client = make(real_sdk(handler))
    with pytest.raises(LLMFatalError) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert caught.value.reason == "key_refused"  # type: ignore[attr-defined]
    assert KEY not in str(caught.value)
    assert KEY not in repr(caught.value)


async def test_aclose_closes_the_sdk_client() -> None:
    sdk = FakeSDK()
    await make(sdk).aclose()
    assert sdk.closed is True


# --- key_status ------------------------------------------------------------------------


def sync_sdk(handler: Any) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=KEY,
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def test_key_status_ok_with_one_models_call() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": claude.OPUS,
                "type": "model",
                "display_name": "Claude Opus 5.5",
                "created_at": "2026-09-01T00:00:00Z",
            },
        )

    assert key_status(KEY, client=sync_sdk(handler)) == ("ok", None)
    (request,) = seen
    assert request.method == "GET"
    assert request.url.path == f"/v1/models/{claude.OPUS}"


@pytest.mark.parametrize(
    ("status", "type_"), [(401, "authentication_error"), (403, "permission_error")]
)
def test_key_status_refused_key(status: int, type_: str) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=error_body(type_, f"invalid x-api-key {KEY}"))

    state, reason = key_status(KEY, client=sync_sdk(handler))
    assert state == "unavailable"
    assert reason == "key refused"


def test_key_status_connection_failure() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"cannot connect with {KEY}", request=request)

    # The check could not be completed, which says nothing about the key: not an outage.
    state, reason = key_status(KEY, client=sync_sdk(handler))
    assert state == "ok"
    assert reason is not None and "key not checked" in reason and "unreachable" in reason
    assert KEY not in reason


@pytest.mark.parametrize("status", [500, 503, 529, 429, 404])
def test_key_status_other_http_error_names_the_status_without_the_key(status: int) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=error_body("api_error", f"boom {KEY}"))

    state, reason = key_status(KEY, client=sync_sdk(handler))
    assert state == "ok"
    assert reason == f"key not checked (Anthropic answered HTTP {status})"


def test_key_status_without_a_key_makes_no_call() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no request expected")

    assert key_status("", client=sync_sdk(handler)) == ("unavailable", "key not set")


# --- a 200 that is not a message (re-review 3, item 6) ---------------------------------

# An integer longer than Python's limit for converting digits: the decoder then raises a
# plain ValueError, not a JSONDecodeError, before it reaches the end of the body.
HUGE_INTEGER = b'{"id":"msg_1","n":' + b"9" * 4301 + b',"content":[{"type":"te'
NOT_A_MESSAGE = {
    "html": (b"<html><body>busy</body></html>", {"Content-Type": "text/html"}),
    "html as json": (b"<html><body>busy</body></html>", {"Content-Type": "application/json"}),
    "no content type, malformed": (b'{"id": "msg_1", "content": [', {}),
    "no content type, empty": (b"", {}),
    "huge integer": (HUGE_INTEGER, {"Content-Type": "application/json"}),
    "huge integer, no content type": (HUGE_INTEGER, {}),
    "echoing the key": (f"upstream says {KEY}".encode(), {"Content-Type": "text/plain"}),
}


@pytest.mark.parametrize("case", list(NOT_A_MESSAGE), ids=list(NOT_A_MESSAGE))
async def test_a_200_that_is_not_a_message_is_a_transient_failure(case: str) -> None:
    """Without a JSON content type the SDK hands back the body as a string instead of
    a message, which used to pass for an empty answer (a 422 after one call, with no
    retry and no step-down); a body the decoder gives up on for another reason than
    its syntax used to be a 500."""
    content, headers = NOT_A_MESSAGE[case]
    hits = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal hits
        hits += 1
        return httpx2.Response(200, headers=headers, content=content)

    client = make(real_sdk(handler))
    with pytest.raises(LLMRetryableError) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert caught.value.reason == "transient"  # type: ignore[attr-defined]
    assert str(caught.value) == "Anthropic answered 200 with a body that is not JSON"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert caught.value.usage == Usage.unreported()
    assert hits == 1


async def test_json_without_a_json_content_type_is_still_an_answer() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.dumps(wire_message(claude.OPUS)).encode()
        return httpx2.Response(200, headers={"Content-Type": "text/plain"}, content=body)

    client = make(real_sdk(handler))
    response = await client.complete(LLMRequest(system="s", user="u"))
    await client.aclose()
    assert response.text == '{"entities": []}' and response.model == claude.OPUS


def _encoding_error() -> ValueError:
    """What ``json.dumps`` raises for a value it will not write: a ``ValueError`` out
    of the ``json`` package, but from its encoder."""
    try:
        json.dumps({"n": float("nan")}, allow_nan=False)
    except ValueError as exc:
        assert exc.__traceback__ is not None
        return exc
    raise AssertionError("json.dumps accepted NaN")


@pytest.mark.parametrize(
    "error",
    [
        _encoding_error(),
        ValueError("Streaming is required for operations that may take longer than 10 minutes"),
        ValueError("Exceeds the limit (4300 digits) for integer string conversion"),
        TypeError("create() got an unexpected keyword argument"),
        KeyError("model"),
    ],
)
async def test_an_error_that_is_not_the_decoders_still_propagates(error: Exception) -> None:
    """Only what was raised while the body was being decoded is the provider's answer
    being unreadable. The same class from anywhere else is a fault of this code or of
    how the SDK was called, whatever its message says, and must stay one."""
    client = make(FakeSDK(error))
    with pytest.raises(type(error)) as caught:
        await client.complete(LLMRequest(system="s", user="u"))
    assert caught.value is error
