"""Claude through the Anthropic API, as an ``xwalk.llm.base.LLMClient``.

Binding: docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md section
3.1. The site's other models go through OpenRouter's Chat Completions; Claude is called
with the operator's own Anthropic key and the official ``anthropic`` SDK, never through an
OpenAI-compatible endpoint and never with hand-built HTTP. Extraction and selection keep
talking to an ``LLMClient`` and learn nothing about providers.

What the adapter sends, and why it differs from what the engine asks for:

- No ``temperature``, ``top_p``, ``top_k`` or seed. Opus 5.5 answers 400 to sampling
  parameters, so ``LLMRequest.temperature``, ``.seed`` and ``.extra`` are ignored.
- No ``thinking`` parameter. Thinking is adaptive by default and cannot be disabled on
  Opus 5.5; ``output_config.effort`` is the only depth control.
- Its own ``max_tokens``. Thinking counts against the limit, so the engine's 2048 / 600
  (``LLMRequest.max_tokens``) would cut answers short; the caller picks
  ``EXTRACT_MAX_TOKENS`` or ``SELECT_MAX_TOKENS`` at construction.
- Schema-enforced output (``output_config.format``) when the request has a schema, sent as
  a normalised copy (``normalise_schema``). The caller still validates the answer.
- The server-side refusal fallback (``fallbacks="default"``): Opus 5.5 has a ``bio``
  classifier and the site is fed biomedical abstracts. It is a beta parameter, so every
  request goes through ``client.beta.messages.create``. The installed SDK (2.10.0) types
  ``fallbacks`` and ``output_config.format`` on that same call, and its fallback entries
  document how a fallback attempt inherits the request's ``format``, so both are sent
  together. This could not be tried against the live API from the offline tests.
  ``refusal_fallback=False`` sends neither the parameter nor the beta flag: a request on
  a visitor's own key (``ner_site.providers``) must not be answered by a substitute
  model billed to them at another price.

What comes back (docs/superpowers/specs/2026-11-18-own-key-daily-cap-design.md section 8):
``complete`` returns a :class:`ClaudeResponse`, an ``LLMResponse`` that also says whether
a server-side fallback model served the answer (``served_by_fallback``, read from the
SDK's own signal, the ``fallback_message`` entry in ``usage.iterations``). Text written
before the last ``fallback`` content block belongs to a model that declined and is left
out of the answer.

One ``complete`` is one Messages request (SDK ``max_retries=0``): retrying and stepping
down to another model are the fallback chain's job (``ner_site.fallback``), so nothing is
retried out of sight of the call budget. Every error ``complete`` raises carries a
``reason`` string the chain reads; see ``_translate`` for the table. No message contains
the API key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import anthropic
from xwalk.fingerprint import hash_value
from xwalk.llm.base import (
    LLMCapabilities,
    LLMError,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
    reported_usage,
)
from xwalk.records import Usage

ADAPTER_VERSION = 1

OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
# Known to the adapter and named in notices, but not one of the site's own models: it
# is offered only to a visitor running on their own Anthropic key (ner_site.providers).
HAIKU = "claude-haiku-5-5"
CLAUDE_MODELS: tuple[str, ...] = (OPUS, SONNET)
DISPLAY_NAMES: dict[str, str] = {
    OPUS: "Claude Opus 5.5",
    SONNET: "Claude Sonnet 5.5",
    HAIKU: "Claude Haiku 5.5",
}

# Thinking tokens count against max_tokens, so these are far above the engine's own limits.
EXTRACT_MAX_TOKENS = 16000
SELECT_MAX_TOKENS = 8000
EFFORTS: tuple[str, ...] = ("low", "medium", "high")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# What the SDK raises for a 200 whose body it cannot decode: the decoders' own errors,
# not an ``anthropic`` one. Caught around the SDK call only (``ClaudeClient._send``).
UNREADABLE: tuple[type[Exception], ...] = (json.JSONDecodeError, UnicodeDecodeError)
MAX_MESSAGE_CHARS = 500


def is_claude(model: str) -> bool:
    """Whether `model` is an Anthropic model id (anything else is an OpenRouter id)."""
    return model.startswith("claude-")


class ClaudeRefused(LLMFatalError):
    """Claude declined the request (``stop_reason: refusal``), after any server fallback."""

    reason = "refused"


@dataclass(frozen=True)
class ClaudeResponse(LLMResponse):
    """What :meth:`ClaudeClient.complete` returns: an ``LLMResponse`` that also carries
    the SDK's signal that a server-side fallback model served the answer.

    ``served_by_fallback`` is True exactly when ``usage.iterations`` has a
    ``fallback_message`` entry; ``model`` is then that entry's model. Comparing ``model``
    with the model asked for cannot tell: a fallback may land on a model the caller also
    uses, and the API may echo the requested id in another spelling. The flag travels
    with the answer rather than on the client, so answers to concurrent calls on one
    client cannot be mixed up. A wrapper that rebuilds the response must keep it
    (``dataclasses.replace``); a plain ``LLMResponse`` has no such attribute, so read it
    with ``getattr(response, "served_by_fallback", False)``."""

    served_by_fallback: bool = False


# What schema-enforced output takes, read from the SDK's own helper
# (anthropic.transform_schema, anthropic/lib/_parse/_transform.py in 2.10.0). The helper
# itself cannot be used here: it leaves optional properties optional and has no branch
# for a list-valued ``type``, which the engine's selection schema has.
_STRING_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)
_NULL: Mapping[str, Any] = {"type": "null"}


def _is_literal(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _allows_null(node: Mapping[str, Any]) -> bool:
    if node.get("type") == "null" or node.get("const", "") is None:
        return True
    enum = node.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    variants = node.get("anyOf")
    return isinstance(variants, list) and any(
        isinstance(variant, Mapping) and _allows_null(variant) for variant in variants
    )


def _nullable(node: dict[str, Any]) -> dict[str, Any]:
    if _allows_null(node):
        return node
    return {"anyOf": [node, dict(_NULL)]}


def _schema_node(schema: Any) -> Any:
    """One schema node in the supported subset. Only called on schemas, never on the
    ``properties`` map itself nor on ``enum`` / ``const`` values, which are data."""
    if not isinstance(schema, Mapping):
        return schema  # a boolean schema, or something the API will name in its 400
    rest = dict(schema)
    out: dict[str, Any] = {}

    defs = rest.pop("$defs", None)
    if isinstance(defs, Mapping):
        out["$defs"] = {name: _schema_node(sub) for name, sub in defs.items()}
    ref = rest.pop("$ref", None)
    if ref is not None:
        out["$ref"] = ref
        return out

    kind = rest.pop("type", None)
    description = rest.pop("description", None)
    title = rest.pop("title", None)

    def finish(node: dict[str, Any], leftover: Mapping[str, Any]) -> dict[str, Any]:
        """Keywords the API does not enforce go into the description, where the model
        may still follow them; the caller validates the answer either way."""
        notes = [description] if isinstance(description, str) else []
        if leftover:
            notes.append("{" + ", ".join(f"{k}: {v}" for k, v in leftover.items()) + "}")
        if notes:
            node["description"] = "\n\n".join(notes)
        if title is not None:
            node["title"] = title
        return node

    if isinstance(kind, list):
        # ["string", "null"] is written as anyOf of single types; the constraints go
        # with the variants they can apply to, the description stays on the outside.
        variants = [
            dict(_NULL) if single == "null" else _schema_node({**rest, "type": single})
            for single in kind
        ]
        if len(variants) == 1:
            return finish(variants[0], {})
        out["anyOf"] = variants
        return finish(out, {})

    # Literal values are data and are copied, never walked. A ``type`` beside them is
    # kept (the SDK helper drops it; the site's extraction schema has one and is accepted
    # as it stands by the strict providers it already runs on).
    if "enum" in rest and isinstance(rest["enum"], list) and all(map(_is_literal, rest["enum"])):
        out["enum"] = list(rest.pop("enum"))
    if "const" in rest and _is_literal(rest["const"]):
        out["const"] = rest.pop("const")

    any_of = rest.pop("anyOf", None)
    one_of = rest.pop("oneOf", None)
    all_of = rest.pop("allOf", None)
    if isinstance(any_of, list):
        out["anyOf"] = [_schema_node(variant) for variant in any_of]
    elif isinstance(one_of, list):
        out["anyOf"] = [_schema_node(variant) for variant in one_of]
    elif isinstance(all_of, list):
        out["allOf"] = [_schema_node(variant) for variant in all_of]
    elif kind is not None:
        out["type"] = kind

    if kind == "object":
        properties = rest.pop("properties", None)
        required = rest.pop("required", None)
        rest.pop("additionalProperties", None)
        named = required if isinstance(required, list) else []
        out["properties"] = {
            # A property that was optional becomes required but nullable.
            name: _schema_node(prop) if name in named else _optional(prop)
            for name, prop in (properties if isinstance(properties, Mapping) else {}).items()
        }
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    elif kind == "string":
        if rest.get("format") in _STRING_FORMATS:
            out["format"] = rest.pop("format")
    elif kind == "array":
        if "items" in rest:
            out["items"] = _schema_node(rest.pop("items"))
        if rest.get("minItems") in (0, 1) and not isinstance(rest["minItems"], bool):
            out["minItems"] = rest.pop("minItems")
    return finish(out, rest)


def _optional(prop: Any) -> Any:
    node = _schema_node(prop)
    return _nullable(node) if isinstance(node, dict) else node


def normalise_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The form of `schema` that Claude's schema-enforced output accepts.

    - Every object forbids additional properties and lists all of its properties as
      required. A property that was optional becomes required but nullable, so the model
      sends null where it would have left the key out; the stage parsers treat null and
      absent alike.
    - A list-valued ``type`` becomes ``anyOf`` of single types, and ``oneOf`` becomes
      ``anyOf``.
    - Keywords outside the supported subset (numeric bounds, string lengths and patterns,
      array sizes beyond ``minItems`` 0 or 1, unknown string formats, anything else) are
      moved into ``description`` instead of being sent, as the SDK's helper does. A 400
      for one of them would be a ``bad_request``, which the fallback chain does not step
      down on, so every call with that schema would fail.

    The input is not modified.
    """
    return dict(_schema_node(schema))


def _scrub(message: str, api_key: str) -> str:
    """`message` without the key (should an upstream ever echo it), bounded in length."""
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return message[:MAX_MESSAGE_CHARS]


def _retryable(reason: str, message: str, *, retry_after: float | None = None) -> LLMError:
    error = LLMRetryableError(message, retry_after=retry_after, usage=Usage.unreported())
    error.reason = reason  # type: ignore[attr-defined]
    return error


def _fatal(reason: str, message: str) -> LLMError:
    error = LLMFatalError(message, usage=Usage.unreported())
    error.reason = reason  # type: ignore[attr-defined]
    return error


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    try:
        seconds = float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _status_message(exc: anthropic.APIStatusError, api_key: str) -> str:
    detail = exc.api_message or "no message"
    kind = f" {exc.type}" if exc.type else ""
    return _scrub(f"Anthropic HTTP {exc.status_code}{kind}: {detail}", api_key)


def _translate(exc: anthropic.APIError, api_key: str) -> LLMError:
    """An SDK error as the engine's class, with the ``reason`` the chain reads.

    Most specific first (spec 3.1). Each one is a dispatched request whose tokens are
    unknown (``Usage.unreported()``), never a free call.

    ========================================== ================= =================
    SDK error                                  raised            reason
    ========================================== ================= =================
    ``RateLimitError`` (429)                   LLMRetryableError rate_limited
    529 / ``overloaded_error``                 LLMRetryableError overloaded
    402 / ``billing_error``, 400 naming the    LLMFatalError     balance_exhausted
    credit balance
    ``AuthenticationError``,                   LLMFatalError     key_refused
    ``PermissionDeniedError``
    ``APITimeoutError`` (no answer within the  LLMRetryableError timeout
    client's timeout)
    other 5xx, 408, 409, other connection      LLMRetryableError transient
    errors, an answer the SDK could not read
    (a 200 that cannot be decoded at all is
    raised as the same, in ``_send``)
    other 4xx                                  LLMFatalError     bad_request
    ========================================== ================= =================

    A timeout is kept apart from ``transient`` because the chain retries a transient
    failure on the same model: waiting out a hung model twice would use up the request.
    """
    if isinstance(exc, anthropic.APIStatusError):
        message = _status_message(exc, api_key)
        status = exc.status_code
        if isinstance(exc, anthropic.RateLimitError):
            return _retryable("rate_limited", message, retry_after=_retry_after(exc))
        if status == 529 or exc.type == "overloaded_error":
            return _retryable("overloaded", message, retry_after=_retry_after(exc))
        if (
            status == 402
            or exc.type == "billing_error"
            # The API has no stable code for an empty prepaid balance on a 400; its
            # message is the only signal.
            or (status == 400 and "credit balance" in (exc.api_message or "").lower())
        ):
            return _fatal("balance_exhausted", message)
        if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return _fatal("key_refused", message)
        # 408 and 409 are transient in the engine's contract (xwalk.llm.base).
        if status >= 500 or status in (408, 409):
            return _retryable("transient", message)
        return _fatal("bad_request", message)
    message = _scrub(f"Anthropic {type(exc).__name__}: {exc.message}", api_key)
    # APITimeoutError is an APIConnectionError: the more specific class first.
    if isinstance(exc, anthropic.APITimeoutError):
        return _retryable("timeout", message)
    # Another APIConnectionError, or a 2xx body the SDK could not parse.
    return _retryable("transient", message)


def _usage(raw: Any) -> Usage:
    """One answered request. Prompt tokens are the uncached input plus both cache counts
    (the API reports them apart); a missing input or output count makes it unknown."""
    prompt = getattr(raw, "input_tokens", None)
    if isinstance(prompt, int) and not isinstance(prompt, bool):
        for name in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            extra = getattr(raw, name, None)
            if isinstance(extra, int) and not isinstance(extra, bool) and extra > 0:
                prompt += extra
    return reported_usage(prompt, getattr(raw, "output_tokens", None))


def _served_by(message: Any) -> tuple[str | None, bool]:
    """The model that answered, as the response reports it, and whether it is a
    server-side fallback model. The SDK's signal for the latter is a ``fallback_message``
    entry in ``usage.iterations`` (2.10.0 ``BetaFallbackMessageIterationUsage``: the
    terminal entry of a fallback-served turn), whose model is then the one that answered;
    otherwise it is the response's own ``model``."""
    model = getattr(message, "model", None)
    answered = str(model) if model else None
    fallback = False
    for entry in getattr(getattr(message, "usage", None), "iterations", None) or ():
        if getattr(entry, "type", None) == "fallback_message":
            fallback = True
            served = getattr(entry, "model", None)
            if served:
                answered = str(served)
    return answered, fallback


def _answer_text(message: Any) -> str:
    """The text of the model whose answer this is. A ``fallback`` content block
    (2.10.0 ``BetaFallbackBlock``) marks the point where a model that declined gives way
    to the next one, so only the text blocks after the last of them count; with no such
    block that is every text block."""
    parts: list[str] = []
    for block in getattr(message, "content", None) or ():
        kind = getattr(block, "type", None)
        if kind == "fallback":
            parts.clear()
        elif kind == "text" and isinstance(block.text, str):
            parts.append(block.text)
    return "".join(parts)


class ClaudeClient:
    """One Claude model behind the ``LLMClient`` protocol (module docstring).

    ``client`` is an injected SDK-shaped object for tests; without it the adapter builds
    an ``anthropic.AsyncAnthropic`` that never retries on its own.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        max_tokens: int,
        effort: str = "low",
        timeout: float = 60.0,
        refusal_fallback: bool = True,
        client: Any | None = None,
    ) -> None:
        if effort not in EFFORTS:
            raise ValueError(f"unknown effort {effort!r}; choose from {list(EFFORTS)}")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._effort = effort
        self._refusal_fallback = refusal_fallback
        self._client = (
            client
            if client is not None
            else anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout)
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            json_schema=True, strict_schema=True, usage_reporting=True, seed=False
        )

    @property
    def generation_defaults(self) -> Mapping[str, Any]:
        """What every request is sent with: no temperature, and the adapter's own limit."""
        return {"temperature": None, "max_tokens": self._max_tokens}

    @property
    def fingerprint(self) -> str:
        """Digest of what shapes the request. The key is absent: rotating it changes no
        answer, and a secret must never reach a manifest."""
        return hash_value(
            {
                "adapter": "anthropic",
                "adapter_version": ADAPTER_VERSION,
                "provider": "anthropic",
                "model": self._model,
                "effort": self._effort,
                "max_tokens": self._max_tokens,
                # Only when it is off, so every fingerprint from before the switch
                # existed is unchanged.
                **({} if self._refusal_fallback else {"refusal_fallback": False}),
            }
        )

    def _arguments(self, request: LLMRequest) -> dict[str, Any]:
        output_config: dict[str, Any] = {"effort": self._effort}
        if request.schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": normalise_schema(request.schema),
            }
        arguments: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }
        if self._refusal_fallback:
            arguments["fallbacks"] = "default"
            arguments["betas"] = [FALLBACK_BETA]
        if request.system:
            arguments["system"] = request.system
        return arguments

    async def _send(self, arguments: dict[str, Any]) -> Any:
        """The SDK call alone, apart from building its arguments and reading its
        answer, so that what the SDK raised can be told from what this adapter did.

        A 200 whose body cannot be decoded makes the SDK raise the decoder's own error
        (``UNREADABLE``), which is no ``anthropic`` error. Here, and only here, such an
        error is the provider's answer being unreadable (``_unreadable``); the same
        class raised while the request is built or the answer is read is a bug of this
        code and is left alone."""
        try:
            return await self._client.beta.messages.create(**arguments)
        except UNREADABLE:
            failure = self._unreadable()
        # Raised out here: the decoder's error quotes the body it could not read.
        raise failure

    def _unreadable(self) -> Exception:
        """What an undecodable 200 is raised as. On the site's key a ``transient``
        failure, like a 5xx: the chain tries the model once more and then steps down.
        Fixed words; nothing of the body. (``providers._OwnClaude`` says it its way.)"""
        return _retryable("transient", "Anthropic answered 200 with a body that is not JSON")

    async def complete(self, request: LLMRequest) -> ClaudeResponse:
        """Send one Messages request. Never retries; errors carry ``reason``."""
        try:
            message = await self._send(self._arguments(request))
        except anthropic.APIError as exc:
            raise _translate(exc, self._api_key) from None

        usage = _usage(getattr(message, "usage", None))
        # Not always the model asked for: ``fallbacks`` lets the API answer from another.
        served, fallback = _served_by(message)
        answered_by = served or self._model
        # The stop reason is read before the content: a refusal may carry no content, or
        # a partial answer that must not be used.
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            category = getattr(getattr(message, "stop_details", None), "category", None)
            detail = f" (category: {category})" if category else ""
            raise ClaudeRefused(
                _scrub(f"{answered_by} declined the request{detail}", self._api_key), usage=usage
            )
        return ClaudeResponse(
            text=_answer_text(message),
            usage=usage,
            model=answered_by,
            structured=request.schema is not None,
            # "length" is what the engine and extract.py recognise as a truncated answer.
            finish_reason="length" if stop_reason == "max_tokens" else stop_reason,
            served_by_fallback=fallback,
        )

    async def aclose(self) -> None:
        await self._client.close()


def key_status(
    api_key: str, *, client: Any | None = None, timeout: float = 5.0
) -> tuple[str, str | None]:
    """Whether Anthropic accepts the key, for ``/api/health``: ``("ok", None)``,
    ``("unavailable", reason)`` for a key that is missing or was refused (401, 403), or
    ``("ok", "key not checked (...)")`` when the check could not be completed, because
    Anthropic was unreachable or answered with any other error. A failed probe says
    nothing about the key, so it is never reported as an outage.

    One synchronous Models API call (free: no tokens are generated). The reason is a
    fixed phrase or names a status or an exception class; it never quotes the upstream
    message, so it cannot carry the key. ``client`` is an SDK-shaped object for tests.
    """
    if not api_key:
        return "unavailable", "key not set"
    owned = client is None
    sdk: Any = (
        anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=timeout) if owned else client
    )
    try:
        sdk.models.retrieve(OPUS, timeout=timeout)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
        return "unavailable", "key refused"
    except anthropic.APIStatusError as exc:
        return "ok", f"key not checked (Anthropic answered HTTP {exc.status_code})"
    except anthropic.APIError as exc:
        return "ok", f"key not checked (Anthropic unreachable: {type(exc).__name__})"
    finally:
        if owned:
            sdk.close()
    return "ok", None


__all__ = [
    "CLAUDE_MODELS",
    "DISPLAY_NAMES",
    "EFFORTS",
    "EXTRACT_MAX_TOKENS",
    "HAIKU",
    "OPUS",
    "SELECT_MAX_TOKENS",
    "SONNET",
    "ClaudeClient",
    "ClaudeRefused",
    "ClaudeResponse",
    "is_claude",
    "key_status",
    "normalise_schema",
]
