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

One ``complete`` is one Messages request (SDK ``max_retries=0``): retrying and stepping
down to another model are the fallback chain's job (``ner_site.fallback``), so nothing is
retried out of sight of the call budget. Every error ``complete`` raises carries a
``reason`` string the chain reads; see ``_translate`` for the table. No message contains
the API key.
"""

from __future__ import annotations

from collections.abc import Mapping
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
CLAUDE_MODELS: tuple[str, ...] = (OPUS, SONNET)
DISPLAY_NAMES: dict[str, str] = {OPUS: "Claude Opus 5.5", SONNET: "Claude Sonnet 5.5"}

# Thinking tokens count against max_tokens, so these are far above the engine's own limits.
EXTRACT_MAX_TOKENS = 16000
SELECT_MAX_TOKENS = 8000
EFFORTS: tuple[str, ...] = ("low", "medium", "high")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_MESSAGE_CHARS = 500


def is_claude(model: str) -> bool:
    """Whether `model` is an Anthropic model id (anything else is an OpenRouter id)."""
    return model.startswith("claude-")


class ClaudeRefused(LLMFatalError):
    """Claude declined the request (``stop_reason: refusal``), after any server fallback."""

    reason = "refused"


def normalise_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """The form of `schema` that Claude's schema-enforced output accepts.

    Every object forbids additional properties and lists all of its properties as
    required. A property that was optional becomes required but nullable, so the model
    sends null where it would have left the key out; the stage parsers treat null and
    absent alike. The input is not modified.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, Mapping):
            return node
        out = {key: walk(value) for key, value in node.items()}
        properties = out.get("properties")
        if isinstance(properties, dict):
            required = node.get("required", ())
            for name, prop in properties.items():
                if name not in required and isinstance(prop, dict):
                    properties[name] = {"anyOf": [prop, {"type": "null"}]}
            out["required"] = list(properties)
            out["additionalProperties"] = False
        return out

    return dict(walk(schema))


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
    other 5xx, connection errors, timeouts,    LLMRetryableError transient
    an answer the SDK could not read
    other 4xx                                  LLMFatalError     bad_request
    ========================================== ================= =================
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
        if status >= 500:
            return _retryable("transient", message)
        return _fatal("bad_request", message)
    # APIConnectionError (APITimeoutError is one), or a 2xx body the SDK could not parse.
    return _retryable(
        "transient", _scrub(f"Anthropic {type(exc).__name__}: {exc.message}", api_key)
    )


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
        client: Any | None = None,
    ) -> None:
        if effort not in EFFORTS:
            raise ValueError(f"unknown effort {effort!r}; choose from {list(EFFORTS)}")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._effort = effort
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
            "fallbacks": "default",
            "betas": [FALLBACK_BETA],
        }
        if request.system:
            arguments["system"] = request.system
        return arguments

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one Messages request. Never retries; errors carry ``reason``."""
        try:
            message = await self._client.beta.messages.create(**self._arguments(request))
        except anthropic.APIError as exc:
            raise _translate(exc, self._api_key) from None

        usage = _usage(getattr(message, "usage", None))
        answered_by = str(getattr(message, "model", None) or self._model)
        # The stop reason is read before the content: a refusal may carry no content, or
        # a partial answer that must not be used.
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason == "refusal":
            category = getattr(getattr(message, "stop_details", None), "category", None)
            detail = f" (category: {category})" if category else ""
            raise ClaudeRefused(
                _scrub(f"{answered_by} declined the request{detail}", self._api_key), usage=usage
            )
        text = "".join(
            block.text
            for block in getattr(message, "content", None) or ()
            if getattr(block, "type", None) == "text" and isinstance(block.text, str)
        )
        return LLMResponse(
            text=text,
            usage=usage,
            model=answered_by,
            structured=request.schema is not None,
            # "length" is what the engine and extract.py recognise as a truncated answer.
            finish_reason="length" if stop_reason == "max_tokens" else stop_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


def key_status(
    api_key: str, *, client: Any | None = None, timeout: float = 5.0
) -> tuple[str, str | None]:
    """``("ok", None)`` or ``("unavailable", reason)``, for ``/api/health``.

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
        return "unavailable", f"Anthropic answered HTTP {exc.status_code}"
    except anthropic.APIError as exc:
        return "unavailable", f"Anthropic unreachable ({type(exc).__name__})"
    finally:
        if owned:
            sdk.close()
    return "ok", None


__all__ = [
    "CLAUDE_MODELS",
    "DISPLAY_NAMES",
    "EFFORTS",
    "EXTRACT_MAX_TOKENS",
    "OPUS",
    "SELECT_MAX_TOKENS",
    "SONNET",
    "ClaudeClient",
    "ClaudeRefused",
    "is_claude",
    "key_status",
    "normalise_schema",
]
