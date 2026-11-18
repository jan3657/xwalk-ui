"""A visitor's own API key: the credential, the client built with it, and what it may say.

Binding: docs/superpowers/specs/2026-11-18-own-key-daily-cap-design.md section 5. A
visitor may run the demo on their own Anthropic, OpenAI or OpenRouter key instead of the
operator's. The key arrives in an ``Authorization: Bearer`` header, is used for that one
request and is gone with it. This module is the only place that touches it:

- :func:`read_credential` turns the request's ``provider`` field and headers into a
  :class:`Credential`, or refuses the request (400 ``bad_credential``).
- :func:`own_llm` builds the request-scoped, budgeted client for one stage.
- :func:`provider_error` and :func:`failure_word` say what went wrong in fixed words.

The rules everything here follows:

- **Nothing from upstream is repeated.** A provider's error text can echo the key, whole
  or masked ("Incorrect API key provided: sk-ab...yz"). Every error that leaves a client
  built here has been replaced by a fixed sentence and a short machine word
  (``_sealed_error``); the original, its traceback and its cause are dropped, because
  their frames hold the request headers. The messages of :func:`provider_error` and
  :func:`read_credential` are fixed too, and never quote the ``provider`` field either:
  it is the visitor's input, and a key pasted into the wrong field must not come back.
- **Nothing is kept.** No module-level state, no cache, no probe; the key is not in
  ``Settings``, a fingerprint, a digest or a log field. :class:`Credential` hides it from
  ``repr``.
- **The visitor chooses nothing but the provider.** Base URLs are the constants below;
  ``NER_SITE_BASE_URL`` and ``ANTHROPIC_BASE_URL`` do not apply. Redirects are not
  followed and the environment's proxies, ``.netrc`` and credentials are not read
  (``_claude``, ``_compat``), so the key goes to the provider's host and nowhere else.
  Every client sends through a transport of this module (``_Wire``) that looks at each
  request as it leaves and each response as it arrives: a request that is not for the
  pinned host with the visitor's key is not sent, and of a response only the status,
  the body and a handful of headers in a strict form reach the HTTP library, so that
  nothing a provider echoes in a header (a redirect's target, a charset, a cookie) is
  parsed into a cache or a cookie jar that outlives the call.
- **The operator's keys are not used.** Each client is given the visitor's key
  explicitly; there is no fallback chain, no server-side refusal fallback, no Claude
  block and no key check. Nothing the operator's environment holds for the site's own
  Anthropic client reaches a visitor's request: not a credential, a profile or a base
  URL, and not a header either (``ANTHROPIC_CUSTOM_HEADERS`` can name ``Authorization``
  and would otherwise replace the visitor's key with the operator's; see ``_claude``).

``service.py`` imports this module, so what is needed from there (the site's OpenRouter
client class, its attribution headers, timeouts and retry counts) is imported inside
:func:`own_llm`, when a request is already running, never at import time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import re
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import anthropic
import httpx
import httpx2
from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMError,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
    ParseError,
    client_generation_defaults,
)
from xwalk.llm.budget import BudgetedLLM, CallBudget, CallLimitExceeded
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.records import Usage

from ner_site.claude import EXTRACT_MAX_TOKENS, SELECT_MAX_TOKENS, ClaudeClient, ClaudeRefused
from ner_site.extract import MAX_TOKENS, TEMPERATURE
from ner_site.schema import ServiceError

SITE = "site"
OWN_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "openrouter")
PROVIDER_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
}
# Constants: nothing a visitor or the environment sends changes where a key is sent.
BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
# The SDK adds the path itself; passed explicitly so ANTHROPIC_BASE_URL is not read.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# Where each provider's requests may go, and nowhere else (``_Wire``).
HOSTS: dict[str, str] = {
    "anthropic": "api.anthropic.com",
    "openai": "api.openai.com",
    "openrouter": "openrouter.ai",
}
# The version of the ``anthropic`` SDK that ``_claude`` was written against: what it
# reads from the environment and what it puts on the wire were read in that version's
# source. It is pinned in ner-site/requirements.txt and pyproject.toml, and a test fails
# on any other version, naming what to look at again before the pin moves.
AUDITED_SDK = "2.10.0"
# The ``anthropic-version`` header of that SDK (its ``ANTHROPIC_API_VERSION``, which it
# does not export).
ANTHROPIC_VERSION = "2026-10-21"
KEY_MIN, KEY_MAX = 20, 512

EXTRACT, SELECT = "extract", "select"
# What service.make_select_llm gives the site's own OpenRouter selector (written there
# as literals): the engine's selection limit, and one retry.
OPENROUTER_SELECT_MAX_TOKENS = 600
SELECT_MAX_RETRIES = 1

_SCHEME = "bearer "
# Printable ASCII without the space: 0x21 to 0x7E.
_KEY = re.compile(r"[\x21-\x7e]+")

_NEEDS_HEADER = (
    "a request with your own API key needs exactly one Authorization header of the form "
    "'Bearer <key>'"
)
_BAD_KEY = f"the API key must be {KEY_MIN} to {KEY_MAX} printable ASCII characters without spaces"
_NOT_ENABLED = "own API keys are not available for that provider on this site"
_SITE_WITH_KEY = (
    "this request carries an API key but asks for the site's models; choose your "
    "provider, or send no Authorization header"
)
_KEY_IN_TEXT = (
    "your API key also appears in the text or the prompt; remove it from there before "
    "running (and consider revoking that key)"
)


class _Token:
    """The key as a callable that shows nothing: what ``Credential.token`` holds."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def __call__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "<token>"

    def __reduce__(self) -> str:
        # Not picklable: a credential is never written anywhere.
        raise TypeError("a credential token cannot be pickled")

    def __deepcopy__(self, memo: dict[int, Any]) -> _Token:
        return self  # so dataclasses.asdict shows "<token>" instead of failing


@dataclass(frozen=True)
class Credential:
    """Whose key a request runs on. ``token()`` returns the secret when a client is
    built for the request; ``repr`` and ``str`` never show it. A later provider (a
    ChatGPT sign-in) can return a refreshed access token from the same call."""

    provider: str
    token: Callable[[], str] = field(repr=False)
    source: str = "visitor"


@dataclass(frozen=True)
class ProviderInfo:
    """One own provider as ``/api/health`` lists it (spec section 6). The model lists
    are the deployment's, so the caller fills this in from its settings."""

    id: str
    name: str
    models: tuple[str, ...]
    default_model: str


def _bad(message: str) -> ServiceError:
    return ServiceError("bad_credential", message, status=400)


def _authorization(headers: Mapping[str, str | Sequence[str]]) -> list[str]:
    """Every ``Authorization`` value, whatever the case of the name and whether a
    repeated header arrives as several items or as one item with a sequence."""
    found: list[str] = []
    for name, value in headers.items():
        if not isinstance(name, str) or name.lower() != "authorization":
            continue
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            found.extend(item if isinstance(item, str) else "" for item in value)
        else:
            found.append("")
    return found


def read_credential(
    provider: str | None,
    headers: Mapping[str, str | Sequence[str]],
    *,
    enabled: Sequence[str],
    text: str,
    prompt: str | None,
) -> Credential | None:
    """The credential a request runs on: None for the site's keys, else the visitor's.

    ``provider`` is the request's field (absent and ``"site"`` both mean the site's
    keys), ``headers`` the request's headers with a repeated header as a sequence of its
    values (an ``email.message.Message`` works as it is: its ``items()`` repeats the
    name), ``enabled`` the own providers this deployment allows, ``text`` and ``prompt``
    what the request will send to the model.

    Raises 400 ``bad_credential`` with a fixed message, which never contains the key or
    the ``provider`` field, when:

    - a site request carries an ``Authorization`` header at all (the caller does not ask
      on a deployment without own keys: there a site request's header is none of the
      site's business, ``http._dispatch``);
    - the provider is not an own provider, or not enabled here;
    - the header is missing, repeated, or not ``Bearer <key>``;
    - the key is not ``KEY_MIN`` to ``KEY_MAX`` printable ASCII characters without spaces;
    - the key also appears in ``text`` or ``prompt``: it would be sent to the model and
      echoed back in the response as part of an entity or of the text itself.
    """
    values = _authorization(headers)
    if provider is None or provider == SITE:
        if values:
            raise _bad(_SITE_WITH_KEY)
        return None
    if not isinstance(provider, str) or provider not in OWN_PROVIDERS or provider not in enabled:
        raise _bad(_NOT_ENABLED)
    if len(values) != 1:
        raise _bad(_NEEDS_HEADER)
    value = values[0].strip(" \t")
    if value[: len(_SCHEME)].lower() != _SCHEME:
        raise _bad(_NEEDS_HEADER)
    # Everything after the scheme and its one space: a second space, a control or a
    # non-ASCII character makes it a bad key, not a longer or a shorter one.
    key = value[len(_SCHEME) :]
    if not KEY_MIN <= len(key) <= KEY_MAX or _KEY.fullmatch(key) is None:
        raise _bad(_BAD_KEY)
    if key in text or (prompt is not None and key in prompt):
        raise _bad(_KEY_IN_TEXT)
    return Credential(provider=provider, token=_Token(key))


# --- what a failure is called -----------------------------------------------------------

# The short machine words (spec section 5), plus three the spec's list has no word for:
# an answer that could not be read, the request's own call limit, and anything else.
KEY_REFUSED, RATE_LIMITED = "key_refused", "rate_limited"
WORDS = frozenset(
    {
        KEY_REFUSED,
        RATE_LIMITED,
        "timeout",
        "overloaded",
        "http_5xx",
        "bad_request",
        "refused",
        "unreachable",
        "bad_answer",
        "call_limit",
        "error",
    }
)
# The ``reason`` an error of ner_site.claude carries -> the word. ``transient`` is not
# here: it covers a 5xx, a dropped connection and an unreadable answer (``_from_text``).
_CLAUDE_REASONS = {
    "rate_limited": RATE_LIMITED,
    "overloaded": "overloaded",
    "balance_exhausted": KEY_REFUSED,
    "key_refused": KEY_REFUSED,
    "timeout": "timeout",
    "refused": "refused",
    "bad_request": "bad_request",
}
# The fixed openings of the messages the two adapters write. Only what the adapter
# itself put there is read (a status, an exception class name); the upstream text after
# it is looked at for one phrase (``_status_word``) and never copied.
_ANTHROPIC_STATUS = re.compile(r"Anthropic HTTP (\d{3})\b")
_ANTHROPIC_CLASS = re.compile(r"Anthropic ([A-Za-z_]\w*): ")
_EXHAUSTED = re.compile(r"exhausted \d+ retries: ")
_BODY_ERROR = re.compile(r"HTTP 200 with error (\d+|None): ")
_STATUS = re.compile(r"HTTP (\d{3}): ")
_CLASS = re.compile(r"([A-Za-z_]\w*): ")


def _status_word(status: int, detail: str = "") -> str:
    # 404 with them: a key that may not use the model is told the model does not exist
    # (OpenAI: ``model_not_found``), and the sentence for a refused key says "or not
    # allowed to use this model".
    if status in (401, 402, 403, 404):
        return KEY_REFUSED
    if status == 429:
        # OpenAI answers 429 both for a rate limit and for an account with no credit
        # left ("You exceeded your current quota"); waiting does not help the second.
        return KEY_REFUSED if "exceeded your current quota" in detail.lower() else RATE_LIMITED
    if status == 408:
        return "timeout"
    if status == 529:
        return "overloaded"
    if status >= 500:
        return "http_5xx"
    if 400 <= status < 500:
        return "bad_request"
    return "error"


def _from_text(text: str) -> str:
    """The word for a message written by ``ner_site.claude`` or by the engine's
    OpenAI-compatible client (also the outcome strings it reports to a call budget)."""
    if text.startswith("Anthropic "):
        status = _ANTHROPIC_STATUS.match(text)
        if status is not None:
            return _status_word(int(status.group(1)), text)
        named = _ANTHROPIC_CLASS.match(text)
        if named is None:
            return "error"
        if named.group(1) == "APITimeoutError":
            return "timeout"
        # Another connection error, else an answer the SDK could not read.
        return "unreachable" if named.group(1) == "APIConnectionError" else "bad_answer"
    exhausted = _EXHAUSTED.match(text)
    rest = text[exhausted.end() :] if exhausted is not None else text
    in_body = _BODY_ERROR.match(rest)
    if in_body is not None:
        code = in_body.group(1)
        return "error" if code == "None" else _status_word(int(code), rest[in_body.end() :])
    status = _STATUS.match(rest)
    if status is not None:
        return _status_word(int(status.group(1)), rest[status.end() :])
    if rest.startswith(("malformed response", "HTTP 200, ")):
        return "bad_answer"
    named = _CLASS.match(rest)
    if named is not None:
        # An httpx exception class, as the engine's client names a transport failure.
        if named.group(1).endswith("Timeout"):
            return "timeout"
        if named.group(1).endswith("Error"):
            return "unreachable"
    return "error"


def failure_word(exc: BaseException) -> str:
    """The short machine word for a failed call: one of ``WORDS``, never text from the
    exception. A failed pick on a visitor's key shows this instead of the exception.

    An error from a client of :func:`own_llm` already carries its word (``reason``);
    anything else is classified here the same way, so the function is safe on any
    exception."""
    if isinstance(exc, CallLimitExceeded):
        return "call_limit"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, ParseError):
        return "bad_answer"
    reason = getattr(exc, "reason", None)
    if isinstance(reason, str):
        if reason in _CLAUDE_REASONS:
            # The adapter files every other 4xx under ``bad_request``; the status it
            # wrote at the start of its message tells a 404 from the rest.
            status = _ANTHROPIC_STATUS.match(str(exc)) if reason == "bad_request" else None
            if status is not None:
                return _status_word(int(status.group(1)))
            return _CLAUDE_REASONS[reason]
        if reason in WORDS:
            return reason
    # Raw SDK and transport errors: none should get this far, all are understood.
    if isinstance(exc, anthropic.APIStatusError):
        return _status_word(exc.status_code, str(exc))
    if isinstance(
        exc, (anthropic.APITimeoutError, httpx.TimeoutException, httpx2.TimeoutException)
    ):
        return "timeout"
    if isinstance(exc, (anthropic.APIConnectionError, httpx.TransportError, httpx2.TransportError)):
        return "unreachable"
    if isinstance(exc, LLMError):
        return _from_text(str(exc))
    return "error"


def _name(provider: str) -> str:
    """The provider's display name; an unknown value is never echoed."""
    known = PROVIDER_NAMES.get(provider) if isinstance(provider, str) else None
    return known or "The provider"


def provider_error(exc: BaseException, provider: str) -> ServiceError:
    """What the visitor is told when their provider failed: a fixed sentence with the
    provider's display name and, for a 502, the machine word. Nothing of ``exc`` but its
    classification is used. The class name of ``exc`` goes with the error as
    ``own_error``, for the log line: the visitor's sentence is the same for a provider
    that is down and for a client that could not be built, and the operator has to be
    able to tell them apart.

    - 401 ``provider_key_refused``: the key was refused (401, 403), or the account
      behind it cannot pay (402, a spent balance or quota).
    - 429 ``provider_rate_limited``.
    - 502 ``provider_error`` for everything else.
    """
    name = _name(provider)
    word = failure_word(exc)
    if word == KEY_REFUSED:
        error = ServiceError(
            "provider_key_refused",
            f"{name} refused this key. It may be mistyped, revoked, out of credit, or not "
            "allowed to use this model.",
            status=401,
        )
    elif word == RATE_LIMITED:
        error = ServiceError(
            "provider_rate_limited",
            f"{name} is rate limiting this key; try again shortly.",
            status=429,
        )
    else:
        error = ServiceError(
            "provider_error", f"{name} could not complete the request ({word}).", status=502
        )
    error.own_error = type(exc).__name__
    return error


# --- clients ----------------------------------------------------------------------------


def _wait(value: object) -> float | None:
    """A ``retry_after`` worth keeping: a finite number of seconds, not negative."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = float(value)
    except OverflowError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _seal(exc: Exception, provider: str) -> Exception:
    """``_sealed_error``, and a fixed error should that ever fail. It is called inside an
    ``except`` block: an exception escaping from here would carry the unsealed original
    out as its context, so nothing escapes. The result is returned, never raised here."""
    try:
        return _sealed_error(exc, provider)
    except Exception:  # noqa: BLE001 - see above
        failure = LLMFatalError("request failed (error)")
        failure.reason = "error"  # type: ignore[attr-defined]
        return failure


def _sealed_error(exc: Exception, provider: str) -> Exception:
    """A new error that says in fixed words what ``exc`` was, and carries nothing of it.

    Same family as the original (``CallLimitExceeded``, ``ClaudeRefused``, retryable,
    fatal), the same ``usage`` and ``retry_after``, ``reason`` set to the machine word.
    The message, the cause, the context and the traceback are not carried over: the
    message may echo the key, and the frames hold the request and its headers."""
    word = failure_word(exc)
    name = _name(provider)
    usage = getattr(exc, "usage", None)
    usage = usage if isinstance(usage, Usage) else None
    sealed: Exception
    if isinstance(exc, WireRefused):
        sealed = WireRefused(f"{name} request failed ({word})", usage=usage)
    elif isinstance(exc, CallLimitExceeded):
        sealed = CallLimitExceeded(f"{name} call limit for this request reached", usage=usage)
    elif isinstance(exc, ClaudeRefused):
        sealed = ClaudeRefused(f"{name} request failed ({word})", usage=usage)
    elif isinstance(exc, LLMRetryableError):
        sealed = LLMRetryableError(
            f"{name} request failed ({word})", retry_after=_wait(exc.retry_after), usage=usage
        )
    else:
        sealed = LLMFatalError(f"{name} request failed ({word})", usage=usage)
    sealed.reason = word  # type: ignore[attr-defined]
    return sealed


class WireRefused(LLMFatalError):
    """A request built for a visitor's key was not sent: it was not for the provider's
    host, did not carry exactly the visitor's key, or (Anthropic) carried a header or a
    value the SDK does not send on its own. The message is the fixed one of any failed
    request; the class name is what the log shows (``own_error``)."""


class _NotSent(RuntimeError):
    """Raised inside the transport; replaced by :class:`WireRefused` before it leaves."""


class _Unreadable(Exception):
    """What a provider sent back cannot be decoded: a body in a coding the HTTP library
    does not know (raised by the wire), or a 200 from Anthropic that is not JSON
    (raised around the SDK call, ``_OwnClaude``). Replaced by a fixed ``bad_answer``
    failure before it leaves. The message is fixed; nothing of the answer is in it."""


def _raised_from(exc: BaseException, kind: type[BaseException]) -> bool:
    """Whether `exc` is, or was raised because of, an exception of `kind`. The SDK and
    the adapter wrap what a transport raises, so the causes and contexts are followed.
    This is asked of one call's own error: a client is shared by picks that run at the
    same time, and a flag on the client would blame them all."""
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        if isinstance(item, kind):
            return True
        seen.add(id(item))
        stack += [item.__cause__, item.__context__]
    return False


# What may fail a call on a visitor's key and is then the provider's failure, said in
# fixed words: the engine's errors (a call limit among them), the SDK's, the two HTTP
# libraries' (a stream read out of turn is a RuntimeError there), a timeout, and a
# request the wire did not send. Anything else is a fault of this site's code and is
# not caught: it must reach the handler, which logs its type and where it was raised
# (and never its message) and answers 500.
_PROVIDER_FAILURES: tuple[type[BaseException], ...] = (
    LLMError,
    anthropic.AnthropicError,
    httpx.HTTPError,
    httpx.StreamError,
    httpx.InvalidURL,
    httpx2.HTTPError,
    httpx2.StreamError,
    httpx2.InvalidURL,
    asyncio.TimeoutError,
    TimeoutError,
    _NotSent,
    _Unreadable,
)

# The response headers that reach the HTTP library, each only in this form. Everything
# else a provider sends in a header is dropped at the wire.
_RESPONSE_HEADERS: dict[bytes, re.Pattern[bytes]] = {
    # The body's size. Its coding is handled apart (``_Wire._coding``).
    b"content-length": re.compile(rb"[0-9]{1,12}"),
    # Read by the engine's client and the Claude adapter (seconds), and by the SDK.
    b"retry-after": re.compile(rb"[0-9]{1,6}(?:\.[0-9]{1,3})?"),
    b"retry-after-ms": re.compile(rb"[0-9]{1,9}"),
}
_JSON_TYPE = re.compile(
    rb"\s*application/(?:[a-z0-9.-]+\+)?json\s*(?:;.*)?", re.IGNORECASE | re.DOTALL
)


class _Wire:
    """The last thing a request passes and the first thing a response meets: a wrapper
    around the real transport of one own-key client (``_HttpxWire`` for the engine's
    client, ``_Httpx2Wire`` for the SDK's).

    Going out, a request is refused unless it is ``https`` to the provider's host on the
    default port, with exactly one ``Authorization`` header holding the visitor's key.
    For Anthropic the whole header set is checked as well (``expect``): every name must
    be one the SDK sends on its own, once, with the value the SDK computes, so nothing
    from the operator's environment rides along under any name. Until ``expect`` has
    been called such a wire refuses everything, and the ``Content-Length`` must be
    the length of the body. A refusal sends nothing and names no header and no value.

    Coming back, before the HTTP library reads the response:

    - a redirect (any 3xx) is replaced by a bare response of the same status. Redirects
      are not followed, but the library still works out where one points, and a
      relative ``Location`` goes through ``urllib.parse``, whose ``urlsplit`` cache
      lives as long as the process: a ``Location`` that echoed the key would stay
      there on a warm instance after the client is closed.
    - any other response keeps its status and body and gets a new set of headers
      (``_headers_for``): a content type that is one of two fixed strings, and the
      few headers of ``_RESPONSE_HEADERS`` that have exactly their strict form, and
      the body's content coding in the spelling the library decodes (``_coding``).
      Searching the headers for the key is not enough: a charset is looked up through
      ``codecs``, and ``encodings`` remembers every name it did not find, also one
      that arrived encoded (``charset*=utf-8''%73%6B...``) and was decoded by the
      library. And a ``Set-Cookie`` would be stored by the client and sent back with
      its next request. The body is not touched: what a body says is replaced
      further in (``_Sealed``)."""

    _response: Any  # the library's Response class
    _headers: Any  # the library's Headers class

    def __init__(self, inner: Any, credential: Credential, *, strict: bool = False) -> None:
        self._inner = inner
        self._host = HOSTS[credential.provider]
        self._token = credential.token  # a callable whose repr shows nothing
        self._strict = strict
        self._expected: dict[str, frozenset[str]] | None = None

    def expect(self, headers: Mapping[str, Sequence[str]]) -> None:
        """The header names a request may carry beside ``Authorization``, ``Host`` and
        ``Content-Length``, each with the values it may have."""
        self._expected = {name.lower(): frozenset(values) for name, values in headers.items()}

    def _allowed(self, request: Any) -> bool:
        url = request.url
        if url.scheme != "https" or url.host != self._host or url.port is not None or url.userinfo:
            return False
        sent: dict[str, list[str]] = {}
        for name, value in request.headers.multi_items():
            sent.setdefault(name.lower(), []).append(value)
        if sent.get("authorization") != [f"Bearer {self._token()}"]:
            return False
        if not self._strict:
            return True
        if self._expected is None or sent.get("host") != [self._host]:
            return False
        for name, values in sent.items():
            if len(values) != 1:
                return False
            if name in ("authorization", "host"):
                continue
            if name == "content-length":
                # Not any number: the length of the body that is about to be sent.
                try:
                    length = len(request.content)
                except Exception:  # noqa: BLE001 - a body that cannot be measured
                    return False
                if values[0] != str(length):
                    return False
            elif values[0] not in self._expected.get(name, ()):
                return False
        return True

    def _headers_for(self, response: Any, request: Any) -> Any:
        """The headers the HTTP library is given for `response`."""
        seen: dict[bytes, list[bytes]] = {}
        for name, value in response.headers.raw:
            seen.setdefault(name.lower(), []).append(value)
        kinds = seen.get(b"content-type", [])
        json_body = len(kinds) == 1 and _JSON_TYPE.fullmatch(kinds[0]) is not None
        kept = [
            (
                b"content-type",
                b"application/json; charset=utf-8" if json_body else b"text/plain; charset=utf-8",
            )
        ]
        for name, pattern in _RESPONSE_HEADERS.items():
            values = seen.get(name, [])
            if len(values) == 1 and pattern.fullmatch(values[0]) is not None:
                kept.append((name, values[0]))
        coding = self._coding(seen.get(b"content-encoding", []), request)
        if coding:
            kept.append((b"content-encoding", coding))
        return self._headers(kept)

    @staticmethod
    def _coding(values: Sequence[bytes], request: Any) -> bytes:
        """The body's content codings in the one spelling the HTTP library decodes:
        lower case, comma and space between them. Coding names are case-insensitive
        and the library compares them as they come, so ``GZip`` would leave the body
        compressed. A coding the client did not offer in its own ``Accept-Encoding``
        (that header is the library's list of what it can decode), or anything that
        is not a plain list of names, cannot be decoded: ``_Unreadable``, not a body
        handed on still encoded."""
        offered = {"identity"}
        for value in request.headers.get_list("accept-encoding", split_commas=True):
            offered.add(value.split(";", 1)[0].strip().lower())
        names: list[bytes] = []
        for value in values:
            for part in value.split(b","):
                name = part.strip().lower()
                if not name:
                    continue
                if name.decode("latin-1") not in offered:
                    raise _Unreadable("the answer is in a content coding that cannot be decoded")
                names.append(name)
        return b", ".join(names)

    async def handle_async_request(self, request: Any) -> Any:
        if not self._allowed(request):
            raise _NotSent("a request was not the one built for this key; nothing was sent")
        response = await self._inner.handle_async_request(request)
        if 300 <= response.status_code < 400:
            await response.aclose()
            return self._response(response.status_code, request=request)
        try:
            response.headers = self._headers_for(response, request)
        except _Unreadable:
            await response.aclose()
            raise
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class _HttpxWire(_Wire, httpx.AsyncBaseTransport):
    _response = httpx.Response
    _headers = httpx.Headers


class _Httpx2Wire(_Wire, httpx2.AsyncBaseTransport):
    _response = httpx2.Response
    _headers = httpx2.Headers


# The dispatch the running task reserved from a ``_QuietBudget`` and has not reported
# yet: (which budget, the dispatch's index). A client is shared by picks that run at
# the same time, each in a task of its own, so this is kept per task, not on the
# client. One variable for the module: a context variable is never collected.
_OPEN_DISPATCH: ContextVar[tuple[int, int] | None] = ContextVar(
    "ner_site_open_dispatch", default=None
)


class _QuietBudget:
    """What the engine's client reserves from and reports to: the real budget, told only
    the machine word of a failed request. A budget remembers the last failure and quotes
    it when it refuses a call, and the client's own report quotes the upstream."""

    def __init__(self, budget: CallBudget) -> None:
        self._budget = budget

    def acquire(self) -> int:
        index = self._budget.acquire()
        _OPEN_DISPATCH.set((id(self), index))
        return index

    def settle(self, index: int, error: str | None) -> None:
        if _OPEN_DISPATCH.get() == (id(self), index):
            _OPEN_DISPATCH.set(None)
        self._budget.settle(index, None if error is None else _from_text(error))

    def settle_open(self, word: str) -> None:
        """Report the dispatch the client reserved and never reported, as `word`. The
        engine's client reports every request it gets an answer or an HTTP error for;
        what the wire raises on its own (``_Unreadable``) passes through it unreported,
        and the budget would go on saying the provider is well."""
        found = _OPEN_DISPATCH.get()
        if found is not None and found[0] == id(self):
            _OPEN_DISPATCH.set(None)
            self._budget.settle(found[1], word)


class _Sealed:
    """An ``LLMClient`` over a client built with a visitor's key, through which no
    upstream text passes: every failure of the provider, the SDK or the HTTP library
    is replaced (``_sealed_error``), and the key is removed from an answer's text and
    model name should either ever contain it. An exception that is none of those
    (``_PROVIDER_FAILURES``) is a bug here, and is left alone."""

    def __init__(self, inner: LLMClient, credential: Credential) -> None:
        self._inner = inner
        self._provider = credential.provider
        self._token = credential.token

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def fingerprint(self) -> str:
        return self._inner.fingerprint

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._inner.capabilities

    @property
    def generation_defaults(self) -> Mapping[str, Any]:
        return client_generation_defaults(self._inner)

    def _clean(self, response: LLMResponse) -> LLMResponse:
        key = self._token()
        if not key or (key not in response.text and key not in response.model):
            return response
        return dataclasses.replace(
            response,
            text=response.text.replace(key, "[redacted]"),
            model=response.model.replace(key, "[redacted]"),
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._inner.complete(request)
        except _PROVIDER_FAILURES as exc:
            failure = _seal(exc, self._provider)
            if _raised_from(exc, _NotSent):
                # Not a provider's failure: the request never left (``_Wire``). The
                # libraries report it as a dropped connection; say what it was.
                failure = WireRefused(
                    f"{_name(self._provider)} request failed (error)",
                    usage=getattr(failure, "usage", None),
                )
                failure.reason = "error"  # type: ignore[attr-defined]
            elif _raised_from(exc, _Unreadable):
                # An answer arrived and could not be decoded; the libraries report
                # that too as a dropped connection, or not at all.
                self._unread()
                failure = LLMRetryableError(
                    f"{_name(self._provider)} request failed (bad_answer)",
                    usage=getattr(failure, "usage", None),
                )
                failure.reason = "bad_answer"  # type: ignore[attr-defined]
        else:
            return self._clean(response)
        # Raised out here, not inside the handler, so that the original is neither the
        # new error's context nor kept alive by its traceback.
        raise failure

    def _unread(self) -> None:
        """An answer could not be decoded. Nothing to do here: the wrapper around this
        client settles the call (``BudgetedLLM``). ``_SealedGated`` has to."""

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if callable(close):
            await close()


class _SealedGated(_Sealed):
    """``_Sealed`` over the engine's OpenAI-compatible client, which reserves from the
    call budget itself, once per HTTP request (``BudgetedLLM`` looks for this hook)."""

    _quiet: _QuietBudget | None = None

    def attach_call_budget(self, budget: CallBudget) -> None:
        self._quiet = _QuietBudget(budget)
        self._inner.attach_call_budget(self._quiet)  # type: ignore[attr-defined]

    def _unread(self) -> None:
        # Before the failure leaves: the dispatch the engine's client reserved for the
        # request is still open, because what the wire raised went straight through it.
        if self._quiet is not None:
            self._quiet.settle_open("bad_answer")


class _NamesOnly(CallBudget):
    """A call budget that remembers how a call ended by the class of its error and
    nothing more. The engine's ``BudgetedLLM`` settles a failed call with ``Class:
    message`` and the budget quotes that when it later refuses a call; the message of
    an error that was not sealed (a fault of this code) may hold anything."""

    def settle(self, index: int, error: str | None) -> None:
        if error is not None:
            name = error.split(":", 1)[0]
            error = name if name.isidentifier() else "error"
        super().settle(index, error)


class _OwnClaude(ClaudeClient):
    """The Claude adapter on a visitor's key. The adapter itself decides what is a 200
    that cannot be decoded (``ClaudeClient._send``, around the SDK call only); here it
    is said as on the other two providers, ``bad_answer``, where the site's key calls
    it ``transient`` for its fallback chain. There is no chain on a visitor's key."""

    def _unreadable(self) -> Exception:
        return _Unreadable("the answer is not JSON")


class _Budgeted(BudgetedLLM):
    """The engine's ``BudgetedLLM``, whose own refusal is sealed too. For a client that
    does not reserve from the budget itself (Claude), the limit error is raised out
    here, around the sealed client, in the engine's words: they quote the last provider
    error and advise a command-line flag. Sealing an error that is already sealed
    changes nothing."""

    def __init__(self, inner: LLMClient, budget: CallBudget, provider: str) -> None:
        super().__init__(inner, budget)
        self._provider = provider

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            return await super().complete(request)
        except _PROVIDER_FAILURES as exc:
            failure = _seal(exc, self._provider)
        raise failure


class OpenAIClient(OpenAICompatClient):
    """The engine's client for ``api.openai.com``, with the two fields OpenAI's reasoning
    models do not take replaced.

    The engine sends ``max_tokens`` and ``temperature``. OpenAI's reasoning models answer
    400 to ``max_tokens`` (they take ``max_completion_tokens``, which every chat model
    accepts) and to any ``temperature`` but the default. So the limit is sent under the
    name that works everywhere and no temperature is sent. The limit is this client's
    own, far above the engine's 2048 / 600, because hidden reasoning counts against it:
    OpenRouter showed GPT-6 Luna spending all 2048 tokens on reasoning and returning
    nothing (``service.OpenRouterClient``). ``reasoning_effort`` is not sent: a model
    that does not reason answers 400 to it, and the lists of models are the operator's.

    Not tried against the live API (rollout check 2 of the spec)."""

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        body = super()._body(request, structured=structured)
        body.pop("max_tokens", None)
        body.pop("temperature", None)
        body["max_completion_tokens"] = self._max_tokens
        return body


# The headers the SDK sends on its own for an API key (2.10.0 ``default_headers``), beside
# its ``X-Stainless-*`` ones.
_SDK_HEADERS = frozenset(
    {"accept", "content-type", "user-agent", "authorization", "anthropic-version"}
)


def _own_headers_only(headers: Mapping[str, Any], key: str) -> None:
    """Refuse a client whose default headers are not the SDK's own with the visitor's
    key. This looks at the client, by name; what is checked on every request, by name
    and value, is ``_Wire``. The message names no header and no value."""
    names = {str(name).lower() for name in headers}
    foreign = {
        name for name in names if name not in _SDK_HEADERS and not name.startswith("x-stainless-")
    }
    sent = [value for name, value in headers.items() if str(name).lower() == "authorization"]
    if foreign or sent != [f"Bearer {key}"]:
        raise RuntimeError("the Anthropic client would send headers it was not given")


def _sdk_wire_headers(sdk: Any, http: Any, timeout: float) -> dict[str, tuple[str, ...]]:
    """What a request of the SDK (2.10.0) carries beside the credential, with the values
    the SDK and its HTTP client compute themselves: ``_base_client._build_headers``,
    ``default_headers`` and ``platform_headers``, read through public attributes only.
    Anything else on a request, or another value under one of these names, did not come
    from the SDK."""
    return {
        "accept": ("application/json",),
        "content-type": ("application/json",),
        # The HTTP client's own defaults (it is built here, with no headers of ours).
        "accept-encoding": (http.headers["accept-encoding"],),
        "connection": (http.headers["connection"],),
        "user-agent": (sdk.user_agent,),
        "anthropic-version": (ANTHROPIC_VERSION,),
        **{name.lower(): (value,) for name, value in sdk.platform_headers().items()},
        "x-stainless-async": ("async:asyncio",),
        "x-stainless-retry-count": ("0",),  # max_retries=0: there is no second attempt
        # The per-request timeout (none is passed) and the client's, as the SDK prints them.
        "x-stainless-timeout": ("NOT_GIVEN", str(timeout)),
        "x-stainless-read-timeout": (str(timeout),),
    }


def _claude(
    credential: Credential, model: str, purpose: str, effort: str, timeout: float, transport: Any
) -> tuple[LLMClient, _Wire]:
    """Claude on the visitor's key: the adapter with the site's effort and token limits,
    no server-side refusal fallback, and an SDK client that can reach one host only.
    Returned with the wire it sends through.

    The SDK's default HTTP client follows redirects and builds its proxies from the
    environment even with ``trust_env=False`` (2.10.0 ``_DefaultAsyncHttpxClient`` reads
    ``get_environment_proxies()`` whenever it is given no transport), so it is given a
    transport of its own.

    What the SDK reads from the environment (2.10.0 ``_client.py``, ``_tracing.py``), and
    what keeps each out of this client:

    - ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, ``ANTHROPIC_PROFILE``, the
      federation variables and the profile on disk: not consulted once ``api_key`` is
      passed.
    - ``ANTHROPIC_BASE_URL``: not read once ``base_url`` is passed.
    - ``ANTHROPIC_OPEN_TELEMETRY*`` (spans, and trace headers on the request): not read
      once ``open_telemetry`` is passed; ``False`` switches it off.
    - ``ANTHROPIC_CUSTOM_HEADERS``: always read, merged into the client's custom headers
      and applied *after* the credential, so an ``Authorization`` line there would send
      the operator's token in place of the visitor's key. The constructor has no switch
      for it. ``default_headers`` wins over the variable for the same name, which covers
      the credential; every other line is dropped by emptying the custom headers once
      the client exists (they then hold nothing else: this client is given none).

    The emptying goes through a private attribute, so it is not what the rest relies
    on. ``_own_headers_only`` refuses a client whose defaults are not the SDK's own, and
    the wire (``_Wire``, strict) refuses every single request that is not: the pinned
    host, the visitor's key once, and no header but the SDK's own with the SDK's own
    value (``_sdk_wire_headers``). An SDK that kept the environment's headers somewhere
    else, or applied them later, would have its requests stopped there, unsent. That
    list of headers is the SDK's of one version, which is why the version is pinned
    (``AUDITED_SDK``).

    One request per call (``max_retries=0``), as on the site's key, so nothing is
    retried out of sight of the call budget."""
    key = credential.token()
    wire = _Httpx2Wire(
        transport if transport is not None else httpx2.AsyncHTTPTransport(trust_env=False),
        credential,
        strict=True,
    )
    http = anthropic.DefaultAsyncHttpxClient(
        transport=wire, follow_redirects=False, trust_env=False
    )
    sdk = anthropic.AsyncAnthropic(
        api_key=key,
        base_url=ANTHROPIC_BASE_URL,
        max_retries=0,
        timeout=timeout,
        open_telemetry=False,
        default_headers={"Authorization": f"Bearer {key}"},
        http_client=http,
    )
    sdk._custom_headers = {}  # noqa: SLF001 - ANTHROPIC_CUSTOM_HEADERS, see above
    _own_headers_only(sdk.default_headers, key)
    wire.expect(_sdk_wire_headers(sdk, http, timeout))
    client = _OwnClaude(
        model,
        api_key=key,
        max_tokens=EXTRACT_MAX_TOKENS if purpose == EXTRACT else SELECT_MAX_TOKENS,
        effort=effort,
        timeout=timeout,
        refusal_fallback=False,
        client=sdk,
    )
    return client, wire


def _compat(
    credential: Credential, model: str, purpose: str, transport: Any, backoff_base: float
) -> tuple[LLMClient, _Wire]:
    """OpenAI or OpenRouter on the visitor's key, through the engine's client: redirects
    are not followed (httpx's default, which the engine keeps) and ``trust_env=False``
    keeps the environment's proxies and ``.netrc`` out of it. Retries, timeout and, on
    OpenRouter, the token limits are the site's own for the same purpose. Returned with
    the wire it sends through (``_Wire``: the pinned host, the visitor's key)."""
    from ner_site import service  # at call time: service imports this module

    extract = purpose == EXTRACT
    wire = _HttpxWire(
        # What the engine's client would build for itself when given no transport.
        transport if transport is not None else httpx.AsyncHTTPTransport(trust_env=False),
        credential,
    )
    common: dict[str, Any] = {
        "api_key": credential.token(),
        "timeout": service.LLM_TIMEOUT_S,
        "max_retries": service.LLM_MAX_RETRIES if extract else SELECT_MAX_RETRIES,
        "backoff_base": backoff_base,
        "transport": wire,
        "trust_env": False,
    }
    if credential.provider == "openai":
        client: LLMClient = OpenAIClient(
            BASE_URLS["openai"],
            model,
            profile="openai",
            max_tokens=EXTRACT_MAX_TOKENS if extract else SELECT_MAX_TOKENS,
            **common,
        )
        return client, wire
    client = service.OpenRouterClient(
        BASE_URLS["openrouter"],
        model,
        profile=service.provider_profile(model),
        temperature=TEMPERATURE if extract else 0.0,
        max_tokens=MAX_TOKENS if extract else OPENROUTER_SELECT_MAX_TOKENS,
        extra_headers={"HTTP-Referer": service.SITE_URL, "X-Title": service.SITE_TITLE},
        **common,
    )
    return client, wire


def own_llm(
    credential: Credential,
    model: str,
    *,
    purpose: Literal["extract", "select"],
    effort: str,
    max_entities: int,
    call_cap: int,
    transport: Any | None = None,
    backoff_base: float = 1.0,
) -> LLMClient:
    """The client for one stage of one request on the visitor's key: built here, used
    for that request, closed by the caller (``await llm.aclose()``).

    ``model`` is an id from the deployment's list for the provider (the caller checks
    that; a visitor never names a free-form model). ``effort`` is the site's Claude
    effort and only Anthropic takes it. The call budget is the site's for the purpose:
    ``call_cap`` requests for an extraction, two per entity allowed for a selection.

    No chain: one model, the provider's own errors (in fixed words, see ``_Sealed``).
    ``transport`` and ``backoff_base`` exist for tests (an ``httpx2`` transport for
    Anthropic, an ``httpx`` one for the others)."""
    if credential.provider not in OWN_PROVIDERS:
        raise ValueError("not a provider a visitor's key can be used with")
    if purpose not in (EXTRACT, SELECT):
        raise ValueError("purpose must be 'extract' or 'select'")
    budget = _NamesOnly(limit=call_cap if purpose == EXTRACT else max_entities * 2)
    if credential.provider == "anthropic":
        from ner_site import service  # at call time: service imports this module

        # No chain to leave time for, so both purposes get the ordinary request timeout
        # rather than the shorter one a chain's Claude extraction has.
        claude, _ = _claude(credential, model, purpose, effort, service.LLM_TIMEOUT_S, transport)
        return _Budgeted(_Sealed(claude, credential), budget, credential.provider)
    inner, _ = _compat(credential, model, purpose, transport, backoff_base)
    return _Budgeted(_SealedGated(inner, credential), budget, credential.provider)


__all__ = [
    "ANTHROPIC_VERSION",
    "AUDITED_SDK",
    "BASE_URLS",
    "HOSTS",
    "KEY_MAX",
    "KEY_MIN",
    "OWN_PROVIDERS",
    "PROVIDER_NAMES",
    "SITE",
    "WORDS",
    "Credential",
    "WireRefused",
    "OpenAIClient",
    "ProviderInfo",
    "failure_word",
    "own_llm",
    "provider_error",
    "read_credential",
]
