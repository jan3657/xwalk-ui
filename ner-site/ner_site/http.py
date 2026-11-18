"""Framework-free request handling: routing, the header and passcode rules, the error
envelope, and one log line per request (docs/ner-site/PLAN.md sections 3.1 and 7).

:func:`handle` is what ``api/*.py`` (through :func:`serve`), ``tools/serve_local.py``
and the tests call. It never raises: every outcome is ``(status, headers, body)`` with a
JSON body, ``Cache-Control: no-store`` and no ``Access-Control-*`` header.

Order of checks for ``POST /api/ner``: path (404), method (405; ``OPTIONS`` too, on
purpose, so no preflight ever succeeds), body size (413), ``X-Requested-With:
xwalk-ner`` (400 ``bad_header``), passcode, JSON, credential, site configuration,
provider-specific validation, indexes and finally admission before any chat client
is built. Site-key Claude deployments apply the hourly and daily allowances and
refuse a request if the counters cannot be consulted. Own-key runs have separate
flood limits and can proceed during a counter outage. The visitor is a
keyed hash of the client address; neither it nor the address is ever logged or returned.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from functools import partial
from http.server import BaseHTTPRequestHandler
from typing import Any, TypeVar
from urllib.parse import urlsplit

from pydantic import ValidationError
from xwalk.llm.base import LLMCapabilities, LLMClient, LLMRequest, LLMResponse, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.retrieval.postgres import NeonHTTPExecutor, SQLExecutor

from ner_site import fake, fallback, limits, providers
from ner_site.claude import OPUS, SONNET, is_claude
from ner_site.indexes import OntologyIndex, get_indexes
from ner_site.ontologies import catalog, full_catalog
from ner_site.schema import (
    FallbackNote,
    HealthResponse,
    NerRequest,
    NerResponse,
    OntologySummary,
    OwnKeys,
    OwnProvider,
    RetrieverStatus,
    ServiceError,
    VisitorStatus,
)
from ner_site.select import JEV, RETRIEVAL_ONLY
from ner_site.service import (
    BLOCK_TEXT,
    ENV_NAMES,
    EXTRACT,
    SELECT,
    ClaudeFactory,
    Settings,
    SettingsError,
    chain_models,
    check_claude_key,
    check_key,
    make_chain,
    make_decider,
    make_llm,
    make_select_llm,
    retrieval_status,
    run,
    uses_database,
    validate_request,
)

T = TypeVar("T")
Response = tuple[int, dict[str, str], bytes]
LLMFactory = Callable[[Settings, str], LLMClient]

HEALTH_PATH = "/api/health"
NER_PATH = "/api/ner"
ALLOWED = {HEALTH_PATH: ("GET", "HEAD"), NER_PATH: ("POST",)}
REQUESTED_WITH = "xwalk-ner"
KNOWN_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
MAX_LENGTH_DIGITS = 18
REQUEST_WORKERS = 16  # threads per request for retriever searches (BM25 and embeddings)
# One round trip to Neon before the request may run: short, so a slow database becomes
# a prompt refusal on site keys, or an uncounted own-key run. It is twice the
# limiter's own bound on waiting for a visitor's lock (limits.LOCK_TIMEOUT_MS), so a
# queue of one visitor's requests ends as a contended lock (429), not as this timeout.
LIMITS_TIMEOUT_S = 3.0
CONTENDED_RETRY_AFTER_S = 5
LIMITS_UNAVAILABLE = "limits_unavailable"
CLAUDE_UNAVAILABLE = "claude_unavailable"


# The SDK logs response headers and both HTTP clients log the reason phrase, even
# when a provider echoes credentials there. Filters on the emitting loggers run
# before any handler can format an exception. The context carries only a boolean.
_OWN_RUN = ContextVar("ner_site_own_run", default=False)


def _outside_own_run(record: logging.LogRecord) -> bool:
    return not _OWN_RUN.get()


for _logger in (
    "anthropic._base_client",
    "anthropic._response",
    "httpx",
    "httpx2",
    *(
        f"{core}.{part}"
        for core in ("httpcore", "httpcore2")
        for part in ("connection", "http11", "http2")
    ),
):
    logging.getLogger(_logger).addFilter(_outside_own_run)


def _headers(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        **(extra or {}),
    }


def _redact(data: Any, secrets: Sequence[str]) -> Any:
    """Scrub strings before JSON escaping can split a quoted or backslashed secret."""
    if isinstance(data, str):
        for secret in secrets:
            data = data.replace(secret, "[redacted]")
    elif isinstance(data, dict):
        return {_redact(k, secrets): _redact(v, secrets) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_redact(item, secrets) for item in data]
    return data


def _json_body(data: object, settings: Settings | None, extra_secrets: Sequence[str] = ()) -> bytes:
    secrets = (*extra_secrets, *(settings.secrets() if settings else ()))
    return json.dumps(_redact(data, secrets), ensure_ascii=False).encode("utf-8")


def _error(
    exc: ServiceError,
    settings: Settings | None,
    extra: Mapping[str, str] | None = None,
    extra_secrets: Sequence[str] = (),
) -> Response:
    secrets = (*extra_secrets, *(settings.secrets() if settings else ()))
    return (
        exc.status,
        _redact(_headers(extra), secrets),
        _json_body(exc.envelope(), settings, extra_secrets),
    )


def _log(
    record: dict[str, Any], settings: Settings | None, extra_secrets: Sequence[str] = ()
) -> None:
    secrets = (*extra_secrets, *(settings.secrets() if settings else ()))
    line = json.dumps(_redact({"event": "ner_site.request", **record}, secrets), ensure_ascii=True)
    print(line, file=sys.stderr, flush=True)


def default_llm_factory(settings: Settings, model: str) -> LLMClient:
    """The offline stand-in in offline mode, otherwise the budgeted OpenRouter client."""
    if settings.offline:
        return fake.offline_llm()
    return make_llm(settings, model)


# --- the visitor allowance: one limiter per warm instance and database -------------

_LIMITS_LOCK = threading.Lock()
_LIMITS: dict[tuple[str, str, limits.Allowance], limits.VisitorLimits] = {}


def make_limits_executor(database_url: str) -> SQLExecutor:
    """The executor the visitor limiter queries with. Not the indexes' one
    (``indexes.make_executor``): that one retries, and the limiter's statement counts a
    visit, so a retried request could count one visit twice. Tests replace it."""
    return NeonHTTPExecutor(database_url, timeout=LIMITS_TIMEOUT_S, max_retries=0)


def _now() -> float:
    return time.time()


def _visitor_limits(settings: Settings) -> limits.VisitorLimits | None:
    """The limiter, or None offline or without a database. A warm instance reuses
    the limiter for each database, schema and allowance. A setup failure is handled
    like a database that cannot be reached."""
    if settings.offline or settings.database_url is None:
        return None
    key = (settings.database_url, settings.neon_schema, settings.allowance)
    with _LIMITS_LOCK:
        found = _LIMITS.get(key)
        if found is None:
            try:
                executor = make_limits_executor(settings.database_url)
                found = limits.VisitorLimits(
                    executor, settings.neon_schema, settings.allowance, clock=_now
                )
            except ValueError as exc:  # a connection string the executor cannot use
                raise limits.LimitsUnavailable(
                    f"visitor limits could not be set up ({type(exc).__name__})"
                ) from None
            _LIMITS[key] = found
        return found


def _visitor(settings: Settings, headers: Mapping[str, str]) -> str:
    """The calling visitor's id (``limits.visitor_id``). Never log or return it."""
    salt = settings.visitor_salt or limits.default_salt(settings.database_url or "")
    return limits.visitor_id(headers, salt)


@dataclass(frozen=True)
class _Admitted:
    """Where an admitted request starts: its tier (``opus``, ``sonnet`` or ``other``),
    the step of ``chain_models`` its chain starts on, the step-downs that decided it
    (stage ``admit``), and why Claude was known to be unusable at that moment
    (``fallback.claude_block``, read once; None when it was usable)."""

    tier: str
    start: int = 0
    notes: tuple[FallbackNote, ...] = ()
    blocked: str | None = None
    window: str = "hour"


def _admit(
    settings: Settings, request: NerRequest, headers: Mapping[str, str], log: dict[str, Any]
) -> _Admitted | None:
    """Count own-key visits independently. Only site-key Claude deployments fail closed."""
    own = request.provider not in (None, providers.SITE)
    if not own and not settings.claude_offered:
        return None
    assert request.model is not None
    lead = next((m for m in (request.model, request.selector) if m and is_claude(m)), None)
    wanted: limits.Tier = "other"
    if own:
        wanted = "own"
    elif lead == OPUS:
        wanted = "opus"
    elif lead == SONNET:
        wanted = "sonnet"
    tier: str = wanted
    reason: str | None = None
    window = "hour"
    own_clause = ", or use your own API key" if settings.own_providers else ""
    try:
        limiter = _visitor_limits(settings)
        if limiter is not None:
            admission = limiter.admit(_visitor(settings, headers), wanted)
            tier, reason, window = admission.tier, admission.reason, admission.window
        elif not own and settings.passcode is None:
            raise ServiceError(LIMITS_UNAVAILABLE, "visitor limits are not configured", status=503)
    except limits.VisitorLimitReached as exc:
        minutes = max(1, math.ceil(exc.resets_in_s / 60))
        wait = f"try again in {minutes} minute{'' if minutes == 1 else 's'}"
        if own:
            message = f"too many requests with your own key; {wait}"
        elif exc.window == "day":
            message = f"you have used today's requests; they reset at midnight UTC{own_clause}"
        else:
            message = f"you have used this hour's requests; {wait}{own_clause}"
        raise ServiceError(
            "visitor_limit_reached",
            message,
            status=429,
            headers={"Retry-After": str(exc.resets_in_s)},
        ) from None
    except limits.LimitsUnavailable as exc:
        log["limits"] = str(exc)  # the limiter exposes only the class and SQLSTATE
        if exc.code == limits.LOCK_NOT_AVAILABLE:
            raise ServiceError(
                "visitor_limit_reached",
                "too many requests at once; try again in a few seconds",
                status=429,
                headers={"Retry-After": str(CONTENDED_RETRY_AFTER_S)},
            ) from None
        if not own:
            raise ServiceError(
                LIMITS_UNAVAILABLE,
                "the site cannot count requests right now; try again shortly" + own_clause,
                status=503,
                headers={"Retry-After": "30"},
            ) from None
    log["tier"] = tier
    if own or lead is None:
        return _Admitted(tier)

    models = chain_models(settings, lead)
    start = 0
    notes: list[FallbackNote] = []
    if wanted == "opus" and tier == "sonnet":
        start = 1
        notes.append(
            FallbackNote(stage="admit", from_=OPUS, to=SONNET, reason=reason or limits.OPUS_USED)
        )
    last = len(models) - 1
    blocked = fallback.claude_block()
    if blocked is not None and not is_claude(models[last]) and start < last:
        notes.append(
            FallbackNote(
                stage="admit", from_=models[start], to=models[last], reason=CLAUDE_UNAVAILABLE
            )
        )
        start = last
    return _Admitted(tier, start, tuple(notes), blocked, window)


def _passcode_error(settings: Settings, headers: Mapping[str, str]) -> ServiceError | None:
    if settings.passcode is None:
        return None
    given = headers.get("x-passcode", "")
    if not given:
        return ServiceError("passcode_required", "this demo needs a passcode", status=401)
    if not hmac.compare_digest(given.encode("utf-8"), settings.passcode.encode("utf-8")):
        return ServiceError("passcode_invalid", "the passcode is not correct", status=401)
    return None


def _parse(body: bytes) -> NerRequest:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ServiceError(
            "bad_json", "the request body is not valid UTF-8 JSON", status=400
        ) from None
    if not isinstance(data, dict):
        raise ServiceError("bad_json", "the request body must be a JSON object", status=400)
    try:
        return NerRequest.model_validate(data, strict=True)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = str(first["loc"][0]) if first["loc"] else ""
        if field == "text":
            raise ServiceError(
                "missing_text", "'text' is required and must be a string", status=400
            ) from None
        if field == "provider":
            raise ServiceError(
                "bad_credential",
                "own API keys are not available for that provider on this site",
                status=400,
            ) from None
        if field == "top_k":
            raise ServiceError("bad_top_k", "'top_k' must be an integer", status=400) from None
        if first["type"] == "extra_forbidden":
            message = (
                "unknown request field; send text, prompt, model, selector, top_k and provider"
            )
        else:
            message = f"'{field}' must be a string or null"
        raise ServiceError("bad_json", message, status=400) from None


def _check_encodable(request: NerRequest) -> NerRequest:
    """JSON may carry a lone surrogate (``"\\ud800"``) that no UTF-8 encoder accepts; it
    would fail later, in the provider call or the response, as a 500."""
    for name in ("text", "prompt", "model"):
        value = getattr(request, name)
        if value is None:
            continue
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise ServiceError(
                "bad_json",
                f"'{name}' is not valid UTF-8 text (it contains a lone surrogate escape)",
                status=400,
            ) from None
    return request


def run_request(coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run`` without its wait for executor threads.

    Retriever searches run in threads (``asyncio.to_thread``). A search cut off by its
    timeout or the request deadline cannot be stopped: ``asyncio.run`` would then wait
    for it in ``shutdown_default_executor`` (up to the encoder's own HTTP timeout, or a
    Retry-After sleep) after the answer is ready. Here each request gets its own pool,
    shut down without waiting: an abandoned thread finishes in the background on a warm
    instance, and its result goes to a future of this closed loop, never into a later
    response (nothing it writes is shared with one)."""
    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(max_workers=REQUEST_WORKERS, thread_name_prefix="ner-site")
    loop.set_default_executor(executor)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            loop.close()


def _catalog(settings: Settings, indexes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The bundled samples' catalog, or for the full catalogs the counts the database
    holds (0 for an ontology that is not loaded)."""
    if settings.catalog != "full":
        return catalog()
    return full_catalog(
        {slug: getattr(index.store, "record_count", None) for slug, index in indexes.items()}
    )


def _claude_health(settings: Settings) -> RetrieverStatus:
    """Whether a request can run on Claude now: the key is set, no request of this warm
    instance has met an exhausted balance or a refused key in the last minutes, and
    Anthropic accepts the key (one free call, remembered). Only those three are
    ``unavailable``; a key check that could not be completed is ``ok`` with a reason
    saying the key was not checked."""
    if settings.anthropic_api_key is None:
        return RetrieverStatus(status="unavailable", reason="key not set")
    blocked = fallback.claude_block()
    if blocked is not None:
        return RetrieverStatus(status="unavailable", reason=BLOCK_TEXT.get(blocked, blocked))
    return check_claude_key(settings) or RetrieverStatus(status="ok")


def _visitor_health(settings: Settings, headers: Mapping[str, str]) -> dict[str, Any]:
    """Report the read-only peek, or how admission behaves without usable counters."""
    allowance = settings.allowance
    try:
        limiter = _visitor_limits(settings)
        if limiter is not None:
            seen = limiter.peek(_visitor(settings, headers))
            balances = asdict(seen)
            del balances["reason"]
            return {"limits": "on", **balances}
        unmetered = settings.passcode is not None
    except limits.LimitsUnavailable:
        unmetered = False
    now = _now()
    return {
        "limits": "off" if unmetered else "unavailable",
        "tier": "opus" if unmetered else "blocked",
        "opus_left": allowance.opus_per_hour if unmetered else 0,
        "sonnet_left": allowance.sonnet_per_hour if unmetered else 0,
        "opus_left_today": allowance.opus_per_day if unmetered else 0,
        "sonnet_left_today": allowance.sonnet_per_day if unmetered else 0,
        "own_left": None,
        "own_left_today": None,
        "day_resets_in_s": math.ceil(limits.DAY_S - now % limits.DAY_S),
        "resets_in_s": math.ceil(limits.HOUR_S - now % limits.HOUR_S),
        "window": "hour",
    }


def _health(settings: Settings, headers: Mapping[str, str]) -> dict[str, Any]:
    indexes = get_indexes(settings.index_settings())
    health = HealthResponse(
        mode=settings.mode,
        default_model=settings.default_model,
        models=list(settings.offered_models),
        limits=settings.limits.public(),
        passcode_required=settings.passcode is not None,
        default_prompt=settings.default_prompt,
        ontologies=[OntologySummary.model_validate(o) for o in _catalog(settings, indexes)],
        retrieval=retrieval_status(indexes, settings),
        selectors=list(settings.offered_selectors),
        default_selector=settings.default_selector,
        own_keys=OwnKeys(
            enabled=bool(settings.own_providers),
            providers=[OwnProvider.model_validate(info) for info in settings.own_providers],
        ),
        visitor=VisitorStatus.model_validate(_visitor_health(settings, headers))
        if settings.claude_offered
        else None,
    )
    data: dict[str, Any] = health.model_dump(mode="json")
    # Not in HealthResponse: what an operator needs to see when /api/ner would be 503.
    if settings.offline:
        llm = RetrieverStatus(status="ok", reason="offline stand-in extractor (NER_SITE_OFFLINE=1)")
    elif settings.api_key is None:
        llm = RetrieverStatus(
            status="unavailable",
            reason="OPENROUTER_API_KEY is not set; "
            + (
                "only the Claude models are offered"
                if settings.configured
                else "POST /api/ner on the site's keys answers 503 not_configured"
            ),
        )
    else:
        llm = check_key(settings) or RetrieverStatus(status="ok")
    data["llm"] = llm.model_dump(mode="json")
    # Offline mode spends no key, so it reports no operator Claude probe.
    if not settings.offline:
        data["claude"] = _claude_health(settings).model_dump(mode="json")
    # Not in HealthResponse either: where the terms and indexes come from.
    data["backend"] = settings.backend
    if settings.backend == "neon":
        data["neon"] = {"schema": settings.neon_schema, "catalog": settings.catalog}
    return data


class _OwnJSONClient:
    """Clean decoded model strings before stages can use them in regex caches or diagnostics.

    The adapter removes literal echoes. JSON escapes can hide the token until the
    stage parses its answer. Use the stage parser so fenced and repaired JSON go
    through the same check. Leave unreadable answers to the usual stage errors.

    A call that fails is named in the request's log line (``own_error``, the class of
    what was raised), here because this is the one place both stages pass: a failed
    extraction becomes an error response, but a failed pick is part of a 200, and
    without this a request the site itself refused to send would leave no trace.
    Picks fail side by side and for different reasons. ``own_error`` is the first
    class seen, except that ``WireRefused`` (the site's own guard stopped a request)
    takes its place whenever it occurs: a provider failing is ordinary, that is not.
    When there was more than one class, ``own_errors`` lists them all, sorted, at
    most ``MAX_OWN_ERRORS``.
    """

    MAX_OWN_ERRORS = 8

    def __init__(
        self, client: LLMClient, credential: providers.Credential, log: dict[str, Any]
    ) -> None:
        self._client = client
        self._token = credential.token
        self._log = log

    @property
    def model(self) -> str:
        return self._client.model

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._client.capabilities

    @property
    def fingerprint(self) -> str:
        return self._client.fingerprint

    def _failed(self, name: str) -> None:
        log = self._log
        first = log.get("own_error")
        seen = sorted({*log.get("own_errors", [first] if first else []), name})
        if len(seen) > 1:
            log["own_errors"] = seen[: self.MAX_OWN_ERRORS]
        if first is None or name == providers.WireRefused.__name__:
            log["own_error"] = name

    @property
    def budget(self) -> Any:
        """The call budget of the client underneath, which counts every request it
        sends (``select.dispatched`` reads it)."""
        return getattr(self._client, "budget", None)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            answer = await self._client.complete(request)
        except Exception as exc:
            self._failed(type(exc).__name__)
            raise
        key = self._token()
        if "\\" not in answer.text and key not in answer.text:
            return answer
        try:
            data = parse_json_object(answer.text)
        except ParseError:
            return answer
        safe = _redact(data, (key,))
        return answer if safe == data else replace(answer, text=json.dumps(safe))


@contextmanager
def _provider_errors(provider: str) -> Iterator[None]:
    """A failure building an own client has the same envelope as a failed call; the
    log line names its class (``own_error``)."""
    try:
        yield
        return
    except asyncio.CancelledError:
        error = providers.provider_error(asyncio.TimeoutError(), provider)
    except Exception as exc:
        error = providers.provider_error(exc, provider)
    raise error


async def _run_own(
    credential: providers.Credential,
    valid: NerRequest,
    settings: Settings,
    indexes: Mapping[str, OntologyIndex],
    started: float,
    log: dict[str, Any],
) -> NerResponse:
    """Each constructed stage is closed even if construction of the next stage fails.
    A stage that fails to close is named in `log` and nothing more: the request's
    answer, or its own error, is what the visitor gets."""
    assert valid.model is not None and valid.selector is not None
    build = partial(
        providers.own_llm,
        credential,
        effort=settings.claude_effort,
        max_entities=settings.limits.max_entities,
        call_cap=settings.limits.llm_call_cap,
    )

    async def close(client: Any) -> None:
        try:
            await client.aclose()
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - see above
            log.setdefault("own_close_error", type(exc).__name__)

    async with AsyncExitStack() as stack:
        with _provider_errors(credential.provider):
            llm = build(valid.model, purpose="extract")
            stack.push_async_callback(close, llm)
            selector = None
            if valid.selector != RETRIEVAL_ONLY:
                selector = build(valid.selector, purpose="select")
                stack.push_async_callback(close, selector)
        return await run(
            valid,
            settings=settings,
            indexes=indexes,
            started_at=started,
            llm=_OwnJSONClient(llm, credential, log),
            select_llm=_OwnJSONClient(selector, credential, log) if selector is not None else None,
            extra_secrets=(credential.token(),),
        )


async def _ner(
    settings: Settings,
    valid: NerRequest,
    factory: LLMFactory,
    log: dict[str, Any],
    *,
    started: float,
    indexes: Mapping[str, OntologyIndex],
    admitted: _Admitted | None = None,
    credential: providers.Credential | None = None,
    claude_factory: ClaudeFactory | None = None,
) -> dict[str, Any]:
    """Build the clients for a validated, admitted request and run it on `indexes`.

    A request for a Claude model runs on one fallback chain, started where `admitted`
    says; a Claude selector uses that chain, or one of its own (for selection only) when
    the extraction model is an OpenRouter model. Everything else is built as before."""
    assert valid.model is not None
    model, selector = valid.model, valid.selector or RETRIEVAL_ONLY
    if credential is not None:
        token = _OWN_RUN.set(True)
        try:
            response = await _run_own(credential, valid, settings, indexes, started, log)
        finally:
            _OWN_RUN.reset(token)
    else:
        start = 0 if admitted is None else admitted.start
        chain: LLMClient | None = None
        if admitted is not None and is_claude(model):
            chain = make_chain(
                settings,
                model,
                start=start,
                purposes=(EXTRACT, SELECT) if is_claude(selector) else (EXTRACT,),
                claude_factory=claude_factory,
                llm_factory=factory,
                select_factory=make_select_llm,
            )
        llm = chain if chain is not None else factory(settings, model)
        decider = make_decider(settings) if selector == JEV else None
        select_llm: LLMClient | None = None
        if admitted is not None and is_claude(selector):
            select_llm = chain or make_chain(
                settings,
                selector,
                start=start,
                purposes=(SELECT,),
                claude_factory=claude_factory,
                select_factory=make_select_llm,
            )
        elif selector not in (JEV, RETRIEVAL_ONLY):
            select_llm = make_select_llm(settings, selector)
        try:
            response = await run(
                valid,
                settings=settings,
                llm=llm,
                indexes=indexes,
                started_at=started,
                decider=decider,
                select_llm=select_llm,
                admitted=() if admitted is None else admitted.notes,
                claude_blocked=None if admitted is None else admitted.blocked,
                allowance_window="hour" if admitted is None else admitted.window,
            )
        except BaseException:
            if admitted is not None:  # a failed request still says how far down it went
                log["fallbacks"] = [note.model_dump(mode="json") for note in admitted.notes] + [
                    {"stage": f.stage, "from": f.from_model, "to": f.to_model, "reason": f.reason}
                    for client in dict.fromkeys(c for c in (llm, select_llm) if c is not None)
                    for f in getattr(client, "fallbacks", ())
                ]
            raise
        finally:
            closed: list[object] = []
            for client in (llm, decider, select_llm):
                close = getattr(client, "aclose", None)
                if callable(close) and not any(client is seen for seen in closed):
                    closed.append(client)
                    await close()
    if admitted is not None:
        log["fallbacks"] = [note.model_dump(mode="json") for note in response.fallbacks]
        log["ran"] = response.model
    log.update(
        entities=len(response.entities),
        matches=sum(len(e.matches) for e in response.entities),
        warnings=len(response.warnings),
        llm_calls=response.usage.llm_calls,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        embedding_requests=response.usage.embedding_requests,
        selector_calls=response.usage.selector_calls,
        selector_cost_usd=response.usage.selector_cost_usd,
        picks=sum(1 for e in response.entities if e.pick is not None and e.pick.id),
        dense=response.retrieval["dense"].status if "dense" in response.retrieval else None,
        timing_ms=response.timing_ms,
    )
    return response.model_dump(mode="json")


def handle(
    method: str,
    path: str,
    headers: Mapping[str, str | Sequence[str]],
    body: bytes,
    *,
    settings: Settings,
    llm_factory: LLMFactory | None = None,
    claude_factory: ClaudeFactory | None = None,
) -> Response:
    """One request. ``headers`` keys in any case; ``path`` with or without a query.

    ``llm_factory`` builds the OpenRouter extraction client and ``claude_factory`` the
    Claude clients (``service.make_claude``); both exist for tests."""
    started = time.monotonic()
    method = method.upper()
    path = urlsplit(path).path or "/"
    lowered = {}
    for name in headers:
        if name.lower() != "authorization":
            value = headers[name]
            lowered[name.lower()] = value if isinstance(value, str) else ", ".join(value)
    extra_secrets: list[str] = []
    log: dict[str, Any] = {
        "method": method if method in KNOWN_METHODS else "(unknown)",
        "path": path if path in ALLOWED else "(unknown)",
        "mode": settings.mode,
        "body_bytes": len(body),
    }
    try:
        result = _dispatch(
            method,
            path,
            lowered,
            body,
            settings,
            llm_factory,
            log,
            claude_factory,
            headers,
            extra_secrets,
        )
    except ServiceError as exc:
        log["code"] = exc.code
        if exc.own_error is not None:
            log["own_error"] = exc.own_error
        allow = ALLOWED.get(path) if exc.code == "method_not_allowed" else None
        extra = {**exc.headers, **({"Allow": ", ".join(allow)} if allow else {})}
        result = _error(exc, settings, extra, extra_secrets)
    except Exception as exc:  # noqa: BLE001 - a 500 with no detail; type and place to the log
        frames = traceback.extract_tb(exc.__traceback__)
        log["code"] = "internal"
        for name in ("own_error", "own_errors"):  # not a provider's: `exception` says what
            log.pop(name, None)
        log["exception"] = type(exc).__name__
        log["where"] = [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames]
        result = _error(
            ServiceError("internal", "internal error; see the function log", status=500),
            settings,
            extra_secrets=extra_secrets,
        )
    log["status"] = result[0]
    log.setdefault("timing_ms", {})["handle"] = round((time.monotonic() - started) * 1000)
    _log(log, settings, extra_secrets)
    return result


def _dispatch(
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    settings: Settings,
    llm_factory: LLMFactory | None,
    log: dict[str, Any],
    claude_factory: ClaudeFactory | None,
    raw_headers: Mapping[str, str | Sequence[str]],
    extra_secrets: list[str],
) -> Response:
    allowed = ALLOWED.get(path)
    if allowed is None:
        raise ServiceError("not_found", "no such endpoint", status=404)
    if method not in allowed:
        raise ServiceError(
            "method_not_allowed",
            f"{method[:16]} is not allowed on {path}; use {' or '.join(allowed)}",
            status=405,
        )
    if path == HEALTH_PATH:
        return 200, _headers(), _json_body(_health(settings, headers), settings)

    if len(body) > settings.limits.max_body_bytes:
        raise ServiceError(
            "body_too_large",
            f"the request body is larger than {settings.limits.max_body_bytes} bytes",
            status=413,
        )
    if headers.get("x-requested-with") != REQUESTED_WITH:
        raise ServiceError(
            "bad_header", f"missing header X-Requested-With: {REQUESTED_WITH}", status=400
        )
    passcode = _passcode_error(settings, headers)
    if passcode is not None:
        raise passcode
    request = _check_encodable(_parse(body))
    # Record only a recognised mode, even when its credential is missing or refused.
    if request.provider in (None, providers.SITE):
        log.update(provider=providers.SITE, credential="site")
    elif request.provider in providers.OWN_PROVIDERS:
        log.update(provider=request.provider, credential="visitor")
    credential: providers.Credential | None = None
    # A deployment without own keys looks at no Authorization header on a request for
    # the site's models, as before there were own keys: HTTP basic auth, `curl -u` and
    # an authenticating proxy all add one. A request that names an own provider is
    # still refused there, and with own keys on a site request must not carry a key.
    if settings.own_providers or request.provider not in (None, providers.SITE):
        credential = providers.read_credential(
            request.provider,
            raw_headers,
            enabled=tuple(info.id for info in settings.own_providers),
            text=request.text,
            prompt=request.prompt,
        )
    if credential is not None:
        key = credential.token()
        extra_secrets.append(key)
        # A value pasted into a forwarding header is not an address to hash into SQL.
        headers = {
            name: headers[name]
            for name in limits.ADDRESS_HEADERS
            if name in headers and key not in headers[name]
        }
    elif not settings.configured:
        raise ServiceError(
            "not_configured",
            "the service has no ANTHROPIC_API_KEY and no OPENROUTER_API_KEY; the operator "
            "must set one (or NER_SITE_OFFLINE=1 for the stand-in extractor)",
            status=503,
        )
    log["text_chars"] = len(request.text)
    log["custom_prompt"] = bool(request.prompt and request.prompt.strip())
    started = time.monotonic()  # the request deadline also covers admission and the indexes
    valid = validate_request(request, settings)
    log["model"] = valid.model
    log["top_k"] = valid.top_k
    log["selector"] = valid.selector
    # Opened (once per warm instance) before admission: a request that cannot run
    # because the indexes cannot be opened must not have been counted as a visit.
    indexes = get_indexes(settings.index_settings())
    admitted = _admit(settings, valid, headers, log)  # last: a refused request is not counted
    data = run_request(
        _ner(
            settings,
            valid,
            llm_factory or default_llm_factory,
            log,
            started=started,
            indexes=indexes,
            admitted=admitted,
            credential=credential,
            claude_factory=claude_factory,
        )
    )
    return 200, _headers(), _json_body(data, settings, extra_secrets)


# --- Vercel: settings from the process environment, and the shim's I/O -----------

_SETTINGS_LOCK = threading.Lock()
_SETTINGS: dict[tuple[str | None, ...], Settings | SettingsError] = {}


def env_settings(env: Mapping[str, str] | None = None) -> Settings | SettingsError:
    """``Settings.from_env`` for the process environment, read once per distinct set of
    values (a warm instance reuses it); a bad value is returned, not raised."""
    source = os.environ if env is None else env
    key = tuple(
        source.get(name) if name != "DATABASE_URL" or uses_database(source) else None
        for name in ENV_NAMES
    )
    with _SETTINGS_LOCK:
        found = _SETTINGS.get(key)
        if found is None:
            try:
                found = Settings.from_env(source)
            except SettingsError as exc:
                found = SettingsError(str(exc))  # cached errors must not retain request frames
            _SETTINGS[key] = found
        return found


def reset() -> None:
    """Forget the settings and the visitor limiter of this instance (tests)."""
    with _SETTINGS_LOCK:
        _SETTINGS.clear()
    with _LIMITS_LOCK:
        _LIMITS.clear()


def _shim_response(request: BaseHTTPRequestHandler, settings: Settings) -> Response:
    """Read the body (at most Content-Length bytes, nothing when it is over the cap)
    and handle the request; refusals before reading are logged like any other."""
    method = request.command.upper()
    cap = settings.limits.max_body_bytes
    raw_length = request.headers.get("Content-Length")
    early: ServiceError | None = None
    length = 0
    if raw_length is None:
        if method == "POST":
            early = ServiceError(
                "length_required", "a POST needs a Content-Length header", status=411
            )
    elif not re.fullmatch(rf"[0-9]{{1,{MAX_LENGTH_DIGITS}}}", raw_length.strip()):
        # ASCII digits only: str.isdigit() also accepts "²", which int() refuses.
        early = ServiceError("bad_header", "Content-Length is not a valid length", status=400)
    else:
        length = int(raw_length.strip())
        if length > cap:
            early = ServiceError(
                "body_too_large", f"the request body is larger than {cap} bytes", status=413
            )
    if early is not None:
        request.close_connection = True
        _log(
            {
                "method": method if method in KNOWN_METHODS else "(unknown)",
                "path": route if (route := urlsplit(request.path).path) in ALLOWED else "(unknown)",
                "status": early.status,
                "code": early.code,
                "body_bytes": length,
            },
            settings,
        )
        return _error(early, settings)
    body = request.rfile.read(length) if length else b""
    headers: dict[str, str | Sequence[str]] = {
        k.lower(): request.headers[k] for k in request.headers if k.lower() != "authorization"
    }
    if method == "POST" and urlsplit(request.path).path == NER_PATH:
        headers["authorization"] = request.headers.get_all("Authorization", [])

    return handle(method, request.path, headers, body, settings=settings)


def serve(request: BaseHTTPRequestHandler) -> None:
    """Answer one request on a ``BaseHTTPRequestHandler`` (``api/*.py``)."""
    settings = env_settings()
    if isinstance(settings, SettingsError):
        error = ServiceError(
            "not_configured", f"the service is misconfigured: {settings}", status=503
        )
        _log(
            {
                "method": request.command if request.command in KNOWN_METHODS else "(unknown)",
                "path": route if (route := urlsplit(request.path).path) in ALLOWED else "(unknown)",
                "status": 503,
                "code": "not_configured",
            },
            None,
        )
        status, headers, body = _error(error, None)
    else:
        status, headers, body = _shim_response(request, settings)
    request.send_response(status)
    for key, value in headers.items():
        request.send_header(key, value)
    request.send_header("Content-Length", str(len(body)))
    if request.close_connection:
        request.send_header("Connection", "close")
    request.end_headers()
    if request.command.upper() != "HEAD":
        request.wfile.write(body)


__all__ = [
    "default_llm_factory",
    "env_settings",
    "handle",
    "make_limits_executor",
    "reset",
    "serve",
]
