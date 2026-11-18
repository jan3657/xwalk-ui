"""Framework-free request handling: routing, the header and passcode rules, the error
envelope, and one log line per request (docs/ner-site/PLAN.md sections 3.1 and 7).

:func:`handle` is what ``api/*.py`` (through :func:`serve`), ``tools/serve_local.py``
and the tests call. It never raises: every outcome is ``(status, headers, body)`` with a
JSON body, ``Cache-Control: no-store`` and no ``Access-Control-*`` header.

Order of checks for ``POST /api/ner``: path (404), method (405; ``OPTIONS`` too, on
purpose, so no preflight ever succeeds), body size (413), ``X-Requested-With:
xwalk-ner`` (400 ``bad_header``), passcode (401), configuration (503), JSON (400), field
limits (400), the visitor's allowance (429 ``visitor_limit_reached``), then the run
(422/429/502). An unexpected exception is a 500 ``internal`` whose log line names only
the exception type and where it was raised: its message could carry the visitor's text.

Admission (docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md
sections 3.3 and 5) comes after every check that can refuse a request, so a refused
request is never counted. It applies where Claude is offered (an Anthropic key, not
offline): the visitor's hourly allowance decides whether a request for Opus starts on
Opus or on Sonnet, and a Claude that is known to be unusable moves the start to the
OpenRouter step. The visitor is a keyed hash of the client address; neither it nor the
address is ever logged or returned.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any, TypeVar
from urllib.parse import urlsplit

from pydantic import ValidationError
from xwalk.llm.base import LLMClient
from xwalk.retrieval.postgres import NeonHTTPExecutor, SQLExecutor

from ner_site import fake, fallback, limits
from ner_site.claude import OPUS, SONNET, is_claude
from ner_site.indexes import get_indexes
from ner_site.ontologies import catalog, full_catalog
from ner_site.schema import (
    FallbackNote,
    HealthResponse,
    NerRequest,
    OntologySummary,
    RetrieverStatus,
    ServiceError,
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
    validate_request,
)

T = TypeVar("T")
Response = tuple[int, dict[str, str], bytes]
LLMFactory = Callable[[Settings, str], LLMClient]

HEALTH_PATH = "/api/health"
NER_PATH = "/api/ner"
ALLOWED = {HEALTH_PATH: ("GET", "HEAD"), NER_PATH: ("POST",)}
REQUESTED_WITH = "xwalk-ner"
MAX_LOGGED_PATH = 80
MAX_LENGTH_DIGITS = 18
REQUEST_WORKERS = 16  # threads per request for retriever searches (BM25 and embeddings)
# One round trip to Neon before the request may run: short, so a slow database becomes
# "limits unavailable" (Sonnet, uncounted) and not a slow request.
LIMITS_TIMEOUT_S = 3.0
LIMITS_UNAVAILABLE = "limits_unavailable"
CLAUDE_UNAVAILABLE = "claude_unavailable"


def _headers(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        **(extra or {}),
    }


def _json_body(data: object, settings: Settings | None) -> bytes:
    text = json.dumps(data, ensure_ascii=False)
    if settings is not None:  # last line of defence: no secret ever leaves
        for secret in settings.secrets():
            text = text.replace(secret, "[redacted]")
    return text.encode("utf-8")


def _error(
    exc: ServiceError, settings: Settings | None, extra: Mapping[str, str] | None = None
) -> Response:
    return exc.status, _headers(extra), _json_body(exc.envelope(), settings)


def _log(record: dict[str, Any], settings: Settings | None) -> None:
    line = json.dumps({"event": "ner_site.request", **record}, ensure_ascii=True)
    if settings is not None:
        for secret in settings.secrets():
            line = line.replace(secret, "[redacted]")
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
    """The limiter, or None where it is off: Claude is not offered (nothing to meter),
    or there is no ``DATABASE_URL``. Kept per database, schema and allowance, so a warm
    instance creates the table once. Raises ``LimitsUnavailable`` when it cannot be set
    up, which is handled like a database that cannot be reached."""
    if not settings.claude_offered or settings.database_url is None:
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
    the step of ``chain_models`` its chain starts on, and the step-downs that decided
    it (stage ``admit``)."""

    tier: str
    start: int = 0
    notes: tuple[FallbackNote, ...] = ()


def _admit(
    settings: Settings, request: NerRequest, headers: Mapping[str, str], log: dict[str, Any]
) -> _Admitted | None:
    """Count the visit and decide where the request starts (spec sections 3.3 and 5).

    None where Claude is not offered: no allowance, no chain, the request runs as it
    always did. Otherwise the wanted tier comes from the model, or from the selector
    when only the selector is Claude, and:

    - limiter on: ``limits.admit`` grants Opus or Sonnet and records the visit, or the
      hour is used (429 ``visitor_limit_reached`` with ``Retry-After``);
    - limiter unreachable: the request is not counted and a request for Opus starts on
      Sonnet;
    - limiter off (no ``DATABASE_URL``): with a passcode Claude is unmetered, without
      one a request for Opus starts on Sonnet, as for unreachable.

    Then a Claude that is known to be unusable (``fallback.claude_block``) moves the
    start to the OpenRouter step, when the chain has one."""
    if not settings.claude_offered:
        return None
    assert request.model is not None
    lead = next((m for m in (request.model, request.selector) if m and is_claude(m)), None)
    wanted: limits.Tier = "opus" if lead == OPUS else "sonnet" if lead == SONNET else "other"
    tier: str = wanted
    reason: str | None = None
    unmetered = False
    try:
        limiter = _visitor_limits(settings)
        if limiter is None:
            unmetered = settings.passcode is None
        else:
            admission = limiter.admit(_visitor(settings, headers), wanted)
            tier, reason = admission.tier, admission.reason
    except limits.VisitorLimitReached as exc:
        minutes = max(1, math.ceil(exc.resets_in_s / 60))
        raise ServiceError(
            "visitor_limit_reached",
            "you have used this hour's requests; try again in "
            f"{minutes} minute{'' if minutes == 1 else 's'}",
            status=429,
            headers={"Retry-After": str(exc.resets_in_s)},
        ) from None
    except limits.LimitsUnavailable as exc:
        log["limits"] = str(exc)  # the error's class and SQLSTATE, nothing else
        unmetered = True
    if unmetered and wanted == "opus":
        tier, reason = "sonnet", LIMITS_UNAVAILABLE
    log["tier"] = tier
    if lead is None:
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
    if fallback.claude_block() is not None and not is_claude(models[last]) and start < last:
        notes.append(
            FallbackNote(
                stage="admit", from_=models[start], to=models[last], reason=CLAUDE_UNAVAILABLE
            )
        )
        start = last
    return _Admitted(tier, start, tuple(notes))


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
        if field == "top_k":
            raise ServiceError("bad_top_k", "'top_k' must be an integer", status=400) from None
        if first["type"] == "extra_forbidden":
            message = f"unknown field {field[:40]!r}; send text, prompt, model and top_k"
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
    Anthropic accepts the key (one free call, remembered)."""
    if settings.anthropic_api_key is None:
        return RetrieverStatus(status="unavailable", reason="key not set")
    blocked = fallback.claude_block()
    if blocked is not None:
        return RetrieverStatus(status="unavailable", reason=BLOCK_TEXT.get(blocked, blocked))
    return check_claude_key(settings) or RetrieverStatus(status="ok")


def _visitor_health(settings: Settings, headers: Mapping[str, str]) -> dict[str, Any]:
    """The calling visitor's allowance as ``limits.peek`` reads it (nothing is
    recorded): ``tier`` is where a request for Opus would start now. With the limiter
    unreachable or off there are no counts: the numbers then say what `_admit` would
    do, Sonnet uncounted, or the whole allowance on a private (passcode) deployment."""
    allowance = settings.allowance
    state = "off"
    try:
        limiter = _visitor_limits(settings)
        if limiter is not None:
            seen = limiter.peek(_visitor(settings, headers))
            return {
                "limits": "on",
                "tier": seen.tier,
                "opus_left": seen.opus_left,
                "sonnet_left": seen.sonnet_left,
                "resets_in_s": seen.resets_in_s,
            }
    except limits.LimitsUnavailable:
        state = "unavailable"
    unmetered = state == "off" and settings.passcode is not None
    now = _now()
    return {
        "limits": state,
        "tier": "opus" if unmetered else "sonnet",
        "opus_left": allowance.opus_per_hour if unmetered else 0,
        "sonnet_left": allowance.sonnet_per_hour,
        "resets_in_s": max(1, math.ceil(limits.HOUR_S - now % limits.HOUR_S)),
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
                else "POST /api/ner answers 503 not_configured"
            ),
        )
    else:
        llm = check_key(settings) or RetrieverStatus(status="ok")
    data["llm"] = llm.model_dump(mode="json")
    # Not in HealthResponse either (spec 2026-11-18 section 6): the Anthropic key's
    # status, and where Claude is offered the calling visitor's allowance. Offline mode
    # spends no key and counts no visitor, so it reports neither.
    if not settings.offline:
        data["claude"] = _claude_health(settings).model_dump(mode="json")
    if settings.claude_offered:
        data["visitor"] = _visitor_health(settings, headers)
    # Not in HealthResponse either: where the terms and indexes come from.
    data["backend"] = settings.backend
    if settings.backend == "neon":
        data["neon"] = {"schema": settings.neon_schema, "catalog": settings.catalog}
    return data


async def _ner(
    settings: Settings,
    valid: NerRequest,
    factory: LLMFactory,
    log: dict[str, Any],
    *,
    started: float,
    admitted: _Admitted | None = None,
    claude_factory: ClaudeFactory | None = None,
) -> dict[str, Any]:
    """Build the clients for a validated, admitted request and run it.

    A request for a Claude model runs on one fallback chain, started where `admitted`
    says; a Claude selector uses that chain, or one of its own (for selection only) when
    the extraction model is an OpenRouter model. Everything else is built as before."""
    assert valid.model is not None
    model, selector = valid.model, valid.selector or RETRIEVAL_ONLY
    indexes = get_indexes(settings.index_settings())
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
    headers: Mapping[str, str],
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
    lowered = {k.lower(): v for k, v in headers.items()}
    log: dict[str, Any] = {
        "method": method,
        "path": path[:MAX_LOGGED_PATH],
        "mode": settings.mode,
        "body_bytes": len(body),
    }
    try:
        result = _dispatch(method, path, lowered, body, settings, llm_factory, log, claude_factory)
    except ServiceError as exc:
        log["code"] = exc.code
        allow = ALLOWED.get(path) if exc.code == "method_not_allowed" else None
        extra = {**exc.headers, **({"Allow": ", ".join(allow)} if allow else {})}
        result = _error(exc, settings, extra)
    except Exception as exc:  # noqa: BLE001 - a 500 with no detail; type and place to the log
        frames = traceback.extract_tb(exc.__traceback__)
        log["code"] = "internal"
        log["exception"] = type(exc).__name__
        log["where"] = [f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}" for f in frames]
        result = _error(
            ServiceError("internal", "internal error; see the function log", status=500),
            settings,
        )
    log["status"] = result[0]
    log.setdefault("timing_ms", {})["handle"] = round((time.monotonic() - started) * 1000)
    _log(log, settings)
    return result


def _dispatch(
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes,
    settings: Settings,
    llm_factory: LLMFactory | None,
    log: dict[str, Any],
    claude_factory: ClaudeFactory | None = None,
) -> Response:
    allowed = ALLOWED.get(path)
    if allowed is None:
        raise ServiceError("not_found", f"no such endpoint: {path[:MAX_LOGGED_PATH]}", status=404)
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
    if not settings.configured:
        raise ServiceError(
            "not_configured",
            "the service has no ANTHROPIC_API_KEY and no OPENROUTER_API_KEY; the operator "
            "must set one (or NER_SITE_OFFLINE=1 for the stand-in extractor)",
            status=503,
        )
    request = _check_encodable(_parse(body))
    log["text_chars"] = len(request.text)
    log["custom_prompt"] = bool(request.prompt and request.prompt.strip())
    started = time.monotonic()  # the request deadline also covers admission and the indexes
    valid = validate_request(request, settings)
    log["model"] = valid.model
    log["top_k"] = valid.top_k
    log["selector"] = valid.selector
    admitted = _admit(settings, valid, headers, log)  # last: a refused request is not counted
    data = run_request(
        _ner(
            settings,
            valid,
            llm_factory or default_llm_factory,
            log,
            started=started,
            admitted=admitted,
            claude_factory=claude_factory,
        )
    )
    return 200, _headers(), _json_body(data, settings)


# --- Vercel: settings from the process environment, and the shim's I/O -----------

_SETTINGS_LOCK = threading.Lock()
_SETTINGS: dict[tuple[str | None, ...], Settings | SettingsError] = {}


def env_settings(env: Mapping[str, str] | None = None) -> Settings | SettingsError:
    """``Settings.from_env`` for the process environment, read once per distinct set of
    values (a warm instance reuses it); a bad value is returned, not raised."""
    source = os.environ if env is None else env
    key = tuple(source.get(name) for name in ENV_NAMES)
    with _SETTINGS_LOCK:
        found = _SETTINGS.get(key)
        if found is None:
            try:
                found = Settings.from_env(source)
            except SettingsError as exc:
                found = exc
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
                "method": method,
                "path": urlsplit(request.path).path[:MAX_LOGGED_PATH],
                "status": early.status,
                "code": early.code,
                "body_bytes": length,
            },
            settings,
        )
        return _error(early, settings)
    body = request.rfile.read(length) if length else b""
    headers = {k.lower(): v for k, v in request.headers.items()}
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
                "method": request.command,
                "path": urlsplit(request.path).path[:MAX_LOGGED_PATH],
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
