"""Framework-free request handling: routing, the header and passcode rules, the error
envelope, and one log line per request (docs/ner-site/PLAN.md sections 3.1 and 7).

:func:`handle` is what ``api/*.py`` (through :func:`serve`), ``tools/serve_local.py``
and the tests call. It never raises: every outcome is ``(status, headers, body)`` with a
JSON body, ``Cache-Control: no-store`` and no ``Access-Control-*`` header.

Order of checks for ``POST /api/ner``: path (404), method (405; ``OPTIONS`` too, on
purpose, so no preflight ever succeeds), body size (413), ``X-Requested-With:
xwalk-ner`` (400 ``bad_header``), passcode, JSON, credential, site configuration,
provider-specific validation, the response cache, indexes and finally admission before
any chat client is built. Site-key Claude deployments apply the hourly and daily allowances and
refuse a request if the counters cannot be consulted. Own-key runs have separate
flood limits and can proceed during a counter outage. The visitor is a
keyed hash of the client address; neither it nor the address is ever logged or returned.

With ``NER_SITE_STATS_PASSCODE`` set (site analytics design, sections 3 and 4), every
``POST /api/ner`` that got past the passcode and body parsing is recorded as one row of
``usage.SiteRuns`` once its response is built, and ``GET /api/stats`` (its own passcode,
no other check) reads the rows back. Unset, neither exists: nothing is built or sent,
and ``/api/stats`` is the 404 of any unknown path.

The response cache (``ner_site.cache``). A validated request the site pays for, whose
text, prompt, model, selector and ``top_k`` are those of a recent clean run of this warm
instance on the same index view, is answered with that run's response: no model is
called, nothing is searched or embedded, and it is looked up before admission, so it
does not use up one of the visitor's runs. Everything that refuses a request before
admission still refuses it first, because the lookup comes after all of it. The key is
built once per request, from the indexes that request opened (remembered by a warm
instance, so a hit pays nothing for them), and the same value is used to look the answer
up and to store it. A request on a visitor's own key is never looked up and never
stored: its provider answers it, or refuses it. A hit is not served while a provider the
stored answer needs is known to refuse the operator's key (``service.known_refusal``;
the request then proceeds as a fresh one would). Only the response of a wholly clean run
is stored (``service.cacheable``), as it was sent; a hit says so in ``cache``
(``{"response": "hit"}`` and nothing more) and reports its own ``timing_ms`` and
``usage``. How old the answer was is in the log line only.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
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
from urllib.parse import parse_qs, urlsplit

from pydantic import ValidationError
from xwalk.llm.base import LLMCapabilities, LLMClient, LLMRequest, LLMResponse, ParseError
from xwalk.llm.parsing import parse_json_object
from xwalk.records import Usage
from xwalk.retrieval.postgres import NeonHTTPExecutor, SQLExecutor

from ner_site import cache as caching
from ner_site import fake, fallback, limits, providers, usage
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
    cacheable,
    chain_models,
    check_claude_key,
    check_key,
    known_refusal,
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
STATS_PATH = "/api/stats"
ALLOWED = {HEALTH_PATH: ("GET", "HEAD"), NER_PATH: ("POST",)}
STATS_METHODS = ("GET",)
STATS_PASSCODE_HEADER = "x-stats-passcode"
COUNTRY_HEADER = "x-vercel-ip-country"
REQUESTED_WITH = "xwalk-ner"
KNOWN_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
MAX_LENGTH_DIGITS = 18
REQUEST_WORKERS = 16  # threads per request for retriever searches (BM25 and embeddings)
# One round trip to Neon before the request may run: short, so a slow database becomes
# a prompt refusal on site keys, or an uncounted own-key run. It is twice the
# limiter's own bound on waiting for a visitor's lock (limits.LOCK_TIMEOUT_MS), so a
# queue of one visitor's requests ends as a contended lock (429), not as this timeout.
LIMITS_TIMEOUT_S = 3.0
# Recording a run happens while the visitor waits for an answer that is already built:
# one round trip, and never longer than this. Reading the statistics is the operator's
# own request and may wait for a database that has to wake up.
USAGE_TIMEOUT_S = 2.0
STATS_TIMEOUT_S = 10.0
# The recording writes one warm instance may have under way at once. A write whose wait
# ran out is not stopped: it keeps its thread and its connection until the executor's
# own timeout ends it, so a database that hangs would otherwise hold one of each for
# every request. A healthy write takes tens of milliseconds and its request waits for
# it, so four is never reached by a site that works, and bounds one that does not.
USAGE_MAX_IN_FLIGHT = 4
USAGE_SATURATED = "saturated"
STATS_UNAVAILABLE = "stats_unavailable"
CONTENDED_RETRY_AFTER_S = 5
LIMITS_UNAVAILABLE = "limits_unavailable"
CLAUDE_UNAVAILABLE = "claude_unavailable"


# The SDK logs response headers and both HTTP clients log the reason phrase, even
# when a provider echoes credentials there. Filters on the emitting loggers run
# before any handler can format an exception. The context carries only a boolean.
_OWN_RUN = ContextVar("ner_site_own_run", default=False)
# The same for an exchange with the database of the visitor limiter or of the record of
# runs (``_Unlogged``): the connection string is the credential there, and a server, or
# something in front of it, that echoes it in a reason phrase would put it in the log.
_DATABASE_CALL = ContextVar("ner_site_database_call", default=False)


def _outside_own_run(record: logging.LogRecord) -> bool:
    return not _OWN_RUN.get() and not _DATABASE_CALL.get()


# The HTTP stack of the engine's Neon executor. Its loggers are not named one by one:
# the packages have more than the ones that show in an ordinary exchange (through a
# proxy, ``httpcore.proxy`` logs the host it is asked to reach).
_HTTP_STACK = ("httpx", "httpcore")
_HUSHED = 0  # how many loggers there were when the stack's were last looked for


def _hush_http_stack() -> None:
    """Put the filter on every logger under ``httpx`` and ``httpcore`` that exists by
    now, the two themselves included. A filter on a logger does not reach its
    children, so each one gets it; this runs again before every database exchange,
    for loggers made since, and looks only when there are more loggers than before."""
    global _HUSHED
    loggers = logging.root.manager.loggerDict
    if len(loggers) == _HUSHED:
        return
    for name in _HTTP_STACK:
        logging.getLogger(name)  # so that the top of each hierarchy is a logger too
    for name, found in list(loggers.items()):
        if isinstance(found, logging.Logger) and name.split(".", 1)[0] in _HTTP_STACK:
            found.addFilter(_outside_own_run)  # once: a filter is not added twice
    _HUSHED = len(loggers)


class _Unlogged:
    """An ``SQLExecutor`` whose exchanges the HTTP libraries' loggers say nothing
    about (``_outside_own_run``). The mark is set for the length of a call, in the
    thread that makes it, and nowhere else."""

    def __init__(self, inner: SQLExecutor) -> None:
        self._inner = inner

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        _hush_http_stack()
        token = _DATABASE_CALL.set(True)
        try:
            return self._inner.execute(sql, params)
        finally:
            _DATABASE_CALL.reset(token)

    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]:
        _hush_http_stack()
        token = _DATABASE_CALL.set(True)
        try:
            return self._inner.transaction(statements, read_only=read_only)
        finally:
            _DATABASE_CALL.reset(token)


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
_hush_http_stack()


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


def make_usage_executor(database_url: str, timeout: float) -> SQLExecutor:
    """The executor the record of runs is written and read with: the limiter's kind,
    built the same way (an insert is no more idempotent than a counted visit, so no
    retries), with a timeout of its own. It is the engine's client, the one the site
    already runs against Neon. Its timeout is the HTTP library's, for each step of an
    exchange and not for the exchange: see ``_record`` for what that leaves. Tests
    replace it."""
    return NeonHTTPExecutor(database_url, timeout=timeout, max_retries=0)


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
                executor = _Unlogged(make_limits_executor(settings.database_url))
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


# --- the record of runs: one store per warm instance, database and timeout ----------

_USAGE_LOCK = threading.Lock()
_USAGE: dict[tuple[str, str, float], usage.SiteRuns] = {}


class _Flights:
    """The recording writes of this instance that have not ended, at most ``limit``."""

    def __init__(self, limit: int) -> None:
        self._lock = threading.Lock()
        self._limit = limit
        self.count = 0

    def enter(self) -> bool:
        """Take a place for one write; False, at once, when every place is taken."""
        with self._lock:
            if self.count >= self._limit:
                return False
            self.count += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self.count -= 1


_USAGE_FLIGHTS = _Flights(USAGE_MAX_IN_FLIGHT)


def _usage_store(settings: Settings, timeout: float, deadline: float) -> usage.SiteRuns | None:
    """Where runs are kept, or None when they are not: the stats passcode is unset, or
    there is no database to keep them in. `deadline` (``time.monotonic()``) is the
    caller's: the lock the instance's requests share is waited for no longer than to
    it, and ``TimeoutError`` says that it was not had in time. Raises ``ValueError``
    for a connection string or a schema that cannot be used."""
    if settings.stats_passcode is None or settings.database_url is None:
        return None
    key = (settings.database_url, settings.neon_schema, timeout)
    if not _USAGE_LOCK.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise TimeoutError("the store of runs was not free in time")
    try:
        found = _USAGE.get(key)
        if found is None:
            executor = _Unlogged(make_usage_executor(settings.database_url, timeout))
            found = _USAGE[key] = usage.SiteRuns(executor, settings.neon_schema, clock=_now)
        return found
    finally:
        _USAGE_LOCK.release()


def _time_left(deadline: float) -> float:
    """The seconds to `deadline`; ``TimeoutError`` when there are none, so that what
    has run out of time does not start."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("no time was left")
    return left


def _plain_address(headers: Mapping[str, str]) -> dict[str, str]:
    """The address header of a request whose credential has not been read, for the
    day's visitor of its row. It is the header the visitor id is made from (the first
    of ``limits.ADDRESS_HEADERS`` with a value, as for an admitted request), kept only
    when that value is an IP address: a request that is then refused for its credential
    is counted under its own address, and what a client put there instead of one (a
    key, pasted in the wrong place) is not hashed into a row."""
    for name in limits.ADDRESS_HEADERS:
        value = headers.get(name) or ""
        first = (value.split(",", 1)[0] if name == "x-forwarded-for" else value).strip()
        if not first:
            continue
        try:
            ipaddress.ip_address(first)
        except ValueError:
            return {}
        return {name: value}
    return {}


def _known_models(settings: Settings) -> frozenset[str]:
    """The model and selector ids this deployment lists: what the row of a run may
    name (``usage.run_of`` adds the ids every deployment knows)."""
    return frozenset(
        {
            *settings.models,
            *settings.offered_models,
            *settings.selectors,
            settings.fallback_model,
            *(model for info in settings.own_providers for model in info.models),
        }
    )


def _failed_calls(recorded: dict[str, Any] | None, exc: BaseException) -> None:
    """Keep, for the row of a run that is being recorded, what the model calls of a
    failed request are known to have used (``ServiceError.usage``): the sum over every
    model its chain tried, which the row books under the model the request asked for
    (``usage.run_of``). Tokens only when a call reported them: a failure that
    reported none used an unknown amount, not zero."""
    carried = getattr(exc, "usage", None)
    if not recorded or not isinstance(carried, Usage) or carried.calls <= 0:
        return
    reported = carried.calls > carried.unknown_calls
    recorded["failed"] = {
        "llm_calls": carried.calls,
        "prompt_tokens": carried.prompt_tokens if reported else None,
        "completion_tokens": carried.completion_tokens if reported else None,
    }


def _allowed(path: str, settings: Settings) -> tuple[str, ...] | None:
    """The methods of a path, None for a path that does not exist. ``/api/stats``
    exists only on a deployment with the stats passcode set."""
    if path == STATS_PATH:
        return STATS_METHODS if settings.stats_passcode is not None else None
    return ALLOWED.get(path)


def _record(
    settings: Settings,
    log: dict[str, Any],
    recorded: Mapping[str, Any],
    country: str | None,
) -> None:
    """Keep one row for a finished run. Called with the response already built, and
    nothing here can change it: every failure ends as ``log["usage"]``, the name of
    what went wrong.

    What the request waits for has one deadline: ``USAGE_TIMEOUT_S`` from the moment
    this is called, on the monotonic clock. Everything that can wait happens in a
    daemon thread of its own, and this waits for that thread no longer than to the
    deadline. In the thread, the lock the store is found behind is waited for no longer
    than to it either, and a write whose turn comes after it does not start.

    A write that has started cannot be stopped. When the request stops waiting the
    write is left to the executor's own timeouts (``make_usage_executor``): they end
    it promptly against a database that is down or says nothing, and they do not end
    it against one that keeps sending an answer very slowly, because they limit each
    read and not the whole. Until it ends it holds one of ``USAGE_MAX_IN_FLIGHT``
    places; with all of them taken a run is not recorded and does not wait
    (``usage: "saturated"``). So the worst a database can do is keep four daemon
    threads of an instance busy and pause its recording until the connections end or
    the instance is recycled. No answer is changed or delayed beyond the one wait."""
    deadline = time.monotonic() + USAGE_TIMEOUT_S
    try:
        if settings.stats_passcode is None or settings.database_url is None:
            return
        now = _now()
        salt = settings.visitor_salt or limits.default_salt(settings.database_url or "")
        row = usage.run_of(
            log,
            at=now,
            visitor_day=usage.visitor_day(_visitor(settings, recorded["address"]), salt, now),
            country=country,
            selector_tokens=recorded.get("selector_tokens", (None, None)),
            models=_known_models(settings),
            failed=recorded.get("failed"),
        )
        if not _USAGE_FLIGHTS.enter():
            log["usage"] = USAGE_SATURATED
            return
        # How the write ended: None, the name of an Exception, or what is no Exception
        # (a test's own failure, an interpreter exit), which is handed on as it is.
        ended: list[str | BaseException | None] = []

        def send() -> None:
            try:
                store = _usage_store(settings, USAGE_TIMEOUT_S, deadline)
                if store is not None:
                    _time_left(deadline)  # too late: the write does not start
                    store.record(row)
                ended.append(None)
            except Exception as exc:  # noqa: BLE001 - named in the log, never raised
                ended.append(getattr(exc, "kind", None) or type(exc).__name__)
            except BaseException as exc:  # noqa: BLE001 - raised again by the request
                ended.append(exc)
            finally:
                _USAGE_FLIGHTS.leave()

        thread = threading.Thread(target=send, name="ner-site-usage", daemon=True)
        try:
            thread.start()
        except BaseException:
            _USAGE_FLIGHTS.leave()
            raise
        thread.join(max(0.0, deadline - time.monotonic()))
        outcome = ended[0] if ended else TimeoutError.__name__
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not None:
            log["usage"] = outcome
    except Exception as exc:  # noqa: BLE001 - recording never costs the visitor an answer
        log["usage"] = type(exc).__name__


def _stats(
    settings: Settings, headers: Mapping[str, str], query: str, log: dict[str, Any]
) -> Response:
    """``GET /api/stats?days=N``: the stats passcode, then the bounds on ``days``, then
    one read. The visitor passcode, ``X-Requested-With`` and the limiter play no part."""
    assert settings.stats_passcode is not None
    given = headers.get(STATS_PASSCODE_HEADER, "")
    if not given:
        raise ServiceError(
            "stats_passcode_required", "the statistics need their passcode", status=401
        )
    if not hmac.compare_digest(given.encode("utf-8"), settings.stats_passcode.encode("utf-8")):
        raise ServiceError(
            "stats_passcode_invalid", "the stats passcode is not correct", status=401
        )
    days = usage.DEFAULT_DAYS
    asked = parse_qs(query, keep_blank_values=True).get("days")
    if asked is not None:
        # ASCII digits without a sign, a space or a leading zero, and only once.
        if len(asked) != 1 or not re.fullmatch(r"[1-9][0-9]{0,2}", asked[0], re.ASCII):
            days = 0
        else:
            days = int(asked[0])
        if not 1 <= days <= usage.MAX_DAYS:
            raise ServiceError(
                "bad_days",
                f"'days' must be a whole number from 1 to {usage.MAX_DAYS}",
                status=400,
            )
    log["days"] = days
    deadline = time.monotonic() + STATS_TIMEOUT_S  # for the wait for the store's lock
    try:
        store = _usage_store(settings, STATS_TIMEOUT_S, deadline)
        if store is None:
            raise usage.UsageUnavailable("site runs have no database")
        data = store.stats(days, models=_known_models(settings))
    except Exception as exc:  # noqa: BLE001 - whatever kept the statistics from being read
        # The store exposes only the class and SQLSTATE. Anything else is named by its
        # class alone: an executor that could not be built (a connection string it
        # refuses, a certificate file that is not there) may quote the URL or a path.
        log["stats"] = str(exc) if isinstance(exc, usage.UsageUnavailable) else type(exc).__name__
        raise ServiceError(
            STATS_UNAVAILABLE,
            "the statistics cannot be read right now; try again shortly",
            status=503,
        ) from None
    return 200, _headers(), _json_body(data, settings)


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
    When there was more than one class, ``own_errors`` lists them all, sorted: every
    one, so that the class ``own_error`` names is always among them (they are class
    names of this code and its libraries, a handful at most).
    """

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
            log["own_errors"] = seen
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
    caches: caching.Caches | None = None,
    reused: dict[str, int] | None = None,
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
            caches=caches,
            reused=reused,
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
    recorded: dict[str, Any] | None = None,
    caches: caching.Caches | None = None,
    private: Sequence[str] = (),
) -> tuple[dict[str, Any], bool]:
    """Build the clients for a validated, admitted request and run it on `indexes`.
    Returns the response as JSON data and whether it may be kept in the response
    cache (``service.cacheable``, the one rule, judged here on the typed response and
    on what the run counted). `private` holds what belongs to this request and must
    not occur in a stored response (:func:`_private`).

    A request for a Claude model runs on one fallback chain, started where `admitted`
    says; a Claude selector uses that chain, or one of its own (for selection only) when
    the extraction model is an OpenRouter model. Everything else is built as before."""
    assert valid.model is not None
    model, selector = valid.model, valid.selector or RETRIEVAL_ONLY
    reused: dict[str, int] = {}
    if credential is not None:
        token = _OWN_RUN.set(True)
        try:
            response = await _run_own(
                credential, valid, settings, indexes, started, log, caches, reused
            )
        except ServiceError as exc:
            _failed_calls(recorded, exc)
            raise
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
                caches=caches,
                reused=reused,
            )
        except BaseException as exc:
            _failed_calls(recorded, exc)
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
    storable = False
    if response.cache is not None:
        # The counts of what was reused are for the operator: the log line, never the
        # response.
        log["cache"] = {
            "response": response.cache.response,
            "retrieval_hits": reused.get("retrieval_hits", 0),
            "pick_hits": reused.get("pick_hits", 0),
        }
        # The whole run was clean (the one rule), and it was not an own-key one.
        storable = credential is None and cacheable(
            response, indexes, settings=settings, extra_secrets=private, run=reused
        )
    if recorded:  # a run that is being recorded: for its row only, never the log line
        recorded["selector_tokens"] = (
            response.usage.selector_prompt_tokens,
            response.usage.selector_completion_tokens,
        )
    return response.model_dump(mode="json"), storable


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
    target = urlsplit(path)
    path = target.path or "/"
    lowered = {}
    for name in headers:
        if name.lower() != "authorization":
            value = headers[name]
            lowered[name.lower()] = value if isinstance(value, str) else ", ".join(value)
    extra_secrets: list[str] = []
    # What the row of a run needs and the log line must not hold (`_dispatch` fills it
    # once a request has become a run; empty, the request is not recorded).
    recorded: dict[str, Any] = {}
    log: dict[str, Any] = {
        "method": method if method in KNOWN_METHODS else "(unknown)",
        "path": path if _allowed(path, settings) else "(unknown)",
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
            target.query,
            recorded,
        )
    except ServiceError as exc:
        log["code"] = exc.code
        if exc.own_error is not None:
            log["own_error"] = exc.own_error
        allow = _allowed(path, settings) if exc.code == "method_not_allowed" else None
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
    if "address" in recorded:  # after the response is built; `result` is returned as it is
        _record(settings, log, recorded, lowered.get(COUNTRY_HEADER))
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
    query: str = "",
    recorded: dict[str, Any] | None = None,
) -> Response:
    allowed = _allowed(path, settings)
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
    if path == STATS_PATH:
        return _stats(settings, headers, query, log)

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
    if recorded is not None and settings.stats_passcode is not None:
        # From here on the request is a run and gets a row. Until the credential step
        # has looked at the headers (below) its visitor is the plain address it came
        # from; a request refused at that step keeps it.
        recorded["address"] = _plain_address(headers)
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
    if recorded:  # only ever filled above, with the stats passcode set
        recorded["address"] = headers
    if credential is None and not settings.configured:
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
    # The response cache, after every check that can refuse the request and before
    # admission: an answer this instance already has costs nothing and is not counted
    # as one of the visitor's runs. Only for a request the site pays for: one on a
    # visitor's own key is never answered from the cache and never stored in it, so
    # that key's provider is always the one that answers (or refuses).
    caches = caching.caches_for(settings.cache_scope, settings.cache, family=settings.cache_family)
    own = credential is not None or valid.provider not in (None, providers.SITE)
    shared = caches is not None and caches.responses.enabled and not own
    # Opened once per warm instance and remembered, so this costs a stored answer
    # nothing; before admission, because a request that cannot run for want of its
    # indexes must not have been counted as a visit.
    indexes = get_indexes(settings.index_settings())
    # The key of this request, built once, from the indexes this request opened, and
    # kept here: the lookup below and the store after the run use this value. Nothing
    # shared is read again when the answer is stored, so a request that ran on one
    # view cannot leave its answer under the view another request opened meanwhile.
    key: str | None = None
    if caches is not None and shared:
        assert valid.prompt is not None and valid.model is not None
        key = caching.response_key(
            text=valid.text,
            prompt=valid.prompt,
            model=valid.model,
            selector=valid.selector or RETRIEVAL_ONLY,
            top_k=valid.top_k,
            indexes=caching.index_identity(indexes),
        )
        found = caches.responses.get(key)
        if found is not None:
            # A stored answer was made by the models that were asked for. While a
            # provider it needs is known to refuse the operator's key (a refused key,
            # an exhausted balance) the request is not answered from the cache: it
            # proceeds as a fresh one would, which fails or falls back and says so.
            # Read from what this instance already remembers, no provider is asked;
            # and only once there is an answer to hand out (admission reads the block
            # itself).
            assert valid.model is not None
            refused = known_refusal(
                settings,
                valid.model,
                valid.selector,
                embeds=any(index.encoder is not None for index in indexes.values()),
            )
            if refused is not None:
                log["cache_skipped"] = refused
            else:
                data = _from_cache(found[0], found[1], started, log)
                return 200, _headers(), _json_body(data, settings, extra_secrets)
    admitted = _admit(settings, valid, headers, log)  # last: a refused request is not counted
    data, storable = run_request(
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
            recorded=recorded,
            caches=caches,
            private=_private(headers, raw_headers),
        )
    )
    # Redacted once, here, so that what is kept is exactly what is sent.
    safe = _redact(data, (*extra_secrets, *settings.secrets()))
    body = json.dumps(safe, ensure_ascii=False)
    if caches is not None:
        # Whether it may be kept was judged on the typed response, before redaction
        # (`_ner`, ``service.cacheable``). Behind that rule, a response that redaction
        # changed held a secret and is not kept either.
        stored = key is not None and storable and safe == data and caches.responses.put(key, body)
        if "cache" in log:
            log["cache"]["stored"] = bool(stored)
    return 200, _headers(), body.encode("utf-8")


def _private(
    headers: Mapping[str, str], raw_headers: Mapping[str, str | Sequence[str]]
) -> tuple[str, ...]:
    """What belongs to one request and must not occur in a stored response: the client
    address in each address header (for ``x-forwarded-for`` every entry), the passcode
    the request sent, and whatever came in an ``Authorization`` header (a key this
    deployment did not use: a request that runs on a visitor's key is never stored at
    all), with and without its scheme. A response in which one of them occurs is
    served as it is and not kept."""
    values: list[str] = []
    for name in limits.ADDRESS_HEADERS:
        values += [part.strip() for part in (headers.get(name) or "").split(",")]
    values.append((headers.get("x-passcode") or "").strip())
    for name, given in raw_headers.items():  # names in any case, one value or several
        if name.lower() != "authorization":
            continue
        for value in (given,) if isinstance(given, str) else given:
            values += [value.strip(), value.strip().partition(" ")[2].strip()]
    return tuple(dict.fromkeys(v for v in values if v))


def _from_cache(stored: str, age: float, started: float, log: dict[str, Any]) -> dict[str, Any]:
    """The response of a request answered from the response cache: the stored answer,
    parsed anew (so nothing a request does to it reaches the next one), with what is
    this request's own put in place of the original run's. ``usage`` says that no
    call was made and ``timing_ms`` how long the lookup took. ``cache`` says that it
    was a hit and nothing more: how old the answer is, and what the run that made it
    took and used, would tell one visitor about another's run. The age goes to the log
    line, which gets the same counts as a run's, with zero calls."""
    data: dict[str, Any] = json.loads(stored)
    entities = data.get("entities") or []
    timing = {"total": round((time.monotonic() - started) * 1000)}
    data["cache"] = {"response": "hit"}
    data["usage"] = {
        "llm_calls": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "embedding_requests": 0,
        "selector_calls": 0,
        "selector_prompt_tokens": None,
        "selector_completion_tokens": None,
        "selector_cost_usd": None,
    }
    data["timing_ms"] = timing
    dense = (data.get("retrieval") or {}).get("dense")
    log.update(
        entities=len(entities),
        matches=sum(len(e.get("matches") or ()) for e in entities),
        warnings=len(data.get("warnings") or ()),
        llm_calls=0,
        prompt_tokens=None,
        completion_tokens=None,
        embedding_requests=0,
        selector_calls=0,
        selector_cost_usd=None,
        picks=sum(1 for e in entities if (e.get("pick") or {}).get("id")),
        dense=dense.get("status") if isinstance(dense, dict) else None,
        timing_ms=dict(timing),
        cache={"response": "hit", "age_s": int(age)},
    )
    return data


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
    """Forget the settings, the visitor limiter and the caches of this instance (tests)."""
    with _SETTINGS_LOCK:
        _SETTINGS.clear()
    with _LIMITS_LOCK:
        _LIMITS.clear()
    with _USAGE_LOCK:
        _USAGE.clear()
    caching.reset()


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
                "path": route
                if _allowed(route := urlsplit(request.path).path, settings)
                else "(unknown)",
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
    # Every line of a repeated X-Forwarded-For, in order (`handle` joins them with a
    # comma, as a proxy would have): the visitor is still its first entry, and the
    # storage check (`_private`) sees every address the request carried.
    forwarded = request.headers.get_all("X-Forwarded-For", [])
    if len(forwarded) > 1:
        headers["x-forwarded-for"] = forwarded
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
    "make_usage_executor",
    "reset",
    "serve",
]
