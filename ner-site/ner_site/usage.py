"""One row per run of the demo, and the statistics the operator reads back from them
(site analytics design, sections 3 and 4).

Why. The handler already works out, for its one log line per request, which model ran,
whether it stepped down, whose key paid, the tokens and the timings. A log line is gone
in days; a row is not. This module keeps those values, and only those: the row has no
text, prompt, entity, term id, address, visitor id, passcode, key or user agent, because
``run_of`` copies named, typed fields out of the log record and nothing else, and a
field of the wrong type becomes null. No string in the row is free text either: a
model or selector is an id the deployment itself lists, an error code and a step-down
reason are words this site defines, and anything else is stored as the one word
``other`` (the model that answered is named by the provider's response, which is not
ours to trust with a column; a known id with a dated or versioned suffix of digits
and hyphens is stored as the known id).

Who a visitor is. ``visitor_day`` is a keyed hash of the visitor id (itself a keyed hash
of the address, ``limits.visitor_id``) and the UTC date. Distinct values per day count
that day's visitors; the same visitor has another value the next day, so rows cannot be
joined across days or back to an address, and the visitor id is never stored.

Where the rows live. ``<schema>.site_runs`` in the Neon schema the indexes and the
visitor allowance already use, reached through the same ``SQLExecutor``. As for the
allowance, a serverless instance has no memory worth trusting and every call is an HTTPS
round trip, so each call here is exactly one:

- ``record`` sends one ``INSERT``. Its one parameter is the row as a JSON document,
  which Postgres reads into the table's own column types (``jsonb_populate_record``): a
  null stays a null and no value is ever part of the SQL text.
- ``stats`` sends one ``SELECT`` that answers with one JSON document: the per-day
  counts (zero-filled with ``generate_series``, distinct visitors, the median and the
  95th percentile of the run time), the token sums per day, credential, model and
  requested selector, and the counts per error code, step-down reason and country
  (a null country is the runs that came without one).
- The first call of an instance puts ``CREATE TABLE IF NOT EXISTS`` and the index on
  ``at`` in front, in the same transaction: still one round trip. Two cold instances
  that do this at the same moment on a database without the table can collide in
  Postgres's catalog; the loser's transaction is rolled back and its one row is lost
  (noted in its log line, like any failed write). ``create`` makes the table without
  a row, and ``tools/setup_usage.py`` runs it once for the operator, before recording
  is turned on, so that no instance ever has anything to create.

Days are UTC days whatever the session's time zone: a row's day is
``(at AT TIME ZONE 'UTC')::date`` and the series is built from dates, not instants.

What the SQL leaves to Python is the money (``shape``): the price table lives here, in
code, and an estimate is plain arithmetic over the token sums. Only runs on the site's
keys are priced. Extraction tokens are priced at the model that answered, selection
tokens at the selector that was asked for when it is a model, and a cost the provider
itself reported (Jev) is added where the selector has no price of its own, on a row of
``by_model`` named after that selector (``jev``), so that the rows' dollars add up to
the total. A run that failed after a model had answered keeps the tokens its failure
carried, under the model the request asked for (which can only overstate it: a chain
steps down to cheaper models), and is priced like any other. A run on the
site's keys counts once in ``unpriced_runs`` when either of its stages has no price:
the model that extracted, or a model selector whose provider reported no cost. It is
an estimate; the provider's console is the bill.

When the database cannot be asked, every failure becomes ``UsageUnavailable`` with a
message made here from the error's class and SQLSTATE only. The executor's own text is
dropped (it may name the host). Recording a run is not idempotent, so the caller gives
this an executor that does not retry, and decides what a failure may cost: nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any

from xwalk.retrieval.postgres import SQLExecutor, Statement

from ner_site import fallback, limits
from ner_site.claude import CLAUDE_MODELS, HAIKU
from ner_site.providers import OWN_PROVIDERS
from ner_site.select import JEV, RETRIEVAL_ONLY

__all__ = [
    "CODES",
    "COLUMNS",
    "DEFAULT_DAYS",
    "MAX_DAYS",
    "OTHER",
    "PRICES",
    "REASONS",
    "TABLE",
    "Run",
    "SiteRuns",
    "UsageUnavailable",
    "country_of",
    "run_of",
    "shape",
    "visitor_day",
]

TABLE = "site_runs"
DEFAULT_DAYS = 30
MAX_DAYS = 365
# US dollars per million tokens, (input, output), for runs on the site's keys. A model
# that is not here is counted in ``unpriced_runs`` and adds nothing to the estimate.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}
# Dollars are rounded to this many places: fine enough that a priced model with even
# one token never shows as exactly zero, which the page reads as "no price".
USD_DIGITS = 6
SITE = "site"
TIERS = ("opus", "sonnet", "other", "own")
# Selectors that are not a model, so that have no list price to be missing.
_NOT_MODELS = frozenset({JEV, RETRIEVAL_ONLY})

# What a string of the row is stored as when it is not one the site defines.
OTHER = "other"
# The model and selector ids every deployment knows; ``run_of`` is given the rest (the
# deployment's own lists).
_MODELS = frozenset({*CLAUDE_MODELS, HAIKU, JEV, RETRIEVAL_ONLY})
_CREDENTIALS = frozenset({SITE, "visitor"})
_PROVIDERS = frozenset({SITE, *OWN_PROVIDERS})
# Every error code a response of this site can carry (``ServiceError`` and its
# subclasses, wherever they are raised; tests/test_site_usage.py reads the source for
# them, so a new code that is missing here fails a test instead of becoming ``other``).
CODES = frozenset(
    {
        "bad_credential",
        "bad_days",
        "bad_header",
        "bad_json",
        "bad_top_k",
        "body_too_large",
        "call_budget_exhausted",
        "extraction_schema",
        "extraction_unparseable",
        "internal",
        "length_required",
        "limits_unavailable",
        "method_not_allowed",
        "missing_text",
        "not_configured",
        "not_found",
        "passcode_invalid",
        "passcode_required",
        "prompt_too_long",
        "provider_error",
        "provider_key_refused",
        "provider_rate_limited",
        "stats_passcode_invalid",
        "stats_passcode_required",
        "stats_unavailable",
        "text_too_long",
        "unknown_model",
        "unknown_selector",
        "visitor_limit_reached",
    }
)
# Every reason a step-down note can give: the chain's (``ner_site.fallback``), the
# limiter's, and the two the handler and the service write themselves
# (``http.CLAUDE_UNAVAILABLE``, ``service.FOLLOWS_EXTRACTION``; both import this module,
# so they are spelled out here and compared in the tests).
REASONS = frozenset(
    {
        *fallback.STEP_DOWN_REASONS,
        fallback.TRANSIENT,
        fallback.CALL_LIMIT,
        limits.OPUS_USED,
        "claude_unavailable",
        "follows_extraction",
    }
)

_IDENT = re.compile(r"[a-z_][a-z0-9_]{0,62}")
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")
_COUNTRY = re.compile(r"[A-Z]{2}")
_UNDEFINED_TABLE = "42P01"  # the SQLSTATE of a statement on a table that is not there
# What may follow a known model id and its hyphen: a date or a version.
_SUFFIX = re.compile(r"[0-9][0-9-]{0,15}")


@dataclass(frozen=True)
class Run:
    """One row of ``site_runs``: the fields, in the table's column order."""

    at: str
    visitor_day: str
    status: int
    code: str | None = None
    credential: str = SITE
    provider: str = SITE
    tier: str | None = None
    requested_model: str | None = None
    model: str | None = None
    requested_selector: str | None = None
    fallback_reasons: str = ""
    country: str | None = None
    text_chars: int | None = None
    custom_prompt: bool | None = None
    entities: int | None = None
    matches: int | None = None
    picks: int | None = None
    llm_calls: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    selector_calls: int | None = None
    selector_prompt_tokens: int | None = None
    selector_completion_tokens: int | None = None
    selector_cost_usd: float | None = None
    embedding_requests: int | None = None
    extract_ms: int | None = None
    map_ms: int | None = None
    select_ms: int | None = None
    total_ms: int | None = None


COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Run))

# Every statement starts with a tag, like the engine's own and the limiter's, so a log
# or a fake executor can tell them apart. `{table}` is the validated schema and the
# fixed table name; every value is a `$n` parameter.
_CREATE = """/* xwalk:site_runs_table */
CREATE TABLE IF NOT EXISTS {table} (
    at timestamptz NOT NULL,
    visitor_day text NOT NULL,
    status integer NOT NULL,
    code text,
    credential text NOT NULL,
    provider text NOT NULL,
    tier text,
    requested_model text,
    model text,
    requested_selector text,
    fallback_reasons text NOT NULL,
    country text,
    text_chars integer,
    custom_prompt boolean,
    entities integer,
    matches integer,
    picks integer,
    llm_calls integer,
    prompt_tokens integer,
    completion_tokens integer,
    selector_calls integer,
    selector_prompt_tokens integer,
    selector_completion_tokens integer,
    selector_cost_usd double precision,
    embedding_requests integer,
    extract_ms integer,
    map_ms integer,
    select_ms integer,
    total_ms integer
)"""

# Only ``create`` sends this (the operator's provisioning). The site itself never
# does: its schema is the one the indexes and the allowance already live in.
_SCHEMA = """/* xwalk:site_runs_schema */
CREATE SCHEMA IF NOT EXISTS {schema}"""

_INDEX = """/* xwalk:site_runs_index */
CREATE INDEX IF NOT EXISTS site_runs_at ON {table} (at)"""

# $1 the row, a JSON object with one key per column. Postgres reads it into the table's
# row type, so each value is converted by its column and a JSON null is a NULL.
_INSERT = f"""/* xwalk:site_run */
INSERT INTO {{table}} ({", ".join(COLUMNS)})
SELECT {", ".join(COLUMNS)}
FROM jsonb_populate_record(NULL::{{table}}, $1::jsonb)"""

# $1 the first UTC day, $2 the last (dates, both included). One row, one column: the
# whole answer as a JSON document, cast to text so that every executor returns it alike.
_STATS = f"""/* xwalk:site_runs_stats */
WITH runs AS (
    SELECT r.*, (r.at AT TIME ZONE 'UTC')::date AS day
    FROM {{table}} AS r
    WHERE r.at >= (($1::date)::timestamp AT TIME ZONE 'UTC')
      AND r.at < (($2::date + 1)::timestamp AT TIME ZONE 'UTC')
), days AS (
    SELECT g::date AS day
    FROM generate_series(($1::date)::timestamp, ($2::date)::timestamp, interval '1 day') AS g
), per_day AS (
    SELECT day,
           count(*) AS runs,
           count(*) FILTER (WHERE status = 200) AS ok,
           count(DISTINCT visitor_day) AS visitors,
           {", ".join(f"count(*) FILTER (WHERE tier = '{t}') AS {t}" for t in TIERS)},
           count(*) FILTER (WHERE fallback_reasons <> '') AS stepped_down,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY total_ms) AS p50_ms,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms) AS p95_ms
    FROM runs
    GROUP BY day
), spend AS (
    SELECT day, credential, model, requested_selector AS selector,
           count(*) AS runs,
           coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
           coalesce(sum(completion_tokens), 0) AS completion_tokens,
           coalesce(sum(selector_prompt_tokens), 0) AS selector_prompt_tokens,
           coalesce(sum(selector_completion_tokens), 0) AS selector_completion_tokens,
           coalesce(sum(selector_cost_usd), 0) AS selector_cost_usd,
           count(*) FILTER (
               WHERE selector_calls > 0 AND selector_cost_usd IS NULL
           ) AS unreported_selections
    FROM runs
    GROUP BY day, credential, model, requested_selector
), codes AS (
    SELECT code, count(*) AS runs FROM runs WHERE code IS NOT NULL GROUP BY code
), reasons AS (
    SELECT reason, count(*) AS runs
    FROM runs CROSS JOIN LATERAL unnest(string_to_array(fallback_reasons, ',')) AS reason
    WHERE fallback_reasons <> ''
    GROUP BY reason
), countries AS (
    SELECT country, count(*) AS runs FROM runs GROUP BY country
)
SELECT jsonb_build_object(
    'by_day', (
        SELECT coalesce(jsonb_agg(jsonb_build_object(
                   'day', d.day,
                   'runs', coalesce(p.runs, 0),
                   'ok', coalesce(p.ok, 0),
                   'visitors', coalesce(p.visitors, 0),
                   {", ".join(f"'{t}', coalesce(p.{t}, 0)" for t in TIERS)},
                   'stepped_down', coalesce(p.stepped_down, 0),
                   'p50_ms', coalesce(round(p.p50_ms::numeric), 0),
                   'p95_ms', coalesce(round(p.p95_ms::numeric), 0)
               ) ORDER BY d.day), '[]'::jsonb)
        FROM days AS d LEFT JOIN per_day AS p ON p.day = d.day
    ),
    'spend', (SELECT coalesce(jsonb_agg(to_jsonb(s)), '[]'::jsonb) FROM spend AS s),
    'by_code', (
        SELECT coalesce(jsonb_agg(to_jsonb(c) ORDER BY c.runs DESC, c.code), '[]'::jsonb)
        FROM codes AS c
    ),
    'by_fallback', (
        SELECT coalesce(jsonb_agg(to_jsonb(f) ORDER BY f.runs DESC, f.reason), '[]'::jsonb)
        FROM reasons AS f
    ),
    'by_country', (
        SELECT coalesce(jsonb_agg(to_jsonb(n) ORDER BY n.runs DESC, n.country), '[]'::jsonb)
        FROM countries AS n
    )
)::text AS stats"""


class UsageUnavailable(Exception):
    """The database could not be queried, or answered with something unreadable. The
    message names the error's class and its SQLSTATE, nothing the executor wrote.
    ``kind`` is that class name and ``code`` the SQLSTATE (None when there was none)."""

    def __init__(self, message: str, *, kind: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind or type(self).__name__
        self.code = code


def _unavailable(exc: Exception) -> UsageUnavailable:
    """What an executor's failure is raised as: its class and SQLSTATE, not its text."""
    code = getattr(exc, "code", None)
    state = code if isinstance(code, str) and _SQLSTATE.fullmatch(code) else None
    return UsageUnavailable(
        "site runs could not be queried "
        f"({type(exc).__name__}{'' if state is None else ' ' + state})",
        kind=type(exc).__name__,
        code=state,
    )


def visitor_day(visitor: str, salt: str, now: float) -> str:
    """The visitor of one UTC day: the first 32 hex characters of
    ``HMAC-SHA256(salt, visitor id + date)``. ``visitor`` is ``limits.visitor_id`` and
    ``now`` the time in seconds since the epoch."""
    day = datetime.fromtimestamp(now, tz=timezone.utc).date().isoformat()
    return hmac.new(
        salt.encode("utf-8"), (visitor + day).encode("utf-8"), hashlib.sha256
    ).hexdigest()[:32]


def country_of(value: str | None) -> str | None:
    """The ``x-vercel-ip-country`` header when it is a two-letter code, else None."""
    return value if isinstance(value, str) and _COUNTRY.fullmatch(value) else None


def _whole(value: Any) -> int | None:
    return value if type(value) is int else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _one_of(value: Any, known: Collection[str]) -> str | None:
    """`value` when it is a string the site defines, ``other`` for any other string
    that is not empty, None for what is no string at all."""
    if not isinstance(value, str) or not value:
        return None
    return value if value in known else OTHER


def _model_of(value: Any, known: Collection[str]) -> str | None:
    """`value` as a model or selector id of the row: a known id as it is; a known id
    followed by a hyphen and a date or version made of digits and hyphens (a provider
    may answer as ``claude-opus-5-5-20261101`` when asked for ``claude-opus-5-5``) as
    that known id; ``other`` for any other string that is not empty. The longest
    known id that fits is taken, so an id that is the start of another does not
    claim the other's snapshots. Nothing of the suffix is kept, and nothing but
    digits and hyphens can be one."""
    found = _one_of(value, known)
    if found != OTHER:
        return found
    for base in sorted(known, key=len, reverse=True):
        if value.startswith(base + "-") and _SUFFIX.fullmatch(value, len(base) + 1):
            return base
    return OTHER


def run_of(
    log: Mapping[str, Any],
    *,
    at: float,
    visitor_day: str,
    country: str | None = None,
    selector_tokens: tuple[int | None, int | None] = (None, None),
    models: Collection[str] = (),
    failed: Mapping[str, Any] | None = None,
) -> Run:
    """The row of a finished request, from the record the handler logs (``http.handle``)
    and what that record does not hold: the day's visitor, the country, the selection
    step's tokens and, for a run that failed, what its failed calls carried. ``at`` is
    the time in seconds since the epoch.

    ``models`` are the model and selector ids this deployment lists, beside the ones
    every deployment knows. A model or selector that is one of them with a date or
    version suffix is stored as that id (``_model_of``); one that is none of them is
    stored as ``other``, like an error code or a step-down reason the site does not define: the
    model that answered comes out of the provider's response, and a row holds no text
    that came from anywhere but this site. A tier that is none of the four is null
    (``other`` is one of the four).

    ``model`` is the model that produced the extraction (as in the response), so it is
    null for a run that did not get that far. ``failed`` (``llm_calls``,
    ``prompt_tokens``, ``completion_tokens``) is read for a run that did not end in a
    200: its calls are counted, and when any of them reported tokens those are kept
    under the model the request asked for, so that a failure that was paid for is
    priced. The tokens are the sum over every model the chain tried, and nothing says
    which model used which; the chain only steps down to cheaper models, so the
    requested model's price can overstate such a run and never understate it."""
    known = _MODELS | frozenset(models)
    status = _whole(log.get("status")) or 0
    requested = _model_of(log.get("model"), known)
    timing = log.get("timing_ms")
    timing = timing if isinstance(timing, Mapping) else {}
    notes = log.get("fallbacks")
    reasons = [
        _one_of(note.get("reason"), REASONS)
        for note in (notes if isinstance(notes, list) else ())
        if isinstance(note, Mapping)
    ]
    cost = log.get("selector_cost_usd")
    custom = log.get("custom_prompt")
    tier = log.get("tier")
    ran = _model_of(log.get("ran"), known) or (requested if status == 200 else None)
    calls = _whole(log.get("llm_calls"))
    prompt_tokens = _whole(log.get("prompt_tokens"))
    completion_tokens = _whole(log.get("completion_tokens"))
    if status != 200 and isinstance(failed, Mapping):
        calls = _whole(failed.get("llm_calls"))
        prompt_tokens = _whole(failed.get("prompt_tokens"))
        completion_tokens = _whole(failed.get("completion_tokens"))
        if prompt_tokens is not None or completion_tokens is not None:
            ran = requested
    return Run(
        at=datetime.fromtimestamp(at, tz=timezone.utc).isoformat(),
        visitor_day=visitor_day,
        status=status,
        code=_one_of(log.get("code"), CODES),
        credential=_one_of(log.get("credential"), _CREDENTIALS) or SITE,
        provider=_one_of(log.get("provider"), _PROVIDERS) or SITE,
        tier=tier if isinstance(tier, str) and tier in TIERS else None,
        requested_model=requested,
        model=ran,
        requested_selector=_model_of(log.get("selector"), known),
        fallback_reasons=",".join(dict.fromkeys(r for r in reasons if r is not None)),
        country=country_of(country),
        text_chars=_whole(log.get("text_chars")),
        custom_prompt=custom if isinstance(custom, bool) else None,
        entities=_whole(log.get("entities")),
        matches=_whole(log.get("matches")),
        picks=_whole(log.get("picks")),
        llm_calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        selector_calls=_whole(log.get("selector_calls")),
        selector_prompt_tokens=_whole(selector_tokens[0]),
        selector_completion_tokens=_whole(selector_tokens[1]),
        selector_cost_usd=float(cost) if type(cost) in (int, float) else None,
        embedding_requests=_whole(log.get("embedding_requests")),
        extract_ms=_whole(timing.get("extract")),
        map_ms=_whole(timing.get("map")),
        select_ms=_whole(timing.get("select")),
        total_ms=_whole(timing.get("total")),
    )


def _count(value: Any) -> int:
    """A count or a sum as the database sent it: a number, or a string of one."""
    return int(round(float(value)))


def _usd(model: str | None, prompt: int, completion: int) -> float:
    price = PRICES.get(model or "")
    return 0.0 if price is None else (prompt * price[0] + completion * price[1]) / 1_000_000


def shape(
    raw: Any, *, days: int, generated_at: float, models: Collection[str] = ()
) -> dict[str, Any]:
    """The answer of ``GET /api/stats`` (design section 4) from the document the
    statistics statement returns (a JSON string, or the object an executor parsed it
    into). Raises ``KeyError``, ``TypeError`` or ``ValueError`` on a document that is
    not that statement's.

    Every string of the answer goes through the mapping a row's strings go through
    when it is written (``run_of``; ``models`` as there): a row may be older than that
    rule or written by hand, and what it holds must not reach the answer either.
    Groups that become the same word are added up."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Mapping):
        raise TypeError("the statistics document is not an object")
    known = _MODELS | frozenset(models)

    day_usd: dict[str, float] = {}
    models: dict[tuple[str, str], dict[str, Any]] = {}
    prompt_tokens = completion_tokens = unpriced = 0

    def row_of(model: str | None, credential: str) -> dict[str, Any]:
        return models.setdefault(
            (model, credential),
            {
                "model": model,
                "credential": credential,
                "runs": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_usd": 0.0,
            },
        )

    for group in raw["spend"]:
        day = str(group["day"])
        credential = _one_of(group["credential"], _CREDENTIALS) or OTHER
        model = _model_of(group["model"], known)
        selector = _model_of(group["selector"], known)
        runs = _count(group["runs"])
        prompt, completion = _count(group["prompt_tokens"]), _count(group["completion_tokens"])
        picked_prompt = _count(group["selector_prompt_tokens"])
        picked_completion = _count(group["selector_completion_tokens"])
        reported = float(group["selector_cost_usd"])
        # The group's runs whose selection ran and whose provider reported no cost.
        selections = _count(group["unreported_selections"])
        site = credential == SITE
        prompt_tokens += prompt + picked_prompt
        completion_tokens += completion + picked_completion
        spent = 0.0
        is_model = selector is not None and selector not in _NOT_MODELS
        # A run that reached no model (refused, or failed first) is in the row whose
        # model is null, so that the rows' runs add up to every run.
        row = row_of(model, credential)
        row["runs"] += runs
        row["prompt_tokens"] += prompt
        row["completion_tokens"] += completion
        if site and model is not None:
            cost = _usd(model, prompt, completion)
            row["estimated_usd"] += cost
            spent += cost
        if site:
            # Once per run, whichever stage it is that has no price: every run of an
            # extraction model without one, else the runs whose model selector has
            # none and reported no cost of its own.
            if model is not None and model not in PRICES:
                unpriced += runs
            elif is_model and selector not in PRICES:
                unpriced += selections
        # Never both: a selector with a price is priced by its tokens, any other by
        # what its provider reported.
        extra = reported if site and selector not in PRICES else 0.0
        # Jev is no model, but what it reported is shown on a row of its own name, so
        # that the rows' dollars add up to the total.
        named = selector is not None and selector != RETRIEVAL_ONLY
        if named and (picked_prompt or picked_completion or extra):
            row = row_of(selector, credential)
            row["prompt_tokens"] += picked_prompt
            row["completion_tokens"] += picked_completion
            if site:
                cost = _usd(selector, picked_prompt, picked_completion) + extra
                row["estimated_usd"] += cost
                spent += cost
        else:
            spent += extra
        day_usd[day] = day_usd.get(day, 0.0) + spent

    by_day = [
        {
            "day": str(found["day"]),
            "runs": _count(found["runs"]),
            "ok": _count(found["ok"]),
            "visitors": _count(found["visitors"]),
            **{tier: _count(found[tier]) for tier in TIERS},
            "stepped_down": _count(found["stepped_down"]),
            "estimated_usd": round(day_usd.get(str(found["day"]), 0.0), USD_DIGITS),
            "p50_ms": _count(found["p50_ms"]),
            "p95_ms": _count(found["p95_ms"]),
        }
        for found in raw["by_day"]
    ]
    runs_total = sum(day["runs"] for day in by_day)
    ok_total = sum(day["ok"] for day in by_day)
    by_model = sorted(
        models.values(),
        key=lambda row: (
            -row["runs"],
            row["model"] is None,
            row["model"] or "",
            row["credential"],
        ),
    )
    for row in by_model:
        row["estimated_usd"] = round(row["estimated_usd"], USD_DIGITS)

    def counted(name: str, key: str, word: Callable[[Any], str | None]) -> list[dict[str, Any]]:
        found: dict[str | None, int] = {}
        for item in raw[name]:
            value = word(item[key])
            found[value] = found.get(value, 0) + _count(item["runs"])
        # As the statement orders them: most runs first, then by name, null last.
        ordered = sorted(found.items(), key=lambda kv: (-kv[1], kv[0] is None, kv[0] or ""))
        return [{key: value, "runs": runs} for value, runs in ordered]

    stamp = datetime.fromtimestamp(generated_at, tz=timezone.utc)
    return {
        "ok": True,
        "days": days,
        "generated_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals": {
            "runs": runs_total,
            "ok": ok_total,
            "errors": runs_total - ok_total,
            # Visitor-days: one visitor on two days counts twice.
            "visitors": sum(day["visitors"] for day in by_day),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_usd": round(sum(day_usd.values()), USD_DIGITS),
            "unpriced_runs": unpriced,
        },
        "by_day": by_day,
        "by_model": by_model,
        "by_code": counted("by_code", "code", lambda v: _one_of(v, CODES) or OTHER),
        "by_fallback": counted("by_fallback", "reason", lambda v: _one_of(v, REASONS) or OTHER),
        "by_country": counted("by_country", "country", country_of),
    }


class SiteRuns:
    """The runs of the demo, kept in ``<schema>.site_runs``.

    One instance per warm serverless instance: it remembers that it has created the
    table. ``clock`` gives the time in seconds since the epoch; it is a parameter so
    that tests can set the day.
    """

    def __init__(
        self, executor: SQLExecutor, schema: str, *, clock: Callable[[], float] = time.time
    ) -> None:
        if not _IDENT.fullmatch(schema):
            raise ValueError(f"schema {schema!r} must match {_IDENT.pattern}")
        self._executor = executor
        self._clock = clock
        table = f"{schema}.{TABLE}"
        self._schema = _SCHEMA.format(schema=schema)
        self._create = _CREATE.format(table=table)
        self._index = _INDEX.format(table=table)
        self._insert = _INSERT.format(table=table)
        self._stats = _STATS.format(table=table)
        self._created = False

    def create(self) -> None:
        """Make the schema, the table and its index where they are missing: one round
        trip, and the same result however often it runs. For provisioning before
        recording is turned on (``tools/setup_usage.py``). Raises ``UsageUnavailable``
        when the database cannot be asked."""
        try:
            self._executor.transaction([(self._schema, ()), (self._create, ()), (self._index, ())])
        except Exception as exc:
            raise _unavailable(exc) from None
        self._created = True

    def record(self, run: Run) -> None:
        """Store one run: one round trip. Raises ``UsageUnavailable`` when the database
        cannot be asked; the caller must not let that reach the visitor."""
        document = json.dumps(asdict(run), ensure_ascii=False, allow_nan=False)
        self._send((self._insert, (document,)))

    def stats(self, days: int = DEFAULT_DAYS, *, models: Collection[str] = ()) -> dict[str, Any]:
        """The statistics of the last ``days`` UTC days, today included (1 to
        ``MAX_DAYS``, else ``ValueError``): one round trip. ``models`` are the ids the
        deployment lists (``run_of``): a model the table names that is none of them
        is ``other`` in the answer. Raises ``UsageUnavailable`` when the database
        cannot be asked or its answer cannot be read."""
        if type(days) is not int or not 1 <= days <= MAX_DAYS:
            raise ValueError(f"days must be a whole number from 1 to {MAX_DAYS}")
        now = self._clock()
        last = datetime.fromtimestamp(now, tz=timezone.utc).date()
        first = last - timedelta(days=days - 1)
        rows = self._send((self._stats, (first.isoformat(), last.isoformat())))
        try:
            answer = shape(rows[0]["stats"], days=days, generated_at=now, models=models)
            if len(answer["by_day"]) != days:
                raise ValueError("the statistics do not cover the days asked for")
        except (LookupError, TypeError, ValueError):
            raise UsageUnavailable("site runs answered with an unreadable row") from None
        return answer

    def _send(self, statement: Statement) -> list[dict[str, Any]]:
        """One round trip: the statement, behind the table's creation when this instance
        has not sent it yet. Returns the statement's rows."""
        try:
            if self._created:
                rows = self._executor.execute(*statement)
            else:
                lead: list[Statement] = [(self._create, ()), (self._index, ())]
                rows = self._executor.transaction([*lead, statement])[-1]
        except Exception as exc:
            failure = _unavailable(exc)
            if failure.code == _UNDEFINED_TABLE:
                # The table this instance created is gone (dropped to delete the data).
                # This statement is lost; the next one creates the table again.
                self._created = False
            raise failure from None
        self._created = True
        return rows
