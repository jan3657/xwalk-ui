"""One row per run, and the statistics read back from them (site analytics design,
sections 3 and 4).

Two layers, as for the visitor allowance, because the aggregation lives in SQL.

Offline (the default run): ``FakeRuns`` is an in-memory stand-in for the ``site_runs``
table. It tells the module's four statements apart by their ``/* xwalk:<tag> */``, keeps
the rows that were inserted, refuses a statement that arrives before the table exists,
and answers the statistics statement with the meaning of its SQL, computed here in
Python. These tests prove the Python around the SQL (the row, what is sent, the price
arithmetic, the shape of the answer, the bounds on ``days``); they cannot prove the SQL.

Against Postgres (``integration`` marker): the same fixed set of rows goes through the
real statements and must give the same hand-computed answer. As for the limiter, these
need a Postgres URL in ``NER_SITE_TEST_DATABASE_URL`` and the ``psql`` client (16 or
later), and skip without either:

    NER_SITE_TEST_DATABASE_URL=postgresql://user@localhost:5432/postgres \\
        .venv/bin/python -m pytest -q -m integration tests/test_site_usage.py

Run them with ``PGTZ=Europe/Ljubljana`` as well: the days are UTC days whatever the
session's time zone is. Each test creates a schema of its own and drops it.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import secrets
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from ner_site import usage
from ner_site.usage import (
    COLUMNS,
    PRICES,
    TABLE,
    Run,
    SiteRuns,
    UsageUnavailable,
    country_of,
    run_of,
    shape,
    visitor_day,
)
from xwalk.retrieval.postgres import PostgresError, SQLExecutor, tag_of

from tests.test_site_limits import DATABASE_URL_ENV, PsqlExecutor

# 18 November 2026, 14:20:00 UTC.
NOW = datetime(2026, 11, 18, 14, 20, tzinfo=timezone.utc).timestamp()
SENTINEL_URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"
SALT = "test-salt"
VISITOR = "a" * 32
OTHER_VISITOR = "b" * 32
OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
GEMINI = "google/gemini-3.1-flash-lite"


def _percentile(values: list[int], fraction: float) -> float:
    """``percentile_cont``: linear interpolation between the two nearest ranks."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class FakeRuns:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.created = False
        self.indexed = False
        self.sent: list[tuple[str, list[Any]]] = []
        self.round_trips = 0
        self.fail: Exception | None = None

    # --- the executor protocol
    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        self.round_trips += 1
        return self._run(sql, params)

    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]:
        self.round_trips += 1
        before = (list(self.rows), self.created, self.indexed)
        try:
            return [self._run(sql, params) for sql, params in statements]
        except Exception:
            self.rows, self.created, self.indexed = before  # all or nothing
            raise

    def tags(self) -> list[str | None]:
        return [tag_of(sql) for sql, _ in self.sent]

    def _run(self, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        p = list(params)
        self.sent.append((sql, p))
        if self.fail is not None:
            raise self.fail
        placeholders = {int(n) for n in re.findall(r"\$(\d+)", sql)}
        assert placeholders == set(range(1, len(p) + 1)), "placeholders and parameters differ"
        tag = tag_of(sql)
        if tag not in ("site_runs_schema", "site_runs_table") and not self.created:
            raise PostgresError('relation "site_runs" does not exist', code="42P01")
        return getattr(self, f"_{tag}")(sql, p)

    # --- the statements
    def _site_runs_schema(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "CREATE SCHEMA IF NOT EXISTS" in sql
        return []

    def _site_runs_table(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "CREATE TABLE IF NOT EXISTS" in sql
        self.created = True
        return []

    def _site_runs_index(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "CREATE INDEX IF NOT EXISTS" in sql and "(at)" in sql
        self.indexed = True
        return []

    def _site_run(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        (document,) = p
        row = json.loads(document)
        assert set(row) == set(COLUMNS)
        row["at"] = datetime.fromisoformat(row["at"]).astimezone(timezone.utc)
        self.rows.append(row)
        return []

    def _site_runs_stats(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        first, last = (date.fromisoformat(value) for value in p)
        runs = [row for row in self.rows if first <= row["at"].date() <= last]
        by_day = []
        day = first
        while day <= last:
            rows = [row for row in runs if row["at"].date() == day]
            timed = [row["total_ms"] for row in rows if row["total_ms"] is not None]
            by_day.append(
                {
                    "day": day.isoformat(),
                    "runs": len(rows),
                    "ok": sum(row["status"] == 200 for row in rows),
                    "visitors": len({row["visitor_day"] for row in rows}),
                    **{
                        tier: sum(row["tier"] == tier for row in rows)
                        for tier in ("opus", "sonnet", "other", "own")
                    },
                    "stepped_down": sum(row["fallback_reasons"] != "" for row in rows),
                    "p50_ms": round(_percentile(timed, 0.5)) if timed else 0,
                    "p95_ms": round(_percentile(timed, 0.95)) if timed else 0,
                }
            )
            day += timedelta(days=1)
        spend: dict[tuple[Any, ...], dict[str, Any]] = {}
        sums = (
            "prompt_tokens",
            "completion_tokens",
            "selector_prompt_tokens",
            "selector_completion_tokens",
            "selector_cost_usd",
        )
        for row in runs:
            key = (
                row["at"].date().isoformat(),
                row["credential"],
                row["model"],
                row["requested_selector"],
            )
            group = spend.setdefault(
                key,
                {
                    **dict(zip(("day", "credential", "model", "selector"), key, strict=True)),
                    "runs": 0,
                    "unreported_selections": 0,
                    **dict.fromkeys(sums, 0),
                },
            )
            group["runs"] += 1
            group["unreported_selections"] += bool(
                (row["selector_calls"] or 0) > 0 and row["selector_cost_usd"] is None
            )
            for name in sums:
                group[name] += row[name] or 0

        def counted(name: str, values: list[Any]) -> list[dict[str, Any]]:
            found = {value: values.count(value) for value in values}
            ordered = sorted(
                found.items(), key=lambda item: (-item[1], item[0] is None, item[0] or "")
            )
            return [{name: value, "runs": count} for value, count in ordered]

        stats = {
            "by_day": by_day,
            "spend": list(spend.values()),
            "by_code": counted("code", [row["code"] for row in runs if row["code"] is not None]),
            "by_fallback": counted(
                "reason",
                [
                    reason
                    for row in runs
                    if row["fallback_reasons"]
                    for reason in row["fallback_reasons"].split(",")
                ],
            ),
            "by_country": counted("country", [row["country"] for row in runs]),
        }
        return [{"stats": json.dumps(stats)}]


def make(executor: Any | None = None, *, now: float = NOW) -> tuple[SiteRuns, FakeRuns]:
    fake = FakeRuns() if executor is None else executor
    return SiteRuns(fake, "xwalk", clock=lambda: now), fake


def at(day: int, hour: int = 0, minute: int = 0, second: int = 0) -> str:
    return datetime(2026, 11, day, hour, minute, second, tzinfo=timezone.utc).isoformat()


def a_run(**over: Any) -> Run:
    base = Run(
        at=at(18, 10),
        visitor_day=visitor_day(VISITOR, SALT, NOW),
        status=200,
        credential="site",
        provider="site",
    )
    return replace(base, **over)


# --- the visitor of a day -----------------------------------------------------------


def test_visitor_day_is_stable_within_a_utc_day_and_differs_across_days_salts_and_visitors() -> (
    None
):
    midnight = datetime(2026, 11, 18, tzinfo=timezone.utc).timestamp()
    today = visitor_day(VISITOR, SALT, NOW)
    assert re.fullmatch(r"[0-9a-f]{32}", today)
    assert today == visitor_day(VISITOR, SALT, midnight) == visitor_day(VISITOR, SALT, NOW + 3600)
    assert today == visitor_day(VISITOR, SALT, midnight + 86399)
    assert today != visitor_day(VISITOR, SALT, midnight - 1)
    assert today != visitor_day(VISITOR, SALT, midnight + 86400)
    assert today != visitor_day(VISITOR, "another-salt", NOW)
    assert today != visitor_day(OTHER_VISITOR, SALT, NOW)
    assert VISITOR not in today and today != VISITOR


def test_visitor_day_is_the_keyed_hash_of_the_visitor_and_the_utc_date() -> None:
    import hashlib
    import hmac

    expected = hmac.new(SALT.encode(), (VISITOR + "2026-11-18").encode(), hashlib.sha256)
    assert visitor_day(VISITOR, SALT, NOW) == expected.hexdigest()[:32]


@pytest.mark.parametrize(
    "given,expected",
    [
        ("SI", "SI"),
        ("US", "US"),
        ("si", None),
        ("SVN", None),
        ("S", None),
        ("S1", None),
        ("", None),
        (None, None),
        ("SI\n", None),
        ("SI, US", None),
    ],
)
def test_only_a_two_letter_upper_case_country_is_kept(
    given: str | None, expected: str | None
) -> None:
    assert country_of(given) == expected


# --- the row -------------------------------------------------------------------------

OK_LOG: dict[str, Any] = {
    "method": "POST",
    "path": "/api/ner",
    "mode": "live",
    "body_bytes": 512,
    "provider": "site",
    "credential": "site",
    "text_chars": 33,
    "custom_prompt": True,
    "model": OPUS,
    "top_k": 5,
    "selector": OPUS,
    "tier": "sonnet",
    "fallbacks": [
        {"stage": "admit", "from": OPUS, "to": SONNET, "reason": "opus_allowance_used"},
        {"stage": "extract", "from": SONNET, "to": GEMINI, "reason": "rate_limited"},
        {"stage": "select", "from": SONNET, "to": GEMINI, "reason": "rate_limited"},
    ],
    "ran": GEMINI,
    "entities": 3,
    "matches": 12,
    "warnings": 1,
    "llm_calls": 2,
    "prompt_tokens": 412,
    "completion_tokens": 88,
    "embedding_requests": 2,
    "selector_calls": 3,
    "selector_cost_usd": 0.0125,
    "picks": 2,
    "dense": "ok",
    "timing_ms": {"extract": 1800, "map": 900, "select": 1400, "total": 4200, "handle": 4210},
    "status": 200,
}


def test_the_row_of_a_successful_run_is_read_from_the_log_fields() -> None:
    row = run_of(
        OK_LOG,
        at=NOW,
        visitor_day="f" * 32,
        country="SI",
        selector_tokens=(700, 70),
        models=(GEMINI,),
    )
    assert asdict(row) == {
        "at": "2026-11-18T14:20:00+00:00",
        "visitor_day": "f" * 32,
        "status": 200,
        "code": None,
        "credential": "site",
        "provider": "site",
        "tier": "sonnet",
        "requested_model": OPUS,
        "model": GEMINI,
        "requested_selector": OPUS,
        "fallback_reasons": "opus_allowance_used,rate_limited",
        "country": "SI",
        "text_chars": 33,
        "custom_prompt": True,
        "entities": 3,
        "matches": 12,
        "picks": 2,
        "llm_calls": 2,
        "prompt_tokens": 412,
        "completion_tokens": 88,
        "selector_calls": 3,
        "selector_prompt_tokens": 700,
        "selector_completion_tokens": 70,
        "selector_cost_usd": 0.0125,
        "embedding_requests": 2,
        "extract_ms": 1800,
        "map_ms": 900,
        "select_ms": 1400,
        "total_ms": 4200,
    }
    assert tuple(f.name for f in fields(Run)) == COLUMNS


def test_a_successful_run_without_a_chain_ran_on_the_model_it_asked_for() -> None:
    log = {k: v for k, v in OK_LOG.items() if k not in ("ran", "fallbacks", "tier")}
    row = run_of(log, at=NOW, visitor_day="f" * 32)
    assert (row.requested_model, row.model, row.tier, row.fallback_reasons) == (
        OPUS,
        OPUS,
        None,
        "",
    )
    assert (row.country, row.selector_prompt_tokens, row.selector_completion_tokens) == (
        None,
        None,
        None,
    )


def test_the_row_of_a_refused_run_has_nulls_for_what_did_not_happen() -> None:
    log = {
        "method": "POST",
        "path": "/api/ner",
        "provider": "openai",
        "credential": "visitor",
        "text_chars": 12,
        "custom_prompt": False,
        "model": "gpt-6-luna",
        "top_k": 5,
        "selector": "retrieval",
        "code": "visitor_limit_reached",
        "status": 429,
        "timing_ms": {"handle": 12},
    }
    row = run_of(log, at=NOW, visitor_day="f" * 32, country=None, models=("gpt-6-luna",))
    assert (row.status, row.code, row.credential, row.provider) == (
        429,
        "visitor_limit_reached",
        "visitor",
        "openai",
    )
    assert (row.requested_model, row.model, row.requested_selector) == (
        "gpt-6-luna",
        None,
        "retrieval",
    )
    assert row.custom_prompt is False and row.text_chars == 12 and row.fallback_reasons == ""
    for name in ("tier", "entities", "llm_calls", "prompt_tokens", "total_ms", "extract_ms"):
        assert getattr(row, name) is None, name


def test_a_log_field_of_the_wrong_type_is_dropped_not_stored() -> None:
    """The row has only what the handler computed: a field that is not what the column
    holds (and so anything a future log field might carry) becomes null."""
    log = {
        **OK_LOG,
        "entities": "three",
        "prompt_tokens": True,
        "custom_prompt": "yes",
        "timing_ms": {"total": 4200.5, "extract": None},
        "selector_cost_usd": "0.5",
        "tier": 7,
        "fallbacks": [{"reason": 5}, "x", {"reason": "rate_limited"}],
    }
    row = run_of(log, at=NOW, visitor_day="f" * 32, country="si")
    assert row.entities is None and row.prompt_tokens is None and row.custom_prompt is None
    assert row.total_ms is None and row.extract_ms is None and row.selector_cost_usd is None
    assert row.tier is None and row.country is None and row.fallback_reasons == "rate_limited"


# --- the row holds only strings the site defines (fix wave, A3) ----------------------

SENTINEL = "SENTINEL-upstream-text"


def test_the_ids_the_site_always_knows_need_no_list() -> None:
    for model in (OPUS, SONNET, "claude-haiku-5-5", "jev", "retrieval"):
        log = {**OK_LOG, "model": model, "ran": model, "selector": model}
        row = run_of(log, at=NOW, visitor_day="f" * 32)
        assert (row.requested_model, row.model, row.requested_selector) == (model,) * 3


@pytest.mark.parametrize(
    "upstream",
    [
        f"claude-opus-5-5-{SENTINEL}",
        SENTINEL,
        "claude-opus-5-5-latest",
        "claude-opus-5-5-2026x",
        "claude-opus-5-5/20261101",
        "claude-opus-5-5-",
        "claude-opus-5-5--1",
        "claude-opus-5-5-20261101\n",
        "claude-opus-5-5-20261101-" + "1" * 9,
        "claude-opus-5-520261101",
        "CLAUDE-OPUS-5-5",
        " claude-opus-5-5",
        "openai/gpt-6-luna",
        "",
    ],
)
def test_a_model_name_the_site_does_not_know_is_stored_as_other(upstream: str) -> None:
    """The answering model is what the provider's response says it was. Only an id from
    the deployment's own lists is kept; anything else is one fixed word."""
    known = ("google/gemini-3.1-flash-lite", "gpt-6-luna")
    log = {**OK_LOG, "model": OPUS, "ran": upstream, "selector": "gpt-6-luna"}
    row = run_of(log, at=NOW, visitor_day="f" * 32, models=known)
    assert row.model == ("other" if upstream else OPUS)
    assert (row.requested_model, row.requested_selector) == (OPUS, "gpt-6-luna")
    assert SENTINEL not in json.dumps(asdict(row))
    asked = {**OK_LOG, "model": upstream or SENTINEL, "selector": upstream or SENTINEL}
    row = run_of(asked, at=NOW, visitor_day="f" * 32, models=known)
    assert (row.requested_model, row.requested_selector) == ("other", "other")


@pytest.mark.parametrize(
    ("echoed", "base"),
    [
        ("claude-opus-5-5-20261101", OPUS),
        ("claude-sonnet-5-5-20261101", SONNET),
        ("claude-sonnet-5-5-2026-11-01", SONNET),
        ("claude-opus-5-5-1", OPUS),
        ("claude-opus-5-5-" + "1" * 16, OPUS),
        ("google/gemini-3.1-flash-lite-001", GEMINI),
    ],
)
def test_a_known_id_with_a_date_or_version_suffix_is_recorded_as_the_known_id(
    echoed: str, base: str
) -> None:
    """A provider may answer with a dated snapshot of the id that was asked for. Only
    digits and hyphens may follow the id, so nothing free-form passes."""
    log = {**OK_LOG, "model": echoed, "ran": echoed, "selector": echoed}
    row = run_of(log, at=NOW, visitor_day="f" * 32, models=(GEMINI,))
    assert (row.requested_model, row.model, row.requested_selector) == (base,) * 3


def test_of_two_known_ids_that_could_match_the_longer_one_is_taken() -> None:
    known = ("vendor/model", "vendor/model-2", "vendor/model-2-5")
    for echoed, expected in [
        ("vendor/model-2", "vendor/model-2"),  # itself an id: never read as a suffix
        ("vendor/model-2-5", "vendor/model-2-5"),
        ("vendor/model-2-5-20261101", "vendor/model-2-5"),
        ("vendor/model-2-20261101", "vendor/model-2"),
        ("vendor/model-3", "vendor/model"),
        ("vendor/model-2-5x", "other"),
    ]:
        for order in (known, known[::-1]):
            row = run_of({**OK_LOG, "ran": echoed}, at=NOW, visitor_day="f" * 32, models=order)
            assert row.model == expected, echoed


def test_dated_claude_ids_are_priced_as_their_base_ids() -> None:
    """From the log record to the dollars: the row holds the base id, so ``by_model``,
    the estimate and ``unpriced_runs`` all see it."""
    runs, fake = make()
    for echoed in ("claude-opus-5-5-20261101", "claude-sonnet-5-5-20261101"):
        log = {**OK_LOG, "fallbacks": [], "ran": echoed, "selector": "retrieval"}
        log.update(prompt_tokens=1_000_000, completion_tokens=100_000, selector_cost_usd=None)
        runs.record(run_of(log, at=NOW, visitor_day="f" * 32))
    answer = runs.stats(1)
    assert [(row["model"], row["estimated_usd"]) for row in answer["by_model"]] == [
        (OPUS, 6.0),
        (SONNET, 3.0),
    ]
    assert answer["totals"]["estimated_usd"] == 9.0 and answer["totals"]["unpriced_runs"] == 0
    assert "20261101" not in json.dumps([fake.sent, [str(row) for row in fake.rows]], default=str)


def test_codes_reasons_and_the_rest_are_the_sites_own_words_or_other() -> None:
    log = {
        **OK_LOG,
        "status": 502,
        "code": f"provider said {SENTINEL}",
        "credential": SENTINEL,
        "provider": SENTINEL,
        "tier": SENTINEL,
        "fallbacks": [
            {"reason": "rate_limited"},
            {"reason": SENTINEL},
            {"reason": "a,b"},
            {"reason": "transient"},
            {"reason": "follows_extraction"},
        ],
    }
    row = run_of(log, at=NOW, visitor_day="f" * 32, models=(GEMINI,))
    assert (row.code, row.credential, row.provider, row.tier) == ("other", "other", "other", None)
    assert row.fallback_reasons == "rate_limited,other,transient,follows_extraction"
    assert SENTINEL not in json.dumps(asdict(row))


def test_every_code_and_reason_the_site_can_raise_is_one_the_row_keeps() -> None:
    """A code that is missing from ``usage.CODES`` would be stored as ``other``: this
    reads the site's own source for the codes it raises and the reasons it notes."""
    from ner_site import fallback, http, limits, service

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((Path(usage.__file__).parent).glob("*.py"))
    )
    raised = set(re.findall(r'(?:ServiceError|ExtractionError|_bad)\(\s*"([a-z_]+)"', source))
    assert len(raised) > 25 and "visitor_limit_reached" in raised
    constants = {http.STATS_UNAVAILABLE, http.LIMITS_UNAVAILABLE}
    assert raised | constants <= usage.CODES
    reasons = {
        *fallback.STEP_DOWN_REASONS,
        fallback.TRANSIENT,
        fallback.CALL_LIMIT,
        limits.OPUS_USED,
        http.CLAUDE_UNAVAILABLE,
        service.FOLLOWS_EXTRACTION,
    }
    assert reasons == usage.REASONS
    assert "other" not in usage.CODES | usage.REASONS
    assert all("," not in word for word in usage.REASONS)


# --- a run that failed after paid calls (fix wave, A4) --------------------------------

FAILED_LOG: dict[str, Any] = {
    "method": "POST",
    "path": "/api/ner",
    "provider": "site",
    "credential": "site",
    "text_chars": 33,
    "custom_prompt": False,
    "model": OPUS,
    "selector": "retrieval",
    "tier": "opus",
    "code": "extraction_unparseable",
    "status": 422,
    "timing_ms": {"handle": 2100},
}


def test_a_failed_run_keeps_the_tokens_its_failure_carried_under_the_model_asked_for() -> None:
    """The tokens of a failed run are the sum over every model its chain tried, and
    the chain only steps down to cheaper models: priced at the model the request asked
    for, such a run can be overstated and never understated. (They used to go to the
    last model tried: a paid Opus refusal was priced as Sonnet.)"""
    carried = {"llm_calls": 2, "prompt_tokens": 900, "completion_tokens": 90}
    log = {**FAILED_LOG, "fallbacks": [{"stage": "extract", "reason": "refused"}]}
    row = run_of(log, at=NOW, visitor_day="f" * 32, failed={**carried, "model": SONNET})
    assert (row.status, row.code, row.requested_model, row.model) == (
        422,
        "extraction_unparseable",
        OPUS,
        OPUS,
    )
    assert (row.llm_calls, row.prompt_tokens, row.completion_tokens) == (2, 900, 90)
    assert row.entities is None and row.total_ms is None


def test_a_failure_without_reported_tokens_names_no_model() -> None:
    """Calls were sent and nothing came back that was billed by the token: the run is
    still one that reached no model."""
    carried = {"model": OPUS, "llm_calls": 3, "prompt_tokens": None, "completion_tokens": None}
    row = run_of(FAILED_LOG, at=NOW, visitor_day="f" * 32, failed=carried)
    assert (row.model, row.llm_calls, row.prompt_tokens, row.completion_tokens) == (
        None,
        3,
        None,
        None,
    )


def test_what_a_failure_carried_is_typed_and_named_like_the_rest_of_the_row() -> None:
    carried = {"model": SENTINEL, "llm_calls": "2", "prompt_tokens": 9.5, "completion_tokens": 9}
    row = run_of(FAILED_LOG, at=NOW, visitor_day="f" * 32, failed=carried)
    assert (row.model, row.llm_calls, row.prompt_tokens, row.completion_tokens) == (
        OPUS,
        None,
        None,
        9,
    )
    assert SENTINEL not in json.dumps(asdict(row))
    # A successful run is read from its log record alone.
    ok = run_of(OK_LOG, at=NOW, visitor_day="f" * 32, models=(GEMINI,), failed=carried)
    assert (ok.model, ok.llm_calls, ok.prompt_tokens) == (GEMINI, 2, 412)


# --- recording -----------------------------------------------------------------------


def test_fake_is_an_executor_and_the_defaults() -> None:
    assert isinstance(FakeRuns(), SQLExecutor)
    assert TABLE == "site_runs" and len(COLUMNS) == 29
    assert PRICES == {OPUS: (4.0, 20.0), SONNET: (2.0, 10.0)}
    assert (usage.DEFAULT_DAYS, usage.MAX_DAYS) == (30, 365)


def test_one_round_trip_per_row_and_the_table_is_created_once() -> None:
    runs, fake = make()
    runs.record(a_run())
    assert fake.round_trips == 1
    assert fake.tags() == ["site_runs_table", "site_runs_index", "site_run"]
    runs.record(a_run(status=429, code="visitor_limit_reached"))
    assert fake.round_trips == 2 and fake.tags()[3:] == ["site_run"]
    assert [row["status"] for row in fake.rows] == [200, 429]


def test_every_value_is_a_parameter() -> None:
    runs, fake = make()
    row = a_run(
        tier="opus",
        requested_model=OPUS,
        model=SONNET,
        country="SI",
        fallback_reasons="rate_limited",
        prompt_tokens=412,
        selector_cost_usd=0.0125,
        custom_prompt=True,
    )
    runs.record(row)
    for sql, _ in fake.sent:
        assert "xwalk.site_runs" in sql
        for value in (row.visitor_day, OPUS, SONNET, "SI", "rate_limited", "412", "0.0125"):
            assert value not in sql
    (document,) = fake.sent[-1][1]
    assert json.loads(document) == asdict(row)
    assert fake.rows[-1]["model"] == SONNET and fake.rows[-1]["code"] is None


def test_a_second_instance_creates_the_table_again_harmlessly() -> None:
    first, fake = make()
    first.record(a_run())
    second, _ = make(fake)
    second.record(a_run())
    assert fake.tags().count("site_runs_table") == 2 and len(fake.rows) == 2


@pytest.mark.parametrize("call", ["record", "stats"])
def test_an_executor_failure_is_usage_unavailable_without_secrets(call: str) -> None:
    runs, fake = make()
    fake.fail = PostgresError(f"could not reach Neon at ep-x-pooler ({SENTINEL_URL})", code="08006")
    with pytest.raises(UsageUnavailable) as caught:
        runs.record(a_run()) if call == "record" else runs.stats(7)
    message = str(caught.value)
    assert message == "site runs could not be queried (PostgresError 08006)"
    assert caught.value.kind == "PostgresError" and caught.value.code == "08006"
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    for secret in (SENTINEL_URL, "npg_SENTINELpw0123", "ep-x-pooler"):
        assert secret not in message


def test_any_exception_from_the_executor_is_usage_unavailable() -> None:
    runs, fake = make()
    fake.fail = TimeoutError(SENTINEL_URL)
    with pytest.raises(UsageUnavailable) as caught:
        runs.record(a_run())
    assert str(caught.value) == "site runs could not be queried (TimeoutError)"
    assert caught.value.kind == "TimeoutError" and caught.value.code is None


def test_a_table_that_was_dropped_is_created_again_by_the_run_after_the_one_that_found_out() -> (
    None
):
    """A warm instance remembers that it created the table. After a ``DROP TABLE`` its
    next insert fails (that row is lost, and said so), and the one after creates."""
    runs, fake = make()
    runs.record(a_run())
    fake.created, fake.rows = False, []  # DROP TABLE, behind the instance's back
    with pytest.raises(UsageUnavailable) as caught:
        runs.record(a_run())
    assert caught.value.code == "42P01" and fake.tags()[3:] == ["site_run"]
    runs.record(a_run(status=429))
    assert fake.tags()[4:] == ["site_runs_table", "site_runs_index", "site_run"]
    assert [row["status"] for row in fake.rows] == [429]
    # Any other failure leaves the instance's memory of the table alone.
    fake.fail = PostgresError("down", code="08006")
    with pytest.raises(UsageUnavailable):
        runs.record(a_run())
    fake.fail = None
    runs.record(a_run())
    assert fake.tags()[7:] == ["site_run", "site_run"]
    # The statistics notice it too.
    fake.created, fake.rows = False, []
    with pytest.raises(UsageUnavailable):
        runs.stats(1)
    assert runs.stats(1)["totals"]["runs"] == 0


def test_a_late_answer_to_an_insert_sent_before_the_drop_does_not_bring_the_flag_back() -> None:
    """Two writes of one instance. The first was sent while the table was there and is
    acknowledged late; in between the table is dropped and the second finds out. The
    late acknowledgement used to set "created" again, and the run after lost its row
    as well."""
    import threading

    runs, fake = make()
    runs.record(a_run())
    sent, release = threading.Event(), threading.Event()
    real = fake.execute

    def slow(sql: str, params: Any = ()) -> Any:
        rows = real(sql, params)  # reaches the table before the drop
        sent.set()
        release.wait(10)
        return rows

    fake.execute = slow  # type: ignore[method-assign]
    early = threading.Thread(target=lambda: runs.record(a_run(status=201)))
    early.start()
    assert sent.wait(10)
    fake.execute = real  # type: ignore[method-assign]
    fake.created, fake.rows = False, []  # DROP TABLE
    with pytest.raises(UsageUnavailable) as caught:
        runs.record(a_run(status=202))
    assert caught.value.code == "42P01"
    release.set()
    early.join(10)
    # The next run provisions and is kept, whatever arrived in the meantime.
    runs.record(a_run(status=203))
    assert [row["status"] for row in fake.rows] == [203]
    assert fake.tags()[-3:] == ["site_runs_table", "site_runs_index", "site_run"]
    runs.record(a_run(status=204))
    assert fake.tags()[-1] == "site_run" and fake.tags().count("site_runs_table") == 2


def test_a_call_may_be_given_less_time_than_the_executor_has() -> None:
    """``record(run, timeout=...)`` and ``stats`` hand the time that is left to an
    executor that can be told (``within``); any other executor is used as it is."""
    given: list[float] = []

    class Timed(FakeRuns):
        def within(self, seconds: float) -> Any:
            given.append(seconds)
            return self

    fake = Timed()
    runs = SiteRuns(fake, "xwalk", clock=lambda: NOW)
    runs.record(a_run(), timeout=0.75)
    runs.record(a_run())
    runs.stats(1, timeout=3.5)
    runs.create()
    assert given == [0.75, 3.5] and len(fake.rows) == 2
    plain, other = make()
    plain.record(a_run(), timeout=0.75)
    assert len(other.rows) == 1


def test_a_failed_first_call_creates_the_table_on_the_next() -> None:
    runs, fake = make()
    fake.fail = PostgresError("down")
    with pytest.raises(UsageUnavailable):
        runs.record(a_run())
    fake.fail = None
    runs.record(a_run())
    assert fake.tags().count("site_runs_table") == 2 and len(fake.rows) == 1


def test_create_makes_the_schema_table_and_index_in_one_round_trip_and_no_row() -> None:
    runs, fake = make()
    runs.create()
    runs.create()
    assert fake.round_trips == 2 and fake.rows == []
    assert fake.tags() == ["site_runs_schema", "site_runs_table", "site_runs_index"] * 2
    assert "CREATE SCHEMA IF NOT EXISTS xwalk" in fake.sent[0][0]
    assert all(params == [] for _, params in fake.sent)
    # An instance that provisioned has nothing left to create.
    runs.record(a_run())
    assert fake.tags()[6:] == ["site_run"]


def test_a_failed_create_is_usage_unavailable_without_secrets() -> None:
    runs, fake = make()
    fake.fail = PostgresError(f"permission denied ({SENTINEL_URL})", code="42501")
    with pytest.raises(UsageUnavailable) as caught:
        runs.create()
    assert str(caught.value) == "site runs could not be queried (PostgresError 42501)"
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


@pytest.mark.parametrize("schema", ["Xwalk", "xwalk; DROP TABLE x", "x.y", "", "1abc", 'x"y'])
def test_a_schema_that_is_not_a_lower_case_identifier_is_refused(schema: str) -> None:
    with pytest.raises(ValueError, match="schema"):
        SiteRuns(FakeRuns(), schema)


# --- the statistics: the Python around the SQL ---------------------------------------


@pytest.mark.parametrize("days", [0, -1, 366, 1000, True, "7", 7.0, None])
def test_days_outside_the_bounds_are_refused_before_any_query(days: Any) -> None:
    runs, fake = make()
    with pytest.raises(ValueError, match="days"):
        runs.stats(days)
    assert fake.round_trips == 0


@pytest.mark.parametrize(
    "days,first", [(1, "2026-11-18"), (7, "2026-11-12"), (30, "2026-10-20"), (365, "2025-11-19")]
)
def test_the_window_is_the_last_n_utc_days_including_today(days: int, first: str) -> None:
    runs, fake = make()
    answer = runs.stats(days)
    assert fake.round_trips == 1
    assert fake.tags() == ["site_runs_table", "site_runs_index", "site_runs_stats"]
    assert fake.sent[-1][1] == [first, "2026-11-18"]
    assert answer["days"] == days and len(answer["by_day"]) == days
    assert answer["by_day"][0]["day"] == first and answer["by_day"][-1]["day"] == "2026-11-18"


def test_the_utc_day_is_used_just_before_and_after_midnight() -> None:
    late = datetime(2026, 11, 18, 23, 59, 59, tzinfo=timezone.utc).timestamp()
    runs, fake = make(now=late)
    assert runs.stats(1)["by_day"][0]["day"] == "2026-11-18"
    runs, fake = make(now=late + 1)
    assert runs.stats(1)["by_day"][0]["day"] == "2026-11-19"


EMPTY_DAY = {
    "runs": 0,
    "ok": 0,
    "visitors": 0,
    "opus": 0,
    "sonnet": 0,
    "other": 0,
    "own": 0,
    "stepped_down": 0,
    "estimated_usd": 0.0,
    "p50_ms": 0,
    "p95_ms": 0,
}


def test_no_runs_is_an_answer_of_zeros_with_every_day_present() -> None:
    runs, _ = make()
    answer = runs.stats(3)
    assert answer == {
        "ok": True,
        "days": 3,
        "generated_at": "2026-11-18T14:20:00Z",
        "totals": {
            "runs": 0,
            "ok": 0,
            "errors": 0,
            "visitors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
            "unpriced_runs": 0,
        },
        "by_day": [{"day": day, **EMPTY_DAY} for day in ("2026-11-16", "2026-11-17", "2026-11-18")],
        "by_model": [],
        "by_code": [],
        "by_fallback": [],
        "by_country": [],
    }
    assert list(answer) == [
        "ok",
        "days",
        "generated_at",
        "totals",
        "by_day",
        "by_model",
        "by_code",
        "by_fallback",
        "by_country",
    ]


def _raw(spend: list[dict[str, Any]], days: tuple[str, ...] = ("2026-11-18",)) -> dict[str, Any]:
    runs = {day: sum(int(g["runs"]) for g in spend if g["day"] == day) for day in days}
    return {
        "by_day": [
            {"day": day, **{**EMPTY_DAY, "runs": runs[day], "ok": runs[day]}} for day in days
        ],
        "spend": spend,
        "by_code": [],
        "by_fallback": [],
        "by_country": [],
    }


def _group(**over: Any) -> dict[str, Any]:
    return {
        "day": "2026-11-18",
        "credential": "site",
        "model": OPUS,
        "selector": "retrieval",
        "runs": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "selector_prompt_tokens": 0,
        "selector_completion_tokens": 0,
        "selector_cost_usd": 0,
        "unreported_selections": 0,
        **over,
    }


def _priced(*groups: dict[str, Any]) -> dict[str, Any]:
    return shape(_raw(list(groups)), days=1, generated_at=NOW, models=(GEMINI,))


def test_prices_are_per_million_tokens_input_and_output() -> None:
    opus = _priced(_group(prompt_tokens=1_000_000, completion_tokens=1_000_000))
    assert opus["totals"]["estimated_usd"] == 24.0
    assert opus["by_day"][0]["estimated_usd"] == 24.0
    assert opus["by_model"] == [
        {
            "model": OPUS,
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "estimated_usd": 24.0,
        }
    ]
    sonnet = _priced(_group(model=SONNET, prompt_tokens=1_000_000, completion_tokens=1_000_000))
    assert sonnet["totals"]["estimated_usd"] == 12.0
    small = _priced(_group(prompt_tokens=412, completion_tokens=88))
    assert small["totals"]["estimated_usd"] == round(412 * 4e-6 + 88 * 20e-6, 6) == 0.003408


@pytest.mark.parametrize("model", sorted(PRICES))
def test_a_priced_model_with_any_tokens_never_shows_as_zero_dollars(model: str) -> None:
    """The page reads a site-key row with tokens and zero dollars as "no price"."""
    for tokens in ({"prompt_tokens": 1}, {"completion_tokens": 1}):
        answer = _priced(_group(model=model, **tokens))
        assert answer["by_model"][0]["estimated_usd"] > 0
        assert answer["by_day"][0]["estimated_usd"] > 0 and answer["totals"]["estimated_usd"] > 0
    picked = _priced(_group(model=GEMINI, selector=model, selector_prompt_tokens=1))
    row = next(row for row in picked["by_model"] if row["model"] == model)
    assert row["runs"] == 0 and row["estimated_usd"] > 0


def test_only_runs_on_the_sites_keys_are_priced() -> None:
    answer = _priced(
        _group(credential="visitor", prompt_tokens=1_000_000, completion_tokens=1_000_000),
        _group(prompt_tokens=1000, completion_tokens=100),
    )
    assert answer["totals"]["estimated_usd"] == 0.006
    assert answer["totals"]["unpriced_runs"] == 0
    assert (answer["totals"]["prompt_tokens"], answer["totals"]["completion_tokens"]) == (
        1_001_000,
        1_000_100,
    )
    visitor = next(row for row in answer["by_model"] if row["credential"] == "visitor")
    assert visitor["estimated_usd"] == 0.0 and visitor["prompt_tokens"] == 1_000_000


def test_selection_tokens_are_priced_at_the_requested_selector_when_it_is_a_model() -> None:
    answer = _priced(
        _group(
            model=SONNET,
            selector=OPUS,
            prompt_tokens=1_000_000,
            selector_prompt_tokens=1_000_000,
            selector_completion_tokens=100_000,
        )
    )
    assert answer["totals"]["estimated_usd"] == 2.0 + 4.0 + 2.0
    assert answer["by_model"] == [
        {
            "model": SONNET,
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "estimated_usd": 2.0,
        },
        {
            "model": OPUS,
            "credential": "site",
            "runs": 0,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 100_000,
            "estimated_usd": 6.0,
        },
    ]


@pytest.mark.parametrize("selector", ["retrieval", None])
def test_selection_tokens_without_a_selector_go_to_no_model(selector: str | None) -> None:
    answer = _priced(
        _group(selector=selector, selector_prompt_tokens=500, selector_completion_tokens=50)
    )
    assert [row["model"] for row in answer["by_model"]] == [OPUS]
    assert answer["by_model"][0]["prompt_tokens"] == 0
    assert answer["totals"]["prompt_tokens"] == 500 and answer["totals"]["estimated_usd"] == 0.0


def test_jev_has_a_row_of_its_own_with_the_cost_it_reported() -> None:
    """So that the dollars of ``by_model`` add up to the total (fix wave, A4)."""
    answer = _priced(
        _group(
            selector="jev",
            runs=2,
            prompt_tokens=1_000_000,
            selector_prompt_tokens=500,
            selector_completion_tokens=50,
            selector_cost_usd=0.0375,
        ),
        _group(credential="visitor", selector="jev", selector_cost_usd=9.0),
    )
    assert answer["by_model"] == [
        {
            "model": OPUS,
            "credential": "site",
            "runs": 2,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "estimated_usd": 4.0,
        },
        {
            "model": OPUS,
            "credential": "visitor",
            "runs": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
        },
        {
            "model": "jev",
            "credential": "site",
            "runs": 0,
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "estimated_usd": 0.0375,
        },
    ]
    assert answer["totals"]["estimated_usd"] == 4.0375 == answer["by_day"][0]["estimated_usd"]
    assert sum(row["estimated_usd"] for row in answer["by_model"]) == 4.0375
    assert answer["totals"]["unpriced_runs"] == 0


def test_a_run_is_unpriced_once_when_either_stage_has_no_list_price() -> None:
    """A priced extraction with an unpriced model selector used to count as priced."""
    answer = _priced(
        # Priced extraction, unpriced selector that ran in two of the three runs.
        _group(runs=3, selector=GEMINI, prompt_tokens=1000, unreported_selections=2),
        # Both stages unpriced: each run once, not twice.
        _group(model=GEMINI, runs=4, selector=GEMINI, unreported_selections=4),
        # Unpriced extraction, priced selector.
        _group(model=GEMINI, runs=5, selector=OPUS, unreported_selections=5),
        # Priced on both stages, Jev, and no selector: nothing unpriced.
        _group(runs=6, selector=SONNET, unreported_selections=6),
        _group(runs=7, selector="jev", unreported_selections=7),
        _group(runs=8, selector="retrieval"),
        # A visitor's key is never priced, so never unpriced either.
        _group(
            credential="visitor", model=GEMINI, runs=9, selector=GEMINI, unreported_selections=9
        ),
        # No model answered and no selection ran.
        _group(model=None, runs=10, selector=GEMINI),
    )
    assert answer["totals"]["unpriced_runs"] == 2 + 4 + 5
    assert answer["totals"]["runs"] == 52


def test_a_reported_selector_cost_is_added_and_a_model_without_a_price_is_unpriced() -> None:
    answer = _priced(
        _group(
            model=GEMINI,
            selector="jev",
            runs=3,
            prompt_tokens=9_000_000,
            completion_tokens=900_000,
            selector_cost_usd=0.0375,
        ),
        _group(credential="visitor", model=GEMINI, runs=2, selector_cost_usd=5.0),
    )
    assert answer["totals"]["estimated_usd"] == 0.0375 and answer["totals"]["unpriced_runs"] == 3
    assert answer["by_day"][0]["estimated_usd"] == 0.0375
    assert [
        (row["model"], row["credential"], row["estimated_usd"]) for row in answer["by_model"]
    ] == [
        (GEMINI, "site", 0.0),
        (GEMINI, "visitor", 0.0),
        ("jev", "site", 0.0375),
    ]


def test_a_reported_cost_is_not_added_on_top_of_a_priced_selector() -> None:
    """Never both: a selector with a price in the table is priced by its tokens."""
    answer = _priced(
        _group(selector=OPUS, selector_prompt_tokens=1_000_000, selector_cost_usd=99.0)
    )
    assert answer["totals"]["estimated_usd"] == 4.0


def test_a_reported_cost_of_a_selector_model_is_shown_on_that_model() -> None:
    answer = _priced(_group(selector=GEMINI, selector_prompt_tokens=10, selector_cost_usd=0.25))
    gemini = next(row for row in answer["by_model"] if row["model"] == GEMINI)
    assert gemini == {
        "model": GEMINI,
        "credential": "site",
        "runs": 0,
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "estimated_usd": 0.25,
    }
    assert answer["totals"]["estimated_usd"] == 0.25 and answer["totals"]["unpriced_runs"] == 0


def test_runs_that_never_reached_a_model_are_in_the_null_row_and_not_unpriced() -> None:
    answer = _priced(_group(model=None, selector=None, runs=4), _group(prompt_tokens=10))
    assert answer["by_model"][0] == {
        "model": None,
        "credential": "site",
        "runs": 4,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_usd": 0.0,
    }
    assert answer["by_model"][1]["model"] == OPUS and answer["totals"]["unpriced_runs"] == 0
    assert answer["totals"]["runs"] == 5


def test_numbers_that_arrive_as_strings_or_floats_are_read() -> None:
    raw = _raw([_group(runs="2", prompt_tokens="1000000", selector_cost_usd="0")])
    raw["by_day"][0].update(runs="2", ok="1", visitors="1", p50_ms="1500", p95_ms=2900.0)
    raw["by_code"] = [{"code": "provider_error", "runs": "1"}]
    answer = shape(json.dumps(raw), days=1, generated_at=NOW)
    assert answer["totals"] == {
        "runs": 2,
        "ok": 1,
        "errors": 1,
        "visitors": 1,
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "estimated_usd": 4.0,
        "unpriced_runs": 0,
    }
    assert (answer["by_day"][0]["p50_ms"], answer["by_day"][0]["p95_ms"]) == (1500, 2900)
    assert answer["by_code"] == [{"code": "provider_error", "runs": 1}]


@pytest.mark.parametrize(
    "answer",
    [
        [],
        [{}],
        [{"stats": None}],
        [{"stats": "not json"}],
        [{"stats": "[]"}],
        [{"stats": '{"by_day": []}'}],
        [{"stats": '{"by_day": [{"day": "2026-11-18"}], "spend": [], "by_code": []}'}],
    ],
)
def test_an_unreadable_answer_is_usage_unavailable(answer: Any) -> None:
    runs, fake = make()
    fake._site_runs_stats = lambda sql, p: answer  # type: ignore[method-assign]
    with pytest.raises(UsageUnavailable, match="unreadable"):
        runs.stats(1)


# --- the statistics of a fixed set of rows: the fake, and Postgres --------------------

A_17 = visitor_day(VISITOR, SALT, NOW - 86400)
A_18 = visitor_day(VISITOR, SALT, NOW)
B_18 = visitor_day(OTHER_VISITOR, SALT, NOW)

FIXED: tuple[Run, ...] = (
    # Opus for both steps.
    a_run(
        at=at(18, 10),
        visitor_day=A_18,
        tier="opus",
        requested_model=OPUS,
        model=OPUS,
        requested_selector=OPUS,
        country="SI",
        text_chars=900,
        custom_prompt=False,
        entities=4,
        matches=20,
        picks=4,
        llm_calls=1,
        prompt_tokens=1000,
        completion_tokens=100,
        selector_calls=4,
        selector_prompt_tokens=2000,
        selector_completion_tokens=200,
        embedding_requests=1,
        extract_ms=400,
        map_ms=200,
        select_ms=400,
        total_ms=1000,
    ),
    # Stepped down at admission: extraction on Sonnet, the selector asked for was Opus.
    a_run(
        at=at(18, 11),
        visitor_day=A_18,
        tier="sonnet",
        requested_model=OPUS,
        model=SONNET,
        requested_selector=OPUS,
        fallback_reasons="opus_allowance_used",
        country="SI",
        prompt_tokens=500_000,
        completion_tokens=100_000,
        selector_calls=1,
        selector_prompt_tokens=1000,
        selector_completion_tokens=100,
        total_ms=3000,
    ),
    # Refused at admission: no tier, no model, no timing.
    a_run(
        at=at(18, 12),
        visitor_day=B_18,
        status=429,
        code="visitor_limit_reached",
        requested_model=OPUS,
        requested_selector=OPUS,
        country="US",
    ),
    # The visitor's own key, in the last second of the day.
    a_run(
        at=at(18, 23, 59, 59),
        visitor_day=B_18,
        credential="visitor",
        provider="anthropic",
        tier="own",
        requested_model=OPUS,
        model=OPUS,
        requested_selector="retrieval",
        country="US",
        prompt_tokens=10_000,
        completion_tokens=1000,
        total_ms=2000,
    ),
    # The first instant of the day before: an OpenRouter model, Jev's reported cost.
    a_run(
        at=at(17),
        visitor_day=A_17,
        tier="other",
        requested_model=GEMINI,
        model=GEMINI,
        requested_selector="jev",
        fallback_reasons="rate_limited,overloaded",
        prompt_tokens=3000,
        completion_tokens=300,
        selector_calls=2,
        selector_prompt_tokens=50,
        selector_completion_tokens=5,
        selector_cost_usd=0.0125,
        total_ms=500,
    ),
    # A provider failure after admission.
    a_run(
        at=at(17, 8),
        visitor_day=A_17,
        status=502,
        code="provider_error",
        tier="opus",
        requested_model=OPUS,
        requested_selector="retrieval",
        fallback_reasons="rate_limited",
        country="SI",
    ),
    # A priced extraction whose selector is a model without a list price.
    a_run(
        at=at(17, 9),
        visitor_day=A_17,
        tier="opus",
        requested_model=OPUS,
        model=OPUS,
        requested_selector=GEMINI,
        country="SI",
        prompt_tokens=2000,
        completion_tokens=200,
        selector_calls=2,
        selector_prompt_tokens=400,
        selector_completion_tokens=40,
        total_ms=1500,
    ),
    # Failed after a paid call: the model answered and the answer could not be used.
    a_run(
        at=at(18, 13),
        visitor_day=B_18,
        status=422,
        code="extraction_unparseable",
        tier="sonnet",
        requested_model=SONNET,
        model=SONNET,
        requested_selector="retrieval",
        country="US",
        llm_calls=1,
        prompt_tokens=100_000,
        completion_tokens=10_000,
    ),
    # Outside the three days, on either side.
    a_run(at=at(15, 23, 59, 59), model=OPUS, prompt_tokens=1_000_000, total_ms=9, country="DE"),
    a_run(at=at(19), model=OPUS, prompt_tokens=1_000_000, total_ms=9, country="DE"),
)

EXPECTED: dict[str, Any] = {
    "ok": True,
    "days": 3,
    "generated_at": "2026-11-18T14:20:00Z",
    "totals": {
        "runs": 8,
        "ok": 5,
        "errors": 3,
        "visitors": 3,
        "prompt_tokens": 619_450,
        "completion_tokens": 111_945,
        "estimated_usd": 2.3485,
        # The OpenRouter extraction, and the Opus run whose selector has no list price.
        "unpriced_runs": 2,
    },
    "by_day": [
        {"day": "2026-11-16", **EMPTY_DAY},
        {
            "day": "2026-11-17",
            "runs": 3,
            "ok": 2,
            "visitors": 1,
            "opus": 2,
            "sonnet": 0,
            "other": 1,
            "own": 0,
            "stepped_down": 2,
            "estimated_usd": 0.0245,
            "p50_ms": 1000,
            "p95_ms": 1450,
        },
        {
            "day": "2026-11-18",
            "runs": 5,
            "ok": 3,
            "visitors": 2,
            "opus": 1,
            "sonnet": 2,
            "other": 0,
            "own": 1,
            "stepped_down": 1,
            "estimated_usd": 2.324,
            "p50_ms": 2000,
            "p95_ms": 2900,
        },
    ],
    "by_model": [
        {
            "model": OPUS,
            "credential": "site",
            "runs": 2,
            "prompt_tokens": 6000,
            "completion_tokens": 600,
            "estimated_usd": 0.036,
        },
        # One answered run and the one that failed after its paid call.
        {
            "model": SONNET,
            "credential": "site",
            "runs": 2,
            "prompt_tokens": 600_000,
            "completion_tokens": 110_000,
            "estimated_usd": 2.3,
        },
        # The refused run and the one whose provider failed: no model answered.
        {
            "model": None,
            "credential": "site",
            "runs": 2,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
        },
        {
            "model": OPUS,
            "credential": "visitor",
            "runs": 1,
            "prompt_tokens": 10_000,
            "completion_tokens": 1000,
            "estimated_usd": 0.0,
        },
        {
            "model": GEMINI,
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 3400,
            "completion_tokens": 340,
            "estimated_usd": 0.0,
        },
        # What Jev reported for its selections, on no run of its own.
        {
            "model": "jev",
            "credential": "site",
            "runs": 0,
            "prompt_tokens": 50,
            "completion_tokens": 5,
            "estimated_usd": 0.0125,
        },
    ],
    "by_code": [
        {"code": "extraction_unparseable", "runs": 1},
        {"code": "provider_error", "runs": 1},
        {"code": "visitor_limit_reached", "runs": 1},
    ],
    "by_fallback": [
        {"reason": "rate_limited", "runs": 2},
        {"reason": "opus_allowance_used", "runs": 1},
        {"reason": "overloaded", "runs": 1},
    ],
    "by_country": [
        {"country": "SI", "runs": 4},
        {"country": "US", "runs": 3},
        {"country": None, "runs": 1},
    ],
}


class Database:
    """A throwaway schema on the Postgres of ``NER_SITE_TEST_DATABASE_URL``."""

    def __init__(self, executor: PsqlExecutor, schema: str) -> None:
        self.executor = executor
        self.schema = schema

    def runs(self, now: float = NOW) -> SiteRuns:
        return SiteRuns(self.executor, self.schema, clock=lambda: now)

    def value(self, sql: str) -> Any:
        rows = self.executor.execute(f"SELECT ({sql.format(table=f'{self.schema}.{TABLE}')}) AS v")
        return rows[0]["v"] if rows else None


def _database() -> Iterator[Database]:
    url = os.environ.get(DATABASE_URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{DATABASE_URL_ENV} is not set")
    if shutil.which("psql") is None:
        pytest.skip("the psql client is not installed")
    executor = PsqlExecutor(url)
    schema = f"usage_test_{secrets.token_hex(4)}"
    executor.execute(f"CREATE SCHEMA {schema}")
    try:
        yield Database(executor, schema)
    finally:
        executor.execute(f"DROP SCHEMA {schema} CASCADE")


database = pytest.fixture(_database)


@pytest.fixture(params=["fake", pytest.param("postgres", marks=pytest.mark.integration)])
def stored(request: pytest.FixtureRequest) -> Iterator[SiteRuns]:
    """The fixed rows, recorded one by one through ``SiteRuns.record``."""
    if request.param == "fake":
        runs, _ = make()
        for row in FIXED:
            runs.record(row)
        yield runs
        return
    for found in _database():
        runs = found.runs()
        for row in FIXED:
            runs.record(row)
        yield runs


def test_the_statistics_of_a_fixed_set_of_rows(stored: SiteRuns) -> None:
    answer = stored.stats(3, models=(GEMINI,))
    assert answer == EXPECTED
    # The dollars per model and per day both add up to the total.
    for part in ("by_model", "by_day"):
        assert round(sum(row["estimated_usd"] for row in answer[part]), 6) == 2.3485
    # The page draws "refused, no tier" as the runs that are in none of the four tiers.
    for day in stored.stats(30, models=(GEMINI,))["by_day"]:
        assert sum(day[tier] for tier in ("opus", "sonnet", "other", "own")) <= day["runs"]


def test_a_shorter_window_leaves_out_the_earlier_day(stored: SiteRuns) -> None:
    answer = stored.stats(1, models=(GEMINI,))
    assert answer["by_day"] == [EXPECTED["by_day"][2]]
    assert answer["totals"]["runs"] == 5 and answer["totals"]["visitors"] == 2
    assert answer["by_code"] == [
        {"code": "extraction_unparseable", "runs": 1},
        {"code": "visitor_limit_reached", "runs": 1},
    ]
    assert answer["by_fallback"] == [{"reason": "opus_allowance_used", "runs": 1}]
    assert answer["by_country"] == [{"country": "US", "runs": 3}, {"country": "SI", "runs": 2}]
    assert answer["totals"]["unpriced_runs"] == 0


def test_a_longer_window_is_zero_filled_and_takes_in_the_older_run(stored: SiteRuns) -> None:
    answer = stored.stats(30, models=(GEMINI,))
    assert len(answer["by_day"]) == 30 and answer["by_day"][0]["day"] == "2026-10-20"
    assert [day["runs"] for day in answer["by_day"][-4:]] == [1, 0, 3, 5]
    assert all(day == {"day": day["day"], **EMPTY_DAY} for day in answer["by_day"][:-4])
    assert answer["totals"]["runs"] == 9
    assert answer["by_country"][-2:] == [{"country": "DE", "runs": 1}, {"country": None, "runs": 1}]
    assert sum(row["runs"] for row in answer["by_country"]) == 9
    assert sum(row["runs"] for row in answer["by_model"]) == 9
    # One million more Opus input tokens on the site's key: four dollars.
    assert answer["totals"]["estimated_usd"] == 6.3485


# --- what is read back is normalised like what is written (fix round 2) ---------------

STRAY = "STRAY-SENTINEL-text"
# Rows as older code or a hand could have left them: `record` stores a Run as it is.
STRAY_ROWS: tuple[Run, ...] = (
    a_run(
        at=at(18, 9),
        tier="opus",
        model=f"claude-{STRAY}",
        requested_selector=STRAY,
        country=STRAY,
        prompt_tokens=1000,
        selector_calls=1,
        selector_prompt_tokens=10,
    ),
    a_run(
        at=at(18, 10),
        status=502,
        code=f"provider said {STRAY}",
        model=STRAY,
        requested_selector="retrieval",
        fallback_reasons=f"rate_limited,{STRAY},another {STRAY}",
        country="xx",
        prompt_tokens=500,
    ),
    a_run(at=at(18, 11), status=429, code="visitor_limit_reached", credential=STRAY, country="SI"),
    a_run(at=at(18, 12), model="claude-opus-5-5-20261101", prompt_tokens=1_000_000, country="SI"),
    a_run(at=at(18, 13), model="vendor/listed", prompt_tokens=7, tier=STRAY),
)
STRAY_EXPECTED: dict[str, Any] = {
    "by_model": [
        {
            "model": "other",
            "credential": "site",
            "runs": 2,
            "prompt_tokens": 1510,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
        },
        {
            "model": OPUS,
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "estimated_usd": 4.0,
        },
        {
            "model": "vendor/listed",
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 7,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
        },
        {
            "model": None,
            "credential": "other",
            "runs": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "estimated_usd": 0.0,
        },
    ],
    "by_code": [{"code": "other", "runs": 1}, {"code": "visitor_limit_reached", "runs": 1}],
    "by_fallback": [{"reason": "other", "runs": 2}, {"reason": "rate_limited", "runs": 1}],
    "by_country": [{"country": None, "runs": 3}, {"country": "SI", "runs": 2}],
}


def _stray_answer(runs: SiteRuns) -> dict[str, Any]:
    for row in STRAY_ROWS:
        runs.record(row)
    return runs.stats(1, models=("vendor/listed",))


def _assert_no_stray(answer: dict[str, Any]) -> None:
    assert STRAY not in json.dumps(answer) and "20261101" not in json.dumps(answer)
    for name, expected in STRAY_EXPECTED.items():
        assert answer[name] == expected, name
    assert answer["totals"]["runs"] == 5 and answer["totals"]["estimated_usd"] == 4.0
    # The two `other` runs and the one on a listed model without a price.
    assert answer["totals"]["unpriced_runs"] == 3
    assert answer["by_day"][0]["opus"] == 1 and answer["by_day"][0]["other"] == 0


def test_strings_already_in_the_table_are_normalised_on_the_way_out() -> None:
    """New rows hold only the site's own words, but a row may be older than that rule,
    or written by hand: the answer must not carry what such a row holds either."""
    runs, fake = make()
    answer = _stray_answer(runs)
    assert STRAY in json.dumps([str(row) for row in fake.rows])  # it is in the table
    _assert_no_stray(answer)
    # Without the deployment's list, a model only it knows is `other` as well.
    unlisted = runs.stats(1)
    assert {row["model"] for row in unlisted["by_model"]} == {"other", OPUS, None}


@pytest.mark.integration
def test_postgres_strings_already_in_the_table_are_normalised_on_the_way_out(
    database: Database,
) -> None:
    _assert_no_stray(_stray_answer(database.runs()))


@pytest.mark.integration
def test_postgres_a_dropped_table_is_created_again_after_one_lost_row(database: Database) -> None:
    runs = database.runs()
    runs.record(FIXED[0])
    database.executor.execute(f"DROP TABLE {database.schema}.{TABLE}")
    with pytest.raises(UsageUnavailable) as caught:
        runs.record(FIXED[0])
    assert caught.value.code == "42P01"
    runs.record(FIXED[0])
    assert database.value("SELECT count(*) FROM {table}") == "1"


# --- against Postgres only -----------------------------------------------------------

SPEC_COLUMNS = {
    "at": "timestamp with time zone",
    "visitor_day": "text",
    "status": "integer",
    "code": "text",
    "credential": "text",
    "provider": "text",
    "tier": "text",
    "requested_model": "text",
    "model": "text",
    "requested_selector": "text",
    "fallback_reasons": "text",
    "country": "text",
    "text_chars": "integer",
    "custom_prompt": "boolean",
    "entities": "integer",
    "matches": "integer",
    "picks": "integer",
    "llm_calls": "integer",
    "prompt_tokens": "integer",
    "completion_tokens": "integer",
    "selector_calls": "integer",
    "selector_prompt_tokens": "integer",
    "selector_completion_tokens": "integer",
    "selector_cost_usd": "double precision",
    "embedding_requests": "integer",
    "extract_ms": "integer",
    "map_ms": "integer",
    "select_ms": "integer",
    "total_ms": "integer",
}


def test_the_columns_are_the_specs() -> None:
    assert tuple(SPEC_COLUMNS) == COLUMNS


@pytest.mark.integration
def test_postgres_the_table_has_the_specs_columns_and_an_index_on_at(database: Database) -> None:
    runs = database.runs()
    runs.record(FIXED[0])
    columns = json.loads(
        database.value(
            "SELECT json_object_agg(column_name, data_type) FROM information_schema.columns "
            f"WHERE table_schema = '{database.schema}' AND table_name = '{TABLE}'"
        )
    )
    assert columns == SPEC_COLUMNS
    indexes = database.value(
        f"SELECT string_agg(indexdef, ' ') FROM pg_indexes "
        f"WHERE schemaname = '{database.schema}' AND tablename = '{TABLE}'"
    )
    assert indexes is not None and "(at)" in indexes


@pytest.mark.integration
def test_postgres_create_provisions_an_empty_database_and_can_be_repeated(
    database: Database,
) -> None:
    """What ``tools/setup_usage.py`` runs: the schema, the table and the index, twice
    over, on a schema that exists and on one that does not."""
    fresh = Database(database.executor, database.schema + "_new")
    try:
        for found in (database, fresh):
            found.runs().create()
            found.runs().create()
            assert found.value("SELECT count(*) FROM {table}") == "0"
            indexes = found.value(
                f"SELECT string_agg(indexdef, ' ') FROM pg_indexes "
                f"WHERE schemaname = '{found.schema}' AND tablename = '{TABLE}'"
            )
            assert indexes is not None and "(at)" in indexes
            found.runs().record(FIXED[0])
            found.runs().create()  # with a row in it: nothing is lost
            assert found.value("SELECT count(*) FROM {table}") == "1"
    finally:
        database.executor.execute(f"DROP SCHEMA IF EXISTS {fresh.schema} CASCADE")


@pytest.mark.integration
def test_postgres_a_row_is_stored_as_it_was_given_nulls_included(database: Database) -> None:
    runs = database.runs()
    full, refused = FIXED[0], FIXED[2]
    runs.record(full)
    runs.record(refused)
    database.runs().record(replace(full, at=at(18, 13)))  # a second instance: created again
    assert database.value("SELECT count(*) FROM {table}") == "3"
    stored = json.loads(
        database.value(
            "SELECT json_agg(to_jsonb(r) - 'at' ORDER BY r.at) FROM {table} AS r "
            "WHERE r.at < '2026-11-18T12:30:00+00:00'"
        )
    )
    expected = [{k: v for k, v in asdict(row).items() if k != "at"} for row in (full, refused)]
    assert stored == expected
    assert database.value("SELECT min(at) = '2026-11-18T10:00:00+00:00'::timestamptz FROM {table}")


@pytest.mark.integration
def test_postgres_statistics_before_any_run_create_the_table_and_answer_zeros(
    database: Database,
) -> None:
    answer = database.runs().stats(2)
    assert answer["totals"]["runs"] == 0 and answer["by_model"] == []
    assert answer["by_day"] == [
        {"day": "2026-11-17", **EMPTY_DAY},
        {"day": "2026-11-18", **EMPTY_DAY},
    ]
    assert database.value("SELECT count(*) FROM {table}") == "0"


@pytest.mark.integration
def test_postgres_a_year_of_days_across_both_clock_changes(database: Database) -> None:
    """365 days, each once, whatever the session's time zone (run with PGTZ set)."""
    runs = database.runs()
    runs.record(a_run(at="2026-03-29T00:30:00+00:00", total_ms=10))
    runs.record(a_run(at="2026-10-25T23:30:00+00:00", total_ms=30))
    answer = runs.stats(365)
    days = [day["day"] for day in answer["by_day"]]
    start = date(2025, 11, 19)
    assert days == [(start + timedelta(days=n)).isoformat() for n in range(365)]
    found = {day["day"]: day["runs"] for day in answer["by_day"] if day["runs"]}
    assert found == {"2026-03-29": 1, "2026-10-25": 1}


@pytest.mark.integration
def test_postgres_a_failed_statement_is_usage_unavailable_with_its_sqlstate(
    database: Database,
) -> None:
    runs = database.runs()
    with pytest.raises(UsageUnavailable) as caught:
        runs.record(replace(FIXED[0], at="not a time"))
    assert caught.value.code == "22007" and caught.value.kind == "PostgresError"
    runs.record(FIXED[0])  # the failed first call rolled back; the table is created now
    assert database.value("SELECT count(*) FROM {table}") == "1"


# --- tools/setup_usage.py: the table, once, before recording is turned on -------------

SETUP_TOOL = Path(__file__).resolve().parents[1] / "ner-site" / "tools" / "setup_usage.py"


def _setup_tool() -> Any:
    spec = importlib.util.spec_from_file_location("setup_usage_under_test", SETUP_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(
    argv: list[str], env: dict[str, str], executor: Any = None
) -> tuple[int, list[str], list[str]]:
    said: list[str] = []
    opened: list[str] = []

    def factory(database_url: str) -> Any:
        opened.append(database_url)
        if isinstance(executor, Exception):
            raise executor
        return executor

    code = _setup_tool().main(argv, env=env, executor_factory=factory, out=said.append)
    return code, said, opened


def test_the_setup_tool_creates_the_table_and_index_and_can_be_run_again() -> None:
    fake = FakeRuns()
    env = {"DATABASE_URL": SENTINEL_URL}
    for _ in range(2):
        code, said, opened = _setup([], env, fake)
        assert code == 0 and opened == [SENTINEL_URL]
        assert said == ["xwalk.site_runs and its index on at are in place"]
    assert fake.tags() == ["site_runs_schema", "site_runs_table", "site_runs_index"] * 2
    assert fake.created and fake.indexed and fake.rows == [] and fake.round_trips == 2


def test_the_setup_tool_takes_the_schema_from_the_environment_or_the_flag() -> None:
    fake = FakeRuns()
    env = {"DATABASE_URL": SENTINEL_URL, "NER_SITE_NEON_SCHEMA": "demo"}
    assert _setup([], env, fake)[1] == ["demo.site_runs and its index on at are in place"]
    assert "demo.site_runs" in fake.sent[1][0]
    assert _setup(["--schema", "other_one"], env, fake)[1] == [
        "other_one.site_runs and its index on at are in place"
    ]


@pytest.mark.parametrize(
    ("argv", "env", "executor", "code", "said"),
    [
        ([], {}, None, 2, "error: DATABASE_URL is not set"),
        ([], {"DATABASE_URL": "  "}, None, 2, "error: DATABASE_URL is not set"),
        (
            ["--schema", "Bad; DROP"],
            {"DATABASE_URL": SENTINEL_URL},
            FakeRuns(),
            2,
            "error: the schema must be a lower-case identifier (letters, digits, _)",
        ),
        (
            [],
            {"DATABASE_URL": SENTINEL_URL},
            ValueError(f"cannot use {SENTINEL_URL}"),
            2,
            "error: DATABASE_URL is not a postgres:// or postgresql:// URL with a host",
        ),
    ],
)
def test_the_setup_tool_refuses_what_it_cannot_use_without_showing_the_url(
    argv: list[str], env: dict[str, str], executor: Any, code: int, said: str
) -> None:
    assert _setup(argv, env, executor)[:2] == (code, [said])


def test_the_setup_tool_reports_a_database_failure_by_class_and_sqlstate_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeRuns()
    fake.fail = PostgresError(f"could not reach Neon at ep-x-pooler ({SENTINEL_URL})", code="08006")
    code, said, _ = _setup([], {"DATABASE_URL": SENTINEL_URL}, fake)
    assert code == 1
    assert said == ["error: site runs could not be queried (PostgresError 08006)"]
    captured = capsys.readouterr()
    for secret in (SENTINEL_URL, "npg_SENTINELpw0123", "ep-x-pooler"):
        assert secret not in "".join(said) + captured.out + captured.err


def test_the_setup_tool_builds_a_neon_executor_and_prints_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _setup_tool()
    built: list[tuple[str, dict[str, Any]]] = []
    fake = FakeRuns()

    def neon(url: str, **options: Any) -> Any:
        built.append((url, options))
        return fake

    monkeypatch.setattr(tool, "NeonHTTPExecutor", neon)
    assert tool.main([], env={"DATABASE_URL": f"  {SENTINEL_URL} "}) == 0
    assert built == [(SENTINEL_URL, {})] and fake.created
    captured = capsys.readouterr()
    assert captured.out == "xwalk.site_runs and its index on at are in place\n"
    assert "SENTINEL" not in captured.out + captured.err


def test_module_exports() -> None:
    assert set(usage.__all__) == {
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
    }
    for name in usage.__all__:
        assert hasattr(usage, name)
