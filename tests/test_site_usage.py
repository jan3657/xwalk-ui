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

import json
import math
import os
import re
import secrets
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timedelta, timezone
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
        if tag != "site_runs_table" and not self.created:
            raise PostgresError('relation "site_runs" does not exist', code="42P01")
        return getattr(self, f"_{tag}")(sql, p)

    # --- the statements
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
                    **dict.fromkeys(sums, 0),
                },
            )
            group["runs"] += 1
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
    row = run_of(OK_LOG, at=NOW, visitor_day="f" * 32, country="SI", selector_tokens=(700, 70))
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
    row = run_of(log, at=NOW, visitor_day="f" * 32, country=None)
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


def test_a_failed_first_call_creates_the_table_on_the_next() -> None:
    runs, fake = make()
    fake.fail = PostgresError("down")
    with pytest.raises(UsageUnavailable):
        runs.record(a_run())
    fake.fail = None
    runs.record(a_run())
    assert fake.tags().count("site_runs_table") == 2 and len(fake.rows) == 1


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
        **over,
    }


def _priced(*groups: dict[str, Any]) -> dict[str, Any]:
    return shape(_raw(list(groups)), days=1, generated_at=NOW)


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


@pytest.mark.parametrize("selector", ["jev", "retrieval", None])
def test_selection_tokens_of_a_selector_that_is_not_a_model_go_to_no_model(
    selector: str | None,
) -> None:
    answer = _priced(
        _group(selector=selector, selector_prompt_tokens=500, selector_completion_tokens=50)
    )
    assert [row["model"] for row in answer["by_model"]] == [OPUS]
    assert answer["by_model"][0]["prompt_tokens"] == 0
    assert answer["totals"]["prompt_tokens"] == 500 and answer["totals"]["estimated_usd"] == 0.0


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
    assert [row["estimated_usd"] for row in answer["by_model"]] == [0.0, 0.0]


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
    # Outside the three days, on either side.
    a_run(at=at(15, 23, 59, 59), model=OPUS, prompt_tokens=1_000_000, total_ms=9, country="DE"),
    a_run(at=at(19), model=OPUS, prompt_tokens=1_000_000, total_ms=9, country="DE"),
)

EXPECTED: dict[str, Any] = {
    "ok": True,
    "days": 3,
    "generated_at": "2026-11-18T14:20:00Z",
    "totals": {
        "runs": 6,
        "ok": 4,
        "errors": 2,
        "visitors": 3,
        "prompt_tokens": 517_050,
        "completion_tokens": 101_705,
        "estimated_usd": 2.0365,
        "unpriced_runs": 1,
    },
    "by_day": [
        {"day": "2026-11-16", **EMPTY_DAY},
        {
            "day": "2026-11-17",
            "runs": 2,
            "ok": 1,
            "visitors": 1,
            "opus": 1,
            "sonnet": 0,
            "other": 1,
            "own": 0,
            "stepped_down": 2,
            "estimated_usd": 0.0125,
            "p50_ms": 500,
            "p95_ms": 500,
        },
        {
            "day": "2026-11-18",
            "runs": 4,
            "ok": 3,
            "visitors": 2,
            "opus": 1,
            "sonnet": 1,
            "other": 0,
            "own": 1,
            "stepped_down": 1,
            "estimated_usd": 2.024,
            "p50_ms": 2000,
            "p95_ms": 2900,
        },
    ],
    "by_model": [
        # The refused run and the failed one: no model answered.
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
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 4000,
            "completion_tokens": 400,
            "estimated_usd": 0.024,
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
            "model": SONNET,
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 500_000,
            "completion_tokens": 100_000,
            "estimated_usd": 2.0,
        },
        {
            "model": GEMINI,
            "credential": "site",
            "runs": 1,
            "prompt_tokens": 3000,
            "completion_tokens": 300,
            "estimated_usd": 0.0,
        },
    ],
    "by_code": [
        {"code": "provider_error", "runs": 1},
        {"code": "visitor_limit_reached", "runs": 1},
    ],
    "by_fallback": [
        {"reason": "rate_limited", "runs": 2},
        {"reason": "opus_allowance_used", "runs": 1},
        {"reason": "overloaded", "runs": 1},
    ],
    "by_country": [
        {"country": "SI", "runs": 3},
        {"country": "US", "runs": 2},
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
    assert stored.stats(3) == EXPECTED
    # The page draws "refused, no tier" as the runs that are in none of the four tiers.
    for day in stored.stats(30)["by_day"]:
        assert sum(day[tier] for tier in ("opus", "sonnet", "other", "own")) <= day["runs"]


def test_a_shorter_window_leaves_out_the_earlier_day(stored: SiteRuns) -> None:
    answer = stored.stats(1)
    assert answer["by_day"] == [EXPECTED["by_day"][2]]
    assert answer["totals"]["runs"] == 4 and answer["totals"]["visitors"] == 2
    assert answer["by_code"] == [{"code": "visitor_limit_reached", "runs": 1}]
    assert answer["by_fallback"] == [{"reason": "opus_allowance_used", "runs": 1}]
    assert answer["by_country"] == [{"country": "SI", "runs": 2}, {"country": "US", "runs": 2}]


def test_a_longer_window_is_zero_filled_and_takes_in_the_older_run(stored: SiteRuns) -> None:
    answer = stored.stats(30)
    assert len(answer["by_day"]) == 30 and answer["by_day"][0]["day"] == "2026-10-20"
    assert [day["runs"] for day in answer["by_day"][-4:]] == [1, 0, 2, 4]
    assert all(day == {"day": day["day"], **EMPTY_DAY} for day in answer["by_day"][:-4])
    assert answer["totals"]["runs"] == 7
    assert answer["by_country"][-2:] == [{"country": "DE", "runs": 1}, {"country": None, "runs": 1}]
    assert sum(row["runs"] for row in answer["by_country"]) == 7
    assert sum(row["runs"] for row in answer["by_model"]) == 7
    # One million more Opus input tokens on the site's key: four dollars.
    assert answer["totals"]["estimated_usd"] == 6.0365


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


def test_module_exports() -> None:
    assert set(usage.__all__) == {
        "COLUMNS",
        "DEFAULT_DAYS",
        "MAX_DAYS",
        "PRICES",
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
