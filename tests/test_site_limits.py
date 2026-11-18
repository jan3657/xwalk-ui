"""The visitor allowance (own-key daily-cap design, section 3).

Two layers, because the grant rule lives in SQL.

Offline (the default run): ``FakeVisits`` is an in-memory stand-in for the ``site_visits``
table. It was chosen over ``sqlite3`` because the limiter's statements are Postgres only
(a data-modifying CTE, ``FILTER``, ``$n`` placeholders, ``timestamptz``, advisory locks).
It tells the limiter's six statements apart by their ``/* xwalk:<tag> */`` and answers
each with the meaning of its SQL: it keeps the counts, makes the same grant decision the
``site_visit`` statement makes from the parameters it was sent, and refuses a statement
that arrives before the table exists, a visit that does not follow its lock, or a lock
taken before its wait was bounded. It also
records every statement and counts round trips. These tests prove the Python around the
SQL and what is sent; they cannot prove the SQL.

Against Postgres (``integration`` marker, at the end of the file): the same cases run
through the real statements, plus a burst of overlapping requests. They need a Postgres
URL in ``NER_SITE_TEST_DATABASE_URL`` and the ``psql`` client (16 or later, for
``\\bind``), and skip without either:

    NER_SITE_TEST_DATABASE_URL=postgresql://user@localhost:5432/postgres \\
        .venv/bin/python -m pytest -q -m integration tests/test_site_limits.py

No Postgres driver is installed in the project's environment and none may be added, so
``PsqlExecutor`` drives ``psql``. Each test creates a schema of its own and drops it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from ner_site import limits
from ner_site.limits import (
    PRUNE_AFTER_S,
    PRUNE_CHANCE,
    TABLE,
    Admission,
    Allowance,
    LimitsUnavailable,
    VisitorLimitReached,
    VisitorLimits,
    default_salt,
    visitor_id,
)
from xwalk.retrieval.postgres import PostgresError, SQLExecutor, tag_of

# 18 November 2026, 14:20:00 UTC: 40 minutes before the next clock hour.
HOUR = datetime(2026, 11, 18, 14, tzinfo=timezone.utc)
NOW = HOUR.timestamp() + 20 * 60
TO_MIDNIGHT = 9 * 3600 + 40 * 60
SENTINEL_URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"
RAW_IP = "203.0.113.77"
SALT = "test-salt"
VISITOR = "a" * 32
OTHER_VISITOR = "b" * 32


def _when(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None, "a timestamptz parameter must carry its offset"
    return parsed.astimezone(timezone.utc)


class FakeVisits:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, datetime, str], int] = {}
        self.created = False
        self.sent: list[tuple[str, list[Any]]] = []
        self.round_trips = 0
        self.fail: Exception | None = None
        self.locked: set[str] = set()  # visitors locked by the transaction now running
        self.bounded = False  # whether that transaction has set its lock timeout

    # --- the executor protocol
    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        self.round_trips += 1
        return self._run(sql, params)

    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]:
        self.round_trips += 1
        before, created = dict(self.rows), self.created
        try:
            return [self._run(sql, params) for sql, params in statements]
        except Exception:
            self.rows, self.created = before, created  # all or nothing
            raise
        finally:
            self.locked.clear()  # a transaction lock ends with its transaction
            self.bounded = False  # and so does SET LOCAL

    def _run(self, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        p = list(params)
        self.sent.append((sql, p))
        if self.fail is not None:
            raise self.fail
        placeholders = {int(n) for n in re.findall(r"\$(\d+)", sql)}
        assert placeholders == set(range(1, len(p) + 1)), "placeholders and parameters differ"
        tag = tag_of(sql)
        free = ("site_visits_table", "site_visit_lock_timeout", "site_visit_lock")
        if tag not in free and not self.created:
            raise PostgresError('relation "site_visits" does not exist', code="42P01")
        return getattr(self, f"_{tag}")(sql, p)

    # --- the statements
    def _site_visits_table(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "CREATE TABLE IF NOT EXISTS" in sql
        self.created = True
        return []

    def _counts(self, visitor: str, hour: datetime, midnight: datetime) -> dict[str, int]:
        found = {
            key + suffix: 0
            for suffix in ("", "_today")
            for key in ("opus", "sonnet", "own", "total")
        }
        next_midnight = midnight + timedelta(days=1)
        for (who, when, tier), count in self.rows.items():
            if who != visitor or not midnight <= when < next_midnight:
                continue
            for suffix in ("", "_today") if when == hour else ("_today",):
                if tier != "own":
                    found["total" + suffix] += count
                if tier != "other":
                    found[tier + suffix] += count
        return found

    def _site_visit_lock_timeout(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "SET LOCAL lock_timeout = '1500ms'" in sql
        self.bounded = True
        return []

    def _site_visit_lock(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "pg_advisory_xact_lock" in sql
        assert self.bounded, "the lock wait must be bounded before the lock is asked for"
        self.locked.add(p[0])
        return [{"pg_advisory_xact_lock": ""}]

    def _site_visit(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        visitor, hour_text, midnight, wanted, opus, sonnet, opus_day, sonnet_day, own, own_day = p
        assert visitor in self.locked, "a visit must follow its visitor's lock in a transaction"
        hour = _when(hour_text)
        seen = self._counts(visitor, hour, _when(midnight))
        if wanted == "own":
            tier = "own" if seen["own"] < own and seen["own_today"] < own_day else None
        elif seen["total"] >= opus + sonnet or seen["total_today"] >= opus_day + sonnet_day:
            tier = None
        elif wanted == "other":
            tier = "other"
        elif wanted == "opus" and seen["opus"] < opus and seen["opus_today"] < opus_day:
            tier = "opus"
        elif seen["sonnet"] < sonnet and seen["sonnet_today"] < sonnet_day:
            tier = "sonnet"
        else:
            tier = None
        if tier is not None:
            key = (visitor, hour, tier)
            self.rows[key] = self.rows.get(key, 0) + 1
        return [{**seen, "tier": tier}]  # the counts as they were before this visit

    def _site_visits_read(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        visitor, hour_text, midnight = p
        return [self._counts(visitor, _when(hour_text), _when(midnight))]

    def _site_visits_prune(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        (cutoff_text,) = p
        cutoff = _when(cutoff_text)
        self.rows = {key: n for key, n in self.rows.items() if not key[1] < cutoff}
        return []

    # --- what the tests ask
    def tags(self) -> list[str | None]:
        return [tag_of(sql) for sql, _ in self.sent]

    def total(self, visitor: str = VISITOR) -> int:
        return sum(n for (who, _, _), n in self.rows.items() if who == visitor)


class Clock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make(
    allowance: Allowance | None = None,
    *,
    fake: FakeVisits | None = None,
    clock: Clock | None = None,
    rng: float = 0.5,
) -> tuple[VisitorLimits, FakeVisits, Clock]:
    fake = fake or FakeVisits()
    clock = clock or Clock()
    made = VisitorLimits(fake, "xwalk", allowance or Allowance(), clock=clock, rng=lambda: rng)
    return made, fake, clock


# --- the allowance -----------------------------------------------------------------


def test_defaults_and_fake_is_an_executor() -> None:
    assert (Allowance().opus_per_hour, Allowance().sonnet_per_hour) == (10, 20)
    assert TABLE == "site_visits"
    assert (PRUNE_AFTER_S, PRUNE_CHANCE) == (86400, 0.02)
    assert isinstance(FakeVisits(), SQLExecutor)


def test_first_ten_opus_visits_are_opus() -> None:
    lim, fake, _ = make()
    got = [lim.admit(VISITOR, "opus") for _ in range(10)]
    assert [a.tier for a in got] == ["opus"] * 10
    assert [a.opus_left for a in got] == list(range(9, -1, -1))
    assert [a.sonnet_left for a in got] == [20] * 10
    assert all(a.reason is None for a in got)
    assert all(a.resets_in_s == 40 * 60 for a in got)
    assert fake.rows == {(VISITOR, HOUR, "opus"): 10}


def test_eleventh_is_sonnet_thirtieth_is_last_thirty_first_is_refused() -> None:
    lim, fake, _ = make()
    got = [lim.admit(VISITOR, "opus") for _ in range(30)]
    assert got[10] == Admission(
        "sonnet", 0, 19, 40 * 60, "opus_allowance_used", 10, 39, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert [a.tier for a in got[10:]] == ["sonnet"] * 20
    assert all(a.reason == "opus_allowance_used" for a in got[10:])
    assert got[29].sonnet_left == 0
    before = dict(fake.rows)
    with pytest.raises(VisitorLimitReached) as caught:
        lim.admit(VISITOR, "opus")
    assert caught.value.window == "hour"
    assert 1 <= caught.value.resets_in_s <= 3600
    assert caught.value.resets_in_s == 40 * 60
    assert fake.rows == before == {(VISITOR, HOUR, "opus"): 10, (VISITOR, HOUR, "sonnet"): 20}


def test_wanting_sonnet_uses_only_the_sonnet_allowance() -> None:
    lim, fake, _ = make()
    got = [lim.admit(VISITOR, "sonnet") for _ in range(20)]
    assert [a.tier for a in got] == ["sonnet"] * 20
    assert [a.sonnet_left for a in got] == list(range(19, -1, -1))
    assert all(a.opus_left == 10 and a.reason is None for a in got)
    with pytest.raises(VisitorLimitReached):
        lim.admit(VISITOR, "sonnet")
    assert fake.rows == {(VISITOR, HOUR, "sonnet"): 20}
    # The Opus allowance is untouched: an Opus request is still served on Opus.
    assert lim.admit(VISITOR, "opus").tier == "opus"


def test_other_counts_toward_the_total_only() -> None:
    lim, fake, _ = make()
    got = [lim.admit(VISITOR, "other") for _ in range(10)]
    assert all(
        a == Admission("other", 10, 20, 40 * 60, None, 20, 40, 30, 60, TO_MIDNIGHT, "hour")
        for a in got
    )
    assert lim.admit(VISITOR, "opus") == Admission(
        "opus", 9, 19, 40 * 60, None, 19, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    for _ in range(14):
        lim.admit(VISITOR, "other")
    assert [lim.admit(VISITOR, "opus").tier for _ in range(5)] == ["opus"] * 5
    assert fake.total() == 30
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached):
            lim.admit(VISITOR, wanted)
    assert fake.rows == {(VISITOR, HOUR, "other"): 24, (VISITOR, HOUR, "opus"): 6}


def test_what_is_left_is_capped_by_the_rest_of_the_hours_total() -> None:
    lim, _, _ = make()
    # 30 in the hour. Other models use up the total, so less than a full Claude allowance
    # can be left although no Claude request has been made.
    got = [lim.admit(VISITOR, "other") for _ in range(25)]
    assert [(a.opus_left, a.sonnet_left) for a in got[:10]] == [(10, 20)] * 10
    assert [(a.opus_left, a.sonnet_left) for a in got[10:20]] == [(10, n) for n in range(19, 9, -1)]
    assert [(a.opus_left, a.sonnet_left) for a in got[20:]] == [(n, n) for n in range(9, 4, -1)]
    assert lim.peek(VISITOR) == Admission(
        "opus", 5, 5, 40 * 60, None, 20, 35, 30, 60, TO_MIDNIGHT, "hour"
    )
    opus = [lim.admit(VISITOR, "opus") for _ in range(5)]
    assert [(a.tier, a.opus_left, a.sonnet_left) for a in opus] == [
        ("opus", n, n) for n in range(4, -1, -1)
    ]
    assert lim.peek(VISITOR) == Admission(
        "blocked", 0, 0, 40 * 60, None, 15, 30, 30, 60, TO_MIDNIGHT, "hour"
    )


def test_a_second_visitor_is_unaffected() -> None:
    lim, fake, _ = make()
    for _ in range(30):
        lim.admit(VISITOR, "opus")
    assert lim.admit(OTHER_VISITOR, "opus") == Admission(
        "opus", 9, 20, 40 * 60, None, 19, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert fake.total(OTHER_VISITOR) == 1


def test_the_next_clock_hour_starts_fresh() -> None:
    lim, fake, clock = make()
    for _ in range(30):
        lim.admit(VISITOR, "opus")
    clock.now = HOUR.timestamp() + 3600
    assert lim.admit(VISITOR, "opus") == Admission(
        "opus", 9, 20, 3600, None, 9, 20, 30, 60, 9 * 3600, "hour"
    )
    assert fake.rows[(VISITOR, HOUR + timedelta(hours=1), "opus")] == 1


def test_resets_in_is_at_least_one_second() -> None:
    lim, _, clock = make()
    clock.now = HOUR.timestamp() + 3599.9
    assert lim.admit(VISITOR, "opus").resets_in_s == 1
    clock.now = HOUR.timestamp() + 3598.5
    assert lim.peek(VISITOR).resets_in_s == 2


def test_an_unknown_tier_is_refused_before_any_query() -> None:
    lim, fake, _ = make()
    with pytest.raises(ValueError):
        lim.admit(VISITOR, "haiku")  # type: ignore[arg-type]
    assert fake.sent == []


# --- peek --------------------------------------------------------------------------


def test_peek_reports_the_same_numbers_and_records_nothing() -> None:
    lim, fake, _ = make()
    assert lim.peek(VISITOR) == Admission(
        "opus", 10, 20, 40 * 60, None, 20, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert fake.rows == {}
    for count in range(1, 31):
        admitted = lim.admit(VISITOR, "opus")
        before = dict(fake.rows)
        peeked = lim.peek(VISITOR)
        assert fake.rows == before
        assert replace(peeked, tier=admitted.tier, reason=admitted.reason) == admitted
        if count < 10:
            assert (peeked.tier, peeked.reason) == ("opus", None)
        elif count < 30:
            assert (peeked.tier, peeked.reason) == ("sonnet", "opus_allowance_used")
    assert lim.peek(VISITOR) == Admission(
        "blocked", 0, 0, 40 * 60, None, 10, 20, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert fake.total() == 30


def test_peek_is_blocked_at_the_total_even_with_opus_unused() -> None:
    lim, _, _ = make()
    for _ in range(30):
        lim.admit(VISITOR, "other")
    assert lim.peek(VISITOR) == Admission(
        "blocked", 0, 0, 40 * 60, None, 20, 30, 30, 60, TO_MIDNIGHT, "hour"
    )


# --- allowances of zero ------------------------------------------------------------


def test_opus_allowance_of_zero_starts_on_sonnet() -> None:
    lim, fake, _ = make(Allowance(opus_per_hour=0, sonnet_per_hour=2))
    assert lim.peek(VISITOR) == Admission(
        "sonnet", 0, 2, 40 * 60, "opus_allowance_used", 20, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert lim.admit(VISITOR, "opus") == Admission(
        "sonnet", 0, 1, 40 * 60, "opus_allowance_used", 20, 39, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert lim.admit(VISITOR, "opus").sonnet_left == 0
    with pytest.raises(VisitorLimitReached):
        lim.admit(VISITOR, "opus")
    assert fake.rows == {(VISITOR, HOUR, "sonnet"): 2}


def test_both_allowances_of_zero_refuse_the_first_visit() -> None:
    lim, fake, _ = make(Allowance(opus_per_hour=0, sonnet_per_hour=0))
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached):
            lim.admit(VISITOR, wanted)
    assert lim.peek(VISITOR).tier == "blocked"
    assert fake.rows == {}


# Exercise the same boundary cases against the in-memory table and the actual SQL.
Metered = tuple[Callable[..., VisitorLimits], Clock]


@pytest.fixture(params=["fake", pytest.param("postgres", marks=pytest.mark.integration)])
def metered(request: pytest.FixtureRequest) -> Metered:
    if request.param == "postgres":
        db = request.getfixturevalue("database")
        return db.limits, db.clock
    clock = Clock()
    return lambda allowance=None: make(allowance, clock=clock)[0], clock


def test_twentieth_opus_of_the_day_is_last(metered: Metered) -> None:
    build, clock = metered
    lim = build()
    for hour, visits in enumerate((7, 7, 5)):
        clock.now = NOW + hour * 3600
        for _ in range(visits):
            assert lim.admit(VISITOR, "opus").tier == "opus"
    last = lim.admit(VISITOR, "opus")
    assert (last.tier, last.opus_left, last.opus_left_today, last.window) == ("opus", 0, 0, "day")
    next_visit = lim.admit(VISITOR, "opus")
    assert (next_visit.tier, next_visit.reason, next_visit.window) == (
        "sonnet",
        "opus_allowance_used",
        "day",
    )


@pytest.mark.parametrize(
    "later_hour",
    [HOUR.replace(hour=15), datetime(2026, 3, 29, 23, tzinfo=timezone.utc)],
    ids=["15h", "last-utc-hour-on-dst-change"],
)
def test_delayed_visit_sees_a_later_hours_daily_opus_spend(
    metered: Metered, later_hour: datetime
) -> None:
    build, clock = metered
    lim = build(Allowance(opus_per_day=1))
    clock.now = later_hour.timestamp()
    assert lim.admit(VISITOR, "opus").tier == "opus"
    clock.now = (later_hour - timedelta(minutes=1)).timestamp()
    delayed = lim.admit(VISITOR, "opus")
    assert (delayed.tier, delayed.reason, delayed.window) == (
        "sonnet",
        "opus_allowance_used",
        "day",
    )


def test_sixtieth_site_visit_is_last_until_utc_midnight(metered: Metered) -> None:
    build, clock = metered
    lim = build()
    for _ in range(30):
        lim.admit(VISITOR, "opus")
    clock.now += 3600
    for _ in range(30):
        last = lim.admit(VISITOR, "opus")
    assert (last.tier, last.sonnet_left_today) == ("sonnet", 0)
    before = lim.peek(VISITOR)
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached) as caught:
            lim.admit(VISITOR, wanted)
        assert (caught.value.window, caught.value.resets_in_s) == ("day", 8 * 3600 + 40 * 60)
    assert before == lim.peek(VISITOR)
    assert (before.tier, before.window) == ("blocked", "day")
    assert lim.admit(VISITOR, "own").tier == "own"


def test_midnight_restores_both_windows(metered: Metered) -> None:
    build, clock = metered
    lim = build()
    midnight = (HOUR + timedelta(hours=10)).timestamp()
    clock.now = midnight - 0.2
    for wanted in ("opus", "sonnet", "own", "other"):
        last = lim.admit(VISITOR, wanted)
    assert (last.resets_in_s, last.day_resets_in_s) == (1, 1)
    clock.now = midnight
    assert lim.peek(VISITOR) == Admission("opus", 10, 20, 3600, None, 20, 40, 30, 60, 86400, "hour")


def test_other_spends_both_site_totals(metered: Metered) -> None:
    build, clock = metered
    lim = build(Allowance(2, 2, 3, 2))
    for _ in range(3):
        last = lim.admit(VISITOR, "other")
    assert (last.opus_left, last.sonnet_left) == (1, 1)
    assert (last.opus_left_today, last.sonnet_left_today) == (2, 2)
    clock.now += 3600
    assert lim.admit(VISITOR, "sonnet").opus_left == 1
    assert lim.admit(VISITOR, "other").window == "day"
    with pytest.raises(VisitorLimitReached) as caught:
        lim.admit(VISITOR, "other")
    assert caught.value.window == "day"


def test_own_visits_have_independent_hourly_and_daily_limits(metered: Metered) -> None:
    build, clock = metered
    lim = build()
    for hour in range(2):
        for wanted in ("opus", "sonnet", "other"):
            site = lim.admit(VISITOR, wanted)
            assert (site.own_left, site.own_left_today) == (30, 60 - hour * 30)
        for n in range(1, 31):
            own = lim.admit(VISITOR, "own")
            assert own.tier == "own" and own.reason is None
            assert (own.own_left, own.own_left_today) == (30 - n, 60 - hour * 30 - n)
            assert (own.opus_left, own.sonnet_left) == (site.opus_left, site.sonnet_left)
            assert (own.opus_left_today, own.sonnet_left_today) == (
                site.opus_left_today,
                site.sonnet_left_today,
            )
        before = lim.peek(VISITOR)
        with pytest.raises(VisitorLimitReached) as caught:
            lim.admit(VISITOR, "own")
        expected = ("hour", 2400) if hour == 0 else ("day", 8 * 3600 + 2400)
        assert (caught.value.window, caught.value.resets_in_s) == expected
        assert lim.peek(VISITOR) == before
        clock.now += 3600
    assert lim.peek(VISITOR).own_left == 0
    with pytest.raises(VisitorLimitReached) as caught:
        lim.admit(VISITOR, "own")
    assert caught.value.window == "day"
    assert lim.admit(VISITOR, "opus").tier == "opus"


def test_hourly_opus_room_is_capped_by_daily_opus_room(metered: Metered) -> None:
    build, clock = metered
    lim = build(Allowance(opus_per_day=3))
    lim.admit(VISITOR, "opus")
    clock.now += 3600
    got = lim.admit(VISITOR, "opus")
    assert got.opus_left == got.opus_left_today == 1
    assert got == lim.peek(VISITOR) == lim.peek(VISITOR)


def test_retry_after_an_hour_can_admit_opus_on_sonnet(metered: Metered) -> None:
    build, clock = metered
    lim = build(Allowance(1, 1, 1, 3))
    lim.admit(VISITOR, "opus")
    lim.admit(VISITOR, "sonnet")
    with pytest.raises(VisitorLimitReached) as caught:
        lim.admit(VISITOR, "opus")
    assert (caught.value.window, caught.value.resets_in_s) == ("hour", 2400)
    clock.now += caught.value.resets_in_s
    assert lim.admit(VISITOR, "opus").tier == "sonnet"


@pytest.mark.parametrize(
    ("cap", "wanted", "window"),
    [
        ("sonnet_per_day", "sonnet", "day"),
        ("own_per_hour", "own", "hour"),
        ("own_per_day", "own", "day"),
    ],
)
def test_zero_cap_refuses_without_recording(
    metered: Metered, cap: str, wanted: limits.Tier, window: str
) -> None:
    build, _ = metered
    lim = build(Allowance(**{cap: 0}))
    before = lim.peek(VISITOR)
    with pytest.raises(VisitorLimitReached) as caught:
        lim.admit(VISITOR, wanted)
    assert caught.value.window == window
    assert lim.peek(VISITOR) == before


def test_zero_daily_opus_starts_on_sonnet(metered: Metered) -> None:
    build, _ = metered
    lim = build(Allowance(opus_per_day=0))
    last = lim.admit(VISITOR, "opus")
    assert (last.tier, last.reason, last.window) == ("sonnet", "opus_allowance_used", "day")
    assert last.opus_left == last.opus_left_today == 0


# --- round trips, the table, pruning -----------------------------------------------


def test_one_round_trip_per_call_and_the_table_is_created_once() -> None:
    lim, fake, _ = make()
    lim.admit(VISITOR, "opus")
    assert fake.round_trips == 1
    assert fake.tags() == [
        "site_visits_table", "site_visit_lock_timeout", "site_visit_lock", "site_visit",
    ]  # fmt: skip
    lim.admit(VISITOR, "opus")
    lim.peek(VISITOR)
    assert fake.round_trips == 3
    assert fake.tags()[4:] == [
        "site_visit_lock_timeout", "site_visit_lock", "site_visit", "site_visits_read",
    ]  # fmt: skip
    no_table = ("site_visit_lock_timeout", "site_visit_lock")
    assert all(f"xwalk.{TABLE}" in sql for sql, _ in fake.sent if tag_of(sql) not in no_table)


def test_every_visit_locks_its_visitor_first_in_a_statement_of_its_own() -> None:
    # The fake refuses a visit that does not follow its visitor's lock in one transaction.
    lim, fake, _ = make(rng=0.0)
    lim.admit(VISITOR, "opus")
    lim.admit(OTHER_VISITOR, "own")
    assert fake.round_trips == 2
    assert fake.tags() == [
        "site_visits_table",
        "site_visit_lock_timeout", "site_visit_lock", "site_visit", "site_visits_prune",
        "site_visit_lock_timeout", "site_visit_lock", "site_visit", "site_visits_prune",
    ]  # fmt: skip
    locks = [(sql, params) for sql, params in fake.sent if tag_of(sql) == "site_visit_lock"]
    assert [params for _, params in locks] == [[VISITOR], [OTHER_VISITOR]]
    assert all("pg_advisory_xact_lock(hashtextextended($1::text, 0))" in sql for sql, _ in locks)
    visit = next(sql for sql, _ in fake.sent if tag_of(sql) == "site_visit")
    assert "pg_advisory" not in visit


def test_the_lock_wait_is_bounded_below_the_executors_timeout() -> None:
    """The fake refuses a lock whose transaction has not set ``lock_timeout`` first. The
    bound is a constant in the statement (``SET`` takes no parameter) and well under the
    3 s the limiter's executor waits for the whole round trip (``http.LIMITS_TIMEOUT_S``),
    so a queue behind one visitor's lock ends as SQLSTATE 55P03, not as a timed-out call."""
    from ner_site import http as site_http

    lim, fake, _ = make()
    lim.admit(VISITOR, "opus")
    (bound,) = [(sql, params) for sql, params in fake.sent if "lock_timeout" in sql]
    assert bound == (
        "/* xwalk:site_visit_lock_timeout */\nSET LOCAL lock_timeout = '1500ms'",
        [],
    )
    assert limits.LOCK_TIMEOUT_MS == 1500
    assert limits.LOCK_TIMEOUT_MS / 1000 < site_http.LIMITS_TIMEOUT_S
    assert limits.LOCK_NOT_AVAILABLE == "55P03"


def test_peek_takes_no_lock() -> None:
    lim, fake, _ = make()
    lim.peek(VISITOR)
    lim.peek(VISITOR)
    assert "site_visit_lock" not in fake.tags()
    assert "site_visit_lock_timeout" not in fake.tags()


def test_peek_on_a_new_instance_creates_the_table_in_the_same_round_trip() -> None:
    lim, fake, _ = make()
    assert lim.peek(VISITOR).tier == "opus"
    assert fake.round_trips == 1
    assert fake.tags() == ["site_visits_table", "site_visits_read"]


def test_a_second_instance_creates_the_table_again_harmlessly() -> None:
    first, fake, _ = make()
    first.admit(VISITOR, "opus")
    second, _, _ = make(fake=fake)
    assert second.admit(VISITOR, "opus").opus_left == 8
    assert fake.tags().count("site_visits_table") == 2


def test_every_value_is_a_parameter() -> None:
    lim, fake, _ = make(Allowance(7, 13, 11, 31, 17, 47), rng=0.0)
    lim.admit(VISITOR, "opus")
    lim.peek(VISITOR)
    by_tag = {tag_of(sql): (sql, params) for sql, params in fake.sent}
    bounds = [VISITOR, "2026-11-18T14:00:00+00:00", "2026-11-18T00:00:00+00:00"]
    assert by_tag["site_visit"][1] == [*bounds, "opus", 7, 13, 11, 31, 17, 47]
    assert by_tag["site_visits_read"][1] == bounds
    assert by_tag["site_visits_prune"][1] == ["2026-11-17T14:20:00+00:00"]
    for sql, _ in fake.sent:
        assert VISITOR not in sql and "2026" not in sql


def test_pruning_runs_when_the_draw_is_low_and_deletes_only_old_rows() -> None:
    lim, fake, _ = make(rng=0.0)
    old = (OTHER_VISITOR, HOUR - timedelta(hours=25), "opus")
    recent = (OTHER_VISITOR, HOUR - timedelta(hours=23), "sonnet")
    fake.created = True
    fake.rows.update({old: 4, recent: 6})
    lim.admit(VISITOR, "opus")
    assert fake.round_trips == 1
    assert "site_visits_prune" in fake.tags()
    assert fake.rows == {recent: 6, (VISITOR, HOUR, "opus"): 1}


def test_pruning_does_not_run_when_the_draw_is_high() -> None:
    lim, fake, _ = make(rng=0.99)
    old = (OTHER_VISITOR, HOUR - timedelta(hours=25), "opus")
    fake.created = True
    fake.rows[old] = 4
    lim.admit(VISITOR, "opus")
    lim.peek(VISITOR)
    assert "site_visits_prune" not in fake.tags()
    assert fake.rows[old] == 4


def test_pruning_draw_is_compared_with_prune_chance() -> None:
    below, fake_below, _ = make(rng=PRUNE_CHANCE - 0.001)
    below.admit(VISITOR, "opus")
    assert "site_visits_prune" in fake_below.tags()
    at, fake_at, _ = make(rng=PRUNE_CHANCE)
    at.admit(VISITOR, "opus")
    assert "site_visits_prune" not in fake_at.tags()


def test_peek_never_prunes() -> None:
    lim, fake, _ = make(rng=0.0)
    lim.peek(VISITOR)
    assert "site_visits_prune" not in fake.tags()


# --- failures ----------------------------------------------------------------------


@pytest.mark.parametrize("call", ["admit", "peek"])
def test_an_executor_failure_is_limits_unavailable_without_secrets(call: str) -> None:
    lim, fake, _ = make()
    fake.fail = PostgresError(f"could not reach Neon with {SENTINEL_URL}", code="08006")
    with pytest.raises(LimitsUnavailable) as caught:
        lim.admit(VISITOR, "opus") if call == "admit" else lim.peek(VISITOR)
    shown = f"{caught.value} {caught.value!r} {caught.value.__cause__!r}"
    for secret in (SENTINEL_URL, "npg_SENTINELpw0123", "neon.tech", "owner"):
        assert secret not in shown
    assert caught.value.__suppress_context__  # a traceback does not print the cause either


def test_any_exception_from_the_executor_is_limits_unavailable() -> None:
    lim, fake, _ = make()
    fake.fail = RuntimeError(f"boom {SENTINEL_URL}")
    with pytest.raises(LimitsUnavailable) as caught:
        lim.admit(VISITOR, "opus")
    assert "SENTINEL" not in str(caught.value)
    assert caught.value.code is None


@pytest.mark.parametrize(
    ("given", "code"),
    [("55P03", "55P03"), ("40001", "40001"), (None, None), ("not a state", None), (55, None)],
)
def test_limits_unavailable_carries_the_sqlstate(given: Any, code: str | None) -> None:
    """The caller tells a contended lock (55P03) from a database that cannot be asked."""
    lim, fake, _ = make()
    fake.fail = PostgresError("canceling statement due to lock timeout", code="none")
    fake.fail.code = given
    with pytest.raises(LimitsUnavailable) as caught:
        lim.admit(VISITOR, "opus")
    assert caught.value.code == code
    assert (code or "PostgresError") in str(caught.value)
    with pytest.raises(LimitsUnavailable) as unreadable:
        make(fake=type("Empty", (FakeVisits,), {"_site_visit": lambda *a: []})())[0].admit(
            VISITOR, "opus"
        )
    assert unreadable.value.code is None


def test_an_unreadable_answer_is_limits_unavailable() -> None:
    class Empty(FakeVisits):
        def _site_visit(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
            return []

    lim, _, _ = make(fake=Empty())
    with pytest.raises(LimitsUnavailable):
        lim.admit(VISITOR, "opus")


def test_a_failed_first_call_creates_the_table_on_the_next() -> None:
    lim, fake, _ = make()
    fake.fail = PostgresError("down")
    with pytest.raises(LimitsUnavailable):
        lim.admit(VISITOR, "opus")
    fake.fail = None
    assert lim.admit(VISITOR, "opus").opus_left == 9
    assert fake.created


def test_counts_that_arrive_as_strings_are_read() -> None:
    class Stringly(FakeVisits):
        def _site_visits_read(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
            return [{k: str(v) for k, v in super()._site_visits_read(sql, p)[0].items()}]

    lim, _, _ = make(fake=Stringly())
    lim.admit(VISITOR, "opus")
    assert lim.peek(VISITOR).opus_left == 9


@pytest.mark.parametrize(
    "schema", ["Xwalk", "xwalk; drop table t", "1abc", "", "a.b", 'x"y', "x" * 64, "x-y"]
)
def test_a_schema_that_is_not_a_lower_case_identifier_is_refused(schema: str) -> None:
    with pytest.raises(ValueError):
        VisitorLimits(FakeVisits(), schema, Allowance())


def test_a_plain_schema_is_accepted() -> None:
    fake = FakeVisits()
    lim = VisitorLimits(fake, "ner_site_2", Allowance(), clock=Clock(), rng=lambda: 0.5)
    lim.admit(VISITOR, "opus")
    lim.peek(VISITOR)
    no_table = ("site_visit_lock_timeout", "site_visit_lock")
    tables = [sql for sql, _ in fake.sent if tag_of(sql) not in no_table]
    assert len(tables) == 3 and all(f"ner_site_2.{TABLE}" in sql for sql in tables)


# --- the visitor id ----------------------------------------------------------------


def _expected(ip: str, salt: str = SALT) -> str:
    return hmac.new(salt.encode(), ip.encode(), hashlib.sha256).hexdigest()[:32]


def test_visitor_id_is_the_keyed_hash_of_the_address() -> None:
    got = visitor_id({"x-real-ip": RAW_IP}, SALT)
    assert got == _expected(RAW_IP)
    assert re.fullmatch(r"[0-9a-f]{32}", got)
    assert RAW_IP not in got


def test_visitor_id_header_priority() -> None:
    every = {
        "x-real-ip": "198.51.100.1",
        "x-vercel-forwarded-for": "198.51.100.2",
        "x-forwarded-for": "198.51.100.3",
    }
    assert visitor_id(every, SALT) == _expected("198.51.100.1")
    del every["x-real-ip"]
    assert visitor_id(every, SALT) == _expected("198.51.100.2")
    del every["x-vercel-forwarded-for"]
    assert visitor_id(every, SALT) == _expected("198.51.100.3")


def test_visitor_id_skips_empty_headers() -> None:
    headers = {"x-real-ip": "  ", "x-vercel-forwarded-for": "", "x-forwarded-for": RAW_IP}
    assert visitor_id(headers, SALT) == _expected(RAW_IP)
    assert visitor_id({"x-forwarded-for": " , 10.0.0.1"}, SALT) == _expected("")


def test_visitor_id_takes_the_first_forwarded_entry_and_strips_whitespace() -> None:
    assert visitor_id({"x-forwarded-for": f" {RAW_IP} , 10.0.0.1, 10.0.0.2"}, SALT) == _expected(
        RAW_IP
    )
    assert visitor_id({"x-real-ip": f"  {RAW_IP}\t"}, SALT) == _expected(RAW_IP)


def test_visitor_id_without_headers_is_the_id_of_the_empty_string() -> None:
    assert visitor_id({}, SALT) == _expected("")
    assert visitor_id({"user-agent": "x"}, SALT) == visitor_id({}, SALT)


def test_visitor_id_differs_by_salt_and_by_address() -> None:
    assert visitor_id({"x-real-ip": RAW_IP}, "one") != visitor_id({"x-real-ip": RAW_IP}, "two")
    assert visitor_id({"x-real-ip": RAW_IP}, SALT) != visitor_id(
        {"x-real-ip": "203.0.113.78"}, SALT
    )


def test_default_salt_is_the_digest_of_the_database_url() -> None:
    salt = default_salt(SENTINEL_URL)
    assert salt == hashlib.sha256(SENTINEL_URL.encode()).hexdigest()
    assert "SENTINEL" not in salt
    assert default_salt("postgresql://other/db") != salt


def test_no_sql_parameter_ever_holds_the_raw_address() -> None:
    lim, fake, _ = make(rng=0.0)
    who = visitor_id({"x-forwarded-for": f"{RAW_IP}, 10.0.0.1"}, SALT)
    for wanted in ("opus", "sonnet", "other", "own"):
        lim.admit(who, wanted)
    lim.peek(who)
    assert fake.sent
    for sql, params in fake.sent:
        assert RAW_IP not in sql
        assert all(RAW_IP not in str(value) for value in params)


def test_module_exports() -> None:
    for name in limits.__all__:
        assert hasattr(limits, name)
    assert {"VisitorLimits", "visitor_id", "default_salt", "Allowance", "Admission"} <= set(
        limits.__all__
    )


# --- against Postgres --------------------------------------------------------------

DATABASE_URL_ENV = "NER_SITE_TEST_DATABASE_URL"
_SEPARATOR = "-----statement-----"


class PsqlExecutor:
    """The smallest ``SQLExecutor`` over the ``psql`` client that the limiter needs.

    Parameters are bound as text (``\bind``), as Neon's HTTPS endpoint sends them. A
    transaction is one ``psql`` session at READ COMMITTED, the level Neon's endpoint uses
    when none is asked for. At most one row per statement is read back, and every value
    comes back as a string (NULL as None), which is also the least a real executor may
    give. A failed statement is a ``PostgresError`` with its SQLSTATE, as from the real
    executor. The URL goes to ``psql`` through ``PG*`` environment variables, never on
    the command line, and appears in no message.
    """

    def __init__(self, url: str) -> None:
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        settings = {
            "PGHOST": (query.get("host") or [parts.hostname or ""])[0],
            "PGPORT": str(parts.port or ""),
            "PGUSER": unquote(parts.username or ""),
            "PGPASSWORD": unquote(parts.password or ""),
            "PGDATABASE": unquote(parts.path.lstrip("/")),
            "PGSSLMODE": (query.get("sslmode") or [""])[0],
            "PGCONNECT_TIMEOUT": "10",
        }
        self._env = {**os.environ, **{k: v for k, v in settings.items() if v}}

    @staticmethod
    def _statement(sql: str, params: Sequence[Any]) -> str:
        bound = " ".join("'" + str(p).replace("'", "''") + "'" for p in params)
        return f"{sql}\n\\bind {bound} \\gx\n"

    def spawn(self) -> subprocess.Popen[str]:
        return subprocess.Popen(
            ["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-v", "VERBOSITY=verbose"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._env,
        )

    def _run(self, script: str) -> str:
        process = self.spawn()
        out, err = process.communicate(script, timeout=60)
        if process.returncode:
            # VERBOSITY=verbose: "ERROR:  55P03: canceling statement due to lock timeout".
            state = re.search(r"^[A-Z]+:\s+([0-9A-Z]{5}):", err, re.MULTILINE)
            raise PostgresError("psql failed", code=state.group(1) if state else None)
        return out

    @staticmethod
    def _rows(out: str) -> list[dict[str, Any]]:
        lines = [line for line in out.splitlines() if "|" in line]
        row = dict(line.split("|", 1) for line in lines)
        return [{name: value or None for name, value in row.items()}] if row else []

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return self._rows(self._run(self._statement(sql, params)))

    def transaction(
        self, statements: Sequence[tuple[str, Sequence[Any]]], *, read_only: bool = False
    ) -> list[list[dict[str, Any]]]:
        body = f"\\echo {_SEPARATOR}\n".join(self._statement(s, p) for s, p in statements)
        out = self._run(f"BEGIN ISOLATION LEVEL READ COMMITTED;\n{body}COMMIT;\n")
        return [self._rows(part) for part in out.split(_SEPARATOR + "\n")]


class Database:
    """A throwaway schema on the Postgres of ``NER_SITE_TEST_DATABASE_URL``."""

    def __init__(self, executor: PsqlExecutor, schema: str) -> None:
        self.executor = executor
        self.schema = schema
        self.clock = Clock()

    def limits(self, allowance: Allowance | None = None, *, rng: float = 0.5) -> VisitorLimits:
        return VisitorLimits(
            self.executor, self.schema, allowance or Allowance(), clock=self.clock, rng=lambda: rng
        )

    def counts(self, visitor: str) -> dict[str, int]:
        """What the table holds for a visitor, per tier, over every hour."""
        rows = self.executor.execute(
            "SELECT coalesce(json_object_agg(t.tier, t.n), '{}') AS counts FROM ("
            f"SELECT tier, sum(count) AS n FROM {self.schema}.{TABLE} "
            "WHERE visitor = $1::text GROUP BY tier) AS t",
            (visitor,),
        )
        return json.loads(rows[0]["counts"])


@pytest.fixture
def database() -> Iterator[Database]:
    url = os.environ.get(DATABASE_URL_ENV, "").strip()
    if not url:
        pytest.skip(f"{DATABASE_URL_ENV} is not set")
    if shutil.which("psql") is None:
        pytest.skip("the psql client is not installed")
    executor = PsqlExecutor(url)
    schema = f"limits_test_{secrets.token_hex(4)}"
    executor.execute(f"CREATE SCHEMA {schema}")
    try:
        yield Database(executor, schema)
    finally:
        executor.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.mark.integration
def test_postgres_ten_opus_then_twenty_sonnet_then_refused(database: Database) -> None:
    lim = database.limits()
    assert lim.peek(VISITOR) == Admission(
        "opus", 10, 20, 40 * 60, None, 20, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    got = [lim.admit(VISITOR, "opus") for _ in range(30)]
    assert [a.tier for a in got] == ["opus"] * 10 + ["sonnet"] * 20
    assert [a.opus_left for a in got[:10]] == list(range(9, -1, -1))
    assert all(a.reason is None and a.sonnet_left == 20 for a in got[:10])
    assert got[10] == Admission(
        "sonnet", 0, 19, 40 * 60, "opus_allowance_used", 10, 39, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert [a.sonnet_left for a in got[10:]] == list(range(19, -1, -1))
    assert lim.peek(VISITOR) == Admission(
        "blocked", 0, 0, 40 * 60, None, 10, 20, 30, 60, TO_MIDNIGHT, "hour"
    )
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached) as caught:
            lim.admit(VISITOR, wanted)
        assert caught.value.resets_in_s == 40 * 60
    assert database.counts(VISITOR) == {"opus": 10, "sonnet": 20}
    # A second visitor is unaffected, and the next clock hour starts fresh.
    assert lim.admit(OTHER_VISITOR, "opus") == Admission(
        "opus", 9, 20, 40 * 60, None, 19, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    database.clock.now = HOUR.timestamp() + 3600
    assert lim.admit(VISITOR, "opus") == Admission(
        "opus", 9, 20, 3600, None, 9, 20, 30, 60, 9 * 3600, "hour"
    )
    assert database.counts(VISITOR) == {"opus": 11, "sonnet": 20}
    stored = database.executor.execute(
        f"SELECT to_char(min(hour) AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') AS first, "
        f"count(DISTINCT hour) AS hours FROM {database.schema}.{TABLE} WHERE visitor = $1::text",
        (VISITOR,),
    )
    assert stored == [{"first": "2026-11-18 14:00:00", "hours": "2"}]


@pytest.mark.integration
def test_postgres_sonnet_uses_only_the_sonnet_allowance(database: Database) -> None:
    lim = database.limits()
    got = [lim.admit(VISITOR, "sonnet") for _ in range(20)]
    assert [(a.tier, a.sonnet_left, a.reason) for a in got] == [
        ("sonnet", n, None) for n in range(19, -1, -1)
    ]
    assert all(a.opus_left == 10 for a in got)
    with pytest.raises(VisitorLimitReached):
        lim.admit(VISITOR, "sonnet")
    assert database.counts(VISITOR) == {"sonnet": 20}
    assert lim.peek(VISITOR) == Admission(
        "opus", 10, 0, 40 * 60, None, 20, 20, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert lim.admit(VISITOR, "opus") == Admission(
        "opus", 9, 0, 40 * 60, None, 19, 20, 30, 60, TO_MIDNIGHT, "hour"
    )


@pytest.mark.integration
def test_postgres_other_counts_toward_the_total_only(database: Database) -> None:
    lim = database.limits()
    got = [lim.admit(VISITOR, "other") for _ in range(25)]
    assert all(a.tier == "other" and a.reason is None for a in got)
    assert (got[9].opus_left, got[9].sonnet_left) == (10, 20)
    assert (got[24].opus_left, got[24].sonnet_left) == (5, 5)
    assert lim.peek(VISITOR) == Admission(
        "opus", 5, 5, 40 * 60, None, 20, 35, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert [lim.admit(VISITOR, "opus").tier for _ in range(5)] == ["opus"] * 5
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached):
            lim.admit(VISITOR, wanted)
    assert database.counts(VISITOR) == {"other": 25, "opus": 5}


@pytest.mark.integration
def test_postgres_allowances_of_zero(database: Database) -> None:
    no_opus = database.limits(Allowance(opus_per_hour=0, sonnet_per_hour=2))
    assert no_opus.peek(VISITOR) == Admission(
        "sonnet", 0, 2, 40 * 60, "opus_allowance_used", 20, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
    assert [no_opus.admit(VISITOR, "opus").tier for _ in range(2)] == ["sonnet"] * 2
    with pytest.raises(VisitorLimitReached):
        no_opus.admit(VISITOR, "opus")
    assert database.counts(VISITOR) == {"sonnet": 2}
    nothing = database.limits(Allowance(opus_per_hour=0, sonnet_per_hour=0))
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached):
            nothing.admit(OTHER_VISITOR, wanted)
    assert nothing.peek(OTHER_VISITOR).tier == "blocked"
    assert database.counts(OTHER_VISITOR) == {}


@pytest.mark.integration
def test_postgres_pruning_deletes_only_rows_older_than_a_day(database: Database) -> None:
    database.limits(rng=0.99).admit(VISITOR, "opus")  # creates the table, prunes nothing
    for visitor, hours in (("old", 25), ("recent", 23)):
        database.executor.execute(
            f"INSERT INTO {database.schema}.{TABLE} VALUES ($1::text, $2::timestamptz, 'opus', 3)",
            (visitor, (HOUR - timedelta(hours=hours)).isoformat()),
        )
    database.limits(rng=0.99).admit(VISITOR, "opus")
    assert database.counts("old") == {"opus": 3}
    database.limits(rng=0.0).admit(VISITOR, "opus")
    assert database.counts("old") == {}
    assert database.counts("recent") == {"opus": 3}
    assert database.counts(VISITOR) == {"opus": 3}


def _burst(
    database: Database, lim: VisitorLimits, size: int, hold_s: float, wanted: limits.Tier = "opus"
) -> dict[str, Any]:
    """`size` requests of one visitor that reach the visit statement at the same instant.

    A gate session holds the table locked for `hold_s` seconds while every request
    starts, so that all of them are released together: without the per-visitor lock they
    would all take the same snapshot and all be granted Opus. Returns the tiers granted,
    how many were refused for an allowance, and the SQLSTATE of every other failure."""
    lim.peek(VISITOR)  # the table exists and this instance knows it
    gate = database.executor.spawn()
    assert gate.stdin is not None and gate.stdout is not None
    gate.stdin.write(
        f"BEGIN;\nLOCK TABLE {database.schema}.{TABLE} IN ACCESS EXCLUSIVE MODE;\n"
        f"\\echo held\nSELECT pg_sleep({hold_s});\nCOMMIT;\n"
    )
    gate.stdin.close()
    assert gate.stdout.readline().strip() == "held"

    granted: list[str] = []
    refused: list[int] = []
    contended: list[str | None] = []
    errors: list[BaseException] = []

    def visit() -> None:
        try:
            granted.append(lim.admit(VISITOR, wanted).tier)
        except VisitorLimitReached as reached:
            refused.append(reached.resets_in_s)
        except LimitsUnavailable as unavailable:
            contended.append(unavailable.code)
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=visit) for _ in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert gate.wait(timeout=30) == 0
    gate.stdout.close()
    assert gate.stderr is not None
    gate.stderr.close()
    assert errors == []
    return {"granted": granted, "refused": len(refused), "contended": contended}


@pytest.mark.integration
def test_postgres_an_overlapping_burst_cannot_exceed_the_allowances(database: Database) -> None:
    """Forty at once, released after half a second: the visits run one after another.
    A visit that waited longer than the lock timeout for its turn fails with 55P03 and
    takes nothing; however many do, the rest are granted in order and counted once."""
    burst = 40
    seen = _burst(database, database.limits(), burst, 0.5)
    granted = seen["granted"]
    assert set(seen["contended"]) <= {limits.LOCK_NOT_AVAILABLE}
    assert len(granted) + seen["refused"] + len(seen["contended"]) == burst
    assert granted, "no visit got through the queue"
    opus, sonnet = granted.count("opus"), granted.count("sonnet")
    assert opus == min(10, len(granted)) and sonnet == len(granted) - opus <= 20
    assert seen["refused"] == 0 or (opus, sonnet) == (10, 20)
    # Every granted visit is in the table once, and nothing else is.
    expected = {tier: n for tier, n in (("opus", opus), ("sonnet", sonnet)) if n}
    assert database.counts(VISITOR) == expected


@pytest.mark.integration
@pytest.mark.parametrize("wanted", ["opus", "own"])
def test_postgres_only_one_visit_can_spend_the_last_daily_slot(
    database: Database, wanted: limits.Tier
) -> None:
    lim = database.limits(Allowance(opus_per_day=2, sonnet_per_day=0, own_per_day=2))
    lim.admit(VISITOR, wanted)
    database.clock.now += 3600
    seen = _burst(database, lim, 8, 0.3, wanted)
    assert seen["granted"] == [wanted]
    assert seen["refused"] + len(seen["contended"]) == 7
    assert set(seen["contended"]) <= {limits.LOCK_NOT_AVAILABLE}
    assert database.counts(VISITOR) == {wanted: 2}


@pytest.mark.integration
def test_postgres_a_held_visitor_lock_makes_admit_raise_55p03(database: Database) -> None:
    """Another session holds this visitor's advisory lock: ``admit`` gives up after the
    lock timeout with ``code == "55P03"``, another visitor is not held up, and the
    visitor is admitted again once the lock is gone."""
    lim = database.limits()
    lim.peek(VISITOR)
    holder = database.executor.spawn()
    assert holder.stdin is not None and holder.stdout is not None
    holder.stdin.write(
        f"BEGIN;\nSELECT pg_advisory_xact_lock(hashtextextended('{VISITOR}', 0));\n"
        "\\echo held\nSELECT pg_sleep(4);\nCOMMIT;\n"
    )
    holder.stdin.close()
    while holder.stdout.readline().strip() != "held":
        pass

    assert lim.admit(OTHER_VISITOR, "opus").tier == "opus"
    began = time.monotonic()
    with pytest.raises(LimitsUnavailable) as caught:
        lim.admit(VISITOR, "opus")
    waited = time.monotonic() - began
    assert caught.value.code == limits.LOCK_NOT_AVAILABLE == "55P03"
    assert "55P03" in str(caught.value)
    assert 1.0 < waited < 3.0  # the 1500 ms bound, not the holder's 4 s
    assert database.counts(VISITOR) == {}

    assert holder.wait(timeout=30) == 0
    holder.stdout.close()
    assert holder.stderr is not None
    holder.stderr.close()
    assert lim.admit(VISITOR, "opus") == Admission(
        "opus", 9, 20, 40 * 60, None, 19, 40, 30, 60, TO_MIDNIGHT, "hour"
    )
