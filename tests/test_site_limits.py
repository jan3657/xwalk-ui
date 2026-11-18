"""The visitor allowance (spec 2026-11-18, section 3.3): offline, no database.

``FakeVisits`` is an in-memory stand-in for the ``site_visits`` table. It was chosen over
``sqlite3`` because the limiter's one statement is Postgres only (a data-modifying CTE,
``FILTER``, ``$n`` placeholders, ``timestamptz``). It tells the limiter's four statements
apart by their ``/* xwalk:<tag> */`` and answers each with the meaning of its SQL: it
keeps the counts, makes the same grant decision the ``site_visit`` statement makes from
the parameters it was sent, and refuses a statement that arrives before the table exists.
It also records every statement and counts round trips, so the tests can check what was
sent as well as what came back.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

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

    def _run(self, sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        p = list(params)
        self.sent.append((sql, p))
        if self.fail is not None:
            raise self.fail
        placeholders = {int(n) for n in re.findall(r"\$(\d+)", sql)}
        assert placeholders == set(range(1, len(p) + 1)), "placeholders and parameters differ"
        tag = tag_of(sql)
        if tag != "site_visits_table" and not self.created:
            raise PostgresError('relation "site_visits" does not exist', code="42P01")
        return getattr(self, f"_{tag}")(sql, p)

    # --- the statements
    def _site_visits_table(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        assert "CREATE TABLE IF NOT EXISTS" in sql
        self.created = True
        return []

    def _counts(self, visitor: str, hour: datetime) -> dict[str, int]:
        found = {"opus": 0, "sonnet": 0, "total": 0}
        for (who, when, tier), count in self.rows.items():
            if who == visitor and when == hour:
                found["total"] += count
                if tier in found:
                    found[tier] += count
        return found

    def _site_visit(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        visitor, hour_text, wanted, opus_cap, sonnet_cap = p
        hour = _when(hour_text)
        seen = self._counts(visitor, hour)
        if seen["total"] >= opus_cap + sonnet_cap:
            tier = None
        elif wanted == "other":
            tier = "other"
        elif wanted == "opus" and seen["opus"] < opus_cap:
            tier = "opus"
        elif seen["sonnet"] < sonnet_cap:
            tier = "sonnet"
        else:
            tier = None
        if tier is not None:
            key = (visitor, hour, tier)
            self.rows[key] = self.rows.get(key, 0) + 1
        return [{**seen, "tier": tier}]  # the counts as they were before this visit

    def _site_visits_read(self, sql: str, p: list[Any]) -> list[dict[str, Any]]:
        visitor, hour_text = p
        return [self._counts(visitor, _when(hour_text))]

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
    assert got[10] == Admission("sonnet", 0, 19, 40 * 60, "opus_allowance_used")
    assert [a.tier for a in got[10:]] == ["sonnet"] * 20
    assert all(a.reason == "opus_allowance_used" for a in got[10:])
    assert got[29].sonnet_left == 0
    before = dict(fake.rows)
    with pytest.raises(VisitorLimitReached) as caught:
        lim.admit(VISITOR, "opus")
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
    got = [lim.admit(VISITOR, "other") for _ in range(25)]
    assert all(a == Admission("other", 10, 20, 40 * 60, None) for a in got)
    assert [lim.admit(VISITOR, "opus").opus_left for _ in range(5)] == [9, 8, 7, 6, 5]
    assert fake.total() == 30
    for wanted in ("opus", "sonnet", "other"):
        with pytest.raises(VisitorLimitReached):
            lim.admit(VISITOR, wanted)
    assert fake.rows == {(VISITOR, HOUR, "other"): 25, (VISITOR, HOUR, "opus"): 5}


def test_a_second_visitor_is_unaffected() -> None:
    lim, fake, _ = make()
    for _ in range(30):
        lim.admit(VISITOR, "opus")
    assert lim.admit(OTHER_VISITOR, "opus") == Admission("opus", 9, 20, 40 * 60, None)
    assert fake.total(OTHER_VISITOR) == 1


def test_the_next_clock_hour_starts_fresh() -> None:
    lim, fake, clock = make()
    for _ in range(30):
        lim.admit(VISITOR, "opus")
    clock.now = HOUR.timestamp() + 3600
    assert lim.admit(VISITOR, "opus") == Admission("opus", 9, 20, 3600, None)
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
    assert lim.peek(VISITOR) == Admission("opus", 10, 20, 40 * 60, None)
    assert fake.rows == {}
    for count in range(1, 31):
        admitted = lim.admit(VISITOR, "opus")
        before = dict(fake.rows)
        peeked = lim.peek(VISITOR)
        assert fake.rows == before
        assert (peeked.opus_left, peeked.sonnet_left, peeked.resets_in_s) == (
            admitted.opus_left,
            admitted.sonnet_left,
            admitted.resets_in_s,
        )
        if count < 10:
            assert (peeked.tier, peeked.reason) == ("opus", None)
        elif count < 30:
            assert (peeked.tier, peeked.reason) == ("sonnet", "opus_allowance_used")
    assert lim.peek(VISITOR) == Admission("blocked", 0, 0, 40 * 60, None)
    assert fake.total() == 30


def test_peek_is_blocked_at_the_total_even_with_opus_unused() -> None:
    lim, _, _ = make()
    for _ in range(30):
        lim.admit(VISITOR, "other")
    assert lim.peek(VISITOR) == Admission("blocked", 10, 20, 40 * 60, None)


# --- allowances of zero ------------------------------------------------------------


def test_opus_allowance_of_zero_starts_on_sonnet() -> None:
    lim, fake, _ = make(Allowance(opus_per_hour=0, sonnet_per_hour=2))
    assert lim.peek(VISITOR) == Admission("sonnet", 0, 2, 40 * 60, "opus_allowance_used")
    assert lim.admit(VISITOR, "opus") == Admission("sonnet", 0, 1, 40 * 60, "opus_allowance_used")
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


# --- round trips, the table, pruning -----------------------------------------------


def test_one_round_trip_per_call_and_the_table_is_created_once() -> None:
    lim, fake, _ = make()
    lim.admit(VISITOR, "opus")
    assert fake.round_trips == 1
    assert fake.tags() == ["site_visits_table", "site_visit"]
    lim.admit(VISITOR, "opus")
    lim.peek(VISITOR)
    assert fake.round_trips == 3
    assert fake.tags()[2:] == ["site_visit", "site_visits_read"]
    assert f"xwalk.{TABLE}" in fake.sent[0][0]
    assert all(f"xwalk.{TABLE}" in sql for sql, _ in fake.sent)


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
    lim, fake, _ = make(Allowance(opus_per_hour=7, sonnet_per_hour=13), rng=0.0)
    lim.admit(VISITOR, "opus")
    lim.peek(VISITOR)
    by_tag = {tag_of(sql): (sql, params) for sql, params in fake.sent}
    assert by_tag["site_visit"][1] == [VISITOR, "2026-11-18T14:00:00+00:00", "opus", 7, 13]
    assert by_tag["site_visits_read"][1] == [VISITOR, "2026-11-18T14:00:00+00:00"]
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
    VisitorLimits(fake, "ner_site_2", Allowance(), clock=Clock(), rng=lambda: 0.5).peek(VISITOR)
    assert all(f"ner_site_2.{TABLE}" in sql for sql, _ in fake.sent)


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
    for wanted in ("opus", "sonnet", "other"):
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
