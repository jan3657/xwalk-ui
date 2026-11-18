"""Per-visitor hourly and daily allowances for the public demo.

The site tiers share a total in each UTC window. Opus can step down to Sonnet,
and other site models spend only the shared total. Own-key visits have independent
hourly and daily flood limits. See the own-key daily-cap design, section 3.

Who a visitor is. ``visitor_id`` is a keyed hash of the client address, so the table, the
SQL parameters and the logs never hold a raw IP, and the ids cannot be reversed without
the salt. Requests with no usable address header share the id of the empty string, and
with it one allowance.

Where the counts live. One table in the Neon schema the indexes already use, reached
through the same ``SQLExecutor``: ``site_visits(visitor, hour, tier, count)`` with one row
per visitor, UTC clock hour and tier. A serverless instance has no memory worth trusting,
and every ``execute`` is an HTTPS round trip, so each call here is exactly one round trip:

- ``admit`` sends one transaction of three statements: a bound on how long it may wait
  for a lock, a lock on the visitor, then one statement that reads the visitor's counts
  for both windows, decides the tier and records the visit (an upsert that increments
  ``count``). Deciding in SQL is what keeps that to one round trip.
- ``peek`` sends one read and decides in Python. It records nothing and takes no lock.
- The first call of an instance puts ``CREATE TABLE IF NOT EXISTS`` in front, and about
  one ``admit`` in fifty puts a ``DELETE`` of rows older than a day behind, in the same
  transaction: still one round trip.

Under concurrent requests. A statement decides from the snapshot it starts with, so
without more, a burst of parallel requests from one visitor would all read the same
counts and all be granted the same tier. The lock closes that: every ``admit`` first
takes ``pg_advisory_xact_lock`` on a hash of the visitor id, held to the end of the
transaction, so the visits of one visitor run one after another however they arrive. The
lock has to be a statement of its own. The transaction runs at READ COMMITTED (the
default of Postgres and of Neon's endpoint; the executor asks for no other level), where
each statement takes a fresh snapshot, so the visit statement, which starts only once the
lock is held, sees every earlier visit committed. Each granted transaction increments
one bucket. Counting the request's whole UTC day includes visits that committed first
even when their sampled hour is later. Different visitors lock independently unless
their 64-bit hashes collide.

The wait for that lock is bounded. One visitor's parallel requests queue on it, and a
request whose turn does not come within ``LOCK_TIMEOUT_MS`` gives up: Postgres cancels
its statement with SQLSTATE 55P03 (``LOCK_NOT_AVAILABLE``), nothing is recorded, and
``admit`` raises ``LimitsUnavailable`` with that ``code``. The bound is well under the
caller's timeout for the whole round trip, so a long queue ends as that distinct answer,
which the caller refuses with a 429, and never as a call that timed out and was let
through uncounted.

One thing the statements cannot make safe is a retry. ``NeonHTTPExecutor`` sends a
request again after a transport error or a 5xx, and the increment is not idempotent, so a
visit whose first attempt was applied but whose answer was lost would be counted twice:
the caller should give the limiter an executor that does not retry.

When the database cannot be asked, every failure becomes ``LimitsUnavailable`` with a
message made here from the error's class and SQLSTATE only, and the SQLSTATE again as
its ``code``. The executor's own text is dropped (it may name the host), and the caller
decides what an unmetered request may do.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, get_args

from xwalk.retrieval.postgres import SQLExecutor, Statement

__all__ = [
    "ADDRESS_HEADERS",
    "LOCK_NOT_AVAILABLE",
    "LOCK_TIMEOUT_MS",
    "PRUNE_AFTER_S",
    "PRUNE_CHANCE",
    "TABLE",
    "Admission",
    "Allowance",
    "LimitsUnavailable",
    "Tier",
    "VisitorLimitReached",
    "VisitorLimits",
    "default_salt",
    "visitor_id",
]

Tier = Literal["opus", "sonnet", "other", "own"]
_Window = Literal["hour", "day"]
DAY_S = 86400
TABLE = "site_visits"
PRUNE_AFTER_S = 86400
PRUNE_CHANCE = 0.02
HOUR_S = 3600
# Where the client address is looked for, in this order (header names in lower case).
ADDRESS_HEADERS = ("x-real-ip", "x-vercel-forwarded-for", "x-forwarded-for")
OPUS_USED = "opus_allowance_used"
# How long one visit waits for its visitor's lock (and for any other lock) before it
# gives up, and the SQLSTATE Postgres then answers with (lock_not_available).
LOCK_TIMEOUT_MS = 1500
LOCK_NOT_AVAILABLE = "55P03"

_TIERS: tuple[str, ...] = get_args(Tier)
_IDENT = re.compile(r"[a-z_][a-z0-9_]{0,62}")
_SQLSTATE = re.compile(r"[0-9A-Z]{5}")

# Every statement starts with a tag, like the engine's own, so a log or a fake executor
# can tell them apart. `{table}` is the validated schema and the fixed table name; every
# value is a `$n` parameter, cast where Postgres could not infer its type.
_CREATE = """/* xwalk:site_visits_table */
CREATE TABLE IF NOT EXISTS {table} (
    visitor text NOT NULL,
    hour timestamptz NOT NULL,
    tier text NOT NULL,
    count integer NOT NULL,
    PRIMARY KEY (visitor, hour, tier)
)"""

# No parameter: SET takes none, so the bound is written into the statement. LOCAL keeps
# it to this transaction. It goes ahead of `_LOCK`, whose wait it is there to bound.
_LOCK_TIMEOUT = f"""/* xwalk:site_visit_lock_timeout */
SET LOCAL lock_timeout = '{LOCK_TIMEOUT_MS}ms'"""

# $1 visitor. Serialises the visits of one visitor to the end of the transaction. It must
# stay a statement of its own, ahead of `_VISIT`: a lock taken inside the visit statement
# would come after that statement's snapshot, and the counts read would be stale.
_LOCK = """/* xwalk:site_visit_lock */
SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))"""

# $1 visitor, $2 UTC hour, $3 UTC midnight. Both paths read exactly the same buckets.
# A UTC day is 24 hours even when the database session's local day has a DST change.
_COUNTS = """SELECT
    coalesce(sum(count) FILTER (WHERE hour = $2 AND tier = 'opus'), 0) AS opus,
    coalesce(sum(count) FILTER (WHERE hour = $2 AND tier = 'sonnet'), 0) AS sonnet,
    coalesce(sum(count) FILTER (WHERE hour = $2 AND tier <> 'own'), 0) AS total,
    coalesce(sum(count) FILTER (WHERE hour = $2 AND tier = 'own'), 0) AS own,
    coalesce(sum(count) FILTER (WHERE tier = 'opus'), 0) AS opus_today,
    coalesce(sum(count) FILTER (WHERE tier = 'sonnet'), 0) AS sonnet_today,
    coalesce(sum(count) FILTER (WHERE tier <> 'own'), 0) AS total_today,
    coalesce(sum(count) FILTER (WHERE tier = 'own'), 0) AS own_today
FROM {table}
WHERE visitor = $1::text AND hour >= $3::timestamptz
  AND hour < $3::timestamptz + interval '24 hours'"""

_READ = "/* xwalk:site_visits_read */\n" + _COUNTS

# $4 wanted tier, $5/$6 hourly site caps, $7/$8 daily site caps, $9/$10 own caps.
# Return pre-increment counts even on refusal, so a retry can use the right reset.
_VISIT = f"""/* xwalk:site_visit */
WITH seen AS (
    {_COUNTS}
), pick AS (
    SELECT seen.*,
           CASE WHEN $4::text = 'own' THEN
                    CASE WHEN own < $9::int AND own_today < $10::int THEN 'own' END
                WHEN total >= $5::int + $6::int OR total_today >= $7::int + $8::int THEN NULL
                WHEN $4 = 'other' THEN 'other'
                WHEN $4 = 'opus' AND opus < $5 AND opus_today < $7 THEN 'opus'
                WHEN sonnet < $6 AND sonnet_today < $8 THEN 'sonnet'
           END AS tier
    FROM seen
), put AS (
    INSERT INTO {{table}} AS v (visitor, hour, tier, count)
    SELECT $1, $2, tier, 1 FROM pick WHERE tier IS NOT NULL
    ON CONFLICT (visitor, hour, tier) DO UPDATE SET count = v.count + 1
)
SELECT * FROM pick"""

# $1 the cutoff: now minus PRUNE_AFTER_S.
_PRUNE = """/* xwalk:site_visits_prune */
DELETE FROM {table} WHERE hour < $1::timestamptz"""


@dataclass(frozen=True)
class Allowance:
    """Per-visitor caps. Each window's two Claude caps also set its shared site total."""

    opus_per_hour: int = 10
    sonnet_per_hour: int = 20
    opus_per_day: int = 20
    sonnet_per_day: int = 40
    own_per_hour: int = 30
    own_per_day: int = 60


@dataclass(frozen=True)
class Admission:
    """Where the visit starts (or the next Opus request in a peek), and remaining starts.
    Only ``peek`` can return ``"blocked"``. Balances from ``admit`` are after the visit.
    Hourly balances include the daily constraint. Even when ``window == "day"``,
    ``resets_in_s`` gives the hourly reset and ``day_resets_in_s`` the reset at UTC midnight.
    A step-down reports the window that limited Opus. Other grants report the window
    that would next limit starts on the requested tier.
    """

    tier: Tier | Literal["blocked"]
    opus_left: int
    sonnet_left: int
    resets_in_s: int
    reason: str | None
    opus_left_today: int
    sonnet_left_today: int
    own_left: int
    own_left_today: int
    day_resets_in_s: int
    window: _Window


class LimitsUnavailable(Exception):
    """The database could not be queried. The message names the error's class and its
    SQLSTATE, nothing the executor wrote. ``code`` is that SQLSTATE, None when the
    failure had none; ``LOCK_NOT_AVAILABLE`` means the visitor's lock was contended."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class VisitorLimitReached(Exception):
    """The request was refused and nothing was recorded. The binding window sets the retry time."""

    def __init__(self, resets_in_s: int, window: _Window = "hour") -> None:
        super().__init__(f"visitor limit reached; resets in {resets_in_s} s")
        self.resets_in_s = resets_in_s
        self.window = window


def visitor_id(headers: Mapping[str, str], salt: str) -> str:
    """The visitor behind a request: the first 32 hex characters of
    ``HMAC-SHA256(salt, ip)``. ``headers`` has lower-cased names. ``ip`` is the first
    non-empty of ``ADDRESS_HEADERS`` (for ``x-forwarded-for`` its first entry), stripped,
    or the empty string when there is none."""
    ip = ""
    for name in ADDRESS_HEADERS:
        value = headers.get(name) or ""
        if name == "x-forwarded-for":
            value = value.split(",", 1)[0]
        ip = value.strip()
        if ip:
            break
    return hmac.new(salt.encode("utf-8"), ip.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def default_salt(database_url: str) -> str:
    """The salt when ``NER_SITE_VISITOR_SALT`` is unset: the SHA-256 of the database URL,
    a secret the deployment already has and that is the same on every instance."""
    return hashlib.sha256(database_url.encode("utf-8")).hexdigest()


def _room(
    row: Mapping[str, Any], suffix: str, opus: int, sonnet: int, own: int, spent: Tier | None
) -> dict[Tier, int]:
    """Starts still possible in one window, including a grant just recorded by the SQL."""
    try:
        site_visit = spent is not None and spent != "own"
        total = opus + sonnet - int(row["total" + suffix]) - site_visit
        return {
            "opus": max(0, min(total, opus - int(row["opus" + suffix]) - (spent == "opus"))),
            "sonnet": max(
                0, min(total, sonnet - int(row["sonnet" + suffix]) - (spent == "sonnet"))
            ),
            "other": max(0, total),
            "own": max(0, own - int(row["own" + suffix]) - (spent == "own")),
        }
    except (KeyError, TypeError, ValueError):
        raise LimitsUnavailable("visitor limits answered with an unreadable row") from None


def _refusal_window(wanted: Tier, day: dict[Tier, int], caps: Allowance) -> _Window:
    """An Opus retry only needs one eligible starting tier after the hour rolls over."""
    hourly = {
        "opus": caps.opus_per_hour,
        "sonnet": caps.sonnet_per_hour,
        "other": caps.opus_per_hour + caps.sonnet_per_hour,
        "own": caps.own_per_hour,
    }
    candidates: tuple[Tier, ...] = ("opus", "sonnet") if wanted == "opus" else (wanted,)
    enabled = [tier for tier in candidates if hourly[tier] > 0]
    # A disabling zero cap gives its window's next boundary because no finite retry can admit.
    if not enabled or any(day[tier] for tier in enabled):
        return "hour"
    return "day"


def _stamp(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class VisitorLimits:
    """The allowance of every visitor, kept in ``<schema>.site_visits``.

    One instance per warm serverless instance: it remembers that it has created the
    table. ``clock`` gives the time in seconds since the epoch and ``rng`` a number in
    [0, 1); both are parameters so that tests can set the hour and the pruning draw.
    """

    def __init__(
        self,
        executor: SQLExecutor,
        schema: str,
        allowance: Allowance,
        *,
        clock: Callable[[], float] = time.time,
        rng: Callable[[], float] = random.random,
    ) -> None:
        if not _IDENT.fullmatch(schema):
            raise ValueError(f"schema {schema!r} must match {_IDENT.pattern}")
        self._executor = executor
        self._allowance = allowance
        self._clock = clock
        self._rng = rng
        table = f"{schema}.{TABLE}"
        self._create = _CREATE.format(table=table)
        self._visit = _VISIT.format(table=table)
        self._read = _READ.format(table=table)
        self._prune = _PRUNE.format(table=table)
        self._created = False

    def admit(self, visitor: str, wanted: Tier) -> Admission:
        """Count one visit and say where it starts. Raises ``VisitorLimitReached`` (and
        records nothing) when an allowance is used, ``LimitsUnavailable`` when the database
        cannot be asked; with ``code == LOCK_NOT_AVAILABLE`` when the visitor's other
        requests held the lock for longer than ``LOCK_TIMEOUT_MS`` (nothing recorded)."""
        if wanted not in _TIERS:
            raise ValueError(f"wanted must be one of {_TIERS}, got {wanted!r}")
        caps = self._allowance
        now = self._clock()
        hour, midnight, hour_reset, day_reset = self._windows(now)
        statements: list[Statement] = [
            (_LOCK_TIMEOUT, ()),
            (_LOCK, (visitor,)),
            (
                self._visit,
                (
                    visitor,
                    hour,
                    midnight,
                    wanted,
                    caps.opus_per_hour,
                    caps.sonnet_per_hour,
                    caps.opus_per_day,
                    caps.sonnet_per_day,
                    caps.own_per_hour,
                    caps.own_per_day,
                ),
            ),
        ]
        if self._rng() < PRUNE_CHANCE:
            statements.append((self._prune, (_stamp(int(now) - PRUNE_AFTER_S),)))
        row = self._send(statements, answer=2)
        if "tier" not in row or (row["tier"] is not None and row["tier"] not in _TIERS):
            raise LimitsUnavailable("visitor limits answered with an unreadable row")
        return self._admission(row, wanted, hour_reset, day_reset, peek=False)

    def peek(self, visitor: str) -> Admission:
        """Predict the next Opus start and report both balances without taking a lock."""
        hour, midnight, hour_reset, day_reset = self._windows(self._clock())
        row = self._send([(self._read, (visitor, hour, midnight))], answer=0)
        return self._admission(row, "opus", hour_reset, day_reset, peek=True)

    def _admission(
        self, row: Mapping[str, Any], wanted: Tier, hour_reset: int, day_reset: int, *, peek: bool
    ) -> Admission:
        caps = self._allowance
        tier = None if peek else row["tier"]
        hour = _room(row, "", caps.opus_per_hour, caps.sonnet_per_hour, caps.own_per_hour, tier)
        day = _room(row, "_today", caps.opus_per_day, caps.sonnet_per_day, caps.own_per_day, tier)
        if peek:
            if hour["opus"] and day["opus"]:
                tier = "opus"
            elif hour["sonnet"] and day["sonnet"]:
                tier = "sonnet"
        reason = OPUS_USED if wanted == "opus" and tier == "sonnet" else None
        if tier is None:
            window = _refusal_window(wanted, day, caps)
            if not peek:
                raise VisitorLimitReached(day_reset if window == "day" else hour_reset, window)
        elif reason:
            # A step-down spends a site slot in both windows. Attribute it to the
            # Opus cap before that spend, when the SQL actually made its decision.
            window = "day" if int(row["opus_today"]) >= caps.opus_per_day else "hour"
        else:
            window = "day" if day[wanted] == 0 or day[wanted] < hour[wanted] else "hour"
        return Admission(
            tier=tier or "blocked",
            opus_left=min(hour["opus"], day["opus"]),
            sonnet_left=min(hour["sonnet"], day["sonnet"]),
            resets_in_s=hour_reset,
            reason=reason,
            opus_left_today=day["opus"],
            sonnet_left_today=day["sonnet"],
            own_left=min(hour["own"], day["own"]),
            own_left_today=day["own"],
            day_resets_in_s=day_reset,
            window=window,
        )

    @staticmethod
    def _windows(now: float) -> tuple[str, str, int, int]:
        """Take both UTC boundaries and ceiling-rounded retry intervals from one clock sample."""
        hour = int(now // HOUR_S) * HOUR_S
        day = int(now // DAY_S) * DAY_S
        return (
            _stamp(hour),
            _stamp(day),
            math.ceil(hour + HOUR_S - now),
            math.ceil(day + DAY_S - now),
        )

    def _send(self, statements: list[Statement], *, answer: int) -> Mapping[str, Any]:
        """One round trip: the statements, behind the table's creation when this instance
        has not sent it yet. Returns the single row of ``statements[answer]``."""
        lead = 0 if self._created else 1
        if lead:
            statements = [(self._create, ()), *statements]
        try:
            if len(statements) == 1:
                rows = self._executor.execute(*statements[0])
            else:
                rows = self._executor.transaction(statements)[lead + answer]
            row = rows[0]
        except Exception as exc:
            code = getattr(exc, "code", None)
            state = code if isinstance(code, str) and _SQLSTATE.fullmatch(code) else None
            raise LimitsUnavailable(
                "visitor limits could not be queried "
                f"({type(exc).__name__}{'' if state is None else ' ' + state})",
                code=state,
            ) from None
        self._created = True
        return row
