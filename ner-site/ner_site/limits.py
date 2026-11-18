"""The visitor allowance: how many requests an hour one visitor may start on Claude.

Binding: docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md section
3.3. The site is open and Claude runs on the operator's prepaid Anthropic balance, so the
only thing between one visitor and that balance is this module. Each visitor gets
``opus_per_hour`` requests starting on Opus in a clock hour, then ``sonnet_per_hour``
starting on Sonnet, then is refused until the hour rolls over. A request for another
model (``other``) spends none of the Claude allowances but counts toward the hour's total.

Who a visitor is. ``visitor_id`` is a keyed hash of the client address, so the table, the
SQL parameters and the logs never hold a raw IP, and the ids cannot be reversed without
the salt. Requests with no usable address header share the id of the empty string, and
with it one allowance.

Where the counts live. One table in the Neon schema the indexes already use, reached
through the same ``SQLExecutor``: ``site_visits(visitor, hour, tier, count)`` with one row
per visitor, UTC clock hour and tier. A serverless instance has no memory worth trusting,
and every ``execute`` is an HTTPS round trip, so each call here is exactly one round trip:

- ``admit`` sends one transaction of two statements: a lock on the visitor, then one
  statement that reads the visitor's counts for the hour, decides the tier and records
  the visit (an upsert that increments ``count``). Deciding in SQL is what keeps that to
  one round trip; ``_grant`` is the same rule in Python, used by ``peek``.
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
lock is held, sees every earlier visit committed. The allowances and the hour's total are
therefore exact under any interleaving: no visit is lost, none is counted twice, and none
is granted beyond an allowance. Different visitors take different locks and do not wait
for each other (two ids whose 64-bit hashes collide would merely take turns).

One thing the statements cannot make safe is a retry. ``NeonHTTPExecutor`` sends a
request again after a transport error or a 5xx, and the increment is not idempotent, so a
visit whose first attempt was applied but whose answer was lost would be counted twice:
the caller should give the limiter an executor that does not retry.

When the database cannot be asked, every failure becomes ``LimitsUnavailable`` with a
message made here from the error's class and SQLSTATE only. The executor's own text is
dropped (it may name the host), and the caller decides what an unmetered request may do.
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

Tier = Literal["opus", "sonnet", "other"]
TABLE = "site_visits"
PRUNE_AFTER_S = 86400
PRUNE_CHANCE = 0.02
HOUR_S = 3600
# Where the client address is looked for, in this order (header names in lower case).
ADDRESS_HEADERS = ("x-real-ip", "x-vercel-forwarded-for", "x-forwarded-for")
OPUS_USED = "opus_allowance_used"

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

# $1 visitor. Serialises the visits of one visitor to the end of the transaction. It must
# stay a statement of its own, ahead of `_VISIT`: a lock taken inside the visit statement
# would come after that statement's snapshot, and the counts read would be stale.
_LOCK = """/* xwalk:site_visit_lock */
SELECT pg_advisory_xact_lock(hashtextextended($1::text, 0))"""

# $1 visitor, $2 hour, $3 wanted tier, $4 Opus allowance, $5 Sonnet allowance. One row
# comes back: the counts as they were before this visit and the tier granted, NULL when
# the visit is refused (then `put` inserts nothing). The CASE is `_grant` in SQL.
_VISIT = """/* xwalk:site_visit */
WITH seen AS (
    SELECT coalesce(sum(s.count) FILTER (WHERE s.tier = 'opus'), 0)::int AS opus,
           coalesce(sum(s.count) FILTER (WHERE s.tier = 'sonnet'), 0)::int AS sonnet,
           coalesce(sum(s.count), 0)::int AS total
    FROM {table} AS s
    WHERE s.visitor = $1::text AND s.hour = $2::timestamptz
), pick AS (
    SELECT seen.opus, seen.sonnet, seen.total,
           CASE WHEN seen.total >= $4::int + $5::int THEN NULL
                WHEN $3::text = 'other' THEN 'other'
                WHEN $3::text = 'opus' AND seen.opus < $4::int THEN 'opus'
                WHEN seen.sonnet < $5::int THEN 'sonnet'
                ELSE NULL
           END AS tier
    FROM seen
), put AS (
    INSERT INTO {table} AS v (visitor, hour, tier, count)
    SELECT $1::text, $2::timestamptz, pick.tier, 1 FROM pick WHERE pick.tier IS NOT NULL
    ON CONFLICT (visitor, hour, tier) DO UPDATE SET count = v.count + 1
)
SELECT pick.opus, pick.sonnet, pick.total, pick.tier FROM pick"""

# $1 visitor, $2 hour.
_READ = """/* xwalk:site_visits_read */
SELECT coalesce(sum(s.count) FILTER (WHERE s.tier = 'opus'), 0)::int AS opus,
       coalesce(sum(s.count) FILTER (WHERE s.tier = 'sonnet'), 0)::int AS sonnet,
       coalesce(sum(s.count), 0)::int AS total
FROM {table} AS s
WHERE s.visitor = $1::text AND s.hour = $2::timestamptz"""

# $1 the cutoff: now minus PRUNE_AFTER_S.
_PRUNE = """/* xwalk:site_visits_prune */
DELETE FROM {table} WHERE hour < $1::timestamptz"""


@dataclass(frozen=True)
class Allowance:
    """Requests one visitor may start on each Claude tier in a clock hour. Their sum is
    the hour's total for every kind of request."""

    opus_per_hour: int = 10
    sonnet_per_hour: int = 20


@dataclass(frozen=True)
class Admission:
    """Where a request starts and what is left of the hour.

    ``opus_left`` and ``sonnet_left`` are what a visitor can still start on each tier this
    hour: the tier's own allowance, capped by what remains of the hour's total (requests
    for other models use up the total too). From ``admit`` they are after this visit.
    ``tier`` is ``"blocked"`` only from ``peek``. ``reason`` is ``"opus_allowance_used"``
    when Opus was wanted and Sonnet granted, else None.
    """

    tier: Literal["opus", "sonnet", "other", "blocked"]
    opus_left: int
    sonnet_left: int
    resets_in_s: int
    reason: str | None


class LimitsUnavailable(Exception):
    """The database could not be queried. The message names the error's class and its
    SQLSTATE, nothing the executor wrote."""


class VisitorLimitReached(Exception):
    """The visitor has used the hour; nothing was recorded. ``resets_in_s`` is the time
    to the next clock hour."""

    def __init__(self, resets_in_s: int) -> None:
        super().__init__(f"visitor limit reached; resets in {resets_in_s} s")
        self.resets_in_s = resets_in_s


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


def _grant(wanted: Tier, opus: int, sonnet: int, total: int, allowance: Allowance) -> Tier | None:
    """The tier a request for ``wanted`` starts on given the hour's counts, or None when
    it is refused. The CASE in ``_VISIT`` is this rule in SQL; keep the two alike."""
    if total >= allowance.opus_per_hour + allowance.sonnet_per_hour:
        return None
    if wanted == "other":
        return "other"
    if wanted == "opus" and opus < allowance.opus_per_hour:
        return "opus"
    if sonnet < allowance.sonnet_per_hour:
        return "sonnet"
    return None


def _left(opus: int, sonnet: int, total: int, allowance: Allowance) -> tuple[int, int]:
    """(opus_left, sonnet_left) for the hour's counts: each tier's own allowance, capped
    by what remains of the hour's total, never negative."""
    room = allowance.opus_per_hour + allowance.sonnet_per_hour - total
    return (
        max(0, min(allowance.opus_per_hour - opus, room)),
        max(0, min(allowance.sonnet_per_hour - sonnet, room)),
    )


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
        records nothing) when the hour is used, ``LimitsUnavailable`` when the database
        cannot be asked."""
        if wanted not in _TIERS:
            raise ValueError(f"wanted must be one of {_TIERS}, got {wanted!r}")
        allowance = self._allowance
        now = self._clock()
        hour, resets_in_s = self._hour(now)
        statements: list[Statement] = [
            (_LOCK, (visitor,)),
            (
                self._visit,
                (visitor, hour, wanted, allowance.opus_per_hour, allowance.sonnet_per_hour),
            ),
        ]
        if self._rng() < PRUNE_CHANCE:
            statements.append((self._prune, (_stamp(int(now) - PRUNE_AFTER_S),)))
        row = self._send(statements, answer=1)
        try:
            opus, sonnet, total = int(row["opus"]), int(row["sonnet"]), int(row["total"])
            tier = row["tier"]
            if tier is not None and tier not in _TIERS:
                raise ValueError("unknown tier")
        except (KeyError, TypeError, ValueError):
            raise LimitsUnavailable("visitor limits answered with an unreadable row") from None
        if tier is None:
            raise VisitorLimitReached(resets_in_s)
        opus_left, sonnet_left = _left(
            opus + (tier == "opus"), sonnet + (tier == "sonnet"), total + 1, allowance
        )
        return Admission(
            tier=tier,
            opus_left=opus_left,
            sonnet_left=sonnet_left,
            resets_in_s=resets_in_s,
            reason=OPUS_USED if wanted == "opus" and tier == "sonnet" else None,
        )

    def peek(self, visitor: str) -> Admission:
        """What a request for Opus would get now, and what is left of the hour before it.
        Records nothing; ``tier`` is ``"blocked"`` where ``admit`` would raise
        ``VisitorLimitReached``."""
        allowance = self._allowance
        hour, resets_in_s = self._hour(self._clock())
        row = self._send([(self._read, (visitor, hour))], answer=0)
        try:
            opus, sonnet, total = int(row["opus"]), int(row["sonnet"]), int(row["total"])
        except (KeyError, TypeError, ValueError):
            raise LimitsUnavailable("visitor limits answered with an unreadable row") from None
        tier = _grant("opus", opus, sonnet, total, allowance)
        opus_left, sonnet_left = _left(opus, sonnet, total, allowance)
        return Admission(
            tier=tier or "blocked",
            opus_left=opus_left,
            sonnet_left=sonnet_left,
            resets_in_s=resets_in_s,
            reason=OPUS_USED if tier == "sonnet" else None,
        )

    @staticmethod
    def _hour(now: float) -> tuple[str, int]:
        """The UTC clock hour of ``now`` as a timestamptz literal, and the whole seconds
        to the next one (at least 1)."""
        start = int(now // HOUR_S) * HOUR_S
        return _stamp(start), max(1, math.ceil(start + HOUR_S - now))

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
            state = f" {code}" if isinstance(code, str) and _SQLSTATE.fullmatch(code) else ""
            raise LimitsUnavailable(
                f"visitor limits could not be queried ({type(exc).__name__}{state})"
            ) from None
        self._created = True
        return row
