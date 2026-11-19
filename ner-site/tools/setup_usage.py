"""Create the table the NER demo records its runs in, once, before recording is on.

    DATABASE_URL=postgresql://... python ner-site/tools/setup_usage.py
    DATABASE_URL=postgresql://... python ner-site/tools/setup_usage.py --schema xwalk

For the operator, before setting `NER_SITE_STATS_PASSCODE` (ner-site/README.md,
"Analytics"). The site creates `<schema>.site_runs` by itself on the first run of each
warm instance, but two cold instances that do so at the same moment on a database
without the table can collide, and the one that loses also loses its row. With the
table already there nothing is left to create and nothing to collide over.

What it does, and nothing else: one request to the database named by `DATABASE_URL` (a
Neon connection string; the request goes over HTTPS), holding three statements in one
transaction: `CREATE SCHEMA IF NOT EXISTS`, `CREATE TABLE IF NOT EXISTS` and
`CREATE INDEX IF NOT EXISTS` (`ner_site.usage.SiteRuns.create`). It writes no row, drops
nothing, and running it again changes nothing. The schema is `--schema`, else
`NER_SITE_NEON_SCHEMA`, else the site's default; give it the value the deployment has.

The connection string is read from the environment only and is never printed: a
failure is reported by the class of the error and its SQLSTATE. Exit code 0 when the
table is in place, 1 when the database could not be asked, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]
if str(SITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SITE_ROOT))

from ner_site.indexes import NEON_SCHEMA  # noqa: E402
from ner_site.usage import TABLE, SiteRuns, UsageUnavailable  # noqa: E402
from xwalk.retrieval.postgres import NeonHTTPExecutor, SQLExecutor  # noqa: E402

DATABASE_ENV = "DATABASE_URL"
SCHEMA_ENV = "NER_SITE_NEON_SCHEMA"
# What the site accepts for the schema (ner_site.service.Settings.from_env).
_SCHEMA = re.compile(r"[a-z_][a-z0-9_]{0,40}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create <schema>.site_runs and its index (idempotent).",
    )
    parser.add_argument(
        "--schema",
        help=f"the Postgres schema (default: {SCHEMA_ENV}, else {NEON_SCHEMA})",
    )
    return parser


def _executor(database_url: str) -> SQLExecutor:
    return NeonHTTPExecutor(database_url)


def main(
    argv: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    executor_factory: Callable[[str], SQLExecutor] | None = None,
    out: Callable[[str], None] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    env = env if env is not None else os.environ
    say = out or (lambda line: print(line, flush=True))

    database_url = (env.get(DATABASE_ENV) or "").strip()
    if not database_url:
        say(f"error: {DATABASE_ENV} is not set")
        return 2
    schema = args.schema or (env.get(SCHEMA_ENV) or "").strip() or NEON_SCHEMA
    if not _SCHEMA.fullmatch(schema):
        say("error: the schema must be a lower-case identifier (letters, digits, _)")
        return 2
    try:
        executor = (executor_factory or _executor)(database_url)
    except ValueError:  # its message may quote the connection string
        say(f"error: {DATABASE_ENV} is not a postgres:// or postgresql:// URL with a host")
        return 2
    try:
        SiteRuns(executor, schema).create()
    except UsageUnavailable as exc:  # the class and the SQLSTATE, nothing the server wrote
        say(f"error: {exc}")
        return 1
    say(f"{schema}.{TABLE} and its index on at are in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
