# Site analytics: traffic and usage

Date: 18 November 2026. Scope: `ner-site/` only. Builds on the two earlier specs of this
date (Claude by default; own keys and daily cap).

## 1. Goal and decisions

The operator wants to see who visits and what the demo is used for and costs.

| # | Decision (operator, 18 November 2026) |
|---|---|
| 1 | Traffic through Vercel Web Analytics: no cookies, script from the site's own origin. |
| 2 | Usage through one row per run in the existing Neon database: no text, no prompt, no IP address, no key. |
| 3 | A private stats page on the site, behind its own passcode, summarising usage and estimated spend. |

Everything is off unless the operator turns it on: an existing deployment behaves exactly
as before.

## 2. Traffic: Vercel Web Analytics

- `public/index.html` loads two same-origin scripts, both `defer`: a new
  `public/analytics.js` holding the one-line queue stub Vercel's HTML snippet puts
  inline (`window.va = window.va || function () { (window.vaq = window.vaq ||
  []).push(arguments); };`), then `/_vercel/insights/script.js`. No inline script, so
  the Content-Security-Policy (`default-src 'self'`) is unchanged.
- Until the operator enables Web Analytics for the project in the Vercel dashboard and
  redeploys, the second script answers 404, which is harmless and must not break the
  page (verify in a browser with the canned server).
- Only page views (custom events need a paid plan). The stats page does not load it.
- The footer gains: "This site counts visits without cookies."

## 3. Usage: one row per run (`usage.py`, new)

### 3.1 Switch

`NER_SITE_STATS_PASSCODE` (at least 16 characters, refused if shorter; a secret, scrubbed
like the others). Set: runs are recorded and the stats page works. Unset: nothing is
recorded, `/api/stats` answers 404, and `DATABASE_URL` is read no more than before.
When set, `DATABASE_URL` is read (the `uses_database` rule gains this case); set without
a usable `DATABASE_URL`, recording is skipped and `/api/stats` answers 503
`stats_unavailable`.

### 3.2 Table

`<schema>.site_runs`, created with `CREATE TABLE IF NOT EXISTS` on first use of a warm
instance, in the same schema and through the same kind of non-retrying, short-timeout
executor as the limiter:

| Column | Type | Meaning |
|---|---|---|
| `at` | timestamptz | when the request finished (server clock, UTC) |
| `visitor_day` | text | `HMAC(salt, visitor id + UTC date)`, 32 hex: counts distinct visitors per day, cannot be joined across days or back to an address |
| `status` | integer | HTTP status of the answer |
| `code` | text null | error code, null on success |
| `credential` | text | `site` or `visitor` |
| `provider` | text | `site`, `anthropic`, `openai`, `openrouter` |
| `tier` | text null | admitted tier (`opus`, `sonnet`, `other`, `own`), null when not admitted |
| `requested_model`, `model` | text null | as in the response |
| `requested_selector` | text null | as in the response |
| `fallback_reasons` | text | comma-joined distinct `reason` values, empty when none |
| `country` | text null | two-letter code from `x-vercel-ip-country` when it matches `[A-Z]{2}` |
| `text_chars` | integer null | length of the submitted text |
| `custom_prompt` | boolean null | |
| `entities`, `matches`, `picks` | integer null | counts |
| `llm_calls`, `prompt_tokens`, `completion_tokens` | integer null | extraction usage |
| `selector_calls`, `selector_prompt_tokens`, `selector_completion_tokens` | integer null | selection usage |
| `selector_cost_usd` | double precision null | what the provider reported (Jev) |
| `embedding_requests` | integer null | |
| `extract_ms`, `map_ms`, `select_ms`, `total_ms` | integer null | timings |

An index on `at`. Rows are kept; nothing prunes them (a year of a busy demo is a few
megabytes). Every value is a parameter. Nothing else is stored: in particular no text,
prompt, entity, term id, IP address, visitor id, passcode, key or user agent.

### 3.3 What is recorded

Every `POST /api/ner` that got past the passcode and body parsing (so 200s, the 4xx and
5xx that follow, 429s and 503s from admission), exactly once, from the values the
handler already puts in its log line. Health, stats and static requests are not
recorded. Requests refused for a bad header, a wrong passcode or an unreadable body are
not recorded (they have no usable fields and would let anyone fill the table).

### 3.4 Never in the way

Recording happens after the response is built and must not change it: one statement,
one round trip, a 2 s timeout, no retry; any failure is swallowed and noted in the log
line as `usage: "<ExceptionClass>"`. A recording failure never turns a 200 into an
error and never delays the answer by more than the timeout. Own-key requests are
recorded like any other (the key is not among the fields).

## 4. The stats API (`/api/stats`)

`GET /api/stats?days=N` (1..365, default 30). Requires header `X-Stats-Passcode` equal
to `NER_SITE_STATS_PASSCODE` (`hmac.compare_digest`); 401 `stats_passcode_required` /
`stats_passcode_invalid` otherwise, 404 when the variable is unset. It is independent of
the visitor passcode and the limiter. Responses are `no-store`. A new Vercel function
shim `api/stats.py` like the two existing ones.

Answer, computed in SQL over the last N UTC days (one or two statements):

```json
{"ok": true, "days": 30, "generated_at": "...",
 "totals": {"runs": 0, "ok": 0, "errors": 0, "visitors": 0,
            "prompt_tokens": 0, "completion_tokens": 0,
            "estimated_usd": 0.0, "unpriced_runs": 0},
 "by_day": [{"day": "2026-11-18", "runs": 0, "ok": 0, "visitors": 0,
             "opus": 0, "sonnet": 0, "other": 0, "own": 0,
             "stepped_down": 0, "estimated_usd": 0.0,
             "p50_ms": 0, "p95_ms": 0}],
 "by_model": [{"model": "...", "credential": "site", "runs": 0,
               "prompt_tokens": 0, "completion_tokens": 0, "estimated_usd": 0.0}],
 "by_code": [{"code": "visitor_limit_reached", "runs": 0}],
 "by_fallback": [{"reason": "rate_limited", "runs": 0}],
 "by_country": [{"country": "SI", "runs": 0}]}
```

- `visitors` is the count of distinct `visitor_day` values (per day; the total is the
  sum of the daily counts, described on the page as "visitor-days").
- **Estimated dollars** are for runs on the operator's keys only (`credential = site`),
  from a price table in code (per million tokens, input / output): `claude-opus-5-5`
  4 / 20, `claude-sonnet-5-5` 2 / 10; plus `selector_cost_usd` where reported. Tokens of
  a run are attributed to its `model` (extraction) and, for selection tokens, to the
  requested selector when it is a model. Models without a price count in
  `unpriced_runs` and contribute nothing. The page says plainly that this is an
  estimate (a run that stepped down mid-way is priced at the model that answered the
  extraction; the Anthropic Console is the bill).
- Days with no runs are present with zeros so charts have no gaps.

## 5. The stats page (`public/stats.html`, `stats.js`, `stats.css` or shared CSS)

- At `/stats` (a `vercel.json` rewrite to `stats.html`, or `/stats.html`; whichever
  needs no CSP change), linked from nowhere.
- A passcode field (type password, kept in `sessionStorage`, sent only to
  `GET /api/stats`), a days selector (7, 30, 90), then:
  - headline numbers: runs, successful runs, visitor-days, estimated dollars, and the
    share of runs that stepped down;
  - two small charts drawn as SVG with DOM calls (no library, no inline style): runs
    per day stacked by tier, and estimated dollars per day;
  - tables: by model (with tokens and dollars), by error code, by step-down reason, by
    country, and latency (p50, p95) per day.
- Same visual language as the main page, readable at phone width, light and dark.
  Everything through `textContent`. `noindex` meta tag.
- Empty and error states: wrong passcode, stats not enabled (404), database unavailable
  (503), no runs yet.

## 6. Settings, wire, docs

- `NER_SITE_STATS_PASSCODE` joins `ENV_NAMES`, `secrets()` and the README variables
  table. The log line gains `usage` only on a recording failure.
- `/api/health` gains nothing (whether stats are on is not public).
- README: a "Analytics" section (what is collected and what is not, how to turn each
  half on, how to open the stats page, how to delete the data: `DROP TABLE` or a date
  `DELETE`), and the footer sentence. `docs/ner-site/PLAN.md` section 16.
- The main page's footer sentence about the salted hash stays; add "Runs are counted
  without your text."

## 7. Testing

Offline. Recording: each recorded status class writes exactly one row with the expected
fields and none of the sentinel text, prompt, IP address, visitor id, passcode or key;
unrecorded requests write nothing; a failing or slow recorder leaves the response
untouched and is noted in the log; unset variable means no executor is built and no
`DATABASE_URL` read. `visitor_day` differs across days and salts and equals for the same
visitor on one day. Stats: passcode rules, 404 when off, the aggregation on a fixed set
of rows (through the in-memory fake the limiter tests use, and as `integration`-marked
tests on Postgres), zero-filled days, the price arithmetic, `days` bounds. Page:
fixtures for a populated and an empty answer; verified in a browser.

## 8. Rollout (operator)

1. In the Vercel dashboard, enable Web Analytics for the project; redeploy; a visit
   shows a request to `/_vercel/insights/view` and appears in the dashboard.
2. Set `NER_SITE_STATS_PASSCODE` (sensitive, 16+ characters) and redeploy; do a run;
   open `/stats`, enter the passcode, see the run.
3. Confirm in the Neon console that `site_runs` holds no text or address.

## 9. As built (18 November 2026, after review)

The sections above are the design. Where the code differs, this section holds.

- **Enablement is not secret (amends 4 and 6).** With the variable unset `/api/stats`
  answers the application's 404 and that is sufficient; with it set, a request without
  the passcode answers 401, so the two can be told apart from outside. Nothing is done
  to hide that, in the shim or elsewhere. `/api/health` still says nothing about stats.
- **The passcode is at least 24 characters (amends 3.1 and 8).** The endpoint has no
  attempt limit of its own, so the passcode must be a generated secret
  (`openssl rand -hex 24`). The comparison is constant-time and comes before any
  database call. A Vercel Firewall rate-limit rule on `/api/stats` is the optional
  second layer.
- **Recording is capped (amends 3.4).** Finding or building the store and the insert
  run in one worker thread, and the request waits for it at most 2 s in all. At most
  four writes are under way per warm instance; a run that finds them all taken is not
  recorded and does not wait, and its log line says `usage: "saturated"`. The write
  goes through an executor of the site's own (`ner_site/deadline.py`) whose timeout
  is a wall-clock deadline for the whole exchange, name lookup and complete answer
  included: at the deadline the write is cancelled and its connection closed. The
  deadline is one, for the whole attempt, taken on the monotonic clock before any
  setup: the lock the store is found behind is waited for no longer than to it, a
  write that gets the lock after it does not start, the SQL call is given only the
  time that is left, and a write the loop had not got to by then is never sent. So a
  place is free again when the deadline passes (tested to within one second: with
  the store's lock held past the deadline and four writes waiting, none of which
  starts when the lock is released; against an answer that trickles in the body or
  the headers; a silent server; and a resolver that never answers). No failure of
  that executor carries the connection string or anything the server sent: each is
  one of a few fixed-message errors with no cause and no context, and a SQLSTATE is
  kept only when it has the form of one. The engine's executor limits each read, and a database that
  sent a byte now and then held a place without end. A name lookup cannot be
  cancelled: it keeps one of at most four daemon threads until the system's resolver
  gives up, and while all four are stuck a write fails at once.
- **The table can be provisioned (amends 3.2).** The lazy `CREATE TABLE IF NOT EXISTS`
  stays; two cold instances racing on a database without the table may lose one row.
  `tools/setup_usage.py` creates the schema, the table and the index once from
  `DATABASE_URL`, so that nothing is left to create.
- **An unusable `DATABASE_URL` (amends 3.1).** Where the stats passcode is the only
  reason the variable is read, a value that cannot be used (wrong scheme, no host,
  malformed) is not a settings error: recording is skipped, an authenticated
  `/api/stats` answers 503 `stats_unavailable`, and `/api/ner` and `/api/health`
  answer as with stats off. Where the index backend or the limiter needs the
  variable, validation is as strict as before.
- **Only the site's own words are stored (amends 3.2).** A model or selector that is
  not an id the deployment lists (the answering model comes from the provider's
  response), and an error code or step-down reason the site does not define, are
  stored as `other`. An unknown tier is null. One exception, decided after the
  fixes: a known id followed by `-` and a date or version made of digits and hyphens
  (`[0-9][0-9-]{0,15}`, to the end of the string; `claude-opus-5-5-20261101`) is
  stored as the known id, the longest one that fits, so that a dated echo of a Claude
  id is priced. Nothing of the suffix is stored.
- **Spend (amends 3.3 and 4).** A run that failed after a model answered records the
  calls and tokens its failure carried under the model the request asked for (the
  tokens are a sum over the models the chain tried; it only steps down to cheaper
  ones, so this can overstate and never understate). `unpriced_runs`
  counts a site-key run once when either stage has no list price: the extraction
  model, or a model selector whose provider reported no cost. Jev's reported cost is
  a `by_model` row of its own (`model: "jev"`, `credential: "site"`, `runs: 0`), so
  the rows' dollars add up to the total. Still not counted: a successful run in which
  any call did not report its tokens (its token columns are null).
- **The answer is normalised too (amends 4).** The strings of `/api/stats` (models,
  selectors, credentials, codes, reasons, countries) go through the same mapping when
  they are read as when a row is written, so a row from before the rule or written
  by hand adds nothing free-form to the answer; groups that become the same word are
  added up.
- **A dropped table.** An instance remembers that it created the table, and only a
  statement that created it sets that memory (an insert's success does not: its
  answer may be late, from before the drop). After a `DROP TABLE` the first
  statement an instance sends fails (SQLSTATE 42P01, that row is lost, as are any
  that were under way at that moment) and the next one it sends creates the table
  again; `tools/setup_usage.py` after the drop avoids the loss.
- **A port in an analytics-only `DATABASE_URL`.** "Unusable" includes what the URL
  parser only checks when asked: a port that is not a number in range, a host that
  is not a DNS name (labels of 1 to 63 letters, digits and hyphens with no hyphen at
  an end, one dot allowed at the very end; an address, bracketed or not, is refused:
  a Neon host is a name), anything that is not printable ASCII. Nothing of an
  unusable URL is kept, so nothing of it is scrubbed from answers.
- **Null rows (amends 4).** `by_model` has a row with `"model": null` per credential
  for runs no model answered, and `by_country` a row with `"country": null` for runs
  without a country, so that each list's runs add up to every run.
- **`days` is refused, not clamped (amends 4).** Outside 1 to 365, repeated, signed,
  zero-padded or not a number: 400 `bad_days`, after the passcode check.
- **The read has its own timeout (amends 3.2).** Reading the statistics waits up to
  10 s (no retry), so that a Neon compute that has to wake up does not fail the
  operator's page. Recording keeps 2 s.
- **`visitor_day` of a refused request.** A request refused for its credential is
  counted under the address it came from when that is an IP address, like an admitted
  one; anything else in an address header is not hashed.
- **The page.** The residual chart segment is "No tier"; the step-down headline says
  runs "had a step-down note"; the passcode field is emptied once the passcode is
  accepted and the session value is used from then on.
