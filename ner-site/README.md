# ner-site: NER-to-ontology demo on Vercel

A minimal web page where a visitor pastes text and an extraction prompt; one LLM call
extracts named entities (chemical, disease, gene, food, other) with character spans, and
xwalk retrieval maps each entity onto the matching bundled sample ontology (ChEBI, CTD
disease, NCBI Gene, FoodOn).

This directory is a separate Vercel project (Root Directory `ner-site`). It is not part
of the `xwalk` package and is not installed with it. The plan, contracts and workstreams
are in [docs/ner-site/PLAN.md](../docs/ner-site/PLAN.md).

- `ner_site/`: all logic, importable and tested offline from `tests/test_site_*.py`
  (pytest puts `ner-site/` on `sys.path`).
- `api/ner.py`, `api/health.py`: the two Vercel Python functions, thin shims over
  `ner_site.http.handle`.
- `public/`: the page (`index.html`, `app.js`, `app.css`; no build step, no
  dependencies). It builds every element with `textContent`, so model output is only
  ever rendered as text.
- `vercel.json`: function duration and bundled files, and the security headers.
- `tools/serve_local.py`: a standard-library server for `public/` and `/api/*`.
- `tools/claude_check.py`: posts text files to a running site and prints the model that
  ran, the step-downs, timings and tokens per request (for the rollout below; every run
  makes real requests).
- `tools/screenshot.mjs`: Playwright smoke check that loads the page, runs the example
  and writes a PNG.
- `tools/fixtures/`: canned `/api` responses (`health*.json`, `ner*.json`,
  `error*.json`) for working on the page without the service.

## Deploy

### Run it locally first

From the repository root, with the venv active (`. .venv/bin/activate`):

```bash
# Page only, canned API answers (no ner_site import, no key, no network)
python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures
# ... the same with dense retrieval unavailable and offline mode, to see the banner
python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures --scenario degraded
# ... or one of the Claude states: sonnet, blocked, fallback (see below)
python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures --scenario sonnet

# The real service with the stand-in extractor (NER_SITE_OFFLINE=1; no key needed)
python ner-site/tools/serve_local.py --offline

# The real service against OpenRouter (spends money on your key)
OPENROUTER_API_KEY=... python ner-site/tools/serve_local.py

# The real service with Claude first (spends your Anthropic balance; add
# OPENROUTER_API_KEY for Jev, dense retrieval and the last step of the chain)
ANTHROPIC_API_KEY=... python ner-site/tools/serve_local.py
```

The canned scenarios (`--scenario NAME`). A scenario needs only the fixture files it
changes; the others fall back to `health.json` and `ner.json`:

| Scenario | What the page is given |
|---|---|
| (none) | Claude Opus 5.5 extracted and picked; the visitor is on the Opus allowance (`visitor.tier: opus`). No notices. |
| `sonnet` | The hour's Opus allowance is used: health says `visitor.tier: sonnet`, and the response carries the `admit` step-down with its notice. |
| `blocked` | The hour's requests are used: health says `visitor.tier: blocked`, and every POST answers 429 `visitor_limit_reached` with `Retry-After` (`error-blocked.json`). |
| `fallback` | Opus stepped down while extracting (`rate_limited`) and Sonnet while selecting (`overloaded`): two notices, picks made by two models, one failed pick. |
| `degraded` | Offline mode with dense retrieval unavailable, in the response shape of the API before the Claude change (no `claude`, `visitor`, `fallbacks` or `notices`), so it also checks the page against an older API. |

The server binds `127.0.0.1`, port 8765 by default (`--port`), and adds the headers
from `vercel.json` to every response, so the page runs under the production
Content-Security-Policy. With `--canned`, posting text that contains `ERROR` returns
`fixtures/error.json` with HTTP 400 to show the error box (a scenario with an
`error-NAME.json` of its own answers every POST with it); `--delay SECONDS` slows the
POST to show the busy state.

Smoke check (Playwright is resolved from `NODE_PATH` or `npm root -g`):

```bash
node ner-site/tools/screenshot.mjs http://127.0.0.1:8765 /tmp/ner.png --width 1280
node ner-site/tools/screenshot.mjs http://127.0.0.1:8765 /tmp/ner-360.png --width 360
```

It clicks **Load example** and **Run**, waits for the result cards and exits non-zero
on a page error, a console error, a failed run, or horizontal scrolling. `--expect
error --text "...ERROR..."` checks the error state instead; `--dark` uses the dark
colour scheme.

### Deploy on Vercel (operator)

1. Import the repository into Vercel. Set **Root Directory** to `ner-site` and
   **Framework Preset** to "Other" (no build command). Vercel serves `public/` as the
   static site and `api/*.py` as Python functions (`.python-version` selects 3.12).
2. Under Settings, Environment Variables, add `OPENROUTER_API_KEY` (mark it
   sensitive). Optionally add `NER_SITE_PASSCODE` and `NER_SITE_MODELS`; see the table
   below. Redeploy after changing variables. `ANTHROPIC_API_KEY` puts Claude first and
   switches on the visitor allowance: follow "Rollout: Claude by default and open
   access" below before adding it to a public deployment.
3. Open `https://<deployment>/api/health` and check `"mode": "live"`,
   `"retrieval": {"dense": {"status": "ok"}}` and the model list. `dense: unavailable`
   with a reason means the committed indexes under `indexes/` are missing or stale, or
   the key is not set; mapping then runs on BM25 alone and the page shows a banner.
4. Load the page, run the example text, and confirm the entities, their highlighted
   spans and the matches. Check the function log for the cold-start time. If a
   passcode is set, confirm that a request without it gets 401
   (`curl -si -X POST -H 'Content-Type: application/json' -H 'X-Requested-With: xwalk-ner' -d '{"text":"aspirin"}' https://<deployment>/api/ner`).
5. Set a spend limit on the OpenRouter key (see the security notes), then record what
   was verified in a "Verified" section of this file.

### Rollout: Claude by default and open access (operator)

The code for this is merged but, as of 18 November 2026, nothing below has been done:
no deployment, no change to a production variable, no request to the Anthropic API.
Five things could not be checked offline. They are marked **Check** below, each with
what to look for, and all five come before the passcode is removed (step 8). Until
then the passcode is what stands between the public and the key.

`tools/claude_check.py` is the measuring tool. It GETs `/api/health` and POSTs
`/api/ner` on the URL it is given and nothing else; every text is a real request that
spends the balance of the key behind the site and counts against your own hourly
allowance there (`--help` says the same). The 12 real-world texts of 4 November are not
in this repository, so save the texts you want to measure as UTF-8 files first.

1. **Anthropic Console.** Create a workspace for the site and a key in it.
   **Check (auto-reload):** auto-reload must be off. The site has no dollar budget of
   its own: the prepaid balance is the only ceiling on what open access can spend, and
   auto-reload removes it.
2. **Vercel, production variables.** Add `ANTHROPIC_API_KEY` (sensitive). Remove
   `NER_SITE_MODELS`, `NER_SITE_SELECTORS` and `NER_SITE_SELECTOR`, or set them to the
   new defaults: production has `NER_SITE_MODELS=google/gemini-3.1-flash-lite,openai/gpt-6-luna`
   and `NER_SITE_SELECTOR=jev` from 4 November, and with those Claude is not offered
   for extraction and Jev stays the default selector. Keep `NER_SITE_PASSCODE` for now.
   Keep `OPENROUTER_API_KEY` (Jev, query embeddings, the last step of the chain) and
   `DATABASE_URL` (the allowance is counted there). Optionally set
   `NER_SITE_VISITOR_SALT`.
3. **Deploy, then `GET /api/health`.** Look for `"claude": {"status": "ok"}`,
   `"visitor": {"limits": "on", "tier": "opus", "opus_left": 10, "sonnet_left": 20, ...}`,
   `default_model` and `default_selector` both `claude-opus-5-5`, and
   `passcode_required: true`. This first call also creates the table
   `<NER_SITE_NEON_SCHEMA>.site_visits`. `visitor.limits: "unavailable"` means the
   limiter could not query Neon: do not go on (see step 7).
   From here on the allowance is counted even with the passcode set, your own requests
   included: 10 an hour start on Opus, the next 20 on Sonnet.
4. **Check (the live API accepts the requests).** Every Claude request sends the
   server-side refusal fallback (`fallbacks: "default"`, beta
   `server-side-fallback-2026-07-01`) together with schema-enforced output
   (`output_config.format`), and the schema is a normalised copy of the engine's
   (below, "What is sent to Claude"). Neither the combination nor the normalised
   schemas have met the live API. Run one short text:

   ```bash
   read -rs NER_SITE_PASSCODE && export NER_SITE_PASSCODE   # typed, not in history
   python ner-site/tools/claude_check.py --url https://www.xwalk.si \
       --passcode-env NER_SITE_PASSCODE one.txt
   ```

   Accepted looks like `200`, `model claude-opus-5-5 -> claude-opus-5-5`, picks
   `by claude-opus-5-5`, `fallbacks none`. Rejected looks like `502 FAILED
   provider_error` naming an Anthropic HTTP 400 (the extraction request was refused),
   or a 200 whose picks are all `failed` with a `failed picks:` reason naming an
   Anthropic HTTP 400 (the selection schema was refused). The chain does not step down
   on a rejected request (`bad_request`), so nothing hides it and nothing works around
   it: every Claude request would fail the same way. That needs a code change in
   `ner_site/claude.py` before going further (the design says that if the two cannot
   be combined, schema-enforced output stays and the refusal fallback goes).
5. **Check (time and tokens).** Run the real texts, at least one of which yields the
   full 25 entities, with Opus doing both steps:

   ```bash
   python ner-site/tools/claude_check.py --url https://www.xwalk.si \
       --passcode-env NER_SITE_PASSCODE --sequential-delay 5 texts/*.txt
   ```

   The request deadline is 100 s (extraction has 90 s of it, selection what is left,
   8 picks at a time). Look at the `most entities` and `slowest` lines: `total` in
   `timing_ms` should be well under 100000, and no line should show `failed picks: cut
   off by the request deadline`. The `tokens` lines give the input and output tokens
   per request for extraction and for selection; the site computes no dollars, so
   multiply by the prices of the models yourself and compare with the usage the
   Console shows (whether the reported tokens include an attempt the refusal fallback
   replaced was not checked). If selection on Opus is too slow or too dear, set
   `NER_SITE_SELECTOR=jev`: Opus then extracts and Jev picks.
   More than 10 texts in one clock hour start on Sonnet from the 11th (the line shows
   `admit:claude-opus-5-5>claude-sonnet-5-5:opus_allowance_used`). To measure them all
   on Opus, run 10 an hour (`visitor.resets_in_s` in the health block says when the
   hour ends) or raise `NER_SITE_OPUS_PER_HOUR` while measuring and set it back.
6. **Check (the allowance holds under parallel requests).** One visitor's requests are
   serialised with a Postgres advisory lock, and the guarantee depends on Neon's batch
   endpoint running a transaction at READ COMMITTED, which was verified on a local
   Postgres only. Fire a burst from one address, with a one-sentence text to keep it
   cheap:

   ```bash
   python ner-site/tools/claude_check.py --url https://www.xwalk.si \
       --passcode-env NER_SITE_PASSCODE --selector retrieval --burst 15 short.txt
   ```

   The tool compares where the requests started with the `opus_left` and `sonnet_left`
   health reported just before. Good: `within the allowance`, for example 10 on Opus
   and 5 on Sonnet in a fresh hour. Bad: `OVER THE ALLOWANCE` (exit code 1), more
   starts on a tier than were left; do not open the site. Requests reported as
   `uncounted (limits_unavailable)` waited more than 3 s for the lock or the database
   and ran on Sonnet outside the count; a few in a large burst are expected, and a
   burst that is mostly uncounted proves nothing, so repeat it smaller.
7. **Check (know what a Neon outage does, by design).** When the limiter cannot query
   its database (unreachable, or no answer within 3 s), no request is refused and none
   is counted: a request for Opus starts on Sonnet, with the notice "Visitor limits are
   unavailable; this ran on Claude Sonnet 5.5.". So for as long as Neon is down,
   Sonnet is unmetered for everyone and only the prepaid balance bounds it. It shows as
   `visitor.limits: "unavailable"` in `/api/health` and as a `limits` field (error
   class and SQLSTATE) in the request log. If it lasts, set `NER_SITE_PASSCODE` again
   or remove `ANTHROPIC_API_KEY` until it is over.
8. **Open the site.** Remove `NER_SITE_PASSCODE`, redeploy, and check
   `passcode_required: false`. Then confirm the allowance from one address in one
   clock hour (31 short requests; `--selector retrieval` keeps each to one call, and
   earlier requests of that hour count too): the 11th starts on Sonnet with the notice
   "Your Opus allowance for this hour is used; this ran on Claude Sonnet 5.5.", and
   the 31st answers 429 `visitor_limit_reached` with a `Retry-After` header. With the
   tool, in a fresh hour (the 429 makes its exit code 1, as it should):

   ```bash
   python ner-site/tools/claude_check.py --url https://www.xwalk.si \
       --selector retrieval --sequential-delay 1 $(printf 'short.txt %.0s' {1..31})
   ```

   Lines 1 to 10 read `model claude-opus-5-5 -> claude-opus-5-5`, lines 11 to 30 carry
   `admit:claude-opus-5-5>claude-sonnet-5-5:opus_allowance_used`, and line 31 reads
   `429 ... FAILED visitor_limit_reached`.
9. Record what was verified in the "Verified" section, as for earlier deployments.

Going back: removing `ANTHROPIC_API_KEY` and redeploying takes Claude out of the lists
and switches the allowance off; the site then runs on OpenRouter as before, with Jev
as the default selector unless `NER_SITE_SELECTOR` says otherwise.

### Environment variables (function runtime)

| Name | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | The operator's Anthropic key. With it the Claude models are offered and the visitor allowance applies (below). Mark it sensitive; it is scrubbed from every response and log line like the other secrets. |
| `OPENROUTER_API_KEY` | unset | The operator's OpenRouter key, used for the OpenRouter models, Jev, query embeddings and the last step of the fallback chain. With neither key (and not offline) `/api/ner` answers 503 `not_configured`. |
| `NER_SITE_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL for the **chat call only**. Query embeddings always go to OpenRouter (`https://openrouter.ai/api/v1/embeddings`) with `OPENROUTER_API_KEY`, whatever this says: the committed dense indexes record that embeddings URL in their identity, so embedding elsewhere would need a code change and rebuilt indexes. |
| `NER_SITE_MODELS` | `claude-opus-5-5,claude-sonnet-5-5,google/gemini-3.1-flash-lite,openai/gpt-6-luna` | Comma-separated model allowlist. An id starting with `claude-` is called on the Anthropic key, anything else through OpenRouter; only ids whose key is set are offered, and the first offered one is the default. The only Claude ids are `claude-opus-5-5` and `claude-sonnet-5-5`; another `claude-` id is refused (503 `not_configured`), as is a list that no set key can run. |
| `NER_SITE_SELECTORS` | `claude-opus-5-5,claude-sonnet-5-5,jev,google/gemini-3.1-flash-lite,openai/gpt-6-luna` | What may pick the match, after `retrieval`, which is always offered first. The same rule: only ids whose key is set are offered (Jev needs the OpenRouter key). |
| `NER_SITE_SELECTOR` | unset | The default selector. It must be `retrieval` or in the list above; unset, or when its key is not set, the default is the first selector that can run: `claude-opus-5-5` with the Anthropic key, `jev` with only the OpenRouter key. |
| `NER_SITE_CLAUDE_EFFORT` | `low` | `output_config.effort` of every Claude request: `low`, `medium` or `high`. |
| `NER_SITE_FALLBACK_MODEL` | `google/gemini-3.1-flash-lite` | The OpenRouter model at the end of the fallback chain, for extraction and for selection. Must be an OpenRouter model id (not a Claude id, `jev` or `retrieval`). Used only when `OPENROUTER_API_KEY` is set; without it the chain ends at Sonnet. |
| `NER_SITE_OPUS_PER_HOUR` / `NER_SITE_SONNET_PER_HOUR` | 10 / 20 | The per-visitor allowance for a clock hour (UTC): requests that start on Opus, then requests that start on Sonnet. Their sum is the hour's total for every kind of request. Whole numbers 0 to 1000; anything else is refused. |
| `NER_SITE_VISITOR_SALT` | derived | Salt for the visitor ids (a keyed hash of the client address). Unset, it is the SHA-256 of `DATABASE_URL`. Changing it, or the database URL when it is unset, gives every visitor a new id, which resets the current hour's counts. |
| `NER_SITE_PASSCODE` | unset | When set, every `/api/ner` request must send it as `X-Passcode`; the page asks for it and keeps it in `sessionStorage`. ASCII only (8 characters or more recommended): header values arrive as Latin-1, so a non-ASCII passcode could never match and is refused with 503 `not_configured`. It does not exempt anyone from the visitor allowance. |
| `NER_SITE_MAX_CHARS` | 5000 | Maximum text length (characters). |
| `NER_SITE_MAX_ENTITIES` | 25 | Maximum entities kept per request; the default prompt (and `/api/health`'s `default_prompt`) states this number. |
| `NER_SITE_MAX_PROMPT_CHARS` | 4000 | Maximum prompt length (characters). |
| `NER_SITE_OFFLINE` | unset | `1` replaces the LLM with the stand-in extractor; `/api/health` reports `"mode": "offline"`. |
| `NER_SITE_DENSE_DIR` | `<project root>/indexes` | Where the committed dense indexes live. |
| `NER_SITE_CACHE_DIR` | `/tmp/xwalk-ner-site` | Where BM25 indexes are built at cold start. |
| `NER_SITE_BACKEND` | `bundled` | `neon` reads the records, BM25 and vectors from Neon Postgres instead of building BM25 in `/tmp` and opening the committed dense indexes (see "Neon backend" below). |
| `DATABASE_URL` | unset | The Neon connection string (mark it sensitive; scrubbed from every response and log line). The indexes use it only with `NER_SITE_BACKEND=neon`; without it every ontology then reports "DATABASE_URL is not set (NER_SITE_BACKEND=neon)". Since 18 November 2026 it is read on every backend, because the visitor allowance is counted there too; a value that is not a Postgres URL is refused. |
| `NER_SITE_NEON_SCHEMA` | `xwalk` | The schema `tools/load_neon.py` wrote. The allowance table `site_visits` is created in it; the schema itself must already exist. |
| `NER_SITE_NEON_DIMENSIONS` | 1024 | The loaded vector dimension (`load_neon.py --dimensions`); a different value makes dense stale. |
| `NER_SITE_NEON_CATALOG` | `samples` | `full` serves the full catalogs (`chebi`, `ctd-disease`, `ncbi-gene`, `foodon`) once they are loaded; needs `NER_SITE_BACKEND=neon`. |

### Claude, the fallback chain and the visitor allowance

As built on 18 November 2026 (decisions and wire contract:
[docs/ner-site/PLAN.md](../docs/ner-site/PLAN.md) section 14).

**Which models are offered.** With `ANTHROPIC_API_KEY` the two Claude ids lead both
lists and Opus is the default for extraction and for selection. With only that key, the
OpenRouter ids and Jev are not offered, dense retrieval is unavailable (query embeddings
need the OpenRouter key) and the chain ends at Sonnet. With only `OPENROUTER_API_KEY`
the Claude ids are not offered, the default model is `google/gemini-3.1-flash-lite` and
the default selector is `jev` (it was `retrieval` before this change; set
`NER_SITE_SELECTOR=retrieval` to keep that).

**The chain.** A request for a Claude model runs on a chain from that model down:
Opus, Sonnet, then `NER_SITE_FALLBACK_MODEL` through OpenRouter (the fallback model
does the picking too when selection reaches that step, not Jev). One chain serves the
whole request, so a step-down while extracting also moves selection, and a step that
was left is not tried again in that request.

| What happened on a step | What the chain does | `reason` in `fallbacks` |
|---|---|---|
| Anthropic rate limit (429) | next step at once | `rate_limited` |
| Anthropic overloaded (529) | next step at once | `overloaded` |
| Balance exhausted (402, or a 400 naming the credit balance) | next step at once; remembered for 300 s on that warm instance | `balance_exhausted` |
| Key refused (401, 403) | next step at once; remembered for 300 s | `key_refused` |
| Claude declined the text, after the server-side refusal fallback | next step at once | `refused` |
| The step's call cap for the request was reached | next step at once | `call_limit` |
| 5xx, 408, 409, connection error, timeout | the same step once more, then the next | `transient` |
| A request the API rejects (other 4xx), or an unreadable answer | no step-down: extraction answers 502 `provider_error`, a pick is `failed` | none |

On the last step the error is returned as it is. Steps decided before any call have
stage `admit`: `opus_allowance_used` (the visitor's Opus allowance), `limits_unavailable`
(the limiter could not be asked) and `claude_unavailable` (a remembered exhausted
balance or refused key: the request starts on the OpenRouter step). One more reason,
`follows_extraction` (stage `select`), is not a failure: when the model and the selector
are two different Claude ids, selection runs on the request's one chain, that is where
extraction ended, and the response says so. A Claude selector with an OpenRouter
extraction model gets a chain of its own for the picks. Every departure is in
`fallbacks` and, as a sentence, in `notices`; `model` is the model that extracted and
each `pick.selector` the model that made that pick.

**What is sent to Claude.** One Messages request per call through the `anthropic` SDK
with no retries of its own, no sampling parameters and no `thinking` parameter;
`output_config.effort` from `NER_SITE_CLAUDE_EFFORT`; `max_tokens` 16000 for extraction
and 8000 for selection (thinking counts against it); the server-side refusal fallback
(`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`); and schema-enforced
output. The schema is a normalised copy of the engine's: every object lists all of its
properties as required and forbids others, a property that was optional becomes
required but nullable, a list-valued `type` and `oneOf` become `anyOf`, and keywords
outside the supported subset (numeric bounds, lengths, patterns) are moved into the
`description`. The site validates the answer itself, as before.

**The visitor allowance.** It applies where Claude is offered (the Anthropic key is
set and the site is not offline); a deployment without the key counts nothing and
behaves as before. A visitor is the first 32 hex characters of an HMAC-SHA256 of the
client address (`x-real-ip`, else `x-vercel-forwarded-for`, else the first entry of
`x-forwarded-for`) under the salt. The address itself is never stored or logged; the
id is stored only in the table below and is never logged or returned. Requests with no
address header share one allowance.

- Counts are rows in `<NER_SITE_NEON_SCHEMA>.site_visits` (visitor, UTC clock hour,
  tier, count), created on first use; rows older than a day are deleted by about one
  request in fifty.
- A request is counted after every check that can refuse it (size, header, passcode,
  JSON, field limits), under the tier it starts on: `opus`, `sonnet`, or `other` for an
  OpenRouter model with a non-Claude selector. The tier wanted comes from the model,
  or from the selector when only the selector is Claude.
- Opus is granted while fewer than 10 Opus starts are recorded for the hour, then
  Sonnet while fewer than 20 Sonnet starts are. When the hour's total (all three tiers)
  reaches 30, or a request for Sonnet finds the Sonnet allowance used, the answer is
  429 `visitor_limit_reached` with `Retry-After` (seconds to the next clock hour) and
  nothing is recorded.
- One visitor's concurrent requests are serialised with a Postgres advisory lock taken
  in the same transaction, so a parallel burst cannot be granted more than the
  allowance. The limiter has its own Neon executor: 3 s timeout and no retries (a
  retried increment could count one visit twice).
- When the limiter cannot be asked, the request is not counted and a request for Opus
  starts on Sonnet (`limits_unavailable`). The same happens with no `DATABASE_URL` and
  no passcode; with no `DATABASE_URL` and a passcode (a private deployment) Claude is
  unmetered and starts on Opus.

`/api/health` reports `claude` (`{"status": "ok"}`, or `unavailable` with the reason:
"key not set", "key refused", "balance exhausted", "Anthropic answered HTTP n",
"Anthropic unreachable (...)"; the key is checked with a free Models API call,
remembered for 300 s when ok and 60 s otherwise) and, where the allowance applies,
`visitor` for the caller: `limits` (`on`, `off`, `unavailable`), `tier` (where a request
for Opus would start: `opus`, `sonnet`, `blocked`), `opus_left`, `sonnet_left` (each
capped by what remains of the hour's total) and `resets_in_s`. Offline mode reports
neither.

Tests for the limiter's SQL need a real Postgres and are not part of the default suite.
With a Postgres URL in `NER_SITE_TEST_DATABASE_URL` and the `psql` client (16 or later)
on the path, they run the real statements, including 40 overlapping requests of one
visitor, each test in a schema of its own that it drops; without either they skip:

```bash
NER_SITE_TEST_DATABASE_URL=postgresql://user@localhost:5432/postgres \
    python -m pytest -q -m integration tests/test_site_limits.py
```

### Neon backend

The four samples (or, on a large enough plan, the full catalogs) can be served from Neon
Postgres: records, BM25 (`lakebase_text`) and `halfvec` vectors (pgvector HNSW), queried
over Neon's HTTPS SQL endpoint. Full details, sizes and costs:
[docs/ner-site/NEON.md](../docs/ner-site/NEON.md).

```bash
set -a; . ./.env.local; set +a            # DATABASE_URL and OPENROUTER_API_KEY
python ner-site/tools/load_neon.py --dry-run      # counts, size, tokens, cost; writes nothing
python ner-site/tools/load_neon.py                # loads the four samples (about $0.0007)
NER_SITE_BACKEND=neon python ner-site/tools/serve_local.py
```

`/api/health` then shows `"backend": "neon"`; a missing, incomplete or stale load is
reported per ontology with the command that fixes it. The loader is resumable (rerun it
after an interruption; it embeds only what is missing). On Vercel, set
`NER_SITE_BACKEND=neon` and `DATABASE_URL` (the Neon integration provides it), after
re-pinning `requirements.txt` to an xwalk commit that contains
`xwalk.retrieval.postgres`.

### Security notes for the operator

- **Each key has its own hard cap, and the site adds none in dollars.** For the
  OpenRouter key it is the spend limit set on the key; for the Anthropic key it is the
  prepaid balance, with auto-reload off in the Anthropic Console. These are the only
  caps that hold across all function instances. Per request the service allows at most
  3 extraction calls on a step of the chain (retries included; a Claude step makes one
  call, or two after a transient error), selection calls up to twice the entity limit
  on a step (normally one per entity) and 25 entities (query embeddings are never
  retried), and a request finishes by about 105 s (90 s for the extraction, one
  deadline for the rest; Vercel cuts a run off at 120 s), but nothing stops many
  addresses from sending requests.
- **Without the Anthropic key, set `NER_SITE_PASSCODE` for a public deployment.**
  Nothing counts requests there, so anyone who finds the URL can spend the OpenRouter
  key up to its limit. Share the passcode out of band; the service compares it in
  constant time. For more, Vercel's firewall rate limiting is an option.
- **With the Anthropic key, the visitor allowance is what makes an open deployment
  bounded per visitor** (30 requests an hour by default, 10 of them starting on Opus).
  It bounds one address, not the total: many addresses can still spend the balance,
  which is why the balance must be the ceiling. It needs `DATABASE_URL`; when the
  database cannot be asked, requests run on Sonnet uncounted (rollout step 7).
- The keys live only in the function environment. `/api/health` never returns them, and
  error messages pass URLs through redaction. Do not put a key in `NER_SITE_BASE_URL`.
- `vercel.json` sends `Content-Security-Policy: default-src 'self'; connect-src
  'self'; img-src 'self' data:; style-src 'self'`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, and `Cache-Control: no-store` on `/api/*`. No CORS
  headers are sent, and the API requires `X-Requested-With: xwalk-ner`, so other
  sites cannot call it from a browser.
- Visitors' text is sent to the model provider (Anthropic for the Claude models;
  OpenRouter and the model's host for the others, and OpenRouter for query embeddings
  and Jev). The function log records sizes, counts, timings and status only, never the
  text, the prompt or a key; since 18 November 2026 also the tier a request started on,
  its step-downs and the model that ran, and never the visitor id or the client
  address. The page's footer tells visitors that requests are limited per visitor using
  a salted hash of their IP address, kept for one day.

## Verified

### Locally, 3 November 2026 (WS-D; Python 3.13, the repository venv)

What was run, and nothing more:

- `python -m pytest -q tests/test_site_service.py tests/test_site_http.py`: 111 passed.
  Offline only (FakeLLM, the offline stand-in, `httpx.MockTransport`; BM25 built in
  `tmp_path`). They cover every error code in plan section 3.1 through `handle`, the
  passcode, `X-Requested-With`, OPTIONS 405, the health shape (validated with
  `HealthResponse`), a full offline POST, a full live POST against a mocked chat
  endpoint, a full POST with dense indexes built in `tmp_path` by the default Qwen3
  encoder against a mocked 26-letter histogram embeddings endpoint, both `api/*.py`
  shims on a real `ThreadingHTTPServer` socket (including 413 before reading, 411, HEAD,
  and a bad setting giving 503), and that a sentinel key, the passcode, the text and the
  prompt appear in no response and no log line.
- `python ner-site/tools/serve_local.py --offline`, started with `OPENROUTER_API_KEY`
  removed from its environment, then `curl /api/health` (200, `mode: offline`,
  `default_model: offline`, BM25 ok, dense unavailable "OPENROUTER_API_KEY is not set
  ... (all 4 ontologies)"), one `curl -X POST /api/ner` with the page's example text (200,
  11 entities whose offsets equal `tools/fixtures/ner.json`, `llm_calls: 0`, null
  tokens), and `curl -X OPTIONS /api/ner` (405, `Allow: POST`, no CORS headers).
- `node tools/screenshot.mjs` at 1280 px and at `--width 360` against that server: both
  exit 0 with "11 entities found, 9 with candidate terms", no horizontal scroll.
- The four gates from `CONTRIBUTING.md`, all passing.

Not run: anything against OpenRouter (no chat call, no embeddings call, no cost), any
Vercel build or deployment, Python 3.12, and the committed dense indexes opened with a
real key.

### Python 3.12 and live, 3 November 2026 (the orchestrating session)

- A fresh Python 3.12.3 venv with only `pip install -r ner-site/requirements.txt` (xwalk
  from GitHub at the pinned commit, as Vercel installs it): installs cleanly, 125 MB of
  site-packages, and the wheel ships the four ontology samples. All 201
  `tests/test_site_*.py` pass in it against that installed xwalk (checked that `xwalk`
  imported from site-packages, not the repository's `src/`).
- The committed dense indexes, built with the operator's key by
  `tools/build_indexes.py`: 3,132 vectors, 13 MB, reported cost $0.00065; all four open
  as `dense: ok`.
- `tools/serve_local.py` (live, not `--offline`) in that 3.12 venv with the key, then
  `tools/screenshot.mjs` at 1280 and 360 px with the page's example text:
  `/api/health` answered `mode: live`, `llm: ok`, BM25 ok, dense ok; the run found 10
  entities, all with candidate terms, with one `openai/gpt-4o-mini` call (641 + 186
  tokens), 10 embedding requests, 3.9 to 5.2 s in total. Sample terms present in the text
  ranked first in both retrievers (for example haloperidol `CHEBI:5613`, NAT2
  `NCBIGene:10`, Alkaptonuria `MESH:D000474`, cumin seed `FOODON:00003396`); aspirin,
  which is not in the 200-term ChEBI sample, got only loose dense neighbours. The
  request log line held no text, prompt or key.

After the review fixes (request deadline, per-request thread pool, no retry sleeps on
query embeddings), the same live check was repeated in the 3.12 venv: 223 site tests
pass; two runs of the example text gave 10 entities, all with candidates, one LLM call
(641 + 186 tokens), 10 embedding requests, dense ok, in 10.7 and 12.6 s. Mapping took
7.5 to 10 s against about 1 s in the first live run; single query embeddings through
OpenRouter were measured at 0.2 to 3.3 s each at that time, so the spread is provider
latency, not the fixes. Expect a few seconds to about 15 s per request.

### `vercel build`, 4 November 2026 (the orchestrating session)

Vercel CLI 65.11.0, `vercel build --yes` on a copy of `ner-site/` with a placeholder
local project link (no token, no deployment): `Build completed successfully`. It used
Python 3.12 from `.python-version`, installed `requirements.txt` with uv (xwalk from
GitHub at the pinned commit), and produced `api/ner.func` and `api/health.func`
(runtime `python3.12`, `maxDuration` 120) whose file maps hold `ner_site/`, all four
`indexes/<slug>/dense/` directories, xwalk with its four ontology samples, tantivy
0.27.0 and numpy; `public/` is served as static files, and the `vercel.json` headers
appear in the route config. So the `includeFiles` glob is accepted.

Still not run: a Vercel deployment.

### Review fixes, 3 November 2026 (Python 3.13, the repository venv)

What was run, and nothing more (offline only: FakeLLM, the offline stand-in,
`httpx.MockTransport`; no OpenRouter call, no Vercel build):

- New tests first, each seen failing for its reason before the fix: a stalled query
  gateway (6 s) with a 1 s retriever timeout and six entities made `handle` take 11.4 s
  (it waited for abandoned threads); with a 2 s request deadline 11.4 s; a 429 with
  `Retry-After: 120` left the query encoder sleeping; dense failing for every entity
  still reported `dense: ok`; a lone surrogate gave 500 `internal`
  (`UnicodeEncodeError` in httpx); `Content-Length: ²`, `¹` and 5000 digits dropped the
  shim's connection; a non-ASCII passcode was accepted; the default prompt said "at
  most 25" with `NER_SITE_MAX_ENTITIES=7`; a retriever note carried the key and URL
  userinfo into the entity notes. After the fixes the stalled request returns in about
  2 s and the deadline one inside 3.5 s, measured by wall clock in the tests.
- Each redaction step disabled in turn (scrub's secret replacement, scrub's URL
  redaction, the note cleaning in `run`, the body filter, the log filter): every one
  made at least one test fail; restored exactly.
- `python -m pytest -q tests/test_site_*.py`: 223 passed. The four gates from
  `CONTRIBUTING.md`, all passing.
- `python ner-site/tools/serve_local.py --offline --port 8797` (key removed from its
  environment): health 200 `mode: offline`, default prompt "at most 25 entities", dense
  unavailable (key not set); POST with the page's example text 200, 11 entities, 9 with
  matches, `llm_calls: 0`, null tokens; `Content-Length: ²` 400 `bad_header`; a
  `"\ud800"` text 400 `bad_json`. `tools/screenshot.mjs` at 1280 px: exit 0, "11
  entities found, 9 with candidate terms".

Not run: the deadline against a real slow provider, anything live, Python 3.12 again,
a Vercel deployment.

### Deployed on Vercel, 4 November 2026 (the orchestrating session)

Project `xwalk-ner` (Hobby plan), deployed with the Vercel CLI from `git archive` of
`ner-site/` at commit `13ee967` (bundled backend), framework preset "Other", functions
in `fra1` (next to the Neon project in AWS eu-central-1). Production variables, both
sensitive: `OPENROUTER_API_KEY` and `NER_SITE_PASSCODE`. Checked at
https://xwalk-ner.vercel.app: the page and `/api/health` answer 200 (`mode: live`,
`passcode_required: true`, BM25 ok, dense ok, llm ok); `POST /api/ner` without the
passcode is 401 `passcode_required`; with it, a five-entity text returned 200 in 5.1 s
round trip (extract 2.7 s, map 0.7 s, one gpt-4o-mini call, 609 + 96 tokens, 5
embedding requests), every sample term at rank 1 in both retrievers. Domains
`www.xwalk.si` (primary) and `xwalk.si` (308 to www) are attached; DNS at Dynadot
still pointed at Dynadot parking when this was written (apex A records
216.198.79.1 and 64.29.17.1, www CNAME to the project's `*.vercel-dns-017.com`
target, per Vercel's domain config for the project).

### Switched to Neon, 4 November 2026

Production variables added (production target): `DATABASE_URL` (Neon's pooled connection
string, sensitive), `NER_SITE_BACKEND=neon`, `NER_SITE_NEON_CATALOG=samples` (the full
catalogs are not loaded: the Neon org is on the free plan). Deployed with the Vercel CLI
from `git archive` of `ner-site/` at `c20ed3d`, functions in `fra1`. Checked at
https://www.xwalk.si and https://xwalk-ner.vercel.app:

- DNS: Vercel's domain config reports `misconfigured: false` for `www.xwalk.si` (CNAME)
  and `xwalk.si` (A 216.198.79.1, 64.29.17.1); `https://www.xwalk.si` answers 200 with a
  Let's Encrypt certificate, `https://xwalk.si` 308 to `https://www.xwalk.si/`.
- `/api/health` 200: `backend: neon`, `neon: {schema: xwalk, catalog: samples}`, BM25 ok,
  dense ok, llm ok, `passcode_required: true`.
- `POST /api/ner` without the passcode 401 `passcode_required`, with a wrong one 401
  `passcode_invalid`.
- With it, a six-entity text of sample terms: 200, every sample term at rank 1 in both
  `bm25` and `dense-qwen3`, one gpt-4o-mini call (622 + 117 tokens), 6 embedding
  requests; extract 2.2 to 2.7 s, **map 19 to 23 s**. The map time is the query
  embeddings, not Neon: run locally against the same database with httpx logging, every
  Neon SQL request took 120 to 170 ms while OpenRouter's embeddings requests (Qwen3 8B,
  provider DeepInfra) took 0.5 to 12 s that hour (median 6.4 s, four in flight). One
  single-entity run hit the encoder's 15 s timeout; the site answered with BM25 matches
  and the warning "dense retrieval unavailable; matches use BM25 only", as designed.
- Retrieval quality is unchanged by the switch: on the pilot gold sets the fused mean MRR
  is 0.841 on Neon and 0.839 bundled (`benchmarks/results/site-retrieval-2026-11-04/`).

Fallback: the committed dense indexes are still deployed, so setting
`NER_SITE_BACKEND=bundled` (or removing it) and redeploying restores the bundled backend
without a rebuild.

### Full catalogs and selectors, 4 November 2026 (later the same day)

Production now serves the full ChEBI, CTD diseases and FoodOn vocabularies from Neon
(schema `xwalk_full`, `halfvec(768)`; NCBI Gene not loaded, genes not extracted) and lets
the visitor choose what picks each entity's term (docs/ner-site/PLAN.md section 13).
Production variables added or changed: `NER_SITE_NEON_CATALOG=full`,
`NER_SITE_NEON_SCHEMA=xwalk_full`, `NER_SITE_NEON_DIMENSIONS=768`,
`NER_SITE_SELECTOR=jev`, `NER_SITE_MODELS=google/gemini-3.1-flash-lite,openai/gpt-6-luna,anthropic/claude-haiku-5.5`.
Deployed with the Vercel CLI from `git archive` of `ner-site/`. Checked at
https://www.xwalk.si:

- `/api/health` 200: `backend: neon`, `neon: {schema: xwalk_full, catalog: full}`,
  ontologies ChEBI 219,011, CTD diseases 13,323, FoodOn 28,684 terms; BM25, dense and llm
  ok; selectors `retrieval`, `jev`, `openai/gpt-6-luna`, `anthropic/claude-haiku-5.5`,
  `google/gemini-3.1-flash-lite`, default `jev`; the default prompt asks for no genes.
- `POST /api/ner` 401 without the passcode and with a wrong one.
- With it, "Patients with Wilson disease and type 2 diabetes took aspirin and metformin;
  the diet included whole milk yogurt and an apple." (Gemini 3.1 Flash Lite, Jev): 200 in
  7.3 to 9.6 s round trip (extract 1.9 to 3.2 s, map 2.8 to 5.9 s, select 0.8 to 1.0 s),
  695 + 170 tokens, 18 Jev calls reporting $0.0022; every pick looked right to me (this
  text has no gold labels). Jev picked
  "metformin" (CHEBI:6801) where retrieval ranked "metformin(2+)" first; the whole-milk
  yogurt went to "review" on one run and "match" on the other.
- Selectors compared live on the same text: GPT-6 Luna made the same picks in 6.1 s
  (6 calls; its cost is not reported, so "unknown"), Claude Haiku 5.5 and Gemini 3.1 Flash
  Lite answered too. DeepSeek V4.1 Flash returned non-JSON once and Qwen3.7 Flash had no
  endpoint under the account's data policy, so neither is offered.
- Extraction: GPT-6 Luna took 6 to 9 s and 500 to 840 output tokens on that text,
  Gemini 3.1 Flash Lite 1.8 to 3.2 s and about 170, so Gemini is the default.
- The page (headless Chromium, 1280 px): the selector dropdown, Jev's pick with its
  properties above each table, the picked row ticked, the usage line with the selector
  calls and cost.

### Real-world texts, 4 November 2026 (evening)

Visitors hit "not a JSON object" with GPT-6 Luna and Claude Haiku 5.5 on paper
abstracts. Reproduced with 8 PubMed abstracts and 4 Wikipedia food summaries (one in
Slovenian) against production, then fixed and rerun locally with the production
configuration; details and numbers in
`benchmarks/results/site-realworld-2026-11-04/FINDINGS.md`. In short: reasoning models
spent the whole output budget reasoning (now switched off on OpenRouter), Claude Haiku
5.5 is unreliable for extraction (now a selector only), genes were still extracted in
full mode (no longer), an unsure Jev showed as "failed" (now "unsure, needs review"),
mapping made one embeddings request per entity (now one batch: map 0.5 to 8.5 s instead
of up to 76 s), and `/api/health` now reports a key OpenRouter refuses. Production
`NER_SITE_MODELS` is `google/gemini-3.1-flash-lite,openai/gpt-6-luna`.

### Claude by default and open access, 18 November 2026

**Nothing was deployed and nothing was run against the live Anthropic API, Neon or
https://www.xwalk.si.** Production still runs the deployment of 4 November with its
passcode, on OpenRouter models and Jev. This section records code and offline tests
only; the live checks are the operator's, in "Rollout: Claude by default and open
access" above.

What is implemented (branch `feat/claude-provider-open-access`; how it behaves is in
"Claude, the fallback chain and the visitor allowance" above, the decisions in
docs/ner-site/PLAN.md section 14):

- `ner_site/claude.py`: Claude Opus 5.5 and Sonnet 5.5 on the operator's own Anthropic
  key through the `anthropic` SDK (`anthropic>=2.10,<3` in `requirements.txt`).
- `ner_site/fallback.py`: the chain from Opus to Sonnet to an OpenRouter model, with
  every step-down recorded.
- `ner_site/limits.py`: the per-visitor hourly allowance in Neon.
- `ner_site/service.py`, `http.py`, `schema.py`, `select.py`: the new settings and
  defaults, admission before each run, `requested_model`, `requested_selector`,
  `fallbacks` and `notices` in the response, 429 `visitor_limit_reached`, `claude` and
  `visitor` in `/api/health`.
- `public/`: model names, the model that ran, an "About this run" block for notices,
  the allowance line under the model field, the footer sentence about the hashed
  address.
- `tools/claude_check.py` and the `sonnet`, `blocked` and `fallback` scenarios of
  `tools/serve_local.py`.

What was run, and nothing more (Python 3.13, the repository venv, no key in the
environment):

- `python -m pytest -q -m 'not integration' tests/test_site_*.py`: 490 passed, 7
  deselected. Offline only: a fake SDK client for the adapter, scripted clients for
  the chain, a fake SQL executor for the limiter, `httpx.MockTransport` for OpenRouter,
  a local `http.server` stub for `claude_check.py`. They cover the request the adapter
  sends (no sampling parameters, effort, token limits, the schema format and its
  normalisation, the fallback parameter), every row of its error table and that no
  message carries the key; each step-down trigger, the single transient retry,
  stickiness under concurrent calls, the usage of failed attempts and the 300 s block;
  the allowance at its boundaries (10th and 11th, 30th and 31st visit), the hour
  rolling over, the lock statement ahead of every visit, the unreachable and off
  states, and that no raw address reaches a SQL parameter; through `handle`, Opus by
  default, Sonnet after the allowance with its notice, 429 with `Retry-After`, health
  `claude` and `visitor`, no Claude without the key, and a deployment with only the
  OpenRouter key and a passcode running without limiter or chain; and that the keys,
  the database URL, the salt, the client address and the visitor id appear in no body,
  header or log line.
- `ruff check` and `ruff format --check` on `src tests scripts ner-site`: passing.
- `tools/claude_check.py` against `tools/serve_local.py --canned` on 127.0.0.1 (the
  `fallback` scenario, one text and a burst of 3): it printed the canned health blocks,
  the step-downs and the totals. Canned answers, so this shows the tool's output and
  nothing about the service.
- By the implementing sessions, on a throwaway local Postgres 16 (not Neon): the six
  `-m integration` tests of `tests/test_site_limits.py` passed, among them 40
  overlapping requests of one visitor giving exactly 10 Opus, 20 Sonnet and 10 refused.
  They were skipped in the final run (no `NER_SITE_TEST_DATABASE_URL`). The page was
  rendered in headless Chromium against the canned scenarios at 1280 and 390 px.

Not run, and therefore not known:

- Whether the Anthropic API accepts `fallbacks: "default"` (beta
  `server-side-fallback-2026-07-01`) together with `output_config.format`, and whether
  it accepts the normalised schemas (rollout step 4).
- How long a 25-entity request takes with Opus extracting and picking, and what a
  request costs in tokens (step 5). Whether the reported tokens include an attempt
  that the server-side refusal fallback replaced.
- Whether Neon's batch endpoint runs the limiter's transaction at READ COMMITTED, on
  which the advisory lock's guarantee depends (step 6).
- The Anthropic Console settings: workspace, key, auto-reload (step 1).
- Any Vercel build or deployment with `anthropic` installed, Python 3.12 or 3.10, and
  the cold-start time with the SDK imported.

Known and accepted: while the limiter's database cannot be asked, requests run on
Sonnet uncounted (step 7).

### For the operator after deploying

1. `GET /api/health`: `mode: live`, `llm: {"status": "ok"}`, `dense: ok`. Without
   `OPENROUTER_API_KEY` health still answers 200 with `llm` unavailable and says why;
   `POST /api/ner` then answers 503 `not_configured`, unless `ANTHROPIC_API_KEY` is set,
   in which case only the Claude models are offered. With the Anthropic key also
   `claude: {"status": "ok"}` and `visitor.limits: "on"`.
2. That Vercel passes `self.path` as `/api/ner` and `/api/health` (with any query
   string): anything else gets a 404 `not_found` from `handle`. Also that a POST from the
   page carries `Content-Length` (fetch does; a missing one gets 411 `length_required`).
3. The example run on the deployed page: entities, highlighted spans, matches naming
   both `bm25` and `dense-qwen3`, and `usage` with real token counts.
4. The function log: one JSON line per request (`"event": "ner_site.request"`) with
   sizes, counts, timings, status and model, and no text, prompt or key. Where Claude
   is offered it also has `tier`, `fallbacks` and `ran` (the model that extracted), and
   `limits` when the limiter could not be asked; never the visitor id or the address.
   Also note the cold-start time (the first request builds four BM25 indexes in
   `/tmp`).
5. With `NER_SITE_PASSCODE` set: 401 `passcode_required` without it, 401
   `passcode_invalid` with a wrong one. Use an ASCII passcode (browsers reject other
   characters in a header) of 8 characters or more.
6. Query embeddings always go to OpenRouter's default embeddings URL with
   `OPENROUTER_API_KEY`; `NER_SITE_BASE_URL` changes only the chat endpoint.
