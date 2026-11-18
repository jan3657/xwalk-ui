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
  ran, the step-downs, timings and tokens per request (for the rollouts below; every run
  makes real requests, on the site's keys or, with `--provider` and `--key-env`, on an
  API key of your own, which is then billed).
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
# ... or one of the other states: sonnet, blocked, day, unavailable, own-refused,
# own-bad, own-off, fallback (see below)
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
| (none) | Claude Opus 5.5 extracted and picked; the visitor is on the Opus allowance (`visitor.tier: opus`). No notices. Own keys are offered for all three providers, so the page shows "Run with". |
| `sonnet` | The hour's Opus allowance is used: health says `visitor.tier: sonnet`, and the response carries the `admit` step-down with its notice. |
| `blocked` | The hour's requests are used: health says `visitor.tier: blocked`, and every POST on the site's keys answers 429 `visitor_limit_reached` with `Retry-After` (`error-blocked.json`). |
| `day` | The day's requests are used: health says `visitor.tier: blocked` with `window: day`, and every POST on the site's keys answers 429 `visitor_limit_reached` with the day wording (`error-day.json`). An own-key POST still runs. |
| `unavailable` | The limiter cannot be asked: health says `visitor.limits: unavailable` with `tier: blocked`, and every POST on the site's keys answers 503 `limits_unavailable` (`error-unavailable.json`). An own-key POST still runs. |
| `own-refused` | The provider refuses the visitor's key: every own-key POST answers 401 `provider_key_refused` (`error-own-refused.json`, which names Anthropic whichever provider was chosen). POSTs on the site's keys run. |
| `own-bad` | The API refuses the credential: every own-key POST answers 400 `bad_credential` (`error-own-bad.json`). |
| `own-off` | Own keys are turned off (`own_keys.enabled: false`): the page shows no "Run with". |
| `fallback` | Opus stepped down while extracting (`rate_limited`) and Sonnet while selecting (`overloaded`): two notices, picks made by two models, one failed pick. |
| `degraded` | Offline mode with dense retrieval unavailable, in the response shape of the API before the Claude change (no `claude`, `visitor`, `fallbacks` or `notices`), so it also checks the page against an older API. |

The server binds `127.0.0.1`, port 8765 by default (`--port`), and adds the headers
from `vercel.json` to every response, so the page runs under the production
Content-Security-Policy. With `--canned`, posting text that contains `ERROR` returns
`fixtures/error.json` with HTTP 400 to show the error box. A scenario with an
`error-NAME.json` of its own answers POSTs with it, with the HTTP status of its code: an
error about the visitor's own key answers the own-key POSTs only, any other error the
POSTs on the site's keys only. A POST that names an own-key `provider` is answered with
`ner-own.json`, and is refused with 400 `bad_credential` when it lacks a well-formed
`Authorization: Bearer` header (as is a POST on the site's keys that carries one). The
canned server logs each API request with `authorization=present` or `absent`, never the
value, which is how to see that only `POST /api/ner` carries a key. `--delay SECONDS`
slows the POST to show the busy state.

The private stats page (`http://127.0.0.1:8765/stats`, see "Analytics" below) is canned
too. `GET /api/stats` wants the header `X-Stats-Passcode: local-stats-passcode` (not a
secret: it opens fixture data only, and it is shorter than a real deployment accepts)
and answers 401 without it. With it, the answer is a populated one for 7, 30 and 90
days (`stats-7.json`, `stats.json`, `stats-90.json`). Three scenarios change that:
`empty` (stats on, no run recorded yet), `nostats` (stats off: 404, before any passcode
check) and `unavailable` (the database is down: 503). Each canned stats request is
logged with `stats_passcode=present` or `absent`, never the value. Without `--canned`
the real handler answers `/api/stats`, query string included, so
`NER_SITE_STATS_PASSCODE=... DATABASE_URL=... python ner-site/tools/serve_local.py`
shows the page against a real Neon database (the executor speaks Neon's HTTPS
endpoint only, not a local Postgres).

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
   access" below before adding it to a public deployment. Visitors' own API keys are
   off unless `NER_SITE_OWN_PROVIDERS` lists providers: leave that variable unset until
   "Rollout: own keys and the daily cap" below is done.
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
Several things could not be checked offline. They are marked **Check** below, each
with what to look for, and all of them come before the passcode is removed (step 8).
Until then the passcode is what stands between the public and the key.

Two things in this section changed on the same day, with the daily cap and visitors'
own keys ("Rollout: own keys and the daily cap" below, to be done together with this
one): the allowance now also has a daily cap (20 requests starting on Opus and 40 on
Sonnet per UTC day), and a limiter that cannot be asked no longer lets a request run on
Sonnet uncounted: requests on the site's keys are refused with 503 (step 7). The steps
below describe the code as it is now.

`tools/claude_check.py` is the measuring tool. It GETs `/api/health` and POSTs
`/api/ner` on the URL it is given and nothing else; every text is a real request that
spends the balance of the key behind the site and counts against your own hourly and
daily allowance there (`--help` says the same). The 12 real-world texts of 4 November are not
in this repository, so save the texts you want to measure as UTF-8 files first.

1. **Anthropic Console.** Create a workspace for the site and a key in it.
   **Check (auto-reload):** auto-reload must be off. The site has no dollar budget of
   its own: the prepaid balance is the only ceiling on what open access can spend, and
   auto-reload removes it. While there, note the workspace's rate limits for Opus 5.5
   and Sonnet 5.5 (requests, input tokens and output tokens per minute); step 5
   compares them with what one request sends.
2. **Vercel, production variables.** Add `ANTHROPIC_API_KEY` (sensitive). Remove
   `NER_SITE_MODELS`, `NER_SITE_SELECTORS` and `NER_SITE_SELECTOR`, or set them to the
   new defaults: production has `NER_SITE_MODELS=google/gemini-3.1-flash-lite,openai/gpt-6-luna`
   and `NER_SITE_SELECTOR=jev` from 4 November, and with those Claude is not offered
   for extraction and Jev stays the default selector. Keep `NER_SITE_PASSCODE` for now.
   Keep `OPENROUTER_API_KEY` (Jev, query embeddings, the last step of the chain) and
   `DATABASE_URL` (the allowance is counted there). Optionally set
   `NER_SITE_VISITOR_SALT`. Leave `NER_SITE_OWN_PROVIDERS` unset: visitors' own keys
   stay off until their own rollout below is done.
3. **Deploy, then `GET /api/health`.** Look for `"claude": {"status": "ok"}`,
   `"visitor": {"limits": "on", "tier": "opus", "opus_left": 10, "sonnet_left": 20,
   "opus_left_today": 20, "sonnet_left_today": 40, "window": "hour", ...}`,
   `"own_keys": {"enabled": false, "providers": []}`,
   `default_model` and `default_selector` both `claude-opus-5-5`, and
   `passcode_required: true`. This first call also creates the table
   `<NER_SITE_NEON_SCHEMA>.site_visits`. `visitor.limits: "unavailable"` (with
   `tier: "blocked"`) means the limiter could not query Neon, and every request on the
   site's keys is then refused with 503: do not go on (see step 7). `claude` may also be
   `{"status": "ok", "reason": "key not checked (...)"}`: Anthropic did not answer the
   key check, which says nothing about the key; ask again after a minute and go on
   only once it is `ok` without a reason.
   From here on the allowance is counted even with the passcode set, your own requests
   included: 10 an hour start on Opus, the next 20 on Sonnet, and of a UTC day's
   requests 20 start on Opus and 40 on Sonnet. A request is counted when
   it is admitted, before any model is called, so one that then fails with 502 or 422
   has already been counted against the allowance; one refused earlier (400, 401, 413,
   or the 429 `visitor_limit_reached` of the allowance itself) has not.
   **Check (a cold database answers in time).** The limiter gives Neon 3 s for its one
   round trip. Leave the site alone until the Neon compute has suspended (the Neon
   console shows it as idle; five minutes by default), then `GET /api/health` once:
   `visitor.limits` must be `on`. If it is `unavailable` on that first call and `on`
   on the next, waking the database takes longer than 3 s, and after every quiet
   spell the first requests on the site's keys would be refused with 503
   `limits_unavailable` (step 7): a visitor's first run after a quiet spell would
   fail. Keep the compute from suspending, or accept that knowingly, before going on.
   Try it a few times.
   With `NER_SITE_BACKEND=neon`, a function instance that is itself cold opens the
   indexes first, which wakes the database and hides the wait, so an `on` from a slow
   first answer proves nothing.
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
   `by claude-opus-5-5`, `fallbacks none`.
   **Check (the model the API names).** The id after the arrow is the `model` the API
   echoes. Since the change of 18 November 2026 the site no longer reads a refusal
   fallback from that name: it records one only when the SDK signals it (a
   `fallback_message` entry in `usage.iterations`). So an echo under another spelling,
   say `claude-opus-5-5 -> claude-opus-5-5-<date>` with `fallbacks none`, is harmless
   to the notices, but the page then shows that id instead of "Claude Opus 5.5", with
   "(you asked for Claude Opus 5.5)". Note what the API echoes. A harmless text that
   shows `extract:claude-opus-5-5>...:refused` does mean the API served it from its
   fallback model.
   Rejected looks like `502 FAILED provider_error` naming an Anthropic HTTP 400 (the
   extraction request was refused), or a 200 whose picks are all `failed` with a
   `failed picks:` reason naming an Anthropic HTTP 400 (the selection schema was
   refused). The chain does not step down
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
   8 picks at a time). One Claude extraction call has 40 s: if Opus has not answered
   by then the request moves to Sonnet (`extract:claude-opus-5-5>claude-sonnet-5-5:timeout`),
   so a text whose extraction needs longer than that on Opus never runs on Opus. Look
   at the `most entities` and `slowest` lines: `total` in `timing_ms` should be well
   under 100000, `extract` well under 40000, no line should show `failed picks: cut
   off by the request deadline`, and `step-downs by reason` should show no `timeout`.
   **Check (dollars per request).** The `tokens` lines give the input and output
   tokens per request for extraction and for selection; the site computes no dollars,
   so multiply by the prices of the models yourself and compare with the usage the
   Console shows (whether the reported tokens include an attempt the refusal fallback
   replaced was not checked). Do this sum again from the measured tokens before the
   passcode goes, because the allowance sets a rate, it is not a budget: one address
   may make 10 requests an hour starting on Opus and 20 on Sonnet, and with Claude
   picking every entity (up to 25 calls a request) that can be on the order of ten
   dollars an hour from a single address, against a balance of about $180. The daily
   cap bounds the same address at 20 requests starting on Opus and 40 on Sonnet per
   UTC day, so at two such hours a day; it still bounds one address, not the total.
   If the sum says so, set `NER_SITE_SELECTOR=jev` (Opus then extracts and Jev picks)
   or lower `NER_SITE_OPUS_PER_HOUR`, `NER_SITE_SONNET_PER_HOUR` and the two
   `..._PER_DAY` variables for the open site. The same applies when selection on Opus
   is too slow.
   **Check (the workspace's rate limits).** One 25-entity request sends its picks 8
   at a time, so its selection tokens arrive within a few seconds. Compare the tokens
   of the largest request with the per-minute limits noted in step 1: if one request,
   or two visitors at once, can exceed them, requests will step down to Sonnet with
   `rate_limited` as a matter of course (`step-downs by reason` shows it).
   More than 10 texts in one clock hour start on Sonnet from the 11th (the line shows
   `admit:claude-opus-5-5>claude-sonnet-5-5:opus_allowance_used`). To measure them all
   on Opus, run 10 an hour (`visitor.resets_in_s` in the health block says when the
   hour ends) or raise `NER_SITE_OPUS_PER_HOUR` while measuring and set it back. The
   day allows 20 starts on Opus: the 21st of a UTC day starts on Sonnet in the same
   way whatever the hour (`visitor.opus_left_today`, `visitor.day_resets_in_s`), so
   measuring more than 20 texts on Opus in one day needs `NER_SITE_OPUS_PER_DAY`
   raised as well.
6. **Check (the allowance holds under parallel requests).** One visitor's requests are
   serialised with a Postgres advisory lock, and the guarantee depends on Neon's batch
   endpoint running a transaction at READ COMMITTED, which was verified on a local
   Postgres only. Fire a burst from one address, with a one-sentence text to keep it
   cheap:

   ```bash
   python ner-site/tools/claude_check.py --url https://www.xwalk.si \
       --passcode-env NER_SITE_PASSCODE --selector retrieval --burst 15 short.txt
   ```

   The tool checks the limiter from both sides, with the `opus_left` and `sonnet_left`
   health reported just before and just after (each is what can still start this hour,
   already capped by what is left of the day): no more requests may have started on a
   tier than it had left, and each balance must have fallen by the starts on its tier.
   So run the burst alone: nothing else from that address in the same minute, and not
   across the turn of a clock hour. Good: `within the allowance`, then `and counted:
   opus_left 10 -> 0, sonnet_left 20 -> 15`, and exit code 0, for example 10 on Opus
   and 5 on Sonnet in a fresh hour of a fresh day. A request that waits more than
   1.5 s for its turn at the visitor's lock is refused with 429 "too many requests at
   once; try again in a few seconds" and is not counted; the tool reports these as
   `429 too many at once`. That is what the tail of a large burst should get, and it
   is not a failure. Bad, each with exit code 1 and without the `within the allowance`
   line; do not open the site:

   - `OVER THE ALLOWANCE`: more starts on a tier than were left.
   - `THE LIMITER IS FAILING`: at least one request of the burst was refused with 503
     `limits_unavailable` (the summary counts them as `503 limits_unavailable`). The
     limiter's database call failed for those requests, and the site runs no request
     on its own keys uncounted, so it refused them. Waiting for the lock does not
     explain them (that is the 429 above), so do not dismiss them: repeat with
     `--burst 3`, and if the 503s persist in a burst that small, the limiter does not
     work under concurrency. The reason is the error class and SQLSTATE in the
     `limits` field of each such request's log line. `40001` means the isolation
     level is wrong: Neon's batch endpoint is not running the transaction at READ
     COMMITTED, and concurrent visits fail instead of taking turns. That needs a code
     change before the site opens. A `limits` field with no SQLSTATE means the call
     did not come back from the database in time (step 7): the database was slow for
     those calls, and the visitors behind them would have seen an error.
   - `NOT COUNTED`: a balance fell by less than the starts on its tier. Requests ran
     that the limiter did not record.
   - `INCONCLUSIVE`: the run could not establish the count, and proves nothing either
     way. Health showed no `visitor` block with `limits: "on"` and whole-number
     balances before the burst (step 3 was not met; requests that ran all the same
     are reported as `not counted`, under no tier) or after it; no request of the
     burst started at all (every one was refused); the hour or the day ended during
     the burst; a balance fell by more than the starts (something else
     used that address's allowance: run it alone); or the requests used no Claude
     model, for which health has no balance. Fix what it names and run it again.
7. **Check (know what a Neon outage does, by design).** When the limiter cannot query
   its database (unreachable, or no answer within 3 s), nothing runs uncounted on the
   site's keys: every request on them, whatever the model, is refused with 503
   `limits_unavailable` ("the site cannot count requests right now; try again
   shortly, or use your own API key", `Retry-After: 30`; the last clause only where
   own keys are on), and nothing is recorded. This replaces the first design of the
   same day, in which such a request ran on Sonnet uncounted. So for as long as Neon
   is down, the site's models are down for everyone, and nothing is spent on them. A
   visitor with an own key can still run; that run is not counted either, because the
   own-key flood limit lives in the same database. It shows as `visitor.limits:
   "unavailable"` with `tier: "blocked"` in `/api/health`, as "The site cannot count
   requests right now. ..." under the model field on the page, and as a `limits`
   field (error class and SQLSTATE) in the request log. (The contended lock of step 6,
   SQLSTATE `55P03`, is a 429, not an outage.) The same 503, with the message
   "visitor limits are not configured" and no `Retry-After`, answers every site-key
   request on a deployment that has the Anthropic key, no passcode and no
   `DATABASE_URL`.
   There is no setting that makes the open site run without counting. If the outage
   lasts and the site must answer, the choices are a private site (set
   `NER_SITE_PASSCODE` and remove `DATABASE_URL`: with a passcode and no database the
   site's models run unmetered, and with `NER_SITE_BACKEND=neon` the indexes need
   that URL too, so this also means the bundled backend: remove `NER_SITE_BACKEND`
   and `NER_SITE_NEON_CATALOG` as well, or set them to `bundled` and `samples`,
   because production's `NER_SITE_NEON_CATALOG=full` is refused without the Neon
   backend, 503 `not_configured` on every request) or the site without Claude
   (remove `ANTHROPIC_API_KEY`: site-key requests are then not counted at all). Never
   the second without a passcode: without the key and without a passcode the site is
   public and uncounted on the OpenRouter key.
8. **Open the site.** Remove `NER_SITE_PASSCODE`, redeploy, and check
   `passcode_required: false`. Then confirm the allowance from one address in one
   clock hour (31 short requests; `--selector retrieval` keeps each to one call, and
   earlier requests of that hour count too): the 11th starts on Sonnet with the notice
   "Your Opus allowance for this hour is used; this ran on Claude Sonnet 5.5.", and
   the 31st answers 429 `visitor_limit_reached` ("you have used this hour's requests;
   try again in N minutes", plus ", or use your own API key" where own keys are on)
   with a `Retry-After` header. These 30 are also half of that address's day; the
   daily boundaries are step 5 of the next rollout. With the tool, in a fresh hour of
   a fresh day (the 429 makes its exit code 1, as it should):

   ```bash
   python ner-site/tools/claude_check.py --url https://www.xwalk.si \
       --selector retrieval --sequential-delay 1 $(printf 'short.txt %.0s' {1..31})
   ```

   Lines 1 to 10 read `model claude-opus-5-5 -> claude-opus-5-5`, lines 11 to 30 carry
   `admit:claude-opus-5-5>claude-sonnet-5-5:opus_allowance_used`, and line 31 reads
   `429 ... FAILED visitor_limit_reached`.
9. Record what was verified in the "Verified" section, as for earlier deployments.

Going back: set `NER_SITE_PASSCODE` again, remove `ANTHROPIC_API_KEY`, and redeploy.
Do both, and never the second without the first: removing the key takes Claude out of
the lists and switches the allowance off with it, so once the passcode is gone,
removing the key alone would leave a public site that counts nothing and spends the
OpenRouter key. The passcode also stands in front of own-key requests; to take own
keys away as well, remove `NER_SITE_OWN_PROVIDERS`.
With the passcode back, the site runs on OpenRouter as before, with Jev as the default
selector unless `NER_SITE_SELECTOR` says otherwise. Check `passcode_required: true` and
`"claude": {"status": "unavailable", "reason": "key not set"}` in `/api/health`.

### Rollout: own keys and the daily cap (operator)

The code for this is on the branch `feat/own-key-daily-cap` and, as of 18 November
2026, nothing below has been done: no deployment, no change to a production variable,
no request with a real key to Anthropic, OpenAI or OpenRouter. OpenAI's own API in
particular has never been called by anything in this repository. What the code does is
in "The daily cap", "When the limiter cannot be asked" and "Own keys" further down; the
checks that could not be made offline are marked **Check** here, each with what to look
for.

Deploying this code changes two things at once, with no variable touched, and both
only where Claude is offered (a deployment with `ANTHROPIC_API_KEY`):

- the allowance gains a daily cap;
- a limiter that cannot be asked refuses requests on the site's keys with 503 instead
  of letting them run on Sonnet uncounted.

Visitors' own keys are a third change, and that one waits for a variable: they are off
until `NER_SITE_OWN_PROVIDERS` lists the providers to allow (step 9). Until then a
deployment without the Anthropic key behaves as it did before this code, apart from
fields that are added: `own_keys` (off) in health, `provider: "site"` and
`credential_source: "site"` in a response, and `provider: "site"` and `credential:
"site"` in the log line. That includes the
production deployment of 4 November (OpenRouter key, passcode, Neon): the same
models, no "Run with" field, `DATABASE_URL` read only for the Neon backend, and an
`Authorization` header on a request (HTTP basic auth, `curl -u`, a proxy that adds
one) not looked at.

`tools/claude_check.py` runs on a key of your own with `--provider anthropic|openai|openrouter
--key-env NAME`: the key is read from the environment variable NAME only, sent as
`Authorization: Bearer` on the `POST /api/ner` requests, and replaced by `[redacted]`
in everything the tool prints. Such a run bills that key for the model calls. The tool
sends the key only after `/api/health` answered 200 with `own_keys.enabled: true` and
the provider listed. It sends neither the key nor the passcode over plain http to
another machine, and asks a URL on this machine directly, never through a proxy the
environment names. Because its report is redacted, it says itself whether a response
contained the key or the passcode: such a response is marked `CREDENTIAL ECHO` and
fails the run, and a run that sent a secret ends with a `credential echo check:` line.
A value escaped more deeply than the tool decodes (eight rounds of percent decoding)
gets no verdict, and no verdict is not a pass: the value is withheld, the run fails,
and the last line says an echo "was detected or could not be ruled out". A key or
passcode that is itself escaped that deeply is refused before anything is sent (exit
2), because the tool could not recognise its echo.

The examples below are written for a preview deployment that has the passcode, which
a preview has unless its environment says otherwise (it inherits `NER_SITE_PASSCODE`
from the shared variables): export the passcode as `NER_SITE_PASSCODE` in your shell
first. On a preview without a passcode leave `--passcode-env NER_SITE_PASSCODE` out.
Without the flag a protected preview answers every POST with 401 `passcode_required`,
and the check has checked nothing.

1. **Deploy with own keys off, and see that nothing changed.** Deploy this code with
   no `NER_SITE_OWN_PROVIDERS` in the Vercel environment (production and preview:
   check that neither has one). After the deployment `GET /api/health` must show
   `"own_keys": {"enabled": false, "providers": []}`, the page must show no "Run
   with" field, and production must behave as before: the same model and selector
   lists, the passcode asked for, a run of the example text answering as it did. A
   POST that names an own provider answers 400 `bad_credential`. **Check** in the
   Vercel environment, before going on:
   - `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_LOG`, `ANTHROPIC_BASE_URL` and the proxy
     variables (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, in either case) are unset.
     Requests on a visitor's own key ignore all of them. The site's own Claude client
     does not: it would send the operator's Claude requests with those headers, to
     that base URL or through that proxy, and with `ANTHROPIC_LOG` the SDK writes its
     requests to the function log.
   - Nothing in front of production adds an `Authorization` header to requests (an
     authenticating proxy, basic auth on the domain). While own keys are off the
     site does not look at that header; once they are on, a request on the site's
     keys that carries one is refused with 400 `bad_credential`, for every visitor
     behind that proxy.
2. **Check (Vercel records no request headers).** A visitor's key arrives in the
   `Authorization` header of `POST /api/ner`. The site's own log line never holds it,
   and the SDK and HTTP client loggers are silenced for an own-key run, but what the
   platform records is outside the code. In the Vercel project, look at everything
   that can record a request: the runtime logs of the functions, every log drain,
   every observability, tracing or error-reporting integration, and the firewall's
   logs. None of them may store request headers, or `Authorization` must be excluded
   by name. This check is made with step 3: do not turn own keys on for the public
   before both are done.
3. **Check (a wrong key is refused and appears nowhere).** On a preview deployment
   with own keys on for it (`NER_SITE_OWN_PROVIDERS=anthropic,openai,openrouter` in
   the preview environment only), send a made-up key that you can search for:

   ```bash
   export CHECK_KEY=sk-rollout-sentinel-4f9a71c2d08b4e5a   # made up, not a real key
   python ner-site/tools/claude_check.py --url https://<preview> \
       --passcode-env NER_SITE_PASSCODE --provider openai --key-env CHECK_KEY one.txt
   ```

   The made-up key travels to the provider, which refuses it. Expected line:
   `401 ... FAILED provider_key_refused: OpenAI refused this key. It may be mistyped,
   revoked, out of credit, or not allowed to use this model. (sent as provider
   openai)`, exit code 1 (the POST failed, as it should), and as the last line
   `credential echo check: none of the 3 responses contained the API key or the
   passcode this run sent`. A line with `CREDENTIAL ECHO`, or `credential echo check:
   FAILED`, means a response carried the key or the passcode back: the tool prints
   every copy as `[redacted]`, so only those lines show it. Stop, and do not turn
   own keys on. Then search the function log and every sink of step 2 for
   the whole key, for its middle part (`sentinel-4f9a71c2`; providers echo keys
   masked, as in `sk-ro...4e5a`) and for the word `Authorization`: nothing. The
   request's log line shows `"provider": "openai", "credential": "visitor"`,
   `"code": "provider_key_refused"` and no part of the key. Repeat for `anthropic`
   and `openrouter`. A key of the wrong shape (shorter than 20 characters, a space)
   never reaches the provider: the tool refuses it, and the API answers 400
   `bad_credential`.
4. **Check (one real run per provider).** With a real key for each provider (a
   restricted one with a spending limit), one short text each, first with retrieval
   picking and then with the model picking:

   ```bash
   read -rs MY_OPENAI_KEY && export MY_OPENAI_KEY          # typed, not in history
   python ner-site/tools/claude_check.py --url https://<preview> \
       --passcode-env NER_SITE_PASSCODE --provider openai --key-env MY_OPENAI_KEY one.txt
   python ner-site/tools/claude_check.py --url https://<preview> \
       --passcode-env NER_SITE_PASSCODE --provider openai --key-env MY_OPENAI_KEY \
       --selector gpt-6-luna one.txt
   ```

   Good: `200`, `provider openai (credential visitor)`, `model gpt-6-luna ->
   gpt-6-luna`, `entities` above 0, `fallbacks none`, and in the second run picks
   `by gpt-6-luna` with none `failed`. On the page the same run reads "Extracted by
   gpt-6-luna on your OpenAI key." (the page has names for the Claude models only,
   for example "Extracted by Claude Haiku 5.5 on your Anthropic key."). In your own
   dashboards for that minute: no usage on the Anthropic key, and on
   the OpenRouter key the query embeddings only, no chat call. The provider's
   dashboard for your key shows the calls.
   **OpenAI direct needs the closest look**, because nothing has called it:
   - *The default model id exists.* `gpt-6-luna` is the id OpenRouter uses; whether
     OpenAI's API has it under that name is not known. A model the key cannot use, or
     one that does not exist, comes back from OpenAI as HTTP 404, which the site
     reports as the key being refused: so `401 provider_key_refused` with a key you
     know to be good means the model id. Set `NER_SITE_OWN_MODELS_OPENAI` to ids that
     exist.
   - *The request is accepted.* The site sends `max_completion_tokens` (16000 for
     extraction, 8000 for selection) and no `temperature`, with the JSON schema in
     strict mode. `502 provider_error ... (bad_request)` means OpenAI rejected the
     request's shape; picks that are all `failed` with `bad_request` mean it rejected
     the selection request. Either needs a code change in `ner_site/providers.py`.
     A rejected schema can also pass unseen: when OpenAI's error names
     `response_format`, the engine's client asks once more with the JSON described
     in the prompt instead, so also look at the key's usage for failed calls.
   - *JSON comes back inside the token limit.* `422 extraction_unparseable` or
     `extraction_schema`, or picks `failed` with `bad_answer`, mean the answer was
     cut off or was not the JSON asked for.
   - *Latency.* One request to the provider has 60 s, extraction as a whole 90 s
     (up to three requests on OpenAI and OpenRouter, one on Anthropic). `timing_ms`
     `extract` should be far below 60000; `502 provider_error ... (timeout)` means it
     was not.

   For Anthropic also run `--model claude-haiku-5-5`, which only own keys offer and
   which has not met the live API through this adapter. An own Anthropic key gets one
   request per call and no retry, so a single 5xx from Anthropic is a `502
   provider_error ... (http_5xx)`: seen once, run it again before reading anything
   into it.
5. **Check (the daily cap at its boundaries).** Cheapest on a preview deployment with
   the day lowered, `NER_SITE_OPUS_PER_DAY=3` and `NER_SITE_SONNET_PER_DAY=3`, from an
   address with no requests that day:

   ```bash
   python ner-site/tools/claude_check.py --url https://<preview> \
       --passcode-env NER_SITE_PASSCODE --selector retrieval --sequential-delay 1 \
       $(printf 'short.txt %.0s' {1..7})
   ```

   Lines 1 to 3 read `model claude-opus-5-5 -> claude-opus-5-5`; lines 4 to 6 carry
   `admit:claude-opus-5-5>claude-sonnet-5-5:opus_allowance_used`, and the response's
   notice is "Your Opus allowance for today is used; this ran on Claude Sonnet 5.5.";
   line 7 reads `429 ... FAILED visitor_limit_reached: you have used today's
   requests; they reset at midnight UTC` (plus ", or use your own API key" where own
   keys are on), with `Retry-After` the seconds to midnight UTC. Health then shows
   `"tier": "blocked", "window": "day"`, and the page "You have used today's
   requests. They reset at midnight UTC...".
   With the defaults the same shows over two clock hours from one address: 31
   requests in the first (step 8 of the rollout above), 31 in the second. In the
   second hour requests 1 to 10 start on Opus, 11 to 30 on Sonnet with the "for
   today" notice (the day's 20 Opus starts are used), and the 31st is refused with
   the day wording; every request after it, in any hour, is refused the same way
   until midnight UTC.
6. **Check (a failing limiter fails closed, and own keys still run).** On a preview
   deployment that has the Anthropic key and own keys on, set `DATABASE_URL` to a
   Postgres URL that cannot work (the right host with a wrong password). Use the
   bundled backend for this: with `NER_SITE_BACKEND=neon` the indexes read the same
   URL and retrieval fails too. A preview that starts from production's variables
   has `NER_SITE_BACKEND=neon` and `NER_SITE_NEON_CATALOG=full`, so set both for the
   preview, to `bundled` and `samples` (or remove both there): changing the backend
   alone leaves `full`, which is refused without the Neon backend, and every request,
   health included, then answers 503 `not_configured` instead of what this step is
   about. `/api/health` shows `"visitor": {"limits": "unavailable", "tier":
   "blocked", ...}` and the page says "The site cannot count requests right now. Try
   again shortly, or use your own API key." A request on the site's keys (with
   `--passcode-env NER_SITE_PASSCODE` on a preview that has the passcode) answers
   `503 ... FAILED limits_unavailable` with `Retry-After: 30`; `--burst 3` prints
   `THE LIMITER IS FAILING` and exits 1. A run with `--provider ... --key-env ...`
   still answers 200: it is not counted, and its log line has `"tier": "own"` and a
   `limits` field. Afterwards put the preview back: restore the working
   `DATABASE_URL`, `NER_SITE_BACKEND` and `NER_SITE_NEON_CATALOG` (or delete the
   preview's overrides), redeploy, and see `"visitor": {"limits": "on", ...}` and
   the catalog you expect in `/api/health` before using that preview for anything
   else.
7. **Check (the browser's password manager).** In Chrome and in Safari, choose "My
   ... key" on the page, type a key and run. The field is a password field without a
   name and with `autocomplete="off"`, in the same form as the passcode field, and
   whether a browser then offers to save it was not tested. If one does, a saved key
   outlives the tab, which the sentence under the field ("Your key stays in this
   browser tab ...") denies: say so on the page (a change in `public/`), or accept it
   knowingly. Also press "Forget key", change "Run with", and reload: the field must
   be empty after the first two and hold the key after a reload in the same tab.
8. **`ANTHROPIC_CUSTOM_HEADERS` does not reach own-key requests, and the SDK is
   pinned.** The site's own Claude client honours that variable as before (which is
   why step 1 wants it unset). A client built for a visitor's Anthropic key drops it
   (a header there could replace the visitor's key with the operator's), so nothing
   that depends on such a header, a gateway or an attribution header, applies to
   own-key requests: do not rely on it. The same holds for `ANTHROPIC_BASE_URL`,
   `NER_SITE_BASE_URL` and proxy variables: own-key requests go to the provider's
   own host. The drop goes through a private attribute of the `anthropic` SDK, so it
   is not what the site relies on: every own-key Anthropic request is looked at as
   it leaves, and one that is not for `api.anthropic.com` with the visitor's key
   once and exactly the headers and values SDK 2.10.0 sends on its own is not sent
   (`502 provider_error ... (error)` for the visitor, `"own_error": "WireRefused"`
   in the log line). That check, and the reading of what the SDK takes from the
   environment, are of one version, so `anthropic` is pinned to `2.10.0` in
   `requirements.txt` and in the `dev` extra of `pyproject.toml`, and a test fails
   on any other version. An upgrade does not announce itself otherwise: a new
   version whose requests look the same passes the check, and whatever else it
   reads from the environment has not been looked at. So on any bump, before
   deploying it: go through the list in that test's message
   (`tests/test_site_providers.py`), move both pins and `AUDITED_SDK` together, and
   repeat step 4 for Anthropic.
9. **Turn own keys on.** Set `NER_SITE_OWN_PROVIDERS` to the providers that passed
   step 4, by name, for example `anthropic,openrouter` or
   `anthropic,openai,openrouter` (there is no word for all of them; case and spaces
   do not matter), and redeploy. A name that is not one of the three makes every
   request answer 503 `not_configured`, `/api/health` included. With own keys on,
   `DATABASE_URL` is read on that deployment (the flood limit is counted there) and
   must be a Postgres URL, and a request on the site's keys may no longer carry an
   `Authorization` header (step 1). `/api/health` shows `own_keys.enabled: true`
   with each provider's models, and the page shows "Run with". The flood limit can
   be confirmed at your own key's expense: the 31st own-key request of an hour
   answers 429 "too many requests with your own key; try again in N minutes". A
   burst of the tool with `--provider` checks it too where health has a `visitor`
   block (a deployment with the Anthropic key); elsewhere it reports `INCONCLUSIVE`,
   because nothing in health shows the own-key count.
10. **Sign-in with ChatGPT is not built.** OpenAI's programme covers loopback
    callbacks only and a hosted app has to apply: submit OpenAI's interest form if
    sign-in on the hosted site is wanted. Nothing changes on the site until then;
    the code has a place for it (a provider whose credential returns a refreshed
    token) and nothing more.
11. Record what was verified in the "Verified" section.

Turning own keys off again: remove `NER_SITE_OWN_PROVIDERS` and redeploy (to take
one provider away, take its name out of the list). Health reports
`own_keys.enabled: false`, and a POST with an own provider answers 400
`bad_credential`. A page that is open learns it at its next health read (at load, and
after every run): it drops the provider from "Run with", forgets a key kept in the
tab, says that the key can no longer be used, and runs nothing on the site's models
until the visitor chooses that.

### Environment variables (function runtime)

| Name | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | The operator's Anthropic key. With it the Claude models are offered and the visitor allowance applies (below). Mark it sensitive; it is scrubbed from every response and log line like the other secrets. A request on a visitor's own key never uses it. |
| `OPENROUTER_API_KEY` | unset | The operator's OpenRouter key, used for the OpenRouter models, Jev, query embeddings and the last step of the fallback chain. With neither key (and not offline) a request on the site's keys answers 503 `not_configured`; a request on a visitor's own key still runs, without dense retrieval. |
| `NER_SITE_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL for the **chat call only**. Query embeddings always go to OpenRouter (`https://openrouter.ai/api/v1/embeddings`) with `OPENROUTER_API_KEY`, whatever this says: the committed dense indexes record that embeddings URL in their identity, so embedding elsewhere would need a code change and rebuilt indexes. |
| `NER_SITE_MODELS` | `claude-opus-5-5,claude-sonnet-5-5,google/gemini-3.1-flash-lite,openai/gpt-6-luna` | Comma-separated model allowlist. An id starting with `claude-` is called on the Anthropic key, anything else through OpenRouter; only ids whose key is set are offered, and the first offered one is the default. The only Claude ids are `claude-opus-5-5` and `claude-sonnet-5-5`; another `claude-` id is refused (503 `not_configured`), as is a list that no set key can run. |
| `NER_SITE_SELECTORS` | `claude-opus-5-5,claude-sonnet-5-5,jev,google/gemini-3.1-flash-lite,openai/gpt-6-luna` | What may pick the match, after `retrieval`, which is always offered first. The same rule: only ids whose key is set are offered (Jev needs the OpenRouter key). |
| `NER_SITE_SELECTOR` | unset | The default selector. It must be `retrieval` or in the list above; unset, or when its key is not set, the default is the first selector that can run: `claude-opus-5-5` with the Anthropic key, `jev` with only the OpenRouter key. |
| `NER_SITE_CLAUDE_EFFORT` | `low` | `output_config.effort` of every Claude request: `low`, `medium` or `high`. |
| `NER_SITE_FALLBACK_MODEL` | `google/gemini-3.1-flash-lite` | The OpenRouter model at the end of the fallback chain, for extraction and for selection. Must be an OpenRouter model id (not a Claude id, `jev` or `retrieval`). Used only when `OPENROUTER_API_KEY` is set; without it the chain ends at Sonnet. |
| `NER_SITE_OPUS_PER_HOUR` / `NER_SITE_SONNET_PER_HOUR` | 10 / 20 | The per-visitor allowance for a clock hour (UTC): requests that start on Opus, then requests that start on Sonnet. Their sum is the hour's total for every kind of request on the site's keys. Whole numbers 0 to 1000; anything else is refused. |
| `NER_SITE_OPUS_PER_DAY` / `NER_SITE_SONNET_PER_DAY` | 20 / 40 | The same for a UTC calendar day, on top of the hourly allowance: their sum is the day's total for every kind of request on the site's keys. Whole numbers 0 to 10000; anything else is refused. |
| `NER_SITE_OWN_PER_HOUR` / `NER_SITE_OWN_PER_DAY` | 30 / 60 | The flood limit for requests on a visitor's own key, per visitor and clock hour and per UTC day. Counted apart from the allowance above. Whole numbers 0 to 10000; anything else is refused. 0 refuses every own-key request where the limiter can be asked. |
| `NER_SITE_OWN_PROVIDERS` | unset (own keys off) | The providers a visitor may bring an own key for: a comma-separated list of `anthropic`, `openai` and `openrouter` (case and spaces are ignored, as are empty items such as a trailing comma). Unset or empty, own keys are off; an unknown name is refused (503 `not_configured`). Offline mode turns them off whatever this says. |
| `NER_SITE_OWN_MODELS_ANTHROPIC` | `claude-opus-5-5,claude-sonnet-5-5,claude-haiku-5-5` | The models an own Anthropic key may run; the first is the default. Only these three ids are accepted. Read only while `anthropic` is in `NER_SITE_OWN_PROVIDERS`; an empty value counts as unset. |
| `NER_SITE_OWN_MODELS_OPENAI` | `gpt-6-luna` | The models an own OpenAI key may run on `https://api.openai.com/v1`; the first is the default. Any ids except `jev` and `retrieval`; an empty item in the list (a trailing comma) is refused. Read only while `openai` is listed; an empty value counts as unset. Not checked against the live API. |
| `NER_SITE_OWN_MODELS_OPENROUTER` | `google/gemini-3.1-flash-lite,openai/gpt-6-luna` | The models an own OpenRouter key may run; the first is the default. The same rules. |
| `NER_SITE_VISITOR_SALT` | derived | Salt for the visitor ids (a keyed hash of the client address). Unset, it is the SHA-256 of `DATABASE_URL`. Changing it, or the database URL when it is unset, gives every visitor a new id, which resets the current hour's and day's counts. |
| `NER_SITE_PASSCODE` | unset | When set, every `/api/ner` request must send it as `X-Passcode`; the page asks for it and keeps it in `sessionStorage`. ASCII only (8 characters or more recommended): header values arrive as Latin-1, so a non-ASCII passcode could never match and is refused with 503 `not_configured`. It does not exempt anyone from the visitor allowance, and a request on a visitor's own key needs it too. |
| `NER_SITE_STATS_PASSCODE` | unset (analytics off) | Turns on the record of runs and the private stats page ("Analytics" below): with it, every run writes one row to `site_runs` and `GET /api/stats` answers to this value in the `X-Stats-Passcode` header. A generated secret of at least 24 ASCII characters (`openssl rand -hex 24` gives 48); a shorter one is refused (503 `not_configured`). Mark it sensitive; it is scrubbed from every response and log line. Unset, nothing is recorded, `/api/stats` answers 404 and `DATABASE_URL` is read no more than before. It is independent of `NER_SITE_PASSCODE`. |
| `NER_SITE_MAX_CHARS` | 5000 | Maximum text length (characters). |
| `NER_SITE_MAX_ENTITIES` | 25 | Maximum entities kept per request; the default prompt (and `/api/health`'s `default_prompt`) states this number. |
| `NER_SITE_MAX_PROMPT_CHARS` | 4000 | Maximum prompt length (characters). |
| `NER_SITE_OFFLINE` | unset | `1` replaces the LLM with the stand-in extractor; `/api/health` reports `"mode": "offline"`. |
| `NER_SITE_DENSE_DIR` | `<project root>/indexes` | Where the committed dense indexes live. |
| `NER_SITE_CACHE_DIR` | `/tmp/xwalk-ner-site` | Where BM25 indexes are built at cold start. |
| `NER_SITE_BACKEND` | `bundled` | `neon` reads the records, BM25 and vectors from Neon Postgres instead of building BM25 in `/tmp` and opening the committed dense indexes (see "Neon backend" below). |
| `DATABASE_URL` | unset | The Neon connection string (mark it sensitive; scrubbed from every response and log line). The indexes use it only with `NER_SITE_BACKEND=neon`; without it every ontology then reports "DATABASE_URL is not set (NER_SITE_BACKEND=neon)". Since 18 November 2026 it is also read, on either backend, on a site that is not offline and where either `ANTHROPIC_API_KEY` is set or own keys are on (`NER_SITE_OWN_PROVIDERS` lists a provider), because the visitor allowance and the own-key flood limit are counted there. Where it is read for one of these, a value that is not a Postgres URL is refused (503 `not_configured`); anywhere else (the bundled backend without the Anthropic key and with own keys off, or offline) it is ignored unread, as before. With `NER_SITE_STATS_PASSCODE` set it is read on every deployment, for the record of runs. Where that is the only reason it is read, a value that cannot be used (not a Postgres URL, no host, malformed) is not an error: there are then no statistics (`GET /api/stats` answers 503 `stats_unavailable`) and everything else answers as with analytics off. |
| `NER_SITE_NEON_SCHEMA` | `xwalk` | The schema `tools/load_neon.py` wrote. The allowance table `site_visits` and the record of runs `site_runs` are created in it; the schema itself must already exist (`tools/setup_usage.py` creates it when it does not). |
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
was left is not tried again in that request. One Claude call has a timeout of 40 s
while extracting (so that Opus and then Sonnet both fit inside the 90 s the extraction
has) and 60 s while selecting.

| What happened on a step | What the chain does | `reason` in `fallbacks` |
|---|---|---|
| Anthropic rate limit (429) | next step at once | `rate_limited` |
| Anthropic overloaded (529) | next step at once | `overloaded` |
| Balance exhausted (402, or a 400 naming the credit balance) | straight to the OpenRouter step, past Sonnet (the two Claude models share the key and the balance); the last step when the chain has no OpenRouter step; remembered for 300 s on that warm instance | `balance_exhausted` |
| Key refused (401, 403) | the same, past Sonnet; remembered for 300 s | `key_refused` |
| Claude declined the text, after the server-side refusal fallback | next step at once | `refused` |
| No answer within the call's timeout | next step at once, with no second wait on the same model | `timeout` |
| The step's call cap for the request was reached | next step at once | `call_limit` |
| 5xx, 408, 409, connection error, a 200 that cannot be decoded (a body that is not JSON, or not a message at all: an HTML page, a body without a JSON content type that is not JSON either) | the same step once more, then the next | `transient` |
| A request the API rejects (other 4xx) | no step-down: extraction answers 502 `provider_error`, a pick is `failed` | none |
| An answer that cannot be read as the JSON asked for | no step-down: extraction answers 422 (`extraction_unparseable`, or `extraction_schema` without an `entities` list), a pick is `failed` | none |

On the last step the error is returned as it is: a failed provider request while
extracting is 502 `provider_error`. Steps decided before any call have stage `admit`:
`opus_allowance_used` (the visitor's Opus allowance for the hour or for the day) and
`claude_unavailable` (a remembered exhausted balance or refused key: the request starts
on the OpenRouter step). A third, `limits_unavailable`, belonged to the first design, in
which a request ran on Sonnet when the limiter could not be asked; the code no longer
produces it (such a request is now refused with 503, below), and only its sentence is
kept for pages that still hold an old response. One more reason,
`follows_extraction` (stage `select`), is not a failure: when the model and the selector
are two different Claude ids, selection runs on the request's one chain, that is where
extraction ended, and the response says so. A Claude selector with an OpenRouter
extraction model gets a chain of its own for the picks. Every departure is in
`fallbacks` and, as a sentence, in `notices`; `model` is the model that extracted and
each `pick.selector` the model that made that pick.

The refusal fallback on Anthropic's side can also answer from a model nobody asked for.
The adapter recognises that by the SDK's own signal (a `fallback_message` entry in
`usage.iterations`), not by comparing model names, and then uses only the text of the
model that served the answer (the text after the last `fallback` content block). It is
recorded like any other departure: stage `extract` or `select`, from the model asked to
the model that answered, reason `refused`; for the picks once per pair of models, and
the notice names the entities whose picks it concerns ("... the picks for e2, e5 ran
on ..."). A request on a visitor's own Anthropic key does not ask for this fallback.

A notice says what ran only where that is known. Of the departures that decided where
the extraction ran (those of stage `admit` included), and again of those that decided
where the picks ran, only the last sentence ends "this ran on ..." or "the remaining
picks ran on ..."; an earlier one ends "moved on to ...", for example "Claude Opus 5.5
is rate limited; moved on to Claude Sonnet 5.5." followed by "Claude Sonnet 5.5 is rate
limited; this ran on Gemini 3.1 Flash Lite through OpenRouter.". The sentence for
`timeout` is "... did not answer in time; ...". A `claude_unavailable` notice names the
reason that was known when the request was admitted.

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

**The visitor allowance.** It applies to requests on the site's keys where Claude is
offered (the Anthropic key is set and the site is not offline); a deployment without
the key counts no such request and behaves as before (requests on a visitor's own key
are counted apart, on any deployment with a database: "Own keys" below). A visitor is
the first 32 hex characters of an HMAC-SHA256 of the client address (`x-real-ip`, else
`x-vercel-forwarded-for`, else the first entry of `x-forwarded-for`) under the salt. The address itself is never stored or logged; the
id is stored only in the table below and is never logged or returned. Requests with no
address header share one allowance.

- Counts are rows in `<NER_SITE_NEON_SCHEMA>.site_visits` (visitor, UTC clock hour,
  tier, count), created on first use; rows older than a day are deleted by about one
  request in fifty.
- A request is counted after every check that can refuse it (size, header, passcode,
  JSON, field limits) and after the indexes are open, so a request that cannot run for
  one of those reasons costs the visitor nothing. One that fails later (502, 422) stays
  counted. It is counted under the tier it starts on: `opus`, `sonnet`, or `other` for
  an OpenRouter model with a non-Claude selector. The tier wanted comes from the model,
  or from the selector when only the selector is Claude.
- Opus is granted while fewer than 10 Opus starts are recorded for the hour and fewer
  than 20 for the day, then Sonnet while fewer than 20 Sonnet starts are recorded for
  the hour and fewer than 40 for the day. When the total of the three site tiers
  reaches 30 for the hour or 60 for the day, or a request for Sonnet finds a Sonnet
  allowance used, the answer is 429 `visitor_limit_reached` with `Retry-After` and
  nothing is recorded ("The daily cap" below has the windows and the wording).
- One visitor's concurrent requests are serialised with a Postgres advisory lock taken
  in the same transaction, so a parallel burst cannot be granted more than the
  allowance. The wait for that lock is bounded (`SET LOCAL lock_timeout = '1500ms'`
  in the same transaction): a request whose turn does not come in time (SQLSTATE
  `55P03`) is answered 429 `visitor_limit_reached` with the message "too many requests
  at once; try again in a few seconds" and `Retry-After: 5`, and nothing is recorded.
  The limiter has its own Neon executor: 3 s timeout and no retries (a retried
  increment could count one visit twice).
- When the limiter cannot be asked for any other reason, a request on the site's keys
  is refused with 503 `limits_unavailable` and nothing runs uncounted ("When the
  limiter cannot be asked" below).

**The daily cap.** On top of the hour, each visitor has a UTC calendar day: 20 requests
starting on Opus and 40 starting on Sonnet (`NER_SITE_OPUS_PER_DAY`,
`NER_SITE_SONNET_PER_DAY`), and their sum, 60, is the day's total for every request on
the site's keys. Nothing new is stored: the day's counts are the sum of that visitor's
hour rows from midnight UTC to the next midnight, read in the same statement and under
the same lock as the hour's.

- The whole day is summed, not the day up to the request's own hour. A request stamped
  14:59 that gets its turn at the lock after a 15:00 request has committed therefore
  sees that request, and the cap cannot be passed at an hour boundary. The upper bound
  is written as 24 hours after midnight, so a database session in a time zone with a
  clock change still reads a UTC day.
- A request for Opus that finds the day's Opus allowance used starts on Sonnet like one
  that finds the hour's used, with the same reason `opus_allowance_used`; its notice
  says which: "Your Opus allowance for today is used; this ran on Claude Sonnet 5.5."
- The refusal names the window that binds. The hour: "you have used this hour's
  requests; try again in N minutes", `Retry-After` the seconds to the next clock hour.
  The day: "you have used today's requests; they reset at midnight UTC", `Retry-After`
  the seconds to midnight UTC. Both end ", or use your own API key" where own keys are
  on. A request is refused for the day only when no tier it could start on has room
  left today; while one has, the refusal is the hour's.
- In `/api/health`, `opus_left` and `sonnet_left` are what can still start this hour,
  each capped by what is left of the day, so a page written before the daily cap stays
  truthful. `opus_left_today` and `sonnet_left_today` are the day's remainders.
  `resets_in_s` is always the seconds to the next clock hour and `day_resets_in_s` the
  seconds to midnight UTC. `window` is `hour` or `day`: which of the two binds the
  next request for Opus (when it is refused or steps down), or would bind first.
- Rows older than 24 hours are still deleted, which keeps every row of the current UTC
  day.

**When the limiter cannot be asked.** A request on the site's keys never runs
uncounted on an open site. This replaces the first design of 18 November 2026, in
which it ran on Sonnet outside the count.

| The limiter | A request on the site's keys | A request on a visitor's own key | `visitor` in `/api/health` |
|---|---|---|---|
| answers | counted, allowed or refused (429) | counted under the flood limit, allowed or refused (429) | `limits: "on"` with the balances |
| does not answer (database unreachable, an error, or no answer in 3 s) | 503 `limits_unavailable`, "the site cannot count requests right now; try again shortly" (", or use your own API key" where own keys are on), `Retry-After: 30` | runs, not counted | `limits: "unavailable"`, `tier: "blocked"` |
| has a queue on the visitor's lock longer than 1.5 s | 429 `visitor_limit_reached`, "too many requests at once; try again in a few seconds", `Retry-After: 5` | the same 429 | not affected |
| is not configured (no `DATABASE_URL`), no passcode | 503 `limits_unavailable`, "visitor limits are not configured", no `Retry-After` | runs, not counted | `limits: "unavailable"`, `tier: "blocked"` |
| is not configured, with a passcode (a private deployment) | runs unmetered, a request for Opus on Opus | runs, not counted | `limits: "off"`, `tier: "opus"`, the full allowance |

The first column applies where Claude is offered. On a deployment without the
Anthropic key, requests on the site's keys are not metered at all, as before (and a
public one needs a passcode, see the security notes), health has no `visitor` block,
and own-key requests are still counted when `DATABASE_URL` is set. Where the limiter
gives no counts, the numbers in `visitor` say what admission would do (0, or the full
allowance in the last row), `own_left` and `own_left_today` are null, and `window` is
`hour`. The page shows "The site cannot count requests right now. Try again shortly,
or use your own API key." for `limits: "unavailable"` with `tier: "blocked"`.

`/api/health` reports `claude`: `{"status": "ok"}`, or `unavailable` with the reason
"key not set", "key refused" (from the key check, or from a request of that warm
instance in the last 300 s) or "balance exhausted" (from such a request). The key is
checked with a free Models API call. A check that could not be completed is not an
outage: `claude` is then `ok` with the reason "key not checked (Anthropic unreachable:
...)" or "key not checked (Anthropic answered HTTP n)". An accepted key is remembered
for 300 s, any other result for 60 s. Where the allowance applies, health also reports
`visitor` for the caller: `limits` (`on`, `off`, `unavailable`), `tier` (where a request
for Opus would start: `opus`, `sonnet`, `blocked`), `opus_left`, `sonnet_left` (each
capped by what remains of the hour's total and of the day), `resets_in_s`, and since the
daily cap `opus_left_today`, `sonnet_left_today`, `own_left`, `own_left_today` (the
own-key flood limit's remainders), `day_resets_in_s` and `window`. Offline mode reports
neither. `own_keys` is reported on every deployment: `{"enabled": true, "providers":
[{"id", "name", "models", "default_model"}, ...]}`, or `{"enabled": false,
"providers": []}`.

**Own keys.** A visitor may run the demo on their own API key for Anthropic, OpenAI or
OpenRouter instead of the site's keys (binding design:
[the spec of 18 November 2026](../docs/superpowers/specs/2026-11-18-own-key-daily-cap-design.md),
with an "As built" section; decisions in docs/ner-site/PLAN.md section 15). Off unless
`NER_SITE_OWN_PROVIDERS` lists the providers to allow, and always off in offline mode.

*What travels where.*

- The page keeps the key in the tab's session storage (one entry, with the provider),
  puts it back in the field after a reload in the same tab, and removes it on "Forget
  key", on any change of "Run with", and when a health read (at load, and after every
  run) no longer offers its provider. It sends it only as `Authorization: Bearer
  <key>` on `POST /api/ner`, with `redirect: "error"`, and with `"provider": "<id>"`
  in the body. Health requests carry nothing.
- The page never moves a run from the visitor's key to the site's keys on its own.
  After a reload the saved choice comes back only when health has answered, so until
  then Run (the button and the keyboard shortcut) is held. If health cannot be
  loaded, or no longer offers the provider, the page says so and runs on the site's
  models only after the visitor presses "Use the site's models" (or picks another
  key under "Run with"). That the visitor chose an own key is kept in session storage
  on its own (the provider id, never the key) and is removed only by that choice, so
  the page asks again after every reload, also when the key itself has been forgotten.
- The API reads the key from that one header, builds a client for that request, and
  closes it after. The key is not written to the settings, a cache, the database, a log
  line or a response, and no length or digest of it is kept. For the length of the
  request it is one more value scrubbed from the response and the log line, and the
  SDK and HTTP client loggers are silent during an own-key run.
- The key goes to the provider's own host and nowhere else: `https://api.anthropic.com`,
  `https://api.openai.com/v1` or `https://openrouter.ai/api/v1`. These are constants:
  `NER_SITE_BASE_URL`, `ANTHROPIC_BASE_URL`, proxy variables, `.netrc` and
  `ANTHROPIC_CUSTOM_HEADERS` do not apply, and redirects are not followed. Each
  request is checked as it leaves (the pinned host, the visitor's key once; for
  Anthropic every header and its value, and a `Content-Length` that is the body's)
  and is not sent otherwise. Of a response, the HTTP client is given the status, the
  body and a header set the site writes itself: one of two fixed content types;
  `content-length`, `retry-after` and `retry-after-ms` only in a strict form; and the
  body's `content-encoding` rewritten in lower case, when every coding in it is one
  the client offered to decode (an answer in any other coding is 502 `provider_error`
  with `bad_answer`). A redirect's `Location`, a charset, a cookie and every other header
  stop there, so a provider that echoes the key in one (in any encoding) leaves it
  in no cache of the warm instance, and no cookie is stored or sent back. The text
  and the prompt go to that provider
  with the key. An own OpenRouter request carries the site's `HTTP-Referer` and
  `X-Title` headers, as the site's own do.
- A provider's error text is never repeated (it can echo the key, whole or masked):
  the visitor gets a fixed sentence with the provider's name, and the log the error
  code.

*What a request may ask for.* `provider` is absent or `site` (the site's keys), or one
of the enabled providers. With an own provider, `model` must be in that provider's list
(`NER_SITE_OWN_MODELS_*`; the first is the default) and `selector` is `retrieval` (the
default) or a model of the same list: picks run on the same key or not at all, and Jev
is not offered, because it bills the operator. The response says `provider` and
`credential_source: "visitor"`, `model` is the model asked for, and `fallbacks` and
`notices` are empty: there is no chain, no step-down and no server-side refusal
fallback, so the visitor is billed for the model they chose and no other. A set
`NER_SITE_PASSCODE` is required as for any request.

*What each provider is sent.* Anthropic: the same request as on the site's key (effort,
token limits, schema-enforced output) without the refusal fallback, 60 s per call for
both steps, and no retry, so one transient failure is a 502. Claude Haiku 5.5 is
offered here and only here. OpenAI: the engine's OpenAI-compatible client with
`max_completion_tokens` (16000 for extraction, 8000 for selection) in place of
`max_tokens`, no `temperature` and no `reasoning_effort`, the schema in strict mode;
60 s per request, up to three requests for an extraction and two for a pick. This path
has never met the live API. OpenRouter: the site's own OpenRouter client and limits
with the visitor's key. The call caps per request are the site's.

*What the operator still pays for.* The model calls are billed to the visitor. The
operator pays for the term lookup: the query embeddings for dense retrieval (OpenRouter,
`OPENROUTER_API_KEY`; without that key own-key requests run on BM25 alone), the Neon
queries, and the function's time, up to about 105 s a request. An own-key request needs
neither of the site's LLM keys to be set.

*The flood limit.* Own-key requests do not count against the Opus and Sonnet
allowance. They have their own: 30 per visitor and clock hour and 60 per UTC day
(`NER_SITE_OWN_PER_HOUR`, `NER_SITE_OWN_PER_DAY`), counted in the same table under the
tier `own`, on any deployment with `DATABASE_URL`, also one without the Anthropic key.
Over it: 429 `visitor_limit_reached`, "too many requests with your own key; try again
in N minutes" (N counts to the reset of the window that binds, the hour or midnight
UTC), with `Retry-After`. It bounds what one address can make the operator pay for
lookups; it is not counted when the limiter cannot be asked or is not configured (the
table above), and then nothing bounds it but the function's own limits.

*Errors.* All with fixed text.

| Status and code | When |
|---|---|
| 400 `bad_credential` | An own provider without exactly one `Authorization: Bearer <key>` header; a key that is not 20 to 512 printable ASCII characters without spaces; a provider that is unknown or not enabled; the key also present in the text or the prompt; or, where own keys are on, a request on the site's keys that carries an `Authorization` header at all. Where own keys are off, that header is not looked at on a request for the site's keys. |
| 401 `provider_key_refused` | "<Provider> refused this key. It may be mistyped, revoked, out of credit, or not allowed to use this model." The provider answered 401, 402, 403 or 404 (a model the key may not use, or one that does not exist, is a 404), Anthropic named the credit balance, or OpenAI answered 429 "exceeded your current quota". |
| 429 `provider_rate_limited` | "<Provider> is rate limiting this key; try again shortly." No `Retry-After`. |
| 502 `provider_error` | "<Provider> could not complete the request (<word>)." The word is one of `timeout`, `overloaded`, `http_5xx`, `bad_request`, `refused`, `unreachable`, `bad_answer`, `call_limit`, `error`. The request deadline and the call cap are reported this way too (`timeout`, `call_limit`), not as the site-key codes. |

An extraction answer that cannot be read is 422 as on the site's keys. A pick that
failed carries one of the same words as its explanation (also `key_refused` and
`rate_limited`), never the provider's text.

*In the log.* The log line of a `POST /api/ner` that passed the passcode and JSON
checks says `provider` and `credential` (`site` or `visitor`), also when the
credential was then refused. One of the three provider errors above also has
`own_error`, the class name of what failed (`LLMFatalError` or `LLMRetryableError`
for a provider's answer, `TimeoutError` for the deadline, `CallLimitExceeded`,
`WireRefused` for a request the site did not send, or the class of whatever kept a
client from being built). A call that fails during selection is a failed pick inside
a 200, and the line of that request has `own_error` too: the first class that
failed, or `WireRefused` whenever a pick was refused, whatever else failed beside
it. So `"status": 200` with `"own_error": "WireRefused"` means picks the site itself
did not send. When picks failed for more than one reason, `own_errors` lists the
classes, sorted and all of them, so the one `own_error` names is always in the list.
Only a failure of the provider, the SDK, the HTTP client or a timeout is reported as
the provider's error, and so is a 200 that cannot be decoded (`bad_answer`): a body
that is not JSON, one that is not a message at all (an HTML page, or a body without a
JSON content type that is not JSON either), or one in a content coding the client did
not offer. Any other exception is a fault of the site's code.
During extraction it answers 500 `internal`, with `exception` and `where` in the log
line as for any request; during selection it is a failed pick with the word `error`
and its class in `own_error`. A client that fails to close after the run changes
nothing for the visitor and adds `own_close_error` (a class name) to the line. No
part of the key is in any of these.

*The Anthropic SDK is pinned.* `anthropic==2.10.0`, in `requirements.txt` and in the
`dev` extra. What keeps the operator's environment out of a visitor's Anthropic
request was written against that version's source; see step 8 of the own-key rollout
before changing it.

*Turning it off.* Own keys are off while `NER_SITE_OWN_PROVIDERS` is unset or empty; a
list keeps only the providers named. Offline mode has none.

Tests for the limiter's SQL need a real Postgres and are not part of the default suite.
With a Postgres URL in `NER_SITE_TEST_DATABASE_URL` and the `psql` client (16 or later)
on the path, they run the real statements, including 40 overlapping requests of one
visitor and a visitor's lock held past the lock timeout, each test in a schema of its
own that it drops; without either they skip:

```bash
NER_SITE_TEST_DATABASE_URL=postgresql://user@localhost:5432/postgres \
    python -m pytest -q -m integration tests/test_site_limits.py
```

### Analytics: traffic and usage

As built on 18 November 2026 (design:
[docs/superpowers/specs/2026-11-18-site-analytics-design.md](../docs/superpowers/specs/2026-11-18-site-analytics-design.md);
decisions: [docs/ner-site/PLAN.md](../docs/ner-site/PLAN.md) section 16). Two halves
that do not depend on each other. Both are off until the operator turns them on, and a
deployment that turns neither on behaves as before.

**Traffic: Vercel Web Analytics.** `public/index.html` loads two scripts from the
site's own origin, both deferred and after `app.js`: `analytics.js` (the one-line queue
stub of Vercel's snippet, in a file so that the Content-Security-Policy needs no
exception) and `/_vercel/insights/script.js`. Page views only; the site sends no custom
events, and the stats page does not load the scripts. What is collected is collected by
Vercel, not by this code: per page view the path, the referrer, the country, and the
device, operating system and browser, without cookies (Vercel's Web Analytics
documentation has the exact list and how it tells visitors apart). The footer says
"This site counts visits without cookies."

To turn it on: in the Vercel dashboard open the project, Analytics, enable Web
Analytics, and redeploy. Then load the page with the browser's network panel open:
`/_vercel/insights/script.js` answers 200 and a request to `/_vercel/insights/view`
follows; the visit shows in the dashboard's Analytics tab. Until it is enabled
`/_vercel/insights/script.js` answers 404, the browser logs that (Chromium adds a line
about the MIME type of the 404 page), and the page works as before.

**Usage: one row per run.** With `NER_SITE_STATS_PASSCODE` set and a usable
`DATABASE_URL`, every `POST /api/ner` that got past the passcode and body parsing
writes one row to `<schema>.site_runs` (`ner_site/usage.py`) after its response is
built: the 200s, the 4xx and 5xx that follow, and the 429s and 503s of admission.
Requests refused before that (a missing header, a wrong passcode, an unreadable body),
health, stats and static requests write nothing.

What a row holds:

| Column | Meaning |
|---|---|
| `at` | When the request finished (server clock, UTC). |
| `visitor_day` | 32 hex characters: a keyed hash of the visitor id and the UTC date. It counts distinct visitors within a day; the same visitor has another value the next day, so rows cannot be joined across days or back to an address. A request refused for its credential is counted under the address it came from, like an admitted one; a value in an address header that is not an IP address is not hashed. |
| `status`, `code` | The HTTP status and the error code of the answer (null on success). |
| `credential`, `provider`, `tier` | `site` or `visitor`; `site`, `anthropic`, `openai` or `openrouter`; the admitted tier (`opus`, `sonnet`, `other`, `own`) or null. |
| `requested_model`, `model`, `requested_selector` | The ids asked for and the model that extracted. `model` is null for a run that no model answered. |
| `fallback_reasons` | The distinct step-down reasons, comma-joined; empty when none. |
| `country` | The two-letter code Vercel puts in `x-vercel-ip-country`, or null. |
| `text_chars`, `custom_prompt` | The length of the text, and whether a prompt was sent. |
| `entities`, `matches`, `picks` | Counts. |
| `llm_calls`, `prompt_tokens`, `completion_tokens` | The extraction's calls and tokens. |
| `selector_calls`, `selector_prompt_tokens`, `selector_completion_tokens`, `selector_cost_usd` | The selection's calls and tokens, and the cost its provider reported (Jev). |
| `embedding_requests`, `extract_ms`, `map_ms`, `select_ms`, `total_ms` | Embedding requests and timings. |

What it does not hold: the text, the prompt, an entity, a term id, an IP address, the
visitor id, a passcode, an API key (the site's or a visitor's) or a user agent. No
column is free text: a model or selector is stored only when it is an id this
deployment lists (`NER_SITE_MODELS`, `NER_SITE_SELECTORS`, `NER_SITE_FALLBACK_MODEL`,
the own-key lists, the Claude ids, `jev`, `retrieval`), an error code and a step-down
reason only when they are ones the site defines, and anything else is stored as the
word `other`. That matters for `model`: the model that answered is named by the
provider's response. One other spelling is recognised: a known id followed by a hyphen
and a date or version of digits and hyphens (`claude-opus-5-5-20261101`) is stored as
the known id (`claude-opus-5-5`), so a dated snapshot is priced like the model that
was asked for. Any other spelling is a row with `model: other`, which has no price.

A run that failed after a model had answered (an answer that could not be parsed, a
refusal that reported its tokens) records the calls and tokens its failure carried,
under the model the request asked for, so that it is priced. A failure that reported
no tokens records its calls and no model.

What `/api/stats` returns goes through the same mapping again on the way out, with the
deployment's lists as they are when it is asked: a row written before these rules, or
by hand, cannot put a string of its own into the answer, and a model that has since
left the lists is shown as `other`.

**Turning usage on.**

1. Create the table once, before recording is on:
   `DATABASE_URL=postgresql://... python ner-site/tools/setup_usage.py` (add
   `--schema NAME` when `NER_SITE_NEON_SCHEMA` is not the default `xwalk`). It sends one
   request with `CREATE SCHEMA IF NOT EXISTS`, `CREATE TABLE IF NOT EXISTS` and
   `CREATE INDEX IF NOT EXISTS`, writes no row, never prints the connection string, and
   can be run again. The site also creates the table by itself on the first run of a
   warm instance, but two cold instances doing that at the same moment on a database
   without the table can collide, and the one that loses also loses its row; with the
   table in place there is nothing to collide over.
2. Generate the passcode: `openssl rand -hex 24`. Set it as `NER_SITE_STATS_PASSCODE`
   (sensitive) and redeploy.
3. Do a run on the site, then open `https://<deployment>/stats` (linked from nowhere,
   `noindex`), enter the passcode and press Show.

**The stats page and API.** `/stats` is a `vercel.json` rewrite to `public/stats.html`.
It asks for the passcode, keeps it in session storage for the tab (the field is emptied
once the passcode is accepted; "Forget passcode" removes it), and calls one endpoint:
`GET /api/stats?days=N` with the header `X-Stats-Passcode`. `days` is 1 to 365, default
30; the page offers 7, 30 and 90. Answers: 401 `stats_passcode_required` or
`stats_passcode_invalid`; 400 `bad_days` for a `days` outside the range (after the
passcode check); 404 when analytics is off; 503 `stats_unavailable` when the
statistics cannot be read: no usable `DATABASE_URL`, a database client that cannot be
built, an error, a step of the exchange that takes longer than 10 s, or an answer in
which any value is not what it has to be (the days are exactly the UTC days asked for,
counts are whole numbers that are not negative, words are strings; nothing is passed
on from an answer that fails this). The 10 s are the client's timeout for each
operation (connecting, sending, each read), not a deadline for the whole read: a
database that answers very slowly can keep the request longer, up to the function's
own limit. Else 200 with `totals`, `by_day` (every UTC day of the period, zeros included), `by_model`,
`by_code`, `by_fallback` and `by_country`. `by_model` has a row with `model: null` for
runs no model answered, a row `other` for model names the site does not know, and a
row `jev` (with `runs: 0`) for what Jev reported; `by_country` has a row with
`country: null`. The page shows the estimated spend, the share of runs with a step-down
note, runs by tier (a grey "No tier" segment is the runs in none of the four tiers:
most were refused before admission), successful runs, visitor-days, two charts and the
tables.

**What the estimate includes and what it does not.** Dollars are computed in code from
the token sums, for runs on the site's keys only, at list prices per million tokens
(input / output): `claude-opus-5-5` 4 / 20, `claude-sonnet-5-5` 2 / 10
(`usage.PRICES`). Extraction tokens are priced at the model that answered, selection
tokens at the selector that was asked for, and what Jev reported is added as it is.
The tokens of a failed run are priced at the model the request asked for. They are a
sum over every model the chain tried and nothing says which model used which; the
chain only steps down to cheaper models, so this can overstate such a run (a request
for Opus that failed on Sonnet is priced as Opus) and never understate it.
The per-model and per-day dollars both add up to the total. Not in the estimate:

- Models without a list price: every OpenRouter model, and `other`. Their runs are
  counted in `unpriced_runs` (once per run, when the extraction model or a model
  selector has no price), and the page says so above the charts.
- A successful run in which any model call did not report its tokens (a retry after a
  failed attempt, a step-down in the middle): its token columns are null and it adds
  nothing. Such runs are not in `unpriced_runs`. This is the largest known gap.
- Query embeddings on the OpenRouter key, and all cache pricing (cached input tokens
  are priced as input).
- A run that stepped down is priced at the model that answered the extraction; the
  tokens of the attempts before it are unknown.
- Runs on a visitor's own key cost the operator nothing for the model and are shown
  with "visitor pays".

The Anthropic Console is the bill.

**What recording costs a request.** One HTTPS round trip to Neon (a single `INSERT`),
made after the response is built and before it is sent, through the same client the
visitor allowance uses (the engine's Neon client, with no retries). The write runs in
a worker thread. What the request waits for has one deadline: 2 s on the clock from
the moment recording begins. The worker waits no longer than that for a lock the
instance's requests share (behind it the database client is found or built), a write
whose turn comes after the deadline does not start, and the request stops waiting at
the deadline and answers, whatever the write is doing. At most four writes are under
way on a warm instance at once; a run that finds all four taken is not recorded and
does not wait. So recording costs a request at most 2 s and never an answer.

What is left, accepted as it is: a write that has started cannot be stopped. When its
request stops waiting, the write ends when the client's own timeouts end it. They
are 2 s for each step of the exchange (connecting, sending, each read of the answer),
so a database that is down or silent ends it promptly. A database that keeps sending
an answer very slowly, a little at a time, never lets a read time out, and the write
stays. With four such writes, four worker threads of that warm instance stay busy
and its recording pauses: later runs log `usage: "saturated"`. Visitors' answers are
not affected. Recording resumes when the connections end or the instance is
recycled. The workers are daemon threads and never keep the process alive.

`timing_ms.handle` in the log does not include recording. The log line of a request
gains one field, and only when its run was not recorded:

| `usage` in the log line | Meaning |
|---|---|
| absent | The row was written, or recording is off. |
| `"TimeoutError"` | The write was not done within 2 s and the request stopped waiting. The write itself goes on until it ends, so the row may still be written. |
| `"saturated"` | Four writes were already under way; this run was skipped at once. |
| `"PostgresError"` | The database refused the statement or could not be reached. Only the class and, when the server gave one, its five-character SQLSTATE are kept of such an error (the statistics request logs both, as `stats`). |
| another class name | What else failed (`ValueError`: a connection string the client cannot use). |

**Deleting the data.** In the Neon SQL editor or with `psql`, on the schema of the
deployment (`xwalk` by default):

```sql
DELETE FROM xwalk.site_runs WHERE at < '2026-12-01';   -- everything before a date
DROP TABLE xwalk.site_runs;                              -- everything
```

Nothing prunes the table. After a `DROP TABLE`, a deployment that still has
`NER_SITE_STATS_PASSCODE` set creates the table again, but not on its next run: a
warm instance remembers that it created the table, so the first write it sends after
the drop fails (`usage: "PostgresError"`, that row is lost), and the next write it
sends after learning that creates the table. Writes that were already under way when
the instance learned it fail the same way, so an instance loses one row, or a few
when it was serving several requests at that moment; each warm instance learns it
for itself. To lose nothing, run `tools/setup_usage.py` right after the drop. To stop recording instead, unset the passcode
first.

**Turning it off.** Unset `NER_SITE_STATS_PASSCODE` and redeploy: nothing is recorded,
`/api/stats` answers 404, no database client is built for it, and the rows stay until
they are deleted. Web Analytics is turned off in the Vercel dashboard.

**Rollout checks (operator).**

1. Enable Web Analytics and redeploy; a visit shows a request to
   `/_vercel/insights/view` and appears in the dashboard.
2. Run `tools/setup_usage.py`, set `NER_SITE_STATS_PASSCODE` and redeploy; do a run;
   open `/stats`, enter the passcode, see the run.
3. Watch the first production log lines of `POST /api/ner` for a `usage` field. The
   statements were run on a real Postgres 16, but through `psql`: Neon's HTTPS
   endpoint has not been exercised with them. `usage: "PostgresError"` on every run
   means the insert is refused there and nothing is being recorded.
4. In the Neon console, look at the first rows: no text, no address, and `model`
   showing the model ids rather than `other`. A dated snapshot id is stored as its
   base id; if Anthropic's responses name the model in some other spelling, every
   Claude run is `other` and unpriced until that spelling is known to the site.
5. `curl -si https://<deployment>/api/stats` answers 401 `stats_passcode_required`,
   and with a wrong `X-Stats-Passcode` 401 `stats_passcode_invalid`.

Tests for the statistics SQL need a real Postgres, like the limiter's:

```bash
NER_SITE_TEST_DATABASE_URL=postgresql://user@localhost:5432/postgres \
    python -m pytest -q -m integration tests/test_site_usage.py
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
  call, or two after a transient error, each of at most 40 s), selection calls up to twice the entity limit
  on a step (normally one per entity) and 25 entities (query embeddings are never
  retried), and a request finishes by about 105 s (90 s for the extraction, one
  deadline for the rest; Vercel cuts a run off at 120 s), but nothing stops many
  addresses from sending requests.
- **Without the Anthropic key, set `NER_SITE_PASSCODE` for a public deployment.**
  Nothing counts requests on the site's keys there, so anyone who finds the URL can
  spend the OpenRouter key up to its limit. Share the passcode out of band; the service
  compares it in constant time. For more, Vercel's firewall rate limiting is an option.
- **With the Anthropic key, the visitor allowance is what makes an open deployment
  bounded per visitor** (by default 30 requests an hour, 10 of them starting on Opus,
  and 60 a day, 20 of them starting on Opus). It bounds one address, not the total:
  many addresses can still spend the balance, which is why the balance must be the
  ceiling. It is a rate, not a budget: what one address can spend in a day is the
  daily cap times the cost of a request (rollout step 5). It needs `DATABASE_URL`;
  when the database cannot be asked, requests on the site's keys are refused with 503
  and none runs uncounted (rollout step 7), so a database outage takes the site's
  models down rather than opening them. It exists only with the Anthropic key:
  removing the key from an open deployment without setting the passcode again leaves
  the site public and uncounted on the OpenRouter key.
- **A visitor's own key is the visitor's secret, and the site must not keep it.** It
  arrives in the `Authorization` header of `POST /api/ner`. The code never stores,
  logs or returns it ("Own keys" above), but the platform in front of the code must
  not either: confirm that Vercel's logs, log drains and integrations record no
  request headers before turning own keys on (own-key rollout, steps 2 and 3), and
  leave `NER_SITE_OWN_PROVIDERS` unset until then. The page keeps the key in session
  storage, which any script of the origin can read: the Content-Security-Policy
  below, which allows scripts from the site only, is what protects it, so do not
  loosen it. Own-key requests cost the operator the term lookup; the flood limit (30
  an hour, 60 a day per address) bounds that, and does not count while the limiter
  cannot be asked.
- **The stats passcode is the only thing between the internet and the usage numbers.**
  `GET /api/stats` has no attempt limit of its own and answers 401 to a wrong passcode
  as often as it is asked, so the passcode must be a generated secret
  (`openssl rand -hex 24`), never a phrase; the site refuses anything under 24
  characters and compares in constant time, before any database call. For a second
  layer, add a Vercel Firewall rate-limit rule on the path `/api/stats`. Whether
  analytics is on is not a secret: the endpoint answers 404 when it is off and 401
  when it is on. The page keeps the passcode in session storage, like a visitor's
  key, under the same Content-Security-Policy.
- The keys live only in the function environment. `/api/health` never returns them, and
  error messages pass URLs through redaction. Do not put a key in `NER_SITE_BASE_URL`.
- `vercel.json` sends `Content-Security-Policy: default-src 'self'; connect-src
  'self'; img-src 'self' data:; style-src 'self'`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, and `Cache-Control: no-store` on `/api/*`. No CORS
  headers are sent, and the API requires `X-Requested-With: xwalk-ner`, so other
  sites cannot call it from a browser.
- Visitors' text is sent to the model provider (Anthropic for the Claude models;
  OpenRouter and the model's host for the others, and OpenRouter for query embeddings
  and Jev). With an own key it is sent to the provider the visitor chose, on their
  key; the entity names still go to OpenRouter on the operator's key for the query
  embeddings. The function log records sizes, counts, timings and status only, never
  the text, the prompt or a key; since 18 November 2026 also the tier a request
  started on, its step-downs, the model that ran, the provider and whose credential
  it ran on (`site` or `visitor`), and never the visitor id, the client address or
  any part of a visitor's key. The page's footer tells visitors that requests are
  limited per visitor using a salted hash of their IP address, kept for one day.

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

- `python -m pytest -q -m 'not integration' tests/test_site_*.py`: 533 passed, 8
  deselected (after the review fixes below; 490 before them). Offline only: a fake SDK client for the adapter, scripted clients for
  the chain, a fake SQL executor for the limiter, `httpx.MockTransport` for OpenRouter,
  a local `http.server` stub for `claude_check.py`. They cover the request the adapter
  sends (no sampling parameters, effort, token limits, the schema format and its
  normalisation, the fallback parameter), every row of its error table and that no
  message carries the key; each step-down trigger, the single transient retry, a
  timeout stepping down after one call, a balance or key failure skipping Sonnet,
  stickiness under concurrent calls, the usage of failed attempts and the 300 s block;
  the allowance at its boundaries (10th and 11th, 30th and 31st visit), the hour
  rolling over, the lock timeout and the lock statement ahead of every visit, the
  unreachable and off states, and that no raw address reaches a SQL parameter; through
  `handle`, Opus by default, Sonnet after the allowance with its notice, 429 with
  `Retry-After` (for a used hour and for a contended lock), health `claude` and
  `visitor`, an index failure counting no visit, no Claude without the key, and a deployment with only the
  OpenRouter key and a passcode running without limiter or chain; and that the keys,
  the database URL, the salt, the client address and the visitor id appear in no body,
  header or log line.
- `ruff check` and `ruff format --check` on `src tests scripts ner-site`: passing.
- `tools/claude_check.py` against `tools/serve_local.py --canned` on 127.0.0.1 (the
  `fallback` scenario, one text and a burst of 3): it printed the canned health blocks,
  the step-downs and the totals. Canned answers, so this shows the tool's output and
  nothing about the service.
- By the implementing sessions, on a throwaway local Postgres 16 (not Neon): the
  seven `-m integration` tests of `tests/test_site_limits.py` passed, with the lock
  timeout in the transaction. Among them: 40 overlapping requests of one visitor, of
  which those that got their turn within the lock timeout were granted in order (Opus
  up to 10, then Sonnet) and counted once each, the others failing with SQLSTATE
  `55P03` and counting nothing; and a visitor's lock held by another session, which
  made `admit` give up after about 1.5 s with `55P03` while another visitor was
  admitted at once. They are skipped in a run without `NER_SITE_TEST_DATABASE_URL`.
  The page was rendered in headless Chromium against the canned scenarios at 1280 and
  390 px (before the review fixes; the page itself did not change in them).

Changed after the final review, 18 November 2026 (all offline, same tests): a timeout
steps down at once and a Claude extraction call has 40 s; a balance or key failure
skips Sonnet; only the last notice of a part of a request says what ran; an answer
from Anthropic's refusal fallback model is recorded as a departure; a contended
visitor lock is a 429, not an uncounted request; the indexes are opened before a visit
is counted; `DATABASE_URL` is ignored where nothing uses it; a key check that could
not be completed is not an outage; `claude_check.py --burst` fails when the limiter is
failing.

Not run, and therefore not known:

- Whether the Anthropic API accepts `fallbacks: "default"` (beta
  `server-side-fallback-2026-07-01`) together with `output_config.format`, and whether
  it accepts the normalised schemas (rollout step 4).
- How long a 25-entity request takes with Opus extracting and picking, and what a
  request costs in tokens (step 5). Whether the reported tokens include an attempt
  that the server-side refusal fallback replaced.
- Whether Neon's batch endpoint runs the limiter's transaction at READ COMMITTED, on
  which the advisory lock's guarantee depends, and whether it honours `SET LOCAL
  lock_timeout` there as a plain Postgres does (step 6).
- Whether the `model` the API echoes is the id that was requested (step 4), how the
  workspace's per-minute limits compare with one 25-entity request (step 5), and
  whether a suspended Neon compute answers inside the limiter's 3 s (step 3).
- Whether Opus answers a 25-entity extraction inside the 40 s one call now has.
- The Anthropic Console settings: workspace, key, auto-reload (step 1).
- Any Vercel build or deployment with `anthropic` installed, Python 3.12 or 3.10, and
  the cold-start time with the SDK imported.

Known and accepted when this section was written: while the limiter's database cannot
be asked, requests run on Sonnet uncounted. That is no longer so: the change recorded
in the next section refuses them with 503 instead (step 7 as it reads now). The same
change added the daily cap and replaced the way a server-side refusal fallback is
recognised, so the fourth item of the list above (the model the API echoes) no longer
decides whether a false notice appears.

### Own keys and a daily cap, 18 November 2026

**Nothing was deployed, no production variable was changed, and no request was made
with a real key to Anthropic, OpenAI, OpenRouter, Neon or https://www.xwalk.si.**
Production still runs the deployment of 4 November. This section records code and
offline tests only; the live checks are the operator's, in "Rollout: own keys and the
daily cap" above.

What is implemented (branch `feat/own-key-daily-cap`, on top of the Claude change; how
it behaves is in "The daily cap", "When the limiter cannot be asked" and "Own keys"
above, the decisions in docs/ner-site/PLAN.md section 15):

- `ner_site/limits.py`: the daily cap and the own-key tier, in the same table and the
  same single round trip.
- `ner_site/providers.py` (new): the credential, the per-request clients for the three
  providers, the fixed error words.
- `ner_site/claude.py`, `fallback.py`: the refusal fallback can be switched off, Claude
  Haiku 5.5 is a known id, and a server-side fallback is recognised by the SDK's
  signal.
- `ner_site/service.py`, `http.py`, `schema.py`, `select.py`: the new settings, the
  `provider` field, admission for both kinds of request, 503 `limits_unavailable`,
  `own_keys` and the new `visitor` fields in health, `provider` and `credential` in
  the log line.
- `public/`: "Run with", the key field and its sentences, the allowance line for the
  day and for the unavailable limiter.
- `tools/claude_check.py` (`--provider`, `--key-env`, the own-key tier, the day and
  the 503 in the burst summary) and the `day`, `unavailable`, `own-refused`, `own-bad`
  and `own-off` scenarios of `tools/serve_local.py`.

What was run, and nothing more (Python 3.13.3, the repository venv, no key in the
environment):

- `python -m pytest -q -m 'not integration' tests/test_site_*.py`: 1000 passed, 23
  deselected. Offline only: the fakes of the earlier section, plus `httpx` and
  `httpx2` mock transports for the three own-key clients. They cover the daily cap
  at its boundaries (the 20th and 21st Opus start and the 60th and 61st request of a
  day across hours, midnight UTC, a late request at an hour boundary), the own-key
  limits apart from the site's, `peek` giving the same numbers; the credential header
  (missing, repeated, malformed, too long, the key inside the text), each provider's
  client built with the visitor's key and the fixed host, no redirect followed, no
  proxy or environment header used, the fixed error sentences with the key echoed in
  the fake provider's errors, two overlapping requests with different keys; through
  `handle`, 503 for the site's keys and a successful run for an own key when the
  limiter fails, own-key requests not counted as site visits, `own_keys` in health,
  and that a sentinel key appears in no body, header or log line; for the tool, the
  cases in `tests/test_site_claude_check.py` against a stub on 127.0.0.1.
- `ruff check` and `ruff format --check` on `src tests scripts ner-site`: passing.
- `tools/claude_check.py` against `tools/serve_local.py --canned` on 127.0.0.1, with a
  made-up key, in the plain, `unavailable`, `day`, `own-refused` and `own-off`
  scenarios (one text and a burst of 3, on the site's keys and with `--provider`): it
  printed the provider and credential of each run, the 503 and the day refusals in
  the burst summary, sent nothing in `own-off`, and the canned server logged
  `authorization=present` for the own-key POSTs only and the key nowhere. Canned
  answers, so this shows the tool's output and nothing about the service.
- By the implementing sessions: the 22 `-m integration` tests of
  `tests/test_site_limits.py` on a throwaway local Postgres 16 (not Neon), with the
  session in a time zone that has a clock change, among them eight requests racing
  for the last daily slot; and the page in headless Chromium against the canned
  scenarios at 1280 and 390 px, light and dark, with a request listener asserting that
  only `POST /api/ner` carried the key and that it appeared nowhere in the document,
  the URL, local storage or the console.

Not run, and therefore not known:

- Anything against OpenAI's API: whether `gpt-6-luna` exists there, whether it accepts
  `max_completion_tokens`, the system role and the two schemas in strict mode, whether
  JSON comes back inside the token limit, and how long it takes (own-key rollout,
  step 4).
- Any real key on any provider, Claude Haiku 5.5 through this adapter, and whether
  OpenAI's out-of-credit answer still carries the phrase the code looks for.
- Whether Vercel, its log drains or an integration record request headers (step 2).
- Whether Chrome or Safari offer to save the key field; Firefox, Safari and a screen
  reader on the changed page at all (step 7).
- The daily cap and the 503 on Neon and on a deployment (steps 5 and 6).
- Python 3.10 and 3.12 (no such interpreter was at hand; the code is checked by ruff
  against the 3.10 target only), and any Vercel build.

Known and accepted:

- Turning own keys on (`NER_SITE_OWN_PROVIDERS` lists a provider) makes `DATABASE_URL`
  read (and refused when it is not a Postgres URL) on that deployment unless it is
  offline.
- Where own keys are on, a request on the site's keys that carries any
  `Authorization` header is refused with 400 `bad_credential`. A proxy or client that
  adds one to every request would break such a site (own-key rollout, step 1).
- A burst of the check tool cannot show the own-key count on a deployment without the
  Anthropic key (no `visitor` block in health): it reports `INCONCLUSIVE`.
- The page learns that a provider was turned off at its next health read, which is at
  load and after a run: the run before it still carries the key, and the API refuses
  it with 400 `bad_credential`.
- An own Anthropic key gets no retry: one transient failure is a 502.
- Own-key requests are not counted while the limiter cannot be asked, and a response
  does not say whether it was counted.
- On a deployment without the Anthropic key, health has no `visitor` block, so the
  page cannot show an own-key flood limit before the 429 arrives.

### Fixes after the final review, 18 November 2026

Two reviews of the whole branch (one of them by another model) found the points below.
**The sessions that wrote these fixes deployed nothing, changed no production variable
and made no request with a real key to any provider, to Neon or to
https://www.xwalk.si.** The sections above describe the code after these fixes; the
decisions are PLAN.md 15.6, 15.11, 15.12 and 15.16 to 15.20.

- Own keys are off unless `NER_SITE_OWN_PROVIDERS` lists providers, and a deployment
  without them does not read `DATABASE_URL` for the limiter, does not look at an
  `Authorization` header on a site request, and does not validate an own model list.
- Own-key clients send through a transport wrapper that stops a request not built
  for the visitor's key and removes a redirect's `Location`, and any header carrying
  the key, before the HTTP client parses the response. `anthropic` is pinned to
  2.10.0.
- The page holds Run until a saved own-key choice is settled, and follows `own_keys`
  on every health read.
- The check tool keeps its secrets off proxies and plain http, reports a response
  that echoes one, and passes a burst only when the balances account for it.
- Own-key failures are named in the log; a failed close no longer replaces a result;
  picks cut off by the deadline are counted with unknown tokens.

A second review of these fixes found more, fixed the same day:

- Response headers are no longer searched for the key but rebuilt from an allowlist:
  an encoded charset still reached the codec cache, and an ordinary `Set-Cookie`
  from Anthropic was replayed by the HTTP client and refused by the site's own
  check, failing every pick after the first.
- A refusal is attributed to the call that was refused and is logged for selection
  too; the outgoing check compares `Content-Length` with the body.
- A fault inside an adapter is no longer sealed as the provider's failure.
- The page keeps the visitor's choice to pay with an own key apart from the key, so
  a second reload cannot run on the site's keys unasked.
- The tool resolves percent escapes before looking for a secret, and a burst in
  which nothing started is `INCONCLUSIVE`.

And a third, of the second round's fixes:

- The response header allowlist compared `Content-Encoding` by case, so an answer
  sent as `GZip` reached the parser compressed. Codings are now lower-cased and
  checked against what the client offered; one it cannot decode is `bad_answer`.
- A 200 from Anthropic that is not JSON raised the decoder's own error, which the
  narrowed sealing let through as a 500. It is `bad_answer` again; only the SDK
  call is covered, so a fault of the site's code still is not. On the site's own
  key the same answer was a 500 all along: it is now a `transient` failure there,
  so the chain tries the model once more and then steps down, with the usual
  notice.
- The call budget of an own-key request keeps the class of a failed call, not its
  message.
- A refused pick is named in the log whatever failed beside it (`own_errors` lists
  every class).
- The tool looks for a secret at every layer of percent decoding, also in the
  secret's own decoded forms, and a value escaped more deeply than it decodes
  (eight rounds) is withheld and fails the echo check.

And a fourth, of the third round's fixes (18 November 2026, with the analytics fixes
below):

- The tool applied its decoding limit to the response and not to the secret: with a
  key that holds an escape nine layers deep, the site's echo of its decoded form was
  printed and the run exited 0. Such a key or passcode is now refused before anything
  is sent, and when the limit is what failed the check, the last line says an echo
  was detected or could not be ruled out.
- `own_errors` was cut to eight classes, which could drop the one `own_error` names.
  It lists every class.
- A 200 from Anthropic that is not a message (an HTML page, or a body without a JSON
  content type that is not JSON either) was read as an empty answer: 422 after one
  call, with no retry and no step-down on the site's key. A body with an integer too
  long for Python to convert was a 500. Both are response-decoding failures now:
  `transient` on the site's key, `bad_answer` on an own key.
- An answer in a content coding that was not offered left its dispatch open in the
  call budget on OpenAI and OpenRouter. It is settled as `bad_answer`.

What was run (Python 3.13.3, the repository venv, no key in the environment):

- `python -m pytest -q -m 'not integration' tests/test_site_*.py`: 1305 passed, 23
  deselected (1000 before these fixes, 1153 after the first round, 1203 after the
  second).
  The fixes were written test first. Reproduced as failing tests before the code
  changed: for each of the three providers, a 302 with the key in a relative
  `Location` left it in the `urlsplit` cache of `urllib.parse` after the client was
  closed, and a charset that echoed the key left it in the `encodings` cache; an
  operator value under a header name the SDK uses reached the wire once the removal
  of `ANTHROPIC_CUSTOM_HEADERS` was bypassed; the tool sent its requests for
  127.0.0.1 to a configured proxy, printed a key with a quote in its JSON spelling,
  and exited 0 for a burst whose balances had not moved.
- `ruff check` and `ruff format --check` on `src tests scripts ner-site`: passing.
- The page in headless Chromium against `tools/serve_local.py --canned` on 127.0.0.1,
  at 1280 px light and 390 px dark, with a request listener and a made-up key: after a
  reload with a kept key, with the health answer delayed, answering 503, or
  unreachable, neither the button nor the keyboard shortcut posted anything (before
  the fix both posted on the site's keys); the choice came back with health, or after
  "Try again"; "Use the site's models" forgot the key and then posted without one;
  a provider turned off between two runs was removed from "Run with", its key
  forgotten, and nothing posted until a choice was made; three reloads in a row
  with the provider gone, with and without a kept key, each held Run and asked
  again (before the second round's fix the second reload posted on the site's
  keys), and choosing the site's models left nothing in session storage; only
  `POST /api/ner` carried the key, and it was in no URL, body, document or console
  line.

Not run, in addition to the list above: Python 3.10 (the changed files parse with the
3.10 grammar and compile on 3.11 and 3.12; the suite ran on 3.13 only); Firefox and
Safari; a real proxy in front of the tool; any SDK version but 2.10.0.

### Analytics, 18 November 2026

Built and then fixed after a review by another model (the points are PLAN.md 16.4 to
16.10, marked **[review]**). **Nothing was deployed, no production variable was
changed, and no request was made to Vercel, to Neon or to https://www.xwalk.si by the
sessions that wrote this.** "Analytics: traffic and usage" above describes the code
after the fixes.

What was run (Python 3.13.3, the repository venv, no key in the environment):

- `python -m pytest -q -m 'not integration' tests/test_site_*.py`: 1765 passed, 34
  deselected (1431 before the fixes). The fixes were written test first. Reproduced
  before the code changed: four requests in a row against a database that never
  answers gave four `usage: "TimeoutError"` lines and left four live worker threads;
  a wrong-scheme or malformed `DATABASE_URL` on a deployment that only analytics made
  read it was a settings error or an uncaught `ValueError`; an upstream model name
  was stored as it came; a run that failed after a paid call had no tokens; and the
  own-key probes of the fourth round listed under "Fixes after the final review".
- With recording on: a database that hangs (a fake that holds every statement) across
  sequential and concurrent requests left every answer as it is without analytics,
  held at most four writes, skipped the rest at once, and recorded again once it
  answered; the store's shared lock held by another thread cost a request no more
  than the same wait.
- After further reviews, the write's client. A client of the site's own with a
  wall-clock deadline was written and, three review rounds later, removed again:
  each round found new faults in it, and a hand-written database client on the
  request path was judged a greater risk than what it solved. Recording and the
  statistics read use the engine's Neon client, as the visitor allowance does. What
  that leaves is tested as it is documented above: against a socket on 127.0.0.1 that
  accepts and never answers, each write ended by the client's timeout and every
  place came back; against one that answers a byte at a time, four writes kept their
  places six timeouts later, every answer was the same as without analytics, the
  following runs logged `saturated` at once, and the places came back when the
  connections ended.
- With the store's lock held for longer than the deadline and four requests waiting,
  every place was free within one second of the last attempt's own deadline (watched
  while the requests ran) and no write went out when the lock was released; a write
  whose turn came after its deadline did not start.
- A Neon answer that echoes the database URL in its error code, in a body that is
  not JSON, in an error page or in the reason phrase, and a connection string the
  client cannot send: the log lines hold the error's class and a SQLSTATE of the
  right form, and nothing of the URL, also with the HTTP library's own logging at
  DEBUG (it logs the reason phrase; its loggers are now silent during the limiter's
  and the record's database exchanges, as they were during own-key exchanges). That
  holds for every logger under `httpx` and `httpcore` that exists when an exchange
  begins, also with `HTTPS_PROXY` set, where `httpcore.proxy` logs the host it is
  asked to reach (tested through a local socket standing in for a proxy).
- A reply to the statistics statement that is right in shape and carries something
  else in one value (the database host as a day; 42 more cases, one or more for each
  kind of field) is 503 `stats_unavailable` and nothing of it is returned.
- A failed run that had stepped down (a paid Opus refusal, then a Sonnet failure) is
  priced as Opus; a row with strings of its own in it adds none to `/api/stats`;
  a dropped table is created again after one lost row; a `DATABASE_URL` with a port
  that is no number is not kept where only analytics reads it.
- `python -m pytest -q -m integration tests/test_site_usage.py`: 11 passed (33 with
  the limiter's), against a
  local PostgreSQL 16.15 on 127.0.0.1 through `psql`, in throwaway schemas that were
  dropped (none left), also with `PGTZ=Europe/Ljubljana` and
  `PGTZ=America/Los_Angeles`. These run the real statements, including the changed
  spend query and `SiteRuns.create`, which is what `tools/setup_usage.py` sends.
- `ruff check` and `ruff format --check` on `src tests scripts ner-site`: passing.
- The stats page in headless Chromium against `tools/serve_local.py --canned` on
  127.0.0.1, under the production Content-Security-Policy, at 1280 and 390 px, light
  and dark: a wrong passcode stays in the field and is not kept; an accepted one is
  kept for the tab and the field is empty afterwards; the period selector, Show and a
  reload send the kept value in `X-Stats-Passcode` and nowhere else, and it is in no
  URL, other header, document or input value; a kept passcode the server refuses is
  dropped; the legend says "No tier", the headline "had a step-down note"; an answer
  with all four tier counts zero draws every bar as "No tier"; no page error, no CSP
  violation, no inline style, no horizontal page scroll. An earlier session checked
  the main page with `/_vercel/insights/script.js` answering 404.

Not run or not verified:

- Vercel Web Analytics itself: the real `/_vercel/insights/script.js`, the request to
  `/_vercel/insights/view`, the dashboard. The `/stats` rewrite on Vercel.
- Neon's HTTPS endpoint with these statements: that it takes the one text parameter
  of the insert for `$1::jsonb` and returns the statistics column as text. The
  client is the one the visitor allowance already uses against Neon in production.
  `tools/setup_usage.py` was run against a fake executor and, through
  `SiteRuns.create`, on the local Postgres; never against Neon.
- What Anthropic's responses put in `model` for `claude-opus-5-5` and
  `claude-sonnet-5-5`. The id as requested and a dated or versioned form of it are
  both stored as the id; any other spelling would be `other` and unpriced (rollout
  check 4).
- The real `/api/stats` in a browser (the page ran against canned answers only),
  Firefox and Safari, a screen reader, a password manager.
- Python 3.10 (no interpreter on the machine; the code uses nothing newer and ruff
  targets it). A Vercel Firewall rule on `/api/stats`.
- Two timing tests (`test_one_deadline_bounds_the_whole_mapping_step`,
  `test_a_refusal_belongs_to_the_call_that_was_refused`) failed now and then in full
  runs while the machine was under heavy unrelated load (a load average of 25 to 50)
  and passed when run again; both fail the same way on the commit before this work
  (the second 3 times in 12 there).
- A connection the server closes just as a write is sent on it (an idle keep-alive
  connection timing out) is a lost row with `usage: "PostgresError"`: nothing is
  retried, because an insert is not idempotent.

### For the operator after deploying

1. `GET /api/health`: `mode: live`, `llm: {"status": "ok"}`, `dense: ok`. Without
   `OPENROUTER_API_KEY` health still answers 200 with `llm` unavailable and says why;
   `POST /api/ner` then answers 503 `not_configured`, unless `ANTHROPIC_API_KEY` is set,
   in which case only the Claude models are offered. With the Anthropic key also
   `claude: {"status": "ok"}` and `visitor.limits: "on"`; `visitor.limits:
   "unavailable"` there means every request on the site's keys answers 503
   `limits_unavailable`. `own_keys` says whether visitors' own keys are on, and for
   which providers and models: `{"enabled": false, "providers": []}` on every
   deployment where `NER_SITE_OWN_PROVIDERS` lists none, and such a deployment
   behaves as it did before own keys existed.
2. That Vercel passes `self.path` as `/api/ner` and `/api/health` (with any query
   string): anything else gets a 404 `not_found` from `handle`. Also that a POST from the
   page carries `Content-Length` (fetch does; a missing one gets 411 `length_required`).
3. The example run on the deployed page: entities, highlighted spans, matches naming
   both `bm25` and `dense-qwen3`, and `usage` with real token counts.
4. The function log: one JSON line per request (`"event": "ner_site.request"`) with
   sizes, counts, timings, status and model, and no text, prompt or key. Where Claude
   is offered it also has `tier`, `fallbacks` and `ran` (the model that extracted), and
   `limits` when the limiter could not be asked; never the visitor id or the address.
   A `POST /api/ner` also has `provider` and `credential` (`site` or `visitor`), and
   for a visitor's own key `tier` is `own`; no part of that key is in the line. A
   failed own-key request names the class of what failed in `own_error`.
   Also note the cold-start time (the first request builds four BM25 indexes in
   `/tmp`).
5. With `NER_SITE_PASSCODE` set: 401 `passcode_required` without it, 401
   `passcode_invalid` with a wrong one. Use an ASCII passcode (browsers reject other
   characters in a header) of 8 characters or more.
6. Query embeddings always go to OpenRouter's default embeddings URL with
   `OPENROUTER_API_KEY`; `NER_SITE_BASE_URL` changes only the chat endpoint.
