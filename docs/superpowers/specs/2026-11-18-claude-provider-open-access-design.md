# Claude as the default provider, and open access with per-visitor limits

Date: 18 November 2026. Scope: `ner-site/` only (the Vercel project behind
https://www.xwalk.si). The workbench (`src/`, `web/`) is not touched.

## 1. Goal

Visitors use the operator's Anthropic API key by default, without a passcode:

- Extraction and term selection both default to Claude Opus 5.5 through the Anthropic API.
- When Opus cannot be used, the request steps down to Claude Sonnet 5.5, and only then to
  an OpenRouter model. The visitor is always told which model ran and why.
- There is no dollar budget in the site. The operator's prepaid Anthropic balance (about
  $180, auto-reload off) is the hard ceiling.
- A per-visitor hourly allowance, stored in Neon, keeps one visitor from spending it all.

Decisions taken with the operator on 18 November 2026:

| # | Decision |
|---|---|
| 1 | Claude is reached with a direct Anthropic key (`ANTHROPIC_API_KEY`), not through OpenRouter. OpenRouter stays for the other models, Jev and query embeddings. |
| 2 | Claude is the default for both extraction and selection. Model: `claude-opus-5-5`. |
| 3 | The site is open: no passcode in production. |
| 4 | Limits are counters in the existing Neon database. No dollar budget. |
| 5 | Chain: Opus 5.5, then Sonnet 5.5, then OpenRouter as the last resort. Step-down triggers: Anthropic rate limit or overload, and the visitor's hourly Opus allowance. |
| 6 | Allowance: 10 requests an hour starting on Opus, then 20 starting on Sonnet, then 429 until the hour rolls over. |
| 7 | Haiku 5.5 is not in the chain (unreliable for extraction on 4 November 2026). |

## 2. Non-goals

- No change to retrieval, mapping, Jev, the ontologies or the Neon index tables.
- No streaming, no prompt caching, no cost computation in dollars (the Anthropic Console
  shows spend).
- No deployment, no change to production variables and no live calls as part of the
  implementation: the operator does the rollout in section 10.

## 3. Architecture

Three new modules in `ner-site/ner_site/`, each with one job, and small changes to the
existing ones. Extraction (`extract.py`) and the engine's selection stage keep talking to
an `xwalk.llm.base.LLMClient`; nothing in them learns about providers.

```
http.py  ── admit(visitor) ──>  limits.py   (Neon counters -> starting tier or 429)
   │
   ├─ build chain ─────────────> fallback.py (ordered steps, sticky step-down, notices)
   │                                 ├─ claude.py  ClaudeClient(claude-opus-5-5)
   │                                 ├─ claude.py  ClaudeClient(claude-sonnet-5-5)
   │                                 └─ service.make_llm(OpenRouter fallback model)
   └─ service.run(llm=chain, select_llm=chain)
```

### 3.1 `claude.py`: the Anthropic adapter

`ClaudeClient` implements `LLMClient` (`model`, `fingerprint`, `capabilities`,
`complete`, plus `aclose` and `generation_defaults`) with the official `anthropic` SDK
(`AsyncAnthropic`). It never uses an OpenAI-compatible endpoint and never raw HTTP.

- Constructor: `ClaudeClient(model, *, api_key, max_tokens, effort, timeout, client=None)`.
  `client` is an injected SDK-shaped object for tests; no test touches the network.
- One `complete(request)` is one Messages request. SDK `max_retries=0`: retrying and
  stepping down are the chain's job, so nothing is retried out of sight of the budget.
- Request shape, verified against the claude-api skill docs before coding (SDK names are
  never guessed):
  - `system` and one user message from `LLMRequest.system` / `.user`.
  - No `temperature`, `top_p`, `top_k` or seed: Opus 5.5 answers 400 to them.
    `LLMRequest.temperature` and `.seed` are ignored; `capabilities.seed` is false.
  - No `thinking` parameter (thinking cannot be disabled on Opus 5.5 and is adaptive by
    default). `output_config.effort` is the only depth control, default `low`
    (`NER_SITE_CLAUDE_EFFORT`, one of low, medium, high).
  - `max_tokens` is large because thinking counts against it: 16000 for extraction, 8000
    for selection. `LLMRequest.max_tokens` (the engine's 2048 / 600) is ignored for the
    same reason; `generation_defaults` reports the adapter's own value.
  - When `LLMRequest.schema` is set, schema-enforced output through
    `output_config.format` (`json_schema`). If the engine's schema uses a feature the API
    rejects, the adapter sends a normalised copy (for example `additionalProperties:
    false`, all properties required); the caller still validates the answer, as today.
  - Server-side refusal fallback is on (`fallbacks: "default"` with beta
    `server-side-fallback-2026-07-01`): Opus 5.5 has a `bio` classifier, and the site is
    fed biomedical abstracts. If the API rejects that combination with structured
    output, structured output wins and refusals are handled by the chain alone.
- Response handling:
  - `stop_reason` is checked before `content`. `refusal` raises `ClaudeRefused`
    (a `LLMFatalError`). `max_tokens` returns the text with `finish_reason="length"`.
  - Text is the concatenation of `text` blocks; thinking blocks are ignored.
  - Usage through `reported_usage(input + cache tokens, output_tokens)`.
  - `LLMResponse.model` is the model the API says answered (it differs after a
    server-side fallback).
- Errors, most specific first, mapped onto the engine's classes with a `reason`
  attribute the chain reads:

  | SDK error | Raised | `reason` |
  |---|---|---|
  | `RateLimitError` (429) | `LLMRetryableError` | `rate_limited` |
  | 529 `overloaded_error` | `LLMRetryableError` | `overloaded` |
  | 402 `billing_error`, or 400 naming the credit balance | `LLMFatalError` | `balance_exhausted` |
  | `AuthenticationError`, `PermissionDeniedError` | `LLMFatalError` | `key_refused` |
  | other 5xx, `APIConnectionError`, `APITimeoutError` | `LLMRetryableError` | `transient` |
  | other 4xx | `LLMFatalError` | `bad_request` |

  Messages never contain the key.

### 3.2 `fallback.py`: the chain

`FallbackLLM(steps, *, start=0)` is an `LLMClient` over an ordered list of
`Step(label, client)`.

- `complete` tries the current step. On failure it either retries or steps down:
  - `rate_limited`, `overloaded`, `balance_exhausted`, `key_refused`, `ClaudeRefused`,
    `CallLimitExceeded`: step down immediately.
  - `transient`: one retry on the same step, then step down.
  - `bad_request`, `ParseError` and anything else: raised to the caller unchanged (a
    malformed request would fail on every step).
  - The last step's error is raised as is.
- Step-down is sticky and shared: once the chain has moved to step n, every later call
  in the request starts at n. Concurrent selection calls already in flight finish where
  they are.
- Every step-down is recorded once as `Fallback(from_model, to_model, reason)`;
  `fallbacks` lists them in order.
- The returned `LLMResponse.usage` includes the usage of the failed attempts
  (`failure_usage`), so no dispatched call is free.
- `view()` returns an `LLMClient` that delegates to the chain and remembers
  `answered_by`, the model of its last successful answer. Selection creates one view per
  entity so each pick can name the model that made it, even under concurrency.
- `model` is the current step's model. `aclose` closes every step.
- `balance_exhausted` and `key_refused` are remembered in a module-level cache for 300
  seconds (`claude_block()` / `reset()`), so later requests on a warm instance start at
  the OpenRouter step and `/api/health` reports Claude as unavailable with the reason.

One chain object serves a whole request: extraction and selection share it, so a
rate limit met while extracting also moves selection down.

### 3.3 `limits.py`: the visitor allowance

- Visitor id: `HMAC-SHA256(salt, ip)`, hex, first 32 characters. `ip` is the first
  usable of `x-real-ip`, `x-vercel-forwarded-for`, `x-forwarded-for` (first entry);
  none means the id of the empty string (all such requests share one allowance). The
  salt is `NER_SITE_VISITOR_SALT`, or when unset the SHA-256 of `DATABASE_URL`. Raw IPs
  are never stored or logged.
- Storage: one table in the configured Neon schema, created with
  `CREATE TABLE IF NOT EXISTS` once per warm instance:
  `site_visits(visitor text, hour timestamptz, tier text, count integer,
  primary key (visitor, hour, tier))`. `hour` is the UTC clock hour. Queried through the
  same executor as the indexes (`indexes.make_executor`). Rows older than 24 hours are
  deleted opportunistically (about one request in fifty).
- `admit(visitor, wanted) -> Admission(tier, opus_left, sonnet_left, resets_in_s,
  reason)` where `wanted` and `tier` are `"opus"`, `"sonnet"` or `"other"`:
  - Reads the visitor's counts for the hour and records the visit under the tier it
    starts in, in as few round trips as practical (one statement preferred).
  - Opus is granted while fewer than `opus_per_hour` Opus visits are recorded; then
    Sonnet while fewer than `sonnet_per_hour` Sonnet visits are recorded. A request for
    an OpenRouter model is recorded as `other`.
  - When the hour's total reaches `opus_per_hour + sonnet_per_hour`, `admit` raises
    `VisitorLimitReached(resets_in_s)`; nothing is recorded.
- `peek(visitor)` returns the same numbers without recording (for `/api/health`).
- Limiter states:
  - **on**: `DATABASE_URL` is set. Behaves as above.
  - **unreachable**: the query fails. The request starts on Sonnet, uncounted, with the
    reason "visitor limits unavailable".
  - **off**: no `DATABASE_URL`. With a passcode set (a private deployment) Claude is
    unmetered and starts on Opus; without one, requests start on Sonnet, as for
    unreachable. Offline mode never consults the limiter.
- Settings: `NER_SITE_OPUS_PER_HOUR` (default 10) and `NER_SITE_SONNET_PER_HOUR`
  (default 20), whole numbers 0..1000, refused when out of range like the other limits.

## 4. Settings and model lists

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | The operator's Anthropic key. Scrubbed from every response and log line like the other secrets. |
| `NER_SITE_CLAUDE_EFFORT` | `low` | `output_config.effort` for every Claude call. |
| `NER_SITE_FALLBACK_MODEL` | `google/gemini-3.1-flash-lite` | The OpenRouter model at the end of the chain, for extraction and selection. |
| `NER_SITE_OPUS_PER_HOUR` / `NER_SITE_SONNET_PER_HOUR` | 10 / 20 | Per-visitor allowance. |
| `NER_SITE_VISITOR_SALT` | derived | Salt for visitor ids. |

- A model id starting with `claude-` is an Anthropic model; anything else is an
  OpenRouter model. The Claude ids the site knows are `claude-opus-5-5` and
  `claude-sonnet-5-5`; any other `claude-` id in a list is a settings error.
- Code defaults change so that Claude leads:
  `DEFAULT_MODELS = (claude-opus-5-5, claude-sonnet-5-5, google/gemini-3.1-flash-lite,
  openai/gpt-6-luna)`,
  `DEFAULT_SELECTORS = (claude-opus-5-5, claude-sonnet-5-5, jev,
  google/gemini-3.1-flash-lite, openai/gpt-6-luna)`, default selector `claude-opus-5-5`.
  The stale OpenRouter id `anthropic/claude-haiku-5.5` leaves the default selectors, and
  the `anthropic-compat` profile stays only for operators who list such an id themselves.
- Without `ANTHROPIC_API_KEY` the Claude ids are not offered: `offered_models` and
  `offered_selectors` drop them and the defaults become the first remaining entries
  (Gemini; Jev). Without `OPENROUTER_API_KEY` the OpenRouter ids and Jev are not
  offered and the chain ends at Sonnet. `configured` is true with either key.
- `NER_SITE_PASSCODE` keeps working unchanged for private deployments.

## 5. Request flow (`http.py`, `service.py`)

1. Existing checks: path, method, body size, `X-Requested-With`, passcode (if set),
   configuration, JSON, `validate_request`. A request refused here is not counted.
2. The wanted tier comes from the validated model: `claude-opus-5-5` is `opus`,
   `claude-sonnet-5-5` is `sonnet`, anything else `other`. When the model is not Claude
   but the selector is, the tier comes from the selector.
3. `limits.admit` gives the starting tier, or 429 `visitor_limit_reached` with a
   `Retry-After` header and the message "you have used this hour's requests; try again
   in N minutes".
4. `claude_block()` may lower the start to the OpenRouter step.
5. The chain is built from the wanted model down: for Opus
   `[opus, sonnet, fallback model]`, for Sonnet `[sonnet, fallback model]`, started at
   the admitted step. Each Claude step is wrapped in its own `BudgetedLLM`; the
   OpenRouter step is `make_llm` / `make_select_llm` as today. A request for an
   OpenRouter model has no chain and behaves exactly as today.
6. Selection:
   - Selector `retrieval` and `jev` are unchanged.
   - A Claude selector uses the request's chain when the extraction model was Claude
     too, otherwise a chain of its own with the same rules.
   - `select._with_llm` takes a `view()` per entity when the client offers one and
     writes `answered_by` into `Pick.selector`.
7. The response (section 6) and the log line carry the tier, the fallbacks and the
   models that ran. The visitor id is not logged.

Timeouts are unchanged: one Claude request has the existing 60 s timeout, extraction 90 s
overall, the request 100 s. Selection keeps its concurrency of 8.

## 6. Wire contract changes

`POST /api/ner` response, additions only:

- `model`: the model that produced the extraction (it may differ from the request).
- `requested_model`, `requested_selector`: what was asked for, defaults resolved.
- `fallbacks`: `[{"stage": "extract" | "select", "from": "claude-opus-5-5",
  "to": "claude-sonnet-5-5", "reason": "rate_limited"}]`, empty when nothing stepped down.
  Admission step-downs appear with stage `"admit"` and reason `opus_allowance_used`,
  `limits_unavailable` or `claude_unavailable`.
- `notices`: sentences for the visitor, one per distinct fallback, for example
  "Your Opus allowance for this hour is used; this ran on Claude Sonnet 5.5." and
  "Claude is unavailable (balance exhausted); this ran on Gemini 3.1 Flash Lite through
  OpenRouter."
- `entities[].pick.selector`: the model that made that pick.

New error: 429 `visitor_limit_reached`.

`GET /api/health`, additions only:

- `claude`: `{"status": "ok"}` or `{"status": "unavailable", "reason": "..."}` (key not
  set, key refused, balance exhausted). The key is checked with a free Models API call,
  cached like the OpenRouter check.
- `visitor`: `{"limits": "on" | "off" | "unavailable", "tier": "opus" | "sonnet" |
  "blocked", "opus_left": 7, "sonnet_left": 20, "resets_in_s": 1380}` for the calling
  visitor.
- `llm` stays the OpenRouter key's status. `passcode_required` is unchanged.

## 7. The page (`public/`)

- Model and selector options show names: "Claude Opus 5.5", "Claude Sonnet 5.5"; other
  ids as today.
- `notices` render in a dedicated, visible block above the results, distinct from the
  warnings list. The results header names the model that actually ran.
- Before submitting, a short line under the model field reflects `health.visitor`:
  nothing while on Opus; "Your Opus allowance for this hour is used; requests run on
  Claude Sonnet 5.5." on Sonnet; "You have used this hour's requests; try again in N
  minutes." when blocked. Health is re-read after each request.
- `visitor_limit_reached` shows its message like other errors.
- A footer sentence: "Requests are limited per visitor using a salted hash of your IP
  address, kept for one day."
- Everything is still inserted with `textContent`; the CSP is unchanged. The passcode
  field already hides itself when no passcode is required.
- `tools/fixtures/` gain canned health and response files for the Sonnet and blocked
  states so the page can be worked on without the service.

## 8. Error handling summary

| Situation | Outcome |
|---|---|
| Opus rate-limited or overloaded mid-request | Remaining calls run on Sonnet; fallback and notice recorded. |
| Sonnet rate-limited too | Remaining calls run on the OpenRouter fallback model. |
| Anthropic balance exhausted or key refused | OpenRouter fallback model; remembered for 300 s; health says why. |
| Claude declines (refusal survives the server-side fallback) | Next step in the chain; reason `refused`. |
| Every step fails during extraction | 502 `provider_error` as today, naming the last error, scrubbed. |
| A step fails during selection with no step left | That pick is `failed`, as today; never a 5xx. |
| Visitor over the hourly total | 429 `visitor_limit_reached` with `Retry-After`. |
| Neon unreachable for the limiter | Sonnet, uncounted, with a notice. |

Nothing substitutes silently: every departure from the requested model is in `fallbacks`
and `notices`.

## 9. Testing

Offline only, as now: no key, no network. New files `tests/test_site_claude.py`,
`tests/test_site_fallback.py`, `tests/test_site_limits.py`; additions to the HTTP,
service, select and schema tests.

- Adapter: request body (no sampling parameters, effort, max tokens, schema format,
  fallbacks), text and usage extraction, refusal, `max_tokens`, every row of the error
  table, key never in a message. With a fake SDK client.
- Chain: each trigger steps down; transient retries once; sticky across concurrent
  calls; usage of failed attempts counted; `view().answered_by`; the 300 s block.
- Limiter: tiers at the boundaries (10th and 11th, 30th and 31st visit), hour rollover,
  `other` visits, unreachable and off states, visitor id from each header, no raw IP in
  any SQL parameter. With a fake executor.
- HTTP: Opus by default; Sonnet after the allowance with the notice; 429 with
  `Retry-After`; health `claude` and `visitor`; no Claude without the key; passcode
  deployments unchanged.
- `ruff check` and `ruff format --check` pass. The existing suite keeps passing.

`anthropic` is added to `ner-site/requirements.txt` (what Vercel installs) and to the
`dev` extra in `pyproject.toml` (what CI installs).

## 10. Rollout (operator, after the code is merged)

1. In the Anthropic Console: a workspace for the site, a key in it, auto-reload off.
2. On Vercel (production): `ANTHROPIC_API_KEY` (sensitive); `NER_SITE_MODELS`,
   `NER_SITE_SELECTORS`, `NER_SITE_SELECTOR` removed or set to the new defaults. Keep
   `NER_SITE_PASSCODE` for now.
3. Deploy. `/api/health`: `claude: ok`, `visitor.limits: on`.
4. Measure with `ner-site/tools/claude_check.py` (new: runs given texts through the
   deployed site or a local one and prints model, fallbacks, timings and tokens per
   request) on the 12 real-world texts of 4 November. Two things decide the next step:
   a 25-entity request must finish inside the 100 s deadline, and the tokens per
   request give the cost. If selection on Opus is too slow, set `NER_SITE_SELECTOR=jev`
   and keep Opus for extraction.
5. Remove `NER_SITE_PASSCODE`, redeploy, check the 11th request in an hour steps down
   to Sonnet with the notice and the 31st answers 429.
6. Record what was verified in `ner-site/README.md`, as for earlier deployments.

## 11. Documentation

`ner-site/README.md`: the variables table, the deploy steps and a section for this
change. `docs/ner-site/PLAN.md`: a new section 14 with the decisions above and the wire
contract additions.

## Amendments after review, 18 November 2026

The final review of the implementation changed six things; the code follows these, and
the sections named are to be read with them. As built, with reasons:
`docs/ner-site/PLAN.md` section 14 (14.17 to 14.24).

- **A contended lock is a 429 (section 3.3).** The `admit` transaction bounds its wait
  for the visitor's lock at 1500 ms, and a request whose turn does not come in that
  time is answered 429 `visitor_limit_reached` ("too many requests at once; try again
  in a few seconds", `Retry-After: 5`) with nothing recorded, instead of running
  uncounted on Sonnet; every other limiter failure keeps the behaviour of section 3.3.
- **A timeout steps down (sections 3.1, 3.2 and 5).** `anthropic.APITimeoutError` is the
  reason `timeout`, not `transient`: the chain leaves the step at once without trying
  the same model again, and a Claude extraction call has a 40 s timeout (not 60 s) so
  that Opus and then Sonnet both fit inside the 90 s extraction bound.
- **A balance or key failure skips the other Claude step (section 3.2).** On
  `balance_exhausted` or `key_refused` the chain moves straight to its OpenRouter step
  (its last step when it has none) and records one fallback, because Opus and Sonnet
  share the key and the balance.
- **A server-side refusal fallback is recorded (sections 3.1 and 6).** When a Claude
  step's answer names a model that is not a step of the chain, the response carries a
  `fallbacks` entry (stage `extract` or `select`, from the step's model to the answering
  model, reason `refused`, once per stage and model) and its notice.
- **`DATABASE_URL` is read only where it is used (section 4).** It is read and validated
  with `NER_SITE_BACKEND=neon`, or where `ANTHROPIC_API_KEY` is set and the site is not
  offline; anywhere else it is ignored, as it was before this change.
- **A failed key probe is not an outage (section 6).** When the Anthropic key check
  could not be completed (unreachable, or an HTTP error other than 401 or 403), health
  reports `claude` as `ok` with a reason saying the key was not checked; only a missing
  key, a refused key or a remembered block is `unavailable`.
