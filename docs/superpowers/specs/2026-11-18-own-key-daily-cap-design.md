# A daily cap, and visitors' own API keys

Date: 18 November 2026. Scope: `ner-site/` only. Builds on
`2026-11-18-claude-provider-open-access-design.md` (including its amendments), which this
document changes in the places it says so. Design input: a read-only review by GPT-6.2
Astra (Codex CLI) of the code as built.

## 1. Goal and decisions

Bound what one address can spend of the operator's balance per day, and let a visitor who
wants more (or who arrives during an outage) run the demo on their own provider key.

| # | Decision (operator, 18 November 2026) |
|---|---|
| 1 | Per-visitor daily cap on top of the hourly allowance: 20 requests a day starting on Opus, 40 starting on Sonnet. A day is the UTC calendar day. |
| 2 | A visitor may use their own API key for Anthropic, OpenAI or OpenRouter. |
| 3 | The key lives in the browser only (session storage), travels with each request, is used for that request, and is never stored, logged or echoed by the server. |
| 4 | An own-key request does not count against the Opus or Sonnet allowance. It has its own flood limit: 30 an hour and 60 a day per address. |
| 5 | With an own key, terms are picked by a model of the same provider on the same key, or by retrieval alone. Jev is not offered (it bills the operator). |
| 6 | Model ids come from server-side lists per provider, changeable by environment variable. A visitor never names a base URL or a free-form model. |
| 7 | When the limiter cannot be consulted, requests on the operator's keys are refused (503), not run uncounted. Own-key requests keep working. This replaces "Sonnet, uncounted" in the earlier spec. |
| 8 | "Sign in with ChatGPT" is not built: OpenAI's programme covers loopback callbacks only and hosted apps must apply. The credential seam is shaped so it can be added as one more provider. |

## 2. Non-goals

- No ChatGPT OAuth, no server-side storage of any visitor credential, no cookies.
- No change to retrieval, Jev, the fallback chain's rules for site-key requests, or the
  wire fields that exist today.
- No deployment and no live calls as part of the implementation.

## 3. The limiter (`limits.py`)

### 3.1 Daily cap

- `Allowance` gains `opus_per_day = 20`, `sonnet_per_day = 40`, `own_per_hour = 30`,
  `own_per_day = 60`. Environment: `NER_SITE_OPUS_PER_DAY`, `NER_SITE_SONNET_PER_DAY`,
  `NER_SITE_OWN_PER_HOUR`, `NER_SITE_OWN_PER_DAY` (whole numbers 0..10000, refused out
  of range like the others).
- The table is unchanged (hour buckets). The visit statement sums the visitor's rows of
  the request's whole UTC day (from its midnight to the next), giving hourly and daily
  counts per tier in the same statement, still under the advisory lock, the lock timeout
  and one round trip. The day, not the request's own hour, bounds the scan: a request
  stamped 14:59 that takes the lock after a 15:00 visit committed must see that visit,
  or the daily cap could be exceeded at every hour boundary (found by the implementer).
- Site tiers (`opus`, `sonnet`, `other`):
  - The hour's site total must be below `opus_per_hour + sonnet_per_hour` and the day's
    below `opus_per_day + sonnet_per_day`; `own` visits are in neither total.
  - Opus is granted while both its hourly and daily counts have room; otherwise Sonnet
    while both of its have room (reason `opus_allowance_used`); otherwise refusal.
  - Wanted `sonnet` and `other` follow the same two-window rule as today's one-window
    rule.
- Tier `own`: granted while the visitor's `own` counts are below `own_per_hour` and
  `own_per_day`; recorded as `own`.
- Pruning stays at 24 hours, which keeps every bucket of the current UTC day.
- `Admission` keeps its fields and gains `opus_left_today`, `sonnet_left_today`,
  `own_left`, `own_left_today`, `day_resets_in_s` and `window`
  (`"hour"` or `"day"`: which window bound the decision or would bind next).
  `opus_left` and `sonnet_left` become what can actually still start this hour, capped
  by the day's remainder, so existing readers stay truthful.
- `VisitorLimitReached` carries `resets_in_s` for the earliest reset that would admit
  the request, and `window`.

### 3.2 Messages

- Hour: "you have used this hour's requests; try again in N minutes, or use your own API
  key" (the own-key clause only where own keys are enabled).
- Day: "you have used today's requests; they reset at midnight UTC, or use your own API
  key". `Retry-After` is the seconds to that reset.
- Own-key flood limit: "too many requests with your own key; try again in N minutes".

## 4. When the limiter applies, and failing closed (`http.py`)

- `DATABASE_URL` is read when the backend is `neon`, or the site is not offline and
  either `ANTHROPIC_API_KEY` is set or own keys are enabled.
- Site-key request (provider `site`):
  - Claude offered: admitted through the limiter as today.
  - Limiter not configured (no `DATABASE_URL`): with a passcode, unmetered as today;
    without one, **503 `limits_unavailable`** ("visitor limits are not configured").
  - Limiter query fails: **503 `limits_unavailable`** ("the site cannot count requests
    right now; try again shortly, or use your own API key"), `Retry-After: 30`. The
    contended-lock 429 is unchanged.
  - Claude not offered (OpenRouter-only deployment): no limiter, exactly as before.
- Own-key request:
  - Admitted through the limiter as tier `own` when it is configured.
  - Limiter not configured or failing: the request runs, uncounted (decision 7).
- The earlier fallback notes with reason `limits_unavailable` no longer occur; the reason
  string stays in the vocabulary for old pages.
- Health `visitor.tier` becomes `blocked` and `visitor.limits` `unavailable` in the
  failing state, so the page can say so before the visitor submits.

## 5. Credentials and providers (new `providers.py`)

One request flow, with the credential as data:

```python
SITE = "site"
OWN_PROVIDERS = ("anthropic", "openai", "openrouter")

@dataclass(frozen=True)
class Credential:
    provider: str                      # one of OWN_PROVIDERS today; "chatgpt" later
    token: Callable[[], str]           # called at dispatch time; repr never shows it
    source: Literal["visitor"] = "visitor"

@dataclass(frozen=True)
class ProviderInfo:
    id: str; name: str                 # "anthropic", "Anthropic"
    models: tuple[str, ...]; default_model: str

def read_credential(provider: str | None, headers: Mapping[str, str],
                    settings: Settings) -> Credential | None   # None means the site's keys
def own_llm(credential: Credential, model: str, *, purpose: Literal["extract", "select"],
            settings: Settings) -> LLMClient                    # budgeted, request-scoped
def provider_error(exc: BaseException, provider: str) -> ServiceError
```

- **Transport.** JSON body field `provider`: absent or `"site"`, or one of the own
  providers. An own provider requires exactly one `Authorization: Bearer <key>` header.
  Where own keys are on, a `site` request must not carry one; where they are off, the
  header is not looked at on a `site` request (amended after the final review: basic
  auth and authenticating proxies add one). Violations are 400 `bad_credential` with a fixed
  message: missing, duplicated or malformed header; a key outside 20..512 printable
  ASCII characters without spaces; a provider that is not enabled; the key also present
  in `text` or `prompt`. `/api/health` never receives or reads a credential.
- **Clients**, built per request and closed after it:
  - `anthropic`: `ClaudeClient` with the visitor's key, the same effort and token
    limits, and **no server-side refusal fallback** (a substitute model would bill the
    visitor at another price). `ClaudeClient` gains a `refusal_fallback: bool = True`
    parameter. Claude Haiku 5.5 (`claude-haiku-5-5`) is added to the ids the adapter
    accepts; it is offered only with an own key.
  - `openai`: the engine's `OpenAICompatClient` pinned to `https://api.openai.com/v1`.
  - `openrouter`: the site's OpenRouter client with the visitor's key.
  - Base URLs are constants. `NER_SITE_BASE_URL` does not apply. No redirects.
  - Each is wrapped in the same call budgets as today. No fallback chain, no key probe,
    no Claude block, nothing cached per key, and the key never enters `Settings`.
- **What still uses the operator's keys:** query embeddings for dense retrieval
  (OpenRouter) and the Neon queries. Nothing else.
- **Models.** `NER_SITE_OWN_MODELS_ANTHROPIC` (default `claude-opus-5-5,
  claude-sonnet-5-5,claude-haiku-5-5`), `NER_SITE_OWN_MODELS_OPENAI` (default
  `gpt-6-luna`), `NER_SITE_OWN_MODELS_OPENROUTER` (default
  `google/gemini-3.1-flash-lite,openai/gpt-6-luna`); the first is the default.
  `NER_SITE_OWN_PROVIDERS` lists the providers that are on. Own keys are off unless it
  lists at least one: unset or empty means off. (Amended after the final review; the
  first version of this section had all three on by default.) With an own provider
  the request's `model` must be in that provider's list, and `selector` is
  `retrieval` (the default) or a model in the same list.
- **Errors**, fixed text with the provider's display name and nothing from upstream:
  - 401 `provider_key_refused`: "<Provider> refused this key. It may be mistyped,
    revoked, out of credit, or not allowed to use this model."
  - 429 `provider_rate_limited`: "<Provider> is rate limiting this key; try again
    shortly."
  - 502 `provider_error`: "<Provider> could not complete the request (<reason>)." where
    reason is a short machine word (`timeout`, `overloaded`, `http_5xx`, `bad_request`,
    `refused`, `unreachable`).
  - A failed pick with an own key says the same short reason, never the exception text.
- **Redaction.** For the lifetime of the request the key is an extra secret for the
  response and log scrubbers (in addition to the adapter-level fixed messages), applied
  to values before JSON encoding. The log line gains `provider` and
  `credential: "visitor" | "site"`; never the key, its length or a digest.
- **ChatGPT later.** A `chatgpt` provider resolves an opaque session to a `Credential`
  whose `token()` refreshes and returns an access token, and `own_llm` returns the
  engine's `ChatGPTClient(model, token)`. Nothing in the request flow changes. The
  storage that needs (encrypted, shared across instances) is out of scope here.

## 6. Wire contract additions

`POST /api/ner` request: `provider` (optional). Response: `provider`
(`"site"` or the own provider) and `credential_source` (`"site"` or `"visitor"`).
`fallbacks` and `notices` are empty for own-key requests.

New errors: 400 `bad_credential`, 401 `provider_key_refused`, 429
`provider_rate_limited`, 503 `limits_unavailable`. `visitor_limit_reached` gains the day
and own-key wordings.

`GET /api/health` additions:

- `own_keys`: `{"enabled": true, "providers": [{"id": "anthropic", "name": "Anthropic",
  "models": [...], "default_model": "..."}, ...]}` (enabled false and no providers when
  off). Present whether or not the operator's own LLM keys are set.
- `visitor` gains `opus_left_today`, `sonnet_left_today`, `own_left`, `own_left_today`,
  `day_resets_in_s`, `window`.

## 7. The page

- A "Run with" choice above the model field: "The site's models (free, limited)",
  "My Anthropic key", "My OpenAI key", "My OpenRouter key" (only enabled providers; the
  whole control is absent when own keys are off or health does not report them).
- Choosing an own provider shows: a password field (no `name`, `autocomplete="off"`,
  `spellcheck="false"`), a "Forget key" button, the provider's models in the model
  field, and "Pick the match with" limited to "Retrieval only" and the provider's models.
- The key is kept in `sessionStorage` under one entry, cleared on "Forget key" and when
  the provider changes. It is attached only to the `POST /api/ner` call, as
  `Authorization: Bearer`, with `redirect: "error"`. Health requests carry nothing.
- Text shown with the field, verbatim:
  "Your key stays in this browser tab and is sent with each run to this site, which
  passes it to <Provider> for that run and does not store or log it. <Provider> bills
  you for the model calls. The site still pays for term lookup. Use a restricted key
  with a spending limit if you can."
- The allowance line reflects the chosen mode: the existing sentences for the site's
  models, extended for the daily cap ("You have used today's requests. They reset at
  midnight UTC, or you can use your own API key."); nothing for an own key until its
  flood limit is reached.
- When the site's limits are unavailable: "The site cannot count requests right now.
  Try again shortly, or use your own API key."
- Results name the provider when the key was the visitor's: "Extracted by <model> on
  your <Provider> key."
- Everything through `textContent`; CSP unchanged.

## 8. Also fixed here (carried from the last review)

- A server-side refusal fallback is recognised by the SDK's own signal (the
  `fallback_message` entry in `usage.iterations`), not by the answering model being
  outside the chain, so a fallback onto a chain model is not silent and a differently
  spelt echo of the requested id is not a false notice.
- After a server-side fallback only the serving model's text is used (text after the
  last `fallback` content block), so a declined model's partial text cannot corrupt the
  JSON.
- A select-stage server fallback's notice says which picks it concerns rather than
  taking the "remaining picks" wording.

## 9. Testing

Offline, as before. Limiter: mixed tiers across both windows, the 20th and 21st Opus
visit of a day across hours, UTC midnight, own-key limits independent of site limits,
`peek` parity, zero caps, the single transaction and lock order; the Postgres
integration tests extended the same way. Providers: header parsing (missing, duplicate,
malformed, oversized, key inside the text), each provider's client built with the
visitor's key and the pinned base URL, no site key used, no chain, no probe, no cache
entry, fixed error messages with upstream echoes of the key in the fake's errors,
concurrent requests with distinct sentinel keys never crossing. HTTP: fail-closed 503 on
limiter failure for site keys and success for own keys; own-key requests not counted as
site visits; health `own_keys`; an OpenRouter-only passcode deployment unchanged. Page:
fixtures for own-key health and results; provider switch clears the key; health requests
carry no credential.

## 10. Rollout checks (operator)

1. In Vercel, confirm request headers are not written to logs, log drains or error
   reporting (the key is in `Authorization`). Do not enable own keys until this is
   checked: they stay off while `NER_SITE_OWN_PROVIDERS` is unset. After deploying
   with the variable unset, health shows `own_keys.enabled: false` and the deployment
   behaves as before. Also confirm that `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_LOG`,
   `ANTHROPIC_BASE_URL` and proxy variables are unset (own-key requests ignore them,
   the site's own Claude client does not) and that nothing in front of production
   adds an `Authorization` header.
2. With a real key for each provider, one run each: the model answers, the result names
   the provider, and the operator's dashboards show no Claude or chat usage for it.
   OpenAI direct is untested offline in particular: confirm the default model id exists
   and returns JSON within its token limit.
3. Send a wrong key: 401 `provider_key_refused`, and no part of the key in the response
   or the function log.
4. After 20 Opus starts in a day, the next request starts on Sonnet; after 60 site
   requests in a day, 429 with the day wording.
5. Stop the limiter (a wrong `DATABASE_URL` in a preview deployment on the bundled
   backend with the sample catalog): site-key requests answer 503
   `limits_unavailable`, own-key requests still run. Restore the preview afterwards.
6. Submit OpenAI's interest form if ChatGPT sign-in on the hosted site is wanted.
7. Turn own keys on by setting `NER_SITE_OWN_PROVIDERS` to the providers that passed
   check 2, by name. Going back is removing the variable.
8. The `anthropic` SDK is pinned to the version the own-key client was audited
   against. On any change of that pin, repeat the audit and check 2 for Anthropic.

## 11. Documentation

`ner-site/README.md` (variables, the own-key section, rollout checks, the changed
limiter-outage behaviour), `docs/ner-site/PLAN.md` (a section 15), and an amendment note
in the earlier spec pointing here for decision 7.

## As built, 18 November 2026

Where the code departs from this document, or settles what it left open. The code
follows these; reasons are in `docs/ner-site/PLAN.md` section 15 (15.1 to 15.20). The
points marked "final review" were changed after two reviews of the whole branch.

- **The day is the visitor's whole UTC day (section 3.1).** The scan is `hour >=
  midnight AND hour < midnight + 24 hours`, not "through the current hour", and the
  bound is written as 24 hours so that a session time zone with a clock change cannot
  shorten or lengthen it.
- **Balances (section 3.1).** `opus_left`, `sonnet_left` and `own_left` are capped by
  the day's remainder. `resets_in_s` stays the hourly reset in `Admission` and in
  health; `day_resets_in_s` and `window` carry the day. Only `VisitorLimitReached`
  carries the reset of the window that binds. A refusal is for the day only when no
  tier the request could start on has room left today.
- **Allowance variables (section 3.1).** The four new ones take 0 to 10000; the two
  hourly site ones keep 0 to 1000.
- **Own-key errors (section 5).** The words are the six listed and `bad_answer`,
  `call_limit` and `error`; a failed pick can also say `key_refused` or
  `rate_limited`. An HTTP 404 from a provider is reported as the key being refused
  (401 `provider_key_refused`), as are 402, Anthropic's credit-balance 400 and
  OpenAI's 429 "exceeded your current quota". With an own key the request deadline
  and the call cap are 502 `provider_error`; an unreadable extraction answer stays
  422. `provider_rate_limited` carries no `Retry-After`.
- **Signatures (section 5).** `read_credential(provider, headers, *, enabled, text,
  prompt)` and `own_llm(credential, model, *, purpose, effort, max_entities, call_cap)`
  take no `Settings`; `Credential.source` is a `str`; `token()` is called once per
  stage, when the client is built.
- **OpenAI direct (section 5).** A subclass of the engine's client sends
  `max_completion_tokens` (16000 and 8000) and no `temperature` or
  `reasoning_effort`. Never run against the live API.
- **Own Anthropic keys (section 5).** No retry, no server-side refusal fallback, 60 s
  per call for both steps. The client is built so that nothing from the operator's
  environment reaches the request: `ANTHROPIC_CUSTOM_HEADERS` is stripped (through a
  private attribute of the SDK, with a check that refuses to build the client
  otherwise), and the SDK's other environment inputs (keys, profile, base URL,
  telemetry, proxies) do not reach the request. Final review: every request is also
  checked as it leaves (the host, the visitor's bearer once, and each header name
  and value against what SDK 2.10.0 sends on its own) and is not sent otherwise, and
  `anthropic` is pinned to 2.10.0, with a test that fails on any other version.
- **Own keys are off by default (section 5, final review).** Unset or empty
  `NER_SITE_OWN_PROVIDERS` means off; the operator turns them on by listing
  providers (any case, spaces and empty items ignored, unknown names refused). An
  own model list is read only for a provider that is on, and empty counts as unset.
- **A site request with an `Authorization` header (section 5, final review)** is
  refused with 400 `bad_credential` where own keys are on. Where they are off the
  header is not looked at on a site request, so that a deployment from before own
  keys (basic auth, an authenticating proxy) behaves as it did.
- **Redirects and echoing headers (section 5, final review).** "No redirects" was
  not enough: with redirects not followed the HTTP libraries still parse a
  redirect's `Location`, and a relative one that echoed the key stayed in Python's
  `urlsplit` cache after the request. Every own-key client now sends through a
  transport wrapper that replaces a 3xx response with a bare one, before the
  library reads it. Second round: searching the other headers for the key was not
  enough either (a charset can arrive encoded and is decoded by the library, and a
  `Set-Cookie` is stored and sent back), so every response gets a new header set:
  one of two fixed content types, and `content-length`, `content-encoding`,
  `retry-after` and `retry-after-ms` only in a strict form.
- **Own-path failures (section 5, final review).** Only a failure of the provider,
  the SDK, the HTTP library or a timeout is a provider error, also inside the
  clients themselves; any other exception is a fault of the site's code: 500
  `internal`, logged with its type and frames, during extraction, and a failed pick
  during selection. The log line names the class of what failed in either stage
  (`own_error`); a client that fails to close is logged (`own_close_error`) and
  does not replace the result.
- **Selector usage (section 6, final review).** `selector_calls` counts every call
  the request's call budgets dispatched during selection, also one the deadline
  cut off; the selector's tokens and cost are then null.
- **Where requests are metered (section 4).** On an OpenRouter-only deployment (no
  Anthropic key) site requests are still not metered, while own-key requests use the
  flood limit when a database is configured. Health has a `visitor` block only where
  Claude is offered. `DATABASE_URL` is read with the Neon backend, or on a
  deployment that is not offline and has the Anthropic key or lists own providers;
  one with only the OpenRouter key and no own providers does not read it, as before.
  Offline mode has no own keys. A contended lock is a 429 for own-key requests too.
- **Health without counters (section 4).** `own_left` and `own_left_today` are null
  and `window` is `hour`; with a passcode and no database, `limits` is `off` and the
  site numbers are the full allowance.
- **The page (section 7).** The session entry holds the provider with the key and a
  reload in the same tab restores both; the key's shape is checked before sending;
  the own-key limit and the day's Opus step-down have sentences the spec does not
  give (PLAN.md 15.15). Final review: Run is held until a saved own-key choice is
  restored; when it cannot be (health failed, or the provider is no longer offered)
  the page says so and uses the site's models only after the visitor chooses that,
  and the choice (the provider id, not the key) is kept so that the question is
  asked again after every reload until then;
  each health read reconciles the providers offered; the hour-blocked sentence
  names the own-key way out where own keys are on (PLAN.md 15.20).
- **The check tool (rollout).** `tools/claude_check.py --provider ... --key-env ...`;
  it sends a key only to a site whose health answered 200 with own keys enabled and
  the provider listed, and no secret over plain http to another machine or through
  a proxy to this one. Final review: a response that echoes a secret fails the run,
  and a burst passes only when the balances before and after account for its starts
  (PLAN.md 15.16).
