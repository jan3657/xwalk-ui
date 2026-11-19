# Own Keys and Daily Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-visitor daily cap to the NER site's allowance, let visitors run the demo on their own Anthropic, OpenAI or OpenRouter key, and refuse site-key requests when the limiter cannot count them.

**Architecture:** The limiter's single SQL statement gains a daily window and an `own` tier. A new `providers.py` turns a request's `provider` field and `Authorization` header into a request-scoped credential and budgeted client, with fixed error messages; `http.py` and `service.py` route own-key requests around the fallback chain and the operator's keys. The page gains a "Run with" choice and a key field kept in session storage.

**Tech Stack:** Python 3.10+, `anthropic` SDK 2.10 (installed in `.venv`), `xwalk-match` engine (source at `/Users/jandrole/projects/xwalk/src/xwalk`), pydantic, pytest (`asyncio_mode = auto`), ruff; plain JS/CSS page.

**Spec:** `docs/superpowers/specs/2026-11-18-own-key-daily-cap-design.md` is binding. It builds on `docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md` (read its "Amendments after review" too). Where this plan and the spec disagree the spec wins; report it.

**Implementer:** GPT-6.2 Astra through the Codex CLI, one task per run; each task is reviewed before the next.

## Global Constraints

- Work only under `ner-site/`, `tests/test_site_*.py`, `tests/conftest.py`, `docs/`. Do not touch `src/` or `web/`.
- Tests are offline: no real key, no real database, no call to any provider. Run from the repository root: `.venv/bin/python -m pytest -q -m 'not integration' tests/test_site_*.py` (533 passing before this plan; none may be lost without reason). `.venv/bin/ruff check src tests scripts ner-site` and `.venv/bin/ruff format --check src tests scripts ner-site` must pass (line length 100, target py310). Python 3.10 must work.
- A visitor's key must never appear in: a response body or header, a log line, an exception message that can reach either, SQL, `Settings`, any cache or module-level state, or a digest or length derived from it. Tests use sentinel keys and assert their absence.
- A visitor never chooses a base URL, a free-form model id or SDK options. Provider base URLs are constants.
- An own-key request never uses the operator's Anthropic key or the operator's OpenRouter key for a chat call, never uses the fallback chain, never touches the key-probe caches or the Claude block, and never asks Anthropic for a server-side fallback model.
- A deployment with only `OPENROUTER_API_KEY`, a passcode and explicit old model lists behaves exactly as before.
- Existing wire fields keep their names and meanings. New fields and codes are exactly those in spec section 6.
- Claude is called only through the official `anthropic` SDK; SDK names are checked against the installed package, never guessed. Never send sampling parameters or disable thinking.
- The page inserts everything with `textContent`; no inline styles or scripts (CSP `default-src 'self'`).
- User-facing sentences are the ones the spec gives, verbatim where it gives them. No em dashes in docs or user-facing text.
- Follow each file's existing style (docstrings that say why, `from __future__ import annotations`, `__all__`, the existing settings validation and error envelope). Do not restructure `service.py` or `http.py`. No new dependencies.
- No deployment, no production variable changes. Do not commit: the coordinator commits after review.

---

### Task 1: Daily window and the `own` tier in the limiter

**Files:**
- Modify: `ner-site/ner_site/limits.py`
- Test: `tests/test_site_limits.py` (offline tests with the in-memory fake, and the `integration`-marked Postgres tests)

**Interfaces:**
- Consumes: nothing new.
- Produces (spec section 3):

```python
Tier = Literal["opus", "sonnet", "other", "own"]

@dataclass(frozen=True)
class Allowance:
    opus_per_hour: int = 10
    sonnet_per_hour: int = 20
    opus_per_day: int = 20
    sonnet_per_day: int = 40
    own_per_hour: int = 30
    own_per_day: int = 60

@dataclass(frozen=True)
class Admission:            # existing fields unchanged in name
    tier: ...; opus_left: int; sonnet_left: int; resets_in_s: int; reason: str | None
    opus_left_today: int
    sonnet_left_today: int
    own_left: int
    own_left_today: int
    day_resets_in_s: int
    window: Literal["hour", "day"]

class VisitorLimitReached(Exception):
    resets_in_s: int        # earliest reset that would admit this request
    window: Literal["hour", "day"]
```

`VisitorLimits.admit(visitor, wanted)` accepts `"own"`; `peek(visitor)` reports all the new numbers. Signatures are otherwise unchanged; `http.py` is not edited in this task and must keep working (it constructs `Allowance` with the two hourly values and reads the existing `Admission` fields).

- [ ] **Step 1:** Read spec section 3, `limits.py` in full (the lock statement, the lock timeout, the visit statement, how the result row is picked, pruning) and `tests/test_site_limits.py` (the fake executor and the Postgres executor used by the integration tests).
- [ ] **Step 2:** Write failing offline tests, one behaviour each: the 20th Opus visit of a UTC day (spread over three hours) is Opus and the 21st is Sonnet with reason `opus_allowance_used`; the 60th site visit of the day is the last and the 61st raises with `window == "day"` and `resets_in_s` equal to the seconds to UTC midnight; an hourly refusal has `window == "hour"`; the hour's and the day's numbers after UTC midnight; `other` counts toward both site totals; `own` visits are granted up to 30 an hour and 60 a day, do not reduce any site number, and site visits do not reduce `own_left`; wanted `own` beyond its limits raises; `opus_left` is capped by the day's remainder; `peek` equals what the last `admit` returned and records nothing; zero caps for each new field; the batch is still one transaction with the lock timeout first, the lock second and the visit third.
- [ ] **Step 3:** Run `.venv/bin/python -m pytest -q tests/test_site_limits.py` and see them fail for the right reason.
- [ ] **Step 4:** Implement in the single visit statement (filtered sums over the visitor's rows from UTC midnight through the current hour), keeping every value a parameter, the advisory lock as its own statement, and one round trip. Update the fake to mirror it, and extend the integration tests with the same daily and `own` cases.
- [ ] **Step 5:** Run the limiter tests, then the full site suite and both ruff commands. If `psql` and a local Postgres are available to you, run the integration tests as the module docstring describes; if not, say so.
- [ ] **Step 6:** Report: the final SQL, what is and is not guaranteed under concurrency, test counts, whether the integration tests ran.

---

### Task 2: Credentials, provider clients and adapter changes

**Files:**
- Create: `ner-site/ner_site/providers.py`
- Modify: `ner-site/ner_site/claude.py`
- Test: `tests/test_site_providers.py` (new), `tests/test_site_claude.py`

**Interfaces:**
- Consumes: `ClaudeClient`, `xwalk.llm.openai_compat.OpenAICompatClient`, the site's OpenRouter client class and client-building helpers in `service.py` (read them; import what is importable without a cycle, otherwise take what is needed as parameters and say so), `xwalk.llm.budget.BudgetedLLM` / `CallBudget`, `ServiceError`.
- Produces (spec section 5; Task 3 relies on these names):

```python
SITE = "site"
OWN_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "openrouter")
PROVIDER_NAMES: dict[str, str] = {"anthropic": "Anthropic", "openai": "OpenAI", "openrouter": "OpenRouter"}
BASE_URLS: dict[str, str]                 # openai and openrouter; constants
KEY_MIN, KEY_MAX = 20, 512

@dataclass(frozen=True)
class Credential:
    provider: str
    token: Callable[[], str]              # repr and str never show the key
    source: str = "visitor"

def read_credential(provider: str | None, headers: Mapping[str, str | Sequence[str]], *,
                    enabled: Sequence[str], text: str, prompt: str | None) -> Credential | None
    # None for the site's keys; raises ServiceError(400, "bad_credential") per spec section 5
def own_llm(credential: Credential, model: str, *, purpose: Literal["extract", "select"],
            effort: str, max_entities: int, call_cap: int) -> LLMClient
def provider_error(exc: BaseException, provider: str) -> ServiceError
def failure_word(exc: BaseException) -> str   # the short machine word for a failed pick
```

In `claude.py`: `ClaudeClient(..., refusal_fallback: bool = True)`; `HAIKU = "claude-haiku-5-5"` accepted by the adapter and present in `DISPLAY_NAMES` ("Claude Haiku 5.5") but not in `CLAUDE_MODELS` (the site's own list is unchanged); the server-side fallback is detected from the SDK's `fallback_message` usage entry and exposed so the caller can tell (for example a `served_by_fallback` flag carried with the response's model; say exactly how in your report, Task 3 consumes it); after a fallback only text after the last `fallback` content block is returned (spec section 8).

- [ ] **Step 1:** Read spec sections 5 and 8, `claude.py`, the client builders and `scrub` in `service.py`, the engine's `llm/openai_compat.py` (how it builds requests, what its errors contain, how redirects and proxies are configured, which token-limit field it sends) and `llm/base.py`.
- [ ] **Step 2:** Write failing tests. `read_credential`: site and absent provider give None; a `site` request with an `Authorization` header is refused; each own provider with a good bearer key gives a credential whose repr, str and dataclass fields do not contain the key; missing, duplicated (a sequence of two values), non-bearer, too short, too long, non-ASCII, spaced keys are each 400 `bad_credential` with a message that does not contain the key; a provider not in `enabled` is refused; a key contained in `text` or `prompt` is refused. `own_llm`: for each provider the client is built with the visitor's key and the pinned base URL (assert on the request the fake transport receives: host, authorization header, model, no `fallbacks` and no beta flag for Anthropic), the call budget is enforced, and no module-level state changes. `provider_error` and `failure_word`: each failure class maps to the fixed message and code of spec section 5, with a fake upstream error that echoes the sentinel key in its text and the assertion that the result does not contain it. Two credentials used concurrently never cross (each fake transport sees only its own key). `claude.py`: `refusal_fallback=False` sends neither `fallbacks` nor the beta flag; Haiku is accepted; a response with a `fallback_message` iteration is flagged and names the serving model; text before a `fallback` content block is dropped.
- [ ] **Step 3:** See them fail; implement; run the two test files, the full site suite and ruff.
- [ ] **Step 4:** Report, including: how the engine's OpenAI client behaves against `api.openai.com` as far as can be told offline (token-limit parameter name, structured-output mode, anything that would need a live check), and how redirects and environment proxies are disabled.

---

### Task 3: Request flow, settings, wire contract, health

**Files:**
- Modify: `ner-site/ner_site/schema.py`, `service.py`, `http.py`, `select.py`
- Test: `tests/test_site_schema.py`, `test_site_service.py`, `test_site_http.py`, `test_site_select.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: everything Tasks 1 and 2 produce, exactly as named there.
- Produces: spec sections 4, 5 (redaction, logging), 6 and 8 (notice wording, fallback detection by the adapter's signal).

- [ ] **Step 1:** Read the spec in full, `providers.py`, the changed `limits.py` and `claude.py`, then `service.py`, `http.py`, `select.py`, `schema.py` end to end.
- [ ] **Step 2: Settings and schema, test first.** New variables (spec sections 3.1, 5): the four allowance variables, `NER_SITE_OWN_PROVIDERS`, the three `NER_SITE_OWN_MODELS_*` lists (validated: non-empty ids, Anthropic ids must be ones the adapter accepts), all in `ENV_NAMES`. The `DATABASE_URL` rule of spec section 4. `NerRequest.provider`; `NerResponse.provider`, `credential_source`; health `own_keys` and the new `visitor` fields.
- [ ] **Step 3: Own-key flow, test first.** In `_dispatch`: passcode, then `read_credential` (using the raw header values so duplicates are detectable; if the Vercel shim collapses duplicate headers before `handle`, detect them in the shim and say how), then validation against the provider's lists (`validate_request` at both of its call sites), then admission as tier `own`, then clients from `own_llm` for extraction and, when the selector is a model, selection. No chain, no `fallbacks`, no `notices`. Provider failures become `provider_error(...)`; a failed pick's explanation is `failure_word(...)`. The key is an extra secret for the response and log scrubbers for this request only, applied before JSON encoding. Log `provider` and `credential`.
- [ ] **Step 4: Failing closed, test first.** Spec section 4, each row a test: site-key request with the limiter failing is 503 `limits_unavailable` with `Retry-After: 30` and no client is built; limiter not configured without a passcode is 503; with a passcode it is unmetered as today; an own-key request runs in both failing states; the contended-lock 429 is unchanged; the day and own-key 429 wordings and `Retry-After`; an OpenRouter-only passcode deployment is untouched. Health in each state.
- [ ] **Step 5: Carried fixes, test first.** Fallback notes for a server-side fallback come from the adapter's signal (Task 2), for extraction and per pick; the select-stage notice wording of spec section 8.
- [ ] **Step 6:** A test that drives two concurrent own-key requests with different sentinel keys through `handle` and asserts neither key appears in the other's response or in any captured log line, and that no site-key client was constructed. Run the full site suite and ruff.
- [ ] **Step 7:** Report every interpretation of the spec.

---

### Task 4: The page

**Files:**
- Modify: `ner-site/public/index.html`, `app.js`, `app.css`, `ner-site/tools/serve_local.py`
- Create: fixtures under `ner-site/tools/fixtures/` for own-key health, an own-key result, `bad_credential`, `provider_key_refused`, the daily-cap health and 429, and `limits_unavailable`

**Interfaces:**
- Consumes: spec sections 6 and 7 only. Missing new fields must be tolerated (older API).

- [ ] **Step 1:** Read spec sections 6, 7 and the three page files.
- [ ] **Step 2:** Implement section 7: the "Run with" control, the key field and "Forget key", per-provider model and selector options, session storage under one entry cleared on forget and on provider change, the `Authorization: Bearer` header only on the `POST /api/ner` call with `redirect: "error"`, the explanatory text verbatim, the allowance and outage sentences, the results line for an own key. Add the fixtures and `serve_local.py` scenarios for each new state.
- [ ] **Step 3:** Verify in a real browser against the canned server at desktop and phone widths, light and dark: each new state, switching provider clears the field and storage, health requests carry no `Authorization` header (check in the canned server's log or with a request listener). Say plainly what was and was not verified.
- [ ] **Step 4:** Run the site suite and ruff; report.

---

### Task 5: Operator tool and documentation

**Files:**
- Modify: `ner-site/tools/claude_check.py`, `tests/test_site_claude_check.py`, `ner-site/README.md`, `docs/ner-site/PLAN.md`, `docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md` (one amendment line pointing to the new spec for the limiter-outage rule)

- [ ] **Step 1:** `claude_check.py`: `--provider {site,anthropic,openai,openrouter}` and `--key-env NAME` (the key is read from the named environment variable only, sent as `Authorization: Bearer`, and removed from anything printed); the per-file line shows the provider and credential source; the burst summary understands tier `own`, the day window and 503 `limits_unavailable` (which is a failure of the check). Fix the carried miscount: a burst with a Sonnet model on a failing limiter must not be tallied as Sonnet starts. Offline tests for each.
- [ ] **Step 2:** Docs, describing the code as built (read it; do not paraphrase the spec where they differ): the variables table, an "Own keys" section (what travels where, what the operator still pays for, the flood limit, how to turn it off), the daily cap, the changed limiter-outage behaviour everywhere the old one is described, the rollout checks of spec section 10, and a dated section saying what was and was not verified. `docs/ner-site/PLAN.md` section 15.
- [ ] **Step 3:** Run the site suite and ruff; report every place where code and spec differ.
