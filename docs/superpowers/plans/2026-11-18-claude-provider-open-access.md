# Claude Default Provider and Open Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The NER site (`ner-site/`) defaults to Claude Opus 5.5 on the operator's Anthropic key for extraction and selection, steps down to Sonnet 5.5 and then OpenRouter with a visible notice, and enforces a per-visitor hourly allowance stored in Neon so it can run without a passcode.

**Architecture:** Three new single-purpose modules in `ner-site/ner_site/` (`claude.py` adapter, `fallback.py` chain, `limits.py` allowance) behind the engine's existing `LLMClient` protocol, wired together in `service.py` and `http.py`. The page reads two new health blocks and renders notices.

**Tech Stack:** Python 3.10+, the `anthropic` SDK (2.10.0 is installed in `.venv`), `xwalk-match` engine (source readable at `/Users/jandrole/projects/xwalk/src/xwalk`), pydantic, pytest + pytest-asyncio (`asyncio_mode = auto`), ruff; plain JS/CSS page with no build step.

**Spec:** `docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md`. It is binding; this plan adds file boundaries and exact names. Where the plan and spec disagree, the spec wins and the disagreement is reported.

## Global Constraints

- Work only under `ner-site/`, `tests/test_site_*.py`, `docs/`, `pyproject.toml`. Do not touch `src/` or `web/`.
- Tests are offline: no network, no real key. Run with `.venv/bin/python -m pytest -q -m 'not integration' tests/test_site_*.py` from the repository root. Baseline before this work: 269 passed.
- `.venv/bin/ruff check src tests scripts ner-site` and `.venv/bin/ruff format --check src tests scripts ner-site` must pass (line length 100, target py310).
- Claude is called only through the official `anthropic` SDK. No OpenAI-compatible endpoint, no raw HTTP. SDK names (classes, parameters, exception types) are checked against the installed package (`.venv/lib/python3.13/site-packages/anthropic`) or the claude-api skill docs before use, never guessed.
- Model ids, exactly: `claude-opus-5-5`, `claude-sonnet-5-5`. No date suffixes.
- Never send `temperature`, `top_p`, `top_k`, a seed, `thinking: disabled` or `budget_tokens` to Claude.
- No secret (either API key, passcode, database URL) and no raw IP address in any response, log line, SQL parameter or exception message.
- No deployment, no production variable changes, no live API calls.
- Follow the file's existing style: module docstrings that explain why, `from __future__ import annotations`, `__all__`, frozen dataclasses, no new dependencies beyond `anthropic`.
- No em dashes in user-facing copy or docs.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ner-site/ner_site/claude.py` (new) | Anthropic adapter: `ClaudeClient`, error mapping, key check | 1 |
| `ner-site/ner_site/fallback.py` (new) | The chain: `FallbackLLM`, `ChainView`, Claude block cache | 2 |
| `ner-site/ner_site/limits.py` (new) | Visitor id and Neon-backed allowance | 3 |
| `ner-site/ner_site/schema.py`, `service.py`, `http.py`, `select.py` | Settings, wire contract, request flow, health | 4 |
| `ner-site/public/*`, `ner-site/tools/fixtures/*` | Page and canned responses | 5 |
| `ner-site/tools/claude_check.py` (new), `ner-site/README.md`, `docs/ner-site/PLAN.md` | Operator check tool and docs | 6 |

Tasks 1, 2, 3 are independent of each other. Task 4 needs all three. Task 5 needs only the wire contract in spec section 6. Task 6 follows task 4.

---

### Task 1: Anthropic adapter (`claude.py`)

**Files:**
- Create: `ner-site/ner_site/claude.py`
- Test: `tests/test_site_claude.py`
- Modify: `ner-site/requirements.txt` (add `anthropic>=2.10,<3` with a one-line comment), `pyproject.toml` (add `anthropic>=2.10,<3` to the `dev` extra)

**Interfaces:**
- Consumes: `xwalk.llm.base` (`LLMRequest`, `LLMResponse`, `LLMCapabilities`, `LLMFatalError`, `LLMRetryableError`, `reported_usage`).
- Produces:

```python
OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
CLAUDE_MODELS: tuple[str, ...] = (OPUS, SONNET)
DISPLAY_NAMES: dict[str, str] = {OPUS: "Claude Opus 5.5", SONNET: "Claude Sonnet 5.5"}
EXTRACT_MAX_TOKENS = 16000
SELECT_MAX_TOKENS = 8000
EFFORTS: tuple[str, ...] = ("low", "medium", "high")

def is_claude(model: str) -> bool: ...          # model.startswith("claude-")

class ClaudeRefused(LLMFatalError):
    reason = "refused"

class ClaudeClient:                              # satisfies xwalk.llm.base.LLMClient
    def __init__(self, model: str, *, api_key: str, max_tokens: int, effort: str = "low",
                 timeout: float = 60.0, client: Any | None = None) -> None: ...
    model: str                                   # property
    fingerprint: str                             # property; digest of provider, model, effort, max_tokens, adapter version
    capabilities: LLMCapabilities                # json_schema=True, strict_schema=True, usage_reporting=True, seed=False
    generation_defaults: Mapping[str, Any]       # {"temperature": None, "max_tokens": <adapter value>}
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
    async def aclose(self) -> None: ...

def key_status(api_key: str, *, client: Any | None = None, timeout: float = 5.0) -> tuple[str, str | None]:
    """("ok", None) or ("unavailable", reason). One free Models API call, synchronous."""
```

Every exception `complete` raises carries a string attribute `reason`, one of `rate_limited`, `overloaded`, `balance_exhausted`, `key_refused`, `transient`, `bad_request`, `refused` (spec 3.1 error table). Task 2 reads only that attribute.

- [ ] **Step 1: Read first.** The spec section 3.1; `/Users/jandrole/projects/xwalk/src/xwalk/llm/base.py` and `openai_compat.py` (how an adapter builds a fingerprint and reports usage); the installed SDK for the async Messages call, `output_config` (effort and `format` json_schema), the `fallbacks` parameter and its beta flag, the exception classes and their `.type` / `.status_code`, and the Models API retrieve call.
- [ ] **Step 2: Write failing tests** in `tests/test_site_claude.py` using a fake SDK client (an object exposing the same attribute path the adapter calls, recording keyword arguments and returning or raising scripted values). Cases, one test each:
  - the request carries `model`, `system`, one user message, the adapter's `max_tokens`, `output_config.effort`; carries no `temperature`, `top_p`, `top_k`, `thinking`, seed, even when `LLMRequest.temperature=0.0`, `seed=1`, `max_tokens=600` are set
  - with `LLMRequest.schema` the request carries a json_schema format; without it, none
  - a schema lacking `additionalProperties: false` or with optional properties is sent normalised, and the caller's schema object is not mutated
  - refusal fallback is requested (`fallbacks` "default" and the beta flag)
  - text blocks are concatenated, thinking blocks ignored; usage is input plus cache tokens and output tokens with `calls == 1`; `LLMResponse.model` is the response's model
  - `stop_reason == "refusal"` raises `ClaudeRefused` with `reason == "refused"`, even with empty content
  - `stop_reason == "max_tokens"` returns `finish_reason == "length"`
  - each row of the error table maps to the right class and `reason` (build the SDK exceptions the way the SDK's own tests do, or with minimal fake response objects)
  - no raised message contains the API key (use a sentinel key)
  - `key_status`: ok; refused key; connection failure, each without the key in the reason
  - `is_claude` and the constants
- [ ] **Step 3: Run** `.venv/bin/python -m pytest -q tests/test_site_claude.py` and see them fail for the missing module.
- [ ] **Step 4: Implement** `claude.py` to the spec. SDK `max_retries=0`. If the installed SDK rejects structured output together with `fallbacks`, keep structured output, drop `fallbacks` only for schema requests, and say so in the module docstring and your report.
- [ ] **Step 5: Run** the new tests, then the whole site suite and both ruff commands. All pass.
- [ ] **Step 6: Report** (do not commit; the coordinator commits): files changed, test counts, any SDK fact that differed from the spec.

---

### Task 2: The chain (`fallback.py`)

**Files:**
- Create: `ner-site/ner_site/fallback.py`
- Test: `tests/test_site_fallback.py`

**Interfaces:**
- Consumes: `xwalk.llm.base` (`LLMClient`, `LLMRequest`, `LLMResponse`, `LLMError`, `ParseError`, `failure_usage`), `xwalk.llm.budget.CallLimitExceeded`, `xwalk.records.Usage`. It does not import `claude.py`: it reads `getattr(exc, "reason", None)`.
- Produces:

```python
STEP_DOWN_REASONS = frozenset({"rate_limited", "overloaded", "balance_exhausted", "key_refused", "refused"})
BLOCK_REASONS = frozenset({"balance_exhausted", "key_refused"})
BLOCK_TTL_S = 300.0

@dataclass(frozen=True)
class Step:
    label: str            # "opus" | "sonnet" | "other"
    client: LLMClient

@dataclass(frozen=True)
class Fallback:
    stage: str            # "extract" | "select"
    from_model: str
    to_model: str
    reason: str           # a STEP_DOWN_REASONS value, "transient" or "call_limit"

class FallbackLLM:        # satisfies LLMClient
    def __init__(self, steps: Sequence[Step], *, start: int = 0) -> None: ...
    stage: str            # plain attribute, default "extract"; the caller sets "select" before selection
    position: int         # property: index of the current step
    fallbacks: list[Fallback]   # in order, each step-down once
    model: str            # property: current step's client.model
    fingerprint: str      # property: current step's
    capabilities: LLMCapabilities  # property: current step's
    async def complete(self, request: LLMRequest) -> LLMResponse: ...
    def view(self) -> ChainView: ...
    async def aclose(self) -> None: ...     # closes every step that has aclose

class ChainView:          # satisfies LLMClient; delegates to the chain
    answered_by: str | None   # model of this view's last successful answer

def block_claude(reason: str, *, now: float | None = None) -> None: ...
def claude_block(*, now: float | None = None) -> str | None: ...   # the reason while blocked, else None
def reset() -> None: ...
```

Rules (spec 3.2): step down immediately on a `STEP_DOWN_REASONS` reason or `CallLimitExceeded` (recorded as `call_limit`); on `transient` retry the same step once, then step down; any other error, including `ParseError`, is raised unchanged; the last step's error is raised unchanged. Step-down is sticky and never moves up; two concurrent calls failing on the same step record one `Fallback`. A `BLOCK_REASONS` failure calls `block_claude`. The returned response's `usage` is the successful answer's usage plus `failure_usage(exc)` of every failed attempt in that call. An error finally raised carries the accumulated usage in the same way when the error class accepts it (`exc.usage`).

- [ ] **Step 1: Read** spec 3.2, `xwalk/llm/base.py`, `xwalk/llm/budget.py`, `xwalk/llm/fake.py` (`FakeLLM`, used for scripted steps) and `xwalk/records.py` (`Usage` and how usages add).
- [ ] **Step 2: Write failing tests** in `tests/test_site_fallback.py` with small scripted clients (answer, or raise an error with a given `reason`). One test each:
  - first step answers: no fallbacks, `position == 0`
  - each `STEP_DOWN_REASONS` value steps down and is recorded with the right models, reason and stage
  - `CallLimitExceeded` steps down as `call_limit`
  - `transient` is retried once on the same step (two calls to it), then steps down
  - `bad_request` and `ParseError` are raised unchanged with no step-down
  - last step failing raises its error unchanged
  - sticky: after a step-down the next `complete` does not call the earlier step
  - `start=1` never calls step 0
  - eight concurrent `complete` calls against a rate-limited first step record exactly one `Fallback` and all succeed
  - usage of failed attempts is included in the returned usage
  - `stage` set to `"select"` is what later fallbacks record
  - `view()`: `answered_by` is the answering model per view, correct for two views that answered on different steps
  - `balance_exhausted` and `key_refused` set `claude_block()`; it expires after `BLOCK_TTL_S` (pass `now`); `reset()` clears it; `rate_limited` does not block
  - `aclose` closes every step
- [ ] **Step 3: Run** and see them fail. **Step 4: Implement.** **Step 5: Run** the new tests, the whole site suite and ruff. **Step 6: Report** (do not commit).

---

### Task 3: Visitor allowance (`limits.py`)

**Files:**
- Create: `ner-site/ner_site/limits.py`
- Test: `tests/test_site_limits.py`

**Interfaces:**
- Consumes: the synchronous `SQLExecutor` protocol (`execute(sql, params) -> list[Row]`, `transaction(...)`) in `/Users/jandrole/projects/xwalk/src/xwalk/retrieval/postgres.py`; read `NeonHTTPExecutor` there and how `ner-site/ner_site/indexes.py` and `tests/test_site_neon.py` use and fake it.
- Produces:

```python
Tier = Literal["opus", "sonnet", "other"]
TABLE = "site_visits"
PRUNE_AFTER_S = 86400
PRUNE_CHANCE = 0.02

@dataclass(frozen=True)
class Allowance:
    opus_per_hour: int = 10
    sonnet_per_hour: int = 20

@dataclass(frozen=True)
class Admission:
    tier: Literal["opus", "sonnet", "other", "blocked"]   # "blocked" only from peek
    opus_left: int
    sonnet_left: int
    resets_in_s: int
    reason: str | None        # None, or "opus_allowance_used" when wanted opus but granted sonnet

class LimitsUnavailable(Exception): ...          # the database could not be queried; message scrubbed
class VisitorLimitReached(Exception):
    resets_in_s: int

def visitor_id(headers: Mapping[str, str], salt: str) -> str: ...   # lower-cased header keys in
def default_salt(database_url: str) -> str: ...

class VisitorLimits:
    def __init__(self, executor: SQLExecutor, schema: str, allowance: Allowance, *,
                 clock: Callable[[], float] = time.time,
                 rng: Callable[[], float] = random.random) -> None: ...
    def admit(self, visitor: str, wanted: Tier) -> Admission: ...
    def peek(self, visitor: str) -> Admission: ...     # what an Opus request would get now; records nothing
```

Rules (spec 3.3): table `<schema>.site_visits(visitor text, hour timestamptz, tier text, count integer, primary key (visitor, hour, tier))`, created with `CREATE TABLE IF NOT EXISTS` on first use of an instance; `hour` is the UTC clock hour of `clock()`. `admit`: if the hour's total across tiers is at least `opus_per_hour + sonnet_per_hour`, raise `VisitorLimitReached` and record nothing. Otherwise grant: wanted `opus` gets `opus` while the Opus count is below `opus_per_hour`, else `sonnet` while the Sonnet count is below `sonnet_per_hour` (reason `opus_allowance_used`), else `VisitorLimitReached`; wanted `sonnet` gets `sonnet` while below `sonnet_per_hour`, else `VisitorLimitReached`; wanted `other` gets `other`. The visit is recorded under the granted tier with an upsert that increments `count`. `opus_left` / `sonnet_left` are after this visit. `resets_in_s` is the seconds to the next clock hour (at least 1). When `rng() < PRUNE_CHANCE`, rows with `hour` older than `PRUNE_AFTER_S` are deleted. The schema name is validated as a lower-case identifier before it is put into SQL; every value goes in as a parameter. Any executor failure raises `LimitsUnavailable` (no database URL, password or host in the message). `visitor_id` is the first 32 hex characters of `HMAC-SHA256(salt, ip)` where `ip` is the first non-empty of `x-real-ip`, `x-vercel-forwarded-for`, the first entry of `x-forwarded-for`, else the empty string; surrounding whitespace is stripped.

- [ ] **Step 1: Read** spec 3.3 and the files named above.
- [ ] **Step 2: Write failing tests** in `tests/test_site_limits.py` with an in-memory fake executor that actually keeps the counts (interpret the handful of statements the limiter sends, or back it with `sqlite3` if the SQL is portable enough; say which you chose). One test each:
  - visits 1 to 10 wanting Opus get `opus`, `opus_left` counting down to 0, `reason is None`
  - visit 11 gets `sonnet` with reason `opus_allowance_used`; visit 30 is the last; visit 31 raises `VisitorLimitReached` with `1 <= resets_in_s <= 3600` and records nothing
  - wanting `sonnet` uses only the Sonnet allowance (21st raises)
  - `other` visits count toward the total of 30 but not toward Opus or Sonnet
  - a second visitor is unaffected
  - the next clock hour starts fresh
  - `peek` reports the same numbers and records nothing; reports `blocked` at the total
  - allowances of 0 (Opus 0: first visit is Sonnet; both 0: first visit raises)
  - pruning runs when `rng` returns 0.0 and not when it returns 0.99, and deletes only rows older than a day
  - executor raising gives `LimitsUnavailable` whose message has no secret (sentinel password in the fake's error)
  - a schema that is not a lower-case identifier raises `ValueError` at construction
  - `visitor_id`: each header source in priority order, first entry of a list, whitespace, no headers, different salts differ, the raw IP does not appear in the id; no SQL parameter recorded by the fake ever contains the raw IP
- [ ] **Step 3: Run** and see them fail. **Step 4: Implement**, using as few round trips per `admit` as practical (one statement preferred; `transaction` is acceptable). **Step 5: Run** the new tests, the whole site suite and ruff. **Step 6: Report** (do not commit), including the exact SQL and how many round trips `admit` and `peek` make.

---

### Task 4: Settings, wire contract, request flow, health

**Files:**
- Modify: `ner-site/ner_site/schema.py`, `ner-site/ner_site/service.py`, `ner-site/ner_site/http.py`, `ner-site/ner_site/select.py`
- Test: `tests/test_site_schema.py`, `tests/test_site_service.py`, `tests/test_site_http.py`, `tests/test_site_select.py` (extend; update expectations that encode the old defaults)

**Interfaces:**
- Consumes: everything Tasks 1 to 3 produce, exactly as listed there.
- Produces (spec sections 4, 5, 6):
  - `Settings` gains `anthropic_api_key: str | None` (repr off, in `secrets()`), `claude_effort: str`, `fallback_model: str`, `allowance: limits.Allowance`, `visitor_salt: str | None`; `ENV_NAMES` gains `ANTHROPIC_API_KEY`, `NER_SITE_CLAUDE_EFFORT`, `NER_SITE_FALLBACK_MODEL`, `NER_SITE_OPUS_PER_HOUR`, `NER_SITE_SONNET_PER_HOUR`, `NER_SITE_VISITOR_SALT`.
  - `DATABASE_URL` is read for the limiter whenever it is set, not only when `NER_SITE_BACKEND=neon`; the index backend choice is unchanged.
  - New code defaults and the offering rules of spec section 4 (`offered_models`, `offered_selectors`, `default_model`, `default_selector`, `configured`). An unknown `claude-` id in a list is a `SettingsError`.
  - `NerResponse` gains `requested_model: str`, `requested_selector: str`, `fallbacks: list[FallbackNote]`, `notices: list[str]`; `FallbackNote` has `stage`, `from_` serialised as `"from"`, `to`, `reason`. `HealthResponse` (or the extra blocks `_health` adds, following the existing `llm` pattern) gains `claude` and `visitor` as in spec section 6.
  - `http.handle` returns 429 `visitor_limit_reached` with a `Retry-After` header.
  - `select._with_llm` uses `llm.view()` when the client has one and writes `answered_by` to `Pick.selector`.

- [ ] **Step 1: Read** the spec in full, the three new modules and their tests, then `service.py`, `http.py`, `select.py`, `schema.py` and the four test files end to end before changing anything.
- [ ] **Step 2: Settings and schema, test first.** Tests: defaults with both keys (Opus first, default selector Opus); only OpenRouter key (no Claude offered, defaults Gemini and Jev); only Anthropic key (no OpenRouter ids, no Jev, `configured` true); effort and allowance parsing and their refusals; unknown `claude-` id refused; both keys scrubbed by `scrub`; new response fields serialise with `"from"`. Implement. Update existing tests whose expectations encode the old default ids, keeping what each test was actually checking.
- [ ] **Step 3: Chain construction and run, test first.** In `service.py` add the builders the HTTP layer needs (for example `make_chain(settings, wanted_model, *, start, purpose)` returning a `FallbackLLM` whose Claude steps are `BudgetedLLM(ClaudeClient(...))` with `EXTRACT_MAX_TOKENS` or `SELECT_MAX_TOKENS`, and whose last step is the existing `make_llm` / `make_select_llm` for `settings.fallback_model` when an OpenRouter key is set). `run` sets the chain's `stage` to `"select"` before selection, fills `model` with the model that produced the extraction, and fills `requested_*`, `fallbacks` and `notices` (spec section 6 wording; one notice per distinct fallback; Claude display names from `claude.DISPLAY_NAMES`). Usage accounting is unchanged in shape. Tests with scripted fake Claude clients injected through the existing factory seams (extend `LLMFactory` or add a parallel seam; no test may construct a real SDK client that could reach the network).
- [ ] **Step 4: Admission in `http.py`, test first.** Order: existing checks and `validate_request`, then the wanted tier (spec section 5 step 2), then the limiter. Limiter states exactly as spec 3.3 (on, unreachable, off with and without a passcode, offline never consults it). `claude_block()` lowers the start. A warm-instance cache holds the `VisitorLimits` (and its executor) per database URL, with a `reset()` hook for tests, mirroring how `indexes.py` caches. Admission step-downs appear in `fallbacks` with stage `"admit"` and in `notices`. The log line gains `tier` and `fallbacks`; never the visitor id or IP. Tests: Opus by default; 11th request Sonnet with notice; 31st is 429 with `Retry-After`; limiter unreachable gives Sonnet with notice and is not counted; no `DATABASE_URL` without passcode gives Sonnet, with passcode gives Opus; a blocked Claude starts on the fallback model with the notice; a request for an OpenRouter model has no chain and is recorded as `other`; a 400 request is not counted; no IP or key in any response or captured log line.
- [ ] **Step 5: Health, test first.** `claude` block (key unset, `key_status` result cached like `check_key`, `claude_block()` reason) and `visitor` block (`peek`; `limits` on, off or unavailable). Existing autouse fixtures that stub `check_key` get a sibling for the Claude key check.
- [ ] **Step 6: Run** the whole site suite and both ruff commands; all pass. **Step 7: Report** (do not commit): files changed, test counts, every place the implementation had to interpret the spec.

---

### Task 5: The page

**Files:**
- Modify: `ner-site/public/index.html`, `ner-site/public/app.js`, `ner-site/public/app.css`
- Create: `ner-site/tools/fixtures/health-sonnet.json`, `health-blocked.json`, `ner-fallback.json`; modify `health.json` and `ner.json` to carry the new fields; extend `ner-site/tools/serve_local.py` scenarios if that is how fixtures are selected

**Interfaces:**
- Consumes: the wire contract in spec section 6 only (`claude`, `visitor`, `requested_model`, `requested_selector`, `fallbacks`, `notices`, `pick.selector` as a model id, error `visitor_limit_reached`). Missing fields must be tolerated: the page may be served against an older API.
- Produces: nothing other tasks import.

- [ ] **Step 1: Read** spec section 7, the three page files and `tools/serve_local.py`, `tools/screenshot.mjs`.
- [ ] **Step 2: Implement** section 7: display names for the two Claude ids in both dropdowns and wherever a model or selector id is shown; a notices block above the results, visually distinct from warnings and using the page's existing design tokens; the results header naming the model that ran; the allowance line under the model field from `health.visitor` (nothing on Opus, the Sonnet sentence, the blocked sentence with minutes rounded up); health re-read after each request; the footer sentence. All insertion through `textContent`; no inline styles or scripts (the CSP is `default-src 'self'`).
- [ ] **Step 3: Fixtures** for the Sonnet and blocked states and a response with fallbacks and notices.
- [ ] **Step 4: Verify in a browser.** Run `.venv/bin/python ner-site/tools/serve_local.py --canned ner-site/tools/fixtures` (and the new scenarios), load the page, and check each state renders at 1280 px and at phone width, in light and dark if the page has both. Use `tools/screenshot.mjs` if Playwright is available; otherwise say so in the report.
- [ ] **Step 5: Run** the site suite (some tests read `public/` and the fixtures) and ruff. **Step 6: Report** (do not commit), with what was and was not visually verified.

---

### Task 6: Operator check tool and documentation

**Files:**
- Create: `ner-site/tools/claude_check.py`
- Modify: `ner-site/README.md`, `docs/ner-site/PLAN.md`

- [ ] **Step 1: `claude_check.py`**: standard library only, like `serve_local.py`. `python ner-site/tools/claude_check.py --url https://www.xwalk.si [--passcode-env NAME] [--model ID] [--selector ID] FILE...` posts each text file to `/api/ner` with the `X-Requested-With: xwalk-ner` header and prints one line per file: requested and actual model, entity count, picks, fallbacks, `timing_ms` and token counts; then totals and the slowest request. Exit code 1 if any request failed. The passcode is read from the named environment variable, never from an argument. It makes no request unless run by the operator; it has a `--help` and a module docstring saying what it costs to run.
- [ ] **Step 2: Docs.** `ner-site/README.md`: the variables table (spec section 4), the rollout steps (spec section 10) under Deploy, and a dated section "Claude by default and open access, 18 November 2026" stating what is implemented and, plainly, that nothing was deployed or run live. `docs/ner-site/PLAN.md`: a section 14 with the decision table from spec section 1 and the wire additions. Describe the code as it is after Task 4; read it, do not paraphrase the spec where they differ.
- [ ] **Step 3: Run** the site suite and ruff. **Step 4: Report** (do not commit).

---

## Self-review notes

- Spec coverage: 3.1 Task 1; 3.2 Task 2; 3.3 Task 3; 4, 5, 6, 8 Task 4; 7 Task 5; 9 across Tasks 1 to 4; 10 and 11 Task 6.
- The spec's "OpenRouter step for selection" is the fallback model as an LLM selector, not Jev, so the chain has one interface.
- Names used across tasks: `reason` attribute values (Tasks 1, 2), `FallbackLLM.stage` / `fallbacks` / `view()` (Tasks 2, 4), `Admission` / `VisitorLimitReached` / `LimitsUnavailable` / `visitor_id` (Tasks 3, 4), `DISPLAY_NAMES` (Tasks 1, 4).
