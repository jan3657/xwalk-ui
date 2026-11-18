# NER-to-ontology demo site on Vercel: implementation plan

Status: plan, 2026-11-03. Drafted by a planning agent, reviewed and amended by the
orchestrating session (amendments marked **[amended]**). The executing agents have no
`vercel` CLI, no Vercel token and no OpenRouter key: they produce a deploy-ready
`ner-site/` directory verified locally, and the operator deploys and verifies (section
11).

The request: a minimal page where a visitor pastes text and an extraction prompt, an LLM
extracts named entities (chemical, disease, gene, food, other) with character spans, and
each entity is mapped with xwalk onto the matching bundled sample ontology (ChEBI, CTD
disease, NCBI Gene, FoodOn; 200 terms each in `src/xwalk_ui/ontologies/`). Hosted on
Vercel.

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| 1 | New top-level `ner-site/` with its own `vercel.json`, `requirements.txt`, `.python-version`; Vercel project Root Directory = `ner-site`. Nothing added to the `xwalk` package. **[amended]** The directory is not called `site/`: that is the name of a standard-library module and the plan puts the directory on `sys.path`. | Keeps the library light and the Vercel build from scanning the repo (the `web/` package.json would otherwise be picked up as a framework). |
| 2 | Static page in `ner-site/public/` (plain HTML/CSS/JS, no build step) plus two file-based Python functions `ner-site/api/ner.py` and `ner-site/api/health.py`, each defining `handler(BaseHTTPRequestHandler)`. No web framework. | The `/api` form is supported and needs no extra dependency. |
| 3 | Python 3.12 (`ner-site/.python-version`). | Vercel's default; tantivy 0.27.0 publishes a `cp311-abi3-manylinux_2_17_x86_64` wheel, so it installs without a compiler. |
| 4 | xwalk installed from git in `ner-site/requirements.txt`: `xwalk-match @ git+https://github.com/jan3657/xwalk.git@<full sha>`, plus `numpy`. **[amended]** The repository is public (checked: github.com/jan3657/xwalk answers 200 without credentials), so this works on Vercel without a token. | Vercel installs only what the root directory declares; a pinned SHA is reproducible. |
| 5 | NER = one chat completion through `xwalk.llm.openai_compat.OpenAICompatClient` against OpenRouter, wrapped in `BudgetedLLM`. | Reuses xwalk's retry/parse/usage code and its call cap. |
| 6 | Mapping = retrieval only: BM25 + dense fused by reciprocal rank through `xwalk.retrieve.retrieve`, top-k shown with scores. No LLM in the mapping step in v1. | One LLM call per run keeps latency and cost low, and the user's prompt cannot influence which IDs are returned. An LLM-matcher mode is phase 2. |
| 7 | BM25 indexes built at cold start into `/tmp` from the bundled JSONL. Dense indexes precomputed by `ner-site/tools/build_indexes.py` and committed under `ner-site/indexes/<slug>/dense/`. **[amended]** The orchestrating session holds the user's key and builds and commits them after WS-B; no subagent receives the key. | The function filesystem is read-only except `/tmp`; tantivy writes on build. A committed index is verifiable offline. |
| 8 | If a dense index is missing, stale, or the key is unset, the response and `/api/health` report `dense: unavailable` with the reason and mapping runs on BM25 alone. Never silently. | xwalk's no-silent-substitution invariant. |
| 9 | All logic in `ner-site/ner_site/` (importable, tested); `ner-site/api/*.py` are short shims. | Testable without Vercel. |

### Vercel facts relied on (read 2026-11-03 by the planning agent)

- Python runtime, entrypoints, versions, dependencies, 500 MB bundle, `excludeFiles`, cwd is the project root: https://vercel.com/docs/functions/runtimes/python
- File-based `/api` functions, `handler(BaseHTTPRequestHandler)`, `api/ner.py` -> `/api/ner`: https://vercel.com/docs/functions/runtimes/python/api-directory
- Version selection (`.python-version`, 3.12 default): https://vercel.com/docs/functions/runtimes/python/python-version
- Limits: max duration default 300 s (Hobby max 300 s, Pro 800 s); 2 GB / 1 vCPU; request and response body 4.5 MB: https://vercel.com/docs/functions/limitations
- Read-only filesystem with writable `/tmp` up to 500 MB: https://vercel.com/docs/functions/runtimes
- `functions` (`maxDuration`, `includeFiles`, `excludeFiles`), `headers`, `rewrites`: https://vercel.com/docs/project-configuration/vercel-json
- Environment variables: https://vercel.com/docs/environment-variables
- Root Directory for a subdirectory project: https://vercel.com/docs/monorepos
- Unverified: cold-start time with tantivy + pydantic + httpx (measure after deploy); whether `includeFiles` is needed for `ner_site/**` and `indexes/**` (set it anyway).

## 2. Layout

```text
ner-site/
  .python-version          3.12
  requirements.txt         xwalk-match @ git+...@<sha>, numpy
  vercel.json              functions.maxDuration=120, includeFiles, security headers
  README.md                deploy and local-run instructions
  api/ner.py               POST /api/ner  -> ner_site.http.handle
  api/health.py            GET  /api/health -> ner_site.http.handle
  ner_site/__init__.py
  ner_site/schema.py       pydantic request/response models, Limits, ServiceError
  ner_site/extract.py      prompt, JSON schema, LLM call, span relocation
  ner_site/ontologies.py   catalog, term loading, type->ontology routing, term URLs, templates
  ner_site/indexes.py      BM25 cold-start build in /tmp, dense open, status reporting
  ner_site/mapping.py      fused retrieval per entity
  ner_site/service.py      Settings from env, LLM factory, run()
  ner_site/http.py         framework-free request handling, error envelope, passcode
  ner_site/fake.py         offline stand-in LLM (NER_SITE_OFFLINE=1) for demos and smoke tests
  indexes/<slug>/dense/    vectors.f32, record_ids.json, xwalk_meta.json (committed once built)
  public/index.html, app.js, app.css
  tools/build_indexes.py   builds ner-site/indexes with OPENROUTER_API_KEY
  tools/serve_local.py     stdlib server: public/ + /api/* through ner_site.http (--offline)
  tools/screenshot.mjs     Playwright: load page, run offline sample, screenshot
  tools/fixtures/*.json    canned /api responses for frontend work before WS-D lands
tests/test_site_*.py       offline tests (root tests dir; pytest `pythonpath = ["ner-site"]`)
```

`api/*.py` import `ner_site` lazily inside the handler method after
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`. The bundled ontology JSONL
comes from the installed package via `importlib.resources.files("xwalk_ui") / "ontologies"`.

## 3. Contracts (binding for every workstream)

### 3.1 HTTP

`GET /api/health` -> 200:

```json
{"ok": true, "mode": "live", "default_model": "openai/gpt-4o-mini",
 "models": ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"],
 "limits": {"max_chars": 5000, "max_prompt_chars": 4000, "max_entities": 25, "top_k_max": 10},
 "passcode_required": false, "default_prompt": "...",
 "ontologies": [{"slug": "chebi-sample", "name": "ChEBI (sample)", "type": "chemical", "count": 200}],
 "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "unavailable", "reason": "no index at indexes/chebi-sample/dense; run ner-site/tools/build_indexes.py"}},
 "selectors": ["retrieval", "jev", "openai/gpt-6-luna"], "default_selector": "jev"}
```

`POST /api/ner` with `Content-Type: application/json` and `X-Requested-With: xwalk-ner`
(and `X-Passcode` when required):

```json
{"text": "Aspirin reduced fever in rheumatoid arthritis.", "prompt": null, "model": null, "top_k": 5,
 "selector": null}
```

`prompt`/`model`/`selector` null mean the defaults from `/api/health` (section 13 for
`selector`, `pick` and the `selector_*` usage fields). 200 response:

```json
{"ok": true, "model": "openai/gpt-4o-mini", "mode": "live", "selector": "jev",
 "entities": [
   {"id": "e1", "text": "Aspirin", "start": 0, "end": 7, "type": "chemical", "ontology": "chebi-sample",
    "matches": [{"id": "CHEBI:15365", "label": "acetylsalicylic acid", "score": 0.0325,
                 "retrievers": {"bm25": 1, "dense-qwen3": 2},
                 "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365"}],
    "notes": [],
    "pick": {"selector": "jev", "status": "match", "id": "CHEBI:15365", "label": "acetylsalicylic acid",
             "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365", "confidence": 0.93,
             "explanation": "rubric 0.95, properties 0.97"},
    "more": [{"id": "CHEBI:46195", "label": "paracetamol", "score": 0.0161, "retrievers": {"dense-qwen3": 2},
              "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:46195"}]},
   {"id": "e2", "text": "fever", "start": 16, "end": 21, "type": "other", "ontology": null,
    "matches": [], "notes": ["no ontology for type 'other'"], "pick": null,
    "more": []}],
 "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
 "warnings": ["entity 'feverr' dropped: not found in text"],
 "usage": {"llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 88, "embedding_requests": 2,
           "selector_calls": 3, "selector_prompt_tokens": 2900, "selector_completion_tokens": 0,
           "selector_cost_usd": 0.0001},
 "timing_ms": {"extract": 1800, "map": 900, "select": 1400, "total": 4200}}
```

Token counts are `null` when the provider did not report them, never 0. Errors:

```json
{"ok": false, "error": {"code": "text_too_long", "message": "text is 7000 characters; the limit is 5000"}}
```

| HTTP | code |
|---|---|
| 400 | `bad_json`, `missing_text`, `text_too_long`, `prompt_too_long`, `bad_top_k`, `unknown_model`, `bad_header` (missing `X-Requested-With`) |
| 401 | `passcode_required`, `passcode_invalid` |
| 404 / 405 | unknown path / method (OPTIONS is 405 on purpose) |
| 413 | `body_too_large` (> 64 KB) |
| 422 | `extraction_unparseable` (model answer not a JSON object after xwalk's parsing), `extraction_schema` (object lacks `entities`) |
| 429 | `call_budget_exhausted`. **[amended 2026-11-18]** Also `visitor_limit_reached`, with `Retry-After` (section 14) |
| 502 | `provider_error` (LLM or embeddings failure; message without the key or URL userinfo) |
| 503 | `not_configured` (no `OPENROUTER_API_KEY` and not offline mode). **[amended 2026-11-18]** Now: neither `ANTHROPIC_API_KEY` nor `OPENROUTER_API_KEY` |
| 500 | `internal` (generic message; details to the function log only) |

Every response carries `Cache-Control: no-store`. No `Access-Control-*` headers are ever
sent.

**[amended by WS-D]** Gaps settled while implementing `ner_site/http.py`:

- 404 is `not_found`; 405 is `method_not_allowed` and carries `Allow` (`POST` on
  `/api/ner`, `GET, HEAD` on `/api/health`). `OPTIONS` is 405 on both, never a preflight
  answer.
- The shim reads at most `Content-Length` bytes and refuses before reading: no
  `Content-Length` on a POST is **411 `length_required`**, a non-numeric or negative one
  is 400 `bad_header`, one over 64 KB is 413 `body_too_large`. `handle` also checks
  the length of the body it is given (413).
- Order of checks on `POST /api/ner`: path, method, body size, `X-Requested-With`,
  passcode, configuration (503), JSON, then field limits. `Content-Type` is not checked:
  `X-Requested-With` already forces a CORS preflight that always fails.
- `/api/health` needs neither `X-Requested-With` nor the passcode (it returns nothing
  secret, and the operator opens it in a browser). It always answers 200, also without a
  key: `mode` is `"live"` and an extra block `"llm": {"status": "unavailable", "reason":
  "OPENROUTER_API_KEY is not set; POST /api/ner answers 503 not_configured"}` says what
  is missing (`{"status": "ok"}` with a key; `ok` with reason "offline stand-in
  extractor" offline). `llm` is not a `HealthResponse` field; the page ignores it.
- A bad environment value (section 3.2) makes both endpoints answer 503 `not_configured`
  with a message naming the variable (never its value), not a traceback.
- Offline mode: `mode` is `"offline"`, the response `model` and the health
  `default_model` are `"offline"`, `models` is `["offline"]`, and a request may name only
  `"offline"` (or null). Offline mode never uses the key, even when one is set: dense is
  unavailable (reason "OPENROUTER_API_KEY is not set", or "offline mode ... requests no
  query embeddings" when a key is set) and `usage` is `llm_calls: 0` with null tokens.
- `start`/`end` are Python `str` indices: Unicode code points, not bytes or UTF-16
  units. `text_too_long` and `prompt_too_long` count code points too.
- `usage.llm_calls` counts dispatched HTTP requests, retries included. The token counts
  are null unless every request reported them. `embedding_requests` is a lower bound
  (`MapStats`).
- Embedding failures at request time do not fail the request: the entity gets a note,
  `warnings` summarises it, and BM25 matches are still returned. `provider_error` (502)
  is for the extraction call only, and also covers an extraction that takes longer than
  90 s (retries included). Each retriever call then gets `min(30 s, what is left of
  115 s)`, at least 5 s. **[superseded after review, below]**
- When the retrieval block says dense (or BM25) is unavailable and some entity was routed,
  `warnings` also carries `"dense retrieval unavailable; matches use BM25 only"`. A reason
  shared by all four ontologies is reported once, with "(all 4 ontologies)".
- The key, and a passcode of 8 or more characters, are also string-replaced out of every
  response body and log line as a last line of defence. Provider messages pass through
  `redact_url` for every URL and are cut to 300 characters.
- **[amended after review]** Time: the whole mapping step gets one wall-clock deadline,
  what is left of 100 s since the request began (opening the indexes and the
  extraction included; at least 5 s); each retriever call within
  it at most 30 s. A dense search still running at the deadline is cut off and one not
  yet started is skipped (BM25 gets 5 s of grace, so every entity keeps its BM25
  matches); the entity gets the note `"dense-qwen3: cut off by the request deadline"`
  and `warnings` the usual `"<note> (<entity ids>)"` summary. Worst case about 105 s,
  inside `maxDuration` 120 s. Each request runs on its own event loop and thread pool,
  shut down without waiting (`http.run_request`): an abandoned embeddings thread
  finishes in the background and cannot touch a later response. Query embeddings use
  `max_retries=0` (timeout 15 s), so a 429 with a long `Retry-After` fails at once
  instead of sleeping.
- **[amended after review]** Dense at request time: when every entity that reached a
  dense search had it fail, time out or be cut off, the response's `retrieval.dense` is
  `unavailable` with reason `"every dense search failed in this request: <first note>"`
  (plus the `"dense retrieval unavailable; matches use BM25 only"` warning), even though
  `/api/health` says `ok`. When only some failed, dense stays `ok` and `warnings` adds
  `"dense search failed for N of M entities (e2, e5); their matches use BM25 only"`.
- **[amended after review]** Entity `notes` from retrievers pass through the same scrub
  as provider messages (key and passcode removed, every URL through `redact_url`) and
  are cut to 300 characters.
- **[amended after review]** `bad_json` also covers a `text`, `prompt` or `model` that
  is not encodable as UTF-8 (a lone surrogate escape such as `"\ud800"`), which used to
  be a 500. `Content-Length` must be 1 to 18 ASCII digits (`"²"` passes `str.isdigit()`
  but not `int()`); anything else is 400 `bad_header`.
- **[amended after review]** `default_prompt` (health, and the prompt used when a
  request sends none) states the configured `NER_SITE_MAX_ENTITIES`
  (`extract.render_default_prompt(limit)`; `extract.DEFAULT_PROMPT` is the one for 25).
- Vercel (https://vercel.com/docs/functions/runtimes/python/api-directory, read
  2026-11-03): a file-based function defines a top-level `handler` class that inherits
  from `BaseHTTPRequestHandler` (or `app`/`application` for ASGI/WSGI); `api/ner.py`
  serves `/api/ner`. The request is read the standard way (`self.command`, `self.path`,
  `self.headers`, `self.rfile`). Both shims are identical: `do_*` call
  `ner_site.http.serve(self)`, which routes by path, so a request reaching the wrong file
  still gets the right 404/405. `log_message` is silenced so each request logs one line.

### 3.2 Environment variables (function runtime)

| Name | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | unset | the operator's key; used for the chat call and query embeddings |
| `NER_SITE_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL. **[amended after review]** For the chat call only: query embeddings always go to OpenRouter (`https://openrouter.ai/api/v1/embeddings`) with `OPENROUTER_API_KEY`. The committed dense indexes record that embeddings base URL in their identity, so moving embeddings elsewhere means rebuilding them (`tools/build_indexes.py`) and changing the code |
| `NER_SITE_MODELS` | `openai/gpt-4o-mini,google/gemini-2.5-flash-lite,qwen/qwen3-next-80b-a3b-instruct` | allowlist, first is default. **[amended]** All three ids exist on OpenRouter and list `structured_outputs` (checked 2026-11-03). **[amended 2026-11-18]** The default is now `claude-opus-5-5,claude-sonnet-5-5,google/gemini-3.1-flash-lite,openai/gpt-6-luna`, and only ids whose key is set are offered (section 14, which also lists the new variables; section 15 adds those of the daily cap and of visitors' own keys) |
| `NER_SITE_PASSCODE` | unset | when set, `X-Passcode` must equal it (`hmac.compare_digest`). **[amended after review]** ASCII only: `http.server` reads header values as Latin-1, so another value could never match; it is refused (503 `not_configured` naming the variable) |
| `NER_SITE_MAX_CHARS` / `NER_SITE_MAX_ENTITIES` / `NER_SITE_MAX_PROMPT_CHARS` | 5000 / 25 / 4000 | limits. **[amended by WS-D]** Whole numbers in 1..20000 / 1..100 / 1..16000; anything else is refused (503 `not_configured` naming the variable), not clamped. Empty means unset. `NER_SITE_OFFLINE` accepts 1/0, true/false, yes/no, on/off |
| `NER_SITE_OFFLINE` | unset | `1` swaps the LLM for `ner_site.fake.offline_llm`; `/api/health` reports `"mode": "offline"` |
| `NER_SITE_DENSE_DIR` | `<project root>/indexes` | where committed dense indexes live |
| `NER_SITE_CACHE_DIR` | `/tmp/xwalk-ner-site` | BM25 build target |

### 3.3 Module signatures

`ner-site/ner_site/schema.py` (WS-0; pydantic v2):

```python
from typing import Literal

EntityType = Literal["chemical", "disease", "gene", "food", "other"]


class Entity:  # pydantic BaseModel in the real file
    id: str
    text: str
    start: int
    end: int
    type: EntityType


class Match:
    id: str
    label: str
    score: float
    retrievers: dict[str, int]
    url: str | None


class EntityResult(Entity):
    ontology: str | None
    matches: list[Match]
    notes: list[str]


class RetrieverStatus:
    status: Literal["ok", "unavailable"]
    reason: str | None = None


class NerRequest:
    text: str
    prompt: str | None = None
    model: str | None = None
    top_k: int = 5


class ServiceError(Exception):
    def __init__(self, code: str, message: str, *, status: int) -> None: ...
```

Also `NerResponse`, `HealthResponse`, `Usage` (llm_calls, prompt_tokens | None,
completion_tokens | None, embedding_requests) and `Limits(max_chars=5000,
max_prompt_chars=4000, max_entities=25, top_k_max=10, max_body_bytes=65536,
llm_call_cap=3)`.

`ner-site/ner_site/extract.py` (WS-A):

```python
from xwalk.llm.base import LLMClient, LLMRequest

DEFAULT_PROMPT: str  # [amended after review] = render_default_prompt(25)
def render_default_prompt(max_entities: int) -> str: ...  # Settings.default_prompt uses it
SYSTEM_PROMPT: str  # fixed, not user-editable
ENTITY_SCHEMA: dict  # JSON schema passed as LLMRequest.schema


def build_request(text: str, prompt: str) -> LLMRequest: ...
def locate(text: str, mention: str, hint: int | None) -> tuple[int, int] | None: ...
def parse_entities(raw: str, text: str, *, max_entities: int) -> tuple[list, list[str]]: ...
async def extract(llm: LLMClient, text: str, prompt: str, *, max_entities: int) -> tuple[list, list[str], object]: ...
```

(`extract` returns entities, warnings and the xwalk `Usage` of the call.)

`ner-site/ner_site/ontologies.py` (WS-B):

```python
from xwalk.records import Record
from xwalk.templates import TemplateSet

ROUTES: dict[str, str]  # chemical->chebi-sample, disease->ctd-disease-sample, gene->ncbi-gene-sample, food->foodon-sample
TEMPLATES: TemplateSet  # query "{{ mention }}", doc "{{ label }} {{ synonyms | join(' ') }} {{ definition }}"


def catalog() -> list[dict]: ...
def load_terms(slug: str) -> list[Record]: ...
def route(entity_type: str) -> str | None: ...
def term_url(record_id: str) -> str | None: ...
```

`ner-site/ner_site/indexes.py` (WS-B):

```python
from collections.abc import Callable, Mapping
from pathlib import Path


class OntologyIndex:  # dataclass
    slug: str
    store: object  # xwalk.stores.memory.MemoryStore
    retrievers: list  # xwalk Retriever objects, BM25 first
    status: dict  # {"bm25": RetrieverStatus, "dense": RetrieverStatus}


def open_indexes(*, cache_dir: Path, dense_dir: Path | None, api_key: str | None,
                 encoder_factory: Callable[[], object] | None = None) -> Mapping[str, OntologyIndex]: ...
def get_indexes(settings: object) -> Mapping[str, OntologyIndex]: ...  # module-level memo; reset() for tests
```

`ner-site/ner_site/mapping.py` (WS-B):

```python
async def map_entities(entities: list, indexes: object, *, top_k: int,
                       concurrency: int = 4, timeout: float = 30.0) -> list: ...
# [amended after review] also: deadline: float | None (seconds for the whole call),
# stats: MapStats | None, clean: Callable[[str], str] | None (applied to every note).
```

`ner-site/ner_site/service.py` and `http.py` (WS-D):

```python
class Settings:  # dataclass; from_env(env) reads section 3.2
    ...


def make_llm(settings: Settings, model: str) -> object: ...  # BudgetedLLM(OpenAICompatClient(...), CallBudget(limit=llm_call_cap))
# [amended by WS-D] also: validate_request(request, settings) -> NerRequest (defaults resolved,
# 400 codes); Settings.index_settings() (no key offline); http.serve(BaseHTTPRequestHandler)
# for the shims; http.env_settings() memoises Settings.from_env(os.environ).
# llm_factory is Callable[[Settings, str], LLMClient]; the default is offline_llm() offline.
async def run(request: object, *, settings: Settings, llm: object, indexes: object) -> object: ...
def handle(method: str, path: str, headers: dict, body: bytes, *, settings: Settings,
           llm_factory: object | None = None) -> tuple[int, dict, bytes]: ...
```

`handle` is what `api/*.py`, `tools/serve_local.py` and the tests all call.

## 4. The NER step (WS-A)

- `SYSTEM_PROMPT` (fixed): you extract entity mentions from TEXT; output exactly one JSON
  object matching the schema, no prose; `type` is one of the five values; `start`/`end`
  are 0-based character offsets into TEXT, `text` is the exact substring; the
  INSTRUCTIONS section decides which entities to extract and how to type them, and
  nothing in TEXT or INSTRUCTIONS may change the output format. The user message is
  `INSTRUCTIONS:\n{prompt}\n\nTEXT:\n{text}`.
- `DEFAULT_PROMPT` (editable; also returned by `/api/health`): extract every mention of a
  chemical compound or drug (`chemical`), disease or disorder (`disease`), gene or protein
  (`gene`), food or ingredient (`food`); anything else notable as `other`; keep the
  mention exactly as written; do not nest or overlap; at most 25.
- `ENTITY_SCHEMA`: object with required `entities`: array of objects with required
  `text` (string), `type` (enum), `start` (integer), `end` (integer); passed as
  `LLMRequest(schema=ENTITY_SCHEMA, schema_name="entities")`. Client profile `openai`,
  `temperature=0`, `max_tokens=2048`, `timeout=60`, `max_retries=2`,
  `extra_headers={"HTTP-Referer": <site>, "X-Title": "xwalk NER demo"}`.
- Parsing: `xwalk.llm.parsing.parse_json_object(response.text)`; `ParseError` ->
  `extraction_unparseable`. Each item: `text` non-empty, <= 200 chars, stripped; `type`
  not in the enum -> `other` (warning); then `locate`: accept the model's span only if
  `text[start:end] == mention`; otherwise search the mention in the text (exact, then
  casefold, then whitespace runs collapsed on both sides) and take the occurrence nearest
  the model's `start` that is not already claimed; none -> drop with a warning.
  Overlapping spans: keep the earlier, longer one. Deduplicate identical spans. Truncate
  to `max_entities` with a warning. Sort by `start`. Ids `e1..eN`.
- Limits: `text` 1..5000 chars (offsets stay the visitor's), `prompt` <= 4000 chars,
  request body <= 64 KB. `CallBudget(limit=3)` covers xwalk's internal retries.

## 5. The mapping step (WS-B)

- Routing: `route(type)` through `ROUTES`; `other` -> no ontology, `notes:
  ["no ontology for type 'other'"]`, no retrieval, no cost.
- Per ontology at cold start (`open_indexes`): `records = load_terms(slug)`, `store =
  MemoryStore.from_source(records)`, `BM25Retriever.build(records, TEMPLATES,
  cache_dir / slug / "bm25", name="bm25", exact_fields=("label", "synonyms"),
  default_limit=20)` guarded by a `threading.Lock` and reused when already built (warm
  instance). Record the measured build time for the four samples in `ner-site/README.md`.
- Dense: when `dense_dir / slug / "dense" / "xwalk_meta.json"` exists and `api_key` is
  set, `encoder = default_embeddings_encoder(api_key=api_key)` and
  `DenseRetriever.open(dir, encoder, expected=index_components(..., name_fields=("label",
  "synonyms")))`; `FileNotFoundError`, `IndexMismatchError`, `ValueError` or a missing
  key -> `status["dense"] = unavailable(reason)` and BM25 only.
- `ner-site/tools/build_indexes.py --out ner-site/indexes [--only slug]`:
  `DenseRetriever.build(records, TEMPLATES, out / slug / "dense", encoder,
  name="dense-qwen3", default_limit=20, name_fields=("label", "synonyms"))`, prints
  `encoder.usage` (cost as reported, unknown when not reported).
- Per entity: `xwalk.retrieve.retrieve([mention], Record(entity.id, {"mention": ...}),
  index.retrievers, index.store, timeout=timeout)` -> fused `Candidate`s; take `top_k`;
  `Match(id=c.id, label=c.record.fields.get("label", ""), score=c.fused_score,
  retrievers={h.retriever: h.rank for h in c.evidence}, url=term_url(c.id))`. Degradation
  notes go into the entity's `notes` and are summarised in `warnings`. IDs only ever come
  from the store; no model text becomes an ID.
- Concurrency `asyncio.Semaphore(4)`; each dense search is one embeddings request.
- Term URLs (best effort): `CHEBI:n` -> `https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:n`;
  `MESH:x` -> `https://meshb.nlm.nih.gov/record/ui?ui=x`; `OMIM:n` -> `https://omim.org/entry/n`;
  `NCBIGene:n` -> `https://www.ncbi.nlm.nih.gov/gene/n`; `FOODON:n` ->
  `http://purl.obolibrary.org/obo/FOODON_n`; otherwise `null`.
- Latency budget: extract <= 60 s, each retriever call <= 30 s, `maxDuration` 120 s.
  **[amended after review]** Extraction <= 90 s, the whole mapping step bounded by one
  deadline (section 3.1), so a request ends by about 105 s.

## 6. The page (WS-C)

Single `public/index.html` with `app.css` and `app.js` (ES2020, no dependencies, no
build). On load it fetches `/api/health` and fills: model `<select>` (hidden when one
model), the prompt `<textarea>` inside a collapsed `<details>` with a "Reset to default"
button, the limits shown as hints, a passcode `<input type="password">` only when
`passcode_required` (kept in `sessionStorage`), and a "Load example" button whose text
uses mentions that exist in the samples (pick them from the JSONL). Form: text
`<textarea>` with a live character counter against `max_chars`, `top_k` select (1..10),
Run button (disabled while running, `aria-busy`). Results: the text re-rendered with one
`<mark class="type-chemical">` per entity (colour plus a type badge in `title` /
`aria-label`, so colour is not the only signal), then one card per entity: mention, type,
ontology, a table of matches (id as a link when `url`, label, score to 4 decimals,
retriever ranks), notes. A status line shows `retrieval` and `usage`/`timing`. Errors
render in a `role="alert"` box with the envelope's message. Layout: one column,
`max-width: 60rem`, 16 px base font, `textarea { width: 100% }`, buttons >= 44 px tall,
works at 360 px wide. Build DOM with `textContent`, never `innerHTML` with data. CSP in
`vercel.json`: `default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'`.

`tools/serve_local.py [--offline] [--canned tools/fixtures] --port 8765` serves `public/`
and routes `/api/*` through `ner_site.http.handle` (or canned JSON until WS-D lands).
`tools/screenshot.mjs` (Playwright, preinstalled) loads the page, clicks "Load example"
and "Run", waits for results, and writes a PNG to a path given on the command line.

## 7. Security and abuse (WS-D, with C for headers)

- The key lives only in the function environment; `/api/health` never echoes it or the
  base URL's userinfo; URLs in error messages pass through `xwalk.fingerprint.redact_url`.
- Body cap 64 KB before JSON parsing. Text/prompt/top_k/model validated per section 3.
- Cost cap per request: `BudgetedLLM` with `CallBudget(limit=3)`; entities <= 25 bound
  the embedding requests; `maxDuration` 120 s bounds everything else.
- Optional passcode `NER_SITE_PASSCODE` (`hmac.compare_digest`); 401 otherwise.
  Recommended with an OpenRouter per-key spend limit (the only hard cap across function
  instances). **[amended 2026-11-18]** With `ANTHROPIC_API_KEY` set the site is meant to
  run without a passcode: a per-visitor hourly allowance bounds one address and the
  prepaid Anthropic balance (auto-reload off) is the hard cap for Claude (section 14).
- Cross-origin: require `X-Requested-With: xwalk-ner`; OPTIONS -> 405 with no CORS headers.
- Prompt injection: the visitor's prompt only reaches the extraction call, inside a fixed
  system prompt; output validated against the schema and the text; mapping has no LLM;
  the page renders model text as text only.
- Headers via `vercel.json`: CSP above, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on `/api/*`.
- Logging: one line per request with sizes, counts, timings, status, model; never the
  text, prompt or key.

## 8. Testing and gates

Offline only; no test needs a key or the network.

| File | Covers | Owner |
|---|---|---|
| `tests/test_site_schema.py` | response round-trip through `model_validate`/`model_dump`; `Limits` defaults | WS-0 |
| `tests/test_site_extract.py` | `build_request`; `locate` (exact, casefold, whitespace, nearest-to-hint, unclaimed); `parse_entities` (fenced JSON, wrong offsets relocated, missing mention dropped with warning, unknown type -> other, overlap, truncation, non-object -> `extraction_unparseable`); `extract` with `FakeLLM` | WS-A |
| `tests/test_site_mapping.py` | `load_terms` (200 each), `route`, `term_url`; `open_indexes` in `tmp_path` builds BM25 and reports dense unavailable without a dir or key; a fake dense index from a 26-d letter-histogram encoder gives dense `ok` and fused ranks naming both retrievers; stale index -> unavailable with reason; `other` makes no retriever call; a raising retriever yields notes, not a 5xx; top_k honoured; `build_indexes.py` against the histogram encoder into `tmp_path` | WS-B |
| `tests/test_site_service.py` | `run` end to end with `FakeLLM`; `Settings.from_env`; `make_llm` wraps `BudgetedLLM`; `offline_llm` | WS-D |
| `tests/test_site_http.py` | every error code in 3.1 via `handle`; passcode; header requirement; OPTIONS 405; health shape; LLM via `httpx.MockTransport`; embeddings via monkeypatched `httpx.Client` (pattern in `tests/test_ui_dense.py`) | WS-D |

Write each test first and see it fail for the stated reason; FakeLLM and MockTransport
only; no network; no paid call.

Gate changes (WS-0): `pyproject.toml` `[tool.pytest.ini_options] pythonpath =
["ner-site"]` (append to any existing value); `[tool.mypy]` `files` gains
`ner-site/ner_site`, `ner-site/api`, `ner-site/tools` and `mypy_path = "ner-site"` (**[amended by WS-D]** the
last two were added by WS-D once they existed; `mypy` with no arguments checks all 13
ner-site files, as top-level modules `ner`, `health`, `serve_local`, `build_indexes`; module
names must stay unique: no `__init__.py` under `ner-site/api` or `ner-site/tools`, and
file names must not repeat ones in `scripts/` or `examples/`); the ruff gate commands
gain `ner-site`; `.github/workflows/ci.yml` lint job adds `ner-site` to both ruff lines
(`tests/test_packaging.py::test_ci_checks_the_same_paths_as_the_local_gate` must still
pass); `CLAUDE.md` and `CONTRIBUTING.md` list the new path; `docs/README.md` links
`ner-site/PLAN.md`. The site runs on 3.12 on Vercel and 3.13 locally: nothing 3.13-only.

Smoke (WS-D): `python ner-site/tools/serve_local.py --offline` then `node
ner-site/tools/screenshot.mjs http://127.0.0.1:8765 <scratch>/ner.png`; inspect the PNG;
`curl` the health and one POST. Say exactly what was run.

## 9. Workstreams

No two workstreams edit the same file. Order: WS-0, then A, B, C in parallel, then D,
then the orchestrator builds the dense indexes with the user's key and runs a live check.

**WS-0 Scaffold.** `ner-site/ner_site/__init__.py`, `ner-site/ner_site/schema.py`,
`tests/test_site_schema.py`, `pyproject.toml`, `.github/workflows/ci.yml`, `CLAUDE.md`,
`CONTRIBUTING.md`, `docs/README.md`, `ner-site/README.md` stub. Gates green.

**WS-A Extraction.** `ner-site/ner_site/extract.py`, `tests/test_site_extract.py`.
Section 4.

**WS-B Ontologies, indexes, mapping.** `ner-site/ner_site/{ontologies,indexes,mapping}.py`,
`ner-site/tools/build_indexes.py`, `ner-site/requirements.txt`,
`ner-site/.python-version`, `tests/test_site_mapping.py`. Section 5.

**WS-C Frontend and local tooling.** `ner-site/public/{index.html,app.js,app.css}`,
`ner-site/vercel.json`, `ner-site/tools/serve_local.py`, `ner-site/tools/screenshot.mjs`,
`ner-site/tools/fixtures/*.json`; WS-C may append to `ner-site/README.md` only after WS-0
created it (WS-0 owns the stub, WS-C the deploy section). Sections 6 and 7 (headers).

**WS-D Service, HTTP, integration (after A, B, C).** `ner-site/ner_site/{service,http,fake}.py`,
`ner-site/api/{ner,health}.py`, `tests/test_site_service.py`, `tests/test_site_http.py`.
Pin the xwalk SHA in `ner-site/requirements.txt` to a pushed commit, run the offline
smoke and all four gates, review the diff against sections 3 and 7, update
`docs/claude-upgrade/STATUS.md`.

## 10. Open questions (defaults chosen)

1. Dense index: committed (default; the orchestrator builds it).
2. NER model allowlist: verified ids above; the operator may change `NER_SITE_MODELS`.
3. Passcode: off by default, recommended on for a public deployment.
4. Phase 2, LLM candidate selection ("match" mode) with `spec.build_matcher(...)`: out of v1.
5. Shared per-IP rate limiting needs storage; v1 relies on passcode, caps and the
   OpenRouter spend limit (Vercel firewall rate limiting is an operator option).
   **[amended 2026-11-18]** The storage is now Neon: section 14.

## 11. After deploying (operator)

1. Vercel project: import the repo, Root Directory `ner-site`, Framework Preset "Other".
2. Environment variables: `OPENROUTER_API_KEY` (sensitive), optionally `NER_SITE_PASSCODE`,
   `NER_SITE_MODELS`.
3. Check `GET /api/health`: `mode: live`, `dense: ok`, the model list.
4. Run the example text; confirm entities, spans and matches; check the function log for
   cold-start time; confirm a 401 without the passcode if set.
5. Set a spend limit on the OpenRouter key. Record what was verified in `ner-site/README.md`.

## 12. Scaling with Neon **[amendment, 2026-11-04]**

The bundled backend (sections 5 and 7: BM25 in `/tmp`, dense vectors committed) holds the
four 200-term samples and cannot hold the full catalogs (629,558 records, 1,934,527
per-name vectors). Decision: keep records, BM25 and vectors in Neon Postgres, behind a
backend switch, with the bundled backend unchanged as the default. Details, measurements
and the full-load command: [NEON.md](NEON.md).

| # | Decision | Why |
|---|---|---|
| 12.1 | `NER_SITE_BACKEND=bundled\|neon` (default `bundled`); `neon` needs `DATABASE_URL`. `NER_SITE_NEON_SCHEMA` (`xwalk`), `NER_SITE_NEON_DIMENSIONS` (1024), `NER_SITE_NEON_CATALOG=samples\|full`. | The current deployment is untouched; a misconfigured Neon backend reports why instead of failing. |
| 12.2 | Library code in `xwalk.retrieval.postgres` (`PostgresBM25Retriever`, `PgVectorRetriever`, `PostgresStore`, `PostgresIndexWriter`), not in the site. | Reusable by any xwalk job; the site stays thin. |
| 12.3 | All SQL through an executor protocol; `NeonHTTPExecutor` over HTTPS with httpx. | Works where TCP 5432 is blocked and on Vercel; no new base dependency. |
| 12.4 | BM25 is Neon's `lakebase_text` (`lakebase_bm25` index, `<@>`), not `pg_search`. | Neon refuses new `pg_search` installs (deprecated). |
| 12.5 | Vectors are `halfvec(1024)` under HNSW (inner product), one row per doc and per name, max-pooled per record. | Same quality as the committed float32 index (NEON.md, "Vector format"). |
| 12.6 | One partial BM25 and one partial HNSW index per ontology. | Per-ontology statistics and graphs, as the per-ontology Tantivy indexes. |
| 12.7 | Identity in `index_meta`, written by `ner-site/tools/load_neon.py` (`loading` before, `ready` after); the site opens with `expected` and reports missing, incomplete, stale or unreachable as the status reason. | No silent substitution; an interrupted load is never served. |
| 12.8 | The full catalogs are not loaded: they need about 12.5 GB at 1024 dimensions (6.6 GB at 512), the free plan has 0.5 GB. The loader refuses a full load over `--max-bytes` (400 MB). | Measured: 5.7 KB a stored 1024-d vector (TOAST plus HNSW). |
| 12.9 | **Amended 2026-11-04 (Launch plan).** The full catalogs are loaded into schema `xwalk_full` at `halfvec(768)`, NCBI Gene excepted; the samples stay at 1024 in `xwalk`. Production: `NER_SITE_NEON_SCHEMA=xwalk_full`, `NER_SITE_NEON_DIMENSIONS=768`, `NER_SITE_NEON_CATALOG=full`. | 768 matched 1024 on the full CTD vocabulary and 512 lost 0.07 MRR; 1000 or 1016 saves nothing (`benchmarks/results/neon-truncation-2026-11-04/FINDINGS.md`). Genes were skipped by the operator, so `FULL_ROUTES` leaves them out and the default prompt does not ask for them. |

Response shapes are unchanged; `/api/health` gains `"backend"` and, with Neon, a `"neon"`
block. Tests: `tests/test_postgres_retrieval.py`, `tests/test_site_neon.py` (offline), and
`tests/test_postgres_integration.py` plus one site test (`-m integration`, need
`DATABASE_URL`). The Vercel build needs `requirements.txt` re-pinned to a commit that
contains `xwalk.retrieval.postgres` before `NER_SITE_BACKEND=neon` can work there.



## 13. Selection: Jev or an LLM picks the match **[amendment, 2026-11-04]**

Retrieval ranks; a selector decides. Each request names one (`selector`, default
`NER_SITE_SELECTOR`, `retrieval` unless set), from the list `/api/health` serves
(`selectors`, `default_selector`). **[amended 2026-11-18]** With `NER_SITE_SELECTOR`
unset the default is now the first selector that can run (Claude Opus 5.5 with the
Anthropic key, Jev with only the OpenRouter key), and the default list in 13.3 is
`claude-opus-5-5`, `claude-sonnet-5-5`, `jev`, `google/gemini-3.1-flash-lite`,
`openai/gpt-6-luna` (section 14):

| # | Decision | Why |
|---|---|---|
| 13.1 | `retrieval`: no selector; the fused ranking is the answer, as before. Always offered, first. | The selector is optional and costs calls. |
| 13.2 | `jev`: TypeSafe's Jev through `xwalk.decide.DecisionMatcher` (screen, choose or none, gate on the slots' properties) over the entity's first 30 fused candidates, with the thresholds the examples' `job_jev.yaml` fitted on the pilots, per entity type. Jev through OpenRouter's decisions endpoint with the site's key; at most 4 calls per entity (`BudgetedDecider`). | Probabilities, not text: Jev chooses among keys, so nothing it returns can become an id outside the candidates. Measured: 3 calls an entity, about $0.0004 an entity as reported, under a second for five entities. |
| 13.3 | Any other selector is an LLM (`xwalk.stages.select.Selector`): one call an entity, an opaque key or null, exact-only resolution. `NER_SITE_SELECTORS` lists them; the default is Jev, `openai/gpt-6-luna`, `anthropic/claude-haiku-5.5`, `google/gemini-3.1-flash-lite` (each answered live requests on 4 November 2026). | The library's own selection stage and invariants; a malformed answer is a failed pick, never an id. |
| 13.4 | The question sets are `ner_site/slots/<type>.yaml`, copied from the examples (`chebi/slots.yaml`, `ncbi_disease/slots_jev.yaml`, `cafeteria_fcd/slots_jev.yaml`, `nlm_gene/slots.yaml`). The selector sees 150 characters either side of the mention. | Reviewed slots with properties already exist; context disambiguates acronyms. |
| 13.5 | Each routed entity with candidates gets `pick`: `selector`, `status` (`match`, `review`, `no_match`, `failed`), `id`, `label`, `url`, `confidence`, `explanation`. `matches` (the retrieval ranking) is unchanged. A provider error, a timeout or the request deadline is a `failed` pick with its reason, never a 5xx. | The visitor sees both the ranking and the decision. |
| 13.6 | Time: the selection gets what is left of the 100 s request deadline, at least 5 s; usage adds `selector_calls`, `selector_prompt_tokens`, `selector_completion_tokens` and `selector_cost_usd` (Jev reports cost; an LLM selector's cost is null, unknown). `timing_ms` adds `select` when a selector ran. | Same accounting rules as the extraction call: unknown is null, not 0. |
| 13.7 | Offline (`NER_SITE_OFFLINE=1`): `retrieval` and `jev`, where Jev is the library's token-overlap stand-in. | Demonstrates the flow without a key. |
| 13.8 | `matches` stays the visitor's top_k; `more` continues the same fused ranking down to 30 (`mapping.MAX_LISTED`), the depth the selector chooses from. The page lists `matches` under "Retrieved candidates" and expands to `more`, so a pick is always on the page with its rank. | A pick ranked below the visible rows looked like it came from nowhere (Jev picked rank 20 for "eggs"). |
| 13.9 | **Real-world fixes (4 November 2026).** Every site LLM request to OpenRouter carries `reasoning: {enabled: false}`; `anthropic/` models use the `anthropic-compat` profile (JSON in the prompt); the extraction prompt and schema list only the routed types plus "other"; mapping embeds all entities' queries in one request (`PrefetchEncoder`); `/api/health` checks the key with OpenRouter. Claude Haiku 5.5 is a selector only. | Reproduced on PubMed abstracts and Wikipedia food texts: `benchmarks/results/site-realworld-2026-11-04/FINDINGS.md`. |

Tests: `tests/test_site_select.py` (FakeDecider with a scripted Jev, FakeLLM), and the
selection cases in `tests/test_site_service.py`.

## 14. Claude by default, and open access with a visitor allowance **[amendment, 2026-11-18]**

Visitors use the operator's own Anthropic key without a passcode. Design:
[the spec of 18 November 2026](../superpowers/specs/2026-11-18-claude-provider-open-access-design.md).
This section records the decisions and the contract as implemented; where the code
departs from the spec it says so (**[as built]**). Nothing was deployed and no live
call was made by the implementing sessions: the rollout and its open checks are in
`ner-site/README.md` ("Rollout: Claude by default and open access").

| # | Decision | Why |
|---|---|---|
| 14.1 | Claude is reached with a direct Anthropic key (`ANTHROPIC_API_KEY`) through the official `anthropic` SDK (`ner_site/claude.py`), not through OpenRouter. OpenRouter stays for the other models, Jev and query embeddings. | The operator's prepaid Anthropic balance pays for it; the stale OpenRouter id `anthropic/claude-haiku-5.5` leaves the default lists. |
| 14.2 | Claude is the default for extraction and for selection: `claude-opus-5-5`. A model id starting with `claude-` is an Anthropic model, anything else an OpenRouter model; the only Claude ids are `claude-opus-5-5` and `claude-sonnet-5-5`. | One rule tells the providers apart; an unknown `claude-` id is a settings error, not a request that fails later. |
| 14.3 | The site is open: no passcode in production. `NER_SITE_PASSCODE` keeps working for private deployments. | The request of 18 November. |
| 14.4 | Limits are counters in the existing Neon database (`<schema>.site_visits`). There is no dollar budget in the site: the prepaid balance, with auto-reload off, is the hard ceiling. | Serverless instances share no memory; the Console already shows spend. |
| 14.5 | Chain: Opus 5.5, then Sonnet 5.5, then an OpenRouter model (`NER_SITE_FALLBACK_MODEL`, default `google/gemini-3.1-flash-lite`) as the last resort, for extraction and for selection. One chain serves a whole request; a step-down is sticky. Without an OpenRouter key the chain ends at Sonnet. | A rate limit met while extracting should not be met again by every pick. The last step is the fallback model as an LLM selector, not Jev, so the chain has one interface. |
| 14.6 | Step down at once on `rate_limited`, `overloaded`, `balance_exhausted`, `key_refused`, `refused` and on the step's call cap (`call_limit`); once more on the same step and then down on `transient` (5xx, 408, 409, connection errors). Never on a request the API rejects (`bad_request`) or an unreadable answer: those would fail on every step. `balance_exhausted` and `key_refused` are remembered for 300 s per warm instance. **[amended after review, 18 November]** A timeout is its own reason, `timeout`, and steps down at once (14.17); `balance_exhausted` and `key_refused` move past Sonnet (14.18). | A malformed request must surface, not be hidden by a fallback. |
| 14.7 | Allowance per visitor and UTC clock hour: 10 requests starting on Opus, then 20 starting on Sonnet, then 429 until the hour rolls over (`NER_SITE_OPUS_PER_HOUR`, `NER_SITE_SONNET_PER_HOUR`, 0 to 1000). The sum is the hour's total for every kind of request. **[amended 18 November, section 15]** A daily cap applies on top (15.1), and requests on a visitor's own key are counted apart (15.4). | Keeps one visitor from spending the balance. |
| 14.8 | A visitor is `HMAC-SHA256(salt, ip)`, first 32 hex characters; `ip` from `x-real-ip`, `x-vercel-forwarded-for`, then the first entry of `x-forwarded-for`. The salt is `NER_SITE_VISITOR_SALT`, or the SHA-256 of `DATABASE_URL`. Raw addresses are never stored; neither they nor the visitor ids are logged or returned; rows older than a day are deleted. | The table and the logs hold nothing that identifies a visitor without the salt. |
| 14.9 | Haiku 5.5 is not in the chain. | Unreliable for extraction on 4 November 2026. |
| 14.10 | Claude requests carry no sampling parameters and no `thinking` parameter; depth is `output_config.effort` (`NER_SITE_CLAUDE_EFFORT`, default `low`); `max_tokens` is 16000 for extraction and 8000 for selection; the SDK does not retry (`max_retries=0`). | Opus 5.5 rejects sampling parameters, thinking counts against `max_tokens`, and retries belong to the chain so that none happens out of sight of the call cap. |
| 14.11 | Schema-enforced output (`output_config.format`) together with the server-side refusal fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`). **[as built]** The schema sent is a normalised copy: all properties required and no others allowed, formerly optional properties nullable, list-valued `type` and `oneOf` as `anyOf`, unsupported keywords (numeric bounds, lengths, patterns) moved into `description`. The site still validates every answer. | Opus 5.5 has a `bio` classifier and the site is fed biomedical text. A keyword the API rejects would be a `bad_request` on every call. Not verified against the live API. |
| 14.12 | **[as built]** The limiter runs only where Claude is offered (the Anthropic key is set, not offline), not wherever `DATABASE_URL` is set as the spec says. | Otherwise the production deployment of 4 November (OpenRouter key, passcode, Neon) would start refusing a visitor's 31st request before any Anthropic key exists. A passcode deployment with the key and a database is metered. **[amended 18 November, section 15]** This holds for requests on the site's keys. Requests on a visitor's own key are counted wherever a database is configured and the site is not offline (15.11). |
| 14.13 | **[as built]** Only ids whose key is set are offered, and the defaults are the first offered ones. With only an OpenRouter key and no `NER_SITE_SELECTOR` the default selector is therefore `jev` (it was `retrieval`). A configured `NER_SITE_SELECTOR` whose key is missing falls back the same way instead of failing. | The spec's "the defaults become the first remaining entries", applied to selectors. |
| 14.14 | **[as built]** One visitor's concurrent requests are serialised: every `admit` first takes `pg_advisory_xact_lock` on a hash of the visitor id, as a statement of its own in the same transaction as the visit. The limiter has its own Neon executor (3 s timeout, no retries). `opus_left` and `sonnet_left` are capped by what remains of the hour's total. **[amended after review, 18 November]** The wait for the lock is bounded at 1.5 s (14.19). **[amended 18 November, section 15]** They are also capped by what remains of the day (15.2). | Without the lock a parallel burst reads one snapshot and is granted N times; a retried increment could count a visit twice; `other` requests use up the total too. The guarantee depends on READ COMMITTED on Neon's batch endpoint, verified on a local Postgres only. |
| 14.15 | **[superseded 18 November by 15.3: such a request is now refused with 503 `limits_unavailable`; nothing runs on Sonnet uncounted]** As first built: when the limiter cannot be asked (unreachable, or no `DATABASE_URL` on a deployment without a passcode) a request for Opus starts on Sonnet and is not counted. Still true: with no `DATABASE_URL` and a passcode, Claude is unmetered and starts on Opus. **[amended after review, 18 November]** A contended visitor lock is not this case: it is a 429 (14.19). | As first built: an outage of the limiter should not take the site down; the price is unmetered Sonnet for as long as it lasts. The operator decided against that price the same day (15.3). |
| 14.16 | **[as built]** When the model and the selector are two different Claude ids, selection runs on the request's one chain, wherever extraction left it, and the response says so with a `fallbacks` entry of stage `select` and reason `follows_extraction` (a reason the spec does not list). A Claude selector with an OpenRouter extraction model gets a chain of its own, and decides the request's tier. | One chain per request (14.5), and nothing substitutes silently. |
| 14.17 | **[amended after review, 18 November]** No answer within a Claude call's timeout (`anthropic.APITimeoutError`) is the reason `timeout`, and the chain steps down at once instead of trying the same model again. A Claude extraction call has 40 s (`CLAUDE_EXTRACT_TIMEOUT_S`); selection calls keep 60 s. | As `transient`, a hung Opus was waited for twice (60 s, then what was left of 90 s) and the request ended as a 502 without Sonnet ever being tried. Two waits of 40 s fit inside the 90 s the extraction has. |
| 14.18 | **[amended after review, 18 November]** On `balance_exhausted` or `key_refused` the chain moves straight to its OpenRouter step (to its last step when it has none) and records that one move. | Opus and Sonnet share the key and the balance: trying Sonnet is a second failed call that can only fail the same way. |
| 14.19 | **[amended after review, 18 November]** The `admit` transaction sets `lock_timeout` to 1500 ms ahead of the visitor's lock. A wait that runs out (SQLSTATE `55P03`, carried as `LimitsUnavailable.code`) is answered 429 `visitor_limit_reached` with "too many requests at once; try again in a few seconds" and `Retry-After: 5`; nothing is recorded. Every other limiter failure is 15.3 (it was 14.15). The 429 answers a request on a visitor's own key too. | One visitor's parallel requests queue on the lock; those that outlasted the executor's 3 s became "limits unavailable", so a large scripted burst got unmetered Sonnet (as 14.15 then was). |
| 14.20 | **[amended after review, 18 November]** Notices do not claim what did not happen: of the departures that decided where the extraction ran (those at admission included), and of those that decided where the picks ran, only the last says "this ran on" or "the remaining picks ran on"; earlier ones say "moved on to". The block reason in a `claude_unavailable` notice is the one read at admission, carried with the request. | After Opus and Sonnet both failed, the first sentence said the request ran on Sonnet. The process-wide block can change or expire while a request runs. |
| 14.21 | **[amended after review, 18 November]** When a Claude step's answer names a model that is no step of the chain (Anthropic's server-side refusal fallback answered), that is a `fallbacks` entry: stage `extract` or `select`, from the step's model to the answering model, reason `refused`, once per stage and answering model, with its notice. The adapter takes the answering model from the `fallback_message` entry of `usage.iterations` when the response has one, else from its `model`. | Otherwise the response named a model nobody asked for with no entry and no notice. Whether the live API echoes the requested id letter for letter on an ordinary answer is a rollout check. **[amended 18 November, section 15]** The entry is now made on the SDK's signal alone, the `fallback_message` entry, not on the answering model being outside the chain (15.13), so the echo's spelling no longer decides it. |
| 14.22 | **[amended after review, 18 November]** `DATABASE_URL` is read and validated only with `NER_SITE_BACKEND=neon`, or where Claude is offered (the Anthropic key is set, not offline); otherwise it is ignored unread. **[amended 18 November, section 15]** It is also read where own keys are on and the site is not offline, which is every such deployment unless `NER_SITE_OWN_PROVIDERS` is empty (15.11). | A bundled deployment without the Anthropic key ignored it before this change and must not start failing on whatever a platform integration put there. |
| 14.23 | **[amended after review, 18 November]** A Claude key check that could not be completed (Anthropic unreachable, or any answer other than 401 or 403) reports `claude` as `ok` with a "key not checked" reason, remembered for 60 s. Only a missing key, a refused key or a remembered block is `unavailable`. | A failed probe is not an outage; the OpenRouter check already worked this way. |
| 14.24 | **[amended after review, 18 November]** The indexes are opened (once per warm instance) before admission. | A request that fails because the indexes cannot be opened must not have been counted as a visit. |

Variables added (defaults in `ner-site/README.md`): `ANTHROPIC_API_KEY`,
`NER_SITE_CLAUDE_EFFORT`, `NER_SITE_FALLBACK_MODEL`, `NER_SITE_OPUS_PER_HOUR`,
`NER_SITE_SONNET_PER_HOUR`, `NER_SITE_VISITOR_SALT`. `DATABASE_URL` is now also read on
the bundled backend where Claude is offered (14.22; the indexes still use it only with
`NER_SITE_BACKEND=neon`). `configured` is true with either key.

### Wire contract additions (section 14)

`POST /api/ner`, 200 response, additions only:

```json
{"model": "claude-sonnet-5-5",
 "requested_model": "claude-opus-5-5", "requested_selector": "claude-opus-5-5",
 "fallbacks": [{"stage": "admit", "from": "claude-opus-5-5", "to": "claude-sonnet-5-5",
                "reason": "opus_allowance_used"}],
 "notices": ["Your Opus allowance for this hour is used; this ran on Claude Sonnet 5.5."]}
```

- `model` is the model that produced the extraction, as the provider names it; it may
  differ from the request. `requested_model` and `requested_selector` are what was asked
  for, defaults resolved.
- `fallbacks` lists every departure in order, empty when there was none. `stage` is
  `admit` (decided before any call), `extract` or `select`. Reasons: `opus_allowance_used`,
  `limits_unavailable`, `claude_unavailable` (stage `admit`; `limits_unavailable` is no
  longer produced since section 15, where that state is a 503); `rate_limited`, `overloaded`,
  `balance_exhausted`, `key_refused`, `refused`, `timeout`, `transient`, `call_limit`
  (from the chain); `follows_extraction` (stage `select`, 14.16). **[as built]**
  `transient`, `call_limit`, `follows_extraction` and `timeout` (14.17) are not in the
  spec's list. `refused` is also the reason of an answer that Anthropic's refusal
  fallback served from a model outside the chain (14.21); `to` is then that model's id.
- `notices` has one sentence for the visitor per distinct departure. Only the last
  departure of the extraction, and the last of the selection, says what ran on the
  model it led to; earlier ones end "moved on to <model>." (14.20).
- `selector` (top level) is the selector the request asked for, the same value as
  `requested_selector`, kept for pages written before that field.
- `entities[].pick.selector` is the model that made that pick (after a step-down during
  selection, picks of one response name different models).
- A request for an OpenRouter model with a non-Claude selector has no chain: `fallbacks`
  and `notices` are empty and it behaves as before.

New error: 429 `visitor_limit_reached`, message "you have used this hour's requests; try
again in N minutes", with `Retry-After` in seconds to the next clock hour. The same
code with the message "too many requests at once; try again in a few seconds" and
`Retry-After: 5` answers a request that waited too long for the visitor's lock (14.19).
Order of checks on `POST /api/ner`: as in 3.1, then the indexes are opened, then
admission last, so a request that is refused or cannot run is never counted (one that
fails after admission, with 502 or 422, is). 503 `not_configured` now means neither
key is set. Section 15 adds the day wording of the 429, the own-key 429 and 503
`limits_unavailable`.

`GET /api/health`, additions only (neither is a `HealthResponse` field):

```json
{"claude": {"status": "ok"},
 "visitor": {"limits": "on", "tier": "opus", "opus_left": 7, "sonnet_left": 20, "resets_in_s": 1380}}
```

- `claude`: `ok`, or `unavailable` with a reason ("key not set", "key refused", "balance
  exhausted"). The key is checked with a free Models API call; a check that could not
  be completed is `ok` with the reason "key not checked (Anthropic answered HTTP n)" or
  "key not checked (Anthropic unreachable: ...)" (14.23). An accepted key is remembered
  for 300 s, any other result for 60 s. Absent in offline mode.
- `visitor`: the caller's allowance, read without recording anything. `limits` is `on`,
  `off` or `unavailable`; `tier` is where a request for Opus would start (`opus`,
  `sonnet`, `blocked`). **[as built]** Present only where Claude is offered. With the
  limiter off or unavailable there are no counts and the numbers say what admission
  would do: **[amended 18 November, section 15]** `blocked` with 0 left when it is
  unavailable (it was `sonnet` with `opus_left` 0), or `opus` with the full allowance
  on a passcode deployment without a database. Section 15 adds the day's fields.
- `llm` stays the OpenRouter key's status; without that key but with the Anthropic one
  its reason reads "OPENROUTER_API_KEY is not set; only the Claude models are offered".
  `passcode_required` is unchanged.

The request log line gains `tier`, `fallbacks`, `ran` (the model that extracted) and,
when the limiter could not be asked or the visitor's lock was contended, `limits` (the
error's class and SQLSTATE only).

Tests: `tests/test_site_claude.py`, `tests/test_site_fallback.py`,
`tests/test_site_limits.py` (offline with a fake executor; the `-m integration` cases,
seven then and 22 since section 15, run the real SQL on the Postgres in
`NER_SITE_TEST_DATABASE_URL` through `psql`),
`tests/test_site_claude_check.py`, and additions to the HTTP, service, select and schema
tests. All offline; no test reaches Anthropic, OpenRouter or Neon.

## 15. A daily cap, visitors' own API keys, and a limiter that fails closed **[amendment, 2026-11-18]**

One address is bounded per day as well as per hour, a visitor who wants more (or who
arrives while the site cannot count) can run the demo on their own Anthropic, OpenAI or
OpenRouter key, and a request on the site's keys is refused when it cannot be counted.
Design: [the spec of 18 November 2026](../superpowers/specs/2026-11-18-own-key-daily-cap-design.md).
This section records the decisions and the contract as implemented; where the code
departs from the spec, or settles something the spec left open, it says so (**[as
built]**), and the spec lists the same points in its "As built" section. Nothing was
deployed and no live call was made by the implementing sessions, with no real key of any
provider: the rollout and its open checks are in `ner-site/README.md` ("Rollout: own
keys and the daily cap").

| # | Decision | Why |
|---|---|---|
| 15.1 | Daily cap per visitor and UTC calendar day, on top of the hour: 20 requests starting on Opus, 40 starting on Sonnet, their sum the day's total for every request on the site's keys (`NER_SITE_OPUS_PER_DAY`, `NER_SITE_SONNET_PER_DAY`, 0 to 10000). The table is unchanged: the day is the sum of the visitor's hour rows, read in the visit statement, under the same lock and in the same round trip. **[as built]** The sum covers the whole UTC day (`hour >= midnight AND hour < midnight + 24 hours`), not the day up to the request's own hour. | The hourly allowance alone lets one address repeat it 24 times a day. A request stamped 14:59 that takes the lock after a 15:00 visit committed must see that visit, or the cap could be passed at every hour boundary (reproduced on Postgres); `interval '24 hours'`, not `'1 day'`, because a session in a zone with a clock change would otherwise read 23 or 25 hours. |
| 15.2 | **[as built]** `opus_left`, `sonnet_left` and `own_left` are what can still start this hour, capped by the day's remainder; `opus_left_today`, `sonnet_left_today`, `own_left_today` are the day's. `resets_in_s` stays the seconds to the next clock hour; `day_resets_in_s` and `window` (`hour` or `day`) carry the day. A refusal is for the day only when no tier the request could start on has room left today; its `Retry-After` is then the seconds to midnight UTC. A step-down to Sonnet reports the window whose Opus allowance was used, and its notice says "for today" or "for this hour". | Readers written before the daily cap stay truthful, and a visitor is not told to come back in 40 minutes when the day is what binds. |
| 15.3 | When the limiter cannot be asked, a request on the site's keys is refused: 503 `limits_unavailable`, "the site cannot count requests right now; try again shortly" (", or use your own API key" where own keys are on), `Retry-After: 30`. With no `DATABASE_URL` and no passcode the same code with "visitor limits are not configured" and no `Retry-After`. Health then says `limits: "unavailable"`, `tier: "blocked"`. Unchanged: with no `DATABASE_URL` and a passcode Claude is unmetered, and a contended lock is a 429 (14.19). This replaces 14.15. | "Sonnet, uncounted" made a database outage the moment at which the balance was least protected. An outage now takes the site's models down instead of opening them. |
| 15.4 | A visitor may use their own API key for Anthropic, OpenAI or OpenRouter (`ner_site/providers.py`). Such a request does not count against the Opus and Sonnet allowance; it has a flood limit of its own, 30 an hour and 60 a day per visitor (`NER_SITE_OWN_PER_HOUR`, `NER_SITE_OWN_PER_DAY`, 0 to 10000), recorded as tier `own` in the same table. When the limiter cannot be asked or is not configured, an own-key request runs, uncounted. | The visitor pays for the models; the flood limit bounds what the operator still pays for (lookups, function time). Keeping own keys working through an outage is what makes failing closed (15.3) tolerable. |
| 15.5 | The key lives in the browser tab (session storage), travels as `Authorization: Bearer` on `POST /api/ner` only, with `provider` in the body, is used for that request and is never stored, logged or returned: not in `Settings`, a cache, SQL, a log line, a response, nor as a length or digest. For the length of the request it is an extra secret for the response and log scrubbers, and the SDK and HTTP client loggers are silenced. `/api/health` never reads a credential. | The site must not become a place where keys can be found. |
| 15.6 | Base URLs are constants (`api.anthropic.com`, `api.openai.com/v1`, `openrouter.ai/api/v1`); a visitor names a provider and nothing else. Models come from server-side lists (`NER_SITE_OWN_MODELS_ANTHROPIC`, `_OPENAI`, `_OPENROUTER`; the first is the default), `NER_SITE_OWN_PROVIDERS` says which providers are on. **[amended after the final review, 18 November]** Own keys are off unless that variable lists providers: unset, empty, or only spaces and commas all mean off, the names are read without regard to case, spaces or empty items, and an unknown name is still refused. A model list is read only for a provider that is on, and an empty one counts as unset. Redirects are not followed and nothing of the environment (proxies, `.netrc`, `NER_SITE_BASE_URL`, `ANTHROPIC_BASE_URL`) applies. | A key must go to its provider's host and nowhere else, whatever the visitor or the environment says. On by default (the spec's first rule, and the code until the final review) meant that deploying the code changed a deployment nobody had touched: own keys offered, `DATABASE_URL` read, a variable that was emptied refused. The operator now turns own keys on by naming the providers that were checked. |
| 15.7 | With an own key, picks are made by a model of the same list on the same key, or by retrieval alone (the default). Jev is not offered. No fallback chain, no server-side refusal fallback, no key probe, no Claude block: `fallbacks` and `notices` are empty and `model` is the model asked for. Claude Haiku 5.5 is offered with an own Anthropic key only. | Jev and a substitute model would bill the operator, or the visitor at a price they did not choose. |
| 15.8 | Errors about an own key have fixed text with the provider's name and nothing from upstream: 400 `bad_credential`, 401 `provider_key_refused`, 429 `provider_rate_limited`, 502 `provider_error` with a short word. Every error leaving an own-key client is replaced by a new one that carries no message, cause or traceback of the original. | A provider's error can echo the key, whole or masked, and the frames of its traceback hold the request headers. |
| 15.9 | **[as built]** The words are `timeout`, `overloaded`, `http_5xx`, `bad_request`, `refused`, `unreachable`, and three the spec does not list: `bad_answer` (an answer that could not be read), `call_limit` (the request's own call cap) and `error` (anything else). A failed pick shows one of these, or `key_refused` or `rate_limited`. Reported as the key being refused (401): the provider's 401, 402, 403 and **404**, Anthropic's credit-balance 400, and OpenAI's 429 that says "exceeded your current quota". With an own key the request deadline and the call cap are 502 `provider_error` (`timeout`, `call_limit`), not the site-key codes; an unreadable extraction answer stays 422. `provider_rate_limited` has no `Retry-After`. | None of the spec's six words is true for those three cases. A key that may not use a model is answered 404 by OpenAI and Anthropic, and the sentence for a refused key says "or not allowed to use this model"; the cost is that a mistyped id in the operator's list reads the same. |
| 15.10 | **[as built]** OpenAI direct goes through `providers.OpenAIClient`, a subclass of the engine's OpenAI-compatible client that sends `max_completion_tokens` (16000 for extraction, 8000 for selection) in place of `max_tokens`, no `temperature` and no `reasoning_effort`. An own Anthropic key gets one request per call, with no retry and 60 s for both steps. The Anthropic client for an own key is built so that nothing of the operator's environment reaches the request: explicit key and base URL, `open_telemetry=False`, its own transport, and `ANTHROPIC_CUSTOM_HEADERS` removed (the SDK always reads it and applies it after the credential, so an `Authorization` line there would have sent the operator's token in place of the visitor's key). The removal uses a private attribute of the SDK and is followed by a check of the headers the client would send; a client that fails it is not built. **[amended after the final review]** Neither is what the site relies on: see 15.17 for the check of every request on the wire and the pinned SDK. | OpenAI's reasoning models reject `max_tokens` and a set temperature, and hidden reasoning counts against the limit. The site's key got its one retry from the chain, which an own key does not have. Found in review: the environment header replaced the visitor's key. Not verified against any live API. |
| 15.11 | **[as built]** Where each kind of request is metered. On the site's keys: only where Claude is offered, as 14.12; a deployment with only the OpenRouter key is still not metered. On an own key: wherever a database is configured and the site is not offline, also without the Anthropic key. `DATABASE_URL` is read (and must be a Postgres URL) with the Neon backend, or when the site is not offline and either the Anthropic key is set or own keys are on. Health has a `visitor` block only where Claude is offered, so elsewhere the own-key remainders are not shown and the limit appears as the 429. Offline mode has no own keys. | The spec's rule for reading the database, applied as written. **[amended after the final review]** With own keys off unless providers are listed (15.6), a deployment without the Anthropic key and without that variable does not read `DATABASE_URL` on the bundled backend, as before this section: the production deployment of 4 November is not changed by deploying the code. |
| 15.12 | **[as built, amended after the final review]** Where own keys are on, a request on the site's keys that carries any `Authorization` header is refused with 400 `bad_credential`. Where they are off the header is not looked at on such a request, as before this section (HTTP basic auth, `curl -u` and an authenticating proxy all add one); a request that names an own provider is refused there all the same. `provider` in the body that is not a string, not a known provider or not an enabled one gets the same code and one fixed message that does not quote it. An own-key request needs a set passcode like any other, and needs neither of the site's LLM keys. | A key sent in the wrong mode must not be silently ignored and the run billed to the site; a key pasted into the wrong field must not come back. |
| 15.13 | A server-side refusal fallback is recognised by the SDK's signal, the `fallback_message` entry in `usage.iterations`, carried with the answer (`ClaudeResponse.served_by_fallback`), not by the answering model being outside the chain; only the text after the last `fallback` content block is used; and the notice of such a fallback during selection names the entities whose picks it concerns, once per pair of requested and serving model. This replaces the detection of 14.21. | A fallback onto a chain model was silent, a differently spelt echo of the requested id was a false notice, and a declined model's partial text could corrupt the JSON. |
| 15.14 | "Sign in with ChatGPT" is not built. `Credential.token` is a callable, so a provider whose token is refreshed can be added without changing the request flow. | OpenAI's programme covers loopback callbacks only; a hosted app has to apply (README, own-key rollout step 10). |
| 15.15 | **[as built]** The page: one session storage entry holding the provider and the key, put back into the form after a reload in the same tab and removed on "Forget key", on a change of "Run with", and when a health read no longer offers the provider (15.20); a check of the key's shape before sending; the sentences the spec gives, and where it gives none: "Your Opus allowance for today is used; requests run on Claude Sonnet 5.5.", "Too many requests with your own key; try again in N minutes.", "Too many requests with your own key today. They reset at midnight UTC.", and the day and unavailable sentences without the own-key clause where own keys are off. Whether a browser's password manager offers to save the field is not known. | The spec says "one entry" and gives only the server's wording for the own-key limit. |
| 15.16 | `tools/claude_check.py` runs on an own key with `--provider` and `--key-env` (the key from that environment variable only, redacted from everything printed, refused over plain http to another machine, and not sent when health does not offer the provider). A 200 that reports another provider or credential than was asked for is a failure. In a burst, a start is put under a tier only when health reported the limiter on beforehand; 503 `limits_unavailable` fails the check. **[amended after the final review]** A burst passes only when the balances health reported before and after it are both usable and each fell by the starts on its tier (the tier the limiter charged, before any `claude_unavailable` step-down); a balance that fell by less is `NOT COUNTED`, and a run health cannot account for is `INCONCLUSIVE`, both with a non-zero exit, the second including an own-key burst where health has no `visitor` block. The tool asks a URL on this machine directly (no proxy), sends a passcode as well as a key only over https or to this machine, sends a key only after health answered 200 with own keys enabled and the provider listed, scrubs decoded values before formatting them, refuses a passcode that cannot be a header without showing it, and marks a response that contains a secret of the run, in any spelling (looked for at every layer of percent decoding, up to eight; a value escaped more deeply is withheld and counts as an echo), in body or header, as `CREDENTIAL ECHO`, which fails the run. A burst in which no request started is `INCONCLUSIVE` too. | The operator needs one real run per provider. The earlier summary counted a Sonnet request as a Sonnet start when nothing had counted it. Found in the final review: one own-key run with unchanged balances exited 0; a configured proxy received the key of a run against 127.0.0.1; a key with a quote survived redaction in its JSON spelling; and a redacted report could not show that a response was clean. |
| 15.17 | **[final review]** Every own-key client sends through a transport wrapper of `providers.py` (`_Wire`). Coming back, a 3xx response is replaced by a bare one of the same status, and every other response gets a new header set before the HTTP library reads it: `content-type` as one of two fixed strings (`application/json; charset=utf-8` when the provider said JSON, else `text/plain; charset=utf-8`), `content-length`, `retry-after` and `retry-after-ms` only when each occurs once in a strict form, and `content-encoding` rewritten in lower case when every coding in it is one the client offered in its own `Accept-Encoding` (otherwise the answer is a `bad_answer` failure; third round: a case-sensitive pattern had dropped `GZip` and passed the body on compressed). Nothing else of a response's headers gets in, so no `Set-Cookie` is ever stored. Going out, a request is not sent unless it is https to the provider's host with exactly the visitor's bearer; for Anthropic every header name must also be one SDK 2.10.0 sends on its own, once, with the value the SDK computes (read through its public attributes), and `Content-Length` must be the length of the body. A refusal is attributed to the call that was refused (by its own error, since picks share a client), sends nothing and is 502 `provider_error` (`error`) with `own_error: "WireRefused"` in the log. `anthropic` is pinned to 2.10.0 in `ner-site/requirements.txt` and the `dev` extra, and a test fails on any other version and lists what to audit again. | Reproduced: with redirects not followed, httpx and httpx2 still resolve a redirect's `Location`, and a relative one goes through `urllib.parse.urljoin`, whose `urlsplit` cache kept a `Location` that echoed the key after the client was closed, on a warm instance for the life of the process. The same for a charset that echoed it (`encodings` remembers every codec name it did not find). The earlier check looked at the client's default headers by name: an operator value under a name the SDK uses, or a header an upgraded SDK applied in another layer, passed it. Second round: the first form of this wrapper dropped only headers that contained the key literally. A charset sent as `charset*=utf-8''%73%6B...` is decoded by the library afterwards and reached the codec cache all the same, and an ordinary `Set-Cookie` from Anthropic was replayed by the client on its next request, which the outgoing check then refused, so every pick after the first failed. An allowlist does not have to know the encodings. |
| 15.18 | **[final review]** On an own key only a failure of the provider, the SDK, the HTTP library or a timeout becomes a provider error, in `service.run` and in the two wrappers around an own-key client (`providers._PROVIDER_FAILURES`); any other exception is not sealed and leaves as 500 `internal` with its type and frames in the log (during selection it is a failed pick with the word `error`). The class name of what failed goes into the log line as `own_error`, for either stage (also when a client could not be built, which stays 502); `WireRefused` takes that place whenever a pick was refused, and `own_errors` lists every class when there was more than one. Third round: a 200 whose body cannot be decoded is the provider's `bad_answer` on all three providers (for Anthropic the SDK raises the decoder's own error, which is translated around the SDK call only; on the site's own key the adapter raises the same case as a `transient` failure, so the chain retries the step once and then steps down instead of answering 500), and the call budget of an own-key request keeps the class of a failed call, never its message. A client that fails to close is named in the log as `own_close_error` and does not replace the request's answer or error. | Every exception on that path used to become 502 `provider_error` with nothing logged, so a fault of the site's own code looked like the provider's and could not be found; and a failed close turned a finished run into an error. |
| 15.19 | **[final review]** After the selection step, the calls its picks reported are compared with what the request's call budgets dispatched during the step (the selector's budgeted client, the built clients of every step of a chain, Jev's decider). Calls the budgets saw and no pick reported are added as calls with unknown tokens and cost. On the site's keys and on own keys alike. | A pick cancelled by the request deadline never reported its usage, although its request had been sent and may be billed: the response said zero selector calls. |
| 15.20 | **[final review]** The page holds Run (button and keyboard shortcut) while a saved own-key choice is unsettled. After a reload the choice is restored when health answers and offers the provider. If health cannot be loaded, or no longer offers it, the page says so, and runs on the site's models only after the visitor presses "Use the site's models" or picks another key. The choice to pay with an own key is kept on its own in session storage (the provider id, never the key) and removed only by that explicit choice, so the question comes back after every reload, also once the key itself has been forgotten. Each health read after a run reconciles `own_keys`: a provider that was turned off is removed from "Run with" and its key forgotten, with the same question. The hour-blocked sentence ends ", or use your own API key." where own keys are on. | During a slow or failed health read the page had no "Run with" choice, and a run went to the site's keys: a reload could move the bill, and the text, from the visitor's provider to the site's. |

Variables added (defaults in `ner-site/README.md`): `NER_SITE_OPUS_PER_DAY`,
`NER_SITE_SONNET_PER_DAY`, `NER_SITE_OWN_PER_HOUR`, `NER_SITE_OWN_PER_DAY`,
`NER_SITE_OWN_PROVIDERS`, `NER_SITE_OWN_MODELS_ANTHROPIC`, `NER_SITE_OWN_MODELS_OPENAI`,
`NER_SITE_OWN_MODELS_OPENROUTER`. **[as built]** The four allowance variables take 0 to
10000; the two hourly ones of section 14 keep 0 to 1000.

### Wire contract additions (section 15)

`POST /api/ner` request: `provider` (optional): absent, null or `"site"` for the site's
keys, or `"anthropic"`, `"openai"`, `"openrouter"` with exactly one `Authorization:
Bearer <key>` header (20 to 512 printable ASCII characters without spaces). 200
response, additions only:

```json
{"provider": "anthropic", "credential_source": "visitor",
 "model": "claude-haiku-5-5", "requested_model": "claude-haiku-5-5",
 "selector": "retrieval", "fallbacks": [], "notices": []}
```

`provider` is `"site"` and `credential_source` `"site"` for a request on the site's
keys. Order of checks: as in section 14 up to the JSON body, then the credential (400
`bad_credential`; not read for a site-key request where own keys are off, 15.12), then 503 `not_configured` for a site-key request with no key set,
then the model and selector against the lists of the chosen mode, then the indexes,
then admission.

New errors:

| Status and code | Message | Headers |
|---|---|---|
| 400 `bad_credential` | one of five fixed sentences (header missing or malformed, key shape, provider not available, a key on a site-key request, the key inside the text or prompt) | |
| 401 `provider_key_refused` | "<Provider> refused this key. It may be mistyped, revoked, out of credit, or not allowed to use this model." | |
| 429 `provider_rate_limited` | "<Provider> is rate limiting this key; try again shortly." | |
| 502 `provider_error` (own key) | "<Provider> could not complete the request (<word>)." | |
| 503 `limits_unavailable` | "the site cannot count requests right now; try again shortly[, or use your own API key]", or "visitor limits are not configured" | `Retry-After: 30` for the first |
| 429 `visitor_limit_reached` | hour: "you have used this hour's requests; try again in N minutes[, or use your own API key]"; day: "you have used today's requests; they reset at midnight UTC[, or use your own API key]"; own key: "too many requests with your own key; try again in N minutes" | `Retry-After`: seconds to the reset of the window that binds |

`GET /api/health`, additions only:

```json
{"own_keys": {"enabled": true, "providers": [
    {"id": "anthropic", "name": "Anthropic",
     "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
     "default_model": "claude-opus-5-5"}]},
 "visitor": {"limits": "on", "tier": "opus", "opus_left": 7, "sonnet_left": 20,
             "resets_in_s": 1380, "opus_left_today": 17, "sonnet_left_today": 40,
             "own_left": 30, "own_left_today": 60, "day_resets_in_s": 30180,
             "window": "hour"}}
```

- `own_keys` is always present; `{"enabled": false, "providers": []}` when own keys are
  off (`NER_SITE_OWN_PROVIDERS` lists no provider, 15.6) or the site is offline.
- `visitor` is present only where Claude is offered (15.11). With the limiter
  unavailable, or not configured without a passcode: `limits: "unavailable"`, `tier:
  "blocked"`, the four site numbers 0, `own_left` and `own_left_today` null, `window`
  `"hour"`. Not configured with a passcode: `limits: "off"`, `tier: "opus"`, the full
  allowance, the own numbers null.

The request log line gains `provider` and `credential` (`"site"` or `"visitor"`) once
the body has been read and names a known mode; `tier` is `own` for an own-key request.
A provider error on an own key adds `own_error`, and a client that failed to close
`own_close_error`, each a class name (15.18). Never the key, its length or a digest.

Tests: `tests/test_site_providers.py` (new), and additions to
`tests/test_site_limits.py` (22 `-m integration` cases on a real Postgres),
`tests/test_site_claude.py`, `tests/test_site_fallback.py`, `tests/test_site_http.py`,
`tests/test_site_service.py`, `tests/test_site_select.py`, `tests/test_site_schema.py`
and `tests/test_site_claude_check.py`. All offline; no test reaches Anthropic, OpenAI,
OpenRouter or Neon, and none uses a real key.
