"""Wire contract of the NER demo site: request, response and error shapes.

Binding: docs/ner-site/PLAN.md section 3. Every response model serialises with
``model_dump(mode="json")`` to exactly the shapes below.

``GET /api/health`` -> 200::

    {"ok": true, "mode": "live", "default_model": "openai/gpt-4o-mini",
     "models": ["openai/gpt-4o-mini", "google/gemini-2.5-flash-lite"],
     "limits": {"max_chars": 5000, "max_prompt_chars": 4000, "max_entities": 25, "top_k_max": 10},
     "passcode_required": false, "default_prompt": "...",
     "ontologies": [{"slug": "chebi-sample", "name": "ChEBI (sample)", "type": "chemical",
                     "count": 200}],
     "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "unavailable",
       "reason": "no index at indexes/chebi-sample/dense; run ner-site/tools/build_indexes.py"}},
     "selectors": ["retrieval", "jev", "openai/gpt-6-luna"], "default_selector": "jev"}

``POST /api/ner`` request::

    {"text": "Aspirin reduced fever in rheumatoid arthritis.", "prompt": null, "model": null,
     "top_k": 5}

200 response::

    {"ok": true, "model": "openai/gpt-4o-mini", "mode": "live", "selector": "jev",
     "entities": [
       {"id": "e1", "text": "Aspirin", "start": 0, "end": 7, "type": "chemical",
        "ontology": "chebi-sample",
        "matches": [{"id": "CHEBI:15365", "label": "acetylsalicylic acid", "score": 0.0325,
                     "retrievers": {"bm25": 1, "dense-qwen3": 2},
                     "url": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15365"}],
        "notes": [],
        "pick": {"selector": "jev", "status": "match", "id": "CHEBI:15365",
                 "label": "acetylsalicylic acid", "url": "https://www.ebi.ac.uk/...",
                 "confidence": 0.93, "explanation": "rubric 0.95, properties 0.97"},
        "more": [{"id": "CHEBI:46195", "label": "paracetamol", "score": 0.0161, ...}]},
       {"id": "e2", "text": "fever", "start": 16, "end": 21, "type": "other", "ontology": null,
        "matches": [], "notes": ["no ontology for type 'other'"], "pick": null,
        "more": []}],
     "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
     "warnings": ["entity 'feverr' dropped: not found in text"],
     "usage": {"llm_calls": 1, "prompt_tokens": 412, "completion_tokens": 88,
               "embedding_requests": 2},
     "timing_ms": {"extract": 1800, "map": 900, "select": 1400, "total": 4200}}

    ``usage`` also carries ``selector_calls``, ``selector_prompt_tokens``,
    ``selector_completion_tokens`` and ``selector_cost_usd`` (PLAN.md section 13).

Additions of 18 November 2026 (docs/superpowers/specs/
2026-11-18-claude-provider-open-access-design.md section 6). The 200 response also
carries what was asked for and every departure from it::

    "requested_model": "claude-opus-5-5", "requested_selector": "claude-opus-5-5",
    "fallbacks": [{"stage": "extract", "from": "claude-opus-5-5",
                   "to": "claude-sonnet-5-5", "reason": "rate_limited"}],
    "notices": ["Claude Opus 5.5 is rate limited; this ran on Claude Sonnet 5.5."]

``model`` is then the model that produced the extraction and each ``pick.selector`` the
model that made that pick. ``/api/health`` gains ``claude`` and ``visitor`` beside
``llm`` (added by ``ner_site.http``, not part of :class:`HealthResponse`).

Token counts are ``null`` when the provider did not report them, never 0.

Error envelope (see :meth:`ServiceError.envelope`)::

    {"ok": false, "error": {"code": "text_too_long",
                            "message": "text is 7000 characters; the limit is 5000"}}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

EntityType = Literal["chemical", "disease", "gene", "food", "other"]
ENTITY_TYPES: tuple[EntityType, ...] = ("chemical", "disease", "gene", "food", "other")

Mode = Literal["live", "offline"]


class Entity(BaseModel):
    """One mention located in the visitor's text; ``text[start:end] == text`` field."""

    id: str
    text: str
    start: int
    end: int
    type: EntityType


class Match(BaseModel):
    """One ontology term for an entity. ``id`` always comes from the store."""

    id: str
    label: str
    score: float
    retrievers: dict[str, int]
    url: str | None


PickStatus = Literal["match", "review", "no_match", "failed"]


class Pick(BaseModel):
    """The selector's verdict on one entity's retrieved candidates.

    ``id`` is set for "match", may be set for "review" (None: unsure whether any
    candidate fits), and is never set otherwise; when set it is always one of the
    candidates retrieved for the entity (an issued key resolved by exact lookup, or
    Jev's choice among them); a model's raw text never becomes an id."""

    selector: str
    status: PickStatus
    id: str | None = None
    label: str | None = None
    url: str | None = None
    confidence: float | None = None
    explanation: str = ""


class EntityResult(Entity):
    """``matches`` is the visitor's top_k of the fused retrieval ranking; ``more`` the
    rest of it down to ``mapping.MAX_LISTED`` (30), the depth a selector chooses from."""

    ontology: str | None
    matches: list[Match]
    notes: list[str]
    pick: Pick | None = None
    more: list[Match] = []


class RetrieverStatus(BaseModel):
    """``reason`` is omitted from the JSON when it is None (``{"status": "ok"}``)."""

    status: Literal["ok", "unavailable"]
    reason: str | None = None

    @model_serializer(mode="wrap")
    def _drop_missing_reason(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        if data.get("reason") is None:
            data.pop("reason", None)
        return data


class Usage(BaseModel):
    llm_calls: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    embedding_requests: int
    # The selection step (Jev or an LLM): its calls, tokens and the cost the provider
    # reported (null when any call did not report it).
    selector_calls: int = 0
    selector_prompt_tokens: int | None = None
    selector_completion_tokens: int | None = None
    selector_cost_usd: float | None = None


class NerRequest(BaseModel):
    """``prompt``/``model`` None mean the defaults that ``/api/health`` reports.

    Limits (text length, prompt length, top_k range, model allowlist) are checked by the
    service against :class:`Limits`, so each failure gets its own error code.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    prompt: str | None = None
    model: str | None = None
    top_k: int = 5
    selector: str | None = None


class FallbackNote(BaseModel):
    """One departure from the requested model. ``stage`` is ``admit`` (decided before
    any call: the visitor's allowance, the limiter, a Claude that is known to be
    unavailable), ``extract`` or ``select``. ``reason`` is a machine string
    (``rate_limited``, ``opus_allowance_used`` ...); the sentence for the visitor is in
    ``NerResponse.notices``. ``from_`` is ``"from"`` on the wire (a Python keyword)."""

    model_config = ConfigDict(populate_by_name=True)

    stage: str
    from_: str = Field(alias="from")
    to: str
    reason: str

    @model_serializer(mode="plain")
    def _from_on_the_wire(self) -> dict[str, str]:
        return {"stage": self.stage, "from": self.from_, "to": self.to, "reason": self.reason}


class NerResponse(BaseModel):
    """``model`` is the model that produced the extraction; ``requested_model`` and
    ``requested_selector`` are what was asked for, defaults resolved (empty only in
    responses written before those fields existed)."""

    ok: Literal[True] = True
    model: str
    mode: Mode
    selector: str = "retrieval"
    entities: list[EntityResult]
    retrieval: dict[str, RetrieverStatus]
    warnings: list[str]
    usage: Usage
    timing_ms: dict[str, int]
    requested_model: str = ""
    requested_selector: str = ""
    fallbacks: list[FallbackNote] = []
    notices: list[str] = []


class OntologySummary(BaseModel):
    slug: str
    name: str
    type: EntityType
    count: int


class HealthResponse(BaseModel):
    ok: Literal[True] = True
    mode: Mode
    default_model: str
    models: list[str]
    limits: dict[str, int]  # Limits.public()
    passcode_required: bool
    default_prompt: str
    ontologies: list[OntologySummary]
    retrieval: dict[str, RetrieverStatus]
    selectors: list[str] = ["retrieval"]
    default_selector: str = "retrieval"


@dataclass(frozen=True)
class Limits:
    max_chars: int = 5000
    max_prompt_chars: int = 4000
    max_entities: int = 25
    top_k_max: int = 10
    max_body_bytes: int = 65536
    llm_call_cap: int = 3

    def public(self) -> dict[str, int]:
        """The subset ``/api/health`` reports (plan section 3.1)."""
        return {
            "max_chars": self.max_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_entities": self.max_entities,
            "top_k_max": self.top_k_max,
        }


class ServiceError(Exception):
    """A failure with an HTTP status and a code from the table in plan section 3.1.

    ``message`` is shown to the visitor: never put a key, a URL's userinfo or the
    visitor's text in it.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        # Response headers that belong to the failure (``Retry-After`` on a 429).
        self.headers: dict[str, str] = dict(headers or {})

    def envelope(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message}}
