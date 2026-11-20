"""Selection: one model picks the matching term among an entity's retrieved candidates.

Retrieval ranks; a selector decides. The visitor chooses the selector per request:

- ``retrieval`` (no selector): the fused ranking is the answer, as before;
- ``jev``: TypeSafe's Jev through ``xwalk.decide.DecisionMatcher`` (screen every
  candidate, choose among the survivors or none, gate the choice on the slots'
  properties; thresholds fitted on the pilots, ``POLICIES``);
- an LLM id (``openai/gpt-6-luna`` ...): ``xwalk.stages.select.Selector``, one call that
  returns an opaque candidate key or null. A Claude id runs on a fallback chain, and each
  pick's ``selector`` is then the model that made that pick.

Both read the question sets in ``ner_site/slots/`` (copied from the examples). Both only
ever resolve to a candidate retrieved for the entity: Jev chooses among keys, and an LLM's
key is resolved by exact lookup, so a malformed or invented answer is a failed pick, never
an id. Every failure (provider error, timeout, the request deadline) becomes a ``failed``
pick with its reason, never an exception.

The pick cache (``ner_site.cache``): with `cache`, a question that was answered before
is not asked again. It is for requests the site pays for only: the service passes no
cache for a request on a visitor's own key, which therefore neither reads nor writes a
pick (one provider account never answers for another). The question is everything the
selector was shown: the model (or ``jev``), the mention with its type and ontology, the
exact text on either side of it (``CONTEXT_CHARS`` each), and the candidates in order
with the content of their records (``cache.content_of``), so a term whose label or
definition changed is asked about again. The mention alone is never the key: "green
tea" after "drank" is the drink, elsewhere the leaf. On a fallback chain a pick is
looked up under the model the chain would ask now.

Stored, per pick: one that is not ``failed``, was answered by the selector that was
asked for (not by a later step of a chain, and not by another model a provider served
in its place), in a run that had not stepped down or fallen back when the pick was made
(`storable`), on candidates that were retrieved cleanly (`degraded` names the entities
whose were not), and in which `sensitive` finds no secret. A hit costs no call and no
tokens (``SelectStats.cache_hits`` counts them).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from xwalk.decide.base import DecisionClient
from xwalk.decide.matcher import DecisionMatcher
from xwalk.decide.policy import DecisionPolicy
from xwalk.decide.questions import QuestionSet
from xwalk.llm.base import LLMClient, failure_usage
from xwalk.prompts.contract import PromptSet, PromptSlots, load_slots
from xwalk.records import Candidate, MatchStatus, Record, RetrievalHit
from xwalk.records import Usage as XwalkUsage
from xwalk.retrieval.base import SearchRequest
from xwalk.stages.choose import Chooser
from xwalk.stages.keying import Resolution
from xwalk.stages.property_gate import PropertyGate
from xwalk.stages.screen import Screener
from xwalk.stages.select import Selector, SelectorPolicy
from xwalk.templates import TemplateSet

from ner_site.cache import TTLCache, pick_key
from ner_site.indexes import OntologyIndex
from ner_site.mapping import MAX_LISTED
from ner_site.ontologies import DOC_TEMPLATE, term_url
from ner_site.providers import failure_word
from ner_site.schema import EntityResult, Pick, PickStatus

RETRIEVAL_ONLY = "retrieval"
JEV = "jev"
JEV_MODEL = "~typesafe/jev-latest"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_TIMEOUT_S = 20.0
JEV_RETRIES = 1

SLOTS_DIR = Path(__file__).resolve().parent / "slots"
CONTEXT_CHARS = 150  # either side of the mention, for the selector's context
# Of the fused list (two retrievers at depth 20): one screen chunk, and exactly the
# candidates the page lists (matches plus more), so every pick is visible there.
MAX_CANDIDATES = MAX_LISTED
CONCURRENCY = 8  # entities selected at once
DEADLINE_NOTE = "cut off by the request deadline"

SELECT_TEMPLATES = TemplateSet(
    query="{{ mention }}",
    context="{{ context_left }}[{{ mention }}]{{ context_right }}",
    doc=DOC_TEMPLATE,
    candidate=(
        "ID: {{ id }}\nLabel: {{ label }}\n"
        "{% if synonyms %}Synonyms: "
        "{% if synonyms is string %}{{ synonyms }}{% else %}{{ synonyms | join('; ') }}"
        "{% endif %}\n{% endif %}"
        "{% if definition %}Definition: {{ definition }}\n{% endif %}"
        "{% if organism %}Organism: {{ organism }}{% endif %}"
    ),
)

# The thresholds the examples' job_jev.yaml files fitted on the pilots (`xwalk fit`), per
# entity type; depth and chunking sized for the site (one screen call per entity).
_FITTED: dict[str, tuple[float, float, float]] = {  # accept_at, property_floor, choose_at
    "chemical": (0.65, 0.50, 0.70),
    "disease": (0.75, 0.50, 0.50),
    "food": (0.75, 0.70, 0.50),
    "gene": (0.85, 0.50, 0.70),
}
POLICIES: dict[str, DecisionPolicy] = {
    entity_type: DecisionPolicy(
        accept_at=accept,
        property_floor=floor,
        choose_at=choose,
        screen_floor=0.30,
        chunk_size=50,
        max_candidates=MAX_CANDIDATES,
        concurrency=CONCURRENCY,
    )
    for entity_type, (accept, floor, choose) in _FITTED.items()
}

_JEV_STATUS: dict[MatchStatus, PickStatus] = {
    MatchStatus.MATCHED: "match",
    MatchStatus.NEEDS_REVIEW: "review",
    MatchStatus.UNMATCHED: "no_match",
    MatchStatus.FAILED: "failed",
}


@dataclass
class SelectStats:
    """Filled in by `select_entities`: the selector's calls, tokens and reported cost.
    Tokens and cost are None once any call did not report them ("unknown", not 0),
    which includes a call that was sent and then cut off by the request deadline.

    ``served_elsewhere`` records the requested and serving model per pick when the
    adapter signals a server-side refusal fallback."""

    calls: int = 0
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0
    cost_usd: float | None = 0.0
    cost_reported: bool = True  # False for selectors whose provider reports no cost
    served_elsewhere: dict[str, tuple[str, str]] = field(default_factory=dict)
    # entity id -> the model of the chain step that answered its pick (the id the
    # site asks for, which a provider may answer under a dated name).
    asked: dict[str, str] = field(default_factory=dict)
    # Picks answered from the pick cache: no call was made for them.
    cache_hits: int = 0

    def add(self, usage: XwalkUsage) -> None:
        self.calls += usage.calls
        if usage.unknown_calls:
            self.prompt_tokens = self.completion_tokens = self.cost_usd = None
            return
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
        if self.cost_usd is not None:
            self.cost_usd += usage.cost_usd

    @property
    def reported_cost(self) -> float | None:
        if self.calls == 0:
            return None
        return self.cost_usd if self.cost_reported else None


def dispatched(client: object) -> int:
    """How many requests `client` has sent so far, as its call budgets counted them
    when each was dispatched: a budgeted client or decider (``budget.dispatched``,
    also behind a wrapper that passes ``budget`` on), a staged step (the clients it
    has ``built``) or a fallback chain (its ``steps``). 0 for a client with no budget."""
    steps = getattr(client, "steps", None)
    if steps is not None:
        return sum(dispatched(step.client) for step in steps)
    built = getattr(client, "built", None)
    if built is not None:
        return sum(dispatched(part) for part in built)
    count = getattr(getattr(client, "budget", None), "dispatched", None)
    return count if isinstance(count, int) else 0


@cache
def slots(entity_type: str) -> PromptSlots:
    return load_slots(SLOTS_DIR / f"{entity_type}.yaml")


@cache
def _questions(entity_type: str) -> QuestionSet:
    return QuestionSet.from_slots(slots(entity_type))


@cache
def _prompts(entity_type: str) -> PromptSet:
    return PromptSet.from_slots(slots(entity_type))


class _Retrieved:
    """A retriever that answers with the candidates the site already retrieved, in the
    fused order, so the decision matcher re-ranks nothing and searches nothing."""

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._hits = [
            RetrievalHit(c.id, "retrieved", c.fused_score, rank)
            for rank, c in enumerate(candidates, start=1)
        ]

    @property
    def name(self) -> str:
        return "retrieved"

    @property
    def fingerprint(self) -> str:
        return "retrieved"

    @property
    def default_limit(self) -> int:
        return max(1, len(self._hits))

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        return self._hits


def _source(entity: EntityResult, text: str) -> Record:
    return Record(
        entity.id,
        {
            "mention": entity.text,
            "context_left": text[max(0, entity.start - CONTEXT_CHARS) : entity.start],
            "context_right": text[entity.end : entity.end + CONTEXT_CHARS],
        },
    )


def _pick(
    selector: str,
    status: PickStatus,
    candidates: Sequence[Candidate],
    record_id: str | None = None,
    *,
    confidence: float | None = None,
    explanation: str = "",
) -> Pick:
    record = next((c.record for c in candidates if c.id == record_id), None)
    if record_id is not None and record is None:  # cannot happen; refuse rather than invent
        return Pick(selector=selector, status="failed", explanation="chose an unknown term")
    return Pick(
        selector=selector,
        # A match names a term. A review may not: Jev leaning to "none of these" below
        # the policy's none_at is needs_review with no id, an honest "unsure".
        status=status if record is not None or status != "match" else "failed",
        id=None if record is None else record.id,
        label=None if record is None else str(record.fields.get("label", "")),
        url=None if record is None else term_url(record.id),
        confidence=confidence,
        explanation=explanation,
    )


async def _with_jev(
    entity: EntityResult,
    index: OntologyIndex,
    candidates: Sequence[Candidate],
    text: str,
    decider: DecisionClient,
    stats: SelectStats,
) -> Pick:
    entity_type = entity.type
    questions = _questions(entity_type)
    policy = POLICIES[entity_type]
    matcher = DecisionMatcher(
        templates=SELECT_TEMPLATES,
        retrievers=[_Retrieved(candidates)],
        store=index.store,
        screener=Screener(
            decider,
            questions,
            SELECT_TEMPLATES,
            chunk_size=policy.chunk_size,
            shortlist_size=policy.shortlist_size,
            shortlist_floor=policy.shortlist_floor,
        ),
        chooser=Chooser(decider, questions, SELECT_TEMPLATES),
        gate=PropertyGate(decider, questions, SELECT_TEMPLATES),
        policy=policy,
        keep_candidates_in_trace=False,
    )
    result = await matcher.match(_source(entity, text))
    stats.add(result.usage)
    status = _JEV_STATUS[result.status]
    explanation = result.explanation
    error = result.attempts[0].error if result.attempts else None
    if status == "failed" and error:
        # A Jev call failed (after the screen, say): say which, not the partial signals.
        explanation = f"{error} (signals so far: {explanation})" if explanation else error
    return _pick(
        JEV,
        status,
        candidates,
        result.matched_id,
        confidence=result.confidence,
        explanation=explanation,
    )


async def _with_llm(
    entity: EntityResult,
    candidates: Sequence[Candidate],
    text: str,
    llm: LLMClient,
    selector: str,
    stats: SelectStats,
    own_key: bool,
) -> Pick:
    source = _source(entity, text)
    # A fallback chain (ner_site.fallback) may answer this pick from another model than
    # the one asked for, and may move on while other picks are in flight: a view of it
    # remembers which model answered this one, and the pick names that model.
    get_view = None if own_key else getattr(llm, "view", None)
    view = get_view() if callable(get_view) else None
    chooser = Selector(
        llm if view is None else view,
        _prompts(entity.type),
        SELECT_TEMPLATES,
        policy=SelectorPolicy(max_candidates=MAX_CANDIDATES),
    )
    try:
        outcome = await chooser.select(source, SELECT_TEMPLATES.render_context(source), candidates)
    except Exception as exc:
        # The caller turns this into a failed pick; its calls were dispatched all the same.
        stats.add(failure_usage(exc))
        raise
    stats.add(outcome.usage)
    answered = getattr(view, "answered_by", None)
    if answered:
        selector = answered
        asked = getattr(view, "asked", None)
        if asked:
            stats.asked[entity.id] = asked
        if getattr(view, "served_by_fallback", False) and view.asked:
            stats.served_elsewhere[entity.id] = (view.asked, answered)
    choice = outcome.choice
    if outcome.error is not None:
        return _pick(
            selector,
            "failed",
            candidates,
            explanation="bad_answer" if own_key else f"unreadable answer: {outcome.error}",
        )
    if choice.resolution is Resolution.ABSTAIN:
        return _pick(
            selector,
            "no_match",
            candidates,
            confidence=outcome.confidence,
            explanation=outcome.explanation,
        )
    if choice.resolution is not Resolution.EXACT_KEY or choice.record_id is None:
        return _pick(
            selector,
            "failed",
            candidates,
            explanation="bad_answer" if own_key else "the model named no candidate it was shown",
        )
    return _pick(
        selector,
        "match",
        candidates,
        choice.record_id,
        confidence=outcome.confidence,
        explanation=outcome.explanation,
    )


async def select_entities(
    results: Sequence[EntityResult],
    candidates: Mapping[str, Sequence[Candidate]],
    indexes: Mapping[str, OntologyIndex],
    *,
    selector: str,
    text: str,
    stats: SelectStats,
    decider: DecisionClient | None = None,
    llm: LLMClient | None = None,
    deadline: float | None = None,
    own_key: bool = False,
    cache: TTLCache[Pick] | None = None,
    clean: Callable[[str], str] | None = None,
    storable: Callable[[], bool] | None = None,
    degraded: Collection[str] = (),
    sensitive: Callable[[str], bool] | None = None,
) -> list[EntityResult]:
    """`results` with ``pick`` set for every entity that was routed to an ontology and
    has candidates (None otherwise, and for ``retrieval``). `candidates` maps entity ids
    to their fused candidates; `deadline` (seconds from now) bounds the whole step.

    `cache` is the pick cache (module docstring); it is never used with `own_key`.
    `clean` is applied to a pick's explanation before the pick is stored, so that what
    the cache holds is what a response may carry. `storable` is asked when a pick is
    about to be stored and says whether the run is still on the models it asked for
    (no step-down, no fallback); `degraded` holds the ids of the entities whose
    candidates came from degraded retrieval; `sensitive` says whether a text holds a
    secret. A pick that fails any of the three is used and not stored."""
    if selector == RETRIEVAL_ONLY:
        return list(results)
    if selector == JEV and decider is None:
        raise ValueError("selector 'jev' needs a decider")
    if selector != JEV and llm is None:
        raise ValueError(f"selector {selector!r} needs an LLM client")
    stats.cost_reported = selector == JEV
    client = decider if selector == JEV else llm
    sent_before, counted_before = dispatched(client), stats.calls
    deadline_note = "timeout" if own_key else DEADLINE_NOTE
    deadline_at = None if deadline is None else time.monotonic() + deadline
    semaphore = asyncio.Semaphore(CONCURRENCY)
    remember = cache if cache is not None and cache.enabled and not own_key else None
    # A chain is looked up under the model it would ask now; anything else under the
    # selector that was asked for.
    on_chain = selector != JEV and not own_key and callable(getattr(llm, "view", None))

    def key_for(entity: EntityResult, found: Sequence[Candidate], model: str) -> str:
        source = _source(entity, text)
        return pick_key(
            selector=model,
            entity_type=entity.type,
            ontology=entity.ontology or "",
            mention=entity.text,
            left=str(source.fields["context_left"]),
            right=str(source.fields["context_right"]),
            candidates=found,
        )

    async def one(entity: EntityResult) -> EntityResult:
        found = list(candidates.get(entity.id, ()))[:MAX_CANDIDATES]
        index = indexes.get(entity.ontology) if entity.ontology else None
        if index is None or not found or entity.type not in POLICIES:
            return entity
        async with semaphore:
            if remember is not None:
                asking = str(getattr(llm, "model", selector)) if on_chain else selector
                hit = remember.get(key_for(entity, found, asking))
                if hit is not None:
                    stats.cache_hits += 1
                    return entity.model_copy(update={"pick": hit[0].model_copy()})
            remaining = None if deadline_at is None else deadline_at - time.monotonic()
            if remaining is not None and remaining <= 0:
                return entity.model_copy(
                    update={"pick": _pick(selector, "failed", found, explanation=deadline_note)}
                )
            try:
                if selector == JEV:
                    assert decider is not None
                    work = _with_jev(entity, index, found, text, decider, stats)
                else:
                    assert llm is not None
                    work = _with_llm(entity, found, text, llm, selector, stats, own_key)
                pick = await asyncio.wait_for(work, timeout=remaining)
            except asyncio.CancelledError:
                if not own_key:
                    raise
                pick = _pick(selector, "failed", found, explanation="timeout")
            except asyncio.TimeoutError:
                pick = _pick(selector, "failed", found, explanation=deadline_note)
            except Exception as exc:  # noqa: BLE001 - a failed pick, never a 5xx
                pick = _pick(
                    selector,
                    "failed",
                    found,
                    explanation=failure_word(exc) if own_key else f"{type(exc).__name__}: {exc}",
                )
            else:
                # A chain's pick is stored only when the step that answered is the
                # selector that was asked for (a step that is not known is not).
                answered_on = stats.asked.get(entity.id) if on_chain else selector
                if (
                    remember is not None
                    and pick.status != "failed"
                    and entity.id not in stats.served_elsewhere
                    and entity.id not in degraded
                    and answered_on == selector
                    and (storable is None or storable())
                ):
                    # Judged as the model gave it, before anything is scrubbed from it.
                    shown = "\n".join(
                        str(v) for v in pick.model_dump().values() if isinstance(v, str)
                    )
                    if sensitive is None or not sensitive(shown):
                        if clean is not None:
                            pick = pick.model_copy(update={"explanation": clean(pick.explanation)})
                        remember.put(key_for(entity, found, selector), pick.model_copy())
        return entity.model_copy(update={"pick": pick})

    picked = list(await asyncio.gather(*(one(e) for e in results)))
    # A pick the deadline cancelled never reported its usage, but its request had left
    # and may be billed. The budgets counted it at dispatch: what they saw beyond what
    # the picks reported is booked as calls with unknown tokens and cost, never dropped.
    missing = (dispatched(client) - sent_before) - (stats.calls - counted_before)
    if missing > 0:
        stats.add(XwalkUsage.unreported(missing))
    return picked


__all__ = [
    "JEV",
    "JEV_MODEL",
    "JEV_URL",
    "POLICIES",
    "RETRIEVAL_ONLY",
    "SELECT_TEMPLATES",
    "SelectStats",
    "dispatched",
    "select_entities",
]
