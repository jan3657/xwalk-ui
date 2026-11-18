"""The NER site's selection step: Jev or an LLM picks one retrieved candidate per entity.

Offline: FakeDecider, FakeLLM and in-memory stores. Plan: docs/ner-site/PLAN.md section 13.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest
from ner_site.indexes import OntologyIndex
from ner_site.schema import EntityResult
from ner_site.select import JEV, RETRIEVAL_ONLY, SelectStats, select_entities
from xwalk.decide.base import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionResponse,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)
from xwalk.decide.fake import FakeDecider
from xwalk.llm.base import LLMRequest
from xwalk.llm.fake import FakeLLM
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.stores.memory import MemoryStore

TEXT = "Patients took aspirin daily."
ASPIRIN = Record(
    "CHEBI:15365",
    {"label": "aspirin", "synonyms": ["acetylsalicylic acid"], "definition": "a drug"},
)
PARACETAMOL = Record("CHEBI:46195", {"label": "paracetamol", "synonyms": [], "definition": ""})


def _entity(**overrides: Any) -> EntityResult:
    fields: dict[str, Any] = {
        "id": "e1",
        "text": "aspirin",
        "start": 14,
        "end": 21,
        "type": "chemical",
        "ontology": "chebi",
        "matches": [],
        "notes": [],
    }
    return EntityResult(**{**fields, **overrides})


def _setup() -> tuple[dict[str, OntologyIndex], dict[str, list[Candidate]]]:
    store = MemoryStore.from_source([ASPIRIN, PARACETAMOL])
    index = OntologyIndex("chebi", store, [], {})
    # Retrieval ranked the wrong term first: the selector is what fixes that.
    ranked = [PARACETAMOL, ASPIRIN]
    candidates = [
        Candidate(r, 1.0 / (i + 1), (RetrievalHit(r.id, "bm25", None, i + 1),))
        for i, r in enumerate(ranked)
    ]
    return {"chebi": index}, {"e1": candidates}


def _run(selector: str, **kw: Any) -> tuple[list[EntityResult], SelectStats]:
    indexes, candidates = _setup()
    stats = SelectStats()
    results = asyncio.run(
        select_entities(
            kw.pop("results", [_entity()]),
            kw.pop("candidates", candidates),
            indexes,
            selector=selector,
            text=TEXT,
            stats=stats,
            **kw,
        )
    )
    return results, stats


def _knows_aspirin(state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
    """A scripted Jev: yes to the aspirin record, no to anything else, confident."""
    candidates: Mapping[str, Any] = state.get("candidates", {})
    right = {key for key, value in candidates.items() if "CHEBI:15365" in json.dumps(value)}
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        if isinstance(question, Noul):
            text = json.dumps(question.instructions)
            refers = [key for key in candidates if key in text]
            yes = not refers or any(key in right for key in refers)
            answers[name] = NoulAnswer(noul=0.95 if yes else 0.05)
        elif isinstance(question, Choice):
            options = [k for k in question.criteria if k != "NONE"]
            best = next((k for k in options if k in right), "NONE")
            probabilities = {k: (0.95 if k == best else 0.0) for k in question.criteria}
            probabilities["NONE"] = 0.05 if best != "NONE" else 1.0
            answers[name] = ChoiceAnswer(best, probabilities, probabilities[best])
        elif isinstance(question, Score):
            top = len(question.criteria) - 1
            answers[name] = ScoreAnswer(
                score=top,
                probabilities={str(i): float(i == top) for i in range(top + 1)},
                confidence=0.95,
                legend={str(i): str(c) for i, c in enumerate(question.criteria)},
            )
    return answers


def _key_of(request: LLMRequest, record_id: str) -> str:
    found = re.search(r"\[(C\d+)\] ID: " + re.escape(record_id), request.user)
    assert found, request.user
    return found.group(1)


def test_jev_picks_the_right_candidate_over_the_retrieval_ranking() -> None:
    decider = FakeDecider(handler=_knows_aspirin)
    (result,), stats = _run(JEV, decider=decider)
    assert result.pick is not None
    assert result.pick.selector == JEV
    assert result.pick.id == "CHEBI:15365" and result.pick.label == "aspirin"
    assert result.pick.status in {"match", "review"}
    assert result.pick.url and "CHEBI:15365" in result.pick.url
    assert stats.calls == len(decider.calls) >= 2
    # Jev saw the mention in its sentence, not the bare word.
    assert any("took" in json.dumps(state) for state, _ in decider.calls)


def test_an_llm_selector_resolves_only_an_issued_key() -> None:
    llm = FakeLLM(
        handler=lambda r: json.dumps(
            {"chosen_key": _key_of(r, "CHEBI:15365"), "confidence_score": 0.9, "explanation": "x"}
        )
    )
    (result,), stats = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None and result.pick.selector == "openai/gpt-6-luna"
    assert result.pick.status == "match" and result.pick.id == "CHEBI:15365"
    assert result.pick.confidence == pytest.approx(0.9)
    assert stats.calls == 1 and stats.prompt_tokens == 100 and stats.completion_tokens == 20


def test_an_unissued_key_never_becomes_an_id() -> None:
    llm = FakeLLM(['{"chosen_key": "CHEBI:15365", "confidence_score": 0.99, "explanation": ""}'])
    (result,), _ = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None
    assert result.pick.id is None and result.pick.status == "failed"


def test_an_llm_abstention_is_no_match() -> None:
    llm = FakeLLM(['{"chosen_key": null, "confidence_score": 0.8, "explanation": "none fit"}'])
    (result,), _ = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None
    assert result.pick.status == "no_match" and result.pick.id is None
    assert result.pick.explanation == "none fit"


def test_retrieval_only_and_unrouted_entities_call_nothing() -> None:
    decider = FakeDecider()
    (result,), stats = _run(RETRIEVAL_ONLY, decider=decider)
    assert result.pick is None and stats.calls == 0
    other = _entity(id="e2", type="other", ontology=None)
    (result,), stats = _run(JEV, decider=decider, results=[other], candidates={})
    assert result.pick is None and stats.calls == 0 and not decider.calls


class _SlowDecider:
    model = "slow"
    fingerprint = "slow"

    async def decide(self, state: Any, questions: Mapping[str, Question]) -> DecisionResponse:
        await asyncio.sleep(5)
        raise AssertionError("not reached")


def test_the_deadline_cuts_a_slow_selector_off() -> None:
    (result,), _ = _run(JEV, decider=_SlowDecider(), deadline=0.05)
    assert result.pick is not None and result.pick.status == "failed"
    assert result.pick.id is None and "deadline" in result.pick.explanation


def test_a_failing_decider_is_a_failed_pick_not_an_exception() -> None:
    decider = FakeDecider(handler=lambda state, questions: RuntimeError("boom"))
    (result,), _ = _run(JEV, decider=decider)
    assert result.pick is not None and result.pick.status == "failed" and result.pick.id is None


def test_a_failed_jev_pick_names_the_error_not_the_signals() -> None:
    """A gate call that failed after the screen and choice showed "same 0.43; NONE
    0.69" on the page, hiding why the pick failed."""
    from xwalk.decide.base import DecisionRetryableError

    calls: list[int] = []

    def flaky(state: Any, questions: Mapping[str, Question]) -> Any:
        calls.append(1)
        if len(calls) >= 3:  # screen and choose answer; the gate fails
            return DecisionRetryableError("HTTP 503 from the decisions endpoint")
        return _knows_aspirin(state, questions)

    (result,), _ = _run(JEV, decider=FakeDecider(handler=flaky))
    assert result.pick is not None and result.pick.status == "failed"
    assert "HTTP 503" in result.pick.explanation


def test_an_unsure_none_is_a_review_without_a_term_not_a_failure() -> None:
    """Jev leaning to "none of these" below the none_at threshold (p_none 0.68 against
    0.70) is the policy's needs_review with no id; the page showed "Selection failed"."""

    def unsure(state: Any, questions: Mapping[str, Question]) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            if isinstance(question, Noul):
                answers[name] = NoulAnswer(noul=0.5)
            elif isinstance(question, Choice):
                probabilities = {k: 0.0 for k in question.criteria}
                probabilities["NONE"] = 0.6
                first = next(k for k in question.criteria if k != "NONE")
                probabilities[first] = 0.4
                answers[name] = ChoiceAnswer("NONE", probabilities, 0.6)
        return answers

    (result,), _ = _run(JEV, decider=FakeDecider(handler=unsure))
    assert result.pick is not None
    assert result.pick.status == "review" and result.pick.id is None


# --- a fallback chain as the selector (spec 2026-11-18, section 5 step 6) ------------


def _pick_reply(request: LLMRequest) -> str:
    return json.dumps(
        {"chosen_key": _key_of(request, "CHEBI:15365"), "confidence_score": 0.9, "explanation": "x"}
    )


def test_a_pick_names_the_model_of_the_chain_that_answered_it() -> None:
    from ner_site.fallback import FallbackLLM, Step
    from xwalk.llm.base import LLMRetryableError

    limited = LLMRetryableError("429")
    limited.reason = "rate_limited"  # type: ignore[attr-defined]
    opus = FakeLLM([limited], model="claude-opus-5-5")
    sonnet = FakeLLM(handler=_pick_reply, model="claude-sonnet-5-5")
    chain = FallbackLLM([Step("opus", opus), Step("sonnet", sonnet)])
    chain.stage = "select"
    (result,), stats = _run("claude-opus-5-5", llm=chain)
    assert result.pick is not None and result.pick.status == "match"
    assert result.pick.selector == "claude-sonnet-5-5"  # not the selector that was asked for
    assert [(f.stage, f.from_model, f.to_model) for f in chain.fallbacks] == [
        ("select", "claude-opus-5-5", "claude-sonnet-5-5")
    ]
    assert stats.calls == 2 and stats.prompt_tokens is None  # the failed call is counted


def test_concurrent_picks_each_name_their_own_model() -> None:
    """Two entities in flight on one chain that steps down between their answers."""
    from ner_site.fallback import FallbackLLM, Step
    from xwalk.llm.base import LLMFatalError

    refused = LLMFatalError("declined")
    refused.reason = "refused"  # type: ignore[attr-defined]
    seen: list[str] = []

    def opus_reply(request: LLMRequest) -> Any:
        seen.append(request.user)
        return _pick_reply(request) if len(seen) == 1 else refused

    opus = FakeLLM(handler=opus_reply, model="claude-opus-5-5")
    sonnet = FakeLLM(handler=_pick_reply, model="claude-sonnet-5-5")
    chain = FallbackLLM([Step("opus", opus), Step("sonnet", sonnet)])
    _, candidates = _setup()
    first, second = _entity(), _entity(id="e2")
    results, _ = _run(
        "claude-opus-5-5",
        llm=chain,
        results=[first, second],
        candidates={"e1": candidates["e1"], "e2": candidates["e1"]},
    )
    named = [r.pick.selector for r in results if r.pick is not None]
    assert named == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert all(r.pick is not None and r.pick.id == "CHEBI:15365" for r in results)


def test_a_plain_client_keeps_the_selector_it_was_asked_as() -> None:
    llm = FakeLLM(handler=_pick_reply, model="some-provider-name-for-it")
    (result,), _ = _run("openai/gpt-6-luna", llm=llm)
    assert result.pick is not None and result.pick.selector == "openai/gpt-6-luna"


def test_a_failed_llm_pick_still_counts_its_call() -> None:
    from xwalk.llm.base import LLMFatalError

    (result,), stats = _run("openai/gpt-6-luna", llm=FakeLLM([LLMFatalError("HTTP 500")]))
    assert result.pick is not None and result.pick.status == "failed"
    assert result.pick.selector == "openai/gpt-6-luna" and "HTTP 500" in result.pick.explanation
    assert stats.calls == 1 and stats.prompt_tokens is None


@pytest.mark.parametrize(
    "answer,word",
    [
        ("not-json", "bad_answer"),
        ('{"chosen_key":"FAKE"}', "bad_answer"),
        (TimeoutError("unsafe"), "timeout"),
        (RuntimeError("unsafe"), "error"),
    ],
)
def test_own_failed_pick_uses_only_a_machine_word(answer: Any, word: str) -> None:
    (result,), _ = _run("gpt-6-luna", own_key=True, llm=FakeLLM([answer]))
    assert result.pick and (result.pick.status, result.pick.explanation) == ("failed", word)


def test_cancelled_own_call_becomes_a_failed_pick() -> None:
    class Cancelled(FakeLLM):
        async def complete(self, request: LLMRequest) -> Any:
            raise asyncio.CancelledError("unsafe provider detail")

    (result,), _ = _run("gpt-6-luna", own_key=True, llm=Cancelled([]))
    assert result.pick and (result.pick.status, result.pick.explanation) == ("failed", "timeout")


def test_a_site_selector_can_still_supply_its_own_view() -> None:
    class Client(FakeLLM):
        def view(self) -> FakeLLM:
            view = FakeLLM(handler=_pick_reply)
            view.answered_by = "reported-model"
            return view

    (result,), stats = _run("requested-model", llm=Client([]))
    assert result.pick and result.pick.selector == "reported-model"
    assert not stats.served_elsewhere


# --- calls cut off by the deadline are still reported (final review, C3) ---------------


class _Stalls(FakeLLM):
    """Answers the first `quick` picks, then never: a call the deadline has to cancel."""

    def __init__(self, quick: int = 0) -> None:
        super().__init__(handler=_pick_reply, model="stalling")
        self._quick = quick

    async def complete(self, request: LLMRequest) -> Any:
        if self._quick > 0:
            self._quick -= 1
            return await super().complete(request)
        await asyncio.sleep(30)
        raise AssertionError("not reached")


def _two_entities() -> tuple[list[EntityResult], dict[str, list[Candidate]]]:
    _, candidates = _setup()
    return [_entity(), _entity(id="e2")], {"e1": candidates["e1"], "e2": candidates["e1"]}


@pytest.mark.parametrize("own_key", [False, True])
def test_a_pick_cancelled_by_the_deadline_still_counts_its_dispatched_call(own_key: bool) -> None:
    """The call left for the provider and may be billed; the cancellation skips the
    usage the pick would have reported, so the call budget's count is what is left."""
    from xwalk.llm.budget import BudgetedLLM, CallBudget

    llm = BudgetedLLM(_Stalls(), CallBudget(limit=4))
    (result,), stats = _run("gpt-6-luna", llm=llm, deadline=0.05, own_key=own_key)
    assert result.pick is not None and result.pick.status == "failed"
    assert llm.budget.dispatched == 1
    assert stats.calls == 1
    assert stats.prompt_tokens is None and stats.completion_tokens is None
    assert stats.cost_usd is None and stats.reported_cost is None


def test_one_answered_and_one_cancelled_pick_report_two_calls_and_unknown_tokens() -> None:
    from ner_site.service import wire_usage
    from xwalk.llm.budget import BudgetedLLM, CallBudget
    from xwalk.records import Usage

    results, candidates = _two_entities()
    llm = BudgetedLLM(_Stalls(quick=1), CallBudget(limit=4))
    picked, stats = _run("m", llm=llm, deadline=0.05, results=results, candidates=candidates)
    assert sorted(r.pick.status for r in picked if r.pick) == ["failed", "match"]
    assert (stats.calls, stats.prompt_tokens, stats.completion_tokens) == (2, None, None)
    usage = wire_usage(Usage(prompt_tokens=5, completion_tokens=2, calls=1), 0, stats)
    assert usage.selector_calls == 2
    assert usage.selector_prompt_tokens is None and usage.selector_completion_tokens is None
    assert usage.selector_cost_usd is None


def test_calls_that_reported_are_not_counted_twice() -> None:
    from xwalk.llm.budget import BudgetedLLM, CallBudget

    results, candidates = _two_entities()
    llm = BudgetedLLM(FakeLLM(handler=_pick_reply), CallBudget(limit=4))
    _, stats = _run("m", llm=llm, results=results, candidates=candidates)
    assert llm.budget.dispatched == 2 and stats.calls == 2
    assert stats.prompt_tokens is not None and stats.completion_tokens is not None
    # Calls made before this step (an extraction on the same budget) are not the selector's.
    again = SelectStats()
    indexes, _ = _setup()
    asyncio.run(
        select_entities(results, candidates, indexes, selector="m", text=TEXT, stats=again, llm=llm)
    )
    assert llm.budget.dispatched == 4 and again.calls == 2


def test_a_cancelled_pick_on_a_chain_counts_the_call_of_the_step_it_was_on() -> None:
    from ner_site.claude import OPUS
    from ner_site.service import SELECT, Settings, make_chain

    settings = Settings(anthropic_api_key="sk-ant-SENTINEL-0123456789")
    chain = make_chain(
        settings, OPUS, purposes=(SELECT,), claude_factory=lambda s, model, purpose: _Stalls()
    )
    (result,), stats = _run(OPUS, llm=chain, deadline=0.05)
    assert result.pick is not None and "deadline" in result.pick.explanation
    assert stats.calls == 1 and stats.prompt_tokens is None


def test_a_cancelled_pick_through_the_own_key_reader_counts_its_call() -> None:
    from ner_site.http import _OwnJSONClient
    from ner_site.providers import Credential
    from xwalk.llm.budget import BudgetedLLM, CallBudget

    credential = Credential("openai", lambda: "sk-SENTINEL-0123456789abcdef")
    llm = _OwnJSONClient(BudgetedLLM(_Stalls(), CallBudget(limit=4)), credential)
    (result,), stats = _run("gpt-6-luna", llm=llm, deadline=0.05, own_key=True)
    assert result.pick is not None and result.pick.explanation == "timeout"
    assert stats.calls == 1 and stats.prompt_tokens is None


def test_a_cancelled_jev_pick_counts_its_dispatched_call() -> None:
    from xwalk.decide.budget import BudgetedDecider
    from xwalk.llm.budget import CallBudget

    decider = BudgetedDecider(_SlowDecider(), CallBudget(limit=8))
    (result,), stats = _run(JEV, decider=decider, deadline=0.05)
    assert result.pick is not None and "deadline" in result.pick.explanation
    assert decider.budget.dispatched == 1
    assert stats.calls == 1 and stats.reported_cost is None
