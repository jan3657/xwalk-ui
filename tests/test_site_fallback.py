"""The fallback chain (spec 2026-11-18, section 3.2): offline, scripted clients only.

Each step is a small scripted client that answers or raises an error carrying the
``reason`` attribute the chain reads. One case wraps a scripted client in the engine's
real ``BudgetedLLM``, because that is how the Claude steps reach the chain in production.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import pytest
from ner_site import fallback
from ner_site.fallback import (
    BLOCK_TTL_S,
    STEP_DOWN_REASONS,
    Fallback,
    FallbackLLM,
    Step,
    block_claude,
    claude_block,
)
from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMError,
    LLMFatalError,
    LLMRequest,
    LLMResponse,
    LLMRetryableError,
    ParseError,
)
from xwalk.llm.budget import BudgetedLLM, CallBudget, CallLimitExceeded
from xwalk.records import Usage

REQUEST = LLMRequest(system="s", user="u")
ANSWERED = Usage(prompt_tokens=100, completion_tokens=20, calls=1)


@pytest.fixture(autouse=True)
def _clean_block() -> None:
    fallback.reset()
    yield
    fallback.reset()


def failing(reason: str, *, usage: Usage | None = None) -> LLMError:
    """An error as ``claude.py`` raises it: an engine class plus a ``reason`` string."""
    error = LLMFatalError(f"upstream said {reason}", usage=usage)
    error.reason = reason  # type: ignore[attr-defined]
    return error


class Scripted:
    """Answers with its script in order; the last item repeats. Yields before answering,
    so concurrent calls are truly in flight together."""

    def __init__(
        self,
        model: str,
        script: Sequence[str | BaseException] = ("ok",),
        *,
        capabilities: LLMCapabilities | None = None,
    ) -> None:
        self._model = model
        self._script = list(script)
        self._capabilities = capabilities or LLMCapabilities()
        self.calls = 0
        self.closed = False

    @property
    def model(self) -> str:
        return self._model

    @property
    def fingerprint(self) -> str:
        return f"fp-{self._model}"

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    async def complete(self, request: LLMRequest) -> LLMResponse:
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        await asyncio.sleep(0)
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(text=item, usage=ANSWERED, model=self._model)

    async def aclose(self) -> None:
        self.closed = True


def chain_of(*clients: LLMClient, start: int = 0) -> FallbackLLM:
    labels = ("opus", "sonnet", "other")
    return FallbackLLM(
        [Step(label, client) for label, client in zip(labels, clients, strict=False)],
        start=start,
    )


async def test_first_step_answers_without_fallback() -> None:
    opus, sonnet = Scripted("opus-m"), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    response = await chain.complete(REQUEST)

    assert response.text == "ok"
    assert response.model == "opus-m"
    assert response.usage == ANSWERED
    assert chain.fallbacks == []
    assert chain.position == 0
    assert sonnet.calls == 0


def test_chain_is_an_llm_client_showing_the_current_step() -> None:
    capabilities = LLMCapabilities(json_schema=True)
    chain = chain_of(Scripted("opus-m"), Scripted("sonnet-m", capabilities=capabilities), start=1)

    assert isinstance(chain, LLMClient)
    assert isinstance(chain.view(), LLMClient)
    assert chain.stage == "extract"
    assert (chain.model, chain.fingerprint, chain.capabilities) == (
        "sonnet-m",
        "fp-sonnet-m",
        capabilities,
    )
    view = chain.view()
    assert (view.model, view.fingerprint, view.capabilities) == (
        "sonnet-m",
        "fp-sonnet-m",
        capabilities,
    )


def test_a_chain_needs_steps() -> None:
    with pytest.raises(ValueError):
        FallbackLLM([])


@pytest.mark.parametrize("start", [-1, 2])
def test_start_must_name_a_step(start: int) -> None:
    with pytest.raises(ValueError):
        chain_of(Scripted("opus-m"), Scripted("sonnet-m"), start=start)


@pytest.mark.parametrize("reason", sorted(STEP_DOWN_REASONS))
async def test_step_down_reasons_move_to_the_next_step(reason: str) -> None:
    opus, sonnet = Scripted("opus-m", [failing(reason)]), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    response = await chain.complete(REQUEST)

    assert response.model == "sonnet-m"
    assert opus.calls == 1
    assert chain.position == 1
    assert chain.model == "sonnet-m"
    assert chain.fallbacks == [
        Fallback(stage="extract", from_model="opus-m", to_model="sonnet-m", reason=reason)
    ]


def test_step_down_reasons_are_the_specified_set() -> None:
    assert {
        "rate_limited",
        "overloaded",
        "balance_exhausted",
        "key_refused",
        "refused",
        "timeout",
    } == STEP_DOWN_REASONS
    assert {"balance_exhausted", "key_refused"} == fallback.BLOCK_REASONS
    assert BLOCK_TTL_S == 300.0


async def test_call_limit_steps_down_as_call_limit() -> None:
    opus = Scripted("opus-m", [CallLimitExceeded("limit reached", usage=Usage.zero())])
    chain = chain_of(opus, Scripted("sonnet-m"))

    response = await chain.complete(REQUEST)

    assert response.model == "sonnet-m"
    assert chain.fallbacks == [Fallback("extract", "opus-m", "sonnet-m", "call_limit")]
    # Nothing was dispatched for the refused reservation, so it adds no call.
    assert response.usage == ANSWERED


async def test_transient_is_retried_once_then_steps_down() -> None:
    opus, sonnet = Scripted("opus-m", [failing("transient")]), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    response = await chain.complete(REQUEST)

    assert opus.calls == 2
    assert response.model == "sonnet-m"
    assert chain.fallbacks == [Fallback("extract", "opus-m", "sonnet-m", "transient")]
    assert response.usage == ANSWERED + Usage.unreported(2)


async def test_transient_retry_that_answers_stays_on_the_step() -> None:
    opus, sonnet = Scripted("opus-m", [failing("transient"), "ok"]), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    response = await chain.complete(REQUEST)

    assert response.model == "opus-m"
    assert (opus.calls, sonnet.calls) == (2, 0)
    assert chain.fallbacks == []
    assert chain.position == 0
    assert response.usage == ANSWERED + Usage.unreported()


@pytest.mark.parametrize(
    "error",
    [
        failing("bad_request"),
        ParseError("not JSON", raw="{"),
        LLMRetryableError("no reason attribute"),
        RuntimeError("not a provider error"),
    ],
    ids=["bad_request", "parse_error", "no_reason", "other_exception"],
)
async def test_other_errors_are_raised_unchanged_without_step_down(error: Exception) -> None:
    usage_before = getattr(error, "usage", None)
    opus, sonnet = Scripted("opus-m", [error]), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    with pytest.raises(type(error)) as raised:
        await chain.complete(REQUEST)

    assert raised.value is error
    assert getattr(error, "usage", None) is usage_before
    assert (opus.calls, sonnet.calls) == (1, 0)
    assert chain.fallbacks == []
    assert chain.position == 0


async def test_last_step_error_is_raised_unchanged() -> None:
    last_error = failing("rate_limited")
    opus = Scripted("opus-m", [failing("overloaded")])
    sonnet = Scripted("sonnet-m", [failing("rate_limited")])
    other = Scripted("other-m", [last_error])
    chain = chain_of(opus, sonnet, other)

    with pytest.raises(LLMFatalError) as raised:
        await chain.complete(REQUEST)

    assert raised.value is last_error
    assert chain.position == 2
    assert [(f.from_model, f.to_model, f.reason) for f in chain.fallbacks] == [
        ("opus-m", "sonnet-m", "overloaded"),
        ("sonnet-m", "other-m", "rate_limited"),
    ]
    # Three dispatched calls, none of them free.
    assert last_error.usage == Usage.unreported(3)


async def test_single_step_chain_raises_its_error_untouched() -> None:
    error = failing("rate_limited")
    chain = chain_of(Scripted("only-m", [error]))

    with pytest.raises(LLMFatalError) as raised:
        await chain.complete(REQUEST)

    assert raised.value is error
    assert error.usage is None
    assert chain.fallbacks == []


async def test_step_down_is_sticky() -> None:
    opus = Scripted("opus-m", [failing("rate_limited"), "ok"])
    sonnet = Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    await chain.complete(REQUEST)
    second = await chain.complete(REQUEST)

    assert second.model == "sonnet-m"
    assert second.usage == ANSWERED
    assert (opus.calls, sonnet.calls) == (1, 2)
    assert len(chain.fallbacks) == 1


async def test_start_skips_the_earlier_steps() -> None:
    opus = Scripted("opus-m")
    sonnet = Scripted("sonnet-m", [failing("overloaded")])
    other = Scripted("other-m")
    chain = chain_of(opus, sonnet, other, start=1)

    assert chain.position == 1
    response = await chain.complete(REQUEST)

    assert response.model == "other-m"
    assert opus.calls == 0
    assert chain.fallbacks == [Fallback("extract", "sonnet-m", "other-m", "overloaded")]


async def test_concurrent_failures_on_one_step_record_one_fallback() -> None:
    opus, sonnet = Scripted("opus-m", [failing("rate_limited")]), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    responses = await asyncio.gather(*(chain.complete(REQUEST) for _ in range(8)))

    assert [r.model for r in responses] == ["sonnet-m"] * 8
    # All eight were in flight on the first step before any of them failed.
    assert opus.calls == 8
    assert chain.fallbacks == [Fallback("extract", "opus-m", "sonnet-m", "rate_limited")]
    assert chain.position == 1


async def test_late_failure_on_an_abandoned_step_does_not_move_the_chain_again() -> None:
    """A call still in flight on step 0 fails after the chain has already reached step 2:
    it continues on the current step and records nothing."""
    release = asyncio.Event()

    class Held(Scripted):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                await release.wait()
            raise failing("rate_limited")

    opus = Held("opus-m")
    sonnet = Scripted("sonnet-m", [failing("overloaded")])
    other = Scripted("other-m")
    chain = chain_of(opus, sonnet, other)

    held = asyncio.create_task(chain.complete(REQUEST))
    await asyncio.sleep(0)
    assert (await chain.complete(REQUEST)).model == "other-m"
    assert chain.position == 2
    recorded = list(chain.fallbacks)
    release.set()

    assert (await held).model == "other-m"
    assert chain.fallbacks == recorded
    assert len(recorded) == 2
    assert sonnet.calls == 1


async def test_concurrent_transient_failures_on_one_step_record_one_fallback() -> None:
    """Each of the eight calls retries the step once on its own account; the chain still
    moves once, and says so once."""
    opus, sonnet = Scripted("opus-m", [failing("transient")]), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    responses = await asyncio.gather(*(chain.complete(REQUEST) for _ in range(8)))

    assert [r.model for r in responses] == ["sonnet-m"] * 8
    assert chain.fallbacks == [Fallback("extract", "opus-m", "sonnet-m", "transient")]
    assert chain.position == 1
    # The first failures arrive together, so every call may still retry; a retry that
    # fails after the chain has moved is not tried a third time.
    assert 9 <= opus.calls <= 16
    assert all(r.usage.calls == 3 for r in responses)


async def test_a_transient_failure_on_an_abandoned_step_is_not_retried_there() -> None:
    """A call in flight on step 0 fails ``transient`` after another call has moved the
    chain on: it goes to the current step, without the retry and without a record."""
    release = asyncio.Event()

    class Held(Scripted):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                await release.wait()
                raise failing("transient")
            raise failing("rate_limited")

    opus, sonnet = Held("opus-m"), Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)

    held = asyncio.create_task(chain.complete(REQUEST))
    await asyncio.sleep(0)
    assert (await chain.complete(REQUEST)).model == "sonnet-m"
    recorded = list(chain.fallbacks)
    release.set()
    late = await held

    assert late.model == "sonnet-m"
    assert opus.calls == 2  # the held call and the one that moved the chain; no retry
    assert (
        chain.fallbacks == recorded == [Fallback("extract", "opus-m", "sonnet-m", "rate_limited")]
    )
    assert late.usage == ANSWERED + Usage.unreported()


@pytest.mark.parametrize("reason", ["balance_exhausted", "key_refused"])
async def test_a_block_reason_skips_the_other_claude_step(reason: str) -> None:
    """Opus and Sonnet share the key and the balance: what stops one stops the other."""
    opus = Scripted("opus-m", [failing(reason)])
    sonnet, other = Scripted("sonnet-m"), Scripted("other-m")
    chain = chain_of(opus, sonnet, other)

    response = await chain.complete(REQUEST)

    assert response.model == "other-m"
    assert (opus.calls, sonnet.calls, other.calls) == (1, 0, 1)
    assert chain.position == 2
    assert chain.fallbacks == [Fallback("extract", "opus-m", "other-m", reason)]
    assert response.usage == ANSWERED + Usage.unreported()
    # Sticky: later calls stay on the last step.
    assert (await chain.complete(REQUEST)).model == "other-m"
    assert sonnet.calls == 0


async def test_a_block_reason_without_an_other_step_moves_to_the_last_step() -> None:
    dead = failing("balance_exhausted")
    opus, sonnet = Scripted("opus-m", [dead]), Scripted("sonnet-m", [dead])
    chain = chain_of(opus, sonnet)

    with pytest.raises(LLMFatalError) as raised:
        await chain.complete(REQUEST)

    assert raised.value is dead
    assert (opus.calls, sonnet.calls) == (1, 1)
    assert chain.fallbacks == [Fallback("extract", "opus-m", "sonnet-m", "balance_exhausted")]


async def test_a_block_reason_on_sonnet_moves_to_the_other_step() -> None:
    chain = chain_of(
        Scripted("opus-m"), Scripted("sonnet-m", [failing("key_refused")]), Scripted("o-m"), start=1
    )

    assert (await chain.complete(REQUEST)).model == "o-m"
    assert chain.fallbacks == [Fallback("extract", "sonnet-m", "o-m", "key_refused")]


async def test_a_late_block_reason_moves_a_chain_that_only_reached_sonnet() -> None:
    """Opus was rate limited for one call and out of credit for another still in flight:
    the second moves the chain past Sonnet, which shares the balance."""
    release = asyncio.Event()

    class Held(Scripted):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                await release.wait()
                raise failing("balance_exhausted")
            raise failing("rate_limited")

    opus, sonnet, other = Held("opus-m"), Scripted("sonnet-m"), Scripted("other-m")
    chain = chain_of(opus, sonnet, other)

    held = asyncio.create_task(chain.complete(REQUEST))
    await asyncio.sleep(0)
    assert (await chain.complete(REQUEST)).model == "sonnet-m"
    release.set()

    assert (await held).model == "other-m"
    assert chain.position == 2
    assert chain.fallbacks == [
        Fallback("extract", "opus-m", "sonnet-m", "rate_limited"),
        Fallback("extract", "opus-m", "other-m", "balance_exhausted"),
    ]
    assert sonnet.calls == 1


async def test_the_chain_says_which_step_gave_an_answer() -> None:
    """``answer`` is ``complete`` plus the model of the step that answered, which a
    server-side fallback can make differ from the model the answer names."""

    class Elsewhere(Scripted):
        async def complete(self, request: LLMRequest) -> LLMResponse:
            response = await super().complete(request)
            return LLMResponse(text=response.text, usage=response.usage, model="served-m")

    chain = chain_of(Scripted("opus-m", [failing("overloaded")]), Elsewhere("sonnet-m"))
    view = chain.view()
    assert view.asked is None

    response, asked = await chain.answer(REQUEST)
    await view.complete(REQUEST)

    assert (response.model, asked) == ("served-m", "sonnet-m")
    assert (view.answered_by, view.asked) == ("served-m", "sonnet-m")


async def test_failed_attempts_are_in_the_returned_usage() -> None:
    reported = Usage(prompt_tokens=7, completion_tokens=3, calls=1)
    opus = Scripted("opus-m", [failing("refused", usage=reported)])
    sonnet = Scripted("sonnet-m", [failing("rate_limited")])
    chain = chain_of(opus, sonnet, Scripted("other-m"))

    response = await chain.complete(REQUEST)

    # Opus reported its tokens; Sonnet's are unknown, which is not the same as free.
    assert response.usage == ANSWERED + reported + Usage.unreported()
    assert response.usage.calls == 3
    assert response.usage.unknown_calls == 1


async def test_error_raised_after_a_step_down_carries_the_failed_usage() -> None:
    parse_error = ParseError("not JSON", raw="{", usage=ANSWERED)
    chain = chain_of(
        Scripted("opus-m", [failing("overloaded")]), Scripted("sonnet-m", [parse_error])
    )

    with pytest.raises(ParseError) as raised:
        await chain.complete(REQUEST)

    assert raised.value is parse_error
    assert parse_error.usage == ANSWERED + Usage.unreported()
    assert parse_error.raw == "{"


async def test_error_without_a_usage_slot_is_raised_as_is_after_a_step_down() -> None:
    error = RuntimeError("not a provider error")
    chain = chain_of(Scripted("opus-m", [failing("overloaded")]), Scripted("sonnet-m", [error]))

    with pytest.raises(RuntimeError) as raised:
        await chain.complete(REQUEST)

    assert raised.value is error
    assert not hasattr(error, "usage")


async def test_stage_is_recorded_at_the_time_of_the_step_down() -> None:
    opus = Scripted("opus-m", ["ok", failing("rate_limited")])
    sonnet = Scripted("sonnet-m", ["ok", failing("overloaded")])
    chain = chain_of(opus, sonnet, Scripted("other-m"))

    await chain.complete(REQUEST)
    chain.stage = "select"
    await chain.complete(REQUEST)
    await chain.complete(REQUEST)
    await chain.complete(REQUEST)

    assert chain.fallbacks == [
        Fallback("select", "opus-m", "sonnet-m", "rate_limited"),
        Fallback("select", "sonnet-m", "other-m", "overloaded"),
    ]


async def test_stage_extract_then_select() -> None:
    opus = Scripted("opus-m", [failing("rate_limited")])
    sonnet = Scripted("sonnet-m", ["ok", failing("overloaded")])
    chain = chain_of(opus, sonnet, Scripted("other-m"))

    await chain.complete(REQUEST)
    chain.stage = "select"
    await chain.complete(REQUEST)

    assert [(f.stage, f.reason) for f in chain.fallbacks] == [
        ("extract", "rate_limited"),
        ("select", "overloaded"),
    ]


async def test_views_remember_which_model_answered_them() -> None:
    opus = Scripted("opus-m", ["ok", failing("rate_limited")])
    sonnet = Scripted("sonnet-m")
    chain = chain_of(opus, sonnet)
    first, second = chain.view(), chain.view()
    assert first.answered_by is None

    await first.complete(REQUEST)
    response = await second.complete(REQUEST)

    assert first.answered_by == "opus-m"
    assert second.answered_by == "sonnet-m"
    # The view returns the chain's answer, failed attempt included, and shares its state.
    assert response.usage == ANSWERED + Usage.unreported()
    assert chain.position == 1
    assert len(chain.fallbacks) == 1


async def test_concurrent_views_each_name_their_own_model() -> None:
    """Two picks in flight together: one is answered by step 0 while the other fails
    there and is answered by step 1."""
    outcomes: list[str | BaseException] = ["ok", failing("rate_limited")]
    chain = chain_of(Scripted("opus-m", outcomes), Scripted("sonnet-m"))
    first, second = chain.view(), chain.view()

    await asyncio.gather(first.complete(REQUEST), second.complete(REQUEST))

    assert (first.answered_by, second.answered_by) == ("opus-m", "sonnet-m")


async def test_view_keeps_its_last_answer_when_a_later_call_fails() -> None:
    chain = chain_of(Scripted("opus-m", ["ok", failing("bad_request")]))
    view = chain.view()

    await view.complete(REQUEST)
    with pytest.raises(LLMFatalError):
        await view.complete(REQUEST)

    assert view.answered_by == "opus-m"


@pytest.mark.parametrize("reason", ["balance_exhausted", "key_refused"])
async def test_block_reasons_block_claude(reason: str) -> None:
    chain = chain_of(Scripted("opus-m", [failing(reason)]), Scripted("other-m"))
    assert claude_block() is None

    before = time.monotonic()
    await chain.complete(REQUEST)

    assert claude_block() == reason
    assert claude_block(now=before + BLOCK_TTL_S - 1) == reason
    assert claude_block(now=time.monotonic() + BLOCK_TTL_S) is None


@pytest.mark.parametrize(
    "reason", ["rate_limited", "overloaded", "refused", "transient", "timeout"]
)
async def test_other_reasons_do_not_block_claude(reason: str) -> None:
    chain = chain_of(Scripted("opus-m", [failing(reason)]), Scripted("other-m"))

    await chain.complete(REQUEST)

    assert chain.position == 1
    assert claude_block() is None


def test_block_expires_after_the_ttl_and_reset_clears_it() -> None:
    block_claude("key_refused", now=1000.0)

    assert claude_block(now=1000.0) == "key_refused"
    assert claude_block(now=1000.0 + BLOCK_TTL_S - 0.001) == "key_refused"
    assert claude_block(now=1000.0 + BLOCK_TTL_S) is None

    block_claude("balance_exhausted", now=2000.0)
    assert claude_block(now=2000.0) == "balance_exhausted"
    fallback.reset()
    assert claude_block(now=2000.0) is None


async def test_reason_and_usage_survive_a_budgeted_step() -> None:
    """Claude steps arrive wrapped in the engine's ``BudgetedLLM``: the chain must read
    the reason through it, and the wrapper's own refusal steps down as ``call_limit``."""
    reported = Usage(prompt_tokens=7, completion_tokens=3, calls=1)
    opus = BudgetedLLM(Scripted("opus-m", [failing("rate_limited", usage=reported)]))
    inner_sonnet = Scripted("sonnet-m")
    sonnet = BudgetedLLM(inner_sonnet, CallBudget(1))
    chain = chain_of(opus, sonnet, Scripted("other-m"))

    first = await chain.complete(REQUEST)
    second = await chain.complete(REQUEST)

    assert (first.model, second.model) == ("sonnet-m", "other-m")
    assert first.usage == ANSWERED + reported
    assert second.usage == ANSWERED
    assert inner_sonnet.calls == 1
    assert chain.fallbacks == [
        Fallback("extract", "opus-m", "sonnet-m", "rate_limited"),
        Fallback("extract", "sonnet-m", "other-m", "call_limit"),
    ]
    # The wrappers booked the same calls the chain reported.
    assert opus.usage == reported
    assert sonnet.usage == ANSWERED


async def test_aclose_closes_every_step() -> None:
    class NoClose:
        model = "bare-m"
        fingerprint = "fp-bare"
        capabilities = LLMCapabilities()

        async def complete(self, request: LLMRequest) -> LLMResponse:
            raise AssertionError("not called")

    opus, other = Scripted("opus-m"), Scripted("other-m")
    wrapped = Scripted("sonnet-m")
    chain = FallbackLLM(
        [
            Step("opus", opus),
            Step("sonnet", BudgetedLLM(wrapped)),
            Step("other", NoClose()),
            Step("other", other),
        ],
        start=3,
    )

    await chain.aclose()

    assert (opus.closed, wrapped.closed, other.closed) == (True, True, True)


async def test_aclose_closes_the_rest_when_one_step_fails_to_close() -> None:
    class BrokenClose(Scripted):
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    sonnet = Scripted("sonnet-m")
    chain = chain_of(BrokenClose("opus-m"), sonnet)

    with pytest.raises(RuntimeError, match="close failed"):
        await chain.aclose()

    assert sonnet.closed
