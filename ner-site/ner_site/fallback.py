"""The fallback chain: one ``LLMClient`` over an ordered list of models.

Binding: docs/superpowers/specs/2026-11-18-claude-provider-open-access-design.md section
3.2. A request starts on the best model it is allowed (Claude Opus), and when that step
cannot be used it moves to the next one (Claude Sonnet, then an OpenRouter model).
Extraction and selection keep talking to a single ``LLMClient``; which provider answered
is this module's business, and it records every move so that nothing substitutes silently.

How a failed call is read. The chain does not import the Anthropic adapter: it looks only
at the ``reason`` string an error carries (``getattr(exc, "reason", None)``), so any client
that sets the same attribute takes part, and a wrapper that re-raises the inner error
unchanged (the engine's ``BudgetedLLM``) is transparent.

- ``STEP_DOWN_REASONS``, or the engine's ``CallLimitExceeded`` (recorded as
  ``call_limit``): the step is unusable for now, so the chain moves down at once.
- ``transient``: the same step is tried once more, then the chain moves down.
- Anything else, ``bad_request`` and ``ParseError`` included: raised unchanged. A
  malformed request or an unreadable answer would fail the same way on every step.
- The last step has nowhere to go: its error is raised unchanged (after the one retry,
  when it was ``transient``).

Step-down is sticky and shared. The chain serves a whole request, and its position only
ever grows: once it has left a step, no later call tries that step again. A call that was
already in flight on an abandoned step is left to finish there; if it fails, it carries on
from wherever the chain is by then and records nothing, so eight selection calls failing
together on one step produce one ``Fallback``, not eight. This needs no lock: between
reading the position and advancing it there is no ``await``.

No dispatched call is free. The usage of every failed attempt within one ``complete``
(``failure_usage``) is added to the answer's usage, or to the error finally raised when
that error has a ``usage`` slot (an ``LLMError``).

``balance_exhausted`` and ``key_refused`` outlive the request: no later request can use
Claude either, so they are remembered at module level for ``BLOCK_TTL_S`` seconds. A warm
instance then starts later requests on the OpenRouter step instead of paying a failed
call each time, and ``/api/health`` can say why Claude is unavailable.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    failure_usage,
)
from xwalk.llm.budget import CallLimitExceeded
from xwalk.records import Usage

STEP_DOWN_REASONS = frozenset(
    {"rate_limited", "overloaded", "balance_exhausted", "key_refused", "refused"}
)
BLOCK_REASONS = frozenset({"balance_exhausted", "key_refused"})
BLOCK_TTL_S = 300.0

TRANSIENT = "transient"
CALL_LIMIT = "call_limit"

# (reason, monotonic time at which it stops applying), or None while Claude is usable.
_block: tuple[str, float] | None = None


def block_claude(reason: str, *, now: float | None = None) -> None:
    """Remember for ``BLOCK_TTL_S`` seconds that Claude cannot be used, and why.

    ``now`` is a ``time.monotonic()`` reading; tests pass their own.
    """
    global _block
    _block = (reason, (time.monotonic() if now is None else now) + BLOCK_TTL_S)


def claude_block(*, now: float | None = None) -> str | None:
    """The reason Claude is blocked, or None when it is not (or no longer)."""
    if _block is None:
        return None
    reason, until = _block
    return reason if (time.monotonic() if now is None else now) < until else None


def reset() -> None:
    """Forget the block (tests, and an operator who has fixed the key or the balance)."""
    global _block
    _block = None


@dataclass(frozen=True)
class Step:
    """One model in the chain. ``label`` is its tier: ``opus``, ``sonnet`` or ``other``."""

    label: str
    client: LLMClient


@dataclass(frozen=True)
class Fallback:
    """One step-down: while in ``stage`` (``extract`` or ``select``), the chain left
    ``from_model`` for ``to_model``. ``reason`` is a ``STEP_DOWN_REASONS`` value,
    ``transient`` or ``call_limit``."""

    stage: str
    from_model: str
    to_model: str
    reason: str


def _reason(exc: Exception) -> str | None:
    """Why `exc` makes its step unusable, or None when it must reach the caller as is."""
    if isinstance(exc, CallLimitExceeded):
        return CALL_LIMIT
    reason = getattr(exc, "reason", None)
    if reason == TRANSIENT or (isinstance(reason, str) and reason in STEP_DOWN_REASONS):
        return reason
    return None


def _carry(exc: Exception, failed: Usage) -> None:
    """Add the usage of this call's earlier failed attempts to the error about to be
    raised, when it has a ``usage`` slot. With no earlier attempt it is left untouched."""
    if isinstance(exc, LLMError) and failed != Usage.zero():
        exc.usage = failed + failure_usage(exc)


class FallbackLLM:
    """An ``LLMClient`` that answers from the first usable step (module docstring).

    ``stage`` is a plain attribute the caller sets (``extract``, then ``select``); it is
    what a step-down is recorded under. ``model``, ``fingerprint`` and ``capabilities``
    are the current step's.
    """

    def __init__(self, steps: Sequence[Step], *, start: int = 0) -> None:
        self.steps: tuple[Step, ...] = tuple(steps)
        if not self.steps:
            raise ValueError("a fallback chain needs at least one step")
        if not 0 <= start < len(self.steps):
            raise ValueError(f"start must name one of the {len(self.steps)} steps, got {start}")
        self._position = start
        self.stage = "extract"
        self.fallbacks: list[Fallback] = []

    @property
    def position(self) -> int:
        """Index of the current step. It never decreases."""
        return self._position

    @property
    def _current(self) -> LLMClient:
        return self.steps[self._position].client

    @property
    def model(self) -> str:
        return self._current.model

    @property
    def fingerprint(self) -> str:
        return self._current.fingerprint

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._current.capabilities

    async def complete(self, request: LLMRequest) -> LLMResponse:
        failed = Usage.zero()
        index = self._position
        retried = False
        last = len(self.steps) - 1
        while True:
            client = self.steps[index].client
            try:
                response = await client.complete(request)
            except Exception as exc:
                reason = _reason(exc)
                if reason in BLOCK_REASONS:
                    block_claude(reason)
                # A retry is only worth it on the step the chain is still on.
                retry = reason == TRANSIENT and not retried and self._position == index
                if reason is None or (index == last and not retry):
                    _carry(exc, failed)
                    raise
                failed = failed + failure_usage(exc)
                if retry:
                    retried = True
                    continue
                # No await between this check and the assignment: of several calls that
                # failed on the same step, exactly one moves the chain and records it.
                if self._position == index:
                    self._position = index + 1
                    self.fallbacks.append(
                        Fallback(
                            stage=self.stage,
                            from_model=client.model,
                            to_model=self.steps[index + 1].client.model,
                            reason=reason,
                        )
                    )
                index = self._position
                retried = False
                continue
            if failed == Usage.zero():
                return response
            return LLMResponse(
                text=response.text,
                usage=response.usage + failed,
                model=response.model,
                structured=response.structured,
                finish_reason=response.finish_reason,
            )

    def view(self) -> ChainView:
        """A client over this chain that remembers which model answered it."""
        return ChainView(self)

    async def aclose(self) -> None:
        """Close every step that can be closed, also when one of them fails to."""
        first_error: Exception | None = None
        for step in self.steps:
            close = getattr(step.client, "aclose", None)
            if not callable(close):
                continue
            try:
                await close()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error


class ChainView:
    """An ``LLMClient`` that delegates to a chain and remembers ``answered_by``.

    Selection runs several picks at once on one chain, and the chain may step down in
    between, so "the chain's current model" does not say who made a given pick. Each pick
    takes its own view; ``answered_by`` is the model of that view's last successful
    answer (as the answer itself reports it), None before the first.
    """

    def __init__(self, chain: FallbackLLM) -> None:
        self._chain = chain
        self.answered_by: str | None = None

    @property
    def model(self) -> str:
        return self._chain.model

    @property
    def fingerprint(self) -> str:
        return self._chain.fingerprint

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._chain.capabilities

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await self._chain.complete(request)
        self.answered_by = response.model
        return response


__all__ = [
    "BLOCK_REASONS",
    "BLOCK_TTL_S",
    "STEP_DOWN_REASONS",
    "ChainView",
    "Fallback",
    "FallbackLLM",
    "Step",
    "block_claude",
    "claude_block",
    "reset",
]
