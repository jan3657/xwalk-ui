"""Three in-memory caches, so that a warm instance does not pay a model, the embeddings
endpoint or the database again for work it has just done.

- ``responses``: a whole request -> the response it got (``ner_site.http``).
- ``retrieval``: an ontology and a mention -> the fused candidates (``ner_site.mapping``).
- ``picks``: a selector, a mention in its surroundings and the candidates it was shown
  -> the pick (``ner_site.select``).

Each is a :class:`TTLCache`: at most ``max_entries`` entries (the least recently used
one leaves first) and none older than ``ttl_s`` seconds, behind one lock. Two requests
that miss the same key at once both do the work and both store it; the second store
replaces the first with an equal value. In-flight work is not shared.

Where they live. In the module, per deployment (:func:`caches_for`): a warm serverless
instance keeps them for as long as it lives and no longer, another instance has its own,
and nothing here is written to a database, a file or a log. A cache is therefore a
saving when it happens, never a guarantee.

What a key is. A SHA-256 digest of its parts (:func:`digest`), so a key holds no text;
the values do (a response holds the mentions it found). No key and no value is given a
visitor's API key, the passcode, an address or a visitor id: the callers build keys
from the request's own fields, and store a response only as it is after redaction.

What is stored. Only what a fresh identical run would have been allowed to return:
the callers decide (``service.cacheable`` for a response, and see ``mapping`` and
``select``). A value is never changed after it is stored: a response is kept as JSON
text and parsed anew for every hit, a pick is copied in and out, and candidates are
frozen engine records kept in a tuple.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

V = TypeVar("V")

DEFAULT_RESPONSES = 128
DEFAULT_RETRIEVAL = 2048
DEFAULT_PICKS = 8192
DEFAULT_TTL_S = 6 * 3600


@dataclass(frozen=True)
class CacheSettings:
    """How many entries each cache may hold (0: that cache is off) and how long an
    entry lives. The defaults here are "off": a ``Settings`` built in code caches
    nothing unless it is given sizes, and ``Settings.from_env`` gives a deployment
    :data:`ON` unless the environment says otherwise."""

    responses: int = 0
    retrieval: int = 0
    picks: int = 0
    ttl_s: int = DEFAULT_TTL_S

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0 and (self.responses > 0 or self.retrieval > 0 or self.picks > 0)


ON = CacheSettings(DEFAULT_RESPONSES, DEFAULT_RETRIEVAL, DEFAULT_PICKS, DEFAULT_TTL_S)
OFF = CacheSettings()


def digest(*parts: Any) -> str:
    """The key for `parts` (strings, numbers, None, and lists of them): the SHA-256 of
    their JSON, which keeps ``("ab", "c")`` and ``("a", "bc")`` apart."""
    document = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(document.encode("utf-8", "surrogatepass")).hexdigest()


class TTLCache(Generic[V]):
    """A bounded, expiring, thread-safe mapping. ``clock`` gives seconds that only go
    forward (``time.monotonic``); it is a parameter so that tests can move the time."""

    def __init__(
        self, max_entries: int, ttl_s: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._max = max(0, int(max_entries))
        self._ttl = float(ttl_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[Hashable, tuple[float, V]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max > 0 and self._ttl > 0

    def get(self, key: Hashable) -> tuple[V, float] | None:
        """The value stored under `key` and its age in seconds, or None when there is
        none or it has expired (an expired entry is dropped). A hit makes the entry
        the most recently used."""
        if not self.enabled:
            return None
        now = self._clock()
        with self._lock:
            found = self._entries.get(key)
            if found is None:
                return None
            stored_at, value = found
            age = now - stored_at
            if age >= self._ttl:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value, max(0.0, age)

    def put(self, key: Hashable, value: V) -> None:
        """Store `value`, as the newest entry; the oldest leave when there are too
        many. The caller hands over a value nothing else will change."""
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            self._entries[key] = (now, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def values(self) -> list[V]:
        """What is stored now, expired or not (tests look through it)."""
        with self._lock:
            return [value for _, value in self._entries.values()]

    def digests(self) -> list[Hashable]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class Caches:
    """The three caches of one deployment."""

    def __init__(
        self, settings: CacheSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.settings = settings
        # JSON text of a response, after redaction.
        self.responses: TTLCache[str] = TTLCache(settings.responses, settings.ttl_s, clock=clock)
        # A tuple of ``xwalk.records.Candidate`` (frozen), best first.
        self.retrieval: TTLCache[tuple[Any, ...]] = TTLCache(
            settings.retrieval, settings.ttl_s, clock=clock
        )
        # A ``schema.Pick``; ``select`` copies it in and out.
        self.picks: TTLCache[Any] = TTLCache(settings.picks, settings.ttl_s, clock=clock)

    def clear(self) -> None:
        for one in (self.responses, self.retrieval, self.picks):
            one.clear()


def response_key(
    *, text: str, prompt: str, model: str, selector: str, top_k: int, provider: str
) -> str:
    """Everything of a validated request that decides its answer: the text, the prompt
    actually used, the reader model, the selector, ``top_k`` and who pays (``site`` or
    the own-key provider's id). Never the key itself."""
    return digest("response", text, prompt, model, selector, top_k, provider)


def retrieval_key(slug: str, query: str, retrievers: Sequence[Any]) -> str:
    """One ontology, one rendered query, and the retrievers that were there to answer
    it: each one's name, index identity and depth (how many hits it is asked for, which
    is what decides how many candidates the fused list has). An ontology that lost its
    dense retriever, or got it back, has other keys."""
    return digest(
        "retrieval",
        slug,
        query,
        [
            [
                str(getattr(r, "name", "")),
                str(getattr(r, "fingerprint", "")),
                int(getattr(r, "default_limit", 0) or 0),
            ]
            for r in retrievers
        ],
    )


def pick_key(
    *,
    payer: str,
    selector: str,
    entity_type: str,
    ontology: str,
    mention: str,
    left: str,
    right: str,
    candidate_ids: Sequence[str],
) -> str:
    """One selector's question as it was put: who pays (the site's client and an
    own-key client are configured differently), the model or ``jev``, the mention with
    its type and ontology, the exact text shown on either side of it, and the exact
    candidates, in order. The same mention in other surroundings is another key."""
    return digest(
        "pick", payer, selector, entity_type, ontology, mention, left, right, list(candidate_ids)
    )


# --- one set of caches per deployment and warm instance ---------------------------

_LOCK = threading.Lock()
_CACHES: dict[Hashable, Caches] = {}


def _now() -> float:
    return time.monotonic()


def caches_for(deployment: Hashable, settings: CacheSettings) -> Caches | None:
    """The caches of `deployment` (whatever tells one configuration of the site from
    another in this process; ``Settings.cache_scope``), made on first use; None when
    every cache is off. Tests move the time by replacing ``_now``."""
    if not settings.enabled:
        return None
    key = (deployment, settings)
    with _LOCK:
        found = _CACHES.get(key)
        if found is None:
            found = _CACHES[key] = Caches(settings, clock=lambda: _now())
        return found


def reset() -> None:
    """Forget everything (tests)."""
    with _LOCK:
        _CACHES.clear()


__all__ = [
    "DEFAULT_PICKS",
    "DEFAULT_RESPONSES",
    "DEFAULT_RETRIEVAL",
    "DEFAULT_TTL_S",
    "OFF",
    "ON",
    "CacheSettings",
    "Caches",
    "TTLCache",
    "caches_for",
    "digest",
    "pick_key",
    "reset",
    "response_key",
    "retrieval_key",
]
