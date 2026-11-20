"""In-memory caches, so that a warm instance does not pay a model, the embeddings
endpoint or the database again for work it has just done.

- ``responses``: a whole request -> the response it got (``ner_site.http``). The one
  cache a deployment has by default.
- ``retrieval``: an ontology and a mention -> the fused candidates (``ner_site.mapping``).
- ``picks``: a selector, a mention in its surroundings and the candidates it was shown
  -> the pick (``ner_site.select``).

``retrieval`` and ``picks`` are experimental and off unless a deployment gives them a
size: with either on, a run that is not a whole-response hit does less work when an
earlier text shared a mention with it, and its public call counts, tokens and timings
show that. With both off such a run does all its work.

Each is a :class:`TTLCache`: at most ``max_entries`` entries (the least recently used
one leaves first), optionally at most ``max_bytes`` of stored values, and none older
than ``ttl_s`` seconds, behind one lock (the time of a write is read inside the lock, so
the stored times are in the order of the writes). Two requests that miss the same key at once
both do the work and both store it; the second store replaces the first with an equal
value. In-flight work is not shared.

Who shares them. ``retrieval`` holds candidates of the public ontologies and is shared
by every request. ``responses`` and ``picks`` hold what a model answered, and only for
requests the site paid for: a request on a visitor's own key neither reads nor writes
them (``ner_site.http`` and ``ner_site.service`` see to that), so one visitor's provider
account never answers for another's.

Where they live. In the module, per deployment (:func:`caches_for`): a warm serverless
instance keeps them for as long as it lives and no longer, another instance has its own,
and nothing here is written to a database, a file or a log. A cache is therefore a
saving when it happens, never a guarantee.

How long. An entry is not handed out once it is ``ttl_s`` old, and it is also removed:
every write sweeps its cache's expired entries (they are kept in the order they were
stored, so the sweep stops at the first one still alive), and every request that looks
its caches up (:func:`caches_for`) sweeps all caches of the process. An entry therefore
does not outlive its time limit because nobody asked for its key again; it outlives it
only on an instance that receives no request at all, until the instance is recycled.

What a key is. A SHA-256 digest of its parts (:func:`digest`), so a key holds no text;
the values do (a response holds the mentions it found). The deployment a set of caches
belongs to is named by a digest too (``Settings.cache_scope``). No key and no value is
given a visitor's API key, the passcode, an address or a visitor id: the callers build
keys from the request's own fields, and store nothing in which a configured secret, the
passcode (of any length) or the visitor's key occurs; nor a response in which the
address the request came from occurs.

What is stored. Only what a fresh identical run would have been allowed to return:
the callers decide, each for its own stage (``service.cacheable`` for a response, and
see ``mapping`` and ``select``). A value is never changed after it is stored: a response
is kept as JSON text and parsed anew for every hit, a pick is copied in and out, and
candidates are copied in and out with their records' fields (:func:`copy_candidates`).

How much. A response larger than ``MAX_RESPONSE_BYTES`` is not stored, and the stored
candidates are held to ``MAX_RETRIEVAL_BYTES`` in all (their text, estimated by
:func:`candidates_text`), on top of the entry counts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

V = TypeVar("V")

DEFAULT_RESPONSES = 128
# Experimental, off by default (module docstring): a deployment that wants them gives
# them a size.
DEFAULT_RETRIEVAL = 0
DEFAULT_PICKS = 0
DEFAULT_TTL_S = 6 * 3600
# The longest time limit a deployment may set: the page says "normally for a few hours".
MAX_TTL_S = 24 * 3600
# A stored response is JSON text; the recorded example is 103 kB. One above this is
# not stored at all, so 128 responses are at most 32 MB.
MAX_RESPONSE_BYTES = 256 * 1024
# The text of all stored candidates together (ids, field names and values). Records of
# the large ontologies vary in size, so the entry count alone is no ceiling.
MAX_RETRIEVAL_BYTES = 48 * 1024 * 1024


@dataclass(frozen=True)
class CacheSettings:
    """How many entries each cache may hold (0: that cache is off) and how long an
    entry lives. The defaults here are "off": a ``Settings`` built in code caches
    nothing unless it is given sizes, and ``Settings.from_env`` gives a deployment
    :data:`ON` (whole responses only) unless the environment says otherwise."""

    responses: int = 0
    retrieval: int = 0
    picks: int = 0
    ttl_s: int = DEFAULT_TTL_S

    @property
    def enabled(self) -> bool:
        return self.ttl_s > 0 and (self.responses > 0 or self.retrieval > 0 or self.picks > 0)


ON = CacheSettings(DEFAULT_RESPONSES, DEFAULT_RETRIEVAL, DEFAULT_PICKS, DEFAULT_TTL_S)
# All three, with the sizes the experimental two had while they were on by default: for
# a deployment that tries them, and for their tests.
EXPERIMENTAL = CacheSettings(DEFAULT_RESPONSES, 2048, 8192, DEFAULT_TTL_S)
OFF = CacheSettings()


def digest(*parts: Any) -> str:
    """The key for `parts` (strings, numbers, None, and lists of them): the SHA-256 of
    their JSON, which keeps ``("ab", "c")`` and ``("a", "bc")`` apart."""
    document = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(document.encode("utf-8", "surrogatepass")).hexdigest()


class TTLCache(Generic[V]):
    """A bounded, expiring, thread-safe mapping. ``clock`` gives seconds that only go
    forward (``time.monotonic``); it is a parameter so that tests can move the time.

    With ``max_bytes`` and ``size_of`` the stored values are also bounded by their
    total size: a value larger than ``max_entry_bytes`` (or than ``max_bytes``) is not
    stored, and the least recently used entries leave until the rest fits."""

    def __init__(
        self,
        max_entries: int,
        ttl_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_bytes: int | None = None,
        max_entry_bytes: int | None = None,
        size_of: Callable[[V], int] | None = None,
    ) -> None:
        self._max = max(0, int(max_entries))
        self._ttl = float(ttl_s)
        self._clock = clock
        self._max_bytes = max_bytes
        self._max_entry_bytes = max_entry_bytes
        self._size_of = size_of
        self._bytes = 0
        self._lock = threading.Lock()
        # key -> (stored at, size, value), least recently used first.
        self._entries: OrderedDict[Hashable, tuple[float, int, V]] = OrderedDict()
        # key -> stored at, oldest first: the order in which entries expire.
        self._stored: OrderedDict[Hashable, float] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max > 0 and self._ttl > 0

    @property
    def stored_bytes(self) -> int:
        """The total size of what is stored, as ``size_of`` measured it (0 without)."""
        with self._lock:
            return self._bytes

    def _drop(self, key: Hashable) -> None:
        _, size, _ = self._entries.pop(key)
        self._stored.pop(key, None)
        self._bytes -= size

    def _sweep(self, now: float) -> int:
        """Remove every expired entry (the lock is held). They are at the front of
        ``_stored``, so this costs one step per entry removed and one more."""
        removed = 0
        while self._stored:
            key, stored_at = next(iter(self._stored.items()))
            if now - stored_at < self._ttl:
                break
            self._drop(key)
            removed += 1
        return removed

    def sweep(self) -> int:
        """Remove every expired entry; how many there were."""
        if not self.enabled:
            return 0
        with self._lock:
            return self._sweep(self._clock())

    def get(self, key: Hashable) -> tuple[V, float] | None:
        """The value stored under `key` and its age in seconds, or None when there is
        none or it has expired (an expired entry is dropped). A hit makes the entry
        the most recently used."""
        if not self.enabled:
            return None
        with self._lock:
            now = self._clock()
            found = self._entries.get(key)
            if found is None:
                return None
            stored_at, _, value = found
            age = now - stored_at
            if age >= self._ttl:
                self._drop(key)
                return None
            self._entries.move_to_end(key)
            return value, max(0.0, age)

    def put(self, key: Hashable, value: V) -> bool:
        """Store `value`, as the newest entry, and say whether it was stored (False for
        a value over the size limit, which also removes what was under `key`). Expired
        entries leave first, then the least recently used ones while there are too many
        or they are too large together. The caller hands over a value nothing else
        will change. The time is read with the lock held: two writers then cannot store
        their times in the opposite order of their entries, which would leave an
        expired entry behind a younger one where the sweep does not look."""
        if not self.enabled:
            return False
        size = 0 if self._size_of is None else int(self._size_of(value))
        limit = self._max_entry_bytes
        if self._max_bytes is not None:
            limit = self._max_bytes if limit is None else min(limit, self._max_bytes)
        with self._lock:
            now = self._clock()
            self._sweep(now)
            if key in self._entries:
                self._drop(key)
            if limit is not None and size > limit:
                return False
            self._entries[key] = (now, size, value)
            self._stored[key] = now
            self._bytes += size
            while len(self._entries) > self._max or (
                self._max_bytes is not None and self._bytes > self._max_bytes
            ):
                self._drop(next(iter(self._entries)))
            return True

    def values(self) -> list[V]:
        """What is stored now, expired or not (tests look through it)."""
        with self._lock:
            return [value for _, _, value in self._entries.values()]

    def digests(self) -> list[Hashable]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._stored.clear()
            self._bytes = 0


def candidates_text(candidates: Sequence[Any]) -> str:
    """Every string a tuple of candidates holds (ids, field names and values, the
    retrievers' names), one per line: what its size is estimated from and what is
    searched for a secret before it is stored."""
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            for name, inner in value.items():
                walk(name)
                walk(inner)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for inner in value:
                walk(inner)
        elif value is not None:
            parts.append(str(value))

    for candidate in candidates:
        record = getattr(candidate, "record", None)
        walk(getattr(record, "id", ""))
        walk(getattr(record, "fields", None))
        for hit in getattr(candidate, "evidence", ()):
            walk(getattr(hit, "retriever", ""))
    return "\n".join(parts)


def candidates_size(candidates: Sequence[Any]) -> int:
    """An estimate of what a tuple of candidates occupies: the UTF-8 length of its
    text, and a fixed allowance per candidate for the objects around it."""
    return len(candidates_text(candidates).encode("utf-8", "surrogatepass")) + 256 * len(candidates)


def copy_candidates(candidates: Sequence[Any]) -> tuple[Any, ...]:
    """A copy that shares nothing changeable with `candidates`: a frozen record still
    holds a ``fields`` mapping and lists in it (synonyms), which a later consumer could
    edit. Copied on the way into the cache and on the way out."""
    return tuple(copy.deepcopy(list(candidates)))


def content_of(candidates: Sequence[Any]) -> list[Any]:
    """What a selector is shown of `candidates`, in order: each one's id and every field
    of its record (the prompt renders a subset of them: label, synonyms, definition,
    organism). Two lists with the same ids and another label are not the same question."""
    return [
        [
            str(getattr(c, "id", "")),
            json.dumps(
                dict(getattr(getattr(c, "record", None), "fields", None) or {}),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        ]
        for c in candidates
    ]


class Caches:
    """The three caches of one deployment. Nothing here says which index view a
    stored response belongs to: that is part of each response's key
    (:func:`response_key`), which a request builds once from the indexes it opened."""

    def __init__(
        self, settings: CacheSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.settings = settings
        # JSON text of a response, exactly as it was sent.
        self.responses: TTLCache[str] = TTLCache(
            settings.responses,
            settings.ttl_s,
            clock=clock,
            max_entry_bytes=MAX_RESPONSE_BYTES,
            size_of=lambda body: len(body.encode("utf-8", "surrogatepass")),
        )
        # A tuple of ``xwalk.records.Candidate``, best first; ``mapping`` copies it in
        # and out.
        self.retrieval: TTLCache[tuple[Any, ...]] = TTLCache(
            settings.retrieval,
            settings.ttl_s,
            clock=clock,
            max_bytes=MAX_RETRIEVAL_BYTES,
            size_of=candidates_size,
        )
        # A ``schema.Pick``; ``select`` copies it in and out.
        self.picks: TTLCache[Any] = TTLCache(settings.picks, settings.ttl_s, clock=clock)

    def sweep(self) -> int:
        """Remove the expired entries of all three; how many there were."""
        return sum(one.sweep() for one in (self.responses, self.retrieval, self.picks))

    def clear(self) -> None:
        for one in (self.responses, self.retrieval, self.picks):
            one.clear()


def index_identity(indexes: Mapping[str, Any]) -> str:
    """The identity of an opened index view: per ontology, each retriever's name, index
    fingerprint and depth, each retriever kind's status and whether the mode switched
    dense retrieval off. A catalog that was loaded anew, or a retriever that came or
    went, is another identity. A request computes it once, from the indexes it opened,
    and keeps it in its own key."""
    return digest(
        "indexes",
        [
            [
                str(slug),
                [
                    [
                        str(getattr(r, "name", "")),
                        str(getattr(r, "fingerprint", "")),
                        int(getattr(r, "default_limit", 0) or 0),
                    ]
                    for r in getattr(index, "retrievers", ())
                ],
                sorted(
                    [str(kind), str(getattr(status, "status", ""))]
                    for kind, status in (getattr(index, "status", None) or {}).items()
                ),
                bool(getattr(index, "dense_off", False)),
            ]
            for slug, index in indexes.items()
        ],
    )


def response_key(
    *, text: str, prompt: str, model: str, selector: str, top_k: int, indexes: str
) -> str:
    """Everything of a validated, site-paid request that decides its answer: the text,
    the prompt actually used, the reader model, the selector, ``top_k`` and the index
    view the request opened (:func:`index_identity`). Built once per request and used
    for its lookup and for storing its answer, so an answer made on one view is never
    stored under another. A request on a visitor's own key has no response key: it is
    neither looked up nor stored."""
    return digest("response", text, prompt, model, selector, top_k, indexes)


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
    selector: str,
    entity_type: str,
    ontology: str,
    mention: str,
    left: str,
    right: str,
    candidates: Sequence[Any],
) -> str:
    """One selector's question as it was put, on the site's keys (an own-key request
    has no pick key): the model or ``jev``, the mention with its type and ontology, the
    exact text shown on either side of it, and the candidates as they were shown, in
    order (:func:`content_of`: ids and record fields, so a term whose label or
    definition changed is another question). The same mention in other surroundings
    is another key."""
    return digest(
        "pick", selector, entity_type, ontology, mention, left, right, content_of(candidates)
    )


# --- one set of caches per deployment and warm instance ---------------------------

_LOCK = threading.Lock()
_CACHES: dict[Hashable, Caches] = {}
# The family each set of caches belongs to (see :func:`caches_for`).
_FAMILIES: dict[Hashable, Hashable] = {}


def _now() -> float:
    return time.monotonic()


def sweep_all() -> int:
    """Remove the expired entries of every cache of the process; how many."""
    with _LOCK:
        everything = list(_CACHES.values())
    return sum(one.sweep() for one in everything)


def caches_for(
    deployment: Hashable, settings: CacheSettings, *, family: Hashable | None = None
) -> Caches | None:
    """The caches of `deployment` (whatever tells one configuration of the site from
    another in this process; ``Settings.cache_scope``, a digest), made on first use;
    None when every cache is off. Tests move the time by replacing ``_now``.

    `family` names what stays the same when only a secret of the deployment changes
    (``Settings.cache_family``: everything but the passcodes). When the caches of a
    new `deployment` are made, those of every other deployment of the same family are
    emptied and forgotten: after a passcode is changed in a warm process, what was
    stored under the old one does not stay in memory, and the new passcode starts
    with nothing.

    Every call sweeps what has expired in all caches of the process, also when the
    caches are off for `settings`: an entry nobody asks for again, and the caches of a
    configuration no request uses any more, do not stay for the life of the instance."""
    if not settings.enabled:
        sweep_all()
        return None
    key = (deployment, settings)
    with _LOCK:
        found = _CACHES.get(key)
        if found is None:
            if family is not None:
                for other in [k for k, f in _FAMILIES.items() if f == family and k != key]:
                    _CACHES.pop(other).clear()
                    del _FAMILIES[other]
            found = _CACHES[key] = Caches(settings, clock=lambda: _now())
            if family is not None:
                _FAMILIES[key] = family
    sweep_all()
    return found


def reset() -> None:
    """Forget everything (tests)."""
    with _LOCK:
        _CACHES.clear()
        _FAMILIES.clear()


__all__ = [
    "DEFAULT_PICKS",
    "DEFAULT_RESPONSES",
    "DEFAULT_RETRIEVAL",
    "DEFAULT_TTL_S",
    "EXPERIMENTAL",
    "MAX_RESPONSE_BYTES",
    "MAX_RETRIEVAL_BYTES",
    "MAX_TTL_S",
    "OFF",
    "ON",
    "CacheSettings",
    "Caches",
    "TTLCache",
    "caches_for",
    "candidates_size",
    "candidates_text",
    "content_of",
    "copy_candidates",
    "digest",
    "index_identity",
    "pick_key",
    "reset",
    "response_key",
    "retrieval_key",
    "sweep_all",
]
