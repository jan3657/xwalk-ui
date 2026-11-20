"""The NER site's in-memory caches (ner_site.cache): whole responses, which a deployment
has by default, and the two experimental ones (retrieval, picks), which it has only when
it gives them a size (``EXPERIMENTAL``, ``_live3`` and ``_offline3`` here).

Offline: FakeLLM, FakeDecider, BM25 built in a temporary directory, the limiter's and the
recorder's tables in memory, and an injected clock (nothing here sleeps to let time pass).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from ner_site import cache as caching
from ner_site import fake, fallback, service, usage
from ner_site import http as http_mod
from ner_site import indexes as indexes_mod
from ner_site.cache import EXPERIMENTAL, ON, Caches, CacheSettings, TTLCache
from ner_site.indexes import OntologyIndex
from ner_site.mapping import MapStats, map_entities
from ner_site.schema import CacheInfo, Entity, NerRequest, NerResponse, RetrieverStatus
from ner_site.select import JEV, SelectStats, select_entities
from ner_site.service import Settings, SettingsError, cacheable
from xwalk.decide.fake import FakeDecider
from xwalk.llm.base import LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.records import Candidate, Record, RetrievalHit
from xwalk.retrieval.base import SearchRequest

from tests.test_site_select import TEXT as PICK_TEXT
from tests.test_site_select import _entity as _picked_entity
from tests.test_site_select import _key_of, _knows_aspirin, _setup

PROJECT = Path(__file__).resolve().parents[1] / "ner-site"
SITE_KEY = "sk-or-v1-SENTINEL-c0ffee-4242"
ANTHROPIC_KEY = "sk-ant-SENTINEL-feedbeef-9191"
DB_URL = "postgresql://owner:npg_SENTINELpw0123@ep-x-pooler.c-1.eu.aws.neon.tech/neondb"
OWN_KEY = 'visitor-sentinel-"quoted"-\\tail'
PASSCODE = "correct horse battery"
RAW_IP = "203.0.113.77"
OPUS, SONNET, LUNA = "claude-opus-5-5", "claude-sonnet-5-5", "openai/gpt-6-luna"
HEADERS = {"content-type": "application/json", "x-requested-with": "xwalk-ner", "x-real-ip": RAW_IP}
HALO_TEXT = "She took haloperidol for a fever."
KETA_TEXT = "He was given haloperidol and later ketamine."
OK = RetrieverStatus(status="ok")
Handled = tuple[int, dict[str, str], bytes]


def _answer(*items: tuple[str, str, int, int]) -> str:
    return json.dumps(
        {"entities": [{"text": t, "type": ty, "start": s, "end": e} for t, ty, s, e in items]}
    )


HALO = _answer(("haloperidol", "chemical", 9, 20))
PICK = '{"chosen_key": "C01", "confidence_score": 0.7, "explanation": "first"}'


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _fresh() -> Any:
    indexes_mod.reset()
    caching.reset()
    fallback.reset()
    yield
    indexes_mod.reset()
    caching.reset()
    fallback.reset()


@pytest.fixture(scope="module")
def bm25_only(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return indexes_mod.open_indexes(
        cache_dir=tmp_path_factory.mktemp("bm25"), dense_dir=None, api_key=None
    )


@pytest.fixture
def clean(bm25_only: Any) -> dict[str, OntologyIndex]:
    """The bundled samples with every retriever status ``ok``: a deployment whose
    retrieval is not degraded, without a dense index to build."""
    return {
        slug: OntologyIndex(slug, index.store, list(index.retrievers), {"bm25": OK, "dense": OK})
        for slug, index in bm25_only.items()
    }


def _switched_off(index: OntologyIndex) -> OntologyIndex:
    """`index` as offline settings open it: dense retrieval switched off by the mode,
    with the one status that says so."""
    return dataclasses.replace(
        index, dense_off=True, status={**index.status, "dense": indexes_mod.dense_disabled()}
    )


@pytest.fixture
def offline_indexes(bm25_only: Any) -> dict[str, OntologyIndex]:
    """What offline settings open: BM25 only, dense switched off by the mode."""
    return {slug: _switched_off(index) for slug, index in bm25_only.items()}


# --- TTLCache: bounds, expiry, concurrency ------------------------------------------


def test_a_value_is_found_again_with_its_age() -> None:
    clock = Clock()
    cache: TTLCache[str] = TTLCache(4, 100, clock=clock)
    assert cache.get("a") is None
    cache.put("a", "one")
    clock.now += 30
    assert cache.get("a") == ("one", 30.0)


def test_the_least_recently_used_entry_leaves_first() -> None:
    cache: TTLCache[int] = TTLCache(2, 100, clock=Clock())
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is not None  # "a" is now the most recently used
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") is not None and cache.get("c") is not None
    assert len(cache) == 2


def test_an_entry_expires_at_the_time_limit_and_is_dropped() -> None:
    clock = Clock()
    cache: TTLCache[int] = TTLCache(8, 60, clock=clock)
    cache.put("a", 1)
    clock.now += 59.9
    assert cache.get("a") is not None
    clock.now += 0.1
    assert cache.get("a") is None and len(cache) == 0
    # A hit does not make an entry younger: its age counts from when it was stored.
    cache.put("b", 2)
    clock.now += 40
    assert cache.get("b") is not None
    clock.now += 40
    assert cache.get("b") is None


def test_a_write_removes_the_entries_that_have_expired() -> None:
    """An entry does not stay because nobody asked for its key again: the next write to
    its cache removes it, whatever key that write is for."""
    clock = Clock()
    cache: TTLCache[str] = TTLCache(8, 60, clock=clock)
    cache.put("old", "a mention")
    cache.put("older still", "another")
    clock.now += 30
    cache.put("young", "kept")
    assert cache.get("old") is not None  # read since: the newest in use, still the oldest stored
    clock.now += 30
    cache.put("unrelated", "new")
    assert sorted(cache.values()) == ["kept", "new"] and len(cache) == 2
    assert set(cache.digests()) == {"young", "unrelated"}
    # Stored again, an entry is as young as its last store.
    clock.now += 59
    cache.put("unrelated", "newer")
    clock.now += 1
    cache.put("x", "y")
    assert sorted(cache.values()) == ["newer", "y"]
    # And without any write: sweep() does the same.
    clock.now += 60
    assert cache.sweep() == 2 and len(cache) == 0 and cache.sweep() == 0


def test_a_request_sweeps_every_cache_of_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """``caches_for`` runs for every request: what has expired leaves all three caches,
    and the caches of a configuration nobody uses any more."""
    clock = Clock()
    monkeypatch.setattr(caching, "_now", clock)
    one = caching.caches_for("one", EXPERIMENTAL)
    other = caching.caches_for("other", EXPERIMENTAL)
    assert one is not None and other is not None
    one.responses.put("r", "{}")
    one.retrieval.put("c", ())
    one.picks.put("p", "pick")
    other.responses.put("r", "{}")
    clock.now += EXPERIMENTAL.ttl_s - 1
    caching.caches_for("one", EXPERIMENTAL)
    assert (len(one.responses), len(one.retrieval), len(one.picks), len(other.responses)) == (
        1,
        1,
        1,
        1,
    )
    clock.now += 1
    assert caching.caches_for("one", EXPERIMENTAL) is one
    assert (len(one.responses), len(one.retrieval), len(one.picks), len(other.responses)) == (
        0,
        0,
        0,
        0,
    )
    # A request of a deployment whose caches are off sweeps the others' too.
    other.responses.put("r", "{}")
    clock.now += EXPERIMENTAL.ttl_s
    assert caching.caches_for("off", caching.OFF) is None
    assert len(other.responses) == 0


def test_a_cache_is_bounded_by_the_size_of_what_it_holds() -> None:
    cache: TTLCache[str] = TTLCache(100, 60, clock=Clock(), max_bytes=10, size_of=len)
    cache.put("a", "aaaa")
    cache.put("b", "bbbb")
    assert cache.stored_bytes == 8
    assert cache.get("a") is not None  # "b" is now the least recently used
    assert cache.put("c", "cccc") is True
    assert cache.get("b") is None and cache.stored_bytes == 8 and len(cache) == 2
    # A value that could never fit is not stored, and does not push the others out.
    assert cache.put("d", "d" * 11) is False
    assert len(cache) == 2 and cache.get("d") is None
    # Replacing a value counts its new size, not both.
    cache.put("a", "aa")
    assert cache.stored_bytes == 6
    # One entry's own limit.
    capped: TTLCache[str] = TTLCache(100, 60, clock=Clock(), max_entry_bytes=3, size_of=len)
    assert capped.put("k", "abc") is True and capped.put("k", "abcd") is False
    assert capped.get("k") is None and capped.stored_bytes == 0  # the old value is gone too


def test_the_stored_candidates_have_a_ceiling_in_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    def candidates(label: str) -> tuple[Candidate, ...]:
        record = Record("X:1", {"label": label, "synonyms": [label, label]})
        return (Candidate(record, 1.0, (RetrievalHit("X:1", "bm25", None, 1),)),)

    small, large = candidates("a" * 100), candidates("b" * 2000)
    assert 300 < caching.candidates_size(small) < 1000 < 6000 < caching.candidates_size(large)
    assert "a" * 100 in caching.candidates_text(small) and "synonyms" in caching.candidates_text(
        small
    )
    monkeypatch.setattr(caching, "MAX_RETRIEVAL_BYTES", 2 * caching.candidates_size(small))
    cache = Caches(EXPERIMENTAL).retrieval
    for key in ("one", "two", "three"):
        assert cache.put(key, small)
    assert len(cache) == 2 and cache.get("one") is None  # the oldest left for the third
    assert cache.put("big", large) is False and len(cache) == 2
    assert cache.stored_bytes <= caching.MAX_RETRIEVAL_BYTES
    assert caching.MAX_RESPONSE_BYTES == 256 * 1024


def test_the_older_query_vector_cache_has_the_same_time_limit() -> None:
    from tests.test_site_mapping import ToyEncoder

    clock = Clock()
    toy = ToyEncoder()
    encoder = indexes_mod.PrefetchEncoder(toy, clock=clock)
    assert encoder.TTL_S == caching.DEFAULT_TTL_S
    assert encoder.prefetch(["haloperidol", "ketamine"]) == 1
    assert encoder.cached("haloperidol") and len(encoder) == 2
    clock.now += encoder.TTL_S - 1
    encoder.encode(["ketamine"], is_query=True)
    assert len(toy.calls) == 1  # served from memory
    clock.now += 1
    # Expired: not used, embedded again when asked for, and removed by the next write.
    assert not encoder.cached("haloperidol")
    encoder.encode(["aspirin"], is_query=True)
    assert len(toy.calls) == 2 and len(encoder) == 1 and encoder.cached("aspirin")
    assert encoder.prefetch(["haloperidol"]) == 1 and len(toy.calls) == 3


@pytest.mark.parametrize(("size", "ttl"), [(0, 60), (8, 0)])
def test_a_size_or_a_time_limit_of_zero_turns_a_cache_off(size: int, ttl: int) -> None:
    cache: TTLCache[int] = TTLCache(size, ttl, clock=Clock())
    cache.put("a", 1)
    assert not cache.enabled and cache.get("a") is None and len(cache) == 0


def test_concurrent_readers_and_writers_leave_the_cache_whole() -> None:
    cache: TTLCache[int] = TTLCache(50, 100)
    failures: list[BaseException] = []
    start = threading.Barrier(8)

    def work(worker: int) -> None:
        try:
            start.wait(timeout=5)
            for i in range(2000):
                key = f"k{(i * (worker + 1)) % 120}"
                cache.put(key, i)
                found = cache.get(key)
                assert found is None or isinstance(found[0], int)
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            failures.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not failures and 0 < len(cache) <= 50
    assert len(set(cache.digests())) == len(cache)


def test_a_key_is_a_digest_that_keeps_its_parts_apart() -> None:
    assert caching.digest("ab", "c") != caching.digest("a", "bc")
    assert caching.digest("a", ["b", "c"]) != caching.digest("a", ["bc"])
    key = caching.response_key(
        text="secret text", prompt="p", model="m", selector="s", top_k=5, indexes="view-1"
    )
    assert len(key) == 64 and "secret" not in key
    # The index view the answer was made on is part of the key.
    assert key != caching.response_key(
        text="secret text", prompt="p", model="m", selector="s", top_k=5, indexes="view-2"
    )


# --- settings ------------------------------------------------------------------------


def test_the_environment_turns_the_response_cache_on_and_leaves_the_other_two_off() -> None:
    s = Settings.from_env({})
    assert s.cache == ON == CacheSettings(128, 0, 0, 6 * 3600)
    assert s.cache.enabled
    caches = caching.caches_for(s.cache_scope, s.cache)
    assert caches is not None and caches.responses.enabled
    assert not caches.retrieval.enabled and not caches.picks.enabled
    # The experimental two are there for a deployment that gives them a size.
    assert CacheSettings(128, 2048, 8192, 6 * 3600) == EXPERIMENTAL
    # With the response cache alone switched off nothing is left on.
    assert not Settings.from_env({"NER_SITE_CACHE_RESPONSES": "0"}).cache.enabled
    # A Settings built in code caches nothing unless it is given sizes.
    assert not Settings().cache.enabled


@pytest.mark.parametrize("value", ["off", "0", "false", "NO"])
def test_one_variable_turns_all_three_off(value: str) -> None:
    s = Settings.from_env({"NER_SITE_CACHE": value, "NER_SITE_CACHE_PICKS": "9"})
    assert not s.cache.enabled and caching.caches_for(s.cache_scope, s.cache) is None


def test_each_cache_has_its_own_size_and_zero_turns_it_off() -> None:
    s = Settings.from_env(
        {
            "NER_SITE_CACHE": "on",
            "NER_SITE_CACHE_RESPONSES": "0",
            "NER_SITE_CACHE_RETRIEVAL": "7",
            "NER_SITE_CACHE_PICKS": "9",
            "NER_SITE_CACHE_TTL_S": "60",
        }
    )
    assert s.cache == CacheSettings(0, 7, 9, 60)
    caches = caching.caches_for(s.cache_scope, s.cache)
    assert caches is not None
    assert not caches.responses.enabled and caches.retrieval.enabled and caches.picks.enabled
    assert not Settings.from_env({"NER_SITE_CACHE_TTL_S": "0"}).cache.enabled


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("NER_SITE_CACHE", "maybe"),
        ("NER_SITE_CACHE_RESPONSES", "-1"),
        ("NER_SITE_CACHE_RESPONSES", "many"),
        ("NER_SITE_CACHE_RETRIEVAL", "999999999"),
        ("NER_SITE_CACHE_PICKS", "1.5"),
        ("NER_SITE_CACHE_TTL_S", "forever"),
        ("NER_SITE_CACHE_TTL_S", "86401"),  # "a few hours" on the page: a day at most
        ("NER_SITE_CACHE_TTL_S", "604800"),
    ],
)
def test_a_bad_cache_value_is_a_settings_error(name: str, value: str) -> None:
    with pytest.raises(SettingsError, match=name):
        Settings.from_env({name: value})


def test_the_cache_variables_are_part_of_the_settings_a_warm_instance_remembers() -> None:
    names = {
        "NER_SITE_CACHE",
        "NER_SITE_CACHE_RESPONSES",
        "NER_SITE_CACHE_RETRIEVAL",
        "NER_SITE_CACHE_PICKS",
        "NER_SITE_CACHE_TTL_S",
    }
    assert names <= set(service.ENV_NAMES)
    on = http_mod.env_settings({})
    off = http_mod.env_settings({"NER_SITE_CACHE": "off"})
    assert isinstance(on, Settings) and isinstance(off, Settings)
    assert on.cache.enabled and not off.cache.enabled


def test_two_deployments_in_one_process_do_not_share_a_cache() -> None:
    one = Settings.from_env({"NER_SITE_OFFLINE": "1"})
    other = Settings.from_env({"NER_SITE_OFFLINE": "1", "NER_SITE_MAX_ENTITIES": "7"})
    a = caching.caches_for(one.cache_scope, one.cache)
    assert a is caching.caches_for(one.cache_scope, one.cache)
    assert a is not caching.caches_for(other.cache_scope, other.cache)


def test_the_name_of_a_deployments_caches_is_a_digest_that_holds_nothing() -> None:
    """A database URL carries a password, and a path is nobody's business either: the
    scope is one digest, and nothing of the configuration is in it."""
    env = {
        "OPENROUTER_API_KEY": SITE_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
        "NER_SITE_PASSCODE": PASSCODE,
        "DATABASE_URL": DB_URL,
        "NER_SITE_BASE_URL": "https://proxy.example/v1",
        "NER_SITE_CACHE_DIR": "/tmp/SENTINEL-dir",
    }
    keyed = Settings.from_env(env)
    scope = keyed.cache_scope
    assert isinstance(scope, str) and re.fullmatch(r"[0-9a-f]{64}", scope)
    registered = repr(list(caching._CACHES)) + repr(caching.caches_for(scope, keyed.cache))  # noqa: SLF001
    for secret in (
        SITE_KEY,
        ANTHROPIC_KEY,
        PASSCODE,
        "npg_SENTINEL",
        "proxy.example",
        "SENTINEL-dir",
    ):
        assert secret not in scope and secret not in registered, secret
    # Another operator key (another account) or another base URL is another scope, and
    # so is another passcode: what may be stored depends on it.
    assert Settings.from_env({**env, "OPENROUTER_API_KEY": SITE_KEY + "x"}).cache_scope != scope
    assert Settings.from_env({**env, "ANTHROPIC_API_KEY": ANTHROPIC_KEY + "x"}).cache_scope != scope
    assert (
        Settings.from_env({**env, "NER_SITE_BASE_URL": "https://other.example/v1"}).cache_scope
        != scope
    )
    rotated = Settings.from_env({**env, "NER_SITE_PASSCODE": "ok"})
    assert rotated.cache_scope != scope and rotated.cache_family == keyed.cache_family
    assert Settings.from_env(env).cache_scope == scope
    assert PASSCODE not in keyed.cache_family and SITE_KEY not in keyed.cache_family


# --- the retrieval cache (mapping) ---------------------------------------------------


class Counting:
    """A retriever that counts its searches and answers like the one it wraps."""

    def __init__(self, inner: Any, *, depth: int | None = None) -> None:
        self._inner = inner
        self._depth = depth
        self.searches: list[str] = []

    @property
    def name(self) -> str:
        return str(self._inner.name)

    @property
    def fingerprint(self) -> str:
        return str(self._inner.fingerprint)

    @property
    def default_limit(self) -> int:
        return self._depth or int(self._inner.default_limit)

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        self.searches.append(request.text)
        found: Sequence[RetrievalHit] = await self._inner.search(request)
        return found


class Broken:
    """A second retriever that fails, or answers too late."""

    name = "broken"
    fingerprint = "broken-1"
    default_limit = 5

    def __init__(self, delay: float | None = None) -> None:
        self.delay = delay
        self.fixed = False

    async def search(self, request: SearchRequest) -> Sequence[RetrievalHit]:
        if self.fixed:
            return []
        if self.delay is not None:
            await asyncio.sleep(self.delay)
        raise RuntimeError("the index is on fire")


def _counted(clean: dict[str, OntologyIndex]) -> tuple[dict[str, OntologyIndex], dict[str, Any]]:
    counters = {slug: Counting(index.retrievers[0]) for slug, index in clean.items()}
    indexes = {
        slug: dataclasses.replace(index, retrievers=[counters[slug]])
        for slug, index in clean.items()
    }
    return indexes, counters


def _mention(eid: str, text: str, etype: str = "chemical") -> Entity:
    return Entity(id=eid, text=text, start=0, end=len(text), type=etype)  # type: ignore[arg-type]


def _map(entities: list[Entity], indexes: Any, cache: Any, **kw: Any) -> tuple[Any, MapStats]:
    stats = MapStats()
    results = asyncio.run(
        map_entities(entities, indexes, top_k=kw.pop("top_k", 3), stats=stats, cache=cache, **kw)
    )
    return results, stats


def test_the_same_mention_is_not_searched_for_twice(clean: Any) -> None:
    indexes, counters = _counted(clean)
    cache = Caches(EXPERIMENTAL).retrieval
    first, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    assert stats.cache_hits == 0 and counters["chebi-sample"].searches == ["haloperidol"]
    # In another text, under another entity id and another top_k: the same candidates.
    again, stats = _map([_mention("e7", "haloperidol")], indexes, cache, top_k=1)
    assert stats.cache_hits == 1 and counters["chebi-sample"].searches == ["haloperidol"]
    assert again[0].id == "e7" and again[0].notes == []
    assert again[0].matches == first[0].matches[:1]
    assert [m.id for m in (*again[0].matches, *again[0].more)] == [
        m.id for m in (*first[0].matches, *first[0].more)
    ]
    assert [c.id for c in stats.candidates["e7"]] == [m.id for m in first[0].matches] + [
        m.id for m in first[0].more
    ]


def test_without_a_cache_every_mention_is_searched_for(clean: Any) -> None:
    indexes, counters = _counted(clean)
    for _ in range(2):
        _, stats = _map([_mention("e1", "haloperidol")], indexes, None)
        assert stats.cache_hits == 0
    assert len(counters["chebi-sample"].searches) == 2


def test_a_retrieval_key_is_the_ontology_the_mention_and_the_retrievers(clean: Any) -> None:
    indexes, counters = _counted(clean)
    cache = Caches(EXPERIMENTAL).retrieval
    _map([_mention("e1", "haloperidol")], indexes, cache)

    def searched(entity: Entity, using: Any = indexes) -> int:
        _, stats = _map([entity], using, cache)
        return 1 - stats.cache_hits

    # Another mention, and the same mention sent to another ontology: both searched.
    assert searched(_mention("e1", "ketamine")) == 1
    assert searched(_mention("e1", "haloperidol", "food")) == 1
    # Other retrievers for the same ontology: a second one, and one asked for another
    # number of candidates.
    chebi = indexes["chebi-sample"]
    fixed = Broken()
    fixed.fixed = True
    two = {
        **indexes,
        "chebi-sample": dataclasses.replace(chebi, retrievers=[*chebi.retrievers, fixed]),
    }
    assert searched(_mention("e1", "haloperidol"), two) == 1
    deeper = Counting(clean["chebi-sample"].retrievers[0], depth=7)
    other_depth = {**indexes, "chebi-sample": dataclasses.replace(chebi, retrievers=[deeper])}
    assert searched(_mention("e1", "haloperidol"), other_depth) == 1
    # And each of those is itself remembered.
    assert searched(_mention("e1", "haloperidol"), two) == 0
    assert searched(_mention("e1", "haloperidol")) == 0


@pytest.mark.parametrize("kind", ["failed", "timed out", "cut off"])
def test_a_retrieval_that_was_not_clean_is_never_stored(kind: str, clean: Any) -> None:
    broken = Broken(delay=None if kind == "failed" else 5.0)
    chebi = clean["chebi-sample"]
    indexes = {
        **clean,
        "chebi-sample": dataclasses.replace(chebi, retrievers=[*chebi.retrievers, broken]),
    }
    cache = Caches(EXPERIMENTAL).retrieval
    limits = {"timed out": {"timeout": 0.05}, "cut off": {"timeout": 5.0, "deadline": 0.05}}
    results, stats = _map([_mention("e1", "haloperidol")], indexes, cache, **limits.get(kind, {}))
    assert results[0].matches and results[0].notes  # BM25 answered; the other left a note
    assert len(cache) == 0
    # Once the retriever works again the mention is searched for, and then remembered.
    broken.fixed = True
    results, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    assert stats.cache_hits == 0 and results[0].notes == [] and len(cache) == 1
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    assert stats.cache_hits == 1


def test_a_retrieval_hit_embeds_nothing_and_asks_no_retriever(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    from tests.test_site_mapping import ToyEncoder, _tool

    out = tmp_path_factory.mktemp("dense") / "indexes"
    toy = ToyEncoder()
    _tool().build_all(out, ToyEncoder())
    indexes = indexes_mod.open_indexes(
        cache_dir=tmp_path_factory.mktemp("c"),
        dense_dir=out,
        api_key=None,
        encoder_factory=lambda: toy,
    )
    cache = Caches(EXPERIMENTAL).retrieval
    entities = [_mention("e1", "acetic acid"), _mention("e2", "cheese", "food")]
    first, stats = _map(entities, indexes, cache)
    assert stats.embedding_requests == 1 and stats.dense_entities == ["e1", "e2"]
    queries = len([call for call in toy.calls if call[0]])
    # One remembered mention and one new one: only the new one is embedded.
    indexes["chebi-sample"].encoder._cache.clear()  # noqa: SLF001 - so a search would embed
    again, stats = _map(
        [_mention("e1", "acetic acid"), _mention("e3", "citric acid")], indexes, cache
    )
    assert stats.cache_hits == 1 and stats.dense_entities == ["e3"]
    sent = [call[1] for call in toy.calls if call[0]][queries:]
    assert sent == [["citric acid"]] and stats.embedding_requests == 1
    assert again[0].matches == first[0].matches


def test_what_a_request_does_to_its_candidates_does_not_reach_the_cache(clean: Any) -> None:
    indexes, _ = _counted(clean)
    cache = Caches(EXPERIMENTAL).retrieval
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    kept = [c.id for c in stats.candidates["e1"]]
    stats.candidates["e1"].clear()
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    stats.candidates["e1"].reverse()
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    assert stats.cache_hits == 1 and [c.id for c in stats.candidates["e1"]] == kept
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.candidates["e1"][0].fused_score = 9.0  # type: ignore[misc]


def test_a_change_inside_a_candidates_record_does_not_reach_the_cache(clean: Any) -> None:
    """The records are frozen, their ``fields`` and the lists in them are not: a hit is
    a copy all the way down, and so is what was stored."""
    indexes, _ = _counted(clean)
    cache = Caches(EXPERIMENTAL).retrieval
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    first = stats.candidates["e1"][0]
    label, fields = first.record.fields["label"], dict(first.record.fields)
    listed = next((k for k, v in fields.items() if isinstance(v, list)), None)
    # What was stored shares nothing with what the run itself got back (whose records
    # are the store's own, which this test therefore leaves alone).
    kept = cache.values()[0][0]
    assert kept is not first and kept.record is not first.record
    assert kept.record.fields is not first.record.fields
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    hit = stats.candidates["e1"][0]
    assert stats.cache_hits == 1 and hit.record.fields["label"] == label
    hit.record.fields["label"] = "changed by a hit"  # type: ignore[index]
    hit.record.fields["added"] = "by a hit"  # type: ignore[index]
    if listed is not None:
        hit.record.fields[listed].append("a synonym nobody stored")
    results, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    again = stats.candidates["e1"][0]
    assert stats.cache_hits == 1 and dict(again.record.fields) == {**fields, "label": label}
    assert results[0].matches[0].label == label
    assert again is not hit and again.record is not hit.record
    # Nor is the stored value, or the store's own record, touched by what a hit did.
    assert kept.record.fields == fields and first.record.fields == fields
    assert cache.values()[0][0] is kept


def test_candidates_of_an_ontology_without_one_of_its_retrievers_are_not_stored(
    bm25_only: Any, clean: Any
) -> None:
    """Dense retrieval is configured and not there: BM25 answered, and its candidates
    are a degraded answer that no later request may be handed."""
    cache = Caches(EXPERIMENTAL).retrieval
    for _ in range(2):
        results, stats = _map([_mention("e1", "haloperidol")], bm25_only, cache)
        assert results[0].matches and results[0].notes == []
        assert stats.cache_hits == 0 and stats.degraded == {"e1"} and len(cache) == 0
    # No keyword index is no better.
    no_bm25 = {
        slug: dataclasses.replace(
            index, status={"bm25": RetrieverStatus(status="unavailable", reason="x"), "dense": OK}
        )
        for slug, index in clean.items()
    }
    _, stats = _map([_mention("e1", "haloperidol")], no_bm25, cache)
    assert stats.degraded == {"e1"} and len(cache) == 0
    # With every retriever there, it is stored, and nothing is degraded.
    _, stats = _map([_mention("e1", "haloperidol")], clean, cache)
    assert stats.degraded == set() and len(cache) == 1
    # A failed search is degraded too (and was never stored).
    chebi = clean["chebi-sample"]
    broken = {
        **clean,
        "chebi-sample": dataclasses.replace(chebi, retrievers=[*chebi.retrievers, Broken()]),
    }
    _, stats = _map([_mention("e2", "ketamine")], broken, cache)
    assert stats.degraded == {"e2"} and len(cache) == 1


def test_dense_switched_off_by_the_mode_is_the_one_exception(bm25_only: Any) -> None:
    """Offline no embedding is requested, by design: the index says so explicitly
    (``dense_off``) and its BM25 candidates are what a fresh run would get again."""
    assert not any(index.whole for index in bm25_only.values())
    marked = {s: _switched_off(i) for s, i in bm25_only.items()}
    assert all(index.whole for index in marked.values())
    # The mark alone is not the state: the status has to be the one that says so.
    assert not any(dataclasses.replace(i, dense_off=True).whole for i in bm25_only.values())
    cache = Caches(EXPERIMENTAL).retrieval
    _, stats = _map([_mention("e1", "haloperidol")], marked, cache)
    assert stats.degraded == set() and len(cache) == 1
    # The mark does not excuse a missing keyword index.
    chebi = marked["chebi-sample"]
    gone = dataclasses.replace(
        chebi, status={**chebi.status, "bm25": RetrieverStatus(status="unavailable", reason="x")}
    )
    assert not gone.whole


def test_offline_settings_mark_the_indexes_and_live_settings_do_not(tmp_path: Path) -> None:
    offline = indexes_mod.get_indexes(_offline(tmp_path).index_settings())
    assert all(index.dense_off and index.whole for index in offline.values())
    assert all(i.status["dense"] == indexes_mod.dense_disabled() for i in offline.values())
    live = indexes_mod.get_indexes(_live(tmp_path, OPENROUTER_API_KEY="").index_settings())
    assert live is not offline
    assert not any(index.dense_off or index.whole for index in live.values())


def test_candidates_are_not_stored_when_a_secret_is_in_them(clean: Any) -> None:
    indexes, _ = _counted(clean)
    cache = Caches(EXPERIMENTAL).retrieval
    _map([_mention("e1", "haloperidol")], indexes, cache, sensitive=lambda text: "halo" in text)
    assert len(cache) == 0
    _map([_mention("e1", "haloperidol")], indexes, cache, sensitive=lambda text: "zzz" in text)
    assert len(cache) == 1


# --- the pick cache (select) ---------------------------------------------------------


def _choose(record_id: str = "CHEBI:15365", explanation: str = "x") -> Any:
    def reply(request: LLMRequest) -> str:
        return json.dumps(
            {
                "chosen_key": _key_of(request, record_id),
                "confidence_score": 0.9,
                "explanation": explanation,
            }
        )

    return reply


def _select(selector: str, cache: Any, **kw: Any) -> tuple[Any, SelectStats]:
    indexes, candidates = _setup()
    stats = SelectStats()
    results = asyncio.run(
        select_entities(
            kw.pop("results", [_picked_entity()]),
            kw.pop("candidates", candidates),
            indexes,
            selector=selector,
            text=kw.pop("text", PICK_TEXT),
            stats=stats,
            cache=cache,
            **kw,
        )
    )
    return results, stats


def test_the_same_question_is_not_put_to_the_selector_twice() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose())
    (first,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 0 and stats.calls == 1 and len(llm.requests) == 1
    (again,), stats = _select(
        LUNA,
        cache,
        llm=llm,
        results=[_picked_entity(id="e9")],
        candidates={"e9": _setup()[1]["e1"]},
    )
    assert stats.cache_hits == 1 and stats.calls == 0 and len(llm.requests) == 1
    assert stats.prompt_tokens == 0 and again.id == "e9"
    assert again.pick == first.pick and again.pick.id == "CHEBI:15365"


def test_the_same_mention_in_other_surroundings_is_asked_about_again() -> None:
    """ "drank green tea" is the drink and "dried green tea" the leaf: a pick belongs to
    the context the selector saw, never to the mention alone."""
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose())
    _select(LUNA, cache, llm=llm)
    other = "Nobody took aspirin today."
    start = other.index("aspirin")
    moved = _picked_entity(start=start, end=start + 7)
    _, stats = _select(LUNA, cache, llm=llm, text=other, results=[moved])
    assert stats.cache_hits == 0 and len(llm.requests) == 2
    # Only what the selector was shown counts: 150 characters either side. Text
    # further away may differ.
    long = "A" * 200 + " took aspirin daily " + "B" * 200
    at = long.index("aspirin")
    inside = _picked_entity(start=at, end=at + 7)
    _, stats = _select(LUNA, cache, llm=llm, text=long, results=[inside])
    assert stats.cache_hits == 0 and len(llm.requests) == 3
    _, stats = _select(LUNA, cache, llm=llm, text="C" + long[1:-1] + "D", results=[inside])
    assert stats.cache_hits == 1 and len(llm.requests) == 3
    near = long[: at - 150] + "C" + long[at - 149 :]
    _, stats = _select(LUNA, cache, llm=llm, text=near, results=[inside])
    assert stats.cache_hits == 0 and len(llm.requests) == 4


def test_the_same_context_with_other_candidates_is_asked_about_again() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose())
    candidates = _setup()[1]["e1"]
    _select(LUNA, cache, llm=llm)
    for shown in (list(reversed(candidates)), candidates[1:]):
        _, stats = _select(LUNA, cache, llm=llm, candidates={"e1": shown})
        assert stats.cache_hits == 0
    assert len(llm.requests) == 3
    _, stats = _select(LUNA, cache, llm=llm, candidates={"e1": list(reversed(candidates))})
    assert stats.cache_hits == 1


def test_the_same_candidates_with_other_content_are_asked_about_again() -> None:
    """The selector reads labels, synonyms and definitions, not ids: a term whose label
    changed under the same id, in the same order, is another question."""
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose())
    candidates = _setup()[1]["e1"]
    (first,), _ = _select(LUNA, cache, llm=llm)
    assert first.pick.label == "aspirin"

    def edited(field: str, value: Any) -> list[Candidate]:
        return [
            dataclasses.replace(c, record=Record(c.id, {**c.record.fields, field: value}))
            if c.id == "CHEBI:15365"
            else c
            for c in candidates
        ]

    relabelled = edited("label", "acetylsalicylic acid")
    assert [c.id for c in relabelled] == [c.id for c in candidates]
    (again,), stats = _select(LUNA, cache, llm=llm, candidates={"e1": relabelled})
    assert stats.cache_hits == 0 and len(llm.requests) == 2
    assert again.pick.label == "acetylsalicylic acid"  # not the old label from memory
    for field, value in (("definition", "A new definition."), ("synonyms", ["ASA", "new"])):
        _, stats = _select(LUNA, cache, llm=llm, candidates={"e1": edited(field, value)})
        assert stats.cache_hits == 0, field
    assert len(llm.requests) == 4
    # Each content is remembered on its own, and the unchanged one still is.
    _, stats = _select(LUNA, cache, llm=llm, candidates={"e1": relabelled})
    assert stats.cache_hits == 1
    (same,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 1 and same.pick.label == "aspirin" and len(llm.requests) == 4
    assert caching.content_of(candidates) != caching.content_of(relabelled)


def test_a_run_on_a_visitors_own_key_neither_reads_nor_writes_a_pick() -> None:
    cache = Caches(EXPERIMENTAL).picks
    site = FakeLLM(handler=_choose())
    _select(LUNA, cache, llm=site)
    assert len(cache) == 1
    kept = [pick.model_dump() for pick in cache.values()]
    # The same question on an own key: asked of that key's provider, every time.
    own = FakeLLM(handler=_choose(explanation="said on the visitor's account"))
    for asked in (1, 2):
        (result,), stats = _select(LUNA, cache, llm=own, own_key=True)
        assert stats.cache_hits == 0 and len(own.requests) == asked
        assert result.pick.explanation == "said on the visitor's account"
    # A key the provider refuses is refused, with the answer in the cache.
    refused = FakeLLM([LLMFatalError("HTTP 401: invalid api key")])
    (result,), stats = _select(LUNA, cache, llm=refused, own_key=True)
    assert result.pick.status == "failed" and stats.cache_hits == 0 and len(refused.requests) == 1
    # Nothing an own-key run was told is kept, also in a cache that was empty.
    assert [pick.model_dump() for pick in cache.values()] == kept
    empty = Caches(EXPERIMENTAL).picks
    _select(LUNA, empty, llm=own, own_key=True)
    assert len(empty) == 0
    # And the site's next request is still answered with the site's own pick.
    (again,), stats = _select(LUNA, cache, llm=site)
    assert stats.cache_hits == 1 and again.pick.explanation == "x" and len(site.requests) == 1


def test_another_selector_or_another_type_is_asked_again() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose())
    _select(LUNA, cache, llm=llm)
    _, stats = _select("google/gemini-3.1-flash-lite", cache, llm=llm)
    assert stats.cache_hits == 0
    _, stats = _select(LUNA, cache, llm=llm, results=[_picked_entity(type="food")])
    assert stats.cache_hits == 0
    decider = FakeDecider(handler=_knows_aspirin)
    _, stats = _select(JEV, cache, decider=decider)
    assert stats.cache_hits == 0 and decider.calls
    assert len(llm.requests) == 3
    # Jev's own answer is remembered as Jev's.
    asked = len(decider.calls)
    (again,), stats = _select(JEV, cache, decider=decider)
    assert stats.cache_hits == 1 and len(decider.calls) == asked
    assert again.pick is not None and again.pick.selector == JEV


@pytest.mark.parametrize(
    "answer", ["not-json", '{"chosen_key": "FAKE"}', LLMFatalError("HTTP 500")]
)
def test_a_failed_pick_is_never_stored(answer: Any) -> None:
    cache = Caches(EXPERIMENTAL).picks
    (result,), stats = _select(LUNA, cache, llm=FakeLLM([answer]))
    assert result.pick is not None and result.pick.status == "failed" and len(cache) == 0
    llm = FakeLLM(handler=_choose())
    (result,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 0 and result.pick.status == "match" and len(cache) == 1


def test_a_pick_cut_off_by_the_deadline_is_never_stored() -> None:
    class Slow(FakeLLM):
        async def complete(self, request: LLMRequest) -> Any:
            await asyncio.sleep(5)

    cache = Caches(EXPERIMENTAL).picks
    (result,), _ = _select(LUNA, cache, llm=Slow([]), deadline=0.05)
    assert result.pick is not None and result.pick.status == "failed" and len(cache) == 0


def test_a_no_match_is_an_answer_and_is_stored() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(['{"chosen_key": null, "confidence_score": 0.8, "explanation": "none fit"}'])
    _select(LUNA, cache, llm=llm)
    (again,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 1 and again.pick.status == "no_match"


def test_a_pick_another_model_served_is_never_stored() -> None:
    class Client(FakeLLM):
        def view(self) -> FakeLLM:
            view = FakeLLM(handler=_choose())
            view.answered_by = "another-model"  # type: ignore[attr-defined]
            view.asked = "requested-model"  # type: ignore[attr-defined]
            view.served_by_fallback = True  # type: ignore[attr-defined]
            return view

    cache = Caches(EXPERIMENTAL).picks
    (result,), stats = _select("requested-model", cache, llm=Client([]))
    assert result.pick.selector == "another-model" and stats.served_elsewhere
    assert len(cache) == 0


def _claude_error(reason: str) -> Exception:
    error = LLMRetryableError(f"Anthropic said {reason}")
    error.reason = reason  # type: ignore[attr-defined]
    return error


def _chain(scripts: dict[str, Any], *, start: int = 0) -> tuple[Any, dict[str, FakeLLM]]:
    made: dict[str, FakeLLM] = {}

    def factory(settings: Settings, model: str, purpose: str) -> FakeLLM:
        script = scripts.get(model)
        made[model] = (
            FakeLLM(script, model=model)
            if script is not None
            else FakeLLM(handler=_choose(), model=model)
        )
        return made[model]

    settings = Settings(anthropic_api_key=ANTHROPIC_KEY)
    chain = service.make_chain(
        settings, OPUS, start=start, purposes=(service.SELECT,), claude_factory=factory
    )
    return chain, made


def test_a_pick_made_after_a_step_down_is_never_stored() -> None:
    """Opus is rate limited and Sonnet makes the pick: that is this run's answer, and
    no later request is handed it, whichever model it starts on."""
    cache = Caches(EXPERIMENTAL).picks
    chain, made = _chain({OPUS: [_claude_error("rate_limited")]})
    (stepped,), stats = _select(OPUS, cache, llm=chain)
    assert stepped.pick.selector == SONNET and stats.asked == {"e1": SONNET}
    assert len(cache) == 0

    # A request admitted on Sonnet (the Opus allowance is used) asks Sonnet itself,
    # and its pick, made below the selector that was asked for, is not stored either.
    chain, made = _chain({}, start=1)
    (lower,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 0 and lower.pick.selector == SONNET
    assert len(made[SONNET].requests) == 1 and len(cache) == 0

    # A clean run on Opus is stored, as Opus's, and found by the next clean run.
    chain, made = _chain({})
    (fresh,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 0 and fresh.pick.selector == OPUS and len(cache) == 1
    chain, made = _chain({})
    (again,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 1 and again.pick.selector == OPUS and not made
    # It is never handed to a request that would ask Sonnet now.
    chain, made = _chain({}, start=1)
    (other,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 0 and other.pick.selector == SONNET


def test_a_pick_is_not_stored_when_its_run_or_its_candidates_were_not_clean() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose())
    # The run had already stepped down or fallen back when the pick was made.
    asked: list[int] = []

    def stepped_down() -> bool:
        asked.append(1)
        return False

    (result,), _ = _select(LUNA, cache, llm=llm, storable=stepped_down)
    assert result.pick.status == "match" and asked == [1] and len(cache) == 0
    # The candidates came from degraded retrieval.
    (result,), _ = _select(LUNA, cache, llm=llm, degraded={"e1"})
    assert result.pick.status == "match" and len(cache) == 0
    # A secret is in the pick.
    (result,), _ = _select(LUNA, cache, llm=llm, sensitive=lambda text: "aspirin" in text)
    assert result.pick.status == "match" and len(cache) == 0
    assert len(llm.requests) == 3
    # None of the three: stored. And a pick is still read for degraded candidates or
    # in a stepped-down run (it is the answer the same question got from this model).
    _select(LUNA, cache, llm=llm, storable=lambda: True, degraded={"e2"}, sensitive=lambda t: False)
    assert len(cache) == 1
    _, stats = _select(LUNA, cache, llm=llm, storable=lambda: False, degraded={"e1"})
    assert stats.cache_hits == 1 and len(llm.requests) == 4


def test_what_a_request_does_to_a_pick_does_not_reach_the_cache() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose(explanation="as the model said"))
    (first,), _ = _select(LUNA, cache, llm=llm)
    first.pick.explanation = "changed by the first request"
    (second,), _ = _select(LUNA, cache, llm=llm)
    second.pick.explanation = "changed by the second request"
    (third,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 1 and third.pick.explanation == "as the model said"


def test_a_pick_is_stored_as_the_response_may_carry_it() -> None:
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose(explanation=f"the key is {OWN_KEY}"))
    scrub = lambda note: note.replace(OWN_KEY, "[redacted]")  # noqa: E731
    _select(LUNA, cache, llm=llm, clean=scrub)
    assert [pick.explanation for pick in cache.values()] == ["the key is [redacted]"]


# --- which responses may be stored ----------------------------------------------------


def _clean_response() -> dict[str, Any]:
    pick = {"selector": LUNA, "status": "match", "id": "CHEBI:1", "explanation": ""}

    def entity(eid: str, ontology: str | None, notes: list[str], picked: Any) -> dict[str, Any]:
        return {
            "id": eid,
            "text": "aspirin",
            "start": 0,
            "end": 7,
            "type": "chemical" if ontology else "other",
            "ontology": ontology,
            "notes": notes,
            "pick": picked,
            "matches": [],
        }

    return {
        "ok": True,
        "model": LUNA,
        "mode": "live",
        "entities": [
            entity("e1", "chebi", [], pick),
            entity("e2", "chebi", [], None),
            entity("e3", None, ["no ontology for type 'other'"], None),
        ],
        "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
        "warnings": [],
        "usage": {"llm_calls": 1, "embedding_requests": 1},
        "timing_ms": {"total": 9},
        "fallbacks": [],
        "notices": [],
    }


WHOLE = {"chebi": OntologyIndex("chebi", None, [], {"bm25": OK, "dense": OK})}  # type: ignore[arg-type]
NO_DENSE = RetrieverStatus(status="unavailable", reason="no index at indexes/chebi/dense")
# What offline settings open, and the same index when nothing says the mode did it.
DISABLED = indexes_mod.dense_disabled()
SWITCHED_OFF = {
    "chebi": OntologyIndex("chebi", None, [], {"bm25": OK, "dense": DISABLED}, dense_off=True)  # type: ignore[arg-type]
}
# The mark without the status: an index that reports some other reason.
MARKED_ONLY = {
    "chebi": OntologyIndex("chebi", None, [], {"bm25": OK, "dense": NO_DENSE}, dense_off=True)  # type: ignore[arg-type]
}
MISSING = {"chebi": OntologyIndex("chebi", None, [], {"bm25": OK, "dense": NO_DENSE})}  # type: ignore[arg-type]


def _may_store(data: dict[str, Any], indexes: Any = WHOLE, **settings: Any) -> bool:
    return cacheable(NerResponse.model_validate(data), indexes, settings=Settings(**settings))


def test_a_clean_response_may_be_stored() -> None:
    assert _may_store(_clean_response())
    no_match = _clean_response()
    no_match["entities"][0]["pick"] = {"selector": LUNA, "status": "no_match", "id": None}
    assert _may_store(no_match)


def _with(path: Sequence[Any], value: Any) -> dict[str, Any]:
    data = _clean_response()
    target: Any = data
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return data


STEP_DOWN = {"stage": "extract", "from": OPUS, "to": SONNET, "reason": "rate_limited"}


@pytest.mark.parametrize(
    "unclean",
    [
        pytest.param(_with(["fallbacks"], [STEP_DOWN]), id="a step-down"),
        pytest.param(_with(["notices"], ["Claude Opus 5.5 is rate limited."]), id="a notice"),
        pytest.param(
            _with(["warnings"], ["entity 'x' dropped: not found in text"]), id="a warning"
        ),
        pytest.param(
            _with(["entities", 0, "pick"], {"selector": LUNA, "status": "failed"}),
            id="a failed pick",
        ),
        pytest.param(
            _with(["retrieval", "dense"], {"status": "unavailable", "reason": "no key"}),
            id="degraded retrieval",
        ),
        pytest.param(
            _with(["retrieval", "bm25"], {"status": "unavailable", "reason": "gone"}),
            id="no keyword search",
        ),
        pytest.param(
            _with(["entities", 1, "notes"], ["dense-qwen3: timed out after 30s"]),
            id="a search that timed out",
        ),
        pytest.param(
            _with(["entities", 2, "notes"], ["ontology 'x' is not loaded"]), id="an odd note"
        ),
        pytest.param(_with(["credential_source"], "visitor"), id="a visitor's credential"),
        pytest.param(_with(["provider"], "openai"), id="a visitor's provider"),
    ],
)
def test_nothing_unclean_may_be_stored(unclean: dict[str, Any]) -> None:
    assert not _may_store(unclean)
    # Offline too, but for the one thing the stand-in never has (the next test).
    if unclean["retrieval"]["dense"]["status"] == "ok":
        assert not _may_store(unclean, SWITCHED_OFF, offline=True)


def test_the_offline_stand_in_may_be_stored_only_with_dense_switched_off_by_the_mode() -> None:
    """Offline there is never a dense search, by design, and every answer says so with
    one fixed warning; a fresh identical run would say the same. That state is explicit
    on the indexes (``dense_off``); the same response over indexes that merely lack
    their dense index, for whatever reason, is a degraded one, offline too."""
    offline = _clean_response()
    offline["retrieval"]["dense"] = DISABLED.model_dump(mode="json")
    offline["warnings"] = ["dense retrieval unavailable; matches use BM25 only"]
    assert "disabled in offline mode" in (DISABLED.reason or "")
    assert _may_store(offline, SWITCHED_OFF, offline=True)
    # Only that status passes: not the mark over another reason, and not a response
    # whose own dense status says something else.
    assert not _may_store(offline, MARKED_ONLY, offline=True)
    other = {**offline, "retrieval": {**offline["retrieval"], "dense": {"status": "unavailable"}}}
    assert not _may_store(other, SWITCHED_OFF, offline=True)
    other["retrieval"]["dense"]["reason"] = "no index at indexes/chebi/dense"
    assert not _may_store(other, SWITCHED_OFF, offline=True)
    assert not _may_store(offline, MISSING, offline=True)
    assert not _may_store(offline, MISSING) and not _may_store(offline, WHOLE)
    assert not _may_store(offline, {}, offline=True)
    # One index of several without the mark is enough to keep it out.
    assert not _may_store(offline, {**SWITCHED_OFF, "foodon": MISSING["chebi"]}, offline=True)
    offline["warnings"].append("entity 'x' dropped: not found in text")
    assert not _may_store(offline, SWITCHED_OFF, offline=True)


def test_cleanliness_is_judged_before_redaction_can_hide_a_field() -> None:
    """Redaction rewrites strings, dictionary keys among them: with a secret that
    happens to be a field's name, the redacted JSON has no ``warnings`` any more. The
    typed response still has them."""
    unclean = _with(["warnings"], ["entity 'x' dropped: not found in text"])
    secret = "warnings"
    redacted = http_mod._redact(unclean, (secret,))  # noqa: SLF001
    assert "warnings" not in redacted  # what the old check on the JSON was shown
    assert not _may_store(unclean, api_key=secret)
    assert not _may_store(_with(["fallbacks"], [STEP_DOWN]), api_key="fallbacks")


@pytest.mark.parametrize("passcode", ["ok", "x", "correct horse battery"])
def test_a_response_that_holds_the_passcode_is_not_stored(passcode: str) -> None:
    """A passcode shorter than eight characters is not scrubbed from a response (it
    would corrupt ordinary JSON), so a response it occurs in is simply not kept."""
    held = _with(["entities", 0, "text"], f"the code is {passcode}!")
    assert not _may_store(held, passcode=passcode)
    assert not _may_store(held, stats_passcode=passcode)
    assert _may_store(held, passcode="another-code") and _may_store(held)
    explained = _with(["entities", 0, "pick", "explanation"], f"she said {passcode}")
    assert not _may_store(explained, passcode=passcode)
    # The visitor's key, and any configured secret.
    assert not cacheable(
        NerResponse.model_validate(held), WHOLE, settings=Settings(), extra_secrets=(passcode,)
    )
    assert not _may_store(_with(["entities", 0, "text"], SITE_KEY), api_key=SITE_KEY)
    assert Settings(passcode="ok").unstorable(("visitor-key",)) == ("visitor-key", "ok")


# --- one request end to end (service.run) -------------------------------------------


def _offline(tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env(
        {
            "NER_SITE_OFFLINE": "1",
            "NER_SITE_CACHE_DIR": str(tmp_path / "cache"),
            "NER_SITE_DENSE_DIR": str(tmp_path / "no-dense"),
            **env,
        }
    )


# The two experimental caches with the sizes they had while they were on by default.
ALL_THREE = {"NER_SITE_CACHE_RETRIEVAL": "2048", "NER_SITE_CACHE_PICKS": "8192"}


def _offline3(tmp_path: Path, **env: str) -> Settings:
    return _offline(tmp_path, **{**ALL_THREE, **env})


def _run(settings: Settings, text: str, indexes: Any, caches: Any, **kw: Any) -> NerResponse:
    reused = kw.pop("reused", None)
    return asyncio.run(
        service.run(
            NerRequest(text=text, **kw),
            settings=settings,
            llm=fake.offline_llm(),
            indexes=indexes,
            decider=service.make_decider(settings),
            caches=caches,
            reused=reused,
        )
    )


CLEAN_RUN = {"scrubbed": 0, "cut_off": 0, "served_elsewhere": 0, "degraded": 0}


def test_a_run_reuses_what_an_earlier_text_shared_with_it(
    tmp_path: Path, offline_indexes: Any
) -> None:
    """With the two experimental caches given a size."""
    settings = _offline3(tmp_path)
    caches = Caches(settings.cache)
    reused: dict[str, int] = {}
    first = _run(settings, KETA_TEXT, offline_indexes, caches, reused=reused)
    # The response says that it ran and nothing else; the counts are for the log.
    assert first.cache == CacheInfo(response="miss")
    assert reused == {"retrieval_hits": 0, "pick_hits": 0, **CLEAN_RUN}
    assert [e.text for e in first.entities] == ["haloperidol", "ketamine"]
    assert all(e.pick is not None for e in first.entities)
    # Another text with one of the two mentions, in other surroundings: its candidates
    # are reused, its pick is not.
    shared = _run(settings, HALO_TEXT, offline_indexes, caches, reused=reused)
    assert shared.cache == CacheInfo(response="miss")
    assert reused == {"retrieval_hits": 1, "pick_hits": 0, **CLEAN_RUN}
    assert shared.entities[0].matches == first.entities[0].matches
    assert shared.usage.selector_calls > 0
    # The first text again: both mentions, in the same surroundings.
    again = _run(settings, KETA_TEXT, offline_indexes, caches, reused=reused)
    assert reused == {"retrieval_hits": 2, "pick_hits": 2, **CLEAN_RUN}
    assert again.usage.selector_calls == 0 and again.usage.selector_cost_usd is None
    assert again.entities == first.entities
    assert again.model_dump(mode="json")["cache"] == {"response": "miss"}


def test_a_run_over_degraded_indexes_keeps_neither_candidates_nor_picks(
    tmp_path: Path, bm25_only: Any
) -> None:
    """The same offline run on indexes nothing marked: dense is simply missing, so the
    candidates are degraded, and so is every pick made on them."""
    settings = _offline3(tmp_path)
    caches = Caches(settings.cache)
    reused: dict[str, int] = {}
    for _ in range(2):
        out = _run(settings, KETA_TEXT, bm25_only, caches, reused=reused)
        assert all(e.pick is not None and e.pick.status != "failed" for e in out.entities)
        assert reused == {"retrieval_hits": 0, "pick_hits": 0, **CLEAN_RUN, "degraded": 2}
        assert (len(caches.retrieval), len(caches.picks)) == (0, 0)
        assert not cacheable(out, bm25_only, settings=settings)
        assert not cacheable(out, bm25_only, settings=settings, run=reused)


def test_without_caches_a_response_has_no_cache_object(
    tmp_path: Path, offline_indexes: Any
) -> None:
    settings = _offline(tmp_path, NER_SITE_CACHE="off")
    response = _run(settings, HALO_TEXT, offline_indexes, None)
    assert response.cache is None and "cache" not in response.model_dump(mode="json")
    only_picks = Caches(CacheSettings(responses=0, retrieval=0, picks=5))
    response = _run(settings, HALO_TEXT, offline_indexes, only_picks)
    assert response.model_dump(mode="json")["cache"] == {"response": "off"}


# --- the handler: the response cache ---------------------------------------------------


def _live(tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env(
        {
            "OPENROUTER_API_KEY": SITE_KEY,
            "NER_SITE_CACHE_DIR": str(tmp_path / "cache"),
            "NER_SITE_DENSE_DIR": str(tmp_path / "no-dense"),
            "NER_SITE_SELECTOR": "retrieval",
            **env,
        }
    )


def _live3(tmp_path: Path, **env: str) -> Settings:
    return _live(tmp_path, **{**ALL_THREE, **env})


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, clean: Any) -> list[int]:
    """The handler's indexes are the clean ones; the list counts how often they were
    asked for."""
    asked: list[int] = []

    def get_indexes(settings: Any) -> Any:
        asked.append(1)
        return clean

    monkeypatch.setattr(http_mod, "get_indexes", get_indexes)
    return asked


def _post(
    settings: Settings,
    payload: dict[str, Any],
    *,
    llm: Any = None,
    headers: dict[str, Any] | None = None,
    claude: Any = None,
    raw: bytes | None = None,
) -> Handled:
    def never(settings: Settings, model: str) -> Any:
        pytest.fail("a model client was built for a request that should not need one")

    return http_mod.handle(
        "POST",
        "/api/ner",
        dict(HEADERS if headers is None else headers),
        raw if raw is not None else json.dumps(payload).encode("utf-8"),
        settings=settings,
        # Offline, the stand-in is the handler's own default.
        llm_factory=(lambda settings, model: llm)
        if llm is not None
        else (None if settings.offline else never),
        claude_factory=claude,
    )


def _json(result: Handled) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(result[2])
    return data


def _code(result: Handled) -> tuple[int, str]:
    return result[0], _json(result)["error"]["code"]


def _reader(answer: str = HALO) -> FakeLLM:
    return FakeLLM(handler=lambda request: answer)


def _caches(settings: Settings) -> Caches:
    found = caching.caches_for(settings.cache_scope, settings.cache)
    assert found is not None
    return found


def _log_of(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    return record


def test_an_identical_second_request_is_answered_from_the_cache(
    tmp_path: Path,
    opened: list[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    monkeypatch.setattr(caching, "_now", clock)
    s = _live(tmp_path)
    llm = _reader()
    first = _post(s, {"text": HALO_TEXT}, llm=llm)
    ran = _json(first)
    assert first[0] == 200 and ran["cache"] == {"response": "miss"}
    record = _log_of(capsys)
    assert record["cache"] == {
        "response": "miss",
        "retrieval_hits": 0,
        "pick_hits": 0,
        "stored": True,
    }
    assert len(llm.requests) == 1 and opened == [1]

    clock.now += 245
    second = _post(s, {"text": HALO_TEXT})  # no client may be built
    hit = _json(second)
    assert second[0] == 200 and second[1] == first[1]
    # No model. The index view is asked for (a warm instance has it remembered): the
    # request's key is made from it. Nothing is searched in it.
    assert len(llm.requests) == 1 and opened == [1, 1]
    # It says that it is a hit and nothing about the run that made the answer: not its
    # age, not what it took, not what it used.
    assert hit["cache"] == {"response": "hit"}
    told = json.dumps(hit)
    assert "245" not in told and "age_s" not in told and "original" not in told
    assert set(hit["cache"]) == {"response"}
    # This request's own work: no call, and the time of a lookup.
    assert hit["usage"] == {
        "llm_calls": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "embedding_requests": 0,
        "selector_calls": 0,
        "selector_prompt_tokens": None,
        "selector_completion_tokens": None,
        "selector_cost_usd": None,
    }
    assert set(hit["timing_ms"]) == {"total"} and hit["timing_ms"]["total"] < 1000
    assert ran["usage"]["llm_calls"] == 1 and ran["usage"]["prompt_tokens"] == 100
    # Everything else is the answer as it was.
    for name in ("cache", "usage", "timing_ms"):
        ran.pop(name)
        hit.pop(name)
    assert hit == ran and hit["entities"][0]["matches"][0]["label"] == "haloperidol"
    NerResponse.model_validate(_json(second))
    record = _log_of(capsys)
    assert record["status"] == 200 and record["cache"] == {"response": "hit", "age_s": 245}
    assert (record["llm_calls"], record["entities"], record["matches"]) == (0, 1, 1)
    assert record["prompt_tokens"] is None and set(record["timing_ms"]) == {"total", "handle"}
    assert "tier" not in record
    assert HALO_TEXT not in json.dumps(record) and "haloperidol" not in json.dumps(record)


@pytest.mark.parametrize(
    "other",
    [
        {"text": HALO_TEXT + " "},
        {"prompt": "Find the chemicals."},
        {"model": LUNA},
        {"selector": LUNA},
        {"top_k": 3},
    ],
)
def test_a_request_that_differs_in_any_part_of_the_key_runs(
    other: dict[str, Any], tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live(tmp_path)
    llm = _reader()
    base = {"text": HALO_TEXT, "top_k": 5}
    assert _json(_post(s, base, llm=llm))["cache"]["response"] == "miss"
    changed = _post(s, {**base, **other}, llm=llm)
    assert changed[0] == 200 and _json(changed)["cache"]["response"] == "miss"
    assert len(llm.requests) == 2
    # Each is then remembered on its own, and the defaults spelled out are the same key.
    assert _json(_post(s, {**base, **other}))["cache"]["response"] == "hit"
    spelled = {
        **base,
        "prompt": s.default_prompt,
        "model": s.default_model,
        "selector": "retrieval",
    }
    assert _json(_post(s, spelled))["cache"]["response"] == "hit"
    assert len(_caches(s).responses) == 2


def _own_clients(monkeypatch: pytest.MonkeyPatch, refused: Sequence[str] = ()) -> list[str]:
    """Own-key clients that answer like the site's, except for the keys in `refused`,
    which the provider turns away. The list names every client that was built."""
    from ner_site import providers

    built: list[str] = []

    def own_llm(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        built.append(f"{credential.provider}:{purpose}")
        if credential.token() in refused:
            return FakeLLM([LLMFatalError("HTTP 401: this key was revoked")], model=model)
        return FakeLLM(handler=lambda request: HALO if purpose == "extract" else PICK, model=model)

    monkeypatch.setattr(providers, "own_llm", own_llm)
    return built


OTHER_KEY = "another-visitor-key-000000000000"


def test_a_request_on_a_visitors_own_key_is_never_answered_from_the_cache(
    tmp_path: Path,
    opened: list[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The site has this very answer, and so has another visitor's run on the same
    provider: a request on an own key is still sent to that key's provider."""
    built = _own_clients(monkeypatch)
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live3(tmp_path, NER_SITE_OWN_PROVIDERS="openrouter,openai", NER_SITE_PASSCODE=PASSCODE)
    site = {**HEADERS, "x-passcode": PASSCODE}
    model = "openai/gpt-6-luna"
    asked = {"text": HALO_TEXT, "model": model, "selector": model}
    # The site-paid answer is stored, and found again by a site-paid request.
    assert _post(s, asked, llm=_reader(), headers=site)[0] == 200
    assert _json(_post(s, asked, headers=site))["cache"] == {"response": "hit"}
    caches = _caches(s)
    assert (len(caches.responses), len(caches.picks)) == (1, 1)
    kept = (list(caches.responses.values()), [p.model_dump() for p in caches.picks.values()])
    capsys.readouterr()

    # The same request on two visitors' keys, twice each: every one of them runs, on
    # that key, extraction and pick both.
    own_request = {**asked, "provider": "openrouter"}
    for round_, key in enumerate((OWN_KEY, OTHER_KEY, OWN_KEY, OTHER_KEY), start=1):
        result = _post(s, own_request, headers={**site, "authorization": "Bearer " + key})
        out = _json(result)
        assert result[0] == 200, result[2]
        assert out["cache"] == {"response": "off"} and out["credential_source"] == "visitor"
        assert out["usage"]["llm_calls"] == 1 and out["usage"]["selector_calls"] == 1
        assert built == ["openrouter:extract", "openrouter:select"] * round_
        record = _log_of(capsys)
        # Public ontology candidates are shared; the model's answers are not.
        assert record["cache"] == {
            "response": "off",
            "retrieval_hits": 1,
            "pick_hits": 0,
            "stored": False,
        }
    # And none of them left anything in the two caches.
    assert (
        list(caches.responses.values()),
        [p.model_dump() for p in caches.picks.values()],
    ) == kept
    # The site's own next request is still its hit, with nothing of a visitor's run.
    again = _json(_post(s, asked, headers=site))
    assert again["cache"] == {"response": "hit"} and again["credential_source"] == "site"


def test_own_key_runs_contribute_nothing_to_a_later_request(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty caches, then own-key runs: nothing a visitor paid for answers the site's
    next request or another visitor's."""
    built = _own_clients(monkeypatch)
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live3(tmp_path, NER_SITE_OWN_PROVIDERS="openrouter")
    model = "openai/gpt-6-luna"
    asked = {"text": HALO_TEXT, "model": model, "selector": model}
    own = {**HEADERS, "authorization": "Bearer " + OWN_KEY}
    for _ in range(2):
        assert _post(s, {**asked, "provider": "openrouter"}, headers=own)[0] == 200
    caches = _caches(s)
    assert (len(caches.responses), len(caches.picks), len(caches.retrieval)) == (0, 0, 1)
    assert len(built) == 4
    llm = _reader()
    ran = _json(_post(s, asked, llm=llm))
    assert ran["cache"] == {"response": "miss"} and ran["credential_source"] == "site"
    assert len(llm.requests) == 1 and len(picker.requests) == 1  # read and picked anew


def test_a_revoked_or_refused_key_is_refused_with_the_answer_in_the_cache(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _own_clients(monkeypatch, refused=(OTHER_KEY,))
    s = _live(tmp_path, NER_SITE_OWN_PROVIDERS="openrouter")
    model = "openai/gpt-6-luna"
    asked = {"text": HALO_TEXT, "model": model}
    # The site has the answer, and a visitor with a good key has been given it too.
    assert _post(s, asked, llm=_reader())[0] == 200
    assert _json(_post(s, asked))["cache"] == {"response": "hit"}
    good = {**HEADERS, "authorization": "Bearer " + OWN_KEY}
    assert _post(s, {**asked, "provider": "openrouter"}, headers=good)[0] == 200
    # The same request on a key the provider turns away: the provider is asked, and its
    # refusal is the answer. Every time.
    bad = {**HEADERS, "authorization": "Bearer " + OTHER_KEY}
    for attempt in (1, 2):
        refused = _post(s, {**asked, "provider": "openrouter"}, headers=bad)
        body = _json(refused)
        assert refused[0] >= 400 and body["ok"] is False, refused[2]
        assert "entities" not in body and "haloperidol" not in refused[2].decode()
        assert len(built) == 1 + attempt
    # A key that is not even well formed never reaches a provider, or the cache.
    short = {**HEADERS, "authorization": "Bearer short"}
    assert _code(_post(s, {**asked, "provider": "openrouter"}, headers=short)) == (
        400,
        "bad_credential",
    )
    assert len(_caches(s).responses) == 1


def test_a_response_expires_and_the_request_runs_again(
    tmp_path: Path,
    opened: list[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = Clock()
    monkeypatch.setattr(caching, "_now", clock)
    s = _live3(tmp_path, NER_SITE_CACHE_TTL_S="600")
    llm = _reader()
    _post(s, {"text": HALO_TEXT}, llm=llm)
    clock.now += 599
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"] == {"response": "hit"}
    assert _log_of(capsys)["cache"] == {"response": "hit", "age_s": 599}
    clock.now += 1
    late = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
    # The whole answer has expired, and with it what was retrieved for it.
    assert late["cache"] == {"response": "miss"}
    assert _log_of(capsys)["cache"]["retrieval_hits"] == 0
    assert len(llm.requests) == 2
    # Expired entries were removed, not merely passed over.
    caches = _caches(s)
    assert (len(caches.responses), len(caches.retrieval)) == (1, 1)


def test_the_response_cache_is_bounded(tmp_path: Path, opened: list[int]) -> None:
    s = _live(tmp_path, NER_SITE_CACHE_RESPONSES="2")
    llm = _reader('{"entities": []}')
    for text in ("one", "two", "three"):
        _post(s, {"text": text}, llm=llm)
    assert len(_caches(s).responses) == 2
    assert _json(_post(s, {"text": "three"}))["cache"]["response"] == "hit"
    assert _json(_post(s, {"text": "one"}, llm=llm))["cache"]["response"] == "miss"


def test_a_hit_is_a_fresh_copy_every_time(tmp_path: Path, opened: list[int]) -> None:
    s = _live(tmp_path)
    ran = _json(_post(s, {"text": HALO_TEXT}, llm=_reader()))
    stored = list(_caches(s).responses.values())
    first = _json(_post(s, {"text": HALO_TEXT}))
    second = _json(_post(s, {"text": HALO_TEXT}))
    # What a hit puts in place of the run's figures is not written back: what is
    # stored is still the run's answer, and every hit is made from it anew.
    for hit in (first, second):
        hit["timing_ms"].pop("total")
    assert first == second and first["cache"] == {"response": "hit"}
    assert list(_caches(s).responses.values()) == stored
    kept = json.loads(stored[0])
    assert kept["cache"] == {"response": "miss"} and kept["usage"] == ran["usage"]


def test_two_identical_requests_at_once_both_run_and_leave_one_entry(
    tmp_path: Path, opened: list[int]
) -> None:
    """In-flight work is not shared: both miss, both run, both store the same answer."""
    s = _live3(tmp_path)
    together = threading.Barrier(2)

    def answer(request: LLMRequest) -> str:
        together.wait(timeout=10)  # both requests are past the lookup before either answers
        return HALO

    results: list[Handled] = []

    def ask() -> None:
        results.append(_post(s, {"text": HALO_TEXT}, llm=FakeLLM(handler=answer)))

    threads = [threading.Thread(target=ask) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert [r[0] for r in results] == [200, 200]
    assert [_json(r)["cache"]["response"] for r in results] == ["miss", "miss"]
    caches = _caches(s)
    assert len(caches.responses) == 1 and len(caches.retrieval) == 1
    hit = _json(_post(s, {"text": HALO_TEXT}))
    assert hit["cache"]["response"] == "hit"
    assert hit["entities"] == _json(results[0])["entities"] == _json(results[1])["entities"]


# --- the handler: what is never stored -----------------------------------------------


def _runs_again(settings: Settings, payload: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """The same request twice; the second one's answer, which must be a run."""
    first = _post(settings, payload, **kw)
    assert first[0] == 200, first[2]
    second = _json(_post(settings, payload, **kw))
    assert second["cache"]["response"] == "miss"
    assert len(_caches(settings).responses) == 0
    return second


def test_a_response_with_a_warning_is_not_stored(tmp_path: Path, opened: list[int]) -> None:
    llm = _reader(_answer(("haloperidol", "chemical", 9, 20), ("unobtainium", "chemical", 0, 3)))
    out = _runs_again(_live(tmp_path), {"text": HALO_TEXT}, llm=llm)
    assert out["warnings"] and len(llm.requests) == 2


def test_a_response_with_degraded_retrieval_is_not_stored(
    tmp_path: Path,
    bm25_only: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: bm25_only)
    llm = _reader()
    s = _live3(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT}, llm=llm)
    assert out["retrieval"]["dense"]["status"] == "unavailable" and len(llm.requests) == 2
    # Nor are its BM25-only candidates: dense is configured for this ontology and was
    # not there, so the second run searched again.
    record = _log_of(capsys)
    assert record["cache"]["retrieval_hits"] == 0 and record["cache"]["stored"] is False
    assert len(_caches(s).retrieval) == 0


def test_a_response_with_a_failed_search_is_not_stored(
    tmp_path: Path, clean: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    chebi = clean["chebi-sample"]
    broken = {
        **clean,
        "chebi-sample": dataclasses.replace(chebi, retrievers=[*chebi.retrievers, Broken()]),
    }
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: broken)
    s = _live3(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT}, llm=_reader())
    assert out["entities"][0]["notes"] and out["cache"] == {"response": "miss"}
    assert len(_caches(s).retrieval) == 0


def test_picks_made_on_degraded_candidates_are_not_stored(
    tmp_path: Path, bm25_only: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: bm25_only)
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live3(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT, "selector": LUNA}, llm=_reader())
    assert out["entities"][0]["pick"]["status"] == "match"
    assert len(picker.requests) == 2 and len(_caches(s).picks) == 0


def test_offline_too_only_dense_switched_off_by_the_mode_is_stored(
    tmp_path: Path, bm25_only: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offline exception is the explicit state, not "any reason at all": indexes
    that lack dense retrieval without the mode having switched it off are degraded
    offline as they are live."""
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: bm25_only)
    s = _offline3(tmp_path)
    for _ in range(2):
        out = _json(_post(s, {"text": KETA_TEXT}))
        assert out["cache"] == {"response": "miss"}
    caches = _caches(s)
    assert (len(caches.responses), len(caches.retrieval), len(caches.picks)) == (0, 0, 0)


def test_a_response_with_a_failed_pick_is_not_stored(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    picker = FakeLLM(handler=lambda request: "not json at all")
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live3(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT, "selector": LUNA}, llm=_reader())
    assert out["entities"][0]["pick"]["status"] == "failed"
    assert len(picker.requests) == 2 and len(_caches(s).picks) == 0


def test_an_error_is_not_stored(tmp_path: Path, opened: list[int]) -> None:
    s = _live(tmp_path)
    failed = _post(s, {"text": HALO_TEXT}, llm=FakeLLM([LLMFatalError("HTTP 500")]))
    assert _code(failed) == (502, "provider_error") and len(_caches(s).responses) == 0
    llm = _reader()
    assert _json(_post(s, {"text": HALO_TEXT}, llm=llm))["cache"]["response"] == "miss"
    assert len(llm.requests) == 1


# --- the handler: admission, refusals, the allowance ---------------------------------


@pytest.fixture
def visits(monkeypatch: pytest.MonkeyPatch) -> Any:
    from tests.test_site_limits import FakeVisits

    table = FakeVisits()
    monkeypatch.setattr(http_mod, "make_limits_executor", lambda database_url: table)
    return table


class _Claude:
    """Scripted Claude clients, one per model and purpose; counts what was asked."""

    def __init__(self, extracted: str = HALO, **scripts: Any) -> None:
        self.extracted = extracted
        self.scripts = scripts
        self.requests = 0

    def __call__(self, settings: Settings, model: str, purpose: str) -> FakeLLM:
        tier = "opus" if model == OPUS else "sonnet"
        script = self.scripts.get(f"{tier}_{purpose}")

        def answer(request: LLMRequest) -> str:
            self.requests += 1
            return self.extracted if purpose == "extract" else PICK

        if script is not None:
            return FakeLLM(script, model=model)
        return FakeLLM(handler=answer, model=model)


def _open_site(tmp_path: Path, **env: str) -> Settings:
    return _live(
        tmp_path,
        **{
            "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
            "DATABASE_URL": DB_URL,
            "NER_SITE_SELECTOR": "retrieval",
            **env,
        },
    )


def _counted_visits(table: Any) -> int:
    return sum(table.rows.values())


def test_a_hit_is_served_before_admission_and_is_not_counted(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    s = _open_site(tmp_path, NER_SITE_OPUS_PER_HOUR="1", NER_SITE_SONNET_PER_HOUR="0")
    claude = _Claude()
    first = _post(s, {"text": HALO_TEXT}, claude=claude)
    assert first[0] == 200 and _json(first)["model"] == OPUS and _counted_visits(visits) == 1
    # The hour's one run is used: another text is refused, the same text is answered.
    assert _code(_post(s, {"text": "Another text."}, claude=claude)) == (
        429,
        "visitor_limit_reached",
    )
    asked = len(visits.sent)
    for _ in range(3):
        again = _post(s, {"text": HALO_TEXT}, claude=claude)
        assert again[0] == 200 and _json(again)["cache"]["response"] == "hit"
    assert claude.requests == 1 and _counted_visits(visits) == 1
    assert len(visits.sent) == asked  # the limiter was not asked about a hit at all


def test_a_hit_is_served_while_the_limiter_cannot_be_reached(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    s = _open_site(tmp_path)
    assert _post(s, {"text": HALO_TEXT}, claude=_Claude())[0] == 200
    visits.fail = RuntimeError("database unavailable")
    assert _code(_post(s, {"text": "Another text."}, claude=_Claude())) == (
        503,
        "limits_unavailable",
    )
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"]["response"] == "hit"


def test_a_hit_is_not_served_while_the_operators_claude_is_blocked(
    tmp_path: Path, opened: list[int], visits: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """An Opus answer is stored; then this instance learns that the operator's Claude
    key is refused (or its balance used). A fresh identical request would be routed to
    the fallback model, so the stored Opus answer is not handed out either."""
    s = _open_site(tmp_path)
    claude = _Claude()
    body = {"text": HALO_TEXT}
    assert _json(_post(s, body, claude=claude))["model"] == OPUS
    assert _json(_post(s, body))["cache"] == {"response": "hit"}
    # An answer that needed no Claude, stored too.
    other = {"text": HALO_TEXT, "model": LUNA}
    assert _post(s, other, llm=_reader())[0] == 200
    capsys.readouterr()

    for reason in ("key_refused", "balance_exhausted"):
        fallback.block_claude(reason)
        luna = FakeLLM(handler=lambda request: HALO, model=s.fallback_model)
        result = _post(s, body, llm=luna, claude=claude)
        out = _json(result)
        assert result[0] == 200 and out["cache"] == {"response": "miss"}, reason
        assert out["model"] == s.fallback_model and out["requested_model"] == OPUS
        assert out["fallbacks"] and out["notices"] and len(luna.requests) == 1
        record = _log_of(capsys)
        assert record["cache_skipped"] == "claude_blocked" and record["cache"]["stored"] is False
        # The block concerns Claude: the other model's stored answer is still served.
        assert _json(_post(s, other))["cache"] == {"response": "hit"}
        assert "cache_skipped" not in _log_of(capsys)
    assert claude.requests == 1
    # The fallback's answer did not replace the stored one: with Claude back, the hit.
    fallback.reset()
    again = _json(_post(s, body))
    assert again["cache"] == {"response": "hit"} and again["model"] == OPUS


def test_a_blocked_selector_keeps_a_stored_answer_back_too(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    s = _open_site(tmp_path)
    body = {"text": HALO_TEXT, "model": LUNA, "selector": OPUS}
    assert _post(s, body, llm=_reader(), claude=_Claude())[0] == 200
    assert _json(_post(s, body))["cache"] == {"response": "hit"}
    fallback.block_claude("key_refused")
    picker = FakeLLM(handler=lambda request: PICK)
    import ner_site.http as h

    original = h.make_select_llm
    h.make_select_llm = lambda settings, model: picker  # type: ignore[assignment]
    try:
        out = _json(_post(s, body, llm=_reader(), claude=_Claude()))
    finally:
        h.make_select_llm = original  # type: ignore[assignment]
    assert out["cache"] == {"response": "miss"} and out["fallbacks"]


def test_a_pick_made_after_a_step_down_is_not_stored_by_a_request(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    s = _open_site(tmp_path, **ALL_THREE)
    claude = _Claude(opus_select=[_claude_error("rate_limited")])
    out = _json(_post(s, {"text": HALO_TEXT, "selector": OPUS}, claude=claude))
    assert out["model"] == OPUS and out["entities"][0]["pick"]["selector"] == SONNET
    assert out["fallbacks"]
    caches = _caches(s)
    assert (len(caches.responses), len(caches.picks), len(caches.retrieval)) == (0, 0, 1)
    # A run that stepped down before the selection (the reader did): its picks, made
    # by the model it stepped down to, are not stored either.
    claude = _Claude(opus_extract=[_claude_error("rate_limited")])
    out = _json(_post(s, {"text": HALO_TEXT, "selector": OPUS}, claude=claude))
    assert out["model"] == SONNET and out["entities"][0]["pick"]["status"] == "match"
    assert (len(caches.responses), len(caches.picks)) == (0, 0)
    # A clean run's pick is.
    out = _json(_post(s, {"text": HALO_TEXT, "selector": OPUS}, claude=_Claude()))
    assert out["entities"][0]["pick"]["selector"] == OPUS and not out["fallbacks"]
    assert (len(caches.responses), len(caches.picks)) == (1, 1)


def test_a_stepped_down_response_is_not_stored(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    s = _open_site(tmp_path)
    claude = _Claude(opus_extract=[_claude_error("rate_limited")])
    out = _runs_again(s, {"text": HALO_TEXT}, claude=claude)
    assert out["model"] == SONNET and out["fallbacks"] and out["notices"]
    assert _counted_visits(visits) == 2  # both ran, and both were counted


def test_everything_that_refuses_a_request_still_refuses_it_first(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    """The answer is in the cache, and a request that is malformed or not authorised
    does not get it."""
    s = _open_site(tmp_path, NER_SITE_PASSCODE=PASSCODE, NER_SITE_OWN_PROVIDERS="openai")
    good = {**HEADERS, "x-passcode": PASSCODE}
    body = {"text": HALO_TEXT}
    assert _post(s, body, headers=good, claude=_Claude())[0] == 200
    assert _json(_post(s, body, headers=good))["cache"]["response"] == "hit"
    key = {"authorization": "Bearer " + OWN_KEY}
    padded = json.dumps({**body, "prompt": None}).encode() + b" " * 70000
    refused = [
        ({k: v for k, v in good.items() if k != "x-requested-with"}, body, None, "bad_header"),
        (HEADERS, body, None, "passcode_required"),
        ({**good, "x-passcode": "wrong"}, body, None, "passcode_invalid"),
        (good, body, padded, "body_too_large"),
        (good, body, b'{"text": "She took haloperidol for a fever."', "bad_json"),
        (good, {**body, "extra": 1}, None, "bad_json"),
        (good, {**body, "top_k": 99}, None, "bad_top_k"),
        (good, {**body, "model": "no/such-model"}, None, "unknown_model"),
        (good, {**body, "selector": "no/such-selector"}, None, "unknown_selector"),
        # A key sent with a request for the site's models, and a malformed key.
        ({**good, **key}, body, None, "bad_credential"),
        (
            {**good, "authorization": "Bearer short"},
            {**body, "provider": "openai"},
            None,
            "bad_credential",
        ),
        # A provider whose own keys are off, and an own-key request without a key.
        ({**good, **key}, {**body, "provider": "anthropic"}, None, "bad_credential"),
        (good, {**body, "provider": "openai"}, None, "bad_credential"),
    ]
    for headers, payload, raw, code in refused:
        result = _post(s, payload, headers=headers, raw=raw)
        assert result[0] in (400, 401, 413) and _code(result)[1] == code, (code, result[2])
    assert _counted_visits(visits) == 1
    # And with own keys off altogether.
    closed = _open_site(tmp_path, NER_SITE_PASSCODE=PASSCODE)
    assert _post(closed, body, headers=good, claude=_Claude())[0] == 200
    assert _code(_post(closed, {**body, "provider": "openai"}, headers={**good, **key})) == (
        400,
        "bad_credential",
    )


def test_a_site_that_is_not_configured_answers_no_hit(tmp_path: Path, opened: list[int]) -> None:
    s = Settings.from_env({"NER_SITE_CACHE_DIR": str(tmp_path / "c")})
    assert _code(_post(s, {"text": HALO_TEXT})) == (503, "not_configured")


# --- secrets ---------------------------------------------------------------------------


def test_no_key_passcode_or_address_is_in_any_cached_key_or_value(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model echoes the visitor's key wherever it can; none of it is kept."""
    from ner_site import providers

    escaped = json.dumps(OWN_KEY)[1:-1]

    def own_llm(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        token = credential.token()
        pick = json.dumps(
            {"chosen_key": "C01", "confidence_score": 0.7, "explanation": f"key {token} seen"}
        )
        return FakeLLM(handler=lambda request: HALO if purpose == "extract" else pick, model=model)

    monkeypatch.setattr(providers, "own_llm", own_llm)
    s = _live3(tmp_path, NER_SITE_OWN_PROVIDERS="openai", NER_SITE_PASSCODE=PASSCODE)
    headers = {
        **HEADERS,
        "x-passcode": PASSCODE,
        "x-forwarded-for": f"{RAW_IP}, 10.0.0.1",
        "authorization": "Bearer " + OWN_KEY,
    }
    payload = {"text": HALO_TEXT, "provider": "openai", "selector": "gpt-6-luna"}
    first = _post(s, payload, headers=headers)
    assert first[0] == 200, first[2]
    out = _json(first)
    assert out["entities"][0]["pick"]["explanation"] == "key [redacted] seen"
    caches = _caches(s)
    # An own-key run leaves the public candidates and nothing a model said.
    assert (len(caches.responses), len(caches.retrieval), len(caches.picks)) == (0, 1, 0)

    def kept() -> str:
        return "\n".join(
            [
                *(str(v) for v in caches.responses.values()),
                *(repr(v) for v in caches.retrieval.values()),
                *(repr(v) + v.model_dump_json() for v in caches.picks.values()),
                *(
                    str(k)
                    for part in (caches.responses, caches.retrieval, caches.picks)
                    for k in part.digests()
                ),
                repr(list(caching._CACHES)),  # noqa: SLF001 - the names of the caches too
            ]
        )

    for secret in (OWN_KEY, escaped, "visitor-sentinel", PASSCODE, RAW_IP, SITE_KEY):
        assert secret not in kept(), secret
    # A site-paid run whose selector echoes the operator's key and the passcode: the
    # response goes out scrubbed, and neither it nor the pick is kept.
    echo = json.dumps(
        {
            "chosen_key": "C01",
            "confidence_score": 0.7,
            "explanation": f"key {SITE_KEY} and code {PASSCODE} seen",
        }
    )
    picker = FakeLLM(handler=lambda request: echo)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    site = {k: v for k, v in headers.items() if k != "authorization"}
    asked = {"text": HALO_TEXT, "selector": LUNA}
    for calls in (1, 2):
        result = _post(s, asked, headers=site, llm=_reader())
        told = _json(result)["entities"][0]["pick"]["explanation"]
        assert told == "key [redacted] and code [redacted] seen" and len(picker.requests) == calls
        assert _json(result)["cache"] == {"response": "miss"}
    assert (len(caches.responses), len(caches.picks)) == (0, 0)
    for secret in (OWN_KEY, escaped, "visitor-sentinel", PASSCODE, RAW_IP, SITE_KEY):
        assert secret not in kept(), secret
    # A clean site-paid answer is kept, under digests.
    assert _post(s, {"text": HALO_TEXT}, headers=site, llm=_reader())[0] == 200
    assert len(caches.responses) == 1
    assert all(len(str(k)) == 64 for k in caches.responses.digests())
    assert HALO_TEXT not in "".join(str(k) for k in caches.responses.digests())
    for secret in (OWN_KEY, escaped, "visitor-sentinel", PASSCODE, RAW_IP, SITE_KEY):
        assert secret not in kept(), secret


@pytest.mark.parametrize("passcode", ["halo", "haloperidol"])
def test_nothing_that_holds_the_passcode_is_stored_whatever_its_length(
    passcode: str, tmp_path: Path, opened: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    """A passcode under eight characters is not scrubbed from a response. Here it is
    part of a mention: the answer is served, and nothing that holds it is kept, not
    the response, not the candidates found for the mention, not the pick."""
    s = _live3(tmp_path, NER_SITE_PASSCODE=passcode)
    headers = {**HEADERS, "x-passcode": passcode}
    picker = FakeLLM(handler=lambda request: PICK)
    import ner_site.http as h

    original = h.make_select_llm
    h.make_select_llm = lambda settings, model: picker  # type: ignore[assignment]
    try:
        for calls in (1, 2):
            llm = _reader()
            result = _post(s, {"text": HALO_TEXT, "selector": LUNA}, headers=headers, llm=llm)
            assert result[0] == 200 and _json(result)["cache"] == {"response": "miss"}
            assert len(llm.requests) == 1 and len(picker.requests) == calls
            record = _log_of(capsys)
            assert record["cache"] == {
                "response": "miss",
                "retrieval_hits": 0,
                "pick_hits": 0,
                "stored": False,
            }
    finally:
        h.make_select_llm = original  # type: ignore[assignment]
    caches = _caches(s)
    assert (len(caches.responses), len(caches.retrieval), len(caches.picks)) == (0, 0, 0)
    # With another passcode the same request is stored, all three.
    other = _live3(tmp_path, NER_SITE_PASSCODE="zz")
    h.make_select_llm = lambda settings, model: picker  # type: ignore[assignment]
    try:
        _post(
            other,
            {"text": HALO_TEXT, "selector": LUNA},
            headers={**HEADERS, "x-passcode": "zz"},
            llm=_reader(),
        )
    finally:
        h.make_select_llm = original  # type: ignore[assignment]
    caches = _caches(other)
    assert (len(caches.responses), len(caches.retrieval), len(caches.picks)) == (1, 1, 1)


# --- the index view, and how much is kept ---------------------------------------------


class Reloaded(Counting):
    """The same retriever over an index that was loaded anew: another fingerprint."""

    @property
    def fingerprint(self) -> str:
        return super().fingerprint + "-reloaded"


def test_an_answer_belongs_to_the_index_view_it_was_made_on(
    tmp_path: Path, clean: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = {"now": clean}
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: view["now"])
    s = _live(tmp_path)
    llm = _reader()
    assert _json(_post(s, {"text": HALO_TEXT}, llm=llm))["cache"] == {"response": "miss"}
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"] == {"response": "hit"}
    before = caching.index_identity(clean)
    # The catalog is loaded anew and this instance opens it.
    reloaded = {
        slug: dataclasses.replace(index, retrievers=[Reloaded(r) for r in index.retrievers])
        for slug, index in clean.items()
    }
    assert caching.index_identity(reloaded) != before
    view["now"] = reloaded
    # The answer made on the old view is not handed out for the new one, and no other
    # request has to run first for that: the key is made from the view this very
    # request opened.
    again = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
    assert again["cache"] == {"response": "miss"} and len(llm.requests) == 2
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"] == {"response": "hit"}
    # Back on the old view, its own answer is still there.
    view["now"] = clean
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"] == {"response": "hit"}
    assert len(llm.requests) == 2 and not hasattr(_caches(s), "index_identity")
    # A retriever that went away, a status that changed, or dense retrieval switched
    # off by the mode is another view too.
    fewer = {slug: dataclasses.replace(index, retrievers=[]) for slug, index in clean.items()}
    unwell = {
        slug: dataclasses.replace(
            index, status={**index.status, "dense": RetrieverStatus(status="unavailable")}
        )
        for slug, index in clean.items()
    }
    off = {slug: dataclasses.replace(index, dense_off=True) for slug, index in unwell.items()}
    assert len({caching.index_identity(v) for v in (clean, reloaded, fewer, unwell, off)}) == 5


def test_a_response_over_the_size_limit_is_not_stored(
    tmp_path: Path,
    opened: list[int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert caching.MAX_RESPONSE_BYTES == 256 * 1024
    s = _live(tmp_path)
    small = _post(s, {"text": "Nothing here."}, llm=_reader('{"entities": []}'))
    larger = _post(s, {"text": HALO_TEXT}, llm=_reader())
    assert len(small[2]) < len(larger[2])
    caching.reset()
    capsys.readouterr()
    # The limit between the two: the small answer is kept, the larger one is not.
    monkeypatch.setattr(caching, "MAX_RESPONSE_BYTES", (len(small[2]) + len(larger[2])) // 2)
    llm = _reader()
    for calls in (1, 2):
        out = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
        assert out["cache"] == {"response": "miss"} and len(llm.requests) == calls
        assert _log_of(capsys)["cache"]["stored"] is False
    _post(s, {"text": "Nothing here."}, llm=_reader('{"entities": []}'))
    assert _log_of(capsys)["cache"]["stored"] is True
    assert _json(_post(s, {"text": "Nothing here."}))["cache"] == {"response": "hit"}
    assert len(_caches(s).responses) == 1


# --- the caches turned off ------------------------------------------------------------


def test_with_the_caches_off_a_request_runs_as_before(tmp_path: Path, opened: list[int]) -> None:
    off = _live(tmp_path, NER_SITE_CACHE="off")
    on = _live(tmp_path)
    llm = _reader()
    answers = [_json(_post(off, {"text": HALO_TEXT}, llm=llm)) for _ in range(2)]
    assert len(llm.requests) == 2 and opened == [1, 1]
    assert all("cache" not in answer for answer in answers)
    cached = _json(_post(on, {"text": HALO_TEXT}, llm=llm))
    # Apart from the cache object the response is the one a cached deployment gives.
    assert cached.pop("cache")["response"] == "miss"
    for answer in (*answers, cached):
        answer.pop("timing_ms")
    assert answers[0] == answers[1] == cached
    assert caching.caches_for(off.cache_scope, off.cache) is None


def test_the_response_cache_alone_can_be_off(
    tmp_path: Path, opened: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    s = _live3(tmp_path, NER_SITE_CACHE_RESPONSES="0")
    llm = _reader()
    _post(s, {"text": HALO_TEXT}, llm=llm)
    again = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
    assert again["cache"] == {"response": "off"} and len(llm.requests) == 2
    record = _log_of(capsys)
    assert record["cache"] == {
        "response": "off",
        "retrieval_hits": 1,
        "pick_hits": 0,
        "stored": False,
    }


def test_the_offline_stand_in_is_cached_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through the handler's own indexes: offline settings open them with dense
    switched off by the mode, which is what lets the stand-in be tried."""
    s = _offline(tmp_path)
    first = _json(_post(s, {"text": KETA_TEXT}))
    assert first["cache"] == {"response": "miss"} and first["warnings"]
    assert _json(_post(s, {"text": KETA_TEXT}))["cache"] == {"response": "hit"}
    shared = _json(_post(s, {"text": HALO_TEXT}))
    assert shared["cache"] == {"response": "miss"}
    record = _log_of(capsys)
    # Another text sharing a mention: nothing of the earlier run was reused.
    assert record["cache"] == {
        "response": "miss",
        "retrieval_hits": 0,
        "pick_hits": 0,
        "stored": True,
    }
    assert shared["retrieval"]["dense"] == indexes_mod.dense_disabled().model_dump(mode="json")


# --- the record of runs ---------------------------------------------------------------


def test_the_row_of_a_hit_says_that_it_cost_nothing() -> None:
    log = {
        "status": 200,
        "model": OPUS,
        "selector": OPUS,
        "provider": "site",
        "credential": "site",
        "text_chars": 33,
        "custom_prompt": False,
        "entities": 2,
        "matches": 10,
        "picks": 2,
        "llm_calls": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "embedding_requests": 0,
        "selector_calls": 0,
        "selector_cost_usd": None,
        "timing_ms": {"total": 1, "handle": 2},
        "cache": {"response": "hit", "age_s": 30},
    }
    row = usage.run_of(log, at=1795011600.0, visitor_day="v")
    assert (row.status, row.requested_model, row.requested_selector) == (200, OPUS, OPUS)
    assert row.model is None and row.tier is None
    assert (row.llm_calls, row.selector_calls, row.embedding_requests) == (0, 0, 0)
    assert row.prompt_tokens is None and row.selector_prompt_tokens is None
    assert (row.extract_ms, row.map_ms, row.select_ms, row.total_ms) == (None, None, None, None)
    assert (row.entities, row.matches, row.picks) == (2, 10, 2)
    # A run that missed is recorded as before.
    ran = usage.run_of(
        {**log, "llm_calls": 1, "timing_ms": {"total": 900}, "cache": {"response": "miss"}},
        at=1795011600.0,
        visitor_day="v",
    )
    assert ran.model == OPUS and ran.total_ms == 900


def test_a_hit_is_recorded_without_adding_to_the_spend(
    tmp_path: Path, opened: list[int], visits: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_site_usage import FakeRuns

    table = FakeRuns()
    monkeypatch.setattr(http_mod, "make_usage_executor", lambda database_url, timeout: table)
    monkeypatch.setattr(http_mod, "_now", lambda: 1795011600.0)
    passcode = "stats-SENTINEL-passcode-7781"
    s = _open_site(tmp_path, NER_SITE_STATS_PASSCODE=passcode)

    def stats() -> dict[str, Any]:
        result = http_mod.handle(
            "GET", "/api/stats?days=1", {"x-stats-passcode": passcode}, b"", settings=s
        )
        assert result[0] == 200, result[2]
        return _json(result)

    assert _post(s, {"text": HALO_TEXT}, claude=_Claude())[0] == 200
    before = stats()
    for _ in range(3):
        assert _json(_post(s, {"text": HALO_TEXT}))["cache"]["response"] == "hit"
    after = stats()
    assert len(table.rows) == 4
    hit = table.rows[-1]
    assert (hit["status"], hit["model"], hit["requested_model"]) == (200, None, OPUS)
    assert (hit["llm_calls"], hit["prompt_tokens"], hit["total_ms"], hit["tier"]) == (
        0,
        None,
        None,
        None,
    )
    # Counted as requests of the same visitor; no dollars, no tokens, no unpriced run,
    # and the run time is still the one run's.
    assert (before["totals"]["runs"], after["totals"]["runs"]) == (1, 4)
    assert after["totals"]["ok"] == 4 and after["totals"]["visitors"] == 1
    for name in ("estimated_usd", "prompt_tokens", "completion_tokens", "unpriced_runs"):
        assert after["totals"][name] == before["totals"][name], name
    assert before["totals"]["estimated_usd"] > 0
    assert after["by_day"][-1]["p50_ms"] == before["by_day"][-1]["p50_ms"]
    assert after["by_day"][-1]["opus"] == 1
    by_model = {row["model"]: row for row in after["by_model"]}
    assert by_model[None]["runs"] == 3 and by_model[None]["estimated_usd"] == 0
    assert by_model[OPUS] == {row["model"]: row for row in before["by_model"]}[OPUS]


# --- the wire contract ----------------------------------------------------------------


def test_the_cache_object_says_hit_or_miss_and_nothing_else() -> None:
    for word in ("hit", "miss", "off"):
        assert CacheInfo(response=word).model_dump(mode="json") == {"response": word}  # type: ignore[arg-type]
    assert set(CacheInfo.model_fields) == {"response"}
    with pytest.raises(ValueError):
        CacheInfo.model_validate({"response": "maybe"})
    # What the response used to tell about another visitor's run is refused outright.
    for extra in (
        {"age_s": 240},
        {"original": {"timing_ms": {"total": 9}, "usage": {}}},
        {"retrieval_hits": 3},
        {"pick_hits": 2},
    ):
        with pytest.raises(ValueError):
            CacheInfo.model_validate({"response": "hit", **extra})
    fixtures = PROJECT / "tools" / "fixtures"
    for name in ("ner.json", "ner-cached.json"):
        data = json.loads((fixtures / name).read_text(encoding="utf-8"))
        response = NerResponse.model_validate(data)
        assert response.cache is not None, name
        assert response.model_dump(mode="json") == data, name
        assert set(data["cache"]) == {"response"}, name
    cached = NerResponse.model_validate(json.loads((fixtures / "ner-cached.json").read_text()))
    assert cached.cache is not None and cached.cache.response == "hit"
    assert cached.usage.llm_calls == 0 and set(cached.timing_ms) == {"total"}


# --- only whole answers by default: what a run shows of earlier runs -------------------
#
# The second review of the caches. The lettered points are those of its brief: (a) the
# index view in the key, (b) known provider refusals, (c) cut-off output, (d) unclean
# runs, (e) sensitive values, (f) expiry, (g) Jev's store. Item 2 comes first.

ACID_TEXTS = (
    "Use acetic acid for cleaning the bench.",
    "Use acetic acid in the lab, she said.",
    "Use acetic acid with care, always.",
)
ACID = _answer(("acetic acid", "chemical", 4, 15))


@pytest.fixture(scope="module")
def toy_dense(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Bundled samples with a toy dense index next to BM25: every status ``ok``, and an
    encoder whose calls can be counted."""
    from tests.test_site_mapping import ToyEncoder, _tool

    out = tmp_path_factory.mktemp("toy-dense") / "indexes"
    toy = ToyEncoder()
    _tool().build_all(out, ToyEncoder())
    indexes = indexes_mod.open_indexes(
        cache_dir=tmp_path_factory.mktemp("toy-bm25"),
        dense_dir=out,
        api_key=None,
        encoder_factory=lambda: toy,
    )
    return indexes, toy


def _work(out: dict[str, Any]) -> dict[str, Any]:
    """What a response says its run did, without the durations."""
    return {"usage": out["usage"], "stages": sorted(out["timing_ms"])}


def test_three_texts_sharing_a_mention_each_do_and_report_all_their_work(
    tmp_path: Path,
    toy_dense: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Item 2. With the defaults (whole answers only) three different texts that share
    their one mention are three whole-response misses, and each one reads, embeds,
    searches and picks as if it were the first: the public call counts and tokens of
    the three are the same, so none of them shows that the mention was met before."""
    indexes, toy = toy_dense
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: indexes)
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live(tmp_path)
    assert (s.cache.responses, s.cache.retrieval, s.cache.picks) == (128, 0, 0)
    llm = _reader(ACID)
    embedded = len([call for call in toy.calls if call[0]])
    seen: list[dict[str, Any]] = []
    for number, text in enumerate(ACID_TEXTS, start=1):
        result = _post(s, {"text": text, "selector": LUNA}, llm=llm)
        out = _json(result)
        assert result[0] == 200 and out["cache"] == {"response": "miss"}
        assert out["entities"][0]["text"] == "acetic acid"
        assert out["entities"][0]["pick"]["status"] == "match"
        assert out["retrieval"]["dense"] == {"status": "ok"} and not out["warnings"]
        # The work was done, not only reported: one reading, one embedding request and
        # one pick more for every text.
        assert len(llm.requests) == number and len(picker.requests) == number
        assert len([call for call in toy.calls if call[0]]) == embedded + number
        record = _log_of(capsys)
        assert record["cache"] == {
            "response": "miss",
            "retrieval_hits": 0,
            "pick_hits": 0,
            "stored": True,
        }
        seen.append(_work(out))
    assert seen[0] == seen[1] == seen[2]
    assert seen[0]["usage"] == {
        "llm_calls": 1,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "embedding_requests": 1,
        "selector_calls": 1,
        "selector_prompt_tokens": 100,
        "selector_completion_tokens": 20,
        "selector_cost_usd": None,
    }
    assert seen[0]["stages"] == ["extract", "map", "select", "total"]
    # No query vector of a finished request is left for the next one.
    assert all(len(i.encoder) == 0 for i in indexes.values() if i.encoder is not None)
    caches = _caches(s)
    assert (len(caches.responses), len(caches.retrieval), len(caches.picks)) == (3, 0, 0)
    # The whole answer is what is reused, and only for the identical request.
    hit = _json(_post(s, {"text": ACID_TEXTS[0], "selector": LUNA}))
    assert hit["cache"] == {"response": "hit"} and len(llm.requests) == 3


def test_with_the_experimental_caches_on_a_later_text_shows_what_was_reused(
    tmp_path: Path, toy_dense: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contrast, and why the two are off by default: with them on, the second text
    reports no embedding request, which a visitor can read as "somebody ran this
    mention before me"."""
    indexes, _ = toy_dense
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: indexes)
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live3(tmp_path)
    llm = _reader(ACID)
    counts = [
        _json(_post(s, {"text": text, "selector": LUNA}, llm=llm))["usage"]["embedding_requests"]
        for text in ACID_TEXTS
    ]
    assert counts == [1, 0, 0]


# --- (a) the key is made once, from the indexes the request opened ----------------------


def test_an_answer_made_on_one_view_is_not_stored_under_the_view_published_meanwhile(
    tmp_path: Path, clean: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request runs on view A and is held while its model answers. Another request
    opens view B (the catalog was loaded anew). A finishes. Its answer belongs to A:
    the same request on B must run, not be handed what was made on A."""
    view_a = clean
    view_b = {
        slug: dataclasses.replace(index, retrievers=[Reloaded(r) for r in index.retrievers])
        for slug, index in clean.items()
    }
    view = {"now": view_a}
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: view["now"])
    s = _live(tmp_path)
    reading, release = threading.Event(), threading.Event()

    def held(request: LLMRequest) -> str:
        reading.set()
        assert release.wait(timeout=10)
        return HALO

    results: list[Handled] = []
    paused = threading.Thread(
        target=lambda: results.append(_post(s, {"text": HALO_TEXT}, llm=FakeLLM(handler=held)))
    )
    paused.start()
    assert reading.wait(timeout=10)
    # While A's request waits for its model: B is published and a request runs on it.
    view["now"] = view_b
    other = _post(s, {"text": "Another text."}, llm=_reader('{"entities": []}'))
    assert other[0] == 200 and _json(other)["cache"] == {"response": "miss"}
    release.set()
    paused.join(timeout=30)
    assert results and results[0][0] == 200
    caches = _caches(s)
    assert len(caches.responses) == 2  # A's answer was stored, under A
    # The next request of A's shape, while B is the view: it runs.
    llm = _reader()
    under_b = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
    assert under_b["cache"] == {"response": "miss"} and len(llm.requests) == 1
    # A's answer is found only by a request that opens A itself.
    view["now"] = view_a
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"] == {"response": "hit"}
    keys = set(caches.responses.digests())
    assert len(keys) == 3

    def key(text: str, indexes: Any) -> str:
        return caching.response_key(
            text=text,
            prompt=s.default_prompt,
            model=s.default_model,
            selector="retrieval",
            top_k=5,
            indexes=caching.index_identity(indexes),
        )

    assert keys == {key(HALO_TEXT, view_a), key(HALO_TEXT, view_b), key("Another text.", view_b)}


# --- (b) a refusal this instance already knows of ---------------------------------------


def _refuse_claude_probe(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> list[int]:
    """The health probe's free call is refused; returns the list of probe calls."""
    asked: list[int] = []

    def refused(api_key: str, **kw: Any) -> tuple[str, str | None]:
        asked.append(1)
        return "unavailable", "key refused"

    monkeypatch.setattr(service, "key_status", refused)
    status = service.check_claude_key(settings)
    assert status is not None and status.status == "unavailable" and asked == [1]
    return asked


def _refuse_openrouter_key(settings: Settings, status: int = 401) -> None:
    import httpx

    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, json={"error": {"message": "no such key"}})
    )
    health = service.check_key(settings, transport=transport)
    assert health is not None and health.status == "unavailable"


@pytest.fixture
def key_memo() -> Any:
    service.reset_key_check()
    yield
    service.reset_key_check()


def test_a_hit_is_not_served_while_the_health_probe_knows_the_claude_key_is_refused(
    tmp_path: Path,
    opened: list[int],
    visits: Any,
    key_memo: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The reviewer's case: an Opus answer is stored, then ``/api/health`` learns that
    the operator's key is refused. No request has met the refusal, so the routing block
    is empty; the stored answer is still not handed out."""
    s = _open_site(tmp_path)
    claude = _Claude()
    body = {"text": HALO_TEXT}
    assert _json(_post(s, body, claude=claude))["model"] == OPUS
    assert _json(_post(s, body))["cache"] == {"response": "hit"}
    other = {"text": HALO_TEXT, "model": LUNA}
    assert _post(s, other, llm=_reader())[0] == 200
    capsys.readouterr()

    asked = _refuse_claude_probe(monkeypatch, s)
    assert fallback.claude_block() is None
    again = _post(s, body, claude=claude)
    assert again[0] == 200 and _json(again)["cache"] == {"response": "miss"}
    assert claude.requests == 2  # it ran: the provider is what answers now
    record = _log_of(capsys)
    assert record["cache_skipped"] == "claude_key_refused"
    # Deciding it asked no provider anything.
    assert asked == [1]
    # An answer that needed no Claude is still served; so is a Claude selector's kept back.
    assert _json(_post(s, other))["cache"] == {"response": "hit"}
    assert "cache_skipped" not in _log_of(capsys)
    assert service.known_refusal(s, LUNA, OPUS) == "claude_key_refused"
    assert service.known_refusal(s, LUNA, "retrieval") is None
    # The memo expires (a refused key is asked about again after a minute): the hit is back.
    service.reset_key_check()
    assert _json(_post(s, body))["cache"] == {"response": "hit"}


def test_a_hit_is_not_served_while_the_openrouter_key_is_known_to_be_refused(
    tmp_path: Path,
    opened: list[int],
    visits: Any,
    key_memo: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s = _open_site(tmp_path)
    picker = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    luna = {"text": HALO_TEXT, "model": LUNA}
    mixed = {"text": HALO_TEXT, "selector": LUNA}  # Claude reads, an OpenRouter model picks
    opus = {"text": HALO_TEXT}
    jev = {"text": HALO_TEXT, "model": OPUS, "selector": JEV}
    assert _post(s, luna, llm=_reader())[0] == 200
    assert _post(s, mixed, claude=_Claude())[0] == 200
    assert _post(s, opus, claude=_Claude())[0] == 200
    for body in (luna, mixed, opus):
        assert _json(_post(s, body))["cache"] == {"response": "hit"}
    capsys.readouterr()

    _refuse_openrouter_key(s)
    llm = _reader()
    assert _json(_post(s, luna, llm=llm))["cache"] == {"response": "miss"}
    assert len(llm.requests) == 1
    assert _log_of(capsys)["cache_skipped"] == "openrouter_key_refused"
    claude = _Claude()
    assert _json(_post(s, mixed, claude=claude))["cache"] == {"response": "miss"}
    assert claude.requests == 1
    # An answer both of whose stages ran on Claude does not need OpenRouter.
    assert _json(_post(s, opus))["cache"] == {"response": "hit"}
    # Jev runs on the OpenRouter key; so does an index view that embeds its queries.
    assert service.known_refusal(s, jev["model"], jev["selector"]) == "openrouter_key_refused"
    assert service.known_refusal(s, OPUS, "retrieval") is None
    assert service.known_refusal(s, OPUS, "retrieval", embeds=True) == "openrouter_key_refused"


def test_only_a_definitive_and_unexpired_refusal_keeps_a_hit_back(
    tmp_path: Path, key_memo: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time as time_mod

    import httpx

    s = _open_site(tmp_path)
    now = time_mod.monotonic()
    assert service.known_refusal(s, OPUS, LUNA, now=now) is None
    # A probe that could not be completed says nothing about the key.
    monkeypatch.setattr(
        service, "key_status", lambda api_key, **kw: ("ok", "key not checked (unreachable)")
    )
    service.check_claude_key(s)
    unreachable = httpx.MockTransport(lambda request: httpx.Response(500))
    assert service.check_key(s, transport=unreachable).status == "ok"  # type: ignore[union-attr]
    assert service.known_refusal(s, OPUS, LUNA, now=now) is None
    service.reset_key_check()
    # A spending limit that is used up is definitive, like a refused key.
    spent = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": {"limit_remaining": 0}})
    )
    assert service.check_key(s, transport=spent).status == "unavailable"  # type: ignore[union-attr]
    assert service.known_refusal(s, LUNA, None) == "openrouter_key_refused"
    # ... until it expires.
    later = time_mod.monotonic() + service.KEY_REFUSED_TTL_S + 1
    assert service.known_refusal(s, LUNA, None, now=later) is None
    # The exhausted balance a request met (the routing block), with its own time limit.
    fallback.block_claude("balance_exhausted", now=now)
    assert service.known_refusal(s, OPUS, "retrieval", now=now) == "claude_blocked"
    assert service.known_refusal(s, LUNA, SONNET, now=now) == "claude_blocked"
    assert service.known_refusal(s, OPUS, "retrieval", now=now + fallback.BLOCK_TTL_S) is None
    # Offline needs no provider.
    assert service.known_refusal(_offline(tmp_path), "offline", JEV) is None


# --- (c) an answer the provider cut off ------------------------------------------------


@pytest.mark.parametrize("reason", ["length", "max_tokens"])
def test_a_reading_the_provider_cut_off_is_served_and_not_stored(
    reason: str, tmp_path: Path, opened: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    """The answer parses, so the visitor sees what they saw before (no warning exists
    for it); the list may have stopped early, so it is nobody else's answer."""
    s = _live3(tmp_path)
    whole = _json(_post(s, {"text": HALO_TEXT + " "}, llm=_reader()))
    capsys.readouterr()
    cut = FakeLLM(handler=lambda request: HALO, finish_reason=reason)
    for calls in (1, 2):
        out = _json(_post(s, {"text": HALO_TEXT}, llm=cut))
        assert out["cache"] == {"response": "miss"} and len(cut.requests) == calls
        assert out["warnings"] == [] and out["entities"] == whole["entities"]
        assert _log_of(capsys)["cache"]["stored"] is False
    assert len(_caches(s).responses) == 1  # the clean run's, and only it


def test_a_pick_the_provider_cut_off_is_used_and_neither_it_nor_its_run_is_stored(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    picker = FakeLLM(handler=lambda request: PICK, finish_reason="length")
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live3(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT, "selector": LUNA}, llm=_reader())
    assert out["entities"][0]["pick"]["status"] == "match" and out["warnings"] == []
    caches = _caches(s)
    assert len(picker.requests) == 2 and (len(caches.responses), len(caches.picks)) == (0, 0)
    # The same question with a whole answer is stored.
    whole = FakeLLM(handler=lambda request: PICK)
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: whole)
    assert _post(s, {"text": HALO_TEXT, "selector": LUNA}, llm=_reader())[0] == 200
    assert (len(caches.responses), len(caches.picks)) == (1, 1)


def test_the_finish_reason_is_carried_through_both_stages() -> None:
    from ner_site.extract import FinishWatch

    cut = FakeLLM(["{}", "{}"], model="m", finish_reason="length")
    watch = FinishWatch(cut)
    assert not watch.cut_off and watch.model == "m" and watch.fingerprint == cut.fingerprint
    asyncio.run(watch.complete(LLMRequest(system="s", user="u")))
    assert watch.cut_off
    whole = FinishWatch(FakeLLM(["{}"]))
    asyncio.run(whole.complete(LLMRequest(system="s", user="u")))
    assert not whole.cut_off
    # The pick stage, on its own: the pick is used and the cache is given nothing.
    cache = Caches(EXPERIMENTAL).picks
    llm = FakeLLM(handler=_choose(), finish_reason="length")
    (result,), stats = _select(LUNA, cache, llm=llm)
    assert result.pick.status == "match" and stats.cut_off == {"e1"} and len(cache) == 0
    # And the run counts it for the one rule.
    reused: dict[str, int] = {}
    settings = Settings(api_key=SITE_KEY, cache=ON)
    response = asyncio.run(
        service.run(
            NerRequest(text=HALO_TEXT, selector="retrieval"),
            settings=settings,
            llm=FakeLLM(handler=lambda request: HALO, finish_reason="length"),
            indexes=WHOLE,
            caches=Caches(ON),
            reused=reused,
        )
    )
    assert reused["cut_off"] == 1 and response.warnings == []


# --- (d) one rule: only a wholly clean run is stored ------------------------------------


@pytest.mark.parametrize(
    "counted",
    [
        pytest.param({"cut_off": 1}, id="an answer cut off at the token limit"),
        pytest.param({"served_elsewhere": 1}, id="a pick another model served"),
        pytest.param({"degraded": 1}, id="candidates that were not a clean answer"),
        pytest.param({"scrubbed": 1}, id="a secret the run scrubbed"),
    ],
)
def test_what_the_run_counted_beside_the_response_keeps_it_out(counted: dict[str, int]) -> None:
    response = NerResponse.model_validate(_clean_response())
    clean_run = {"retrieval_hits": 0, "pick_hits": 0, **CLEAN_RUN}
    assert cacheable(response, WHOLE, settings=Settings(), run=clean_run)
    assert not cacheable(response, WHOLE, settings=Settings(), run={**clean_run, **counted})
    # What was reused from the experimental caches is not a reason by itself.
    assert cacheable(response, WHOLE, settings=Settings(), run={**clean_run, "pick_hits": 3})


def test_the_handler_asks_the_one_rule_and_nothing_else(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is stored is what ``service.cacheable`` allows: the typed response, the
    indexes of the run and the run's own counts are what it is shown, before anything
    is redacted."""
    seen: list[tuple[Any, Any, Any]] = []

    def rule(response: Any, indexes: Any, **kw: Any) -> bool:
        seen.append((response, indexes, kw))
        return len(seen) > 1

    monkeypatch.setattr(http_mod, "cacheable", rule)
    s = _live(tmp_path)
    llm = _reader()
    for stored in (0, 1):
        assert _json(_post(s, {"text": HALO_TEXT}, llm=llm))["cache"] == {"response": "miss"}
        assert len(_caches(s).responses) == stored
    response, indexes, kw = seen[0]
    assert isinstance(response, NerResponse) and indexes is http_mod.get_indexes(s)
    assert kw["settings"] is s and kw["run"]["cut_off"] == 0 and RAW_IP in kw["extra_secrets"]


def test_a_reader_warning_keeps_the_whole_run_out(tmp_path: Path, opened: list[int]) -> None:
    """The reviewer's first case, with all three caches on: the response is not stored."""
    llm = _reader(_answer(("haloperidol", "chemical", 9, 20), ("unobtainium", "chemical", 0, 3)))
    s = _live3(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT}, llm=llm)
    assert out["warnings"] == ["entity 'unobtainium' dropped: not found in text"]
    assert len(_caches(s).responses) == 0


KETA = _answer(("haloperidol", "chemical", 13, 24), ("ketamine", "chemical", 35, 43))


class _ServedElsewhere(FakeLLM):
    """A Claude selector whose first answer the provider served from another model
    (a server-side refusal fallback); every later answer is its own."""

    async def complete(self, request: LLMRequest) -> Any:
        from ner_site.claude import ClaudeResponse

        answer = await super().complete(request)
        first = len(self.requests) == 1
        return ClaudeResponse(
            text=answer.text,
            usage=answer.usage,
            model=SONNET if first else answer.model,
            finish_reason=answer.finish_reason,
            served_by_fallback=first,
        )


def test_a_server_side_refusal_fallback_on_any_pick_keeps_the_whole_run_out(
    tmp_path: Path, opened: list[int], visits: Any
) -> None:
    """The reviewer's second case: an OpenRouter model reads, Claude picks, and the
    provider served the first of two picks from another model. Neither the response
    nor the later pick, which Opus itself made in that run, is stored."""
    s = _open_site(tmp_path, **ALL_THREE)
    made: list[_ServedElsewhere] = []

    def claude(settings: Settings, model: str, purpose: str) -> FakeLLM:
        if model == OPUS and purpose == "select":
            made.append(_ServedElsewhere(handler=lambda request: PICK, model=OPUS))
            return made[-1]
        return FakeLLM(handler=lambda request: PICK, model=model)

    body = {"text": KETA_TEXT, "model": LUNA, "selector": OPUS}
    for runs in (1, 2):
        result = _post(s, body, llm=_reader(KETA), claude=claude)
        out = _json(result)
        assert result[0] == 200 and out["cache"] == {"response": "miss"}
        picks = [e["pick"] for e in out["entities"]]
        assert [p["status"] for p in picks] == ["match", "match"]
        assert [p["selector"] for p in picks] == [SONNET, OPUS]
        assert [f["reason"] for f in out["fallbacks"]] == ["refused"]
        assert len(made) == runs and len(made[-1].requests) == 2
    caches = _caches(s)
    assert (len(caches.responses), len(caches.picks)) == (0, 0)


# --- (e) values that must not be kept ----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://user:hunter2-SENTINEL@proxy.example/v1",
        "https://user@proxy.example/v1",
        "https://:hunter2-SENTINEL@proxy.example/v1",
        "https://proxy.example/v1?token=tok-SENTINEL",
        "https://proxy.example/v1?",
        "https://user:hunter2-SENTINEL@proxy.example/v1?token=tok-SENTINEL",
    ],
)
def test_a_base_url_with_user_information_or_a_query_is_refused(url: str) -> None:
    with pytest.raises(SettingsError, match="NER_SITE_BASE_URL") as raised:
        Settings.from_env({"NER_SITE_BASE_URL": url})
    message = str(raised.value)
    assert "no user name or password" in message and "no query string" in message
    assert "SENTINEL" not in message and "proxy.example" not in message
    # A Settings built in code is held to the same rule.
    with pytest.raises(SettingsError):
        Settings(base_url=url)
    # And a deployment with such a value answers "misconfigured", it does not start.
    found = http_mod.env_settings({"NER_SITE_BASE_URL": url, "NER_SITE_OFFLINE": "1"})
    assert isinstance(found, SettingsError) and "SENTINEL" not in str(found)


def test_an_ordinary_base_url_is_accepted() -> None:
    for url in ("https://openrouter.ai/api/v1", "http://localhost:8080/v1/", "https://gw.example"):
        assert Settings.from_env({"NER_SITE_BASE_URL": url}).base_url == url.rstrip("/")
    for url in ("ftp://gw.example/v1", "gw.example/v1", "https:///v1"):
        with pytest.raises(SettingsError, match="NER_SITE_BASE_URL"):
            Settings.from_env({"NER_SITE_BASE_URL": url})


def test_a_response_that_holds_the_requests_client_address_is_not_stored(
    tmp_path: Path, opened: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    """The address is in no response unless the visitor's own text holds it; then the
    answer is theirs alone."""
    text = f"Host {RAW_IP} was given haloperidol."
    answer = _answer((RAW_IP, "other", 5, 5 + len(RAW_IP)))
    s = _live(tmp_path)
    for _ in range(2):
        out = _json(_post(s, {"text": text}, llm=_reader(answer)))
        assert out["cache"] == {"response": "miss"} and out["entities"][0]["text"] == RAW_IP
        assert _log_of(capsys)["cache"]["stored"] is False
    assert len(_caches(s).responses) == 0
    # Whichever header carried it, and every entry of a forwarding list.
    forwarded = {k: v for k, v in HEADERS.items() if k != "x-real-ip"}
    forwarded["x-forwarded-for"] = f"198.51.100.4, {RAW_IP}"
    _post(s, {"text": text}, llm=_reader(answer), headers=forwarded)
    assert len(_caches(s).responses) == 0
    # From another address the same text is an ordinary one, and is kept.
    elsewhere = {**HEADERS, "x-real-ip": "198.51.100.9"}
    _post(s, {"text": text}, llm=_reader(answer), headers=elsewhere)
    assert _log_of(capsys)["cache"]["stored"] is True and len(_caches(s).responses) == 1
    assert http_mod._private(forwarded, forwarded) == ("198.51.100.4", RAW_IP)  # noqa: SLF001


def test_a_response_that_holds_a_key_or_passcode_the_request_sent_is_not_stored(
    tmp_path: Path, opened: list[int]
) -> None:
    """A deployment without own keys ignores an ``Authorization`` header; what came in
    it is still the visitor's, and so is a passcode."""
    token = "tok-visitor-0123456789"
    text = f"Key {token} and haloperidol."
    answer = _answer((token, "other", 4, 4 + len(token)))
    s = _live(tmp_path, NER_SITE_PASSCODE=PASSCODE)
    site = {**HEADERS, "x-passcode": PASSCODE}
    for scheme in ("Bearer ", "Basic ", ""):
        carried = {**site, "Authorization": [scheme + token]}
        assert _post(s, {"text": text}, llm=_reader(answer), headers=carried)[0] == 200
        assert len(_caches(s).responses) == 0
    assert _post(s, {"text": text}, llm=_reader(answer), headers=site)[0] == 200
    assert len(_caches(s).responses) == 1
    # The configured passcode, in the text: served, never kept.
    spoken = f"The code is {PASSCODE} for haloperidol."
    said = _answer((PASSCODE, "other", 12, 12 + len(PASSCODE)))
    out = _json(_post(s, {"text": spoken}, llm=_reader(said), headers=site))
    assert out["ok"] and len(_caches(s).responses) == 1
    assert http_mod._private(  # noqa: SLF001
        {"x-passcode": PASSCODE}, {"Authorization": "Bearer " + token}
    ) == (PASSCODE, "Bearer " + token, token)


def test_a_changed_passcode_starts_with_empty_caches(tmp_path: Path, opened: list[int]) -> None:
    """The reviewer's case: an answer is stored, then the passcode becomes a word that
    occurs in it. The new passcode is another scope, so the answer is not found, and
    what the old scope held is emptied, not left in memory for its time limit."""
    old = _live(tmp_path, NER_SITE_PASSCODE="zz")
    asked = {"text": HALO_TEXT}
    assert _post(old, asked, llm=_reader(), headers={**HEADERS, "x-passcode": "zz"})[0] == 200
    before = _caches(old)
    assert len(before.responses) == 1 and "haloperidol" in before.responses.values()[0]

    new = _live(tmp_path, NER_SITE_PASSCODE="halo")
    assert new.cache_scope != old.cache_scope and new.cache_family == old.cache_family
    llm = _reader()
    out = _json(_post(new, asked, llm=llm, headers={**HEADERS, "x-passcode": "halo"}))
    assert out["cache"] == {"response": "miss"} and len(llm.requests) == 1
    # Nothing that holds the new passcode is in memory, under either scope.
    assert len(before.responses) == 0 and len(_caches(new).responses) == 0
    assert len(caching._CACHES) == 1  # noqa: SLF001
    # Another deployment in the same process is not touched by a passcode change here.
    elsewhere = _live(tmp_path, NER_SITE_PASSCODE="zz", NER_SITE_MAX_ENTITIES="7")
    assert elsewhere.cache_family != old.cache_family
    _post(elsewhere, asked, llm=_reader(), headers={**HEADERS, "x-passcode": "zz"})
    back = _live(tmp_path, NER_SITE_PASSCODE="zz")
    _post(back, asked, llm=_reader(), headers={**HEADERS, "x-passcode": "zz"})
    assert len(_caches(elsewhere).responses) == 1 and len(_caches(back).responses) == 1


# --- (f) the time of a write is read inside the lock ------------------------------------


def test_the_time_of_a_write_is_read_with_the_lock_held() -> None:
    """Two writers that read the time before taking the lock could store their times
    in the opposite order of their entries, and the sweep, which stops at the first
    entry still alive, would leave an expired one behind it."""
    held: list[bool] = []
    cache: TTLCache[str]

    class Watching(Clock):
        def __call__(self) -> float:
            held.append(cache._lock.locked())  # noqa: SLF001
            return self.now

    clock = Watching()
    cache = TTLCache(8, 10, clock=clock)
    cache.put("a", "one")
    cache.get("a")
    cache.sweep()
    assert held == [True, True, True]


def test_writers_that_race_leave_their_times_in_the_order_of_their_entries() -> None:
    """The reviewer's sequence, forced: the writer that takes the lock first is the one
    that gets the earlier time, whatever it would have read before waiting."""
    ticks = iter(range(1000))
    lock = threading.Lock()

    def clock() -> float:
        with lock:
            return float(next(ticks))

    cache: TTLCache[int] = TTLCache(1000, 10_000, clock=clock)
    start = threading.Barrier(8)

    def work(worker: int) -> None:
        start.wait(timeout=5)
        for i in range(40):
            cache.put(f"{worker}:{i}", i)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    stored = list(cache._stored.values())  # noqa: SLF001 - the order the sweep walks
    assert len(stored) == 320 and stored == sorted(stored)
    # So nothing expired is hidden behind a younger entry: with a short limit, a sweep
    # at any time removes exactly the entries that are old enough.
    clock_now = Clock()
    short: TTLCache[int] = TTLCache(8, 10, clock=clock_now)
    clock_now.now = 0
    short.put("first", 1)
    clock_now.now = 5
    short.put("second", 2)
    clock_now.now = 11
    assert short.sweep() == 1 and short.digests() == ["second"]


def test_the_query_vectors_are_stamped_inside_their_lock_too() -> None:
    from tests.test_site_mapping import ToyEncoder

    held: list[bool] = []
    encoder: Any

    class Watching(Clock):
        def __call__(self) -> float:
            held.append(encoder._lock.locked())  # noqa: SLF001
            return self.now

    encoder = indexes_mod.PrefetchEncoder(ToyEncoder(), clock=Watching())
    encoder._remember(["haloperidol"], [[0.0]])  # noqa: SLF001
    assert held == [True]


# --- (g) Jev judges the candidates it was given -----------------------------------------


def test_jev_reads_the_candidates_it_was_given_not_the_index_store() -> None:
    """The reviewer's case: the candidates are there (they are what the page lists and
    what the pick's key describes) and the index's store does not hold their records
    at that moment. Jev used to find nothing to screen and answer ``no_match`` after
    no call at all, and that was stored."""
    from xwalk.stores.memory import MemoryStore

    from tests.test_site_select import ASPIRIN, PARACETAMOL

    _, candidates = _setup()
    assert [c.id for c in candidates["e1"]] == [PARACETAMOL.id, ASPIRIN.id]
    emptied = {"chebi": OntologyIndex("chebi", MemoryStore.from_source([]), [], {})}
    changed = Record(ASPIRIN.id, {"label": "something else entirely", "synonyms": []})
    other = {"chebi": OntologyIndex("chebi", MemoryStore.from_source([changed]), [], {})}
    for indexes in (emptied, other):
        cache = Caches(EXPERIMENTAL).picks
        decider = FakeDecider(handler=_knows_aspirin)
        stats = SelectStats()
        (result,) = asyncio.run(
            select_entities(
                [_picked_entity()],
                candidates,
                indexes,
                selector=JEV,
                text=PICK_TEXT,
                stats=stats,
                decider=decider,
                cache=cache,
            )
        )
        assert result.pick is not None and result.pick.status in {"match", "review"}
        assert result.pick.id == ASPIRIN.id and result.pick.label == "aspirin"
        assert stats.calls == len(decider.calls) >= 2
        seen = json.dumps(decider.calls, default=str)
        assert "acetylsalicylic acid" in seen and "something else entirely" not in seen


# --- item 9: offline, dense retrieval is not opened and not checked ---------------------


def test_offline_dense_retrieval_is_neither_opened_nor_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: Any, **kw: Any) -> Any:
        pytest.fail("dense retrieval was opened or checked in offline mode")

    monkeypatch.setattr(indexes_mod, "open_dense", never)
    monkeypatch.setattr(indexes_mod, "default_embeddings_encoder", never)
    disabled = indexes_mod.dense_disabled()
    assert disabled.status == "unavailable" and disabled.reason == indexes_mod.DENSE_OFF_REASON
    assert disabled.reason.startswith("disabled in offline mode")
    # Whatever the dense directory holds: nothing, an index, or a broken one.
    present = tmp_path / "dense" / "chebi-sample" / "dense"
    present.mkdir(parents=True)
    (present / "xwalk_meta.json").write_text("not json")
    seen = []
    for dense_dir, key in ((tmp_path / "missing", ""), (tmp_path / "dense", SITE_KEY)):
        s = _offline(tmp_path, NER_SITE_DENSE_DIR=str(dense_dir), OPENROUTER_API_KEY=key)
        indexes = indexes_mod.get_indexes(s.index_settings())
        for index in indexes.values():
            assert index.status["dense"] == disabled and index.dense_off and index.whole
            assert index.encoder is None and [r.name for r in index.retrievers] == ["bm25"]
        # One status for the deployment, exactly: no per-ontology list, no other reason.
        assert service.retrieval_status(indexes, s)["dense"] == disabled
        out = _json(_post(s, {"text": KETA_TEXT}))
        assert out["retrieval"]["dense"] == disabled.model_dump(mode="json")
        assert out["warnings"] == ["dense retrieval unavailable; matches use BM25 only"]
        assert _json(_post(s, {"text": KETA_TEXT}))["cache"] == {"response": "hit"}
        seen.append(out["retrieval"])
        indexes_mod.reset()
    assert seen[0] == seen[1]


def test_live_a_missing_dense_index_is_still_checked_and_keeps_answers_out(
    tmp_path: Path,
) -> None:
    """The same directory without the mode: the reason is the real one, and it is a
    degraded deployment whose answers are not kept."""
    s = _live(tmp_path)
    indexes = indexes_mod.get_indexes(s.index_settings())
    reasons = {index.status["dense"].reason for index in indexes.values()}
    assert not any(index.dense_off or index.whole for index in indexes.values())
    assert all(reason and "no index at" in reason for reason in reasons)
    for _ in range(2):
        assert _json(_post(s, {"text": HALO_TEXT}, llm=_reader()))["cache"] == {"response": "miss"}
    assert len(_caches(s).responses) == 0
