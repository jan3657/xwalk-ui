"""The NER site's in-memory caches (ner_site.cache): whole responses, retrieval, picks.

Offline: FakeLLM, FakeDecider, BM25 built in a temporary directory, the limiter's and the
recorder's tables in memory, and an injected clock (nothing here sleeps to let time pass).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from ner_site import cache as caching
from ner_site import fake, service, usage
from ner_site import http as http_mod
from ner_site import indexes as indexes_mod
from ner_site.cache import ON, Caches, CacheSettings, TTLCache
from ner_site.indexes import OntologyIndex
from ner_site.mapping import MapStats, map_entities
from ner_site.schema import CacheInfo, Entity, NerRequest, NerResponse, RetrieverStatus
from ner_site.select import JEV, SelectStats, select_entities
from ner_site.service import Settings, SettingsError, cacheable
from xwalk.decide.fake import FakeDecider
from xwalk.llm.base import LLMFatalError, LLMRequest, LLMRetryableError
from xwalk.llm.fake import FakeLLM
from xwalk.records import RetrievalHit
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
    yield
    indexes_mod.reset()
    caching.reset()


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
        text="secret text", prompt="p", model="m", selector="s", top_k=5, provider="site"
    )
    assert len(key) == 64 and "secret" not in key


# --- settings ------------------------------------------------------------------------


def test_the_environment_turns_the_caches_on_with_the_default_sizes() -> None:
    s = Settings.from_env({})
    assert s.cache == ON == CacheSettings(128, 2048, 8192, 6 * 3600)
    assert s.cache.enabled
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
    # Nothing secret tells them apart.
    keyed = Settings.from_env(
        {"OPENROUTER_API_KEY": SITE_KEY, "NER_SITE_PASSCODE": PASSCODE, "DATABASE_URL": DB_URL}
    )
    scope = repr(keyed.cache_scope)
    assert SITE_KEY not in scope and PASSCODE not in scope and "npg_SENTINEL" not in scope


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
    cache = Caches(ON).retrieval
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
    cache = Caches(ON).retrieval
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
    cache = Caches(ON).retrieval
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
    cache = Caches(ON).retrieval
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
    cache = Caches(ON).retrieval
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    kept = [c.id for c in stats.candidates["e1"]]
    stats.candidates["e1"].clear()
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    stats.candidates["e1"].reverse()
    _, stats = _map([_mention("e1", "haloperidol")], indexes, cache)
    assert stats.cache_hits == 1 and [c.id for c in stats.candidates["e1"]] == kept
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.candidates["e1"][0].fused_score = 9.0  # type: ignore[misc]


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
    cache = Caches(ON).picks
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
    cache = Caches(ON).picks
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
    cache = Caches(ON).picks
    llm = FakeLLM(handler=_choose())
    candidates = _setup()[1]["e1"]
    _select(LUNA, cache, llm=llm)
    for shown in (list(reversed(candidates)), candidates[1:]):
        _, stats = _select(LUNA, cache, llm=llm, candidates={"e1": shown})
        assert stats.cache_hits == 0
    assert len(llm.requests) == 3
    _, stats = _select(LUNA, cache, llm=llm, candidates={"e1": list(reversed(candidates))})
    assert stats.cache_hits == 1


def test_another_selector_another_payer_or_another_type_is_asked_again() -> None:
    cache = Caches(ON).picks
    llm = FakeLLM(handler=_choose())
    _select(LUNA, cache, llm=llm)
    _, stats = _select("google/gemini-3.1-flash-lite", cache, llm=llm)
    assert stats.cache_hits == 0
    _, stats = _select(LUNA, cache, llm=llm, payer="openrouter", own_key=True)
    assert stats.cache_hits == 0
    _, stats = _select(LUNA, cache, llm=llm, results=[_picked_entity(type="food")])
    assert stats.cache_hits == 0
    decider = FakeDecider(handler=_knows_aspirin)
    _, stats = _select(JEV, cache, decider=decider)
    assert stats.cache_hits == 0 and decider.calls
    assert len(llm.requests) == 4
    # Jev's own answer is remembered as Jev's.
    asked = len(decider.calls)
    (again,), stats = _select(JEV, cache, decider=decider)
    assert stats.cache_hits == 1 and len(decider.calls) == asked
    assert again.pick is not None and again.pick.selector == JEV


@pytest.mark.parametrize(
    "answer", ["not-json", '{"chosen_key": "FAKE"}', LLMFatalError("HTTP 500")]
)
def test_a_failed_pick_is_never_stored(answer: Any) -> None:
    cache = Caches(ON).picks
    (result,), stats = _select(LUNA, cache, llm=FakeLLM([answer]))
    assert result.pick is not None and result.pick.status == "failed" and len(cache) == 0
    llm = FakeLLM(handler=_choose())
    (result,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 0 and result.pick.status == "match" and len(cache) == 1


def test_a_pick_cut_off_by_the_deadline_is_never_stored() -> None:
    class Slow(FakeLLM):
        async def complete(self, request: LLMRequest) -> Any:
            await asyncio.sleep(5)

    cache = Caches(ON).picks
    (result,), _ = _select(LUNA, cache, llm=Slow([]), deadline=0.05)
    assert result.pick is not None and result.pick.status == "failed" and len(cache) == 0


def test_a_no_match_is_an_answer_and_is_stored() -> None:
    cache = Caches(ON).picks
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

    cache = Caches(ON).picks
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


def test_on_a_chain_a_pick_belongs_to_the_model_that_made_it() -> None:
    """A pick Sonnet made after Opus stepped down is Sonnet's: a request that would ask
    Opus asks Opus, and one that starts on Sonnet gets Sonnet's pick back."""
    cache = Caches(ON).picks
    chain, made = _chain({OPUS: [_claude_error("rate_limited")]})
    (stepped,), stats = _select(OPUS, cache, llm=chain)
    assert stepped.pick.selector == SONNET and stats.asked == {"e1": SONNET}
    assert len(cache) == 1

    chain, made = _chain({})
    (fresh,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 0 and fresh.pick.selector == OPUS and len(made[OPUS].requests) == 1

    chain, made = _chain({}, start=1)  # admitted on Sonnet: the Opus allowance is used
    (again,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 1 and again.pick.selector == SONNET and not made

    chain, made = _chain({})
    (opus,), stats = _select(OPUS, cache, llm=chain)
    assert stats.cache_hits == 1 and opus.pick.selector == OPUS and not made


def test_what_a_request_does_to_a_pick_does_not_reach_the_cache() -> None:
    cache = Caches(ON).picks
    llm = FakeLLM(handler=_choose(explanation="as the model said"))
    (first,), _ = _select(LUNA, cache, llm=llm)
    first.pick.explanation = "changed by the first request"
    (second,), _ = _select(LUNA, cache, llm=llm)
    second.pick.explanation = "changed by the second request"
    (third,), stats = _select(LUNA, cache, llm=llm)
    assert stats.cache_hits == 1 and third.pick.explanation == "as the model said"


def test_a_pick_is_stored_as_the_response_may_carry_it() -> None:
    cache = Caches(ON).picks
    llm = FakeLLM(handler=_choose(explanation=f"the key is {OWN_KEY}"))
    scrub = lambda note: note.replace(OWN_KEY, "[redacted]")  # noqa: E731
    _select(LUNA, cache, llm=llm, clean=scrub)
    assert [pick.explanation for pick in cache.values()] == ["the key is [redacted]"]


# --- which responses may be stored ----------------------------------------------------


def _clean_response() -> dict[str, Any]:
    pick = {"selector": LUNA, "status": "match", "id": "CHEBI:1", "explanation": ""}
    return {
        "ok": True,
        "model": LUNA,
        "mode": "live",
        "entities": [
            {"id": "e1", "ontology": "chebi", "notes": [], "pick": pick, "matches": []},
            {"id": "e2", "ontology": "chebi", "notes": [], "pick": None, "matches": []},
            {
                "id": "e3",
                "ontology": None,
                "notes": ["no ontology for type 'other'"],
                "pick": None,
                "matches": [],
            },
        ],
        "retrieval": {"bm25": {"status": "ok"}, "dense": {"status": "ok"}},
        "warnings": [],
        "fallbacks": [],
        "notices": [],
    }


def test_a_clean_response_may_be_stored() -> None:
    assert cacheable(_clean_response())
    no_match = _clean_response()
    no_match["entities"][0]["pick"] = {"selector": LUNA, "status": "no_match", "id": None}
    assert cacheable(no_match)


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
        pytest.param(_with(["ok"], False), id="an error"),
    ],
)
def test_nothing_unclean_may_be_stored(unclean: dict[str, Any]) -> None:
    assert not cacheable(unclean)
    # Offline too, but for the one thing the stand-in never has (the next test).
    if unclean["retrieval"]["dense"]["status"] == "ok":
        assert not cacheable(unclean, offline=True)


def test_the_offline_stand_in_may_be_stored_without_dense_search() -> None:
    """Offline there is never a dense search, by design, and every answer says so with
    one fixed warning; a fresh identical run would say the same."""
    offline = _clean_response()
    offline["retrieval"]["dense"] = {"status": "unavailable", "reason": "offline mode"}
    offline["warnings"] = ["dense retrieval unavailable; matches use BM25 only"]
    assert cacheable(offline, offline=True) and not cacheable(offline)
    offline["warnings"].append("entity 'x' dropped: not found in text")
    assert not cacheable(offline, offline=True)


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


def _run(settings: Settings, text: str, indexes: Any, caches: Any, **kw: Any) -> NerResponse:
    return asyncio.run(
        service.run(
            NerRequest(text=text, **kw),
            settings=settings,
            llm=fake.offline_llm(),
            indexes=indexes,
            decider=service.make_decider(settings),
            caches=caches,
        )
    )


def test_a_run_reuses_what_an_earlier_text_shared_with_it(tmp_path: Path, bm25_only: Any) -> None:
    settings = _offline(tmp_path)
    caches = Caches(settings.cache)
    first = _run(settings, KETA_TEXT, bm25_only, caches)
    assert first.cache == CacheInfo(response="miss", retrieval_hits=0, pick_hits=0)
    assert [e.text for e in first.entities] == ["haloperidol", "ketamine"]
    assert all(e.pick is not None for e in first.entities)
    # Another text with one of the two mentions, in other surroundings: its candidates
    # are reused, its pick is not.
    shared = _run(settings, HALO_TEXT, bm25_only, caches)
    assert shared.cache == CacheInfo(response="miss", retrieval_hits=1, pick_hits=0)
    assert shared.entities[0].matches == first.entities[0].matches
    assert shared.usage.selector_calls > 0
    # The first text again: both mentions, in the same surroundings.
    again = _run(settings, KETA_TEXT, bm25_only, caches)
    assert again.cache == CacheInfo(response="miss", retrieval_hits=2, pick_hits=2)
    assert again.usage.selector_calls == 0 and again.usage.selector_cost_usd is None
    assert again.entities == first.entities
    data = again.model_dump(mode="json")
    assert data["cache"] == {"response": "miss", "retrieval_hits": 2, "pick_hits": 2}


def test_without_caches_a_response_has_no_cache_object(tmp_path: Path, bm25_only: Any) -> None:
    settings = _offline(tmp_path, NER_SITE_CACHE="off")
    response = _run(settings, HALO_TEXT, bm25_only, None)
    assert response.cache is None and "cache" not in response.model_dump(mode="json")
    only_picks = Caches(CacheSettings(responses=0, retrieval=0, picks=5))
    response = _run(settings, HALO_TEXT, bm25_only, only_picks)
    assert response.model_dump(mode="json")["cache"] == {
        "response": "off",
        "retrieval_hits": 0,
        "pick_hits": 0,
    }


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
    assert first[0] == 200 and ran["cache"] == {
        "response": "miss",
        "retrieval_hits": 0,
        "pick_hits": 0,
    }
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
    assert len(llm.requests) == 1 and opened == [1]  # no model, and no index either
    assert hit["cache"] == {
        "response": "hit",
        "retrieval_hits": 0,
        "pick_hits": 0,
        "age_s": 245,
        "original": {"timing_ms": ran["timing_ms"], "usage": ran["usage"]},
    }
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


def test_who_pays_is_part_of_the_key(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    from ner_site import providers

    built: list[str] = []

    def own_llm(credential: Any, model: str, *, purpose: str, **options: Any) -> Any:
        built.append(credential.provider)
        return FakeLLM(handler=lambda request: HALO, model=model)

    monkeypatch.setattr(providers, "own_llm", own_llm)
    s = _live(tmp_path, NER_SITE_OWN_PROVIDERS="openrouter,openai", NER_SITE_PASSCODE=PASSCODE)
    site = {**HEADERS, "x-passcode": PASSCODE}
    own = {**site, "authorization": "Bearer " + OWN_KEY}
    model = "openai/gpt-6-luna"
    assert _json(_post(s, {"text": HALO_TEXT, "model": model}, llm=_reader(), headers=site))
    paid = _post(s, {"text": HALO_TEXT, "model": model, "provider": "openrouter"}, headers=own)
    assert _json(paid)["cache"]["response"] == "miss" and built == ["openrouter"]
    assert _json(paid)["credential_source"] == "visitor"
    # The same request on an own key again: answered without the key being used.
    again = _post(s, {"text": HALO_TEXT, "model": model, "provider": "openrouter"}, headers=own)
    assert _json(again)["cache"]["response"] == "hit" and built == ["openrouter"]
    assert _json(again)["provider"] == "openrouter"
    # Another provider's key is another payer.
    other = _post(s, {"text": HALO_TEXT, "provider": "openai"}, headers=own)
    assert _json(other)["cache"]["response"] == "miss" and built == ["openrouter", "openai"]


def test_a_response_expires_and_the_request_runs_again(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(caching, "_now", clock)
    s = _live(tmp_path, NER_SITE_CACHE_TTL_S="600")
    llm = _reader()
    _post(s, {"text": HALO_TEXT}, llm=llm)
    clock.now += 599
    assert _json(_post(s, {"text": HALO_TEXT}))["cache"]["age_s"] == 599
    clock.now += 1
    late = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
    # The whole answer has expired, and with it what was retrieved for it.
    assert late["cache"] == {"response": "miss", "retrieval_hits": 0, "pick_hits": 0}
    assert len(llm.requests) == 2


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
    # What a hit puts in place of the run's figures is not written back: the next hit
    # still reports the original run, not the hit before it.
    assert first["cache"]["original"] == second["cache"]["original"]
    assert second["cache"]["original"]["usage"] == ran["usage"]
    assert "original" not in json.dumps(second["cache"]["original"])
    assert list(_caches(s).responses.values()) == stored
    assert json.loads(stored[0])["cache"]["response"] == "miss"


def test_two_identical_requests_at_once_both_run_and_leave_one_entry(
    tmp_path: Path, opened: list[int]
) -> None:
    """In-flight work is not shared: both miss, both run, both store the same answer."""
    s = _live(tmp_path)
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
    tmp_path: Path, bm25_only: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: bm25_only)
    llm = _reader()
    out = _runs_again(_live(tmp_path), {"text": HALO_TEXT}, llm=llm)
    assert out["retrieval"]["dense"]["status"] == "unavailable" and len(llm.requests) == 2
    # What was retrieved cleanly for it (BM25 is all this ontology has) is still reused.
    assert out["cache"]["retrieval_hits"] == 1


def test_a_response_with_a_failed_search_is_not_stored(
    tmp_path: Path, clean: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    chebi = clean["chebi-sample"]
    broken = {
        **clean,
        "chebi-sample": dataclasses.replace(chebi, retrievers=[*chebi.retrievers, Broken()]),
    }
    monkeypatch.setattr(http_mod, "get_indexes", lambda settings: broken)
    s = _live(tmp_path)
    out = _runs_again(s, {"text": HALO_TEXT}, llm=_reader())
    assert out["entities"][0]["notes"] and out["cache"]["retrieval_hits"] == 0
    assert len(_caches(s).retrieval) == 0


def test_a_response_with_a_failed_pick_is_not_stored(
    tmp_path: Path, opened: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    picker = FakeLLM(handler=lambda request: "not json at all")
    monkeypatch.setattr(http_mod, "make_select_llm", lambda settings, model: picker)
    s = _live(tmp_path)
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
    s = _live(tmp_path, NER_SITE_OWN_PROVIDERS="openai", NER_SITE_PASSCODE=PASSCODE)
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
    assert (len(caches.responses), len(caches.retrieval), len(caches.picks)) == (1, 1, 1)
    kept = "\n".join(
        [
            *(str(v) for v in caches.responses.values()),
            *(repr(v) for v in caches.retrieval.values()),
            *(repr(v) + v.model_dump_json() for v in caches.picks.values()),
            *(
                str(k)
                for part in (caches.responses, caches.retrieval, caches.picks)
                for k in part.digests()
            ),
        ]
    )
    for secret in (OWN_KEY, escaped, "visitor-sentinel", PASSCODE, RAW_IP, SITE_KEY):
        assert secret not in kept, secret
    # The text is not in a key either: a key is a digest.
    assert all(len(str(k)) == 64 for k in caches.responses.digests())
    assert HALO_TEXT not in "".join(str(k) for k in caches.responses.digests())
    # A hit for another visitor's key carries neither key.
    other_key = "another-visitor-key-000000000000"
    again = _post(s, payload, headers={**headers, "authorization": "Bearer " + other_key})
    assert _json(again)["cache"]["response"] == "hit"
    for secret in (OWN_KEY, escaped, other_key, PASSCODE, RAW_IP):
        assert secret.encode() not in again[2], secret


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


def test_the_response_cache_alone_can_be_off(tmp_path: Path, opened: list[int]) -> None:
    s = _live(tmp_path, NER_SITE_CACHE_RESPONSES="0")
    llm = _reader()
    _post(s, {"text": HALO_TEXT}, llm=llm)
    again = _json(_post(s, {"text": HALO_TEXT}, llm=llm))
    assert again["cache"] == {"response": "off", "retrieval_hits": 1, "pick_hits": 0}
    assert len(llm.requests) == 2


def test_the_offline_stand_in_is_cached_too(tmp_path: Path) -> None:
    s = _offline(tmp_path)
    first = _json(_post(s, {"text": KETA_TEXT}))
    assert first["cache"]["response"] == "miss" and first["warnings"]
    assert _json(_post(s, {"text": KETA_TEXT}))["cache"]["response"] == "hit"
    shared = _json(_post(s, {"text": HALO_TEXT}))
    assert shared["cache"] == {"response": "miss", "retrieval_hits": 1, "pick_hits": 0}


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


def test_the_cache_object_is_part_of_the_schema() -> None:
    miss = CacheInfo(response="miss", retrieval_hits=3, pick_hits=2)
    assert miss.model_dump(mode="json") == {"response": "miss", "retrieval_hits": 3, "pick_hits": 2}
    hit = CacheInfo.model_validate(
        {"response": "hit", "age_s": 240, "original": {"timing_ms": {"total": 9}, "usage": {}}}
    )
    assert hit.model_dump(mode="json") == {
        "response": "hit",
        "retrieval_hits": 0,
        "pick_hits": 0,
        "age_s": 240,
        "original": {"timing_ms": {"total": 9}, "usage": {}},
    }
    with pytest.raises(ValueError):
        CacheInfo.model_validate({"response": "maybe"})
    fixtures = PROJECT / "tools" / "fixtures"
    for name in ("ner.json", "ner-cached.json"):
        data = json.loads((fixtures / name).read_text(encoding="utf-8"))
        response = NerResponse.model_validate(data)
        assert response.cache is not None, name
        assert response.model_dump(mode="json") == data, name
    cached = NerResponse.model_validate(json.loads((fixtures / "ner-cached.json").read_text()))
    assert cached.cache is not None and cached.cache.response == "hit"
    assert cached.usage.llm_calls == 0 and set(cached.timing_ms) == {"total"}
    assert cached.cache.original is not None and cached.cache.original["usage"]["llm_calls"] == 1
