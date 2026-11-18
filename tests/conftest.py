from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
E5_MODEL = "intfloat/multilingual-e5-small"


@pytest.fixture(autouse=True)
def _site_process_state(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """The NER site keeps a little state per warm instance: that Claude is unusable
    (``fallback.block_claude``), what the key checks found, the settings and the visitor
    limiter. A test that drives a chain into ``balance_exhausted`` would otherwise start
    every later test on the OpenRouter step. The limiter's database is also closed off:
    a test that wants one gives ``http.make_limits_executor`` a fake. Only the site's
    tests pay for the import."""
    if not request.module.__name__.rsplit(".", 1)[-1].startswith("test_site_"):
        yield
        return
    from ner_site import fallback, service
    from ner_site import http as site_http

    def no_database(database_url: str) -> Any:
        raise AssertionError("a site test reached for the visitor limiter's real database")

    monkeypatch.setattr(site_http, "make_limits_executor", no_database)

    if request.module.__name__.rsplit(".", 1)[-1] == "test_site_http":
        # Even a regression admitting a refused request cannot send a chat to a real host.
        import httpx
        import httpx2

        async def no_chat_transport(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("an HTTP test reached a live async HTTP transport")

        for library in (httpx, httpx2):
            monkeypatch.setattr(
                library.AsyncHTTPTransport, "handle_async_request", no_chat_transport
            )

    def clear() -> None:
        fallback.reset()
        service.reset_key_check()
        site_http.reset()

    clear()
    yield
    clear()


def skip_if_loader_missing(target_kind: str) -> None:
    """Skip when the job's target loader needs an extra that is not installed."""
    if target_kind == "owl":
        pytest.importorskip("rdflib")


def load_encoder_or_skip(model: str) -> Any:
    """A real SentenceTransformerEncoder, or a skip when the extra or the model is missing.

    `_env.sh` runs HuggingFace offline, so a machine with xwalk-match[dense] but no cached model
    raises OSError on load; that is a missing prerequisite, not a failing test.
    """
    pytest.importorskip("sentence_transformers")
    from xwalk.retrieval.dense import SentenceTransformerEncoder

    try:
        return SentenceTransformerEncoder(model)
    except OSError as exc:
        pytest.skip(
            f"{model} is not in the HuggingFace cache ({exc.__class__.__name__}); pull it once "
            f'with HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from '
            f"xwalk.retrieval.dense import SentenceTransformerEncoder as E; E('{model}')\""
        )


def e5_retriever(limit: int) -> dict[str, Any]:
    """The multilingual E5 dense block the decider jobs carry beside BM25 (A2).

    Needs xwalk-match[dense] and the model in the HF cache: `_env.sh` runs offline, so pull it
    once with HF_HUB_OFFLINE=0 via `SentenceTransformerEncoder("intfloat/multilingual-e5-small")`.
    """
    return {
        "kind": "dense",
        "name": "e5",
        "model": E5_MODEL,
        "limit": limit,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    }


@pytest.fixture
def targets_csv() -> Path:
    return FIXTURES / "targets_tiny.csv"


@pytest.fixture
def sources_csv() -> Path:
    return FIXTURES / "sources_tiny.csv"


@pytest.fixture
def targets_eval_csv() -> Path:
    """~50-row target collection.

    `targets_tiny.csv` stays at 5 rows because Phase 1's ranking assertions are
    verified against it. This one exists so evaluation work has a population:
    near-misses (anomers, related sugars, the methylxanthine family) make the
    retrieval-ceiling decomposition mean something.
    """
    return FIXTURES / "targets_eval.csv"
