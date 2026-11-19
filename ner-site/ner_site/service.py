"""Settings from the environment, the LLM factory, and one request end to end.

Binding: docs/ner-site/PLAN.md sections 3.2 (environment), 3.3 (signatures) and 7
(security). ``run`` validates the request, makes the one extraction call, maps the
entities and assembles a :class:`NerResponse`; every failure it knows about leaves as a
:class:`ServiceError` with a code from section 3.1. ``ner_site.http`` turns those into
responses.

Since 18 November 2026 (docs/superpowers/specs/
2026-11-18-claude-provider-open-access-design.md sections 4 to 6) Claude on the
operator's Anthropic key leads the model lists, and a request for a Claude model runs
on a fallback chain (:func:`make_chain`) that ``run`` reports on: the model that
answered, every step-down, and a sentence for the visitor about each.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from xwalk.decide.base import DecisionClient
from xwalk.decide.budget import BudgetedDecider
from xwalk.decide.fake import FakeDecider
from xwalk.decide.jev import JevClient
from xwalk.fingerprint import redact_url
from xwalk.llm.base import (
    LLMCapabilities,
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    failure_usage,
)
from xwalk.llm.budget import BudgetedLLM, CallBudget, CallLimitExceeded
from xwalk.llm.openai_compat import OpenAICompatClient
from xwalk.records import Usage as XwalkUsage

from ner_site import providers
from ner_site.cache import ON as CACHE_ON
from ner_site.cache import Caches, CacheSettings
from ner_site.claude import (
    CLAUDE_MODELS,
    DISPLAY_NAMES,
    EFFORTS,
    EXTRACT_MAX_TOKENS,
    HAIKU,
    OPUS,
    SELECT_MAX_TOKENS,
    SONNET,
    ClaudeClient,
    is_claude,
    key_status,
)
from ner_site.extract import MAX_TOKENS, TEMPERATURE, extract, render_default_prompt
from ner_site.fallback import ChainView, FallbackLLM, Step
from ner_site.indexes import (
    NEON_DIMENSIONS,
    NEON_SCHEMA,
    Backend,
    Catalog,
    OntologyIndex,
    summarise_status,
)
from ner_site.limits import Allowance
from ner_site.mapping import OTHER_NOTE, MapStats, map_entities
from ner_site.ontologies import FULL_ROUTES, ROUTES
from ner_site.schema import (
    CacheInfo,
    FallbackNote,
    Limits,
    Mode,
    NerRequest,
    NerResponse,
    RetrieverStatus,
    ServiceError,
    Usage,
)
from ner_site.select import (
    JEV,
    JEV_MODEL,
    JEV_RETRIES,
    JEV_TIMEOUT_S,
    JEV_URL,
    RETRIEVAL_ONLY,
    SelectStats,
    select_entities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# A model id starting with "claude-" is called on the operator's Anthropic key
# (ner_site.claude); anything else goes through OpenRouter. Claude leads both lists; an
# id is only offered while the key it needs is set (Settings.offered_models).
DEFAULT_FALLBACK_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_MODELS = (OPUS, SONNET, DEFAULT_FALLBACK_MODEL, "openai/gpt-6-luna")
# What may pick the match among an entity's retrieved candidates (ner_site.select), after
# "retrieval" (the fused ranking alone), which is always offered. The OpenRouter ones
# answered live requests on 4 November 2026 (cheap, recent, structured output); DeepSeek
# V4.1 Flash returned non-JSON once and Qwen3.7 Flash had no endpoint under the account's
# data policy, so they are left out.
DEFAULT_SELECTORS = (OPUS, SONNET, JEV, DEFAULT_FALLBACK_MODEL, "openai/gpt-6-luna")
# Names for the visitor's notices; a model without one is shown by its id.
OPENROUTER_NAMES = {
    DEFAULT_FALLBACK_MODEL: "Gemini 3.1 Flash Lite",
    "openai/gpt-6-luna": "GPT-6 Luna",
}
SELECTOR_CALLS_PER_ENTITY = 4  # Jev: one screen chunk, choose, gate, and one spare
DEFAULT_CACHE_DIR = Path("/tmp/xwalk-ner-site")
OFFLINE_MODEL = "offline"

# Sent to OpenRouter for its app attribution (https://openrouter.ai/docs/app-attribution).
SITE_URL = "https://github.com/jan3657/xwalk"
SITE_TITLE = "xwalk NER demo"
LLM_TIMEOUT_S = 60.0  # one HTTP request
LLM_MAX_RETRIES = 2  # so at most 3 requests, the same as the call budget
# One Claude extraction request. Shorter than LLM_TIMEOUT_S because a Claude model that
# does not answer in time is left at once for the next step (ner_site.fallback, reason
# "timeout"): two such waits, Opus then Sonnet, must fit inside EXTRACT_TIMEOUT_S, or a
# hung Opus would end the request before Sonnet was ever tried. Selection calls keep
# LLM_TIMEOUT_S; the request deadline bounds that step.
CLAUDE_EXTRACT_TIMEOUT_S = 40.0

# Wall-clock bounds inside the function's 120 s maxDuration (vercel.json): the whole
# extraction step, retries included; then the whole mapping step gets what is left of
# REQUEST_DEADLINE_S (at least MIN_MAP_BUDGET_S), and each retriever call within it at
# most MAP_RETRIEVER_TIMEOUT_S. Worst case: 90 s extracting, 10 s mapping, plus
# mapping.LOCAL_GRACE_S for BM25 only, so a request ends by about 105 s.
EXTRACT_TIMEOUT_S = 90.0
MAP_RETRIEVER_TIMEOUT_S = 30.0
REQUEST_DEADLINE_S = 100.0
MIN_MAP_BUDGET_S = 5.0
# The selection step after mapping gets what is left of REQUEST_DEADLINE_S, at least
# MIN_SELECT_BUDGET_S; worst case the request then ends by about 110 s.
MIN_SELECT_BUDGET_S = 5.0

MAX_ERROR_DETAIL = 300
MIN_SCRUBBED_CHARS = 8
# The stats passcode guards the operator's numbers on a public site, and nothing in
# front of ``/api/stats`` limits how often it may be guessed: only a generated secret
# will do, so anything shorter than ``openssl rand -hex 24`` at half its length is
# refused (site analytics design, section 3.1 and "As built").
MIN_STATS_PASSCODE_CHARS = 24

# (env name, Limits field, lowest, highest). A value outside the range is a settings
# error, not clamped: an operator who typed 500000 should learn that it was refused.
# The highest values keep a request comfortably inside the 64 KB body cap.
_LIMIT_ENV: tuple[tuple[str, str, int, int], ...] = (
    ("NER_SITE_MAX_CHARS", "max_chars", 1, 20000),
    ("NER_SITE_MAX_ENTITIES", "max_entities", 1, 100),
    ("NER_SITE_MAX_PROMPT_CHARS", "max_prompt_chars", 1, 16000),
)
# The two original hourly limits keep their upper bound. New allowances allow 10000.
_ALLOWANCE_ENV = (
    ("NER_SITE_OPUS_PER_HOUR", "opus_per_hour", 1000),
    ("NER_SITE_SONNET_PER_HOUR", "sonnet_per_hour", 1000),
    ("NER_SITE_OPUS_PER_DAY", "opus_per_day", 10000),
    ("NER_SITE_SONNET_PER_DAY", "sonnet_per_day", 10000),
    ("NER_SITE_OWN_PER_HOUR", "own_per_hour", 10000),
    ("NER_SITE_OWN_PER_DAY", "own_per_day", 10000),
)
_OWN_MODELS = {
    "anthropic": (*CLAUDE_MODELS, HAIKU),
    "openai": ("gpt-6-luna",),
    "openrouter": (DEFAULT_FALLBACK_MODEL, "openai/gpt-6-luna"),
}
# (env name, CacheSettings field, highest). 0 turns that cache off (for the time limit:
# all three). The highest values keep a warm instance's memory within reason: a stored
# response is about 100 KB, the candidates of one mention about 40 KB, a pick under 1 KB.
_CACHE_ENV: tuple[tuple[str, str, int], ...] = (
    ("NER_SITE_CACHE_RESPONSES", "responses", 2000),
    ("NER_SITE_CACHE_RETRIEVAL", "retrieval", 20000),
    ("NER_SITE_CACHE_PICKS", "picks", 100000),
    ("NER_SITE_CACHE_TTL_S", "ttl_s", 7 * 86400),
)
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
ENV_NAMES = (
    "OPENROUTER_API_KEY",
    "NER_SITE_BASE_URL",
    "NER_SITE_MODELS",
    "NER_SITE_PASSCODE",
    "NER_SITE_OFFLINE",
    "NER_SITE_DENSE_DIR",
    "NER_SITE_CACHE_DIR",
    "NER_SITE_BACKEND",
    "DATABASE_URL",
    "NER_SITE_NEON_SCHEMA",
    "NER_SITE_NEON_DIMENSIONS",
    "NER_SITE_NEON_CATALOG",
    "NER_SITE_SELECTORS",
    "NER_SITE_SELECTOR",
    *(name for name, _, _, _ in _LIMIT_ENV),
    "ANTHROPIC_API_KEY",
    "NER_SITE_CLAUDE_EFFORT",
    "NER_SITE_FALLBACK_MODEL",
    *(name for name, _, _ in _ALLOWANCE_ENV),
    "NER_SITE_OWN_PROVIDERS",
    *(f"NER_SITE_OWN_MODELS_{p.upper()}" for p in providers.OWN_PROVIDERS),
    "NER_SITE_VISITOR_SALT",
    "NER_SITE_STATS_PASSCODE",
    "NER_SITE_CACHE",
    *(name for name, _, _ in _CACHE_ENV),
)


class SettingsError(ValueError):
    """A bad environment value. The message names the variable, never a secret."""


def _listed_own(env: Mapping[str, str]) -> tuple[str, ...]:
    """The provider names ``NER_SITE_OWN_PROVIDERS`` lists, lower-cased, without empty
    items or repeats, not yet checked. Own keys are off unless the operator lists
    providers: unset, empty and a value of only spaces and commas all give none."""
    raw = env.get("NER_SITE_OWN_PROVIDERS") or ""
    return tuple(dict.fromkeys(p for p in (item.strip().lower() for item in raw.split(",")) if p))


def _site_uses_database(env: Mapping[str, str]) -> bool:
    """Whether the index backend or a live allowance needs the URL: everything that
    read it before there was a record of runs."""
    if env.get("NER_SITE_BACKEND", "").strip().lower() == "neon":
        return True
    if env.get("NER_SITE_OFFLINE", "").strip().lower() in _TRUE:
        return False
    return bool(env.get("ANTHROPIC_API_KEY", "").strip() or _listed_own(env))


def uses_database(env: Mapping[str, str]) -> bool:
    """The index backend, a live allowance or the record of runs (which the stats
    passcode turns on) needs the URL, including in the settings cache."""
    return _site_uses_database(env) or bool((env.get("NER_SITE_STATS_PASSCODE") or "").strip())


# One label of a host name: 1 to 63 letters, digits and hyphens, no hyphen at an end.
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


def _usable_database_url(url: str) -> bool:
    """Whether a URL that only the record of runs reads can be used: a postgres:// or
    postgresql:// URL, in plain ASCII without spaces (it is sent in a header as it
    is), whose host is a DNS name and whose port, if it names one, is a number in
    range. A name is labels of 1 to 63 letters, digits and hyphens, no hyphen at an
    end of one, at most 253 characters, with or without one dot at its very end. An
    address is not a name (a Neon host always is one): in brackets, with a dot at its
    end, or in a resolver's short forms. Nothing in brackets is. Never
    raises: the URL parser leaves the port and the brackets to whoever asks for them,
    and anything it stumbles on then means "not usable"."""
    if not url.isascii() or not all(0x21 <= ord(ch) <= 0x7E for ch in url):
        return False
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        parts.port  # noqa: B018 - raises for a port that is not a number in range
    except ValueError:
        return False
    if parts.scheme not in {"postgres", "postgresql"} or not host or len(host) > 254:
        return False
    if "[" in parts.netloc or "]" in parts.netloc:
        return False  # brackets hold an address, or something that is not a host at all
    name = host[:-1] if host.endswith(".") else host
    labels = name.split(".")
    try:
        ipaddress.ip_address(name)
    except ValueError:
        pass
    else:
        return False
    # A name whose last label is a number is read as an IPv4 address by resolvers and
    # URL parsers, in decimal or hexadecimal and with fewer than four parts (127.1,
    # 2130706433, 0x7f.1): not a name either.
    if re.fullmatch(r"[0-9]+|0x[0-9a-f]*", labels[-1]):
        return False
    return len(name) <= 253 and all(_DNS_LABEL.fullmatch(label) for label in labels)


def _own_providers(env: Mapping[str, str]) -> tuple[providers.ProviderInfo, ...]:
    """The own-key providers the operator turned on, with their model lists. A model
    list is read only for a provider that is on, and an empty one counts as unset, so
    a deployment without own keys is not refused over a variable nothing uses."""
    enabled = _listed_own(env)
    if any(p not in providers.OWN_PROVIDERS for p in enabled):
        raise SettingsError("NER_SITE_OWN_PROVIDERS must list anthropic, openai or openrouter")
    infos = []
    for provider in enabled:
        defaults = _OWN_MODELS[provider]
        name = f"NER_SITE_OWN_MODELS_{provider.upper()}"
        raw = (env.get(name) or "").strip()
        ids = tuple(dict.fromkeys(m.strip() for m in raw.split(","))) if raw else defaults
        if any(not m or m in (JEV, RETRIEVAL_ONLY) for m in ids):
            raise SettingsError(
                f"{name} must list non-empty model ids distinct from jev and retrieval"
            )
        if provider == "anthropic" and any(m not in defaults for m in ids):
            raise SettingsError(f"{name} must list models from {', '.join(defaults)}")
        infos.append(
            providers.ProviderInfo(provider, providers.PROVIDER_NAMES[provider], ids, ids[0])
        )
    return tuple(infos)


@dataclass(frozen=True)
class IndexView:
    """What ``indexes.get_indexes`` reads (its ``IndexSettings`` protocol)."""

    cache_dir: Path
    dense_dir: Path | None
    api_key: str | None = field(default=None, repr=False)
    backend: Backend = "bundled"
    database_url: str | None = field(default=None, repr=False)
    neon_schema: str = NEON_SCHEMA
    neon_dimensions: int = NEON_DIMENSIONS
    catalog: Catalog = "samples"


@dataclass(frozen=True)
class Settings:
    """The function's configuration (section 3.2). Secrets are left out of ``repr``."""

    api_key: str | None = field(default=None, repr=False)
    base_url: str = DEFAULT_BASE_URL
    models: tuple[str, ...] = DEFAULT_MODELS
    passcode: str | None = field(default=None, repr=False)
    limits: Limits = field(default_factory=Limits)
    offline: bool = False
    dense_dir: Path | None = PROJECT_ROOT / "indexes"
    cache_dir: Path = DEFAULT_CACHE_DIR
    backend: Backend = "bundled"
    database_url: str | None = field(default=None, repr=False)
    neon_schema: str = NEON_SCHEMA
    neon_dimensions: int = NEON_DIMENSIONS
    catalog: Catalog = "samples"
    selectors: tuple[str, ...] = DEFAULT_SELECTORS
    # The configured default selector; None means the first one that can run.
    selector: str | None = None
    anthropic_api_key: str | None = field(default=None, repr=False)
    claude_effort: str = "low"
    fallback_model: str = DEFAULT_FALLBACK_MODEL
    allowance: Allowance = field(default_factory=Allowance)
    visitor_salt: str | None = field(default=None, repr=False)
    # Set: runs are recorded and GET /api/stats answers to it. Unset: neither exists.
    stats_passcode: str | None = field(default=None, repr=False)
    # The in-memory caches (``ner_site.cache``). Off for a ``Settings`` built in code;
    # ``from_env`` turns them on with the default sizes unless the environment says
    # otherwise (``NER_SITE_CACHE=off``, or a size of 0).
    cache: CacheSettings = CacheSettings()

    # Own keys are off unless the operator lists providers (NER_SITE_OWN_PROVIDERS).
    own_providers: tuple[providers.ProviderInfo, ...] = ()

    def __post_init__(self) -> None:
        if self.offline:
            object.__setattr__(self, "own_providers", ())

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Read section 3.2. Empty values count as unset. Raises :class:`SettingsError`."""

        def get(name: str) -> str:
            return (env.get(name) or "").strip()

        base_url = get("NER_SITE_BASE_URL") or DEFAULT_BASE_URL
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise SettingsError(
                "NER_SITE_BASE_URL must be an http(s) URL such as https://openrouter.ai/api/v1"
            )

        def known_claude(name: str, ids: tuple[str, ...]) -> None:
            unknown = [m for m in ids if is_claude(m) and m not in CLAUDE_MODELS]
            if unknown:
                raise SettingsError(
                    f"{name} names an unknown Claude model ({', '.join(unknown)[:80]}); "
                    f"the Claude ids are {' and '.join(CLAUDE_MODELS)}"
                )

        raw_models = get("NER_SITE_MODELS")
        if raw_models:
            models = tuple(dict.fromkeys(m.strip() for m in raw_models.split(",") if m.strip()))
            if not models:
                raise SettingsError("NER_SITE_MODELS must list at least one model id")
            known_claude("NER_SITE_MODELS", models)
        else:
            models = DEFAULT_MODELS

        effort = get("NER_SITE_CLAUDE_EFFORT").lower() or "low"
        if effort not in EFFORTS:
            raise SettingsError(f"NER_SITE_CLAUDE_EFFORT must be one of {', '.join(EFFORTS)}")
        fallback_model = get("NER_SITE_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODEL
        if is_claude(fallback_model) or fallback_model in (RETRIEVAL_ONLY, JEV):
            # The last step of the chain is the one that does not depend on Anthropic.
            raise SettingsError(
                "NER_SITE_FALLBACK_MODEL must be an OpenRouter model id such as "
                f"{DEFAULT_FALLBACK_MODEL}"
            )
        allowance: dict[str, int] = {}
        for name, attr, maximum in _ALLOWANCE_ENV:
            raw = get(name)
            if not raw:
                continue
            if (
                len(raw) > len(str(maximum))
                or not re.fullmatch(r"[0-9]+", raw)
                or int(raw) > maximum
            ):
                raise SettingsError(f"{name} must be a whole number from 0 to {maximum}")
            allowance[attr] = int(raw)

        passcode = get("NER_SITE_PASSCODE") or None
        if passcode is not None and not passcode.isascii():
            # http.server decodes header values as Latin-1 and browsers refuse other
            # characters in a header, so a non-ASCII passcode could never match.
            raise SettingsError("NER_SITE_PASSCODE must be ASCII (printable, no accents)")
        stats_passcode = get("NER_SITE_STATS_PASSCODE") or None
        if stats_passcode is not None and not stats_passcode.isascii():
            raise SettingsError("NER_SITE_STATS_PASSCODE must be ASCII (printable, no accents)")
        if stats_passcode is not None and len(stats_passcode) < MIN_STATS_PASSCODE_CHARS:
            raise SettingsError(
                f"NER_SITE_STATS_PASSCODE must be at least {MIN_STATS_PASSCODE_CHARS} "
                "characters; generate one, for example with: openssl rand -hex 24"
            )

        offline_raw = get("NER_SITE_OFFLINE").lower()
        if offline_raw not in _TRUE | _FALSE:
            raise SettingsError("NER_SITE_OFFLINE must be 1 or 0 (or true/false, yes/no, on/off)")

        numbers: dict[str, int] = {}
        for name, attr, low, high in _LIMIT_ENV:
            raw = get(name)
            if not raw:
                continue
            if not re.fullmatch(r"[0-9]+", raw) or not low <= int(raw) <= high:
                raise SettingsError(f"{name} must be a whole number from {low} to {high}")
            numbers[attr] = int(raw)

        cache_raw = get("NER_SITE_CACHE").lower()
        if cache_raw and cache_raw not in _TRUE | _FALSE:
            raise SettingsError("NER_SITE_CACHE must be on or off (or 1/0, true/false, yes/no)")
        cache_sizes: dict[str, int] = {}
        for name, attr, maximum in _CACHE_ENV:
            raw = get(name)
            if not raw:
                continue
            if (
                len(raw) > len(str(maximum))
                or not re.fullmatch(r"[0-9]+", raw)
                or int(raw) > maximum
            ):
                raise SettingsError(f"{name} must be a whole number from 0 to {maximum}")
            cache_sizes[attr] = int(raw)
        cache_settings = (
            CacheSettings()
            if cache_raw and cache_raw in _FALSE
            else CacheSettings(**{**asdict(CACHE_ON), **cache_sizes})
        )

        backend = get("NER_SITE_BACKEND").lower() or "bundled"
        if backend not in {"bundled", "neon"}:
            raise SettingsError("NER_SITE_BACKEND must be bundled or neon")
        schema = get("NER_SITE_NEON_SCHEMA") or NEON_SCHEMA
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", schema):
            raise SettingsError(
                "NER_SITE_NEON_SCHEMA must be a lower-case identifier (letters, digits, _)"
            )
        raw_dims = get("NER_SITE_NEON_DIMENSIONS")
        if raw_dims and (
            not re.fullmatch(r"[0-9]{1,4}", raw_dims) or not 1 <= int(raw_dims) <= 4000
        ):
            raise SettingsError("NER_SITE_NEON_DIMENSIONS must be a whole number from 1 to 4000")
        catalog = get("NER_SITE_NEON_CATALOG").lower() or "samples"
        if catalog not in {"samples", "full"}:
            raise SettingsError("NER_SITE_NEON_CATALOG must be samples or full")
        if catalog == "full" and backend != "neon":
            raise SettingsError("NER_SITE_NEON_CATALOG=full needs NER_SITE_BACKEND=neon")
        own_providers = _own_providers(env)
        database_url = (get("DATABASE_URL") or None) if uses_database(env) else None
        if (
            database_url is not None
            and not _site_uses_database(env)
            and not _usable_database_url(database_url)
        ):
            # Read for the record of runs alone: until the stats passcode was set this
            # deployment did not look at the variable, so a value that cannot be used
            # must not become a settings error that answers 503 on every route. There
            # are then no statistics (GET /api/stats says so, 503), and nothing else
            # changes.
            database_url = None
        if database_url is not None and urlsplit(database_url).scheme not in {
            "postgres",
            "postgresql",
        }:
            raise SettingsError("DATABASE_URL must be a postgres:// or postgresql:// URL")

        raw_selectors = get("NER_SITE_SELECTORS")
        selectors = (
            tuple(dict.fromkeys(m.strip() for m in raw_selectors.split(",") if m.strip()))
            if raw_selectors
            else DEFAULT_SELECTORS
        )
        selectors = tuple(m for m in selectors if m != RETRIEVAL_ONLY)
        known_claude("NER_SITE_SELECTORS", selectors)
        # Checked against the list, not against the keys: a selector whose key is
        # missing is a default that cannot run (default_selector moves on), not a typo.
        selector = get("NER_SITE_SELECTOR") or None
        offered = (RETRIEVAL_ONLY, JEV) if offline_raw in _TRUE else (RETRIEVAL_ONLY, *selectors)
        if selector is not None and selector not in offered:
            raise SettingsError(
                f"NER_SITE_SELECTOR must be one of {', '.join(offered)} (NER_SITE_SELECTORS)"
            )

        dense = get("NER_SITE_DENSE_DIR")
        cache = get("NER_SITE_CACHE_DIR")
        settings = cls(
            api_key=get("OPENROUTER_API_KEY") or None,
            base_url=base_url.rstrip("/"),
            models=models,
            passcode=passcode,
            limits=Limits(**numbers),
            offline=offline_raw in _TRUE,
            dense_dir=Path(dense) if dense else PROJECT_ROOT / "indexes",
            cache_dir=Path(cache) if cache else DEFAULT_CACHE_DIR,
            backend="neon" if backend == "neon" else "bundled",
            database_url=database_url,
            neon_schema=schema,
            neon_dimensions=int(raw_dims) if raw_dims else NEON_DIMENSIONS,
            catalog="full" if catalog == "full" else "samples",
            selectors=selectors,
            selector=selector,
            anthropic_api_key=get("ANTHROPIC_API_KEY") or None,
            claude_effort=effort,
            fallback_model=fallback_model,
            allowance=Allowance(**allowance),
            own_providers=own_providers,
            visitor_salt=get("NER_SITE_VISITOR_SALT") or None,
            stats_passcode=stats_passcode,
            cache=cache_settings,
        )
        if not settings.offered_models:
            raise SettingsError(
                "NER_SITE_MODELS lists no model that the keys that are set can run "
                "(Claude ids need ANTHROPIC_API_KEY, the others OPENROUTER_API_KEY)"
            )
        return settings

    @property
    def mode(self) -> Mode:
        return "offline" if self.offline else "live"

    def _runnable(self, model: str) -> bool:
        """Whether the key `model` needs is set. A Claude id needs the Anthropic key. An
        OpenRouter id (and Jev) is left out only where Claude is all the deployment has:
        with no key at all the lists show what an OpenRouter key would offer, as they
        did before Claude."""
        if is_claude(model):
            return self.anthropic_api_key is not None
        return self.api_key is not None or self.anthropic_api_key is None

    @property
    def offered_models(self) -> tuple[str, ...]:
        """What ``/api/health`` lists and a request may name: the stand-in when offline,
        otherwise the configured models whose key is set."""
        if self.offline:
            return (OFFLINE_MODEL,)
        return tuple(m for m in self.models if self._runnable(m))

    @property
    def default_model(self) -> str:
        return self.offered_models[0]

    @property
    def offered_selectors(self) -> tuple[str, ...]:
        """What may pick the match: ``retrieval`` (no selector) first, then the
        configured selectors whose key is set. Offline, Jev is a token-overlap stand-in
        and no LLM is offered."""
        if self.offline:
            return (RETRIEVAL_ONLY, JEV)
        return (RETRIEVAL_ONLY, *(m for m in self.selectors if self._runnable(m)))

    @property
    def default_selector(self) -> str:
        """``NER_SITE_SELECTOR`` while it is offered, otherwise the first selector that
        can run (``retrieval`` when none can)."""
        offered = self.offered_selectors
        if self.selector in offered:
            return self.selector
        return offered[1] if len(offered) > 1 else RETRIEVAL_ONLY

    @property
    def default_prompt(self) -> str:
        """The extraction prompt used when a request sends none, stating this
        deployment's entity limit; ``/api/health`` serves it."""
        return render_default_prompt(self.limits.max_entities, types=tuple(self.routes))

    @property
    def routes(self) -> dict[str, str]:
        """Entity type -> ontology slug for this deployment's catalog."""
        return FULL_ROUTES if self.catalog == "full" else ROUTES

    @property
    def configured(self) -> bool:
        """Whether ``/api/ner`` can run: offline, or a key to call a provider with
        (OpenRouter's or Anthropic's)."""
        return self.offline or self.api_key is not None or self.anthropic_api_key is not None

    @property
    def cache_scope(self) -> tuple[Any, ...]:
        """What tells this deployment's answers from another's in the same process, for
        ``cache.caches_for``: two settings that differ in any of these never share a
        cached value. The database is named by a digest, and no key or passcode is
        here at all (only whether each key is set)."""
        database = self.database_url or ""
        return (
            self.mode,
            self.base_url,
            self.backend,
            self.catalog,
            self.neon_schema,
            self.neon_dimensions,
            str(self.dense_dir),
            str(self.cache_dir),
            self.limits,
            self.claude_effort,
            self.fallback_model,
            hashlib.sha256(database.encode("utf-8")).hexdigest() if database else "",
            self.api_key is not None,
            self.anthropic_api_key is not None,
        )

    @property
    def claude_offered(self) -> bool:
        """Whether Claude can run on the site's keys and the site allowance applies."""
        return not self.offline and self.anthropic_api_key is not None

    def index_settings(self) -> IndexView:
        """The view the indexes are opened with. Offline mode never spends the key, so
        dense retrieval there reports the key as missing."""
        return IndexView(
            cache_dir=self.cache_dir,
            dense_dir=self.dense_dir,
            api_key=None if self.offline else self.api_key,
            backend=self.backend,
            database_url=self.database_url if self.backend == "neon" else None,
            neon_schema=self.neon_schema,
            neon_dimensions=self.neon_dimensions,
            catalog=self.catalog,
        )

    def secrets(self) -> tuple[str, ...]:
        """Values scrubbed from every response and log line as a last line of defence
        (nothing echoes them in the first place). Values shorter than
        ``MIN_SCRUBBED_CHARS`` are left alone: replacing a passcode like ``ok`` would
        corrupt ordinary JSON, and so short a value cannot be hidden anyway."""
        values: list[str | None] = [
            self.api_key,
            self.anthropic_api_key,
            self.passcode,
            self.database_url,
            self.visitor_salt,
            self.stats_passcode,
        ]
        if self.database_url:
            password = urlsplit(self.database_url).password
            if password:
                values += [password, unquote(password)]
        return tuple(dict.fromkeys(s for s in values if s and len(s) >= MIN_SCRUBBED_CHARS))


def scrub(message: str, settings: Settings, extra_secrets: Sequence[str] = ()) -> str:
    """``message`` with the key and passcode removed and every URL redacted."""
    for secret in (*extra_secrets, *settings.secrets()):
        message = message.replace(secret, "[redacted]")
    return re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s'\"<>]+", lambda m: redact_url(m[0]), message)


class OpenRouterClient(OpenAICompatClient):
    """`OpenAICompatClient` that switches off OpenRouter's reasoning on every request.

    On real abstracts (4 November 2026) GPT-6 Luna and Claude Haiku 5.5 spent the whole
    2048-token output budget on hidden reasoning and returned no answer at all
    (``finish_reason: length``, ``reasoning_tokens: 2048``); with ``reasoning:
    {enabled: false}`` they answer in a few hundred tokens. Only sent to OpenRouter:
    other OpenAI-compatible endpoints do not know the field."""

    def _body(self, request: LLMRequest, *, structured: bool) -> dict[str, Any]:
        body = super()._body(request, structured=structured)
        body.setdefault("reasoning", {"enabled": False})
        return body


def provider_profile(model: str) -> str:
    """The capability profile for `model`. Anthropic models are asked for JSON in the
    prompt, not with a strict json_schema: with reasoning off, Claude Haiku 5.5 under
    the schema returned one entity with zero offsets; in the prompt, all of them."""
    return "anthropic-compat" if model.startswith("anthropic/") else "openai"


def _client(settings: Settings, model: str, **kw: Any) -> OpenAICompatClient:
    host = urlsplit(settings.base_url).hostname or ""
    openrouter = host == "openrouter.ai" or host.endswith(".openrouter.ai")
    kind = OpenRouterClient if openrouter else OpenAICompatClient
    return kind(
        settings.base_url,
        model,
        api_key=settings.api_key,
        profile=provider_profile(model),
        extra_headers={"HTTP-Referer": SITE_URL, "X-Title": SITE_TITLE},
        **kw,
    )


def make_llm(
    settings: Settings,
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    backoff_base: float = 1.0,
) -> BudgetedLLM:
    """``BudgetedLLM(OpenAICompatClient(...), CallBudget(limit=llm_call_cap))``.

    ``transport`` and ``backoff_base`` exist for tests (MockTransport, no sleeping).
    The caller closes it (``await llm.aclose()``)."""
    client = _client(
        settings,
        model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=LLM_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
        backoff_base=backoff_base,
        transport=transport,
    )
    return BudgetedLLM(client, CallBudget(limit=settings.limits.llm_call_cap))


def make_select_llm(
    settings: Settings, model: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> BudgetedLLM:
    """The LLM selector's client: one call per routed entity, capped at one per entity
    allowed (no retries beyond the client's own). The caller closes it."""
    client = _client(
        settings,
        model,
        temperature=0.0,
        max_tokens=600,
        timeout=LLM_TIMEOUT_S,
        max_retries=1,
        transport=transport,
    )
    return BudgetedLLM(client, CallBudget(limit=settings.limits.max_entities * 2))


def make_decider(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
) -> DecisionClient:
    """Jev through OpenRouter with the site's key, capped per request; offline, the
    token-overlap stand-in. The caller closes it when it has ``aclose``."""
    if settings.offline:
        return FakeDecider(model="offline-jev")
    client = JevClient(
        JEV_URL,
        JEV_MODEL,
        api_key=settings.api_key,
        timeout=JEV_TIMEOUT_S,
        max_retries=JEV_RETRIES,
        transport=transport,
    )
    budget = CallBudget(limit=settings.limits.max_entities * SELECTOR_CALLS_PER_ENTITY)
    return BudgetedDecider(client, budget)


# --- Claude and the fallback chain (spec 2026-11-18, sections 3.2 and 5) ------------

EXTRACT, SELECT = "extract", "select"
# (settings, model, purpose) -> the unbudgeted client for one Claude model and one
# purpose ("extract" or "select"). Tests pass a scripted one; nothing else should.
ClaudeFactory = Callable[["Settings", str, str], LLMClient]
LLMMaker = Callable[["Settings", str], LLMClient]


def make_claude(settings: Settings, model: str, purpose: str) -> LLMClient:
    """One Claude model on the operator's key. The token limit is the adapter's own per
    purpose (thinking counts against it), not the engine's 2048 / 600; an extraction
    request has the shorter ``CLAUDE_EXTRACT_TIMEOUT_S``."""
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    return ClaudeClient(
        model,
        api_key=settings.anthropic_api_key,
        max_tokens=EXTRACT_MAX_TOKENS if purpose == EXTRACT else SELECT_MAX_TOKENS,
        effort=settings.claude_effort,
        timeout=CLAUDE_EXTRACT_TIMEOUT_S if purpose == EXTRACT else LLM_TIMEOUT_S,
    )


class StagedLLM:
    """One step of a chain: one model with a client per stage, each built when first used.

    A chain serves the whole request, but extraction and selection are configured
    differently on every provider (token limit, retries, call budget; on OpenRouter
    exactly what :func:`make_llm` and :func:`make_select_llm` set). ``stage`` says which
    client answers; :func:`enter_stage` moves a chain and its steps together. Building
    late means a request that stays on Opus never opens a connection pool for Sonnet or
    for OpenRouter."""

    def __init__(self, model: str, makers: Mapping[str, Callable[[], LLMClient]]) -> None:
        if not makers:
            raise ValueError("a staged step needs at least one stage")
        self._model = model
        self._makers = dict(makers)
        self._clients: dict[str, LLMClient] = {}
        self.stage = next(iter(self._makers))

    def client(self, stage: str | None = None) -> LLMClient:
        """The client of `stage` (the current one by default), built on first use. A
        stage this step was not given is served by its first one."""
        name = stage or self.stage
        if name not in self._makers:
            name = next(iter(self._makers))
        if name not in self._clients:
            self._clients[name] = self._makers[name]()
        return self._clients[name]

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(self._makers)

    @property
    def built(self) -> tuple[LLMClient, ...]:
        """The clients built so far, of every stage (``select.dispatched`` counts the
        requests they sent)."""
        return tuple(self._clients.values())

    @property
    def model(self) -> str:
        return self._model

    @property
    def fingerprint(self) -> str:
        return self.client().fingerprint

    @property
    def capabilities(self) -> LLMCapabilities:
        return self.client().capabilities

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return await self.client().complete(request)

    async def aclose(self) -> None:
        for client in self._clients.values():
            close = getattr(client, "aclose", None)
            if callable(close):
                await close()


def enter_stage(llm: object, stage: str) -> None:
    """Tell a chain, and each staged step in it, which stage the next calls belong to
    (what a step-down is recorded under, and which client answers). Anything that is not
    a chain is left alone."""
    if not isinstance(llm, FallbackLLM):
        return
    llm.stage = stage
    for step in llm.steps:
        if isinstance(step.client, StagedLLM):
            step.client.stage = stage


def chain_models(settings: Settings, wanted: str) -> tuple[str, ...]:
    """The models a request for the Claude model `wanted` may run on, best first: the
    Claude models from `wanted` down, then the OpenRouter fallback model when there is
    an OpenRouter key (without one the chain ends at Sonnet)."""
    if wanted not in CLAUDE_MODELS:
        raise ValueError(f"{wanted!r} is not a Claude model of this site")
    models = CLAUDE_MODELS[CLAUDE_MODELS.index(wanted) :]
    return (*models, settings.fallback_model) if settings.api_key else models


def make_chain(
    settings: Settings,
    wanted: str,
    *,
    start: int = 0,
    purposes: Sequence[str] = (EXTRACT, SELECT),
    claude_factory: ClaudeFactory | None = None,
    llm_factory: LLMMaker | None = None,
    select_factory: LLMMaker | None = None,
) -> FallbackLLM:
    """The fallback chain for one request that asked for the Claude model `wanted`,
    started at step `start` of :func:`chain_models`.

    Every step has one client per purpose in `purposes`, built when the chain first
    reaches it. A Claude client is wrapped in its own ``BudgetedLLM`` with the limit the
    OpenRouter client of that purpose has: the request's call cap for extraction, two
    calls per entity allowed for selection. The last step is :func:`make_llm` /
    :func:`make_select_llm` for the fallback model, so on OpenRouter a request behaves
    as it does without a chain. The factories exist for tests. The caller closes the
    chain."""
    build_claude = claude_factory or make_claude
    makers: dict[str, LLMMaker] = {
        EXTRACT: llm_factory or make_llm,
        SELECT: select_factory or make_select_llm,
    }
    caps = {EXTRACT: settings.limits.llm_call_cap, SELECT: settings.limits.max_entities * 2}

    def claude(model: str, purpose: str) -> Callable[[], LLMClient]:
        return lambda: BudgetedLLM(
            build_claude(settings, model, purpose), CallBudget(limit=caps[purpose])
        )

    def openrouter(model: str, purpose: str) -> Callable[[], LLMClient]:
        return lambda: makers[purpose](settings, model)

    steps: list[Step] = []
    for model in chain_models(settings, wanted):
        label = "opus" if model == OPUS else "sonnet" if model == SONNET else "other"
        make = claude if is_claude(model) else openrouter
        steps.append(Step(label, StagedLLM(model, {p: make(model, p) for p in purposes})))
    chain = FallbackLLM(steps, start=start)
    enter_stage(chain, purposes[0])
    return chain


def display_name(model: str) -> str:
    """A model as a notice names it. An id without a name (a custom fallback model, or
    the Claude id a server-side fallback answered with) is shown as it is."""
    if is_claude(model):
        return DISPLAY_NAMES.get(model, model)
    return f"{OPENROUTER_NAMES.get(model, model)} through OpenRouter"


BLOCK_TEXT = {"balance_exhausted": "balance exhausted", "key_refused": "key refused"}
FOLLOWS_EXTRACTION = "follows_extraction"
_WHY = {
    "rate_limited": "{model} is rate limited",
    "overloaded": "{model} is overloaded",
    "transient": "{model} did not answer",
    "timeout": "{model} did not answer in time",
    "refused": "{model} declined this text",
    "call_limit": "{model} reached its call limit for this request",
    "opus_allowance_used": "Your Opus allowance for this hour is used",
    "limits_unavailable": "Visitor limits are unavailable",
}


def notice(
    note: FallbackNote,
    *,
    last: bool = True,
    picks: Sequence[str] = (),
    allowance_window: str = "hour",
    claude_blocked: str | None = None,
) -> str:
    """The sentence a visitor reads about one departure from the model they asked for.

    ``last`` is whether the request went no further in that part of it. Only then does
    the sentence say what ran on the model the departure led to; an earlier one says
    "moved on to", because nothing may have run there. ``claude_blocked`` is why Claude
    was known to be unusable when the request was admitted (``fallback.claude_block`` as
    the caller read it then), for a ``claude_unavailable`` departure; this function
    does not read the process's block itself, which may have changed since."""
    target = display_name(note.to)
    if picks:
        which = "the pick" if len(picks) == 1 else "the picks"
        then = f"{which} for {', '.join(picks)} ran on {target}"
    elif not last:
        then = f"moved on to {target}"
    elif note.reason == FOLLOWS_EXTRACTION:
        then = f"the picks ran on {target}"
    elif note.stage == SELECT:
        then = f"the remaining picks ran on {target}"
    else:
        then = f"this ran on {target}"
    if note.reason == FOLLOWS_EXTRACTION:
        return f"Selection uses the model that extracted; {then}."
    blocked = note.reason if note.reason in BLOCK_TEXT else None
    if note.reason == "claude_unavailable":
        blocked = claude_blocked
        if blocked is None:
            return f"Claude is unavailable; {then}."
    if blocked is not None:
        # Named as Claude when the request left Claude altogether, else as the model.
        who = display_name(note.from_) if is_claude(note.to) else "Claude"
        return f"{who} is unavailable ({BLOCK_TEXT.get(blocked, blocked)}); {then}."
    why = _WHY.get(note.reason, "{model} could not be used")
    if note.reason == "opus_allowance_used" and allowance_window == "day":
        why = "Your Opus allowance for today is used"
    return f"{why.format(model=display_name(note.from_))}; {then}."


def _notices(
    departures: Sequence[FallbackNote],
    *,
    extracted_on_chain: bool,
    server_picks: Mapping[int, list[str]],
    allowance_window: str,
    claude_blocked: str | None = None,
) -> list[str]:
    """A sentence per departure, in order. The departures fall into two parts, those
    that decided where the extraction ran and those that decided where the picks ran,
    and in each only the last one says what ran where it led (:func:`notice`). A
    departure at admission belongs to the extraction, or to the selection when only
    the selector runs on a chain."""

    def part(note: FallbackNote) -> str:
        return SELECT if note.stage == SELECT or not extracted_on_chain else EXTRACT

    final = {part(note): position for position, note in enumerate(departures)}
    return [
        notice(
            note,
            last=final[part(note)] == position,
            picks=server_picks.get(position, ()),
            allowance_window=allowance_window,
            claude_blocked=claude_blocked,
        )
        for position, note in enumerate(departures)
    ]


def _notes(chain: FallbackLLM, since: int = 0) -> list[FallbackNote]:
    return [
        FallbackNote(stage=f.stage, from_=f.from_model, to=f.to_model, reason=f.reason)
        for f in chain.fallbacks[since:]
    ]


def _served_elsewhere(view: ChainView) -> FallbackNote | None:
    """Only the SDK's signal distinguishes a refusal fallback from an echoed model id."""
    if view.served_by_fallback and view.asked and view.answered_by:
        return FallbackNote(stage=EXTRACT, from_=view.asked, to=view.answered_by, reason="refused")
    return None


# /api/health asks OpenRouter whether the key still works (GET /api/v1/key: free, no
# model call), remembering the answer; an expired key otherwise shows "llm ok" while every
# POST /api/ner fails with 502 (4 November 2026).
KEY_CHECK_URL = "https://openrouter.ai/api/v1/key"
KEY_CHECK_TIMEOUT_S = 5.0
KEY_OK_TTL_S = 300.0
KEY_REFUSED_TTL_S = 60.0
_KEY_CACHE: dict[str, tuple[float, RetrieverStatus]] = {}
_KEY_LOCK = threading.Lock()


def reset_key_check() -> None:
    with _KEY_LOCK:
        _KEY_CACHE.clear()


def _key_status(settings: Settings, transport: httpx.BaseTransport | None) -> RetrieverStatus:
    try:
        with httpx.Client(timeout=KEY_CHECK_TIMEOUT_S, transport=transport) as client:
            response = client.get(
                KEY_CHECK_URL, headers={"Authorization": f"Bearer {settings.api_key}"}
            )
    except httpx.HTTPError as exc:
        return RetrieverStatus(
            status="ok", reason=f"key not checked (OpenRouter unreachable: {type(exc).__name__})"
        )
    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    if response.status_code in (401, 403):
        error = payload.get("error") if isinstance(payload, dict) else None
        detail = error.get("message") if isinstance(error, dict) else None
        reason = (
            f"OPENROUTER_API_KEY was refused by OpenRouter (HTTP {response.status_code}"
            + (f": {str(detail)[:120]}" if detail else "")
            + "); POST /api/ner answers 502 until the key is replaced"
        )
        return RetrieverStatus(status="unavailable", reason=scrub(reason, settings))
    if response.status_code != 200:
        return RetrieverStatus(
            status="ok", reason=f"key not checked (OpenRouter answered HTTP {response.status_code})"
        )
    data = payload.get("data") if isinstance(payload, dict) else None
    remaining = data.get("limit_remaining") if isinstance(data, dict) else None
    if isinstance(remaining, (int, float)) and remaining <= 0:
        return RetrieverStatus(
            status="unavailable",
            reason="the OpenRouter key's spending limit is used up; raise it or replace the key",
        )
    return RetrieverStatus(status="ok")


def check_key(
    settings: Settings, *, transport: httpx.BaseTransport | None = None
) -> RetrieverStatus | None:
    """Whether OpenRouter accepts ``OPENROUTER_API_KEY`` (None without a key). A refused
    or spent key is ``unavailable`` with OpenRouter's reason; a check that could not reach
    OpenRouter is ``ok`` with a "not checked" reason, never an outage. Remembered for
    ``KEY_OK_TTL_S`` (``KEY_REFUSED_TTL_S`` when refused), per key."""
    if not settings.api_key:
        return None
    slot = hashlib.sha256(settings.api_key.encode()).hexdigest()
    now = time.monotonic()
    with _KEY_LOCK:
        cached = _KEY_CACHE.get(slot)
        if cached is not None and cached[0] > now:
            return cached[1]
    status = _key_status(settings, transport)
    ttl = KEY_OK_TTL_S if status.status == "ok" else KEY_REFUSED_TTL_S
    with _KEY_LOCK:
        _KEY_CACHE[slot] = (now + ttl, status)
    return status


def check_claude_key(settings: Settings, *, client: Any | None = None) -> RetrieverStatus | None:
    """Whether Anthropic accepts ``ANTHROPIC_API_KEY`` (None without a key): one free
    Models API call (``claude.key_status``), remembered like :func:`check_key`. Only a
    refused key is ``unavailable``; a check that could not be completed (Anthropic
    unreachable, or any other HTTP error) is ``ok`` with a "key not checked" reason,
    never an outage, and is remembered only for ``KEY_REFUSED_TTL_S``. The reason is a
    fixed phrase and never quotes the upstream. ``client`` is an SDK-shaped object for
    tests."""
    if not settings.anthropic_api_key:
        return None
    slot = "claude:" + hashlib.sha256(settings.anthropic_api_key.encode()).hexdigest()
    now = time.monotonic()
    with _KEY_LOCK:
        cached = _KEY_CACHE.get(slot)
        if cached is not None and cached[0] > now:
            return cached[1]
    state, reason = key_status(
        settings.anthropic_api_key, client=client, timeout=KEY_CHECK_TIMEOUT_S
    )
    status = RetrieverStatus(status="ok" if state == "ok" else "unavailable", reason=reason)
    # Long only for a key that was accepted; refused or not checked is asked again soon.
    ttl = KEY_OK_TTL_S if status.status == "ok" and reason is None else KEY_REFUSED_TTL_S
    with _KEY_LOCK:
        _KEY_CACHE[slot] = (now + ttl, status)
    return status


def _bad(code: str, message: str) -> ServiceError:
    return ServiceError(code, message, status=400)


def _shown(value: str, limit: int = 80) -> str:
    return repr(value if len(value) <= limit else value[:limit] + "...")


def validate_request(request: NerRequest, settings: Settings) -> NerRequest:
    """The request with defaults resolved, or a 400 :class:`ServiceError` (section 3.1).

    Lengths are Python ``len``: Unicode code points, the unit of every offset."""
    limits = settings.limits
    if not request.text.strip():
        raise _bad("missing_text", "text is empty; paste some text to annotate")
    if len(request.text) > limits.max_chars:
        raise _bad(
            "text_too_long",
            f"text is {len(request.text)} characters; the limit is {limits.max_chars}",
        )
    prompt = (
        request.prompt if request.prompt and request.prompt.strip() else settings.default_prompt
    )
    if len(prompt) > limits.max_prompt_chars:
        raise _bad(
            "prompt_too_long",
            f"prompt is {len(prompt)} characters; the limit is {limits.max_prompt_chars}",
        )
    if not 1 <= request.top_k <= limits.top_k_max:
        raise _bad("bad_top_k", f"top_k must be from 1 to {limits.top_k_max}")
    own = request.provider not in (None, providers.SITE)
    if own:
        info = next((p for p in settings.own_providers if p.id == request.provider), None)
        if info is None:
            raise _bad(
                "bad_credential", "own API keys are not available for that provider on this site"
            )
        models, selectors = info.models, (RETRIEVAL_ONLY, *info.models)
        default_model, default_selector = info.default_model, RETRIEVAL_ONLY
    else:
        models, selectors = settings.offered_models, settings.offered_selectors
        default_model, default_selector = settings.default_model, settings.default_selector
    model = request.model if request.model is not None else default_model
    if model not in models:
        label = "model" if own else f"model {_shown(model)}"
        raise _bad("unknown_model", f"{label} is not offered; choose one of {', '.join(models)}")
    selector = request.selector if request.selector is not None else default_selector
    if selector not in selectors:
        label = "selector" if own else f"selector {_shown(selector)}"
        raise _bad(
            "unknown_selector", f"{label} is not offered; choose one of {', '.join(selectors)}"
        )
    return NerRequest(
        text=request.text,
        prompt=prompt,
        model=model,
        top_k=request.top_k,
        selector=selector,
        provider=request.provider,
    )


def wire_usage(
    usage: XwalkUsage, embedding_requests: int, selected: SelectStats | None = None
) -> Usage:
    """Tokens are reported only when every call reported them; otherwise null, never 0.
    No call at all (the offline stand-in) reports no tokens either. The same holds for
    the selector's tokens and the cost its provider reported."""
    known = usage.calls > 0 and usage.unknown_calls == 0
    picked = selected if selected is not None and selected.calls > 0 else None
    return Usage(
        llm_calls=usage.calls,
        prompt_tokens=usage.prompt_tokens if known else None,
        completion_tokens=usage.completion_tokens if known else None,
        embedding_requests=embedding_requests,
        selector_calls=0 if selected is None else selected.calls,
        selector_prompt_tokens=None if picked is None else picked.prompt_tokens,
        selector_completion_tokens=None if picked is None else picked.completion_tokens,
        selector_cost_usd=None if picked is None else picked.reported_cost,
    )


def retrieval_status(
    indexes: Mapping[str, OntologyIndex], settings: Settings
) -> dict[str, RetrieverStatus]:
    """``summarise_status``; when offline mode is what keeps a configured key from dense
    retrieval, the reason says so instead of claiming the key is missing."""
    summary = summarise_status(indexes)
    for kind, status in summary.items():
        reasons = {
            index.status[kind].reason
            for index in indexes.values()
            if kind in index.status and index.status[kind].status != "ok"
        }
        failing = sum(
            1 for i in indexes.values() if kind not in i.status or i.status[kind].status != "ok"
        )
        # One reason shared by every ontology: say it once, not four times.
        if status.status != "ok" and len(reasons) == 1 and failing == len(indexes) > 1:
            (reason,) = reasons
            if reason:
                summary[kind] = RetrieverStatus(
                    status="unavailable", reason=f"{reason} (all {len(indexes)} ontologies)"
                )
    dense = summary.get("dense")
    if settings.offline and settings.api_key and dense is not None and dense.status != "ok":
        summary["dense"] = RetrieverStatus(
            status="unavailable",
            reason="offline mode (NER_SITE_OFFLINE=1) requests no query embeddings; BM25 only",
        )
    return summary


def _clean(note: str, settings: Settings, extra_secrets: Sequence[str] = ()) -> str:
    """A retriever note fit for a response: secrets and URL userinfo removed, capped."""
    note = scrub(note, settings, extra_secrets)
    return note if len(note) <= MAX_ERROR_DETAIL else note[: MAX_ERROR_DETAIL - 3] + "..."


def _dense_outcome(
    retrieval: dict[str, RetrieverStatus], stats: MapStats, notes: list[str]
) -> None:
    """What dense retrieval did in this request, beyond its open-time status: failing
    for every entity that tried it makes it unavailable; failing for some is a warning
    naming them."""
    dense = retrieval.get("dense")
    tried, failed = stats.dense_entities, stats.dense_failed
    if dense is None or dense.status != "ok" or not tried or not failed:
        return
    if all(eid in failed for eid in tried):
        retrieval["dense"] = RetrieverStatus(
            status="unavailable",
            reason=f"every dense search failed in this request: {failed[tried[0]]}",
        )
        return
    ids = [eid for eid in tried if eid in failed]
    notes.append(
        f"dense search failed for {len(ids)} of {len(tried)} entities ({', '.join(ids)}); "
        "their matches use BM25 only"
    )


def _unavailable_note(kind: str) -> str:
    """The warning of a response whose matches lack one retriever."""
    other = "BM25 only" if kind == "dense" else "dense only"
    return f"{kind} retrieval unavailable; matches use {other}"


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


async def run(
    request: NerRequest,
    *,
    settings: Settings,
    extra_secrets: Sequence[str] = (),
    llm: LLMClient,
    indexes: Mapping[str, OntologyIndex],
    started_at: float | None = None,
    decider: DecisionClient | None = None,
    select_llm: LLMClient | None = None,
    admitted: Sequence[FallbackNote] = (),
    allowance_window: str = "hour",
    claude_blocked: str | None = None,
    caches: Caches | None = None,
) -> NerResponse:
    """One request: validate, extract with ``llm`` (one call), map, select, report.

    With ``caches`` (``ner_site.cache``) the mapping and the selection reuse what this
    instance has already retrieved and picked, and the response says how often
    (``cache``); the whole-response cache is the handler's (``ner_site.http``).

    The selection step runs when the request's selector is not ``retrieval``: Jev with
    ``decider``, any other selector with ``select_llm`` (``ner_site.http`` builds them
    with :func:`make_decider` and :func:`make_select_llm`).

    When ``llm`` or ``select_llm`` is a fallback chain (:func:`make_chain`; one chain may
    be both), the response says what it did: ``model`` is the model whose answer the
    extraction used, ``fallbacks`` lists ``admitted`` (the step-downs decided before any
    call) and then the chain's own, and ``notices`` has a sentence for each
    (``claude_blocked`` is the block reason the caller read when it admitted the
    request, which a ``claude_unavailable`` sentence names). The adapter signals
    server-side refusal fallbacks, which are listed with the reason ``refused``.
    With visitor credentials both stages run on the chosen provider without a chain.

    Calls and tokens are read from one place, the usage each answer carries: a chain's
    answer already includes its failed attempts, and the steps' own ``BudgetedLLM``
    totals count those same calls, so they are never added on top.

    ``started_at`` (``time.monotonic()``) is when the request began, if earlier than
    this call (opening the indexes on a cold start); the mapping deadline counts from it.

    Raises :class:`ServiceError`: 400 codes from :func:`validate_request`, 422 from the
    extraction, 429 ``call_budget_exhausted``, 502 ``provider_error``. With visitor
    credentials a failed call or a timeout is the provider's error
    (``providers.provider_error``); any other exception is not converted."""
    req = validate_request(request, settings)
    assert req.prompt is not None and req.model is not None
    own = req.provider not in (None, providers.SITE)
    chain = llm if not own and isinstance(llm, FallbackLLM) else None
    enter_stage(chain, EXTRACT)
    # A view remembers which model its answer came from; the chain may move on later.
    reader = llm if chain is None else chain.view()
    started = time.monotonic()
    try:
        entities, warnings, usage = await asyncio.wait_for(
            extract(
                reader,
                req.text,
                req.prompt,
                max_entities=settings.limits.max_entities,
                types=(*settings.routes, "other"),
            ),
            timeout=EXTRACT_TIMEOUT_S,
        )
    except ServiceError:
        raise
    except asyncio.CancelledError:
        if not own:
            raise
        assert req.provider is not None
        failure = providers.provider_error(asyncio.TimeoutError(), req.provider)
    except Exception as exc:
        if own:
            # Only what a provider (or the wait for it) did is the provider's failure.
            # Anything else is a fault of this code and leaves as it is, so that the
            # handler logs its type and where it was raised (a 500).
            if not isinstance(exc, (LLMError, asyncio.TimeoutError)):
                raise
            assert req.provider is not None
            failure = providers.provider_error(exc, req.provider)
        elif isinstance(exc, CallLimitExceeded):
            failure = ServiceError(
                "call_budget_exhausted",
                f"the limit of {settings.limits.llm_call_cap} model calls per request was "
                "reached before the model answered; try again later",
                status=429,
            )
        elif isinstance(exc, LLMError):
            detail = scrub(f"{type(exc).__name__}: {exc}", settings)[:MAX_ERROR_DETAIL]
            failure = ServiceError(
                "provider_error", f"the model provider failed ({detail})", status=502
            )
        elif isinstance(exc, asyncio.TimeoutError):
            failure = ServiceError(
                "provider_error",
                f"the model did not answer within {EXTRACT_TIMEOUT_S:g} seconds",
                status=502,
            )
        else:
            raise
        if isinstance(exc, LLMError):
            # What the failed calls are known to have used (a refusal reports its
            # tokens): the handler keeps it for the run's row, never for the answer.
            failure.usage = failure_usage(exc)
    else:
        failure = None
    if failure is not None:
        raise failure
    extracted = time.monotonic()

    stats = MapStats()
    began = started if started_at is None else min(started_at, started)
    budget = max(MIN_MAP_BUDGET_S, REQUEST_DEADLINE_S - (extracted - began))
    results = await map_entities(
        entities,
        indexes,
        top_k=req.top_k,
        timeout=min(MAP_RETRIEVER_TIMEOUT_S, budget),
        deadline=budget,
        stats=stats,
        clean=lambda note: _clean(note, settings, extra_secrets),
        routes=settings.routes,
        cache=None if caches is None else caches.retrieval,
    )
    mapped = time.monotonic()

    selector = req.selector or RETRIEVAL_ONLY
    picked: SelectStats | None = None
    taken = 0 if chain is None else len(chain.fallbacks)  # while extracting
    departures = [*admitted, *(_notes(chain) if chain is not None else ())]
    if chain is not None:
        elsewhere = _served_elsewhere(reader)
        if elsewhere is not None:
            departures.append(elsewhere)
    select_chain = select_llm if not own and isinstance(select_llm, FallbackLLM) else None
    if selector != RETRIEVAL_ONLY:
        if (selector == JEV and decider is None) or (selector != JEV and select_llm is None):
            raise ServiceError("not_configured", f"selector {selector!r} has no client", status=503)
        picked = SelectStats()
        if selector != JEV and select_chain is not None:
            enter_stage(select_chain, SELECT)
            if select_chain is chain and selector not in (req.model, select_chain.model):
                # One chain serves the request, built from the extraction model: a
                # different Claude selector runs where that chain is, and says so.
                departures.append(
                    FallbackNote(
                        stage=SELECT,
                        from_=selector,
                        to=select_chain.model,
                        reason=FOLLOWS_EXTRACTION,
                    )
                )
        results = await select_entities(
            results,
            stats.candidates,
            indexes,
            selector=selector,
            text=req.text,
            stats=picked,
            decider=decider,
            llm=select_llm,
            own_key=own,
            deadline=max(MIN_SELECT_BUDGET_S, REQUEST_DEADLINE_S - (mapped - began)),
            cache=None if caches is None else caches.picks,
            payer=req.provider or providers.SITE,
            clean=lambda note: _clean(note, settings, extra_secrets),
        )
        results = [
            r.model_copy(
                update={
                    "pick": r.pick.model_copy(
                        update={"explanation": _clean(r.pick.explanation, settings, extra_secrets)}
                    )
                }
            )
            if r.pick is not None
            else r
            for r in results
        ]
    selected = time.monotonic()
    if chain is not None:
        departures += _notes(chain, taken)
    if select_chain is not None and select_chain is not chain:
        departures += _notes(select_chain)
    server_picks: dict[int, list[str]] = {}
    if select_chain is not None and picked is not None:
        # Iterate in wire order, independent of the order concurrent picks finished.
        groups: dict[tuple[str, str], list[str]] = {}
        for result in results:
            pair = picked.served_elsewhere.get(result.id)
            if pair:
                groups.setdefault(pair, []).append(result.id)
        for (asked, answered), ids in groups.items():
            server_picks[len(departures)] = ids
            departures.append(
                FallbackNote(stage=SELECT, from_=asked, to=answered, reason="refused")
            )

    retrieval = retrieval_status(indexes, settings)
    notes: list[str] = list(warnings) + stats.warnings
    _dense_outcome(retrieval, stats, notes)
    routed = any(r.ontology is not None for r in results)
    for kind in ("bm25", "dense"):
        status = retrieval.get(kind)
        if routed and status is not None and status.status != "ok":
            notes.append(_unavailable_note(kind))
    timing = {"extract": _ms(extracted - started), "map": _ms(mapped - extracted)}
    if picked is not None:
        timing["select"] = _ms(selected - mapped)
    timing["total"] = _ms(selected - started)
    answered_by = None if own else getattr(reader, "answered_by", None)
    return NerResponse(
        model=answered_by or req.model,
        provider=req.provider or providers.SITE,
        credential_source="visitor" if own else "site",
        mode=settings.mode,
        selector=selector,
        entities=results,
        retrieval=retrieval,
        warnings=[scrub(n, settings, extra_secrets) for n in notes],
        usage=wire_usage(usage, stats.embedding_requests, picked),
        timing_ms=timing,
        requested_model=req.model,
        requested_selector=selector,
        fallbacks=departures,
        notices=list(
            dict.fromkeys(
                scrub(sentence, settings)
                for sentence in _notices(
                    departures,
                    extracted_on_chain=chain is not None,
                    server_picks=server_picks,
                    allowance_window=allowance_window,
                    claude_blocked=claude_blocked,
                )
            )
        ),
        cache=None
        if caches is None
        else CacheInfo(
            response="miss" if caches.responses.enabled else "off",
            retrieval_hits=stats.cache_hits,
            pick_hits=0 if picked is None else picked.cache_hits,
        ),
    )


def cacheable(data: Mapping[str, Any], *, offline: bool = False) -> bool:
    """Whether a 200 response (as JSON data) may be kept in the response cache: only
    when a fresh identical run would be allowed to return the same thing. So never
    with a step-down or a notice, a warning, a pick that failed, a retriever that was
    not ``ok``, or a routed entity that carries a note (a search that failed, timed
    out or was cut off).

    One exception, for the offline stand-in only: it never has dense retrieval (it
    requests no embeddings, by design), says so in every response with one fixed
    warning, and would say the same again. That status and that warning do not keep an
    offline response out; anything else does there too."""
    if data.get("ok") is not True or data.get("fallbacks") or data.get("notices"):
        return False
    allowed = {_unavailable_note("dense")} if offline else set()
    if any(warning not in allowed for warning in data.get("warnings") or ()):
        return False
    for kind, status in (data.get("retrieval") or {}).items():
        if status.get("status") != "ok" and not (offline and kind == "dense"):
            return False
    for entity in data.get("entities") or ():
        pick = entity.get("pick")
        if pick is not None and pick.get("status") == "failed":
            return False
        notes = entity.get("notes") or []
        if entity.get("ontology") is not None and notes:
            return False
        if entity.get("ontology") is None and notes != [OTHER_NOTE]:
            return False
    return True


__all__ = [
    "EXTRACT",
    "SELECT",
    "ClaudeFactory",
    "DEFAULT_BASE_URL",
    "cacheable",
    "DEFAULT_MODELS",
    "ENV_NAMES",
    "OFFLINE_MODEL",
    "IndexView",
    "StagedLLM",
    "Settings",
    "SettingsError",
    "chain_models",
    "check_claude_key",
    "check_key",
    "display_name",
    "enter_stage",
    "make_chain",
    "make_claude",
    "make_decider",
    "make_llm",
    "make_select_llm",
    "reset_key_check",
    "retrieval_status",
    "run",
    "scrub",
    "uses_database",
    "validate_request",
    "wire_usage",
]
