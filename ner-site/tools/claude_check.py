"""Run texts through a deployed (or local) NER site and print what each request did.

For the operator, during the rollout in ``ner-site/README.md`` ("Rollout: Claude by
default and open access"). Nothing calls this tool; it makes requests only when you run
it.

What it costs to run. Every text is one real ``POST /api/ner``: it spends the balance of
the key behind the site (the operator's Anthropic balance when the model or the selector
is Claude, the OpenRouter key otherwise) and it counts against the caller's allowance on
that site (by default 10 requests starting on Opus and then 20 on Sonnet in a clock
hour, 20 and 40 in a UTC day, then 429 until the hour or the day ends). ``--burst N``
sends N such requests at once.

With ``--provider anthropic|openai|openrouter --key-env NAME`` the model calls run on
the API key in the environment variable NAME instead, as a visitor's own key would: THAT
KEY IS BILLED for every model call of the run (the site still pays for the term lookup),
and the requests count against the own-key flood limit (by default 30 an hour and 60 a
day) instead of the allowance. The key is sent to the URL you give, as ``Authorization:
Bearer``, on the ``POST /api/ner`` requests only: use a URL you trust with it. The tool
refuses to send it over plain http to anything but this machine, and sends nothing when
``/api/health`` does not offer that provider.

What it does, and nothing else: one ``GET <url>/api/health`` before and one after, and
one ``POST <url>/api/ner`` per text, to the URL it is given. It follows no redirect,
writes no file, and reads only the text files named on the command line and, with
``--passcode-env NAME`` and ``--key-env NAME``, the environment variables so named (the
passcode is sent as ``X-Passcode``). Neither secret is ever taken from an argument and
neither is ever printed: should the site echo one, it is replaced by ``[redacted]`` in
everything this tool prints.

Usage (standard library only; any Python 3.10 or later)::

    python ner-site/tools/claude_check.py --url http://127.0.0.1:8765 text1.txt text2.txt
    python ner-site/tools/claude_check.py --url https://<deployment> \\
        --passcode-env NER_SITE_PASSCODE --sequential-delay 5 texts/*.txt
    python ner-site/tools/claude_check.py --url https://<deployment> \\
        --selector retrieval --burst 15 short.txt
    python ner-site/tools/claude_check.py --url https://<deployment> \\
        --provider openai --key-env MY_OPENAI_KEY one.txt

It prints the health ``claude``, ``visitor`` and ``own_keys`` blocks, then one line per
text: HTTP status, round-trip seconds, the provider and whose credential the site says
it ran on, the model asked for and the model that extracted, the selector, the entity
count, the picks by status and by the model that made them (and why
any failed), every step-down (``stage:from>to:reason``), the server's ``timing_ms`` and
the token counts; then the totals, the slowest request and the one with the most
entities. Exit code 1 if any POST failed, or if a response says it ran on another
provider or credential than the one asked for; 2 for a usage error.

``--burst N`` instead sends N copies of the first text in parallel and reports how many
started on Opus, on Sonnet, on another model or on the own key, how many were refused
with 429 (the hour used, the day used, the own-key limit, or too many at once: a request
that waited too long for its turn at the visitor's lock, which is what the tail of a
large burst should get), how many with 503 ``limits_unavailable`` (the site could not
count them and so did not run them on its own keys) and how many ran without being
counted, against the balances ``/api/health`` reported just before (``opus_left``,
``sonnet_left`` and ``own_left``, each already capped by what is left of the day). A
start is put under a tier only when health reported the limiter ``on`` before the burst:
otherwise nothing counted it, and it is reported as not counted. Exit code 1 when more
started on a tier than its balance had left (the limiter does not serialise one
visitor's requests), when any request got 503 ``limits_unavailable`` or ran uncounted
(the limiter is failing), when health reported no balance to compare with, when the
limiter stopped answering during an own-key burst, or when a request failed otherwise.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HEALTH_PATH = "/api/health"
NER_PATH = "/api/ner"
REQUESTED_WITH = "xwalk-ner"
USER_AGENT = "xwalk-claude-check"
OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
REQUEST_DEADLINE_S = 100  # ner_site.service.REQUEST_DEADLINE_S
DEFAULT_TIMEOUT_S = 130.0  # Vercel cuts a function off at 120 s
MAX_BURST = 100
MAX_MESSAGE_CHARS = 200
# How the messages of the 429s start (ner_site.http._admit); the code is
# visitor_limit_reached for all of them, and anything else is the hour being used.
TOO_MANY_AT_ONCE = "too many requests at once"  # the visitor's lock was contended
DAY_USED = "you have used today's requests"
OWN_KEY_LIMIT = "too many requests with your own key"
LIMITS_UNAVAILABLE = "limits_unavailable"
# ner_site.providers: who a request can run as, and what a key looks like.
SITE = "site"
OWN_PROVIDERS = ("anthropic", "openai", "openrouter")
_KEY = re.compile(r"[\x21-\x7e]{20,512}")
# Where a key may be sent without TLS: this machine (a local server, the tests).
LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

# What must never be printed (the passcode and the API key of this run), whoever sent
# it back.
_REDACTED: list[str] = []


def _redact(text: str) -> str:
    for secret in _REDACTED:
        text = text.replace(secret, "[redacted]")
    return text


def say(text: str) -> None:
    """Print one line of the report. Everything the tool prints goes through here."""
    print(_redact(text))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so no request leaves the URL the operator gave."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


@dataclass(frozen=True)
class Answer:
    """One HTTP exchange. ``status`` is 0 when no response arrived; ``data`` is the JSON
    object of the body, or None when there was none; ``error`` says why it failed."""

    status: int
    data: dict[str, Any] | None
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.data is not None and self.data.get("ok") is True

    @property
    def code(self) -> str | None:
        """The error code of the site's envelope, when the body is one."""
        error = (self.data or {}).get("error")
        code = error.get("code") if isinstance(error, dict) else None
        return code if isinstance(code, str) else None

    def failure(self) -> str:
        error = (self.data or {}).get("error")
        if isinstance(error, dict):
            message = _redact(str(error.get("message", "")))[:MAX_MESSAGE_CHARS]
            return f"{error.get('code', 'error')}: {message}"
        return self.error or f"HTTP {self.status} without a JSON body"


def _send(request: urllib.request.Request, timeout: float) -> Answer:
    started = time.monotonic()
    status, raw, error = 0, b"", None
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:  # any status outside 2xx, redirects included
        status, raw = exc.code, exc.read()
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        # Refused, unresolved, timed out; or an answer that is not HTTP or was cut short
        # (http.client's own errors are not OSErrors).
        reason = getattr(exc, "reason", None) or exc
        detail = _redact(str(reason))[:MAX_MESSAGE_CHARS]
        error = f"no response ({type(reason).__name__}: {detail})"
    seconds = time.monotonic() - started
    data: dict[str, Any] | None = None
    if raw:
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            error = f"HTTP {status} with a body that is not JSON"
        else:
            data = parsed if isinstance(parsed, dict) else None
    return Answer(status, data, seconds, error)


def get_health(base: str, *, timeout: float) -> Answer:
    request = urllib.request.Request(
        base + HEALTH_PATH, headers={"User-Agent": USER_AGENT}, method="GET"
    )
    return _send(request, timeout)


def post_text(
    base: str,
    text: str,
    *,
    model: str | None,
    selector: str | None,
    passcode: str | None,
    timeout: float,
    provider: str = SITE,
    key: str | None = None,
) -> Answer:
    """One ``POST /api/ner``, as the page sends it: ``provider`` is in the body and the
    key in ``Authorization`` only for a run on an own key."""
    body: dict[str, Any] = {"text": text}
    if model:
        body["model"] = model
    if selector:
        body["selector"] = selector
    if provider != SITE:
        body["provider"] = provider
    headers = {
        "Content-Type": "application/json",
        "X-Requested-With": REQUESTED_WITH,
        "User-Agent": USER_AGENT,
    }
    if passcode:
        headers["X-Passcode"] = passcode
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        base + NER_PATH, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    return _send(request, timeout)


def _block(data: dict[str, Any], name: str) -> dict[str, Any]:
    """The object under `name`, or an empty one when it is absent or something else."""
    value = data.get(name)
    return value if isinstance(value, dict) else {}


def _count(counter: Counter[str]) -> str:
    return ", ".join(f"{name} {n}" for name, n in counter.most_common()) or "none"


def _fallbacks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in data.get("fallbacks") or [] if isinstance(f, dict)]


def _tokens(usage: dict[str, Any], prompt: str, completion: str) -> str:
    """``input+output`` token counts; one the response did not report is ``unknown``."""

    def shown(value: object) -> str:
        return str(value) if isinstance(value, int) else "unknown"

    return f"{shown(usage.get(prompt))}+{shown(usage.get(completion))}"


def describe(data: dict[str, Any]) -> str:
    """The part of a text's line that comes from a 200 response."""
    entities = [e for e in data.get("entities") or [] if isinstance(e, dict)]
    picks = [e["pick"] for e in entities if isinstance(e.get("pick"), dict)]
    by_status = Counter(str(p.get("status")) for p in picks)
    by_model = Counter(str(p.get("selector")) for p in picks)
    # Why picks failed, as the site explains it (a deadline, a provider's refusal ...).
    why_failed = Counter(
        _redact(" ".join(str(p.get("explanation") or "no explanation").split()))[:MAX_MESSAGE_CHARS]
        for p in picks
        if p.get("status") == "failed"
    )
    steps = [
        f"{f.get('stage')}:{f.get('from')}>{f.get('to')}:{f.get('reason')}"
        for f in _fallbacks(data)
    ]
    usage = _block(data, "usage")
    timing = _block(data, "timing_ms")
    parts = [
        # Whose key the site says paid for the model calls (absent before own keys).
        f"provider {data.get('provider') or 'not reported'} "
        f"(credential {data.get('credential_source') or 'not reported'})",
        f"model {data.get('requested_model') or '?'} -> {data.get('model')}",
        f"selector {data.get('requested_selector') or data.get('selector')}",
        f"entities {len(entities)}",
        f"picks {len(picks)}" + (f" ({_count(by_status)}; by {_count(by_model)})" if picks else ""),
        *([f"failed picks: {_count(why_failed)}"] if why_failed else []),
        f"fallbacks {' '.join(steps) or 'none'}",
        "timing_ms " + (" ".join(f"{k}={v}" for k, v in timing.items()) or "none"),
        f"tokens extract {_tokens(usage, 'prompt_tokens', 'completion_tokens')}"
        f" in {usage.get('llm_calls', '?')} calls, select "
        + (
            f"{_tokens(usage, 'selector_prompt_tokens', 'selector_completion_tokens')}"
            f" in {usage['selector_calls']} calls"
            if usage.get("selector_calls")
            else "none"
        ),
    ]
    return "  ".join(parts)


def wrong_credential(answer: Answer, provider: str) -> str | None:
    """Why a 200 is not the run that was asked for: the site says it ran on another
    provider or on the other credential (an own-key request billed to the site, or the
    reverse). None when it agrees, and for a site request to an API from before these
    fields, which reports neither."""
    if not answer.ok:
        return None
    data = answer.data or {}
    asked = (provider, "site" if provider == SITE else "visitor")
    said = (data.get("provider"), data.get("credential_source"))
    if said == asked or (provider == SITE and said == (None, None)):
        return None
    return (
        f"asked for provider {asked[0]} on the {asked[1]} credential, but the response "
        f"says provider {said[0]}, credential {said[1]}"
    )


def line(name: str, answer: Answer, provider: str = SITE) -> str:
    head = f"{name}  {answer.status or '---'}  {answer.seconds:.1f}s"
    if answer.ok:
        assert answer.data is not None
        wrong = wrong_credential(answer, provider)
        flag = f"FAILED wrong credential: {wrong}  " if wrong else ""
        return f"{head}  {flag}{describe(answer.data)}"
    return f"{head}  FAILED {answer.failure()} (sent as provider {provider})"


def print_health(label: str, health: Answer) -> None:
    """The ``claude``, ``visitor`` and ``own_keys`` blocks of ``/api/health`` (absent
    ones say so)."""
    if health.status != 200 or health.data is None:
        say(f"{label}: FAILED {health.failure()}")
        return
    data = health.data
    say(
        f"{label}: mode {data.get('mode')}, default model {data.get('default_model')}, "
        f"default selector {data.get('default_selector')}, "
        f"passcode_required {data.get('passcode_required')}"
    )
    for block in ("claude", "visitor", "own_keys"):
        value = data.get(block)
        shown = json.dumps(value, sort_keys=True) if value is not None else "absent"
        say(f"  {block}: {shown}")


class Totals:
    """Sums over the answered texts. A token count a response did not report is left out
    of its sum and counted in ``unreported``, never added as 0."""

    FIELDS = (
        "prompt_tokens",
        "completion_tokens",
        "selector_prompt_tokens",
        "selector_completion_tokens",
    )

    def __init__(self) -> None:
        self.requests = 0
        self.failed = 0
        self.entities = 0
        self.calls = 0
        self.selector_calls = 0
        self.tokens = dict.fromkeys(self.FIELDS, 0)
        self.unreported = dict.fromkeys(self.FIELDS, 0)
        self.reasons: Counter[str] = Counter()
        self.ran: Counter[str] = Counter()
        self.slowest: tuple[float, str, Answer] | None = None
        self.largest: tuple[int, str, Answer] | None = None

    def add(self, name: str, answer: Answer, provider: str = SITE) -> None:
        self.requests += 1
        if self.slowest is None or answer.seconds > self.slowest[0]:
            self.slowest = (answer.seconds, name, answer)
        if not answer.ok or wrong_credential(answer, provider):
            self.failed += 1
            return
        data = answer.data or {}
        usage = _block(data, "usage")
        count = len(data.get("entities") or [])
        self.entities += count
        if self.largest is None or count > self.largest[0]:
            self.largest = (count, name, answer)
        self.ran[str(data.get("model"))] += 1
        self.reasons.update(str(f.get("reason")) for f in _fallbacks(data))
        for key, attr in (("llm_calls", "calls"), ("selector_calls", "selector_calls")):
            if isinstance(usage.get(key), int):
                setattr(self, attr, getattr(self, attr) + usage[key])
        selected = bool(usage.get("selector_calls"))
        for field in self.FIELDS:
            value = usage.get(field)
            if isinstance(value, int):
                self.tokens[field] += value
            elif selected or not field.startswith("selector_"):
                self.unreported[field] += 1

    def _sum(self, field: str) -> str:
        missing = self.unreported[field]
        return f"{self.tokens[field]}" + (f" (+{missing} unreported)" if missing else "")

    @staticmethod
    def _timing(answer: Answer) -> str:
        timing = (answer.data or {}).get("timing_ms")
        total = timing.get("total") if isinstance(timing, dict) else None
        return f"{answer.seconds:.1f} s round trip" + (
            f", {total} ms on the server" if total is not None else ""
        )

    def print(self) -> None:
        say(
            f"totals: {self.requests} requests, {self.requests - self.failed} ok, "
            f"{self.failed} failed; {self.entities} entities; extracted by {_count(self.ran)}"
        )
        say(
            f"  tokens: extract {self._sum('prompt_tokens')} in + "
            f"{self._sum('completion_tokens')} out over {self.calls} calls; select "
            f"{self._sum('selector_prompt_tokens')} in + "
            f"{self._sum('selector_completion_tokens')} out over {self.selector_calls} calls"
        )
        say(f"  step-downs by reason: {_count(self.reasons)}")
        if self.slowest is not None:
            say(
                f"  slowest: {self.slowest[1]} ({self._timing(self.slowest[2])}; "
                f"the site's deadline is {REQUEST_DEADLINE_S} s)"
            )
        if self.largest is not None:
            say(
                f"  most entities: {self.largest[1]} ({self.largest[0]} entities, "
                f"{self._timing(self.largest[2])})"
            )


def started_on(data: dict[str, Any]) -> str:
    """The tier a request was admitted on: ``opus``, ``sonnet`` or ``other``. It is the
    model asked for, or the selector asked for when the model is not a Claude id (the
    site admits by whichever of the two is Claude, the model first), unless an ``admit``
    step-down moved the start."""
    model = data.get("requested_model") or data.get("model")
    if not (isinstance(model, str) and model.startswith("claude-")):
        model = data.get("requested_selector") or data.get("selector")
    for note in _fallbacks(data):
        if note.get("stage") == "admit":
            model = note.get("to")
    return "opus" if model == OPUS else "sonnet" if model == SONNET else "other"


def offers(health: Answer, provider: str) -> bool:
    """Whether ``/api/health`` lists `provider` among the own-key providers."""
    listed = _block(health.data or {}, "own_keys").get("providers")
    return isinstance(listed, list) and any(
        isinstance(p, dict) and p.get("id") == provider for p in listed
    )


def send_burst(count: int, post: Callable[[], Answer]) -> list[Answer]:
    """`count` requests at once."""
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(lambda _: post(), range(count)))


def report_burst(
    name: str, answers: Sequence[Answer], before: Answer, after: Answer, provider: str = SITE
) -> int:
    """Say where the parallel requests `answers` started and compare that with the
    balances `before` (the health answer read just before them) reported; `after` is the
    health answer read just after. Returns the exit code."""
    count = len(answers)
    own = provider != SITE
    visitor = _block(before.data or {}, "visitor")
    # Only a limiter that answered can have counted anything. Without this a request
    # that ran while nothing counted it was tallied under the tier it asked for.
    comparable = visitor.get("limits") == "on"

    tiers: Counter[str] = Counter()
    refusals: Counter[str] = Counter()
    uncounted = legacy = unavailable = failed = 0
    unavailable_message = ""
    for answer in answers:
        if answer.ok:
            assert answer.data is not None
            if wrong_credential(answer, provider):
                failed += 1
                say(line(name, answer, provider))
            elif any(f.get("reason") == LIMITS_UNAVAILABLE for f in _fallbacks(answer.data)):
                legacy += 1
            elif not comparable:
                uncounted += 1
            else:
                tiers["own" if own else started_on(answer.data)] += 1
        elif answer.status == 429 and answer.code == "visitor_limit_reached":
            message = str(_block(answer.data or {}, "error").get("message", ""))
            refusals[
                "contended"
                if message.startswith(TOO_MANY_AT_ONCE)
                else "own"
                if message.startswith(OWN_KEY_LIMIT)
                else "day"
                if message.startswith(DAY_USED)
                else "hour"
            ] += 1
        elif answer.status == 503 and answer.code == LIMITS_UNAVAILABLE:
            unavailable += 1
            unavailable_message = answer.failure()
        else:
            failed += 1
            say(line(name, answer, provider))
    say(
        f"burst: {count} parallel requests of {name} as provider {provider}: "
        f"started on Opus {tiers['opus']}, on Sonnet {tiers['sonnet']}, "
        f"on another model {tiers['other']}, on the own key {tiers['own']}, "
        f"429 hour used {refusals['hour']}, 429 day used {refusals['day']}, "
        f"429 own-key limit {refusals['own']}, 429 too many at once {refusals['contended']}, "
        f"503 limits_unavailable {unavailable}, not counted {uncounted + legacy}, "
        f"failed {failed}"
    )

    balances = (("own", "own_left"),) if own else (("opus", "opus_left"), ("sonnet", "sonnet_left"))
    over = False
    if comparable:
        for tier, key in balances:
            left = visitor.get(key)
            if isinstance(left, int) and tiers[tier] > left:
                over = True
                say(
                    f"  OVER THE ALLOWANCE: {tiers[tier]} requests started on {tier}, "
                    f"but {key} was {left} before the burst"
                )
    else:
        say(
            "  NOTHING TO COMPARE WITH: health reported no visitor block with limits on "
            "before the burst, so this run says nothing about the allowance"
            + (f", and the {uncounted} requests that ran were not counted" if uncounted else "")
            + (
                " (health reports `visitor` only where Claude is offered on the site's keys)"
                if own and not visitor
                else ""
            )
        )
    if unavailable:
        say(
            f"  THE LIMITER IS FAILING: {unavailable} of {count} requests were refused with "
            f"503 ({unavailable_message}): the site could not count them, and it runs no "
            "request on its own keys uncounted. `visitor limits are not configured` means "
            "the deployment has no DATABASE_URL. Otherwise the reason is the error class "
            "and SQLSTATE in the `limits` field of each such request's log line; 40001 "
            "means the database endpoint does not run the limiter's transaction at READ "
            "COMMITTED"
        )
    if legacy:
        say(
            f"  THE LIMITER IS FAILING: {legacy} of {count} requests ran uncounted with a "
            "`limits_unavailable` step-down. A site that does that runs code from before "
            "18 November 2026 (the current code answers 503 instead); the reason is in "
            "the `limits` field of their log lines"
        )
    stopped = False
    if own and comparable and _block(after.data or {}, "visitor").get("limits") != "on":
        stopped = True
        say(
            "  THE LIMITER STOPPED ANSWERING: health reported no visitor block with limits "
            "on after the burst. An own-key request runs uncounted when the limiter cannot "
            "be asked, and its response does not say so: the starts above may not have "
            "been counted"
        )
    if refusals["contended"]:
        say(
            f"  {refusals['contended']} requests were refused because too many arrived at "
            "once: expected for the tail of a large burst, and not counted against the "
            "allowance"
        )
    bad = bool(failed or over or unavailable or legacy or stopped or not comparable)
    if comparable and not (over or unavailable or legacy or stopped):
        # The hourly balances are what the starts are compared with; the day's are shown
        # because they cap them (an API from before the daily cap reports none).
        keys = [key for _, key in balances]
        keys += [f"{key}_today" for key in keys if f"{key}_today" in visitor]
        shown = ", ".join(f"{key} {visitor.get(key)}" for key in keys)
        say(f"  within the allowance health reported before the burst ({shown})")
    return 1 if bad else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post text files to a NER site's /api/ner and print model, picks, step-downs, "
            "timings and tokens per request. It only GETs /api/health and POSTs /api/ner "
            "on the given URL."
        ),
        epilog=(
            "Each run makes real requests: every text counts against your hourly visitor "
            "allowance on that site (and against the daily one) and spends the balance of "
            "the operator's key behind it (Anthropic for Claude models, OpenRouter for "
            "the others). With --provider and --key-env the model calls are billed to "
            "the API key in that environment variable instead (a run with an own key "
            "bills that key), and the requests count against the own-key flood limit. The "
            "key is sent to the given URL as Authorization: Bearer; it is never printed."
        ),
    )
    parser.add_argument("--url", required=True, help="the site, e.g. http://127.0.0.1:8765")
    parser.add_argument(
        "--passcode-env",
        metavar="NAME",
        help="name of the environment variable that holds the passcode (sent as X-Passcode)",
    )
    parser.add_argument(
        "--provider",
        choices=(SITE, *OWN_PROVIDERS),
        default=SITE,
        help="whose key runs the models: the site's (default), or your own key for that "
        "provider, which needs --key-env and bills that key",
    )
    parser.add_argument(
        "--key-env",
        metavar="NAME",
        help="name of the environment variable that holds your own API key for --provider "
        "(sent as Authorization: Bearer on POST /api/ner; this key pays for the model calls)",
    )
    parser.add_argument(
        "--model",
        help="extraction model id; default: the site's default, or the provider's with an own key",
    )
    parser.add_argument(
        "--selector",
        help="selector (retrieval, jev or a model id; with an own key retrieval or one of "
        "the provider's models); default: the site's default, or retrieval with an own key",
    )
    parser.add_argument(
        "--sequential-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="wait this long between requests, to pace them under the visitor allowance",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=0,
        metavar="N",
        help="send N copies of the first file in parallel and count where they started "
        "(Opus, Sonnet, the own key, 429, 503); the other files are not sent",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"per request (default {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="FILE", help="UTF-8 text files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    parts = urlsplit(args.url)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.query:
        parser.error("--url must be an http(s) URL such as https://<deployment>")
    if parts.username or parts.password:
        parser.error("--url must not carry credentials")
    base = args.url.rstrip("/")
    if args.sequential_delay < 0 or args.timeout <= 0:
        parser.error("--sequential-delay must be 0 or more and --timeout above 0")
    if not 0 <= args.burst <= MAX_BURST:
        parser.error(f"--burst must be from 0 to {MAX_BURST}")
    passcode: str | None = None
    if args.passcode_env:
        passcode = os.environ.get(args.passcode_env) or None
        if passcode is None:
            parser.error(f"the environment variable {args.passcode_env} is not set")
    key: str | None = None
    if args.provider == SITE:
        if args.key_env:
            parser.error("--key-env needs --provider anthropic, openai or openrouter")
    else:
        if not args.key_env:
            parser.error(f"--provider {args.provider} needs --key-env NAME (your own API key)")
        key = os.environ.get(args.key_env) or None
        if key is None:
            parser.error(f"the environment variable {args.key_env} is not set")
        if _KEY.fullmatch(key) is None:  # what the site accepts; the value is not shown
            parser.error(
                f"the environment variable {args.key_env} does not hold an API key of 20 to "
                "512 printable ASCII characters without spaces"
            )
        if parts.scheme != "https" and parts.hostname not in LOOPBACK:
            parser.error("--key-env sends an API key: --url must be https (or this machine)")
    # Longest first, so that one secret inside the other leaves no remainder behind.
    _REDACTED[:] = sorted((s for s in (passcode, key) if s), key=len, reverse=True)
    texts: list[tuple[str, str]] = []
    for file in args.files:
        try:
            texts.append((file.name, file.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as exc:
            parser.error(f"cannot read {file}: {type(exc).__name__}")

    before = get_health(base, timeout=args.timeout)
    print_health("health", before)
    provider: str = args.provider
    say(
        f"running as provider {provider}: "
        + (
            f"your own key from the environment variable {args.key_env}, which is billed "
            "for the model calls"
            if key
            else "the site's keys"
        )
    )
    if key and not offers(before, provider):
        # The key goes nowhere that has not said it takes one.
        say(
            f"FAILED: /api/health at this URL does not offer own keys for {provider} "
            "(see own_keys above), so nothing was sent and the key did not leave this machine"
        )
        return 1

    def post(text: str) -> Answer:
        return post_text(
            base,
            text,
            model=args.model,
            selector=args.selector,
            passcode=passcode,
            timeout=args.timeout,
            provider=provider,
            key=key,
        )

    if args.burst:
        name, text = texts[0]
        answers = send_burst(args.burst, lambda: post(text))
        after = get_health(base, timeout=args.timeout)
        code = report_burst(name, answers, before, after, provider)
    else:
        totals = Totals()
        for index, (name, text) in enumerate(texts):
            if index and args.sequential_delay:
                time.sleep(args.sequential_delay)
            answer = post(text)
            totals.add(name, answer, provider)
            say(line(name, answer, provider))
            sys.stdout.flush()
        totals.print()
        code = 1 if totals.failed else 0
        after = get_health(base, timeout=args.timeout)

    print_health("health after", after)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
