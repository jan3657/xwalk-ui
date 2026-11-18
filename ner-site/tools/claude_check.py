"""Run texts through a deployed (or local) NER site and print what each request did.

For the operator, during the rollout in ``ner-site/README.md`` ("Rollout: Claude by
default and open access"). Nothing calls this tool; it makes requests only when you run
it.

What it costs to run. Every text is one real ``POST /api/ner``: it spends the balance of
the key behind the site (the operator's Anthropic balance when the model or the selector
is Claude, the OpenRouter key otherwise) and it counts against the caller's hourly
allowance on that site (by default 10 requests starting on Opus, then 20 on Sonnet, then
429 until the clock hour ends). ``--burst N`` sends N such requests at once.

What it does, and nothing else: one ``GET <url>/api/health`` before and one after, and
one ``POST <url>/api/ner`` per text, to the URL it is given. It follows no redirect,
writes no file, and reads only the text files named on the command line and, with
``--passcode-env NAME``, the environment variable NAME (sent as ``X-Passcode``; the
passcode is never taken from an argument and never printed: should the site echo it,
it is replaced by ``[redacted]`` in everything this tool prints).

Usage (standard library only; any Python 3.10 or later)::

    python ner-site/tools/claude_check.py --url http://127.0.0.1:8765 text1.txt text2.txt
    python ner-site/tools/claude_check.py --url https://<deployment> \\
        --passcode-env NER_SITE_PASSCODE --sequential-delay 5 texts/*.txt
    python ner-site/tools/claude_check.py --url https://<deployment> \\
        --selector retrieval --burst 15 short.txt

It prints the health ``claude`` and ``visitor`` blocks, then one line per text: HTTP
status, round-trip seconds, the model asked for and the model that extracted, the
selector, the entity count, the picks by status and by the model that made them (and why
any failed), every step-down (``stage:from>to:reason``), the server's ``timing_ms`` and
the token counts; then the totals, the slowest request and the one with the most
entities. Exit code 1 if any POST failed, 2 for a usage error.

``--burst N`` instead sends N copies of the first text in parallel and reports how many
started on Opus, how many on Sonnet, how many were refused with 429 (the hour used, or
too many at once: a request that waited too long for its turn at the visitor's lock,
which is what the tail of a large burst should get) and how many ran uncounted, against
the allowance ``/api/health`` reported just before. Exit code 1 when more started on a
tier than that allowance had left (the limiter does not serialise one visitor's
requests), when more than a fifth of the burst ran uncounted (the limiter is failing),
when health reported no allowance to compare with, or when a request failed otherwise.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
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
# The message of the 429 a request gets when the visitor's lock was contended
# (ner_site.http._admit); the code is visitor_limit_reached, as for a used hour.
TOO_MANY_AT_ONCE = "too many requests at once"
# The part of a burst that may run uncounted (limits_unavailable) before the limiter is
# called failing: a single slow answer from its database, not a pattern.
MAX_UNCOUNTED_SHARE = 0.2

# What must never be printed (the passcode of this run), whoever sent it back.
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
) -> Answer:
    body: dict[str, Any] = {"text": text}
    if model:
        body["model"] = model
    if selector:
        body["selector"] = selector
    headers = {
        "Content-Type": "application/json",
        "X-Requested-With": REQUESTED_WITH,
        "User-Agent": USER_AGENT,
    }
    if passcode:
        headers["X-Passcode"] = passcode
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


def line(name: str, answer: Answer) -> str:
    head = f"{name}  {answer.status or '---'}  {answer.seconds:.1f}s"
    if answer.ok:
        assert answer.data is not None
        return f"{head}  {describe(answer.data)}"
    return f"{head}  FAILED {answer.failure()}"


def print_health(label: str, health: Answer) -> None:
    """The ``claude`` and ``visitor`` blocks of ``/api/health`` (absent ones say so)."""
    if health.status != 200 or health.data is None:
        say(f"{label}: FAILED {health.failure()}")
        return
    data = health.data
    say(
        f"{label}: mode {data.get('mode')}, default model {data.get('default_model')}, "
        f"default selector {data.get('default_selector')}, "
        f"passcode_required {data.get('passcode_required')}"
    )
    for block in ("claude", "visitor"):
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

    def add(self, name: str, answer: Answer) -> None:
        self.requests += 1
        if self.slowest is None or answer.seconds > self.slowest[0]:
            self.slowest = (answer.seconds, name, answer)
        if not answer.ok:
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


def run_burst(
    base: str,
    name: str,
    text: str,
    count: int,
    before: Answer,
    *,
    model: str | None,
    selector: str | None,
    passcode: str | None,
    timeout: float,
) -> int:
    """Send `count` copies of one text at once and compare what was granted with the
    allowance `before` (the health answer just read) reported. Returns the exit code."""

    def one(_: int) -> Answer:
        return post_text(
            base, text, model=model, selector=selector, passcode=passcode, timeout=timeout
        )

    with ThreadPoolExecutor(max_workers=count) as pool:
        answers = list(pool.map(one, range(count)))

    tiers: Counter[str] = Counter()
    uncounted = refused = contended = failed = 0
    for answer in answers:
        if answer.ok:
            assert answer.data is not None
            if any(f.get("reason") == "limits_unavailable" for f in _fallbacks(answer.data)):
                uncounted += 1
            else:
                tiers[started_on(answer.data)] += 1
        elif answer.status == 429 and answer.code == "visitor_limit_reached":
            error = _block(answer.data or {}, "error")
            if str(error.get("message", "")).startswith(TOO_MANY_AT_ONCE):
                contended += 1
            else:
                refused += 1
        else:
            failed += 1
            say(line(name, answer))
    say(
        f"burst: {count} parallel requests of {name}: started on Opus {tiers['opus']}, "
        f"on Sonnet {tiers['sonnet']}, on another model {tiers['other']}, "
        f"429 hour used {refused}, 429 too many at once {contended}, "
        f"uncounted (limits_unavailable) {uncounted}, failed {failed}"
    )

    visitor = (before.data or {}).get("visitor")
    comparable = isinstance(visitor, dict) and visitor.get("limits") == "on"
    failing = uncounted > count * MAX_UNCOUNTED_SHARE
    over = False
    if comparable:
        assert isinstance(visitor, dict)
        for tier, key in (("opus", "opus_left"), ("sonnet", "sonnet_left")):
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
        )
    if failing:
        say(
            f"  THE LIMITER IS FAILING: {uncounted} of {count} requests ran on Sonnet "
            "uncounted (limits_unavailable). The reason is the SQLSTATE in the `limits` "
            "field of each such request's log line; 40001 means the database endpoint "
            "does not run the limiter's transaction at READ COMMITTED"
        )
    elif uncounted:
        say(
            f"  {uncounted} of {count} requests ran on Sonnet uncounted "
            "(limits_unavailable): the limiter's database did not answer them; the "
            "SQLSTATE is in the `limits` field of their log lines"
        )
    if contended:
        say(
            f"  {contended} requests were refused because too many arrived at once: "
            "expected for the tail of a large burst, and not counted against the hour"
        )
    if comparable and not over and not failing:
        assert isinstance(visitor, dict)
        say(
            "  within the allowance health reported before the burst "
            f"(opus_left {visitor.get('opus_left')}, sonnet_left {visitor.get('sonnet_left')})"
        )
    return 1 if failed or over or failing or not comparable else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post text files to a NER site's /api/ner and print model, picks, step-downs, "
            "timings and tokens per request. It only GETs /api/health and POSTs /api/ner "
            "on the given URL."
        ),
        epilog=(
            "Each run makes real requests: every text counts against your hourly visitor "
            "allowance on that site and spends the balance of the operator's key behind "
            "it (Anthropic for Claude models, OpenRouter for the others)."
        ),
    )
    parser.add_argument("--url", required=True, help="the site, e.g. http://127.0.0.1:8765")
    parser.add_argument(
        "--passcode-env",
        metavar="NAME",
        help="name of the environment variable that holds the passcode (sent as X-Passcode)",
    )
    parser.add_argument("--model", help="extraction model id; default: the site's default")
    parser.add_argument(
        "--selector", help="selector (retrieval, jev or a model id); default: the site's default"
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
        "(Opus, Sonnet, 429); the other files are not sent",
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
    _REDACTED[:] = [passcode] if passcode else []
    texts: list[tuple[str, str]] = []
    for file in args.files:
        try:
            texts.append((file.name, file.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError) as exc:
            parser.error(f"cannot read {file}: {type(exc).__name__}")

    before = get_health(base, timeout=args.timeout)
    print_health("health", before)
    common = {
        "model": args.model,
        "selector": args.selector,
        "passcode": passcode,
        "timeout": args.timeout,
    }

    if args.burst:
        name, text = texts[0]
        code = run_burst(base, name, text, args.burst, before, **common)
    else:
        totals = Totals()
        for index, (name, text) in enumerate(texts):
            if index and args.sequential_delay:
                time.sleep(args.sequential_delay)
            answer = post_text(base, text, **common)
            totals.add(name, answer)
            say(line(name, answer))
            sys.stdout.flush()
        totals.print()
        code = 1 if totals.failed else 0

    print_health("health after", get_health(base, timeout=args.timeout))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
