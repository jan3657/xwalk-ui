"""The recorded example the page shows on arrival (``ner-site/public/example.json``).

The page renders that file with the code that renders a live answer, without asking the
API. Nothing else ties the file to the API, so these tests do: a change of the response
schema, or of the example sentence in ``app.js``, fails here instead of silently
breaking the first screen.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

import pytest
from ner_site.schema import NerResponse, OntologySummary

PUBLIC = Path(__file__).resolve().parents[1] / "ner-site" / "public"


@pytest.fixture(scope="module")
def example() -> dict[str, Any]:
    return json.loads((PUBLIC / "example.json").read_text(encoding="utf-8"))


def _js_constant(name: str) -> str:
    """The value of ``const NAME = "..." + "...";`` in app.js."""
    source = (PUBLIC / "app.js").read_text(encoding="utf-8")
    found = re.search(rf"const {name} =\s*((?:\"(?:[^\"\\]|\\.)*\"\s*\+?\s*)+);", source)
    assert found, f"{name} not found in app.js"
    parts = re.findall(r"\"(?:[^\"\\]|\\.)*\"", found.group(1))
    return "".join(json.loads(part) for part in parts)


def test_example_response_is_a_valid_ner_response(example: dict[str, Any]) -> None:
    response = NerResponse.model_validate(example["response"])
    # Nothing in the file is dropped or defaulted by the model: it is what the API sends.
    assert response.model_dump(mode="json", by_alias=True) == example["response"]
    assert response.mode == "live"
    assert response.entities


def test_example_text_is_the_full_catalog_example_of_the_page(example: dict[str, Any]) -> None:
    assert example["text"] == _js_constant("EXAMPLE_FULL")


def test_example_mentions_sit_where_the_text_has_them(example: dict[str, Any]) -> None:
    text = example["text"]
    for entity in example["response"]["entities"]:
        assert text[entity["start"] : entity["end"]] == entity["text"]


def test_example_is_dated_and_names_its_ontologies(example: dict[str, Any]) -> None:
    # The page labels the example with this date and does not show it without one.
    datetime.date.fromisoformat(example["recorded"])
    named = {OntologySummary.model_validate(o).slug for o in example["ontologies"]}
    used = {e["ontology"] for e in example["response"]["entities"] if e["ontology"]}
    assert used <= named


def test_example_holds_only_the_expected_fields(example: dict[str, Any]) -> None:
    # A public file: a recording pasted in with anything else (headers, a key) fails.
    assert set(example) == {"text", "recorded", "request", "ontologies", "response"}
    raw = (PUBLIC / "example.json").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_-]{8,}|authorization|bearer|passcode", raw, re.IGNORECASE)


def test_example_records_the_request_it_answers(example: dict[str, Any]) -> None:
    """The page shows the recording in place of a run only on a deployment whose
    default prompt and entity limit are the ones in ``request``. They are what a
    full-catalog deployment with the default limits uses: when the default prompt or
    the default limit changes, this fails, and the example must be recorded again (or
    the page would silently stop using it)."""
    from ner_site.service import Settings

    asked = example["request"]
    assert set(asked) == {"prompt", "top_k", "max_entities"}
    full = Settings(backend="neon", catalog="full")
    assert asked["prompt"] == full.default_prompt
    assert asked["max_entities"] == full.limits.max_entities
    assert asked["top_k"] == 5
    response = example["response"]
    assert len(response["entities"]) <= asked["max_entities"]
    assert all(len(e["matches"]) <= asked["top_k"] for e in response["entities"])


def test_the_page_checks_the_form_and_the_recorded_settings_before_the_shortcut() -> None:
    """Held here by reading app.js; tools/page_smoke.mjs ("example-settings",
    "example-live") holds the behaviour in a browser."""
    source = (PUBLIC / "app.js").read_text(encoding="utf-8")
    run = source[source.index("  async function run(opts) {") :]
    shortcut = run.index("exampleAsked()")
    # The checks a run must pass come first, the shortcut after them.
    assert run.index("state.limits.max_chars") < shortcut
    assert run.index("state.limits.max_prompt_chars") < shortcut
    # A live POST clears "nothing was sent" before it is sent.
    assert run.index("state.shown.pressed = false") < run.index('api("POST", "/api/ner"')
    asked = source[
        source.index("  function exampleAsked() {") : source.index("  function showRecorded()")
    ]
    assert "state.defaultPrompt !== asked.prompt" in asked
    assert "state.limits.max_entities !== asked.max_entities" in asked
    # The prompt held to the limit is the effective one: an empty editor means the
    # deployment's default, as on the server (service.validate_request), both in the
    # form check and in the shortcut itself.
    effective = source[
        source.index("  function effectivePrompt() {") : source.index("  function exampleAsked() {")
    ]
    assert "return typed.trim() ? typed : state.defaultPrompt;" in effective
    assert run.index("codePointLength(effectivePrompt())") < shortcut
    assert "const prompt = effectivePrompt();" in asked
    assert "codePointLength(prompt) > state.limits.max_prompt_chars" in asked
    # No age and no figures of another run are read from the response.
    for gone in ("age_s", "cache.original", "retrieval_hits", "pick_hits"):
        assert gone not in source, gone
    assert "Answered from a recent identical run. No model was called." in source


# ---------- the form without its script ----------
#
# If app.js is slow or never loads, the browser is left with a plain form. Three things
# keep a visitor's text out of the address bar, the history and the request log then;
# each is held here by itself, and tools/page_smoke.mjs ("no-script") checks them in a
# browser.


def _index() -> str:
    return (PUBLIC / "index.html").read_text(encoding="utf-8")


def test_no_form_field_of_the_page_has_a_name() -> None:
    # A native submission sends named fields only: with none, it carries nothing.
    form = re.search(r"<form id=\"ner-form\".*?</form>", _index(), re.DOTALL)
    assert form, "the form is gone from index.html"
    controls = re.findall(r"<(?:input|textarea|select|button)\b[^>]*>", form.group(0))
    assert len(controls) >= 10
    assert [c for c in controls if re.search(r"\sname\s*=", c)] == []


def test_run_is_disabled_until_the_script_enables_it() -> None:
    run = re.search(r"<button type=\"submit\" id=\"run\"[^>]*>", _index())
    assert run and re.search(r"\sdisabled>$", run.group(0))
    source = (PUBLIC / "app.js").read_text(encoding="utf-8")
    assert "el.run.disabled = state.running || Boolean(state.payer);" in source
    assert '<noscript><p class="banner">This demo needs JavaScript.</p></noscript>' in _index()


def test_the_policy_refuses_every_form_submission() -> None:
    config = json.loads((PUBLIC.parent / "vercel.json").read_text(encoding="utf-8"))
    everywhere = next(rule for rule in config["headers"] if rule["source"] == "/(.*)")
    policy = next(
        h["value"] for h in everywhere["headers"] if h["key"] == "Content-Security-Policy"
    )
    directives = [d.strip() for d in policy.split(";")]
    assert "form-action 'none'" in directives
    assert "default-src 'self'" in directives


# ---------- the footer's statements ----------
#
# "How this site handles data" promises only what holds for every run: each statement
# carries its condition, and none says that every run is recorded or that nothing is
# kept. tools/page_smoke.mjs ("production") reads the same lines in a browser.


def test_the_footer_states_its_conditions() -> None:
    footer = re.search(r"<footer class=\"footer\" id=\"privacy\">.*?</footer>", _index(), re.DOTALL)
    assert footer, "the footer is gone from index.html"
    text = footer.group(0)
    assert (
        "<li>This site is built not to keep your text, prompt or results: its logs and its "
        "database are designed to hold run metadata only, for example time, model, token counts "
        "and country, "
        "and a visitor hash that changes every day.</li>"
    ) in text
    assert (
        "<li>When embedding search is used, the mentions found in your text are sent to OpenRouter "
        "to compute search embeddings, and the server may keep recently embedded mentions in "
        "memory. It may also keep recent answers in memory so that an identical request is not "
        "run twice: normally for a few hours, and at most until the server instance is recycled. "
        "None of that is written to its database, and an answer served from memory sends no "
        "text to a model provider.</li>"
    ) in text
    assert "<li>When run recording is enabled, this site attempts to save" in text
    assert "<li>Runs on this site's own models are limited per network" in text
    for promise in ("writes no text", "Each run is recorded", "stores no", "not saved"):
        assert promise not in text
    # Only whole answers are kept by default: the footer no longer speaks of search
    # results, and "a few hours" cannot become days (the variable ends at one day).
    assert "search results" not in text
    from ner_site import cache as caching
    from ner_site.service import Settings, SettingsError

    assert caching.MAX_TTL_S == 24 * 3600 >= caching.DEFAULT_TTL_S
    assert Settings.from_env({"NER_SITE_CACHE_TTL_S": "86400"}).cache.ttl_s == 86400
    with pytest.raises(SettingsError, match="from 0 to 86400"):
        Settings.from_env({"NER_SITE_CACHE_TTL_S": "86401"})
