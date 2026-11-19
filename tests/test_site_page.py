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
    assert set(example) == {"text", "recorded", "ontologies", "response"}
    raw = (PUBLIC / "example.json").read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_-]{8,}|authorization|bearer|passcode", raw, re.IGNORECASE)


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
        "to compute search embeddings. The server may keep recent mentions, search results and "
        "answers in memory for a few hours so that it does not compute them twice; none of that "
        "is written to its database.</li>"
    ) in text
    assert "<li>When run recording is enabled, this site attempts to save" in text
    assert "<li>Runs on this site's own models are limited per network" in text
    for promise in ("writes no text", "Each run is recorded", "stores no", "not saved"):
        assert promise not in text
