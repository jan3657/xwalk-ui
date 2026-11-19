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
