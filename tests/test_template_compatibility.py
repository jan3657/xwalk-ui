"""Workbench templates retain their pre-sandbox rendering."""

from jinja2 import Environment
from xwalk.records import Record
from xwalk.templates import TemplateSet, _EmptyUndefined, _tidy

from xwalk_ui import projects

_SYNTHETIC = [
    Record(
        "x1",
        {
            "text": "glucose",
            "mention": "glucose",
            "label": "Glucose",
            "synonyms": ["dextrose", "D-glucose"],
            "definition": "a sugar",
            "context": "",
            "context_left": "blood ",
            "context_right": " level",
            "aliases": None,
            "ontology": "CHEBI",
        },
    ),
    Record("x2", {"text": "x  y", "label": "L", "synonyms": [], "definition": None}),
    Record("x3", {}),
]


def _reference(source: str, record: Record) -> str:
    """How the pre-sandbox code rendered one template: a plain environment."""
    env = Environment(undefined=_EmptyUndefined, keep_trailing_newline=False)
    variables = {k: v for k, v in record.fields.items() if v is not None}
    variables["id"] = record.id
    return _tidy(env.from_string(source).render(**variables))


def test_workbench_templates_render_as_before():
    ts = TemplateSet(
        **{k: projects._TEMPLATES[k] for k in ("query", "context", "doc", "candidate")}
    )
    for record in _SYNTHETIC:
        for name in ("query", "context", "doc", "candidate"):
            assert ts._render(name, record) == _reference(getattr(ts, name), record)
