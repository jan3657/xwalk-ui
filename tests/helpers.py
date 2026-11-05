import copy
import json
import math
from pathlib import Path

import yaml
from xwalk.llm.fake import FakeLLM

from tests.conftest import FIXTURES


class ToyEncoder:
    """Deterministic, dependency-free: a character-histogram embedding.

    Enough structure for 'glucose' to sit closer to 'fructose' than to 'automobile',
    which is all the retriever's contract requires. Tests must not depend on a
    downloaded model.
    """

    name = "toy"
    dimension = 26

    def encode(self, texts, *, is_query: bool = False):
        vectors = []
        for text in texts:
            vector = [0.0] * 26
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vector[ord(ch) - 97] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


def select_reply(key, score=0.9, explanation="ok"):
    return json.dumps({"chosen_key": key, "confidence_score": score, "explanation": explanation})


def score_reply(score, **extra):
    return json.dumps({"confidence_score": score, "explanation": "", **extra})


JOB = FIXTURES / "job_tiny.yaml"
BASE = yaml.safe_load(JOB.read_text(encoding="utf-8"))


def _llm(score: float = 0.95) -> FakeLLM:
    def handler(request):
        if "## Candidates" not in request.user:
            return score_reply(score)
        return select_reply("C01") if "[C01]" in request.user else select_reply(None)

    return FakeLLM(handler=handler)


def _job_file(tmp_path: Path, edit=None, *, name: str = "job.yaml") -> Path:
    """A copy of the tiny job next to copies of its data, optionally edited."""
    data = copy.deepcopy(BASE)
    data["prompts"]["slots"] = str((FIXTURES / data["prompts"]["slots"]).resolve())
    for role in ("target", "source"):
        source = FIXTURES / data[role]["path"]
        copied = tmp_path / source.name
        if not copied.exists():
            copied.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if edit is not None:
        edit(data)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path
