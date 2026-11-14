"""
The evaluation modules pin the candidate mode of the system they measured (V30).

The default candidate mode (DEFAULT_CANDIDATES) can change; a stored report
must not. So every SearchService an evaluation module builds names its mode,
and the assistant evaluation, which builds its service from the application's
settings, pins SEARCH_CANDIDATES to the mode recorded in its stored runs.
These tests read source files and stored reports only (no network, no write).
"""

import ast
import json
from pathlib import Path

import pytest

from app.evaluation.assistant_evaluation import design as D
from app.infrastructure.config.settings import SearchProfileSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "app" / "evaluation"


def _search_service_calls(path: Path):
    """The SearchService(...) calls of one source file, as AST nodes."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SearchService":
            yield node


def test_every_search_service_of_the_evaluation_modules_names_its_candidate_mode():
    calls = [(path, node) for path in sorted(EVALUATION_DIR.rglob("*.py")) for node in _search_service_calls(path)]
    assert len(calls) >= 11  # the construction sites known when the default changed (V24, V30)
    unpinned = [f"{path.relative_to(REPO_ROOT)}:{node.lineno}" for path, node in calls
                if not any(keyword.arg == "candidates" for keyword in node.keywords)]
    assert unpinned == []


@pytest.mark.parametrize("stored", ["data/evaluation/assistant/system.json",
                                    "data/evaluation/assistant_v25/system.json"])
def test_the_assistant_evaluation_runs_the_candidate_mode_of_its_stored_runs(stored):
    recorded = json.loads((REPO_ROOT / stored).read_text(encoding="utf-8"))["profile"]["candidates"]
    assert SearchProfileSettings.from_env(dict(D.CLOUD_ENV)).candidates == recorded
