"""
Hexagonal boundary: the domain layer imports no adapter, framework or driver.

Every module under app/domain/ is parsed (not imported), so imports inside
`if TYPE_CHECKING:` blocks, functions and try blocks are checked too, and
relative imports are resolved against the module's package.
"""

import ast
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

DOMAIN_DIR = Path(__file__).resolve().parents[2] / "app" / "domain"

# Top-level packages / module prefixes the domain must never import.
FORBIDDEN = (
    "app.infrastructure",
    "app.api",
    "app.main",
    "fastapi",
    "starlette",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",
    "langchain_ollama",
    "langgraph",
    "faiss",
    "sqlite3",
    "psycopg",
    "psycopg2",
    "psycopg_pool",
    "asyncpg",
    "sqlalchemy",
)


def _module_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imported_modules(path: Path, root: Path) -> Iterator[Tuple[int, str]]:
    """(line, absolute module name) of every import statement in a source file under `root`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_name(path, root).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            yield node.lineno, module
            for alias in node.names:  # "from app import infrastructure"
                yield node.lineno, f"{module}.{alias.name}"


def is_forbidden(module: str) -> bool:
    return any(module == name or module.startswith(name + ".") for name in FORBIDDEN)


def violations(paths, domain_dir: Path = DOMAIN_DIR) -> List[str]:
    root = domain_dir.parents[1]  # the directory that holds the app package
    found = []
    for path in paths:
        for line, module in imported_modules(path, root):
            if is_forbidden(module):
                found.append(f"{path.relative_to(root)}:{line} imports {module}")
    return sorted(set(found))


def test_domain_layer_imports_no_adapter_framework_or_driver():
    paths = sorted(DOMAIN_DIR.rglob("*.py"))
    assert paths, "no domain modules found"
    assert violations(paths) == []


@pytest.mark.parametrize("source", [
    "from app.infrastructure.llm.schemas_judge import ExplanationJudgmentLLM\n",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from app.infrastructure.db import x\n",
    "def f():\n    import faiss\n",
    "from ..infrastructure.search import bm25\n",
    "from app import api\n",
    "import langchain_core.messages\n",
    "from fastapi import FastAPI\n",
    "import sqlite3\n",
    "import psycopg\n",
])
def test_the_check_catches_each_kind_of_import(tmp_path, source):
    domain = tmp_path / "app" / "domain"
    domain.mkdir(parents=True)
    module = domain / "bad.py"
    module.write_text(source, encoding="utf-8")
    assert violations([module], domain) != []


def test_the_check_allows_domain_and_standard_library_imports(tmp_path):
    domain = tmp_path / "app" / "domain"
    domain.mkdir(parents=True)
    module = domain / "good.py"
    module.write_text("from typing import Protocol\nfrom .entities import Book\nfrom app.domain import value_objects\n",
                      encoding="utf-8")
    assert violations([module], domain) == []


def test_llm_adapter_schema_satisfies_the_domain_judgment_port():
    """ExplanationJudgmentLLM still provides what the domain port declares (behaviour unchanged)."""
    from app.domain.ports import ExplanationJudgment, JudgmentScore
    from app.infrastructure.llm.schemas_judge import ExplanationJudgmentLLM, JudgmentDimension

    dimension = JudgmentDimension(score=4, reasoning="ok")
    judgment = ExplanationJudgmentLLM(groundedness=dimension, clarity=dimension, relevance=dimension)
    for name in ("groundedness", "clarity", "relevance"):
        assert name in {m for m in dir(ExplanationJudgment) if not m.startswith("_")}
        value = getattr(judgment, name)
        assert all(hasattr(value, attr) for attr in ("score", "reasoning"))
    assert {"score", "reasoning"} <= set(dir(JudgmentScore))
