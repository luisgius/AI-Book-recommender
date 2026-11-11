"""
The cloud profile starts and serves every endpoint without the model libraries (V13).

Each test starts the API in a fresh Python process (tests/api/cloud_profile_driver.py)
in the cloud configuration (SEARCH_PROFILE=cloud: Postgres adapters, API embeddings
and API reranker by default), with the database and OpenRouter faked in that
process, so no network is used. A fresh process is needed because this test
process has already imported torch and faiss through other tests.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("torch", "faiss", "sentence_transformers", "transformers", "rank_bm25")


def run_driver(*flags: str, llm_provider: str = "openrouter") -> dict:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "SEARCH_PROFILE": "cloud",
        "OPENROUTER_API_KEY": "test-key-not-used",
        "DATABASE_URL": "postgresql://user:pass@127.0.0.1:1/none",
        "LLM_PROVIDER": llm_provider,
        "USAGE_ENDPOINT_ENABLED": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "tests.api.cloud_profile_driver", *flags],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def assert_every_endpoint_answers(report: dict) -> None:
    statuses = report["statuses"]
    health = statuses["health"]
    assert health["status"] == "ok" and health["profile"]["name"] == "cloud"
    assert health["reranker"] == {"configured": True, "model": "qwen/qwen3-reranker-8b", "ready": True}
    assert statuses["search"] == [200, 3]
    assert statuses["book"] == 200
    assert statuses["similar"] == [200, 3]
    assert statuses["ui"] == 200
    assert statuses["mcp"] == 200


def test_cloud_profile_serves_every_endpoint_without_importing_model_libraries():
    report = run_driver()

    assert_every_endpoint_answers(report)
    assert report["forbidden_imported"] == []
    assert "numpy" not in report["top_level_modules"]  # the API encoder decodes vectors with struct
    usage = report["statuses"]["usage"]["api"]
    assert usage["embedding"]["calls"] == 1 and usage["rerank"]["calls"] == 1
    assert usage["total"]["reported_cost_usd"] == pytest.approx(1e-7 + 1e-5)
    assert report["statuses"]["usage"]["llm"]["total"]["calls"] == 0  # LLM client built, not called


def test_cloud_profile_runs_with_the_model_libraries_unimportable():
    report = run_driver("--block")

    assert_every_endpoint_answers(report)
    assert report["forbidden_imported"] == []
