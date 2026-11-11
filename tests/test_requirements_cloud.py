"""
requirements-cloud.txt (the deployed backend) agrees with requirements.txt (V13).

requirements.txt is the full pip-freeze file of the local work; the cloud list
is a subset of it. A package of the cloud list that is missing from the full
list, or pinned to another version there, means the two environments have
drifted apart.
"""

import re
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"torch", "faiss-cpu", "sentence-transformers", "transformers", "rank-bm25", "numpy"}


def pins(path: Path) -> Dict[str, str]:
    """Package (normalized name) -> pinned version, for every `name==version` line."""
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s;]+)", line)
        assert match, f"{path.name}: not an exact pin: {line!r}"
        out[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(3)
    return out


def test_every_cloud_package_is_in_requirements_with_the_same_version():
    full = pins(REPO_ROOT / "requirements.txt")
    cloud = pins(REPO_ROOT / "requirements-cloud.txt")

    assert cloud, "requirements-cloud.txt is empty"
    missing = sorted(name for name in cloud if name not in full)
    different = sorted(f"{name} {cloud[name]} != {full[name]}" for name in cloud
                       if name in full and cloud[name] != full[name])
    assert missing == [] and different == []


def test_cloud_list_has_the_backend_and_no_model_library():
    cloud = pins(REPO_ROOT / "requirements-cloud.txt")

    assert {"fastapi", "psycopg", "psycopg-binary", "psycopg-pool", "httpx", "mcp", "python-dotenv"} <= set(cloud)
    assert FORBIDDEN.isdisjoint(cloud)
