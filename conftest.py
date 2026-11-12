import sys
from pathlib import Path

# Add project root to Python path so 'app' package is discoverable
sys.path.insert(0, str(Path(__file__).parent))

import hashlib  # noqa: E402
import os  # noqa: E402
from typing import Dict, List  # noqa: E402

import pytest  # noqa: E402

# Loading a sentence-transformers model contacts the Hugging Face Hub to check
# for updates, even when the model is already in the local cache. With
# hundreds of tests that adds minutes and makes the suite depend on the
# network. When the default embedding model is already cached, tell the hub
# client to work offline. On a fresh machine the model is downloaded once, as
# usual, and later runs are offline.
_HF_HUB_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_HF_HUB_CACHE / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


# --- Guard: the tests never write into the published evaluation files ------------
# The evaluation's results, reports, judgments and query sets are published
# files that the tests only read; tests that build reports do it in temporary
# directories. A default output path that did not follow those directories
# once made a test overwrite a published report. This guard hashes the files
# below once before collection and once after the last test, and fails the
# run naming every file that changed, appeared or disappeared in between. It
# reads the file system only (no git), so it works anywhere the suite runs,
# and it covers any selection of tests (one file, -k, the whole suite).

_ROOT = Path(__file__).resolve().parent
# (directory, glob pattern searched recursively in it)
_GUARDED = (("data/evaluation", "*"), ("app/evaluation", "*.json"))
_SNAPSHOT = pytest.StashKey[Dict[str, str]]()


def _evaluation_files() -> Dict[str, str]:
    """{path relative to the repository: sha256 of the content} of every guarded file."""
    out: Dict[str, str] = {}
    for directory, pattern in _GUARDED:
        for path in sorted((_ROOT / directory).rglob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                with path.open("rb") as handle:
                    out[path.relative_to(_ROOT).as_posix()] = hashlib.file_digest(handle, "sha256").hexdigest()
    return out


def _changes(before: Dict[str, str], after: Dict[str, str]) -> List[str]:
    """One line per file that was modified, added or removed."""
    lines = [f"modified: {p}" for p in sorted(before.keys() & after.keys()) if before[p] != after[p]]
    lines += [f"added:    {p}" for p in sorted(after.keys() - before.keys())]
    lines += [f"removed:  {p}" for p in sorted(before.keys() - after.keys())]
    return lines


def pytest_sessionstart(session: pytest.Session) -> None:
    """Hash the guarded files before collection (a module that writes on import is caught too)."""
    session.config.stash[_SNAPSHOT] = _evaluation_files()


@pytest.fixture(scope="session", autouse=True)
def _evaluation_files_unchanged(request: pytest.FixtureRequest):
    """Fail the run if the session changed any guarded file (checked once, after the last test)."""
    before = request.config.stash.get(_SNAPSHOT, None)
    if before is None:  # sessionstart did not run for this conftest: hash now instead
        before = _evaluation_files()
    yield
    changes = _changes(before, _evaluation_files())
    if changes:
        pytest.fail(
            "The test session changed published evaluation files, which tests must only read "
            "(compared before collection and after the last test; the test named in this error is only "
            "the last one that ran, not necessarily the one that wrote):\n  " + "\n  ".join(changes)
            + "\nSome test, or code it calls, wrote to a default path instead of tmp_path. Find it by running "
            "the test files one at a time, then restore the files (git, or regenerate the reports offline).",
            pytrace=False,
        )
