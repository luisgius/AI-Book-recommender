import sys
from pathlib import Path

# Add project root to Python path so 'app' package is discoverable
sys.path.insert(0, str(Path(__file__).parent))

import os  # noqa: E402

# Loading a sentence-transformers model contacts the Hugging Face Hub to check
# for updates, even when the model is already in the local cache. With
# hundreds of tests that adds minutes and makes the suite depend on the
# network. When the default embedding model is already cached, tell the hub
# client to work offline. On a fresh machine the model is downloaded once, as
# usual, and later runs are offline.
_HF_HUB_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if (_HF_HUB_CACHE / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
