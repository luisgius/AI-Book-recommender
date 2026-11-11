"""
Single import point for FAISS in the search adapters.

On macOS the faiss-cpu and torch wheels each bundle their own OpenMP runtime
(libomp.dylib). If faiss is imported first, loading a large
sentence-transformers model afterwards (e.g. intfloat/multilingual-e5-large-
instruct) crashes the process with a segmentation fault while the weights are
read. Importing torch first avoids the crash, so the adapters import faiss
from this module instead of importing it directly:

    from app.infrastructure.search.faiss_runtime import faiss

An installation without torch (a deployment that encodes through an API and
loads no model) has no second OpenMP runtime and no crash to avoid, so torch
is imported only when it is installed.
"""

try:
    import torch  # noqa: F401  (must be loaded before faiss, see module docstring)
except ImportError:  # no torch installed: nothing to protect against
    pass
import faiss

__all__ = ["faiss"]
