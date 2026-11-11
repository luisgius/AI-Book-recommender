"""
Single import point for FAISS in the search adapters.

On macOS the faiss-cpu and torch wheels each bundle their own OpenMP runtime
(libomp.dylib). If faiss is imported first, loading a large
sentence-transformers model afterwards (e.g. intfloat/multilingual-e5-large-
instruct) crashes the process with a segmentation fault while the weights are
read. Importing torch first avoids the crash, so the adapters import faiss
from this module instead of importing it directly:

    from app.infrastructure.search.faiss_runtime import faiss
"""

import torch  # noqa: F401  (must be loaded before faiss, see module docstring)
import faiss

__all__ = ["faiss"]
