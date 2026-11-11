"""
Cross-encoder implementation of the Reranker port (third retrieval stage).

A cross-encoder reads the query and one book together (a single transformer
pass over "query [SEP] book text") and outputs one relevance score. It is
more accurate than comparing two independently computed embeddings, but it
needs one model pass per (query, book) pair, so it only re-orders the top
fused candidates that SearchService sends it.

Design decisions:
- The model is loaded lazily, on the first call to rerank(). The API starts
  fast, tests never load it, and a configured but unused reranker costs
  nothing. A failed load is remembered: is_ready() turns False and later
  calls fail at once instead of retrying a download on every search.
- The text given to the model for each book is Book.get_searchable_text()
  (title, authors, description, categories), the same text BM25 indexes.
- The score is the raw model output (no sigmoid), requested explicitly so
  every supported model returns the same kind of number whatever its config
  says. Only the order matters for ranking; MMR min-max normalizes it.
- Results are returned as new SearchResult objects (dataclasses.replace),
  so a failure never leaves the fused list half modified.
- Offline mode (HF_HUB_OFFLINE=1): transformers 4.57.3 asks the Hub for the
  model's tags when it loads a tokenizer with a vocabulary over 100k tokens
  (both supported models), even offline, and the load fails. When offline,
  the model is therefore loaded from its folder in the local cache, which
  skips that call.

Import order: this module imports torch (also through sentence-transformers)
and never imports faiss. faiss_runtime.py explains why torch must be loaded
before faiss on macOS; importing this module first is therefore safe.

Supported models (verified with sentence-transformers 3.3.1):
- BAAI/bge-reranker-v2-m3 (multilingual, XLM-RoBERTa large, ~568M params)
- cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual, ~118M params)
Qwen/Qwen3-Reranker-0.6B needs sentence-transformers >= 5.4 (not installed).
"""

import logging
import threading
from dataclasses import replace
from typing import List, Optional

import torch
from huggingface_hub import constants as hf_constants
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder

from app.domain.entities import SearchResult
from app.domain.ports import Reranker

logger = logging.getLogger(__name__)

# One batch covers the default re-ranking depth (30 candidates).
DEFAULT_BATCH_SIZE = 32

# Tokens per (query, book) pair; longer book texts are truncated.
DEFAULT_MAX_LENGTH = 512


class CrossEncoderReranker(Reranker):
    """
    Reranker backed by a sentence-transformers CrossEncoder model.

    Example:
        >>> reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
        >>> reranked = reranker.rerank("novelas de misterio", fused_results[:30])
        >>> reranked[0].rerank_score  # highest cross-encoder score
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        """
        Configure the reranker. The model is not loaded here.

        Args:
            model_name: Hugging Face id of a cross-encoder model
            device: Torch device ("cpu", "mps", "cuda"); None lets
                sentence-transformers pick the best available one
            batch_size: (query, book) pairs scored per model pass
            max_length: Maximum tokens per pair (longer texts are truncated)
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length

        self._model: Optional[CrossEncoder] = None
        self._load_error: Optional[Exception] = None
        # Concurrent API requests must not load the model twice
        self._load_lock = threading.Lock()

    def _model_source(self) -> str:
        """
        What to pass to CrossEncoder: the model id, or its cached folder offline.

        Offline, a cached model is loaded from its local snapshot folder (see
        the module docstring). If it is not cached, the id is returned and
        CrossEncoder reports the missing model itself.
        """
        if hf_constants.HF_HUB_OFFLINE:
            try:
                return snapshot_download(self._model_name, local_files_only=True)
            except Exception:
                pass
        return self._model_name

    def _get_model(self) -> CrossEncoder:
        """Load the model on first use; re-raise a previous load failure."""
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise RuntimeError(
                    f"Reranker model '{self._model_name}' failed to load: {self._load_error}"
                ) from self._load_error
            try:
                self._model = CrossEncoder(
                    self._model_source(),
                    device=self._device,
                    max_length=self._max_length,
                )
            except Exception as e:
                self._load_error = e
                raise RuntimeError(
                    f"Reranker model '{self._model_name}' failed to load: {e}"
                ) from e
            logger.info(
                f"Loaded reranker '{self._model_name}' "
                f"(device={self._device or 'auto'}, max_length={self._max_length})"
            )
            return self._model

    def rerank(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Score each (query, book) pair with the cross-encoder and sort by score.

        Args:
            query_text: The user's original query text
            results: Candidates to re-order

        Returns:
            New SearchResult objects, best first, with rerank_score and rank
            set; the fusion scores (final, lexical, vector) are kept

        Raises:
            RuntimeError: If the model cannot be loaded
        """
        if not results:
            return []

        model = self._get_model()
        pairs = [(query_text, result.book.get_searchable_text()) for result in results]
        scores = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            activation_fct=torch.nn.Identity(),  # raw score, same for every model
            convert_to_numpy=True,
        )

        # sorted() is stable: equal scores keep the fused order
        order = sorted(range(len(results)), key=lambda i: float(scores[i]), reverse=True)
        return [
            replace(results[i], rank=position, rerank_score=float(scores[i]))
            for position, i in enumerate(order, start=1)
        ]

    def get_model_name(self) -> str:
        """Hugging Face id of the cross-encoder model."""
        return self._model_name

    def is_ready(self) -> bool:
        """True unless loading the model has failed (it loads lazily)."""
        return self._load_error is None
