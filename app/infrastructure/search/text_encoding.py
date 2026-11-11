"""
How text becomes a vector: model conventions and the text-encoder protocol.

This module imports no model library (no torch, no sentence-transformers, no
faiss), so an adapter that encodes through an HTTP API can use it in a
process that never loads a model (the cloud profile).

Contents:
- PROMPT_CONVENTIONS: how each known embedding model expects queries and
  documents to be written (prefixes, instructions), taken from each model
  card. Shared by the local sentence-transformers encoder and the API encoder,
  because the convention belongs to the model, not to where it runs.
- TextEncoder: the encoding half of the EmbeddingsStore port (domain), as an
  infrastructure protocol. EmbeddingsStoreFaiss stores and indexes vectors;
  a TextEncoder produces them. Two implementations exist: the local
  sentence-transformers encoder (embeddings_store_faiss.py) and the
  OpenRouter API encoder (openrouter_embedding_encoder.py). It is not a domain
  port: the domain only knows EmbeddingsStore, which keeps both halves.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class PromptConvention:
    """
    How a model expects queries and documents to be written before encoding.

    Attributes:
        query_prefix: Text prepended to every query.
        document_prefix: Text prepended to every document.
        query_prompt_name: Name of a prompt stored in the model's own
            sentence-transformers config (``model.prompts``), passed to
            ``encode(prompt_name=...)`` for queries. None means no stored prompt.
            Only a local model can apply it; an API has no stored prompts.
    """

    query_prefix: str = ""
    document_prefix: str = ""
    query_prompt_name: Optional[str] = None


# Task description used in the query instruction of instruction-tuned E5.
E5_INSTRUCT_TASK = "Given a book search query, retrieve relevant books"

# Query instruction of the Qwen3-Embedding family, verbatim from the prompt
# "query" stored in Qwen/Qwen3-Embedding-0.6B (config_sentence_transformers.json);
# the model card writes it as f"Instruct: {task}\nQuery:{query}" (no space
# after "Query:") and documents take no instruction. The local 0.6B model
# applies it through its stored prompt; through an API it is a text prefix,
# which gives the model the same input text.
QWEN3_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

# Per-model conventions, taken from each model card on huggingface.co.
# Models not listed here are treated as symmetric (no prompt on either side).
PROMPT_CONVENTIONS: Dict[str, PromptConvention] = {
    # Symmetric model: queries and documents are encoded the same way.
    "all-MiniLM-L6-v2": PromptConvention(),
    # Instruction on the query only; documents are encoded raw.
    "intfloat/multilingual-e5-large-instruct": PromptConvention(
        query_prefix=f"Instruct: {E5_INSTRUCT_TASK}\nQuery: ",
    ),
    # E5 family: every input starts with "query: " or "passage: ".
    "intfloat/multilingual-e5-small": PromptConvention(
        query_prefix="query: ", document_prefix="passage: "
    ),
    "intfloat/multilingual-e5-base": PromptConvention(
        query_prefix="query: ", document_prefix="passage: "
    ),
    "intfloat/multilingual-e5-large": PromptConvention(
        query_prefix="query: ", document_prefix="passage: "
    ),
    # Qwen3 ships its query instruction as the stored prompt "query".
    "Qwen/Qwen3-Embedding-0.6B": PromptConvention(query_prompt_name="query"),
    # The larger Qwen3 models through an API: the same instruction as a prefix.
    "Qwen/Qwen3-Embedding-4B": PromptConvention(query_prefix=QWEN3_QUERY_INSTRUCTION),
    "Qwen/Qwen3-Embedding-8B": PromptConvention(query_prefix=QWEN3_QUERY_INSTRUCTION),
    # BGE-M3: "no longer requires adding instructions to the queries" (model card).
    "BAAI/bge-m3": PromptConvention(),
}


def canonical_model_name(model_name: str) -> str:
    """
    Normalize a model name for comparisons.

    sentence-transformers resolves "all-MiniLM-L6-v2" to
    "sentence-transformers/all-MiniLM-L6-v2", and Hugging Face ids are not
    case sensitive, so both spellings must count as the same model.
    OpenRouter ids are the lower-case Hugging Face ids ("baai/bge-m3").
    """
    name = model_name.strip().lower()
    prefix = "sentence-transformers/"
    return name[len(prefix):] if name.startswith(prefix) else name


def prompt_convention_for(model_name: str) -> PromptConvention:
    """Return the prompt convention of a model (no prompt if unknown)."""
    wanted = canonical_model_name(model_name)
    for known_name, convention in PROMPT_CONVENTIONS.items():
        if canonical_model_name(known_name) == wanted:
            return convention
    return PromptConvention()


def model_slug(model_name: str) -> str:
    """
    File-system friendly name of a model, used to keep one index per model.

    Example: "intfloat/multilingual-e5-large-instruct"
             -> "intfloat_multilingual-e5-large-instruct"
    """
    return re.sub(r"[^a-z0-9.-]+", "_", canonical_model_name(model_name)).strip("_")


class TextEncoder(Protocol):
    """
    Turns documents and queries into vectors (the encoding half of EmbeddingsStore).

    Implementations apply the model's own query/document convention, so a
    caller never adds prefixes itself. get_model_name() is the identity that
    an index records: two encoders whose vectors are not interchangeable
    (another model, another API provider, another output dimension) must
    return different identities.
    """

    def generate_embedding(self, text: str) -> List[float]:
        """Vector of one document (book text). Raises ValueError if empty."""
        ...

    def generate_query_embedding(self, text: str) -> List[float]:
        """Vector of one search query. Raises ValueError if empty."""
        ...

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Vectors of several documents, in input order. Raises ValueError if one is empty."""
        ...

    def get_dimension(self) -> int:
        """Length of every vector this encoder returns."""
        ...

    def get_model_name(self) -> str:
        """Identity of the vectors (model, and for an API also provider and dimension)."""
        ...

    def get_device(self) -> str:
        """Where the encoding runs ("cpu", "mps", ... or "api")."""
        ...
