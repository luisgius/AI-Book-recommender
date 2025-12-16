# Search infrastructure package
"""
Search infrastructure adapters and protocols.

This package contains:
- BM25SearchRepository: Lexical search using BM25
- EmbeddingsStoreFaiss: Embedding generation and FAISS index management
- FaissVectorSearchRepository: Vector similarity search using FAISS

Infrastructure-level Protocols:
- FaissIndexProvider: Protocol for accessing FAISS index internals
  (used by FaissVectorSearchRepository to query the index built by EmbeddingsStoreFaiss)
"""

from typing import Protocol, List, Optional, Any


class FaissIndexProvider(Protocol):
    """
    Infrastructure-level protocol for accessing FAISS index internals.

    This protocol defines the contract between FaissVectorSearchRepository and
    EmbeddingsStoreFaiss. It is NOT a domain port - it lives in the infrastructure
    layer and formalizes the coupling between these two infrastructure components.

    Why this exists:
    - The domain EmbeddingsStore port does not expose FAISS internals (correctly so)
    - FaissVectorSearchRepository needs access to the raw FAISS index for search
    - This protocol makes that coupling explicit and testable

    Design decision (MVP pragmatic approach):
    - We accept infrastructure-to-infrastructure coupling here
    - The domain layer remains clean (VectorSearchRepository port is FAISS-agnostic)
    - This protocol allows mocking in tests without depending on concrete FAISS types
    """

    def get_index(self) -> Optional[Any]:
        """
        Get the underlying FAISS index for search operations.

        Returns:
            The FAISS index (e.g., IndexFlatL2), or None if not built.
            The return type is Any to avoid importing faiss in the protocol.
        """
        ...

    def get_id_mapping(self) -> List[str]:
        """
        Get the FAISS index position -> book_id mapping.

        Returns:
            List where index i contains the book_id (UUID as string) at FAISS position i.
        """
        ...

    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            The embedding dimension (e.g., 384 for MiniLM).
        """
        ...
