# ADR 003: FAISS for Vector Indexing

## Status

Accepted

## Context

Our hybrid search system requires a vector index to enable semantic search.
When a user searches for "mystery novels set in Victorian England", we need
to find books with similar semantic meaning, not just keyword matches.

The `EmbeddingsStore` port in `app/domain/ports.py` defines the contract for:
- Generating embeddings from text
- Storing embeddings associated with book UUIDs
- Building and persisting an ANN (Approximate Nearest Neighbor) index

We need a concrete implementation that:
1. Indexes embeddings generated from book metadata (title, description, categories)
2. Performs fast nearest-neighbor search at query time
3. Runs locally without external services (academic project constraint)
4. Is simple enough to explain in a thesis

## Decision

We will use **FAISS** (Facebook AI Similarity Search) for vector indexing,
specifically:

- **MVP**: `IndexFlatL2` for exact nearest-neighbor search
- **Future**: `IndexHNSWFlat` for approximate search if the catalog grows

For embedding generation, we will use **sentence-transformers** with the
`all-MiniLM-L6-v2` model (384 dimensions, good quality/speed balance).

### Implementation Details

The adapter `EmbeddingsStoreFaiss` in `app/infrastructure/search/embeddings_store_faiss.py`:
- Implements the `EmbeddingsStore` Protocol from `app/domain/ports.py`
- Uses `IndexFlatL2` for exact L2 (Euclidean) distance search
- Maintains a UUID string -> FAISS index position mapping
- Persists three files: `faiss_index.bin`, `faiss_id_mapping.json`, `faiss_embeddings.json`

## Alternatives Considered

### 1. Pinecone / Weaviate / Other Managed Vector DBs

**Pros**: Scalable, feature-rich, managed infrastructure
**Cons**: Requires external service, costs money, overkill for our scale

**Decision**: Rejected. We prefer local solutions for this academic project.

### 2. Annoy (Spotify)

**Pros**: Simple API, good for read-heavy workloads
**Cons**: Cannot add vectors after building (requires full rebuild)

**Decision**: Rejected. FAISS offers more flexibility for incremental updates.

### 3. Hnswlib

**Pros**: Pure HNSW implementation, very fast
**Cons**: Less mature ecosystem than FAISS

**Decision**: Rejected. FAISS has better documentation and more index options.

### 4. ChromaDB / LanceDB

**Pros**: Higher-level API, built-in persistence
**Cons**: Additional abstraction layer, less control

**Decision**: Rejected. For a thesis, understanding the underlying mechanics
(FAISS directly) is more valuable than convenience.

## Index Types in FAISS

| Index Type | Search | Memory | Build Time | Use Case |
|------------|--------|--------|------------|----------|
| IndexFlatL2 | Exact O(n) | Low | Fast | < 10k vectors |
| IndexIVFFlat | Approximate | Medium | Medium | 10k - 1M vectors |
| IndexHNSWFlat | Approximate | High | Slow | Any size, best recall |

### IndexFlatL2 (Our MVP Choice)

- **How it works**: Brute-force comparison against all vectors
- **Pros**: Exact results, no training needed, simple to understand
- **Cons**: Slow for large datasets (linear scan)
- **When to use**: Datasets under 10,000 vectors
- **Complexity**: O(n) per query

### IndexHNSWFlat (Future Option)

- **How it works**: Graph-based navigation, each vector connected to neighbors
- **Pros**: O(log n) search, high recall (>95%)
- **Cons**: More memory, slower to build
- **When to use**: Datasets over 10,000 vectors

## Distance Metric Decision

### MVP: L2 (Euclidean) Distance without Normalization

We use `IndexFlatL2` with raw embeddings:
- Simpler implementation
- No preprocessing required
- Works well for sentence-transformers models

### Alternative: Cosine Similarity (for future consideration)

To use cosine similarity with FAISS:
1. Normalize all embeddings to unit length: `emb = emb / ||emb||`
2. Use `IndexFlatIP` (inner product) instead of `IndexFlatL2`

With normalized vectors: `inner_product(a, b) == cosine_similarity(a, b)`

**Trade-off**: Cosine similarity is often preferred for semantic similarity
because it measures angle rather than magnitude. However, for our MVP with
sentence-transformers (which already produces well-scaled embeddings), L2
distance works adequately.

The helper function `_normalize_embeddings()` is provided in the implementation
for future use if we decide to switch to cosine similarity.

## Embedding Model Choice

We use `all-MiniLM-L6-v2` from sentence-transformers:

| Property | Value |
|----------|-------|
| Dimension | 384 |
| Speed | ~14,000 sentences/second on CPU |
| Model size | ~80MB |
| Quality | Good for semantic similarity tasks |

**Alternatives for future consideration:**
- `all-mpnet-base-v2`: Better quality, slower (768d)
- `paraphrase-MiniLM-L3-v2`: Faster, lower quality (384d)
- Multilingual models if non-English support is needed

## Persistence Strategy

The FAISS index and related data are saved to a directory with three files:

```
data/
  faiss_index.bin        # FAISS index binary (faiss.write_index)
  faiss_id_mapping.json  # List: FAISS position -> book UUID string
  faiss_embeddings.json  # Dict: book UUID string -> embedding vector
```

### Why three files?

1. **faiss_index.bin**: The actual FAISS index for fast search
2. **faiss_id_mapping.json**: Maps FAISS internal indices (0, 1, 2...) back to book UUIDs
3. **faiss_embeddings.json**: Allows `get_embedding()` to work after load, and enables
   index rebuilding without re-generating embeddings

### Rebuild Strategy

For the MVP, we rebuild the entire index during ingestion via `build_index()`.
This is acceptable because:
- Catalog size is small (<10k books)
- Ingestion is a batch operation, not real-time
- `IndexFlatL2` handles rebuilds efficiently

For larger datasets, FAISS supports incremental additions, but this adds complexity.

## Reproducibility

To ensure reproducible results across runs (important for evaluation):
- The `_id_mapping` is sorted by UUID string before building the index
- This guarantees the same FAISS internal positions for the same set of books

## Consequences

### Positive

- Simple, local solution that meets all requirements
- No external dependencies or services
- Easy to explain in academic context
- Good performance for expected catalog size (< 10k books)
- Clear upgrade path to HNSW if needed
- Protocol-based design allows swapping implementations

### Negative

- `IndexFlatL2` will become slow if catalog grows beyond ~50k books
- sentence-transformers model download required on first run (~80MB)
- FAISS installation can be tricky on some platforms (use `faiss-cpu`)
- L2 distance may be suboptimal vs cosine for some queries

## Dependencies

```
faiss-cpu
sentence-transformers
```

**Note**: Use `faiss-cpu` (not `faiss-gpu`) for simplicity. The GPU version
requires CUDA and is unnecessary for datasets under 100k vectors.

## References

- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FAISS Index Selection Guide](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [all-MiniLM-L6-v2 Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
