# ADR 002: BM25 for Lexical Search in Hybrid Retrieval

## Status

Accepted

## Context

Our book recommendation system requires a hybrid search approach that combines lexical (keyword-based) and semantic (embedding-based) search methods. We need to implement the lexical component that:

1. Performs efficient keyword matching over book metadata (title, authors, description, categories)
2. Returns ranked results with relevance scores
3. Supports filtering by language, category, and publication year
4. Integrates seamlessly with our hexagonal architecture
5. Can be combined with vector search results using Reciprocal Rank Fusion (RRF)

The lexical search component must be:
- Simple enough to explain in an academic thesis
- Efficient for small-to-medium catalogs (hundreds to thousands of books)
- Well-established and understood in IR literature
- Implementable with minimal dependencies

## Decision

We will use **BM25 (Best Match 25)** as our lexical search algorithm, implemented via the `rank-bm25` Python library.

### Implementation Details

1. **Algorithm**: BM25Okapi variant
   - Probabilistic ranking function based on term frequency (TF) and inverse document frequency (IDF)
   - Includes document length normalization to avoid bias toward longer documents
   - Parameters k1=1.5, b=0.75 (library defaults, empirically validated in IR research)

2. **Index Structure**:
   - In-memory BM25 index built from `Book.get_searchable_text()`
   - Simple whitespace + lowercase tokenization (sufficient for MVP, can be enhanced)
   - Maintains `book_id → Book` mapping to return complete entities without DB lookups

3. **Search Process**:
   - Tokenize query
   - Score all documents using BM25
   - Rank by score (descending)
   - Apply post-search filters (language, category, year range)
   - Return top-k results as `SearchResult` entities

4. **Persistence**:
   - Index serialization via pickle for faster startup
   - Full rebuild on significant catalog changes
   - Incremental updates via index rebuild (acceptable for moderate catalog sizes)
   - **SECURITY NOTE**: Pickle is only safe in controlled environments (local development, trusted deployment). Never load pickle files from untrusted sources as they can execute arbitrary code during deserialization. For production systems with external file access, consider alternative serialization formats (JSON with separate embedding files, msgpack, or managed search services).

### Integration with Hybrid Search

The BM25 repository implements the `LexicalSearchRepository` port, allowing `SearchService` to:
1. Query both BM25 and vector search independently
2. Merge results using Reciprocal Rank Fusion (RRF)
3. Maintain architectural separation between retrieval methods

## Rationale

### Why BM25?

1. **Proven Effectiveness**: BM25 is the de facto standard for lexical search in information retrieval
   - Used in production systems (Elasticsearch, OpenSearch, Solr)
   - Consistently outperforms simpler TF-IDF in benchmark evaluations
   - Well-documented in academic literature (Robertson & Zaragoza, 2009)

2. **Complementary to Embeddings**:
   - BM25 excels at exact keyword matches (e.g., author names, specific titles)
   - Vector search excels at semantic similarity (e.g., "books about AI" matching deep learning)
   - Hybrid combination provides best of both worlds

3. **Simplicity**:
   - Understandable mathematical foundation (suitable for thesis explanation)
   - Easy to implement with `rank-bm25` library (single dependency)
   - No training or parameter tuning required (uses established defaults)

4. **Performance**:
   - O(n) scoring over corpus (acceptable for catalogs with <10k books)
   - In-memory operation provides fast query response
   - Can be optimized with inverted index if needed at scale

### Alternatives Considered

1. **TF-IDF**:
   - Simpler algorithm but generally less effective than BM25
   - Rejected: BM25 is a direct improvement with minimal added complexity

2. **Elasticsearch / OpenSearch**:
   - Full-featured search engines with BM25 support
   - Rejected: Adds external service dependency, overkill for project scope
   - May be considered for production scaling

3. **Full-text search in SQLite**:
   - Native FTS5 module available
   - Rejected: Less control over ranking algorithm, harder to explain scores

4. **Custom TF-IDF implementation**:
   - Educational value in implementing from scratch
   - Rejected: Reinventing the wheel, `rank-bm25` is well-tested

## Consequences

### Positive

1. **Effective Retrieval**: BM25 provides strong baseline for keyword matching
2. **Academic Credibility**: Well-known algorithm, easy to cite and explain
3. **Fast Implementation**: Library handles complexity, we focus on integration
4. **Hybrid Potential**: Scores can be combined with vector search via RRF
5. **Maintainable**: Simple codebase, minimal dependencies

### Negative

1. **Scalability Limits**: In-memory index limits catalog size to ~10-50k books
   - Mitigation: Sufficient for thesis scope; can migrate to Elasticsearch if needed
2. **Basic Tokenization**: Simple whitespace splitting may miss linguistic nuances
   - Mitigation: Can enhance with stemming/lemmatization in future iterations
3. **No Incremental Updates**: BM25Okapi requires full index rebuild
   - Mitigation: Acceptable for batch ingestion pattern; rebuild is fast for moderate sizes
4. **Language Agnostic**: Same tokenization for all languages
   - Mitigation: Works reasonably for Spanish/English; can add language-specific tokenizers later
5. **Pickle Security**: Pickle deserialization is unsafe with untrusted files
   - Mitigation: Only use in controlled local environments; document security warnings in code

### Design Trade-offs

**Chose simplicity over scalability**: For a thesis project with a local catalog, an in-memory BM25 index provides the best balance of simplicity, performance, and explainability. Production systems would typically use Elasticsearch or similar, but that would add operational complexity without educational benefit.

**Post-search filtering over query-time filtering**: Filters (language, year, category) are applied after BM25 scoring rather than during index construction. This is simpler to implement and maintain, though less efficient than filtered indices. For small-to-medium catalogs, the performance difference is negligible.

**Pickle for persistence (controlled environment only)**: Using pickle provides the simplest serialization solution for this MVP. The security limitations are acceptable given the controlled academic/local deployment context. Production deployments should use safer alternatives.

## References

- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
- `rank-bm25` library: https://github.com/dorianbrown/rank_bm25
- Elasticsearch BM25 documentation: https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html

## Notes

This ADR documents the lexical search component of our hybrid retrieval system. See ADR-003 (planned) for the vector search component and ADR-004 (planned) for the RRF fusion strategy.
