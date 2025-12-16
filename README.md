# CLAUDE.md

You are my coding assistant for my Bachelor’s Thesis (TFG) project.

## Project context

I am building a hybrid book recommendation / search system:

“Sistema inteligente de recomendación de libros mediante PLN y LLMs”.

Main goals:

- Allow a user to search for books in natural language.
- Combine:
  - Lexical search (BM25)
  - Semantic/vector search (embeddings + ANN index)
- Use an LLM to:
  - interpret the user query (intent + filters + reformulation),
  - generate short natural-language explanations for results (RAG-style).

This is an academic project. The code must be:

- clear,
- reasonably simple,
- easy to explain in a written report.

## What this project explicitly does (LLM / RAG / retrieval)

This project actively supports these patterns, within a small local system:

- Use an LLM orchestrated with LangChain (and optionally LangGraph) via a dedicated client in the infrastructure layer.
- Implement a local RAG pipeline:
  1. Retrieve relevant books (BM25 + vector search).
  2. Pass retrieved context to the LLM to generate explanations or enriched answers.
- Connect the LLM to external data (SQLite catalog, BM25 index, FAISS index) through ports and adapters, never directly from the API layer.
- Use embeddings and a local vector index (FAISS) for semantic search and hybrid ranking.
- Design explicit prompts for:
  - query understanding,
  - result explanations.
- Do not perform LLM fine-tuning or training. Use pre-trained models via API (or local inference) and combine with retrieval + prompt engineering.

When proposing new LLM features, prefer RAG patterns (retrieve → condition → generate) rather than pure “chat” calls.

## High-level architecture

We follow Hexagonal Architecture (Ports & Adapters).

### Domain layer (core)

Entities / value objects:

- `Book`:
  - `id` (uuid.UUID, UUIDv7)
  - `title`, `authors`, `description`, `language`, `categories`, `published_date`
  - `source`, `source_id`
  - `metadata`
  - `created_at`, `updated_at`
- `SearchQuery` (text, filters, max_results, use_explanations, etc.)
- `SearchFilters` (language, category, min_year, max_year — all optional)
- `SearchResult` (book, final_score, rank, source, lexical_score, vector_score, optional explanation)
- `Explanation` (book_id, query_text, text, model, created_at)

Services / use cases:

- `SearchService`
  - orchestrates lexical + vector search,
  - merges/reranks using Reciprocal Rank Fusion (RRF),
  - applies filters consistently,
  - optionally generates explanations via `LLMClient` (no direct LangChain in domain).
- `CatalogIngestionService`
  - fetches from `ExternalBooksProvider`,
  - persists into `BookCatalogRepository` (SQLite is source of truth),
  - rebuilds BM25 + vector indices from the full catalog.
- `EvaluationService` (planned)
  - runs predefined queries and computes metrics (NDCG, precision@k, recall@k, etc.).

Ports (interfaces), defined in `app/domain/ports.py`:

- `BookCatalogRepository`
- `LexicalSearchRepository`
- `VectorSearchRepository`
- `EmbeddingsStore`
- `ExternalBooksProvider`
- `LLMClient`
- `CacheService` (optional, not yet required)

Domain must remain independent of FastAPI, SQLite, FAISS, LangChain/LangGraph, requests, etc. Domain imports only:

- domain entities, value objects, ports, and small pure utilities.

### Infrastructure / adapters

Infrastructure implements the ports:

- `SqliteBookCatalogRepository` implements `BookCatalogRepository`
- `BM25SearchRepository` implements `LexicalSearchRepository`
- `FaissVectorSearchRepository` implements `VectorSearchRepository`
- `EmbeddingsStoreFaiss` implements `EmbeddingsStore` (sentence-transformers + FAISS)
- `GoogleBooksClient` implements `ExternalBooksProvider`
- `LangChainLLMClient` implements `LLMClient` (LangChain, optionally LangGraph)

Persistence:

- SQLite is the canonical catalog (source of truth).
- BM25 + vector indices are derived artifacts rebuilt from SQLite and stored on disk.

#### Important design decision: `SearchResult` contains fully hydrated `Book` objects

Search repositories return `List[SearchResult]` with full `Book` entities, not IDs.

Recommended approach (Option A, current design):

- Keep a `book_id -> Book` in-memory map loaded from the catalog when building indices.
- Search populates `SearchResult.book` via this map.
- Avoid per-result DB lookups during search.

### Application layer / API + UI

- FastAPI endpoints:
  - `GET /health`
  - `POST /search` (query + filters → results + optional explanations)
  - `POST /evaluate` (planned)
- Minimal UI for manual testing.

The API layer:
- translates request/response schemas (Pydantic) to domain objects,
- wires real adapters and services (composition root),
- contains no domain logic.

### Jobs / scripts

- ingestion job:
  - calls external APIs,
  - persists into SQLite,
  - rebuilds BM25 + vector indices,
  - generates embeddings.
- evaluation job:
  - runs test queries,
  - computes metrics,
  - outputs JSON/CSV.

## Technology choices

- Python 3.11+
- FastAPI
- SQLite
- BM25 (Python implementation)
- FAISS for ANN
- LangChain (required) and optionally LangGraph
- pytest
- Optional: Docker for reproducibility

Prefer local/simple solutions over managed services unless explicitly justified.

## Project constraints & style

- Keep Hexagonal boundaries strict:
  - Domain cannot import frameworks or concrete adapters.
  - Domain depends on ports; infrastructure implements ports.
- Prefer readability over cleverness.
- Avoid over-engineering; keep scope small.
- Use type hints and docstrings.
- Prompts must be explicit and inspectable.
- UUID policy:
  - Do not use UUIDv4 for new IDs.
  - Use UUIDv7 for primary keys and new entity IDs (RFC 9562).
  - Keep domain IDs as `uuid.UUID` type.

No emojis in code, docs, or assistant output.

## How I want you to help

When I ask for help:

1. Think in architecture terms:
   - Decide if the change belongs to domain / infrastructure / API.
   - Respect boundaries.
2. Propose changes that fit:
   - name files and paths,
   - explain responsibilities briefly.
3. When writing code:
   - clean, idiomatic Python,
   - explicit imports,
   - short comments/docstrings when helpful,
   - LLM code must show retrieval + context construction + prompt building explicitly.
4. If an assumption is needed, state it briefly.

If a request is ambiguous, ask up to 2–3 short clarification questions before generating large code.

## Directory structure

Align new modules with this structure (do not invent a different layout):

```text
.
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── search_endpoints.py
│   │   │   └── schemas.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── entities.py
│   │   ├── ports.py
│   │   ├── value_objects.py
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── search_service.py
│   │       ├── catalog_ingestion_service.py
│   │       └── evaluation_service.py  (planned)
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── db/
│   │   ├── search/
│   │   ├── external/
│   │   ├── llm/
│   │   ├── cache/
│   │   └── config/
│   ├── ingestion/
│   ├── evaluation/
│   └── ui/
├── tests/
├── data/
├── docs/
│   ├── arch_decisions/
│   └── notes/
├── scripts/
├── requirements.txt
├── README.md
└── Dockerfile
