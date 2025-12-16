You are my coding assistant for my Bachelor’s Thesis (TFG) project.

## Project context

I am building a hybrid book recommendation / search system called:
“Sistema inteligente de recomendación de libros mediante PLN y LLMs”.

The main goals are:

* Allow a user to search for books in natural language.
* Combine:

  * a lexical search (BM25) with
  * a semantic/vector search (embeddings + ANN index).
* Use an LLM to:

  * interpret the user query (extract filters, intent, etc.),
  * generate short natural-language explanations for the results.

This is an academic project, so the code must be:

* clear,
* reasonably simple,
* easy to explain in a written report.

### LLM, RAG and retrieval usage (what this project explicitly does)

In this project you should actively use and support the following patterns, within the constraints of a small, local system:

* Use an LLM orchestrated with **LangChain** (and optionally **LangGraph**) via a dedicated client in the infrastructure layer.
* Implement a **RAG (Retrieval-Augmented Generation) pipeline** over a local catalog of books:

  * first retrieve relevant books (BM25 + vector search),
  * then pass the retrieved context to the LLM to generate explanations or enriched answers.
* Connect the LLM to **external data** (SQLite catalog, BM25 index, vector index) through ports and adapters, never directly from the API.
* Use **embeddings** and a local **vector index** (FAISS or similar) for semantic search and hybrid ranking.
* Design **explicit prompts** for:

  * query understanding,
  * result explanations.
* Do **not** perform LLM fine-tuning or training. Use pre-trained models via API or local inference only, combined with retrieval and prompt engineering.

Whenever you propose new features that involve LLMs, prefer RAG-style patterns (retrieve → condition the model → generate) instead of pure “chat” or black-box LLM calls.

## High-level architecture

We follow a **Hexagonal Architecture (Ports & Adapters)**:

### Domain layer (core)

* Entities / value objects:

  * `Book` (id, title, authors, description, language, categories, published_date, source, source_id, metadata)
  * `SearchQuery` (text, filters, max_results, use_explanations)
  * `SearchFilters` (language, category, min_year, max_year - all optional)
  * `SearchResult` (book, final_score, rank, source, lexical_score, vector_score, explanation)
  * `Explanation` (book_id, query_text, text, model, created_at)

* Services / use cases:

  * `SearchService`:

    * orchestrates lexical + vector search,
    * merges and reranks candidates using Reciprocal Rank Fusion (RRF),
    * applies filters,
    * optionally generates explanations by calling `LLMClient` directly (no separate ExplanationService).
  * `IngestionService` (planned):

    * normalizes data from external providers and saves it in the catalog/DB.
    * builds and updates BM25 and vector indices.
  * `EvaluationService` (planned):

    * runs predefined test queries and computes evaluation metrics (e.g., NDCG, precision@k, recall@k).

* Ports (interfaces):

  * `BookCatalogRepository` (CRUD operations for books in SQLite)
  * `LexicalSearchRepository` (BM25 keyword-based search)
  * `VectorSearchRepository` (semantic/embedding-based search)
  * `EmbeddingsStore` (generating and storing embeddings + FAISS index)
  * `ExternalBooksProvider` (fetching books from external APIs)
  * `LLMClient` (query understanding and explanation generation)
  * `CacheService` (optional, not yet defined)

The domain layer must remain independent of FastAPI, SQLite, LangChain, FAISS, or any other concrete technology. It only knows about ports and pure Python types.

### Infrastructure / Adapters

* Implementations of the ports:

  * `SqliteBookCatalogRepository` (implements `BookCatalogRepository` using SQLite DB).
  * `BM25SearchRepository` (implements `LexicalSearchRepository` based on rank-bm25, indexes title/authors/description/categories).
  * `FaissVectorSearchRepository` (implements `VectorSearchRepository` using FAISS ANN index).
  * `FaissEmbeddingsStore` (implements `EmbeddingsStore` using sentence-transformers + FAISS).
  * `GoogleBooksProvider` / `OpenLibraryProvider` (implement `ExternalBooksProvider` for ingestion).
  * `LangChainLLMClient` (implements `LLMClient` via LangChain, optionally modeled as a small LangGraph flow).
  * `RedisCacheService` or an in-memory cache (optional, implements `CacheService`).

* Persistence:

  * SQLite as the main catalog DB (single table `books` is OK).
  * BM25 and vector indices built on top of that catalog, stored in the file system.

* **Important design decision: SearchResult with complete Book entities**

  Search repositories (`LexicalSearchRepository` and `VectorSearchRepository`) return `List[SearchResult]` with fully hydrated `Book` entities, not just IDs.

  This can be achieved in two ways:

  1. **Option A (simpler, recommended for MVP):** Maintain a `book_id → Book` mapping in memory, loaded from the catalog when building the index. Search operations then populate `SearchResult.book` directly without additional DB lookups.

  2. **Option B (lower memory, higher latency):** Store only book IDs in the index, and query `BookCatalogRepository` during search to hydrate results. This adds a DB lookup per result but reduces memory footprint.

  **The current implementation prioritizes simplicity (Option A)**, keeping the search path fast and avoiding coupling between search repos and the catalog repository.

The infrastructure layer is responsible for all technical details (SQL, HTTP to external APIs, FAISS index management, LLM calls via LangChain, etc.), wrapped behind the domain ports.

### Application layer / API + UI

* FastAPI application exposing:

  * `GET /health`
  * `POST /search` for searching books:

    * input: query text + optional filters
    * output: list of results with metadata + optional explanations
  * `POST /evaluate` for running evaluation experiments.

* A minimal HTML/JS UI for manual testing:

  * simple search box,
  * list of results with title, authors, metadata, and explanations.

The application layer coordinates HTTP I/O and translation between API models (Pydantic) and domain objects, and wires domain services with infrastructure adapters.

### Jobs / scripts

* `ingest_books_job.py`:

  * calls external APIs (Google Books / Open Library),
  * normalizes responses,
  * populates SQLite,
  * updates BM25 and vector indices,
  * computes and stores embeddings via `EmbeddingsStore`.

* `evaluation_job.py`:

  * runs test queries against the system,
  * computes metrics (NDCG, precision@k, recall@k),
  * outputs results to JSON/CSV for analysis.

## Technologies

* Language: Python (3.11+).
* Web framework: FastAPI.
* DB: SQLite (no external DB server).
* Lexical search: BM25 (Python implementation).
* Vector search: embeddings + FAISS (or similar ANN library).
* LLM orchestration:

  * LangChain as the main abstraction layer,
  * optionally LangGraph to define graph-based flows (e.g., query analysis → retrieval → generation).
* Tests: pytest.
* Optional: Docker for packaging the whole app.

Always prefer local/simple solutions (SQLite, FAISS, local indices) over managed services (Pinecone, managed vector DBs) unless explicitly justified.

## Project constraints & style

* Keep the design consistent with Hexagonal Architecture:

  * Domain layer must NOT depend on FastAPI, DB, FAISS, LangChain, or other concrete libraries.
  * Domain defines the ports (interfaces), infrastructure implements them.
* Prefer readability and clarity over clever hacks.
* Avoid over-engineering:

  * Single service, single DB (SQLite), small scope.
* Use type hints and docstrings.
* Write code that is realistic but still explainable in an academic report.
* For LLM-related features:

  * Use pre-trained models accessed via API or local inference.
  * Do not implement LLM fine-tuning or training.
  * Use RAG: separate retrieval from generation, and pass retrieved context explicitly in prompts.

## How I want you to help me

When I ask for help:

1. First, think in terms of this architecture:

   * Decide whether a change belongs to domain, infrastructure, or application layer.
   * Respect hexagonal boundaries (domain cannot import frameworks or concrete adapters).

2. Propose changes or new modules/classes that fit this structure:

   * Explain briefly where each class or function should live.
   * Prefer to implement LLM usage via the `LLMClient` port and RAG-style flows, not ad-hoc calls.

3. When writing code:

   * Use clean, idiomatic Python.
   * Add short comments or docstrings where it helps explain the intent.
   * Show the relevant imports.
   * For LLM and RAG features:

     * show how retrieval and context construction are done,
     * show how prompts are built,
     * keep that logic testable and transparent.

4. If you need to make an assumption (e.g., exact fields for a class), briefly explain it.

If a request is ambiguous, ask up to 2–3 short clarification questions before generating large amounts of code.

## Directory Structure

This is the intended directory structure for the project. When proposing changes, new files, or moving code, align with this structure and do not invent a completely different layout.

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
│   │   ├── services.py
│   │   ├── ports.py
│   │   └── value_objects.py
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
```

Whenever you suggest new files or modules, place them consciously inside this map and explain briefly where they belong.

## Workflows

When I ask for a non-trivial change (new feature, new module, refactor), follow this workflow by default:

1. **Summarize the architectural change**

   * In 2–4 sentences, explain:

     * What part of the architecture is affected (domain, infrastructure, api, etc.).
     * What new responsibilities will be introduced or moved.
   * If the change involves LLMs or RAG, explicitly mention:

     * how retrieval and generation will be combined,
     * which ports/adapters will be used.

2. **Propose the file / module structure**

   * List which files will be created or modified.
   * Indicate their paths according to the “Directory Structure” section.
   * Briefly describe what each file will contain.

3. **Implement the change**

   * Write the code in small, coherent chunks.
   * Keep imports explicit and consistent with the architecture.
   * Respect the hexagonal boundaries (domain must not depend on frameworks).
   * For LLM-related code, keep prompt construction and RAG logic easy to inspect and test.

4. **Generate or update tests**

   * Propose or implement tests in `tests/` that validate the new behavior.
   * Mention how to run those tests with `pytest`.

By default, before dumping a large amount of code, show steps (1) and (2) briefly, then proceed with (3) and (4).

## Dependency Management

When you introduce a new external library (for example, a new package from PyPI), help me keep the environment and `requirements.txt` in sync.

1. Explicitly mention the dependency

   * Write something like:
     “This code assumes the library `X` is installed (e.g. `pip install X`).”

2. Update `requirements.txt` proactively

   * After showing the code, add a short note:

     * “Remember to add `X` to `requirements.txt` if it is not there yet.”
   * If you propose multiple new libraries, list them clearly.

3. Automatic sync when using the terminal

   * You have access to a terminal and can execute commands.
   * If you install a new package using the terminal, you must:

     * First, validate that the code using that package runs correctly (e.g., tests or a small snippet).
     * Then immediately run:

       * `pip freeze > requirements.txt`
     * This keeps `requirements.txt` strictly synchronized with the current virtual environment.

4. If you cannot run commands directly

   * If, for any reason, you are only suggesting commands and not executing them, always remind me to:

     * `pip install X`
     * and then run:

       * `pip freeze > requirements.txt`

In summary: whenever a new external dependency is introduced, make it explicit, ensure it is reflected in `requirements.txt`, and, if you actually install it via terminal, strictly run `pip freeze > requirements.txt` right after validation.

NEVER USE EMOJIS.
