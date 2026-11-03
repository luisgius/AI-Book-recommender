# Sistema Inteligente de Recomendacion de Libros mediante PLN y LLMs

**TFG (Bachelor's Thesis) - Hybrid Book Recommendation System**

A production-style book recommendation system demonstrating modern AI Engineering skills including RAG pipelines, LangChain/LangGraph, embeddings, and LLM evaluation.

## Project Goals

This project serves two purposes:

1. **Academic**: A Bachelor's Thesis demonstrating hybrid search and LLM-powered recommendations.
2. **Professional**: A portfolio project showcasing junior AI Engineer skills.

### AI Engineering Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| **RAG Pipelines** | BM25 + vector retrieval with LLM-generated explanations |
| **LangChain** | Chains for prompt management and LLM orchestration |
| **LangGraph** | Stateful flows for query understanding and agentic search |
| **Embeddings** | sentence-transformers for semantic search |
| **FAISS** | Approximate nearest neighbor search |
| **Prompt Engineering** | Versioned prompts with Pydantic structured outputs |
| **Agentic Patterns** | Tool use, ReAct, self-reflection |
| **LLM Evaluation** | IR metrics + LLM-as-judge |
| **Pydantic** | Structured outputs, API schemas, configuration |

## Architecture

**Hexagonal Architecture (Ports & Adapters)**

```
                    +------------------+
                    |   FastAPI API    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  Domain Layer    |
                    |  - SearchService |
                    |  - Ports (interfaces)
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------v-------+   +--------v--------+   +------v------+
| BM25 Search   |   | FAISS Vectors   |   | LangChain   |
| (rank-bm25)   |   | (embeddings)    |   | LLM Client  |
+---------------+   +-----------------+   +-------------+
```

## Project Structure

```
app/
├── domain/                 # Core business logic (framework-agnostic)
│   ├── entities.py         # Book, SearchResult, Explanation
│   ├── services.py         # SearchService with RRF fusion
│   ├── ports.py            # Interface definitions
│   └── value_objects.py    # SearchQuery, SearchFilters
├── infrastructure/         # Adapters implementing ports
│   ├── search/             # BM25, FAISS, embeddings
│   ├── llm/                # LangChain client, prompts, graphs
│   ├── db/                 # SQLite repository
│   └── external/           # Google Books API client
├── evaluation/             # IR metrics, LLM-as-judge
└── api/                    # FastAPI endpoints
```

## Installation

```bash
# Clone and setup
git clone <repo-url>
cd luis-tfg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### LLM (optional; local by default)

Query understanding and grounded explanations use an LLM through LangChain.
By default the app uses a local [Ollama](https://ollama.com) server, so no API
key is needed:

```bash
ollama pull qwen3.8:27b-mlx        # or any model that supports structured output
cp .env.example .env               # optional: change model, limits, provider
```

All settings are environment variables (see `.env.example` and
`app/infrastructure/config/settings.py`): `LLM_PROVIDER` (`ollama`, `openai`,
`anthropic` or `none`), `LLM_MODEL`, `LLM_BASE_URL`, `LLM_MAX_TOKENS`,
`LLM_TIMEOUT_S`, `LLM_MAX_EXPLANATIONS`, among others.

If the LLM is disabled (`LLM_PROVIDER=none`) or unreachable, the API keeps
serving plain hybrid search: understanding falls back to the original query
and explanations are skipped or replaced by a fallback message.
`GET /api/v1/health` reports the LLM status separately from the search
components.

To use a hosted model instead, set `LLM_PROVIDER=openai` (needs
`OPENAI_API_KEY`) or `LLM_PROVIDER=anthropic` (needs `ANTHROPIC_API_KEY` and
`pip install langchain-anthropic`). Never commit `.env`.

### Web UI

Start the API (`python -m app.main` or `uvicorn app.main:app`) and open
`http://localhost:8000/ui/`: a single static page (no build step, no external
assets) with the search box, the LLM options, explicit filters, the understood
query intent, explanations with their verified citations, and "similar books".
With a local 27B model, a search with understanding and three explanations
takes around a minute and a half.

### Search API options

`POST /api/v1/search` accepts, besides `text`, `filters` and `max_results`:

- `use_understanding`: the LLM classifies the intent, extracts filters and
  reformulates the query; retrieval then fuses several query variations.
  Filters set explicitly in `filters` take precedence over extracted ones. An
  extracted language is kept only if the query names it, and an extracted
  category is a soft signal unless `LLM_APPLY_CATEGORY_FILTER=true`.
- `use_explanations`: the top `LLM_MAX_EXPLANATIONS` results get a grounded
  explanation with citations verified against the book data.
- `use_diversification` / `diversity_lambda`: MMR re-ranking.
- `include_metadata`: fusion details and candidate counts.

Other endpoints:

- `GET /api/v1/books/{book_id}/similar?max_results=10&language=es`: item-to-item
  recommendations by vector similarity.
- `POST /api/v1/evaluate` with `{"modes": ["hybrid_rrf"], "max_results": 100}`:
  runs the retrieval evaluation on the bundled test collection (44 queries) and
  returns nDCG@k, Recall@k, Precision@k, MRR and ILD@k per mode. It computes
  the same numbers as the CLI job; LLM-based evaluation (query understanding,
  LLM-as-judge) stays in the CLI job because it takes minutes.
- `GET /api/v1/books/{book_id}` and `GET /api/v1/health`.

## Usage

```bash
# Run tests
pytest

# Run API server
python -m app.main

# Ingest books from Google Books
python -m scripts.ingest_and_build --query "machine learning" --max-items 50

# Run evaluation
python -m app.evaluation.evaluation_job --output data/evaluation/results.json

# Run evaluation (v2, reproducible)
python -m app.evaluation.evaluation_job \
  --db-path data/catalog.db \
  --indexes-dir data/indexes \
  --output data/evaluation/results_v2.json \
  --export-pool data/evaluation/pool_candidates_v2.json \
  --pool-per-mode-limit 50
```

### Evaluation artifacts (evidence)

- **results_v2.json**: `data/evaluation/results_v2.json`
- **pool_candidates_v2.json**: `data/evaluation/pool_candidates_v2.json`
- **comparison_v1_v2.txt**: `data/evaluation/comparison_v1_v2.txt`
- **EVALUATION_SUMMARY.md**: `data/evaluation/EVALUATION_SUMMARY.md`

### FAISS dependency note (faiss-cpu vs faiss)

- This project imports the Python module `faiss`.
- On pip, the package that provides `import faiss` is typically **`faiss-cpu`** (CPU-only).
- `requirements.txt` pins: `faiss-cpu==1.9.0.post1`.
- In some environments (often macOS/conda), FAISS is installed via conda-forge (e.g., `faiss-cpu`), but it still exposes the same Python module name: `faiss`.

## Key Components

### Hybrid Search (SearchService)

Combines lexical and semantic search using Reciprocal Rank Fusion:

```python
# Retrieval
lexical_results = bm25_repo.search(query, max_results=20)
vector_results = faiss_repo.search(query_embedding, max_results=20)

# Fusion
fused = rrf_fusion(lexical_results, vector_results, k=60)
```

### RAG Pipeline

```
User Query --> Query Understanding --> Hybrid Retrieval --> Context Construction --> LLM Generation
                (LangGraph)            (BM25 + FAISS)        (top-k books)          (explanation)
```

### LLM Integration (LangChain)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chain = prompt | llm.with_structured_output(BookExplanation)
explanation = chain.invoke({"query": query, "book": book_info})
```

## Technologies

| Category | Technology |
|----------|------------|
| Language | Python 3.11+ |
| Web Framework | FastAPI |
| Database | SQLite |
| Lexical Search | rank-bm25 |
| Vector Search | FAISS |
| Embeddings | sentence-transformers |
| LLM Orchestration | LangChain, LangGraph |
| LLM Backends | OpenAI API, Anthropic API |
| Validation | Pydantic |
| Testing | pytest |

## Development Status

- [x] Phase 1: Retrieval Foundation (BM25, FAISS, hybrid search)
- [x] Phase 2: Basic RAG Pipeline (LangChain client, grounded explanations, citation guardrails)
- [x] Phase 3: Query Understanding (intent extraction, filter extraction, LangGraph)
- [ ] Phase 4: Agentic Patterns (tool use, ReAct)
- [x] Phase 5: Evaluation Pipeline (IR metrics, LLM-as-judge, citation metrics)
- [~] Phase 6: API and Integration (FastAPI endpoints done, UI and Docker pending)

## Author

Luis Gimenez - TFG 2025
