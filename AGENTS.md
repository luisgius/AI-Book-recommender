# AGENTS.md

## Purpose
- This file guides agentic coding assistants working in this repo.
- Keep changes minimal, clear, and easy to explain in a thesis report.
- Follow hexagonal architecture and grounded LLM behavior rules.

## Repository Snapshot
- Language: Python 3.11+
- Framework: FastAPI
- Testing: pytest
- Key deps: LangChain, LangGraph, FAISS, sentence-transformers, Pydantic
- Architecture: Ports & Adapters (domain independent from infrastructure)

## Build / Run / Test / Lint
```bash
# Install deps
pip install -r requirements.txt

# Run API server
python -m app.main

# Run ingestion job
python -m app.ingestion.ingest_books_job --query "machine learning" --max-results 50

# Run evaluation (v1)
python -m app.evaluation.evaluation_job --output data/evaluation/results.json

# Run evaluation (v2, reproducible)
python -m app.evaluation.evaluation_job \
  --db-path data/catalog.db \
  --indexes-dir data/indexes \
  --output data/evaluation/results_v2.json \
  --export-pool data/evaluation/pool_candidates_v2.json \
  --pool-per-mode-limit 50

# Run all tests
pytest

# Run single test file
pytest tests/infrastructure/llm/test_guardrails.py

# Run single test by node id
pytest tests/infrastructure/llm/test_guardrails.py::test_is_valid_snippet_exact_match

# Run tests by keyword
pytest -k "guardrails"

# Coverage (if needed)
pytest --cov=app --cov-report=term-missing

# Format (available dependency)
black .

# Lint (available dependency)
flake8 app tests

# Type check (available dependency)
mypy app
```

## Docker
- `Dockerfile` is currently a stub (commented TODO).
- If you need containers, confirm the Dockerfile is implemented first.
- Typical build command when ready: `docker build -t ai-book-recommender -f Dockerfile .`

## Editor Rules
- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` were found.

## Formatting and Imports
- Use Black-style formatting (4-space indents, trailing commas where appropriate).
- Keep lines reasonably short; Black will reflow.
- Use standard import grouping: stdlib, third-party, local (`app.*`).
- Separate import groups with a single blank line.

## Types and Models
- Use type hints on public functions and methods.
- Prefer `@dataclass` for domain entities/value objects.
- Use Pydantic `BaseModel` only at boundaries (API/LLM/infrastructure).
- Domain must not depend on FastAPI, LangChain, FAISS, or DB adapters.
- Convert between domain types and Pydantic at adapter boundaries.

## Naming Conventions
- Modules, functions, and variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Use descriptive names; avoid one-letter variables unless idiomatic.

## Documentation Style
- Add concise docstrings for classes and public methods.
- Keep comments minimal; prefer clear code over inline commentary.

## Error Handling
- Catch expected failures at adapter boundaries and re-raise meaningful exceptions.
- Preserve root causes with `raise ... from e`.
- Log warnings/errors with context (query, source_id, etc.).
- Avoid silent failure; return graceful fallbacks where specified.

## Logging and Observability
- Use structured, consistent logging (avoid print).
- Log LLM calls with prompt version, model, inputs, outputs, latency when implemented.
- Track evaluation artifacts in `data/evaluation/`.

## LLM and RAG Rules
- Prefer RAG pipelines: retrieve → build context → generate.
- All LLM explanations must be grounded in retrieved evidence.
- Use citations with `book_id`, `chunk_id`, and `snippet`.
- `snippet` must be a substring of the cited field (deterministic validation).
- If no valid citations remain, return a safe fallback explanation.
- Do not use LLM fine-tuning; use API models via LangChain.

## Prompting
- Store prompts in `app/infrastructure/llm/prompts.py` or templates.
- Version prompts explicitly (e.g., `PROMPT_VERSIONS`).
- Use Pydantic schemas for structured output parsing.
- Keep prompt logic out of domain services.

## Retrieval and Ranking
- Hybrid retrieval uses BM25 + FAISS with RRF fusion.
- Do not compare RRF scores across queries.
- Avoid using score thresholds for fallback decisions; use robust signals.
- MMR diversification is optional but supported.

## Architecture Constraints
- Keep the hexagonal architecture boundaries intact.
- Domain layer holds business logic and interfaces (ports).
- Infrastructure layer implements ports (DB, search, LLM, external providers).
- API layer should only orchestrate and convert; no core logic.

## Testing Guidelines
- Use pytest fixtures for shared test data and LLM mocking.
- Prefer small, deterministic unit tests.
- Name tests `test_*` and files `test_*.py`.
- Structure tests with clear arrange/act/assert blocks.
- `conftest.py` at repo root adds project to `sys.path` for imports.

## Data and Security
- Never commit `.env` or API keys.
- Use `.env.example` as the template.
- Avoid loading untrusted pickle files; treat BM25 pickles as trusted only.
- Keep evaluation artifacts and indexes in `data/`.

## Repo Hygiene
- Keep changes focused; do not refactor unrelated code.
- Update README or docs only when behavior changes.
- Avoid emoji usage in code or docs.

## Quick File Map
- `app/domain/`: entities, value objects, services, ports
- `app/infrastructure/`: adapters for search, LLM, DB, external APIs
- `app/api/`: FastAPI endpoints and schemas
- `app/evaluation/`: metrics, evaluation jobs, judging
- `tests/`: pytest suite
- `scripts/`: ingestion and utility CLI scripts
- `data/`: SQLite catalog, indexes (BM25/FAISS), evaluation artifacts
- `docs/`: architecture decisions and technical documentation

## When Adding Features
- Keep the system explainable and thesis-friendly.
- Prefer simple, well-justified implementations over clever hacks.
- Document assumptions and limitations in code or docs when needed.

## Contact Points
- For system behavior and constraints, read `CLAUDE_updated.md`.
- For usage commands and setup, read `README.md`.

## Notes for Agents
- Ask before running expensive jobs (evaluation, ingestion).
- Prefer single-test runs when iterating.
- Keep outputs reproducible and deterministic where possible.

## Non-Goals
- Do not introduce fine-tuning workflows.
- Do not replace FAISS with Chroma unless explicitly requested.
- Do not add new frameworks without strong justification.

## Style Summary (Checklist)
- Domain types: dataclasses, no Pydantic.
- Boundary types: Pydantic models.
- Grounded explanations with validated citations.
- Prompt versions tracked.
- Hexagonal architecture preserved.
- No emojis.

## End
