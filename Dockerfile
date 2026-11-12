# Container of the local profile with the shipped indexes (data/indexes:
# all-MiniLM-L6-v2 embeddings, BM25 without the normalizing tokenizer) and no
# re-ranking: the light configuration, not the evaluated one that `make demo`
# builds (multilingual-e5-large-instruct + bge-reranker-v2-m3). Without
# OPENROUTER_API_KEY (docker run -e ...) the LLM is off and search still works.
# The embedding model is downloaded at the first request (about 90 MB).
FROM python:3.13-slim

# Install OS dependencies needed for C extensions and healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Layer caching: install Python deps first (changes rarely). The CPU-only torch
# goes first: PyPI's Linux torch bundles several GB of CUDA libraries.
COPY requirements.txt .
RUN pip install --no-cache-dir "$(grep -E '^torch==' requirements.txt)" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code (changes frequently)
COPY app/ ./app/

# Copy default data files (~2MB, can be overridden via volume mount)
COPY data/catalog.db ./data/catalog.db
COPY data/indexes/bm25_index.pkl ./data/indexes/bm25_index.pkl
COPY data/indexes/faiss_index/ ./data/indexes/faiss_index/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Environment variables (paths read by app/api/v1/dependencies.py)
ENV DB_PATH=data/catalog.db \
    INDEXES_DIR=data/indexes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
