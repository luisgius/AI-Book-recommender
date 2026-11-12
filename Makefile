# Makefile of the TFG "Sistema inteligente de recomendación de libros mediante
# PLN y LLMs". It gives the examiners one command for each level of checking:
#
#   make help    list the targets
#   make demo    install everything and start the web UI over the 177-book
#                local profile, with the LLM off (no API key needed);
#                make demo LLM=on uses OPENROUTER_API_KEY
#   make test    run the test suite
#   make eval    regenerate the evaluation reports from the cached judgments
#                (no API call)
#   make clean   remove only what make created
#
# Everything is installed in a virtual environment (.venv) with the standard
# venv module and pip; nothing is installed system-wide. Works with the GNU
# Make that ships with macOS (3.81) and with Linux.

# ---------------------------------------------------------------------------
# Settings (each one can be changed on the command line, e.g. make demo PORT=8001)
# ---------------------------------------------------------------------------

# Python used to create .venv: the first of these commands that is Python 3.11
# to 3.13 (3.13 is the tested version). Give another one with
# make demo PYTHON=/path/to/python3
PYTHON_CANDIDATES := python3.13 python3.12 python3.11 python3
ifndef PYTHON
PYTHON := $(shell for p in $(PYTHON_CANDIDATES); do \
	$$p -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null \
	&& { echo $$p; break; }; done)
endif

VENV := .venv
VENV_PY := $(VENV)/bin/python

# Address of the demo server
HOST := 127.0.0.1
PORT := 8000

# Configuration of the demo: the evaluated local pipeline (V06 and V13):
# BM25 with the normalizing tokenizer, multilingual-e5-large-instruct
# embeddings, RRF fusion and the bge-reranker-v2-m3 cross-encoder (system
# hybrid_norm_e5_bge_nopad of data/evaluation/BM25_PADDING.md).
# make demo RERANKER=none switches the re-ranking stage off (no 2.3 GB download,
# faster searches, but without the stage the ablation supports).
EMBEDDING_MODEL := intfloat/multilingual-e5-large-instruct
RERANKER := BAAI/bge-reranker-v2-m3

# LLM of the demo. off (default): no LLM on any machine, even one whose .env
# holds an OpenRouter key; this is what an examiner without a key gets.
# on: the OpenRouter LLM, with OPENROUTER_API_KEY (and LLM_MODEL, if set) from
# the environment or from .env.
LLM := off

# Folder of the demo indexes (built by make from data/catalog.db, ignored by git)
DEMO_INDEXES := data/indexes_demo

# Marker files written by make, so that make clean removes only what make created
VENV_MARKER := $(VENV)/.created-by-make
INSTALLED := $(VENV)/.requirements-installed
ABLATION_DIR := data/indexes_ablation
ABLATION_MARKER := $(ABLATION_DIR)/.created-by-make

# torch pin of requirements.txt (used for the CPU-only wheel on Linux, see below)
TORCH_PIN := $(shell grep -E '^torch==' requirements.txt)

# Environment of the demo server. Every value is set explicitly so that a .env
# file (read by the application, never overriding these) cannot switch the demo
# to another profile, model or LLM. LLM_PROVIDER=none keeps the LLM off even if
# .env holds a key; with LLM=on it is openrouter and the key comes from the
# environment or .env (without one the LLM stays off and the search works).
ifeq ($(LLM),on)
DEMO_LLM := LLM_PROVIDER=openrouter
else ifeq ($(LLM),off)
DEMO_LLM := LLM_PROVIDER=none
else
$(error LLM must be on or off, got '$(LLM)')
endif
DEMO_ENV := SEARCH_PROFILE=local DB_PATH=data/catalog.db INDEXES_DIR=$(DEMO_INDEXES) \
	EMBEDDING_PROVIDER=local EMBEDDING_MODEL=$(EMBEDDING_MODEL) \
	RERANKER_PROVIDER=local RERANKER_MODEL=$(RERANKER) $(DEMO_LLM)

# Evaluation commands run with the Python of .venv from the repository root
EVAL := $(VENV_PY) -m app.evaluation

.DEFAULT_GOAL := help
.PHONY: help demo serve test eval clean check-python demo-notice eval-notice

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

help:
	@echo "Targets (run from the repository root):"
	@echo "  make demo    install, build the demo indexes and start the web UI at http://$(HOST):$(PORT)/ui/"
	@echo "               (LLM off; make demo LLM=on uses OPENROUTER_API_KEY from the environment or .env)"
	@echo "               (first run downloads about 3.6 GB; make demo RERANKER=none: about 1.3 GB)"
	@echo "  make test    run the test suite (pytest)"
	@echo "  make eval    regenerate the evaluation reports in data/evaluation/ from cached judgments, no API call"
	@echo "  make clean   remove what make created (.venv if make made it, the demo and evaluation indexes)"
	@echo "Options: PYTHON=/path/to/python3  PORT=8001  RERANKER=none  LLM=on"
	@echo "Python found for .venv: $(if $(PYTHON),$(PYTHON),none (need Python 3.11 to 3.13))"

# ---------------------------------------------------------------------------
# Python and the virtual environment
# ---------------------------------------------------------------------------

check-python:
	@if [ -z "$(PYTHON)" ]; then \
		echo "Error: no Python 3.11 to 3.13 found (tried: $(PYTHON_CANDIDATES))."; \
		echo "Install Python 3.13 from https://www.python.org/downloads/ or run: make demo PYTHON=/path/to/python3"; \
		exit 1; \
	fi
	@if ! command -v "$(PYTHON)" >/dev/null 2>&1; then \
		echo "Error: PYTHON=$(PYTHON) is not a command on this machine."; exit 1; \
	fi
	@$(PYTHON) -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 13))' || { \
		echo "Error: $(PYTHON) is Python $$($(PYTHON) -c 'import platform; print(platform.python_version())')."; \
		echo "This project needs Python 3.11 to 3.13 (3.13 is the tested one; the pinned faiss-cpu"; \
		echo "has no package for 3.14). Run: make demo PYTHON=/path/to/python3.13"; \
		exit 1; }

# The virtual environment is created once. The marker file records that make
# created it (make clean removes .venv only then).
$(VENV_PY):
	@$(MAKE) --no-print-directory check-python
	@echo "Creating $(VENV) with $(PYTHON) ($$($(PYTHON) --version))"
	$(PYTHON) -m venv $(VENV)
	@touch $(VENV_MARKER)

# Python packages, installed again only when requirements.txt changes.
# On Linux, PyPI's torch package includes CUDA libraries (several GB); the
# models here run on the CPU, so the CPU-only build of the same version is
# installed first from PyTorch's index and pip then keeps it.
$(INSTALLED): requirements.txt | $(VENV_PY)
	@$(VENV_PY) -m pip --version >/dev/null 2>&1 || $(VENV_PY) -m ensurepip
	@if [ "$$(uname -s)" = "Linux" ]; then \
		echo "Linux: installing the CPU-only $(TORCH_PIN)"; \
		$(VENV_PY) -m pip install "$(TORCH_PIN)" --index-url https://download.pytorch.org/whl/cpu; \
	fi
	$(VENV_PY) -m pip install -r requirements.txt
	@touch $(INSTALLED)

# ---------------------------------------------------------------------------
# demo: the 177-book local profile in the browser
# ---------------------------------------------------------------------------

demo-notice:
	@echo "=================================================================="
	@echo " make demo: local profile, 177-book catalog (data/catalog.db)"
	@echo " Downloads the first time (later runs reuse pip's and Hugging Face's caches):"
	@echo "   Python packages (requirements.txt)          about 190 MB"
	@echo "   intfloat/multilingual-e5-large-instruct     1.14 GB (embeddings)"
	@echo "   BAAI/bge-reranker-v2-m3                     2.29 GB (re-ranking; skip with RERANKER=none)"
	@echo " Re-ranking: $(RERANKER)"
	@echo " LLM: $(LLM)$(if $(filter off,$(LLM)), (make demo LLM=on uses OPENROUTER_API_KEY from the environment or .env),)"
	@echo "=================================================================="

# The demo indexes over data/catalog.db: BM25 (normalizing tokenizer) and a
# FAISS index of the embedding model. The file below is written last by the
# build, so its presence means the indexes are complete.
DEMO_INDEX_FILE := $(DEMO_INDEXES)/faiss_index/faiss_metadata.json

$(DEMO_INDEX_FILE): data/catalog.db | $(INSTALLED)
	@echo "Building the demo indexes in $(DEMO_INDEXES) (downloads $(EMBEDDING_MODEL) the first time)"
	EMBEDDING_PROVIDER=local $(VENV_PY) -m scripts.ingest_and_build --rebuild \
		--embedding-model $(EMBEDDING_MODEL) --out-dir $(DEMO_INDEXES) --smoke-test-query "novela"

demo: demo-notice serve

# Start the server (Ctrl+C stops it). The models load on the first request.
serve: $(DEMO_INDEX_FILE)
	@$(VENV_PY) -c 'import socket; socket.create_server(("$(HOST)", $(PORT))).close()' 2>/dev/null || { \
		echo "Error: port $(PORT) is already in use on $(HOST). Run: make demo PORT=8001"; exit 1; }
	@echo ""
	@echo "Web UI:   http://$(HOST):$(PORT)/ui/"
	@echo "API docs: http://$(HOST):$(PORT)/docs   Health: http://$(HOST):$(PORT)/api/v1/health"
	@echo "The first search loads the models (several seconds). Stop the server with Ctrl+C."
	@echo "LLM $(LLM). Without the LLM the search works; query understanding, explanations and"
	@echo "the agent are unavailable, and the assistant answers each message with a plain search."
	@echo ""
	$(DEMO_ENV) $(VENV_PY) -m uvicorn app.main:app --host $(HOST) --port $(PORT)

# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------

test: $(INSTALLED)
	$(VENV_PY) -m pytest -q

# ---------------------------------------------------------------------------
# eval: regenerate the reports of data/evaluation/ without any API call
# ---------------------------------------------------------------------------
#
# Each command reads the relevance judgments and model outputs cached in the
# repository and recomputes the rankings with the local models, so the figures
# can be compared with the committed reports (git diff data/evaluation/).
# Dates, git commit and measured latencies change on every run; the metrics
# should not. Reports that need a live service (the Neon database, the public
# deployment, OpenRouter) are re-rendered from their saved results instead.
# The first run downloads five local models (about 5.2 GB).

eval-notice:
	@echo "=================================================================="
	@echo " make eval: regenerates the reports in data/evaluation/ (no API call)."
	@echo " Downloads the first time: all-MiniLM-L6-v2 0.09 GB,"
	@echo " multilingual-e5-large-instruct 1.14 GB, Qwen3-Embedding-0.6B 1.21 GB,"
	@echo " bge-reranker-v2-m3 2.29 GB, mmarco-mMiniLMv2-L12-H384-v1 0.49 GB."
	@echo " Afterwards, git diff data/evaluation/ shows what changed."
	@echo "=================================================================="

eval: eval-notice $(INSTALLED)
	@if [ ! -d $(ABLATION_DIR) ]; then mkdir -p $(ABLATION_DIR) && touch $(ABLATION_MARKER); fi
	# Baseline of the non-LLM systems on the original judgments (COMPARISON_BASELINE.md)
	$(EVAL).comparison
	# V06 retrieval ablation, judgments of the first LLM judge (RETRIEVAL_ABLATION.md)
	$(EVAL).retrieval_ablation --no-judge
	# Sensitivity of the V06 ablation to the judge (RETRIEVAL_ABLATION_JUDGE_SENSITIVITY.md): the
	# ablation on the second judge's grades, then the comparison. Both are pinned to the 919 pairs
	# both judges graded for the V06 pool: V07 later added 146 grades for other pairs to the second
	# judge's file, which stays as it is (other reports use them).
	$(EVAL).retrieval_ablation --no-judge \
		--extended-judgments-path app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json \
		--llm-pairs-from app/evaluation/relevance_judgments_extended.json \
		--output-json data/evaluation/results_retrieval_ablation_z-ai-glm-5.3-flash.json --output-md ""
	$(EVAL).judge_sensitivity report --judge-model z-ai/glm-5.3-flash
	# LLM search path from the cached query intents (results_llm_path_k10/k100.json). With every
	# intent cached no LLM is called; the LLM labels come from the saved results, not the environment.
	$(EVAL).llm_path_evaluation --max-results 10 \
		--intents-cache data/evaluation/llm_intents_qwen3.8-27b-mlx.json --output data/evaluation/results_llm_path_k10.json
	$(EVAL).llm_path_evaluation --max-results 100 \
		--intents-cache data/evaluation/llm_intents_qwen3.8-27b-mlx.json --output data/evaluation/results_llm_path_k100.json
	# LLM as recommender from its cached outputs (LLM_RECOMMENDER.md); no request is made and the
	# system LLM is named from the saved results
	$(EVAL).llm_recommender_evaluation --no-llm
	# Embeddings and re-ranking by API from the cached responses (API_MODELS.md)
	$(EVAL).api_models_evaluation --offline
	# BM25 padding fix (BM25_PADDING.md): the systems again, every top-10 pair checked against
	# the V13 judgments file (not rewritten)
	$(EVAL).bm25_padding run --no-judge
	# Re-rendered from saved results: the reports of live services
	$(EVAL).profile_parity report
	$(EVAL).cloud_live_check report
	$(EVAL).llm_model_selection report
	@echo ""
	@echo "Done. Compare with the committed reports: git diff --stat data/evaluation/"

# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

# Removes the demo indexes, the evaluation indexes and .venv only when make
# created them. The regenerated reports are tracked files: restore them with
# git checkout -- data/evaluation/ if you do not want to keep them.
clean:
	rm -rf $(DEMO_INDEXES)
	@if [ -f $(ABLATION_MARKER) ]; then echo "rm -rf $(ABLATION_DIR)"; rm -rf $(ABLATION_DIR); \
	else echo "Keeping $(ABLATION_DIR): make did not create it"; fi
	@if [ -f $(VENV_MARKER) ]; then echo "rm -rf $(VENV)"; rm -rf $(VENV); \
	else echo "Keeping $(VENV): make did not create it"; fi
