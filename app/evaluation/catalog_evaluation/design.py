"""
The design of the evaluation on the deployed catalog (task V14), as declared
in docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md (commit 9abf48e).

Every constant the protocol fixes lives here, so the code that runs the
evaluation and the code that writes the report read the same values.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

PROTOCOL_PATH = "docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md"
PROTOCOL_COMMIT = "9abf48e"

# --- Collection --------------------------------------------------------------
SCHEMA = "catalog_deploy"
SOURCE_CATALOG = Path("data/catalog_deploy/catalog.db")  # the loader's source (same ids and texts as Neon)
CATALOG_WORKS = 50_000

# --- Where things are written --------------------------------------------------
DATA_DIR = Path("data/evaluation/catalog50k")
E2_DIR = Path("data/catalog_deploy/bge_m3")  # git-ignored: about 200 MB of vectors
REPORT_MD = Path("data/evaluation/CATALOG_EVALUATION.md")

# --- Models (all through OpenRouter) ---------------------------------------
QWEN_EMBEDDING = "qwen/qwen3-embedding-8b"
QWEN_DIMENSIONS = 1024
BGE_EMBEDDING = "baai/bge-m3"
BGE_DIMENSIONS = 1024
EMBEDDING_PIN = "deepinfra"  # the deployed encoder's pinned provider (V16, V17)
RERANKER = "qwen/qwen3-reranker-8b"
RERANKER_PIN = "primeintellect"
SYSTEM_LLM = "deepseek/deepseek-v4.1-flash"
PRIMARY_JUDGE = "z-ai/glm-5.3-flash"
SECOND_JUDGE = "minimax/minimax-m3.1-flash"
JUDGE_REASONING = "low"  # both judges have mandatory reasoning on OpenRouter (V10)
JUDGES: Tuple[str, str] = (PRIMARY_JUDGE, SECOND_JUDGE)
JUDGE_LABELS: Dict[str, str] = {PRIMARY_JUDGE: "primary", SECOND_JUDGE: "second"}

# --- Deployed search parameters (settings.py, SearchService) ------------------
DEPTH = 100  # run files keep 100 results per query (protocol section 4)
RERANK_DEPTH = 30  # DEFAULT_RERANK_DEPTH of the deployed service
RRF_K = 60
# SearchService retrieves max(max_results, rerank_depth) * 2 candidates per
# side when a reranker is configured: 60 for the deployed page of 10 results.
FUSION_CANDIDATES = 60
# Deadlines of a query embedding while a user waits (CLOUD_API_QUERY_*, V17).
DEPLOYED_TIMEOUT_S = 3.0
DEPLOYED_RETRIES = 1
DEPLOYED_DEADLINE_S = 5.0
# Deadlines of the evaluation's own calls (the long document defaults).
EVAL_TIMEOUT_S = 30.0
EVAL_RETRIES = 3

# --- Statistics ----------------------------------------------------------------
N_RESAMPLES = 10_000
SEED = 0
ALPHA = 0.05
POOL_K = 10
CUTOFF = 10
RELEVANT_GRADE = 1  # binary metrics (as in evaluation_service.py)
ANSWERED_GRADE = 2  # a query "has an answer" when some pooled work has grade >= 2 (sections 3, 9)

# --- Spending ---------------------------------------------------------------------
SPEND_CAP_USD = 5.0

# --- Strata (section 3) ---------------------------------------------------------------
STRATUM_A, STRATUM_B, STRATUM_C, STRATUM_D = "A", "B", "C", "D"
STATISTICAL_STRATA = (STRATUM_A, STRATUM_B)
B_LANGUAGES: Tuple[str, ...] = ("en", "es", "fr", "de", "it", "pt", "ca", "nl", "ru", "zh", "ja", "ar")
B_TYPES: Dict[str, int] = {"thematic": 3, "descriptive": 2, "known_item": 1}
C_NEEDS = 10
C_LANGUAGES: Tuple[str, ...] = ("es", "en", "fr", "de")
D_QUERIES = 12
A_TYPE = "original"
UNANSWERABLE_TYPE = "unanswerable"
EXPECTED_COUNTS = {STRATUM_A: 44, STRATUM_B: 72, STRATUM_C: 40, STRATUM_D: 12}
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ca": "Catalan", "nl": "Dutch", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ar": "Arabic",
}


# --- Systems (section 4) -----------------------------------------------------------
@dataclass(frozen=True)
class System:
    """One evaluated system and how it is built."""

    name: str
    description: str


S1, S2, S3, S4, S5, S6, S6B, E1, E2 = "S1", "S2", "S3", "S4", "S5", "S6", "S6b", "E1", "E2"
SYSTEMS: Tuple[System, ...] = (
    System(S1, "BM25 in Postgres (lakebase_bm25 index of catalog_deploy.work_texts)"),
    System(S2, f"Vector, `{QWEN_EMBEDDING}` (1024 d, {EMBEDDING_PIN}), approximate index of the schema"),
    System(S3, f"Hybrid: RRF (k={RRF_K}) of the top {FUSION_CANDIDATES} of S1 and of S2"),
    System(S4, f"S2 + `{RERANKER}` ({RERANKER_PIN}) over the top {RERANK_DEPTH}"),
    System(S5, f"S3 + `{RERANKER}` ({RERANKER_PIN}) over the top {RERANK_DEPTH} (the deployed configuration)"),
    System(S6, f"S3 + `{SYSTEM_LLM}` as listwise re-ranker of the top {RERANK_DEPTH} (V07, R1), run 1"),
    System(S6B, f"S3 + `{SYSTEM_LLM}` as listwise re-ranker of the top {RERANK_DEPTH} (V07, R1), run 2 (replication)"),
    System(E1, f"Exact vector search, `{QWEN_EMBEDDING}`, in Postgres with the approximate index disabled"),
    System(E2, f"Exact vector search, `{BGE_EMBEDDING}` (1024 d, {EMBEDDING_PIN}), same works and per-work text, local index"),
)
SYSTEM_NAMES: Tuple[str, ...] = tuple(system.name for system in SYSTEMS)
POOL_SYSTEMS: Tuple[str, ...] = SYSTEM_NAMES  # S1..S6 (both runs), E1, E2


# --- Contrasts (section 7) -------------------------------------------------------------
@dataclass(frozen=True)
class Contrast:
    """A declared two-sided contrast: mean(system) - mean(baseline)."""

    name: str
    system: str
    baseline: str
    question: str


CONTRASTS: Tuple[Contrast, ...] = (
    Contrast("H1", S3, S2, "Does hybrid fusion add to vector search alone?"),
    Contrast("H2", S5, S3, "Does the reranker add to the hybrid?"),
    Contrast("H3", S5, S4, "With a reranker, do the BM25 candidates add anything?"),
    Contrast("H4", S2, S1, "Does vector search beat BM25?"),
    Contrast("H5", S6, S5, "Does the LLM as re-ranker beat the cross-encoder? (run 1; run 2 is the replication)"),
    Contrast("H6", E2, E1, "Do the two embedding models differ in quality?"),
)
REPLICATION = Contrast("H5-rep", S6B, S5, "H5 on the second run of S6 (replication, outside the family)")

# --- Sections 9 and 10 ----------------------------------------------------------------------
EXPLANATION_QUERIES = 40
EXPLANATION_RESULTS = 3
OPERATIONAL_REPETITIONS = 3
ANNOTATION_PAIRS = 200
