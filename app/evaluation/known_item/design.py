"""
The design of the known-item evaluation (V29, part B), as declared in
data/evaluation/known_item/PROTOCOL.md.

Every constant the protocol fixes lives here, so the code that samples, runs
and reports reads the same values. The systems, models and search parameters
are those of the catalog evaluation (V14, catalog_evaluation.design).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from app.evaluation.catalog_evaluation import design as CD

PROTOCOL_PATH = Path("data/evaluation/known_item/PROTOCOL.md")

# --- Where things are written --------------------------------------------------
DATA_DIR = Path("data/evaluation/known_item")
SAMPLE_FILE = "sample.json"
SELECTION_FILE = "weight_selection.json"
RESULTS_FILE = "results.json"
REPORT_FILE = "KNOWN_ITEM.md"

# --- Sample ----------------------------------------------------------------------
SOURCE_CATALOG = CD.SOURCE_CATALOG  # local copy of the deployed catalog (same ids and texts as Neon, V14 parity)
CATALOG_WORKS = CD.CATALOG_WORKS
LANGUAGES: Tuple[str, ...] = CD.B_LANGUAGES  # the twelve languages of V14's stratum B
BOOKS_PER_LANGUAGE = 25
BOOKS = BOOKS_PER_LANGUAGE * len(LANGUAGES)  # 300
DEV_BOOKS = BOOKS // 2  # 150 development, 150 test
SEED = 29
DEV, TEST = "dev", "test"
# Author values of the catalog that name no person (checked on 2026-11-14 by
# listing every author name with "unknown", "anonym", "various" or "authors"
# in it): a book whose first author is one of these counts as having no author.
PLACEHOLDER_AUTHORS: Tuple[str, ...] = ("unknown", "anonymous", "various", "various authors", "group of authors")

# --- Queries -----------------------------------------------------------------------
T, TA, A = "T", "TA", "A"
QUERY_TYPES: Tuple[str, ...] = (T, TA, A)
QUERY_TYPE_NAMES: Dict[str, str] = {T: "título tal como está guardado", TA: "título y primer autor",
                                    A: "nombre del primer autor"}

# --- Metrics -------------------------------------------------------------------------
CUTOFF = CD.CUTOFF  # 10
PRIMARY_METRIC: Dict[str, str] = {T: "mrr_at_10", TA: "mrr_at_10", A: "ndcg_at_10"}
METRICS: Dict[str, Tuple[str, ...]] = {T: ("mrr_at_10", "success_at_1", "success_at_10"),
                                       TA: ("mrr_at_10", "success_at_1", "success_at_10"),
                                       A: ("ndcg_at_10", "success_at_10")}

# --- Search parameters (those of V14) ------------------------------------------------
DEPTH = CD.DEPTH  # 100 results kept per single-branch run
FUSION_CANDIDATES = CD.FUSION_CANDIDATES  # 60 per branch enter the fusion
RERANK_DEPTH = CD.RERANK_DEPTH  # 30
RRF_K = CD.RRF_K  # 60
WEIGHT_GRID: Tuple[float, ...] = (0.1, 0.25, 0.75, 0.9)  # lexical weight w; the vector weight is 1 - w
RERANK_CHUNK = RERANK_DEPTH  # documents per reranker request (the deployed request size)

# --- Systems -----------------------------------------------------------------------------
S1, S2, S3, S4, S5 = CD.S1, CD.S2, CD.S3, CD.S4, CD.S5
S3W, S5W = "S3w", "S5w"


@dataclass(frozen=True)
class System:
    """One evaluated system and how it is built."""

    name: str
    description: str


SYSTEMS: Tuple[System, ...] = tuple(System(s.name, s.description) for s in CD.SYSTEMS
                                    if s.name in (S1, S2, S3, S4, S5)) + (
    System(S3W, f"Weighted RRF (k={RRF_K}) of the top {FUSION_CANDIDATES} of S1 and of S2, lexical weight w "
                "chosen on the development half"),
    System(S5W, f"S3w + `{CD.RERANKER}` ({CD.RERANKER_PIN}) over the top {RERANK_DEPTH}"),
)
SYSTEM_NAMES: Tuple[str, ...] = tuple(s.name for s in SYSTEMS)
DEV_SYSTEMS: Tuple[str, ...] = (S1, S2, S3)  # plus one S3w per weight of the grid; no paid reranking on dev
RERANKED: Dict[str, str] = {S4: S2, S5: S3, S5W: S3W}  # reranked system -> the system whose head it reorders


def grid_name(weight: float) -> str:
    """Run name of the weighted fusion with lexical weight `weight` (S3w@0.25)."""
    return f"{S3W}@{weight:g}"


# --- Contrasts (declared before measuring) --------------------------------------------------
@dataclass(frozen=True)
class Contrast:
    """mean(system) - mean(baseline) of the query type's primary metric, on the test half."""

    name: str
    query_type: str
    system: str
    baseline: str


CONTRASTS: Tuple[Contrast, ...] = tuple(
    Contrast(f"{kind}-{qt}", qt, system, S4)
    for qt in QUERY_TYPES for kind, system in (("C1", S5), ("C2", S5W))
)
SECONDARY_DROP_LANGUAGES: Tuple[str, ...] = ("zh", "ja")
N_RESAMPLES = CD.N_RESAMPLES
STAT_SEED = CD.SEED
ALPHA = CD.ALPHA

# --- Spending (phase 2) ------------------------------------------------------------------------
SPEND_CAP_USD = 1.5
LEDGER_FILE = "spend_ledger.jsonl"
# Reranker price observed in V14's ledger: 0.5645 USD for 336 requests of 30
# documents whose texts (cut at 2,000 characters) plus the query added up to
# 8.14 million characters. Used only for the estimate made before reranking.
RERANK_USD_PER_MILLION_CHARS = 0.0693
RERANK_MAX_DOCUMENT_CHARS = 2000
