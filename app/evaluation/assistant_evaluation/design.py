"""
The design of the evaluation of the assistant and the agent (task V22), as
declared in docs/notes/2026-11-12-protocolo-evaluacion-asistente.md (commit
0ccd5f2, saved before any result).

Every constant the protocol fixes, and every choice this implementation had
to make where the protocol is silent, lives here, so the code that runs the
evaluation and the code that writes the report read the same values. The
choices the protocol does not fix are listed in the report's deviations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from app.evaluation.catalog_evaluation import design as V14

PROTOCOL_PATH = "docs/notes/2026-11-12-protocolo-evaluacion-asistente.md"
PROTOCOL_COMMIT = "0ccd5f2"

# --- Where things are read and written ------------------------------------------
V14_DIR = V14.DATA_DIR  # data/evaluation/catalog50k (read only)
V14_QUERIES = V14_DIR / "queries" / "queries.json"
V14_POOL = V14_DIR / "pool.json"
TEST_QUERIES = Path("app/evaluation/test_queries.json")  # stratum A with its categories
DATA_DIR = Path("data/evaluation/assistant")
REPORT_MD = Path("data/evaluation/ASSISTANT_EVALUATION.md")
SOURCE_CATALOG = V14.SOURCE_CATALOG  # the loader's SQLite copy (parity with Neon checked in V14)
SCHEMA = V14.SCHEMA
CATALOG_WORKS = V14.CATALOG_WORKS

# --- The cloud profile as the application builds it ------------------------------
# The variables the composition root reads; everything else keeps the
# defaults of app/infrastructure/config/settings.py for SEARCH_PROFILE=cloud.
CLOUD_ENV: Dict[str, str] = {"SEARCH_PROFILE": "cloud", "DATABASE_SCHEMA": SCHEMA}

# --- Models (all through OpenRouter) ---------------------------------------------
SYSTEM_LLM = V14.SYSTEM_LLM
GENERATOR_MODEL = "google/gemini-3.8-flash"  # V14's generator (generation.GENERATOR_MODEL)
PRIMARY_JUDGE = V14.PRIMARY_JUDGE
SECOND_JUDGE = V14.SECOND_JUDGE
JUDGES: Tuple[str, str] = V14.JUDGES
# Simulated user of set E: another vendor than the system LLM (DeepSeek), and
# than every other model of the evaluation (Google, Z.ai, MiniMax, Qwen).
SIMULATED_USER_MODEL = "mistralai/mistral-small-2603"
SIMULATED_USER_TEMPERATURE = 0.0

# --- Request sets (section 3) ------------------------------------------------------
SET_A, SET_B, SET_D, SET_E, SET_F = "A", "B", "D", "E", "F"
SET_G = "G"  # held-out set of the re-measurement (V25 addendum, section 3)
AMBIGUOUS_CATEGORY = "ambiguous"  # the 10 queries of A reported apart
E_SIZE = 30
E_SEED = 0
E_MAX_WORDS = 3  # "de una a tres palabras"
E_MAX_CJK_CHARS = 8  # Chinese and Japanese have no spaces: a short phrase
E_MAX_SHARED_WORDS = 1  # content words a vague request may share with the original query
F_SIZE = 20
F_SEED = 0
F_KINDS: Dict[str, int] = {"compare": 7, "like_but": 7, "facts": 6}
F_MIN_DESCRIPTION = 300  # characters of description of a work shown to the generator
F_DESCRIPTION_SHOWN = 700
AGENT_SAMPLE = 40
AGENT_SEED = 0

# --- Runs ---------------------------------------------------------------------------
ASSISTANT_MAX_RESULTS = 10  # DEFAULT_MAX_RESULTS of POST /assistant/chat
P_DEPTH = 100  # reference P keeps 100 results (max_results=100)
CUTOFF = 10
RUN_1, RUN_2 = "run1", "run2"  # A and B twice (route stability); quality from run 1
E_MAX_ASSISTANT_TURNS = 4  # safety stop; the assistant itself searches after 2 questions
TURN_ATTEMPTS = 3  # attempts of a turn that raised (each a full request)
EMBED_ATTEMPTS = 3  # attempts of a query embedding (each with the deployed hedge and deadlines)
WORKERS = 4  # requests in flight; the database pool has 2 connections
DB_POOL = 2
# Reranker calls: the evaluation's long deadlines (as V14), not the deployed 5 s.
RERANK_TIMEOUT_S = V14.EVAL_TIMEOUT_S
RERANK_RETRIES = V14.EVAL_RETRIES

# --- Measures (section 5) -------------------------------------------------------------
RECOMMENDED_GRADE = 2  # "nota 2 o 3"
N_RESAMPLES = V14.N_RESAMPLES
SEED = V14.SEED
ALPHA = V14.ALPHA

# --- Section 7 thresholds ---------------------------------------------------------------
OOS_AB_MAX = 0.02  # "fuera de alcance" indebido en A y B
CLARIFY_B_MAX = 0.10  # preguntas innecesarias en las peticiones de B

# --- Spending (section 9) ---------------------------------------------------------------
SPEND_CAP_USD = 4.0

LANGUAGE_NAMES = V14.LANGUAGE_NAMES
B_LANGUAGES = V14.B_LANGUAGES


@dataclass(frozen=True)
class Contrast:
    """A declared two-sided contrast: mean(system) - mean(reference) over paired requests."""

    name: str
    set_name: str
    metric: str
    system: str
    reference: str
    question: str


CONTRASTS: Tuple[Contrast, ...] = (
    Contrast("H7", "A+B", "nDCG@10", "assistant (run 1)", "P on the request",
             "Does passing a clear request through the assistant change the quality?"),
    Contrast("H8", "E", "nDCG@10", "assistant after the conversation", "P on the vague request",
             "Does asking improve the results?"),
    Contrast("H9", "F", "precision of the recommendations (grade >= 2)", "assistant's recommendations",
             "P at the same number of top positions",
             "Does the agent recommend better than retrieval alone?"),
)


# --- Re-measurement after the revised routing instruction (V25) -------------------
# docs/notes/2026-11-12-protocolo-evaluacion-asistente-adenda.md, committed in
# 0a63751 before the instruction changed. The first measurement's files under
# DATA_DIR are only read; the second measurement lives under V25_DIR and
# reuses, by copy, what the instruction cannot change: the request sets A-F,
# P, the depth check and the query vectors. Judgments of pairs already judged
# are read from V14's and the first measurement's caches.
ADDENDUM_PATH = "docs/notes/2026-11-12-protocolo-evaluacion-asistente-adenda.md"
ADDENDUM_COMMIT = "0a63751"
V25_DIR = Path("data/evaluation/assistant_v25")
V25_REPORT_MD = Path("data/evaluation/ASSISTANT_EVALUATION_V25.md")
FIRST_RUNS_COMMIT = "36802ee"  # the commit every run of the first measurement was made at
INSTRUCTION_COMMIT = "5984e71"  # the commit of the revised routing instruction (before G existed)
V25_SPEND_CAP_USD = 2.5
G_VALID = 60
G_OUT_OF_SCOPE = 20
OOS_G_MAX = 0.02  # "fuera de alcance" indebido en las 60 válidas de G: como máximo una


def is_v25(data_dir: Path) -> bool:
    """True for the re-measurement's data directory."""
    return Path(data_dir).resolve() == V25_DIR.resolve()


# --- Outputs of the report, derived from the data directories ----------------------
# The report command writes results.json and per_request.json into each data
# directory and each markdown report next to it. The defaults are derived
# from the directories when the command runs, so a report built from other
# directories (a test's temporary ones) never writes over the published files.
# With the default directories they are REPORT_MD, V25_DIR and V25_REPORT_MD.


def report_md(data_dir: Path) -> Path:
    """The first measurement's markdown report: REPORT_MD's name, next to data_dir."""
    return Path(data_dir).parent / REPORT_MD.name


def second_data_dir(first_dir: Path) -> Path:
    """The re-measurement's data directory: V25_DIR's name, next to the first measurement's."""
    return Path(first_dir).parent / V25_DIR.name


def second_report_md(second_dir: Path) -> Path:
    """The re-measurement's markdown report: V25_REPORT_MD's name, next to second_dir."""
    return Path(second_dir).parent / V25_REPORT_MD.name


def spend_cap(data_dir: Path) -> float:
    """The cap of the measurement stored in data_dir (section 9 of the protocol; section 6 of the addendum)."""
    return V25_SPEND_CAP_USD if is_v25(data_dir) else SPEND_CAP_USD
