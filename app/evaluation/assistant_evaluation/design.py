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
