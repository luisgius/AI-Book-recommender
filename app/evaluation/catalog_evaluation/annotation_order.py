"""
Human annotation of the order of the systems (task V31).

The 200-pair sheet of V14 (annotation.py) measured how one person agrees
with the two LLM judges on pairs sampled by the judge's grade; by design it
cannot rank the systems. This sheet can: for Spanish and English queries of
strata A and B it holds every work that S2, S3, S4 or S5 puts in its top 5
(each pair once), so nDCG@5 of the four systems can be computed with a
person's grades and compared with the order the judges give.

The sheet has two blocks. Block 1 (required) holds 30 sampled queries;
block 2 (optional) the other eligible queries, after block 1. The analysis
uses both blocks only if every query of block 2 is fully graded.

Protocol: data/evaluation/catalog50k/annotation_order/PROTOCOL.md.

- select_queries / remaining_queries: the two blocks of queries.
- top_lists / build_order_sheet: the top 5 of every system from the stored
  run files (no new retrieval), the blind sheet grouped by query and
  shuffled, the key (systems and ranks, kept apart) and the instructions.
- design_power: the smallest difference a set of queries can detect, from
  the per-query standard deviation of the judges' differences.
- analysis_queries / order_agreement: the declared analysis of a filled sheet.

Every function here is pure or reads local files: no network, no database.
"""

import csv
import json
import math
import random
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats

from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.annotation import VALID_GRADES, book_cells, read_cells, read_sheet
from app.evaluation.catalog_evaluation.queries import EvalQuery
from app.evaluation.evaluation_service import ndcg_at_k
from app.evaluation.relevance_judge import agreement_report
from app.evaluation.statistics import bootstrap_ci, paired_bootstrap_ci, paired_permutation_test

# --- Design (fixed by the protocol) ---------------------------------------------
ORDER_SYSTEMS: Tuple[str, ...] = (D.S2, D.S3, D.S4, D.S5)
ORDER_CONTRASTS: Tuple[D.Contrast, ...] = tuple(c for c in D.CONTRASTS if c.name in ("H1", "H3"))
ORDER_DEPTH = 5
ORDER_QUERIES = 30  # block 1
ORDER_LANGUAGES: Tuple[str, ...] = ("en", "es")  # the languages the annotator reads without translation
STRATUM_PREFERENCE: Tuple[str, ...] = (D.STRATUM_B, D.STRATUM_A)  # A queries were also used to choose models
BLOCK_1, BLOCK_2 = 1, 2
SECONDS_PER_PAIR: Tuple[int, int] = (30, 45)  # the earlier sheet recorded no times
POWER = 0.8
KAPPA_RESAMPLES = 2000  # as in the earlier annotation analysis
ZERO = 1e-12

# --- Files ------------------------------------------------------------------------
ORDER_DIR = "annotation_order"
SHEET_FILE = "annotation_order_sheet.csv"
KEY_FILE = "annotation_order_key.json"
SAMPLE_FILE = "sample.json"
RESULT_FILE = "order_agreement.json"
INSTRUCTIONS_FILE = "INSTRUCCIONES.md"
EARLIER_FILLED_SHEET = Path("annotation") / "annotation_sheet_filled.csv"
SHEET_COLUMNS = ("pair_id", "block", "query", "need", "title", "authors", "book_language", "year", "description",
                 "grade", "grade_2")
FIRST, SECOND = "annotator_1", "annotator_2"

Lists = Mapping[str, Mapping[str, Sequence[str]]]  # query_id -> system -> top work ids
ByQuery = Dict[str, Dict[str, int]]  # query_id -> work_id -> grade
Key = Mapping[str, Mapping[str, Any]]  # pair_id -> query, need, language, stratum, block, work, ranks


# =============================================================================
# Query sample
# =============================================================================


def balanced_quota(available: Mapping[str, int], n: int) -> Dict[str, int]:
    """
    Split n between languages as evenly as their counts allow: one query at a
    time to each language in turn, skipping a language once it has none left.
    """
    quota = {language: 0 for language in available}
    left = min(n, sum(available.values()))
    while left:
        for language in available:
            if left and quota[language] < available[language]:
                quota[language] += 1
                left -= 1
    return quota


def eligible_queries(queries: Sequence[EvalQuery], languages: Sequence[str] = ORDER_LANGUAGES) -> List[EvalQuery]:
    """Strata A and B, query language in `languages`."""
    return [q for q in queries if q.stratum in D.STATISTICAL_STRATA and q.language in languages]


def select_queries(queries: Sequence[EvalQuery], n: int = ORDER_QUERIES, languages: Sequence[str] = ORDER_LANGUAGES,
                   seed: int = D.SEED) -> Tuple[List[EvalQuery], Dict[str, Any]]:
    """
    The queries of block 1 and a record of the rule and its counts.

    If at most n are eligible, all are taken. Otherwise the n are split
    between languages by balanced_quota; inside each language stratum B is
    taken first (all of it, or a seeded sample if it exceeds the quota) and
    stratum A fills the rest with a seeded sample. Ids are sorted before
    sampling, so the result only depends on the query set and the seed.
    """
    eligible = eligible_queries(queries, languages)
    available = {lang: {s: sorted(q.query_id for q in eligible if q.language == lang and q.stratum == s)
                        for s in STRATUM_PREFERENCE} for lang in languages}
    quota = balanced_quota({lang: sum(len(ids) for ids in available[lang].values()) for lang in languages}, n)
    rng = random.Random(seed)
    chosen: Dict[str, Dict[str, List[str]]] = {}
    for lang in languages:
        left = quota[lang]
        chosen[lang] = {}
        for stratum in STRATUM_PREFERENCE:
            ids = available[lang][stratum]
            take = list(ids) if len(ids) <= left else sorted(rng.sample(ids, left))
            chosen[lang][stratum] = take
            left -= len(take)
    by_id = {q.query_id: q for q in eligible}
    selected = [by_id[i] for lang in languages for s in STRATUM_PREFERENCE for i in chosen[lang][s]]
    record = {
        "rule": (f"strata A and B, languages {list(languages)}; {n} queries split between the languages as evenly "
                 f"as the counts allow; inside each language every stratum B query first, then a seeded sample "
                 f"of stratum A (random.Random({seed}).sample over sorted ids); all if at most {n} are eligible"),
        "seed": seed, "target": n, "queries": len(selected), "fewer_than_target": len(selected) < n,
        "available": {lang: {s: len(ids) for s, ids in by.items()} for lang, by in available.items()},
        "quota": quota,
        "selected": {lang: {s: len(ids) for s, ids in by.items()} for lang, by in chosen.items()},
        "query_ids": chosen,
    }
    return selected, record


def remaining_queries(queries: Sequence[EvalQuery], block_1: Sequence[EvalQuery],
                      languages: Sequence[str] = ORDER_LANGUAGES) -> Tuple[List[EvalQuery], Dict[str, Any]]:
    """The queries of block 2: every eligible query not in block 1 (sorted by id), and their counts."""
    taken = {q.query_id for q in block_1}
    rest = sorted((q for q in eligible_queries(queries, languages) if q.query_id not in taken), key=lambda q: q.query_id)
    ids = {lang: {s: [q.query_id for q in rest if q.language == lang and q.stratum == s] for s in STRATUM_PREFERENCE}
           for lang in languages}
    return rest, {"rule": "every eligible query not in block 1", "queries": len(rest),
                  "selected": {lang: {s: len(v) for s, v in by.items()} for lang, by in ids.items()}, "query_ids": ids}


# =============================================================================
# Pairs and the blind sheet
# =============================================================================


def top_lists(queries: Sequence[EvalQuery], runs: Mapping[str, Mapping[str, Mapping[str, Any]]],
              systems: Sequence[str] = ORDER_SYSTEMS, depth: int = ORDER_DEPTH) -> Dict[str, Dict[str, List[str]]]:
    """{query_id: {system: its top `depth` work ids}} from the stored run files."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for query in queries:
        out[query.query_id] = {}
        for system in systems:
            record = runs.get(system, {}).get(query.query_id)
            if record is None:
                raise ValueError(f"No run line of {system} for {query.query_id}")
            out[query.query_id][system] = list(record["ids"][:depth])
    return out


def _ranks(by_system: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, int]]:
    """{work_id: {system: rank}} of the union of the top lists of one query."""
    out: Dict[str, Dict[str, int]] = {}
    for system, ids in by_system.items():
        for rank, work_id in enumerate(ids, start=1):
            out.setdefault(work_id, {})[system] = rank
    return out


def _hours(minutes: float) -> str:
    return f"{minutes / 60:.1f}".replace(".", ",")


def time_estimate(pairs: int, seconds: Tuple[int, int] = SECONDS_PER_PAIR) -> Dict[str, Any]:
    """Annotator time at `seconds` per pair (the earlier sheet recorded no times)."""
    return {"pairs": pairs, "seconds_per_pair": list(seconds),
            "minutes": [round(pairs * s / 60) for s in seconds],
            "basis": "30 to 45 s per pair; the pace of the earlier sheet cannot be inferred (no times were recorded)"}


INSTRUCTIONS = """# Hoja de anotación del orden de los sistemas (V31)

Esta hoja contiene {pairs} pares (consulta, obra) de {queries} consultas del
catálogo desplegado: todas las obras que alguno de los sistemas evaluados
pone entre sus cinco primeros resultados para esas consultas. La hoja no
muestra ninguna nota de los jueces, ni qué sistema devolvió la obra, ni en
qué posición. Las filas están agrupadas por consulta: el orden de las
consultas es aleatorio y, dentro de cada consulta, el de las obras también.

## Dos bloques

La columna `block` dice a qué bloque pertenece cada fila.

- **Bloque 1 (obligatorio)**: {pairs_1} filas de {queries_1} consultas, las
  primeras de la hoja. Termínelo antes de empezar el bloque 2.
- **Bloque 2 (opcional)**: {pairs_2} filas de {queries_2} consultas, después
  del bloque 1. Hágalo entero o no lo haga. En el bloque 2 una consulta solo
  cuenta para comparar los sistemas si todas sus filas tienen nota: si empieza
  una consulta, termínela. Una celda vacía en el bloque 2, aunque sea de una
  obra que no se puede valorar, deja fuera esa consulta. Si deja el bloque a
  medias, las consultas completas se informan aparte y las incompletas no se
  usan.

Decida si hace el bloque 2 sin mirar ningún resultado: el análisis se ejecuta
una sola vez, cuando entregue la hoja, nunca entre un bloque y otro.

## Qué hay que hacer

Abra `{sheet}` en una hoja de cálculo y escriba en la columna `grade` una
nota de 0 a 3 para cada fila. No cambie ninguna otra columna ni el
`pair_id`. Valore la obra frente a la columna `need` (la necesidad de
información, en inglés); la columna `query` muestra lo que escribió el
usuario. Base la nota solo en la información de la fila (título, autores,
idioma de la obra, año y descripción).

- Valore sin consultar nada sobre los sistemas: no abra la clave
  (`{key}`), ni las ejecuciones, ni los informes de la evaluación, ni busque
  las consultas en la aplicación mientras anota.
- Algunas filas repiten pares de la hoja de 200 pares que ya valoró. Valórelas
  de nuevo, sin mirar la nota que les puso entonces.

## Escala (la misma que la de los jueces, V05)

| Nota | Significado |
|---|---|
| 3 | Perfectamente relevante: es exactamente lo que se pide |
| 2 | Relevante: responde a la necesidad, aunque no del todo |
| 1 | Algo relevante: toca el tema de forma parcial o tangencial |
| 0 | No relevante |

- Una obra en otro idioma que la consulta puede ser relevante, salvo que la
  consulta pida un idioma.
- Si falta la descripción, valore por el título, los autores y lo que se sepa
  de la obra por esos datos.
- Deje la celda vacía solo si la obra no se puede valorar con el texto que se
  muestra: esa fila no cuenta. Si duda entre dos notas, elija una.

## Cuánto tarda

A un ritmo de 30 a 45 segundos por fila: el bloque 1 ({pairs_1} filas), entre
{low_1} y {high_1} horas; el bloque 2 ({pairs_2} filas), entre {low_2} y
{high_2} horas más. Puede hacerse en varias sesiones; guarde la hoja a menudo.

## Segunda persona (opcional)

La columna `grade_2` es para una segunda persona, con estas mismas
instrucciones. Puede rellenarla para todas las filas o solo para una parte;
para comparar los sistemas solo cuentan sus consultas completas (todas las
filas de la consulta con nota), en los dos bloques. Para que no vea las notas
de la primera persona, conviene que anote sobre una copia de la hoja en
blanco y se pase el fichero con `--second-sheet` (abajo).

## Al terminar

Guarde la hoja como CSV UTF-8 con otro nombre (por ejemplo
`annotation_order_sheet_filled.csv`) en esta carpeta y ejecute desde la raíz
del repositorio:

    .venv/bin/python -m app.evaluation.catalog_evaluation order-agreement --sheet {folder}/annotation_order_sheet_filled.csv

Si la segunda persona anotó en una copia aparte, añada
`--second-sheet <fichero>`. El comando no usa la red y escribe
`{folder}/{result}`.
"""


def _block_rows(block: int, queries: Sequence[EvalQuery], lists: Lists, books_by_id: Mapping[str, Any],
                earlier: Mapping[Tuple[str, str], str], seed: int, first_number: int
                ) -> Tuple[List[str], List[Dict[str, str]], Dict[str, Dict[str, Any]]]:
    """
    The rows and key entries of one block. The queries are shuffled with a
    new random.Random(seed) and, inside each query, so are its works (same
    generator, ids sorted first); pair ids are numbered from first_number.
    """
    rng = random.Random(seed)
    by_id = {q.query_id: q for q in queries}
    order = sorted(by_id)
    rng.shuffle(order)
    rows: List[Dict[str, str]] = []
    key: Dict[str, Dict[str, Any]] = {}
    for query_id in order:
        query = by_id[query_id]
        ranks = _ranks(lists[query_id])
        works = sorted(ranks)
        rng.shuffle(works)
        for work_id in works:
            pair_id = f"r{first_number + len(key):03d}"
            book = books_by_id[work_id]
            year = book.get_published_year()
            rows.append({"pair_id": pair_id, "block": str(block), "query": query.text, "need": query.need,
                         **book_cells(book), "book_language": book.language or "",
                         "year": "" if year is None else str(year), "grade": "", "grade_2": ""})
            key[pair_id] = {"query_id": query_id, "need_id": query.need_id, "language": query.language,
                            "stratum": query.stratum, "block": block, "work_id": work_id, "ranks": ranks[work_id],
                            "earlier_pair_id": earlier.get((query_id, work_id))}
    return order, rows, key


def _block_summary(order: Sequence[str], key: Key) -> Dict[str, Any]:
    per_query = {q: sum(1 for v in key.values() if v["query_id"] == q) for q in order}
    overlap = sorted(v["earlier_pair_id"] for v in key.values() if v["earlier_pair_id"])
    return {"pairs": len(key), "queries": len(order), "query_order": list(order), "pairs_per_query": per_query,
            "pairs_per_query_summary": ({"min": min(per_query.values()), "median": median(per_query.values()),
                                         "max": max(per_query.values())} if per_query else {}),
            "overlap_with_earlier_sheet": {"pairs": len(overlap), "earlier_pair_ids": overlap},
            "time_estimate": time_estimate(len(key))}


def build_order_sheet(blocks: Sequence[Sequence[EvalQuery]], lists: Lists, books_by_id: Mapping[str, Any],
                      out_dir: Path, earlier_key: Optional[Key] = None, seed: int = D.SEED) -> Dict[str, Any]:
    """
    Write the blind sheet, the key and the instructions; returns a summary.

    `blocks` holds the queries of block 1 and, optionally, of block 2. One
    row per (query, work) of the union of the top lists, block 1 first. Each
    block is shuffled on its own with random.Random(seed) (_block_rows), so
    adding block 2 leaves block 1 exactly as it was. The key maps every pair
    id to its query, need, block, work, the rank of the work in each system
    that returned it, and the pair id of the same (query, work) in the
    earlier 200-pair sheet, if any. The sheet is written with a byte-order
    mark so that spreadsheets read it as UTF-8.
    """
    earlier = {(v["query_id"], v["work_id"]): pair_id for pair_id, v in sorted((earlier_key or {}).items())}
    rows: List[Dict[str, str]] = []
    key: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Any] = {}
    for block, queries in enumerate(blocks, start=1):
        order, block_rows, block_key = _block_rows(block, queries, lists, books_by_id, earlier, seed, len(key) + 1)
        rows += block_rows
        key.update(block_key)
        summaries[f"block_{block}"] = _block_summary(order, block_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / SHEET_FILE).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / KEY_FILE).write_text(json.dumps(key, indent=2), encoding="utf-8")
    one = summaries["block_1"]
    two = summaries.get("block_2") or _block_summary([], {})
    (out_dir / INSTRUCTIONS_FILE).write_text(INSTRUCTIONS.format(
        pairs=len(rows), queries=one["queries"] + two["queries"], sheet=SHEET_FILE, key=KEY_FILE,
        folder=out_dir.as_posix(), result=RESULT_FILE,
        pairs_1=one["pairs"], queries_1=one["queries"], pairs_2=two["pairs"], queries_2=two["queries"],
        low_1=_hours(one["time_estimate"]["minutes"][0]), high_1=_hours(one["time_estimate"]["minutes"][1]),
        low_2=_hours(two["time_estimate"]["minutes"][0]), high_2=_hours(two["time_estimate"]["minutes"][1])),
        encoding="utf-8")
    return {"pairs": len(rows), "queries": sum(s["queries"] for s in summaries.values()), "blocks": summaries}


# =============================================================================
# Metrics on the sheet's pairs
# =============================================================================


def lists_from_key(key: Key, systems: Sequence[str] = ORDER_SYSTEMS) -> Dict[str, Dict[str, List[str]]]:
    """{query_id: {system: top work ids}} rebuilt from the ranks the key stores."""
    ranked: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
    for pair in key.values():
        by_system = ranked.setdefault(pair["query_id"], {s: [] for s in systems})
        for system, rank in pair["ranks"].items():
            by_system[system].append((rank, pair["work_id"]))
    return {q: {s: [w for _, w in sorted(items)] for s, items in by.items()} for q, by in ranked.items()}


def check_key_against_runs(key: Key, runs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[str]:
    """Query ids whose top lists in the key differ from the stored run files (empty: they match)."""
    lists = lists_from_key(key)
    return sorted(q for q, by in lists.items()
                  if any(by[s] != list(runs[s][q]["ids"][: ORDER_DEPTH]) for s in ORDER_SYSTEMS))


def by_query(key: Key, grades_by_pair: Mapping[str, int]) -> ByQuery:
    """{query_id: {work_id: grade}} of the pairs that have a grade."""
    out: ByQuery = {}
    for pair_id, grade in grades_by_pair.items():
        pair = key[pair_id]
        out.setdefault(pair["query_id"], {})[pair["work_id"]] = grade
    return out


def judge_by_pair(key: Key, grades_by_need: Mapping[str, Mapping[str, int]]) -> Dict[str, int]:
    """{pair_id: grade} of one judge on the sheet's pairs (pairs it never graded are left out)."""
    out = {}
    for pair_id, pair in key.items():
        grade = grades_by_need.get(pair["need_id"], {}).get(pair["work_id"])
        if grade is not None:
            out[pair_id] = grade
    return out


def ndcg_table(lists: Lists, grades: ByQuery, query_ids: Sequence[str],
               systems: Sequence[str] = ORDER_SYSTEMS) -> Dict[str, Dict[str, float]]:
    """
    {system: {query_id: nDCG@5}} under one grader. A work without a grade
    counts as unjudged: gain 0 and outside the ideal ranking (ndcg_at_k of
    evaluation_service.py), whose candidates are the graded works of the
    query on the sheet.
    """
    return {s: {q: ndcg_at_k(lists[q][s], grades.get(q, {}), ORDER_DEPTH) for q in query_ids} for s in systems}


def _sign(value: float) -> int:
    return 0 if abs(value) <= ZERO else (1 if value > 0 else -1)


def contrast_rows(table: Mapping[str, Mapping[str, float]], query_ids: Sequence[str],
                  contrasts: Sequence[D.Contrast] = ORDER_CONTRASTS) -> Dict[str, Dict[str, Any]]:
    """Mean paired difference, 95 % bootstrap interval and randomisation p of each contrast (descriptive)."""
    out = {}
    for c in contrasts:
        a = [table[c.system][q] for q in query_ids]
        b = [table[c.baseline][q] for q in query_ids]
        if not a:
            out[c.name] = {"system": c.system, "baseline": c.baseline, "queries": 0}
            continue
        diff, p = paired_permutation_test(a, b, n_resamples=D.N_RESAMPLES, seed=D.SEED)
        _, low, high = paired_bootstrap_ci(a, b, n_resamples=D.N_RESAMPLES, seed=D.SEED)
        out[c.name] = {"system": c.system, "baseline": c.baseline, "queries": len(a), "difference": diff,
                       "ci": [low, high], "p": p, "sign": _sign(diff)}
    return out


def grader_figures(lists: Lists, grades: ByQuery, query_ids: Sequence[str]) -> Dict[str, Any]:
    """nDCG@5 per system (mean, 95 % bootstrap interval, per query), the order of the systems and H1, H3."""
    table = ndcg_table(lists, grades, query_ids)
    systems = {}
    for s in ORDER_SYSTEMS:
        values = [table[s][q] for q in query_ids]
        mean, low, high = bootstrap_ci(values, n_resamples=D.N_RESAMPLES, seed=D.SEED) if values else (None, None, None)
        systems[s] = {"mean": mean, "ci": [low, high]}
    ordering = sorted(ORDER_SYSTEMS, key=lambda s: -(systems[s]["mean"] or 0.0))
    return {"queries": len(query_ids), "ndcg_at_5": systems, "ordering": ordering,
            "contrasts": contrast_rows(table, query_ids), "per_query": table}


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Kendall's tau-b of two score vectors over the same items (None if either is constant)."""
    concordant = discordant = tied_x = tied_y = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = _sign(x[i] - x[j]), _sign(y[i] - y[j])
        tied_x += dx == 0
        tied_y += dy == 0
        concordant += dx * dy > 0
        discordant += dx * dy < 0
    pairs = len(x) * (len(x) - 1) // 2
    denominator = math.sqrt((pairs - tied_x) * (pairs - tied_y))
    return (concordant - discordant) / denominator if denominator else None


def compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether two graders give each contrast the same sign, and Kendall's tau-b of their system means."""
    signs = {}
    for c in ORDER_CONTRASTS:
        a, b = first["contrasts"][c.name].get("sign"), second["contrasts"][c.name].get("sign")
        signs[c.name] = {"first": a, "second": b, "agree": a is not None and a == b}
    means = lambda f: [f["ndcg_at_5"][s]["mean"] or 0.0 for s in ORDER_SYSTEMS]  # noqa: E731
    return {"signs": signs, "kendall_tau_b": kendall_tau_b(means(first), means(second))}


def figures_on(mask: Set[str], key: Key, lists: Lists, graders: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    """
    Every grader's figures on the same pairs (`mask`) and the same queries:
    the queries with at least one pair in the mask. A grader's grade outside
    the mask is ignored, so all graders are scored on exactly the same pairs.
    """
    query_ids = sorted({key[p]["query_id"] for p in mask})
    return {name: grader_figures(lists, by_query(key, {p: g for p, g in grades.items() if p in mask}), query_ids)
            for name, grades in graders.items()}


# =============================================================================
# Power of the design (computed before annotating, from the judges)
# =============================================================================


def minimum_detectable_difference(differences: Sequence[float], alpha: float = D.ALPHA,
                                  power: float = POWER) -> Dict[str, Any]:
    """
    Smallest mean paired difference a two-sided paired t-test at level alpha
    detects with the given power: (t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / sqrt(n),
    with sd the standard deviation of the per-query differences.
    """
    n = len(differences)
    sd = float(np.std(differences, ddof=1))
    factor = float(stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(power, n - 1))
    return {"queries": n, "mean_difference": float(np.mean(differences)), "sd": sd, "alpha": alpha, "power": power,
            "minimum_detectable_difference": factor * sd / math.sqrt(n)}


def design_power(key: Key, judges: Mapping[str, Mapping[str, Mapping[str, int]]],
                 query_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    For each judge and contrast: the per-query differences of nDCG@5 on the
    sheet's pairs of `query_ids` (all the key's queries by default; every pair
    graded by the judge) and the smallest difference they let the design detect.
    """
    lists = lists_from_key(key)
    query_ids = sorted(query_ids if query_ids is not None else lists)
    out: Dict[str, Any] = {"metric": f"nDCG@{ORDER_DEPTH} on the sheet's pairs", "queries": len(query_ids),
                           "by_judge": {}}
    for model, grades in judges.items():
        table = ndcg_table(lists, by_query(key, judge_by_pair(key, grades)), query_ids)
        out["by_judge"][model] = {
            "ndcg_at_5_mean": {s: float(np.mean([table[s][q] for q in query_ids])) for s in ORDER_SYSTEMS},
            "contrasts": {c.name: minimum_detectable_difference([table[c.system][q] - table[c.baseline][q]
                                                                 for q in query_ids]) for c in ORDER_CONTRASTS},
        }
    return out


# =============================================================================
# The declared analysis of a filled sheet
# =============================================================================


def cell_counts(key: Key, cells: Mapping[str, str]) -> Dict[str, Any]:
    """Graded, empty (or missing) and invalid cells of one grade column; invalid cells count as empty."""
    graded = [p for p in key if cells.get(p, "") in VALID_GRADES]
    invalid = {p: cells[p] for p in key if cells.get(p, "") not in VALID_GRADES and cells.get(p, "") != ""}
    empty = [p for p in key if cells.get(p, "") == ""]
    empty_by_query: Dict[str, int] = {}
    for p in empty + list(invalid):
        empty_by_query[key[p]["query_id"]] = empty_by_query.get(key[p]["query_id"], 0) + 1
    return {"rows": len(key), "graded": len(graded),
            "graded_by_block": {str(b): sum(1 for p in graded if key[p].get("block", BLOCK_1) == b)
                                for b in sorted({v.get("block", BLOCK_1) for v in key.values()})},
            "empty": len(empty), "invalid": invalid, "empty_or_invalid_by_query": dict(sorted(empty_by_query.items())),
            "unknown_pair_ids": sorted(p for p in cells if p not in key)}


def _grades(key: Key, path: Path, column: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
    cells = read_cells(path, column)
    return {p: g for p, g in read_sheet(path, column).items() if p in key}, cell_counts(key, cells)


def analysis_queries(key: Key, grades: Mapping[str, int],
                     complete_only: Sequence[int] = (BLOCK_2,)) -> Dict[str, Any]:
    """
    Which queries the system figures use (protocol section 6).

    A query is complete when every one of its pairs has a grade, partial when
    some but not all have one, and not started when none has. In the blocks
    listed in `complete_only` only complete queries are usable (there an
    empty cell may mean the annotator stopped); in the other blocks a query is
    usable if any of its pairs has a grade (an empty cell is a work that could
    not be judged). The primary set is blocks 1 and 2 when every query of
    block 2 is complete, else block 1 only; then the complete queries of
    block 2, if any, are reported apart as descriptive. A partial query of a
    `complete_only` block is never used for system figures.
    """
    pairs: Dict[str, List[str]] = {}
    for pair_id, pair in key.items():
        pairs.setdefault(pair["query_id"], []).append(pair_id)
    status: Dict[str, Dict[str, Any]] = {}
    for query_id, ids in sorted(pairs.items()):
        graded = sum(1 for p in ids if p in grades)
        state = "complete" if graded == len(ids) else ("partial" if graded else "not started")
        status[query_id] = {"block": key[ids[0]].get("block", BLOCK_1), "pairs": len(ids), "graded": graded,
                            "state": state}
    usable = {b: [q for q, s in status.items() if s["block"] == b
                  and (s["state"] == "complete" if b in complete_only else s["graded"] > 0)]
              for b in (BLOCK_1, BLOCK_2)}
    block_2 = [q for q, s in status.items() if s["block"] == BLOCK_2]
    both = bool(block_2) and all(status[q]["state"] == "complete" for q in block_2)
    return {
        "status": status, "usable": {str(b): v for b, v in usable.items()},
        "partial_not_used": sorted(q for q, s in status.items() if s["block"] in complete_only and s["state"] == "partial"),
        "block_2_complete": both,
        "primary_set": "blocks 1 and 2" if both else "block 1",
        "primary": usable[BLOCK_1] + usable[BLOCK_2] if both else usable[BLOCK_1],
        "secondary_block_1": usable[BLOCK_1] if both else None,
        "block_2_descriptive": None if both or not usable[BLOCK_2] else usable[BLOCK_2],
    }


def pair_agreement(key: Key, first: Mapping[str, int], second: Mapping[str, int]) -> Dict[str, Any]:
    """agreement_report on the pairs both graded ("original" = first, "llm" = second), by query language."""
    ids = sorted(p for p in first if p in second)
    if not ids:
        return {"pairs": 0}
    report = agreement_report([first[p] for p in ids], [second[p] for p in ids], [key[p]["language"] for p in ids],
                              n_resamples=KAPPA_RESAMPLES, seed=D.SEED)
    return {"pairs": len(ids), **report}


def reading_rule(person: Mapping[str, Any], judges: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """
    The rule written before annotating: a contrast is confirmed by a person
    when the sign of its difference with the person's grades is the sign it
    has with each judge (a zero difference has no sign and does not confirm).
    """
    out: Dict[str, Any] = {}
    for c in ORDER_CONTRASTS:
        sign = person["contrasts"][c.name].get("sign")
        confirmed = bool(sign) and all(j["contrasts"][c.name].get("sign") == sign for j in judges.values())
        out[c.name] = "confirmed by a person" if confirmed else "not confirmed by a person"
    out["order_holds_on_this_sample"] = all(out[c.name] == "confirmed by a person" for c in ORDER_CONTRASTS)
    return out


def _system_block(key: Key, lists: Lists, person: Mapping[str, int], name: str,
                  judges_by_pair: Mapping[str, Mapping[str, int]], query_ids: Sequence[str]) -> Dict[str, Any]:
    """(a)-(c) on `query_ids`: the person's and each judge's figures on the person's graded pairs of those queries."""
    chosen = set(query_ids)
    mask = {p for p in person if key[p]["query_id"] in chosen}
    figures = figures_on(mask, key, lists, {name: person, **judges_by_pair})
    judge_names = list(judges_by_pair)
    return {"queries": len(chosen), "pairs": len(mask), "figures": figures,
            "versus_judges": {m: compare(figures[name], figures[m]) for m in judge_names},
            "between_judges": {f"{a} vs {b}": compare(figures[a], figures[b])
                               for a, b in combinations(judge_names, 2)},
            "reading_rule": reading_rule(figures[name], {m: figures[m] for m in judge_names})}


def _annotator_block(key: Key, lists: Lists, person: Mapping[str, int], name: str,
                     judges_by_pair: Mapping[str, Mapping[str, int]], complete_only: Sequence[int]) -> Dict[str, Any]:
    """The selection of queries, the primary figures with the reading rule, the secondary or descriptive set, (d)."""
    selection = analysis_queries(key, person, complete_only)
    out: Dict[str, Any] = {"selection": selection,
                           "primary": _system_block(key, lists, person, name, judges_by_pair, selection["primary"])}
    for label in ("secondary_block_1", "block_2_descriptive"):
        if selection[label]:
            block = _system_block(key, lists, person, name, judges_by_pair, selection[label])
            block.pop("reading_rule")  # the rule is read on the primary set only
            out[label] = block
    out["pair_agreement_with_judges"] = {m: pair_agreement(key, person, j) for m, j in judges_by_pair.items()}
    return out


def order_agreement(sheet: Path, key: Key, judges: Mapping[str, Mapping[str, Mapping[str, int]]],
                    earlier_sheet: Optional[Path] = None, second_sheet: Optional[Path] = None) -> Dict[str, Any]:
    """
    The declared analysis of a filled sheet (protocol section 6), run once.

    (a)-(c) nDCG@5 of S2-S5, H1 and H3, signs and Kendall's tau with the
    person's grades and each judge's on the same pairs, on the primary set of
    queries (analysis_queries) and, apart, on the secondary or descriptive
    set; (d) agreement per pair with each judge on every graded pair and, on
    the pairs of the earlier sheet, with the annotator's earlier grades; (e)
    the same for `grade_2` when it is filled (from `second_sheet` if given,
    else from `sheet`; only complete queries in both blocks), plus the
    agreement between the two annotators. Empty and invalid cells do not count.
    """
    lists = lists_from_key(key)
    first, first_cells = _grades(key, sheet, "grade")
    judges_by_pair = {model: judge_by_pair(key, grades) for model, grades in judges.items()}
    out: Dict[str, Any] = {
        "sheet": str(sheet), "pairs_in_key": len(key), "queries_in_key": len(lists),
        "cells": {FIRST: first_cells},
        "empty_cells_rule": ("an empty or invalid cell does not count: it is left out of every agreement figure, and "
                             "for nDCG@5 the pair is unjudged (gain 0, outside the ideal ranking) for the person "
                             "and for every judge alike; a block 1 query is used if one of its pairs is graded, "
                             "a block 2 query only if all its pairs are"),
        FIRST: _annotator_block(key, lists, first, FIRST, judges_by_pair, (BLOCK_2,)),
    }

    earlier_ids = {p: v["earlier_pair_id"] for p, v in key.items() if v.get("earlier_pair_id")}
    self_block: Dict[str, Any] = {"overlap_pairs": len(earlier_ids)}
    if earlier_sheet is not None and Path(earlier_sheet).exists():
        earlier_grades = read_sheet(Path(earlier_sheet))
        earlier = {p: earlier_grades[e] for p, e in earlier_ids.items() if e in earlier_grades}
        self_block.update({
            "earlier_sheet": str(earlier_sheet), "graded_earlier": len(earlier),
            "graded_now": sum(1 for p in earlier_ids if p in first),
            "note": "agreement_report labels: 'original' = earlier grade, 'llm' = new grade of the same annotator",
            "agreement": pair_agreement(key, earlier, first),
        })
    out["self_agreement"] = self_block

    second_path = second_sheet if second_sheet is not None else sheet
    second, second_cells = _grades(key, Path(second_path), "grade_2")
    out["cells"][SECOND] = second_cells
    if second:
        block = _annotator_block(key, lists, second, SECOND, judges_by_pair, (BLOCK_1, BLOCK_2))
        common = sorted(set(out[FIRST]["selection"]["primary"]) & set(block["selection"]["primary"]))
        both = {p: g for p, g in first.items() if p in second and key[p]["query_id"] in set(common)}
        between = figures_on(set(both), key, lists, {FIRST: first, SECOND: second}) if both else {}
        out[SECOND] = {
            "sheet": str(second_path), **block,
            "pair_agreement_between_annotators": pair_agreement(key, first, second),
            "versus_first_annotator_on_common_queries": (
                {"queries": len(common), "pairs": len(both), **compare(between[FIRST], between[SECOND])}
                if both else {"queries": 0, "pairs": 0}),
        }
    else:
        out[SECOND] = None
    return out
