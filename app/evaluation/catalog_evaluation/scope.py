"""
Scope of the catalog evaluation's conclusions on subsets of its queries (V29, part A).

An external review (2026-11-13) objected that two features of the query set
could drive the conclusion "adding BM25 candidates lowers nDCG@10" (H1, H3):
the Chinese and Japanese queries, where the Postgres BM25 index does not
segment the text, and the 44 queries of stratum A, which were used to choose
models before the evaluation. This module recomputes, from the saved runs
and judgments only (no network, no model call, no database), the system
table and the contrasts H1-H6 on four subsets:

- (a) stratum B only;
- (b) strata A and B without the Chinese and Japanese queries;
- (c) stratum B without Chinese and Japanese;
- (d) every stratum except A (B, C and D).

The recomputation is review_sensitivity.catalog_figures, the function that
already recomputes the published figures without a set of queries; it calls
the analysis functions of the evaluation (same metrics, resamples, seed and
Holm family). For each subset, every declared decision (each contrast's
significance under each judge, the protocol's conclusion, the hybrid rule and
the embedding rule) is compared with the full set's.

The analysis is post hoc and outside the declared protocol: it is reported
whatever its direction and does not replace the published results. It writes
only scope.json and SCOPE.md under data/evaluation/catalog50k/scope/.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from app.evaluation import review_sensitivity as RS
from app.evaluation.assistant_evaluation.report import _ci, _f, _p, _table
from app.evaluation.catalog_evaluation import design as D
from app.evaluation.catalog_evaluation.queries import EvalQuery, load_queries

COMMAND = ".venv/bin/python -m app.evaluation.catalog_evaluation scope"
OUT_DIR = D.DATA_DIR / "scope"
ANALYSIS_DATE = "2026-11-14"
REVIEW_DATE = "2026-11-13"
UNSEGMENTED_LANGUAGES: Tuple[str, ...] = ("zh", "ja")  # the Postgres BM25 index does not segment them
TABLE_SYSTEMS: Tuple[str, ...] = D.SYSTEM_NAMES


@dataclass(frozen=True)
class Subset:
    """A subset of the query set: the strata it keeps and the languages it drops."""

    key: str
    label: str
    strata: Tuple[str, ...]
    drop_languages: Tuple[str, ...] = ()


SUBSETS: Tuple[Subset, ...] = (
    Subset("a", "Estrato B solo", (D.STRATUM_B,)),
    Subset("b", "Estratos A y B sin chino ni japonés", D.STATISTICAL_STRATA, UNSEGMENTED_LANGUAGES),
    Subset("c", "Estrato B sin chino ni japonés", (D.STRATUM_B,), UNSEGMENTED_LANGUAGES),
    Subset("d", "Todos los estratos salvo A (B, C y D)", (D.STRATUM_B, D.STRATUM_C, D.STRATUM_D)),
)


def excluded_ids(queries: Sequence[EvalQuery], subset: Subset) -> List[str]:
    """Ids of the queries the subset drops by language (strata are selected separately)."""
    return [q.query_id for q in queries if q.language in subset.drop_languages]


def subset_ids(queries: Sequence[EvalQuery], subset: Subset) -> List[str]:
    """Ids of the queries a subset keeps."""
    dropped = set(excluded_ids(queries, subset))
    return [q.query_id for q in queries if q.stratum in subset.strata and q.query_id not in dropped]


def _decisions(full: Mapping[str, Any], sub: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return RS.contrast_decisions(full, sub) + RS.catalog_rule_decisions(full, sub)


def scope_analysis(data_dir: Path = D.DATA_DIR) -> Dict[str, Any]:
    """
    The whole analysis: the full set (checked against the published
    results.json) and, per subset, its query counts, system table, contrasts
    and decisions compared with the full set.
    """
    queries = load_queries(data_dir / "queries" / "queries.json")
    full = RS.catalog_figures(data_dir)
    RS.check_reproduction("Catalog (V14)", RS.published_catalog(data_dir), full)
    subsets = []
    for subset in SUBSETS:
        figures = RS.catalog_figures(data_dir, exclude=excluded_ids(queries, subset), strata=subset.strata)
        kept_ids = set(subset_ids(queries, subset))
        kept = [q for q in queries if q.query_id in kept_ids]
        counts: Dict[str, int] = {}
        for q in kept:
            counts[q.stratum] = counts.get(q.stratum, 0) + 1
        decisions = _decisions(full, figures)
        subsets.append({"key": subset.key, "label": subset.label, "strata": list(subset.strata),
                        "dropped_languages": list(subset.drop_languages), "queries": len(kept),
                        "queries_by_stratum": counts, "figures": figures, "decisions": decisions,
                        "changed_decisions": [d["decision"] for d in decisions if d["changes"]]})
    return {"analysis": "V29 part A: scope of the catalog evaluation on subsets of its queries",
            "post_hoc": True, "date": ANALYSIS_DATE, "review_date": REVIEW_DATE, "command": COMMAND,
            "full": {"queries": len([q for q in queries if q.stratum in D.STATISTICAL_STRATA]),
                     "figures": full},
            "subsets": subsets}


# =============================================================================
# Markdown (Spanish, like the other reports of the evaluation)
# =============================================================================


def _ndcg_row(label: str, figures: Mapping[str, Any], judge: str) -> List[str]:
    summary = figures["metrics"][judge]["summary_ab"]
    return [label] + [_f(summary[s]["ndcg_at_10"]["mean"]) if s in summary else "n/a" for s in TABLE_SYSTEMS]


def _contrast_rows(full: Mapping[str, Any], figures: Mapping[str, Any], judge: str) -> List[List[str]]:
    rows = []
    for name, row in figures["metrics"][judge]["contrasts"].items():
        base = full["metrics"][judge]["contrasts"][name]
        rows.append([f"{name} ({row['system']} - {row['baseline']})", str(row["queries"]),
                     _f(row["difference"], signed=True), _ci(row["ci"], signed=True),
                     _p(row["p_holm"]) if row.get("p_holm") is not None else f"{_p(row['p'])} (sin Holm)",
                     "sí" if RS._significant(row) else "no",
                     "sí" if RS._significant(row) != RS._significant(base) else "no"])
    return rows


def build_markdown(document: Mapping[str, Any]) -> str:
    """SCOPE.md from the analysis document."""
    full = document["full"]["figures"]
    lines = [
        "# Alcance de las conclusiones de la evaluación sobre el catálogo (V29, parte A)",
        "",
        f"**Análisis post hoc, fuera del protocolo declarado.** Se hizo el {document['date']}, después de conocer "
        f"los resultados de V14 y a raíz de una revisión externa del {document['review_date']}. El protocolo de V14 "
        f"(`{D.PROTOCOL_PATH}`, commit {D.PROTOCOL_COMMIT}) no lo declara. Se informa sea cual sea su dirección y no "
        "sustituye ni modifica los resultados publicados en `data/evaluation/CATALOG_EVALUATION.md`.",
        "",
        "## Qué se recalcula",
        "",
        "La tabla de sistemas (nDCG@10) y los contrastes H1 a H6 de V14, con las mismas funciones, el mismo número "
        f"de remuestreos ({D.N_RESAMPLES}), la misma semilla ({D.SEED}) y la misma familia de Holm, sobre cuatro "
        "subconjuntos de las consultas ya evaluadas. No se hace ninguna llamada de red ni a modelos: todo sale de "
        "los ficheros de ejecuciones y de juicios guardados. Antes de nada se comprueba que el cálculo sin "
        "exclusiones reproduce exactamente el `results.json` publicado.",
        "",
        "- (a) Solo el estrato B: quita las 44 consultas del estrato A, que se usaron para elegir modelos.",
        "- (b) Estratos A y B sin las consultas en chino y japonés, donde el índice BM25 de PostgreSQL no segmenta "
        "el texto.",
        "- (c) Estrato B sin chino ni japonés.",
        "- (d) Todos los estratos salvo A: B, C (versiones en cuatro idiomas de diez necesidades, que comparten "
        "juicios y no son independientes) y D (consultas sin respuesta en el catálogo; ninguna obra tiene grado "
        "relevante, así que su nDCG@10 es 0 en todos los sistemas y solo diluye las diferencias).",
        "",
        "La columna «cambia» compara la significación (p corregida por Holm < 0,05) con la del conjunto completo "
        "de V14 (estratos A y B, 116 consultas). Con menos consultas los intervalos se ensanchan y la potencia "
        "baja, así que perder la significación no es lo mismo que encontrar el efecto contrario.",
        "",
        f"Comando: `{document['command']}`.",
        "",
        "## Resumen",
        "",
    ]
    summary_rows = []
    for s in document["subsets"]:
        p1 = s["figures"]["metrics"]["primary"]["contrasts"]
        summary_rows.append([f"({s['key']}) {s['label']}", str(s["queries"]),
                             f"{_f(p1['H1']['difference'], signed=True)} {_ci(p1['H1']['ci'], signed=True)}",
                             f"{_f(p1['H3']['difference'], signed=True)} {_ci(p1['H3']['ci'], signed=True)}",
                             ", ".join(s["changed_decisions"]) or "ninguna"])
    f1 = full["metrics"]["primary"]["contrasts"]
    lines += _table(["Subconjunto", "Consultas", "H1 (juez principal)", "H3 (juez principal)",
                     "Decisiones que cambian respecto al conjunto completo"],
                    [["V14 completo (A y B)", str(document["full"]["queries"]),
                      f"{_f(f1['H1']['difference'], signed=True)} {_ci(f1['H1']['ci'], signed=True)}",
                      f"{_f(f1['H3']['difference'], signed=True)} {_ci(f1['H3']['ci'], signed=True)}", "-"]]
                    + summary_rows)
    lines += ["## nDCG@10 por sistema", ""]
    for judge, title in (("primary", f"juez principal `{D.PRIMARY_JUDGE}`"),
                         ("second", f"segundo juez `{D.SECOND_JUDGE}`")):
        lines += [f"Con el {title}:", ""]
        rows = [_ndcg_row("V14 completo (A y B)", full, judge)]
        rows += [_ndcg_row(f"({s['key']}) {s['label']}", s["figures"], judge) for s in document["subsets"]]
        lines += _table(["Subconjunto"] + list(TABLE_SYSTEMS), rows)
    for s in document["subsets"]:
        by_stratum = ", ".join(f"{k}: {v}" for k, v in sorted(s["queries_by_stratum"].items()))
        lines += [f"## ({s['key']}) {s['label']}", "", f"Consultas: {s['queries']} ({by_stratum}).", ""]
        for judge, title in (("primary", "Juez principal"), ("second", "Segundo juez")):
            lines += [f"{title}:", ""]
            lines += _table(["Contraste", "n", "Diferencia", "IC 95 %", "p Holm", "Significativo", "Cambia"],
                            _contrast_rows(full, s["figures"], judge))
        lines += ["Decisiones declaradas:", ""]
        lines += _table(["Decisión", "Conjunto completo", "Subconjunto", "Cambia"],
                        [[d["decision"], f"{d['published']} {d['published_detail']}".strip(),
                          f"{d['without']} {d['without_detail']}".strip(), "sí" if d["changes"] else "no"]
                         for d in s["decisions"]])
    return "\n".join(lines).rstrip() + "\n"


def write_scope(data_dir: Path = D.DATA_DIR, out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Run the analysis and write scope.json and SCOPE.md."""
    document = scope_analysis(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scope.json").write_text(json.dumps(RS._as_json(document), ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (out_dir / "SCOPE.md").write_text(build_markdown(document), encoding="utf-8")
    return document
