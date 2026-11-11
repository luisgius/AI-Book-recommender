# Relevance judgments (qrels): datasheet

Facts about `app/evaluation/relevance_judgments.json` as of 2026-11-11. Anything
the repository does not record is marked "Unknown: not recorded in the
repository".

## What the judgments are

- Graded relevance of a book for a test query, one entry per (query, book) pair:
  `{"query_id", "judgments": [{"source", "source_id", "relevance"}]}`. Books are
  identified by `(source, source_id)` (all `google_books`) and resolved to
  catalog UUIDs when an evaluation runs (`resolve_judgments_to_uuids`).
- 44 queries (`app/evaluation/test_queries.json`), 637 judgments, 177-book catalog
  (`data/catalog.db`). All 637 judged books resolve to a catalog book; no
  (query, book) pair appears twice.
- No per-judgment provenance: each entry has only `source`, `source_id` and
  `relevance` (no grader, model, prompt or date). The loader ignores extra
  fields, so such fields can be added later.
- The file was added in commit `a695e1f` (2025-12-22, "22-12 evaluation") and
  has not changed since.

## Grade scale

As defined in the `RelevanceJudgment` docstring (`app/evaluation/types.py`, same
commit):

| Grade | Label |
|---|---|
| 3 | Perfectly relevant |
| 2 | Relevant |
| 1 | Somewhat relevant |
| 0 | Not relevant |

Guidelines beyond these labels: Unknown: not recorded in the repository.
How the metrics use the grades (`app/evaluation/evaluation_service.py`): nDCG gain
2^grade - 1; a book is relevant for Recall, Precision and MRR when its grade
is >= 1; a book without a judgment counts as grade 0.

## Queries

- 44 short keyword queries with an id, the text and a category: ai 12,
  fiction 11, cooking 11, ambiguous 10. Added in commit `a695e1f`.
- Language (`language` field, added in task V03 by reading each query): 24 `en`,
  20 `es`. `q35` "python" and `q38` "java" are spelled the same in both
  languages and are labelled `en` by convention.
- Who wrote the queries and how they were chosen: Unknown: not recorded in the
  repository. No written information need (description or narrative) exists
  beyond the query text.

## How the pool was formed

- Code: `export_pool_candidates` in `app/evaluation/evaluation_job.py`. In
  commit `a695e1f` it took, for each query, the union of the top 20 results of
  four systems, in this order: `lexical_only` (BM25), `vector_only` (FAISS),
  `hybrid_rrf` (BM25 + FAISS, RRF) and `hybrid_rrf_mmr` (MMR, lambda 0.6, over
  the top 20 of `hybrid_rrf`), with the job's default `max_results` of 100.
  Commit `64ddc86` made the per-system depth a parameter
  (`--pool-per-mode-limit`, default 50).
- `data/evaluation/pool_candidates.json` (commit `a695e1f`): 1,518 candidates,
  23 to 44 per query (median 35). Re-running the four systems with the current
  code on the unchanged catalog and indexes (`max_results` 100) and taking the
  union of the top 20 per system reproduces the candidate set, in the same
  order, for 44 of 44 queries. The exact command that produced the file:
  Unknown: not recorded in the repository.
- `scripts/build_relevance_judgments_from_pool.py` builds the judgments file from
  the `relevance` field of a pool file, skipping candidates whose `relevance`
  is null. The 637 grades in `pool_candidates.json` are exactly the 637
  judgments (all agree).
- Only part of the pool is graded: 637 of 1,518 candidates (42.0%).
  - 13 queries (`q04` to `q16`) have every pooled candidate graded: 467 of 467,
    29 to 44 per query; grades 0/1/2/3 = 215/176/64/12.
  - The other 31 queries have 170 of 1,051 candidates graded, 1 to 20 per query
    (median 5); grades 0/1/2/3 = 26/75/56/13.
  - Why those 31 pools were only partly graded, and how the graded candidates
    were chosen: Unknown: not recorded in the repository.
- `data/evaluation/pool_candidates_v2.json` (commit `64ddc86`, 2025-12-24): union
  of the top 50 per system (3,159 candidates, 57 to 100 per query; reproduced
  for 44 of 44 queries the same way). It carries the same 637 grades, copied from
  the existing judgments by the export; the other 2,522 candidates are ungraded,
  and the judgments file was not rebuilt from it.

## Who or what assigned the grades, and when

- Grader (person, LLM or tool), number of graders and agreement between them:
  Unknown: not recorded in the repository.
- Whether the grader saw system names, ranks or scores: Unknown: not recorded in
  the repository.
- Date of grading: Unknown: not recorded in the repository. The grades were
  first committed on 2025-12-22 (commit `a695e1f`, author "Luis Gimenez Uso"),
  so they were assigned on or before that date.

## Statistics

Computed by `describe_judgments` (`app/evaluation/comparison.py`) and stored
under `qrels` in `results_comparison_baseline.json`; the pool figures above
come from a one-off script that re-ran the four systems.

| Statistic | Value |
|---|---|
| Queries / books / judgments | 44 / 177 / 637 |
| Judged share of the 44 x 177 matrix (7,788 pairs) | 8.2% |
| Judgments per query, min / median / max | 1 / 6 / 44 |
| Queries with no judgment | 0 |
| Grade 0 / 1 / 2 / 3 | 241 (37.8%) / 251 (39.4%) / 120 (18.8%) / 25 (3.9%) |
| Relevant judgments (grade >= 1) | 396 (62.2%) |
| Queries with no relevant book | 4: `q22`, `q32`, `q38`, `q39` (all `en`) |

| Language | Queries | Judgments | Relevant judgments | Queries with no relevant book |
|---|---|---|---|---|
| en | 24 | 339 | 231 | 4 |
| es | 20 | 298 | 165 | 0 |

Judged@10 of the four non-LLM systems at depth 10 (`COMPARISON_BASELINE.md`):
0.507 (BM25), 0.532 (FAISS), 0.550 (hybrid RRF), 0.550 (hybrid RRF + MMR).
About half of what these systems put in the top 10 has no judgment and is
scored as not relevant.
