"""
Evaluation on the deployed catalog (task V14).

Implements the protocol committed before any result,
docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md (commit 9abf48e):
four query strata, systems S1-S6 (S6 twice), E1 and E2 over the 50,000 works
of the Neon schema catalog_deploy, pooled judgments by two LLM judges,
nDCG@10 and secondary metrics, contrasts H1-H6 with Holm correction, the
embedding decision rule, the unanswerable-query analysis, the explanation
check and the human annotation sheet.

Stages (python -m app.evaluation.catalog_evaluation <stage>), each resumable
and each leaving its output on disk under data/evaluation/catalog50k/:

    queries      generate and check strata B, C and D
    parity       check that the local catalog copy equals the Neon schema
    embed        frozen query embeddings (both models), E2 document vectors
    runs         top-100 run files of every system
    judge        pool and judge with both judges
    operational  embedding operational measurements (section 8)
    explain      explanation check (section 10)
    annotation   blind human annotation sheet (section 5)
    report       results.json and data/evaluation/CATALOG_EVALUATION.md
                 (--offline: no API call, no database connection)
    annotation-agreement --sheet FILE   agreement of a filled sheet with each judge

Module map: design (constants of the protocol), ledger (spend cap),
queries and generation (stage 1), runs and e2_index (stage 2), judging
(stage 3), analysis (metrics, contrasts, decision rule, sections 9-10, pure),
annotation (stage 7), report (markdown), cli (stages).
"""
