"""
Evaluation of the conversational assistant and the agent (task V22).

Implements the protocol committed before any result,
docs/notes/2026-11-12-protocolo-evaluacion-asistente.md (commit 0ccd5f2):
request sets A, B, D (V14's strata), E (vague requests with a hidden need,
answered by a simulated user) and F (multi-step requests about works in the
catalog); the reference P (the deployed retrieval on the request text, no
LLM); the assistant called through its domain port with the cloud profile's
adapters (no HTTP, no public URL, no daily budget); the agent called directly
on a sample of A and B; judgments by V14's two LLM judges; the route
measures, the contrasts H7-H9 with Holm correction and the two thresholds of
section 7.

Stages (python -m app.evaluation.assistant_evaluation <stage>), each
resumable and each leaving its output on disk under data/evaluation/assistant/:

    sets        build A, B, D (V14) and generate and check E and F
    reference   P on every request text (top 100), no LLM
    assistant   the assistant's turns (A and B twice, D, E with the simulated
                user, F) and the agent on its sample
    judge       judge every new (need, work) pair with both judges
    report      results.json and data/evaluation/ASSISTANT_EVALUATION.md
                (--offline: no API call, no database connection)

Module map: design (constants of the protocol), sets and generation
(stage 1), system (the cloud composition, read-only database, persistent
query vectors, capture of the raw model outputs), simulated_user (the E
loop), runs (stages 2-3), judging (stage 4), analysis (pure measures,
contrasts, thresholds), results and report (stage 5), cli.

Re-measurement (V25 addendum, docs/notes/2026-11-12-protocolo-evaluacion-
asistente-adenda.md): after the routing instruction was revised, the whole
measurement is repeated in data/evaluation/assistant_v25/ (--data-dir), which
reuses the first measurement's sets, P, depth check, query vectors and
judgments without changing them, and adds the held-out set G:

    seed-v25    copy the reusable files and record both routing instructions
    set-g       generate and check G (set_g.py)
    assistant   as above, with the part "g"
    report      also writes data/evaluation/ASSISTANT_EVALUATION_V25.md, both
                measurements side by side (report_v25.py)

Reused from V14 (app/evaluation/catalog_evaluation/, unchanged): the spend
ledger and its recorders, the generator call, the query checks, the judges
with their cache, the statistics and the conclusion rule.
"""
