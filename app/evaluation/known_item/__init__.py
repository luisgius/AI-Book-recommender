"""
Known-item evaluation by title and author on the deployed catalog (task V29, part B).

The catalog evaluation (V14) found that adding BM25 candidates to the vector
candidates lowers nDCG@10, but none of its queries named a book by its title
or its author, the case where lexical search is expected to help. This
package measures that case with ground truth taken from the catalog itself
(no LLM judge, no human grade):

- sampling: 300 books of the 50 000-work catalog, stratified by the twelve
  languages of V14, three mechanical queries per book (T, TA, A) and their
  relevant works (pure; reads only the local copy of the catalog);
- fusion: weighted Reciprocal Rank Fusion (pure);
- analysis: metrics, choice of the fusion weight on the development half,
  the six declared contrasts with Holm (pure);
- runner: the runs against the deployed catalog (Neon, OpenRouter), with a
  spend cap (phase 2 only);
- report: results.json and KNOWN_ITEM.md from the saved files.

The protocol is data/evaluation/known_item/PROTOCOL.md.
Command line: python -m app.evaluation.known_item sample | run | report
"""
