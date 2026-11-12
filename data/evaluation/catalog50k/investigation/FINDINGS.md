# Investigation of two V14 results: the bge-m3 gap (H6) and the qwen embedding failures

Date: 2026-11-12. Scope: read-only. No evaluation code, report, committed judgment or deployed setting was changed. Every
script, raw call log and output named below is in this folder (`data/evaluation/catalog50k/investigation/`). Run them from the
repository root with `.venv/bin/python <script>`; the round-2 and round-3 trials also need
`PYTHONPATH=data/evaluation/catalog50k/investigation`.

## Question 1: is the bge-m3 result (E2) sound?

**Conclusion.** No defect was found in how E2 was built or queried, and the published E2 figure (nDCG@10 0.426 against 0.683
for E1) stands. The stored vectors are the ones the API returns. The provider serves the real BGE-M3 (it reproduces the model
card's similarities). The index and ids are aligned for all 50,000 works, and bge-m3 finds a work from its title as well as qwen
does. The gap comes from queries that need knowledge of the book that is not in the catalog text: plot, characters, places
("Buendía, Macondo", "Meursault ... Alger"). The 8-billion-parameter qwen model carries that knowledge and the 568-million-parameter
bge-m3 does not. The size of the gap matches the published multilingual retrieval benchmark: MMTEB retrieval 70.88 for
Qwen3-Embedding-8B against 54.60 for BGE-M3 (Qwen3-Embedding-8B model card on Hugging Face).

H6 and the embedding decision rule of protocol section 8 are therefore unchanged: H6 is sustained under both judges, and the
outcome is "keep qwen/qwen3-embedding-8b". Since no defect was found, nothing was re-embedded and no new judgments were made.

### Evidence

**1. Text, prefixes, dimensions, truncation, provider.**
- Per-work text: `Book.get_searchable_text()`, which is title + authors + one description + subjects
  (`app/domain/entities.py:138`). It does not concatenate descriptions in several languages. A work's text is at most bilingual
  (a description in the work's language plus subject labels, mostly English). Length over the 50,000 works (`dump_texts.py`):
  median 596 characters, p99 1,369, maximum 2,058. No text comes near bge-m3's 8,192-token context on DeepInfra, so provider
  truncation cannot have occurred.
- Documents: the same `OpenRouterEmbeddingEncoder` for both models, with no document prefix (`PROMPT_CONVENTIONS`:
  "BAAI/bge-m3": no prompt; Qwen3: instruction on queries only), in requests of at most 32 texts and 20,000 characters, pinned to
  DeepInfra. bge-m3 at its native 1024 d (no `dimensions` sent); qwen cut to 1024 d (Matryoshka).
- Queries: bge-m3 without a prefix, as its model card prescribes for dense retrieval; qwen with the Qwen3 instruction.
- Similarity: cosine (`ExactIndex` normalises both sides). Stored bge vectors are unit length (norms 0.9999999 to 1.0000001),
  with no NaN.
- Provider: DeepInfra (`deepinfra/fp32`). OpenRouter lists DeepInfra (fp32, 8,192 tokens) and Parasail for `baai/bge-m3`
  (`endpoints_baai__bge-m3.json`).

**2. The provider serves the real model** (`q1_reference_check.py`, `q1_reference_check.json`). The BGE-M3 model card's dense
example gives similarities [[0.6265, 0.3477], [0.3499, 0.678]]. Through OpenRouter: DeepInfra [[0.626, 0.348], [0.350, 0.6695]],
Parasail [[0.6259, 0.348], [0.3497, 0.6696]]. Both providers agree with each other and with the card to fp16 precision.

**3. Stored vectors = what the API returns; batching is harmless** (`q1_batch_vs_single.py`, `.json`). The first two original
batches of shard 7 (23 and 27 works) were re-sent exactly as V14 batched them, then each work alone. Cosine with the stored
vectors: batched, min 0.9999954; single, min 0.9999962; single against batched, min 0.9999962. There is no correlation with text
length (r = 0.002). This extends V14's nine-work check to 50 works and to the batched path.

**4. Ids aligned for the whole set; the index equals brute force** (`check_e2_index.py`, `.out`).
- The 25 shards hold 50,000 unique ids, equal to the loader's ids sorted (the order `build_vectors` used).
- A brute-force cosine over the raw matrix gives the same top 10 as the E2 run file for all 168 queries (0 mismatches).
- One group of identical vectors, matching the one group of identical texts in the catalog.
- The frozen bge query vectors equal fresh ones: 168 queries re-encoded, cosine min 0.999998 (`q1_self_retrieval.py`).

**5. Title self-retrieval: bge-m3 is not broken** (`q1_self_retrieval.py`, `.json`). Setup: 200 works drawn with seed 0, the
title alone as the query. bge-m3 searched the local E2 index over all 50,000 works. qwen used the deployed instruction with an
exact search in Neon, read-only and sequential.

| Model | hit@1 | hit@10 | hit@100 | MRR@100 |
|---|---|---|---|---|
| bge-m3 (E2 index) | 0.975 | 0.995 | 1.000 | 0.981 |
| qwen3-embedding-8b (E1) | 0.940 | 1.000 | 1.000 | 0.964 |

186 of the 200 ranks are equal; bge-m3 is better on 10 and qwen on 4. A misaligned or corrupted index would fail this test badly.

**6. Where bge-m3 loses: knowledge that is not in the text** (`q1_known_items.py`, `.json`). Rank of the 12 known-item targets:

| Query | Target | E1 rank | E2 rank |
|---|---|---|---|
| b-en-6 | Jane Eyre | 1 | 14 |
| b-es-6 | Cien años de soledad | 1 | 2,462 |
| b-fr-6 | L'Étranger | 3 | 14,129 |
| b-pt-6 | Memórias Póstumas de Brás Cubas | 2 | 71 |
| b-ru-6 | Мастер и Маргарита | 1 | 13 |
| b-de-6, b-nl-6, b-zh-6, b-ja-6 | (four works) | 1 | 1 |
| b-it-6 | I promessi sposi | 1 | 2 |
| b-ca-6 | La plaça del Diamant | 18 | 22 |
| b-ar-6 | الأيام | 11 | 2 |

The catalog text of "Cien años de soledad" never mentions Buendía or Macondo, and that of "L'Étranger" never mentions Meursault
or Algiers; the query names only those. bge-m3's top results are surface matches instead: "Araby" for "un Arabe", "Edward II"
for "Edward Rochester". Even Jane Eyre, whose text does contain "Rochester" and "Thornfield Hall", comes 14th (cosine 0.512
against 0.539 for the first result). The descriptive and thematic queries show the same pattern in milder form.

**7. Splits by text length and language** (`q1_pool_and_splits.py`, `.json`). The table gives recall@100 of works graded 2 or 3
by the primary judge, over the A+B queries.

| Split | n | E1 | E2 |
|---|---|---|---|
| text < 400 chars | 212 | 0.858 | 0.481 |
| 400-800 | 491 | 0.862 | 0.534 |
| 800-1,200 | 384 | 0.919 | 0.552 |
| >= 1,200 | 192 | 0.870 | 0.620 |
| work language en | 528 | 0.852 | 0.597 |
| es | 280 | 0.875 | 0.550 |
| fr | 105 | 0.933 | 0.410 |
| de | 58 | 0.897 | 0.241 |
| other | 308 | 0.909 | 0.549 |
| work language = query language | 615 | 0.911 | 0.532 |
| work language ≠ query language | 664 | 0.851 | 0.554 |

The gap is present in every length bin, and it is smallest for the longest texts, so it is not a long-text or truncation
effect. It is larger for French and German works. It is no smaller when the work and the query share a language, so it is not
a cross-lingual alignment problem either. The second judge gives the same picture (in the JSON).

Other checks:
- No hubness: 1,299 distinct works across E2's 168 × 10 results, against 1,318 for E1; the most frequent E2 work appears 7 times.
- No preference for short texts: the median length of E2's top-10 works is 880 characters, against 774 for E1 and 596 for the
  catalog (`q1_length_bias.py`).
- The E1 run scores recompute from the stored qwen vectors to within 1.2e-5.

**8. Method of the 177-book measurement** (`app/evaluation/api_models_evaluation.py`, V16). Same encoder class, same text
function, same 32-text / 20,000-character batching, same DeepInfra pin, no bge prefix, exact cosine (FAISS over L2-normalised
vectors). Only the index library differs (FAISS there, numpy here), and check 4 shows numpy equals brute force. So the two
measurements do not differ in method.

The same 44 stratum-A queries ("original") give:
- 177 books: bge 0.614, qwen 0.654;
- 50,000 works: E2 0.420, E1 0.639.

So the change comes from the collection. With 177 books there are few distractors, and finding the right work rarely needs
knowledge that is missing from the text; with 50,000 works in twelve languages it often does.

### What remains uncertain

- Not compared against a local copy of BAAI/bge-m3. It is 2.2 GB, over this task's download limit. The model-card check on two
  providers stands in for that comparison.
- Why the gap is so much wider here than on 177 books is argued (distractors, knowledge-dependent queries), not isolated by an
  experiment. A collection-size simulation cannot be made with these judgments: any sub-collection that keeps the pool returns
  the same top 10 for both models (checked: within-pool nDCG reproduces 0.683 and 0.426 exactly).
- The splits in check 7 are descriptive and their cells are small (58 German works).
- The queries were written by an LLM with world knowledge. Known-item and descriptive queries therefore reward a model that has
  that knowledge, and real users may phrase such queries differently.

## Question 2: why did 51 of 504 qwen calls fail, and what would fix it?

**Conclusion.**
- **The failures:** all 51 V14 failures were read timeouts (two 3 s attempts that hit the 5 s deadline); there was no HTTP
  error. DeepInfra's `qwen/qwen3-embedding-8b` has episodes of slowness lasting minutes. During an episode a retry or a hedge on
  the same provider fails too, while Nebius and SiliconFlow fail independently of it.
- **The changing vectors:** DeepInfra's single endpoint returns at least two discrete vector variants for the same text (cosine
  0.996-0.998 between them). The majority variant equals the stored documents and the frozen E1 queries. Measured on the
  existing judgments, the variants change the order of the top 10 but not its quality.
- **Recommendation:** keep DeepInfra first and hedge to Nebius: if DeepInfra has not answered after 1.0 s, send the same request
  to Nebius and take whichever answers first (5 s deadline, a second DeepInfra request at 2.5 s). In rounds 2 and 3 it failed in 0 of
  540 calls (the deployed policy in 3) at no loss of quality, and it costs about 0.2 extra requests per search. Its rate during a
  DeepInfra episode is estimated from round 1 (about 3 % instead of 12 %), because no full episode recurred in rounds 2 and 3.

### Evidence

**1. What the V14 failures were** (`q2_analyse_operational.py`, `.out`; raw: `../operational/`).

| | qwen3-embedding-8b | bge-m3 |
|---|---|---|
| Failures | 51 / 504, all "ReadTimeout (after 2 attempts)", 5.07-5.30 s | 6 / 504: 4 × HTTP 429 "Model busy", 2 × ReadTimeout |
| When | 39 in 12:31-12:39, 12 in 12:40-12:43 | all in repetition 3 |
| Query length (chars, median) | failed 43, succeeded 46 | failed 72, succeeded 46 |
| Successful latency, median / p95 / p99 / max | 399 / 1,977 / 4,490 / 4,811 ms | 398 / 728 / 1,432 / 1,712 ms |
| Distinct vectors per query (1/2/3) | 36 / 111 / 21 | 74 / 66 / 28 |
| Minimum cosine between repetitions (median query) | 0.99787 | 0.999999 |

There is no dependence on query length. The failures by language are proportional to the number of calls. The operational log
does not record the provider, but every call was pinned to DeepInfra without fallbacks.

**2. Why the same text gives different vectors** (`q2_determinism.py`, `q2_doc_variant.py`, `q2_generation_metadata.out`). Four
queries were each sent 15 times per provider with a 30 s timeout.
- DeepInfra gives exactly 2 bit-distinct variants per text, cosine 0.9972-0.9982 between them. The minority variant answers
  faster (median about 0.3 s against 0.8-1.5 s for the majority), which points to separate replicas or builds behind the one
  endpoint.
- Nebius and SiliconFlow give a new bit pattern on almost every call (6-9 distinct in 15 calls) but agree with themselves and
  with each other to cosine 0.9998-0.9999. That is ordinary bf16/fp8 non-determinism.
- Both sit at cosine 0.9963-0.9966 from DeepInfra's variants: DeepInfra's output is the outlier.
- **Which variant the stored data is:** 8 pooled works were re-embedded as documents 8 times each on DeepInfra. 54 of 64 calls
  equal the stored Neon vectors (cosine 1.000); the other 10 are at 0.9982-0.9993. Nebius is at 0.9965-0.9977 on every call. For
  6 queries, the frozen E1 vector equals the majority DeepInfra variant. The share of calls returning the other variant changed
  over the day: 1-4 of 15 in one test, up to 7 of 8 for one query later.
- OpenRouter exposes nothing that tells the variants apart. The response carries only `provider: "DeepInfra"`. The generation
  record has `provider_name`, latency and an `upstream_id` but no replica or quantisation. OpenRouter lists one DeepInfra endpoint
  for this model (`deepinfra/us`, quantisation "unknown"), so there is no slug that pins one variant. OpenRouter's documentation
  says a base slug like `deepinfra` matches all of a provider's endpoints, and a full slug pins one endpoint. Its embeddings page
  claims embeddings of the same text "will always be identical", which these measurements contradict for DeepInfra's qwen.

**3. The variants do not change quality** (`q2_variant_quality.py`, `.json`). The V14 operational repetitions were split into
calls that match the stored variant (162) and calls that do not (144). Scoring their top 10 on the existing judgments (A+B):

| | nDCG@10 primary | nDCG@10 second | Judged@10 |
|---|---|---|---|
| matching variant | 0.679 | 0.695 | 1.000 |
| other variant | 0.680 | 0.697 | 0.975 |

Paired over 81 queries, other minus matching: -0.007 (both judges), mean absolute difference 0.015. So the "116 of 167 queries
whose top 10 changed" is a reshuffle within relevant works. It is a reproducibility nuisance, not a quality loss.

**4. Other providers' query vectors against DeepInfra's documents** (`q2_nebius_quality.py`, `.json`). Exact search in Neon with
the Nebius or SiliconFlow query vector, 60 queries:
- Nebius: nDCG@10 (A+B, 39 queries) 0.689 primary / 0.694 second, against 0.682 / 0.690 for E1 on the same queries.
- SiliconFlow: 0.682 / 0.688.
- Top-10 overlap with E1: 0.95 for both.

A query vector from Nebius is therefore usable against the stored documents.

**5. Mitigations, real calls.** Each arm is a full policy, as a web request would run it. 60 queries (seed 0) × 3 repetitions =
180 calls per arm, all arms interleaved call by call in a shuffled order, so every arm saw the same provider conditions. "Same
vector" is the number of queries whose three vectors agree to cosine 0.99999; "stored variant" is the share of vectors equal to
the stored-document variant. All wall times are in seconds, failures included.

Round 1, 13:44-14:16 (`q2_mitigations.py`, `q2_mitigations_calls.jsonl`, `q2_mitigations_summary.json`), which included a
DeepInfra episode:

| Arm | Fail | Median | p95 | Requests/call | Same vector | Stored variant |
|---|---|---|---|---|---|---|
| A0 qwen DeepInfra, deployed (3 s × 2, 5 s) | 22/180 (12.2 %) | 0.69 | 5.11 | 1.13 | 27/60 | 0.74 |
| A1 qwen Nebius, deployed | 21/180 (11.7 %) | 1.90 | 5.09 | 1.27 | 1/59 (all ≥ 0.99984) | 0.00 |
| A2 qwen SiliconFlow (fp8), deployed | 29/180 (16.1 %) | 1.57 | 5.12 | 1.32 | 2/58 (all ≥ 0.99978) | 0.00 |
| A3 qwen DeepInfra, 1.5 s × 3, 5 s | 20/180 (11.1 %) | 0.61 | 4.78 | 1.31 | 27/60 | 0.70 |
| A4 qwen DeepInfra, hedged on DeepInfra (2nd request at 1.0 s, max 3) | 10/180 (5.6 %) | 0.68 | 5.00 | 1.46 | 23/60 | 0.72 |
| A5 qwen order [DeepInfra, Nebius], fallbacks allowed | 22/180 (12.2 %) | 0.64 | 5.10 | 1.13 | 28/60 | 0.71 |
| A6 bge-m3 DeepInfra, deployed (reference) | 1/180 (0.6 %, HTTP 429) | 0.46 | 1.00 | 1.01 | 60/60 | 1.00 |

Failures are concentrated in time (`q2_failure_timing.out`): across the qwen arms, 5-7 failures per 3-minute window before
13:54, 17-26 per window between 13:54 and 14:05, then 0-7. A0 failed in 2, 20 and 0 calls in its three repetitions.
- Same-provider strategies fail together. A0 and A3 share 16 of their 22 and 20 failed slots, where 2.4 would be expected if
  they were independent; A0 and A4 share 8, against 1.2 expected.
- Other providers fail roughly independently of DeepInfra. A0 and A1 share 6 failed slots, against 2.6 expected; A0 and A2
  share 4, against 3.5.
- OpenRouter's own fallback (A5) never fired. A slow answer is a client-side timeout, not an error OpenRouter can fall back
  from, so A5 equals A0.

Estimated from these slots, a DeepInfra+Nebius hedge would have failed in 6/180 slots (5 of 62 inside the episode, against 20 for
A0), and adding SiliconFlow in 1/180 (`q2_hedge_estimate_from_round1.out`).

Round 2, 14:27-14:36 (`q2_round2.py`, `q2_round2_calls.jsonl`, `q2_round2_summary.json`), a calm period:

| Arm | Fail | Median | p95 | p99 | Requests/call | Same vector | Stored variant | Answered by |
|---|---|---|---|---|---|---|---|---|
| B0 DeepInfra, deployed [control] | 1/180 (0.6 %) | 0.47 | 1.91 | 2.82 | 1.01 | 19/60 | 0.64 | DeepInfra 179 |
| B1 DeepInfra, one attempt, 10 s | 0/180 | 0.51 | 2.65 | 3.29 | 1.00 | 26/60 | 0.72 | DeepInfra 180 |
| B2 hedge DeepInfra 0 s / Nebius 1.0 s / DeepInfra 2.5 s, 5 s | 0/180 | 0.46 | 1.84 | 2.84 | 1.18 | 16/60 | 0.63 | DeepInfra 174, Nebius 6 |
| B3 same, Nebius at 0.6 s | 0/180 | 0.52 | 1.69 | 2.47 | 1.41 | 17/60 | 0.63 | DeepInfra 167, Nebius 13 |

B1 had 5 successful calls over 3 s (3.07-5.34 s), the slowest of which the deployed 5 s deadline would also have lost.

Round 3, 14:37-14:49 (`q2_round3.py`, `q2_round3_calls.jsonl`, `q2_round3_summary.json`, `q2_round3_timing.out`): 60 queries ×
6 repetitions = 360 calls per arm. The period was mostly calm. The control's only 2 failures and 5 of its 6 calls over 3 s fell in
the first repetition, so a short slow spell, not a full episode.

| Arm | Fail | Median | p95 | p99 | Requests/call | Same vector (6 reps) | Stored variant | Answered by |
|---|---|---|---|---|---|---|---|---|
| B0 DeepInfra, deployed [control] | 2/360 (0.6 %) | 0.46 | 1.60 | 4.21 | 1.02 | 5/60 | 0.62 | DeepInfra 358 |
| B2 hedge DeepInfra 0 s / Nebius 1.0 s / DeepInfra 2.5 s, 5 s | 0/360 | 0.50 | 1.57 | 2.51 | 1.17 | 5/60 | 0.59 | DeepInfra 354, Nebius 6 |
| B4 hedge DeepInfra 0 s / Nebius 1.0 s / SiliconFlow 2.0 s, 5 s | 0/360 | 0.46 | 1.46 | 2.52 | 1.15 | 4/60 | 0.62 | DeepInfra 350, Nebius 9, SiliconFlow 1 |

Over rounds 2 and 3, the recommended hedge (B2) failed 0 of 540 calls and the control 3 of 540. Its median stayed at the
control's level, its p99 fell from 2.8-4.2 s to 2.5-2.8 s, and it averaged about 1.17 requests per search. The three-provider
variant (B4) did as well, but needs a third provider.

No DeepInfra episode of round 1's size occurred during rounds 2 and 3. The hedge's failure rate during an episode is therefore
the round-1 estimate (about 3 % instead of 12 %, or 0.6 % with SiliconFlow as a third request), not a direct measurement.

### Recommendation

**What to change.** In the cloud profile's query-embedding call:
- send the request to DeepInfra as now;
- if no answer has arrived after 1.0 s, or DeepInfra returns an error, send the same request to Nebius pinned (`provider:
  {"order": ["nebius"], "allow_fallbacks": false}`) and use the first successful answer;
- send one more DeepInfra request at 2.5 s;
- keep the 5 s deadline.

The documents stay as they are; Nebius query vectors work against them (evidence 4). The query cache key, if any, should not
include the provider.

**Measured numbers.**
- Calm periods (rounds 2 and 3): 0/540 failures against 3/540 for the deployed policy, median 0.46-0.50 s, p95 1.57-1.84 s,
  about 1.17 requests per search.
- Episode (estimated from round 1 slots): 6/180 failures against 22/180 for the deployed policy.
- Quality on the existing judgments: unchanged.

**Cost.**
- About 0.2 extra embedding requests per search (0.4 if the second request goes at 0.6 s), at 1e-8 USD per token, so negligible.
- One more provider in the deployment's routing and its data policy. Nebius's terms should be checked against the site's
  privacy statement.
- About 30 lines in `OpenRouterHttp` / `OpenRouterEmbeddingEncoder` (two requests in flight, first success wins), with tests.
  The serverless function must allow two concurrent outbound requests.

**What it does not fix.**
- Repeatability: the same query still gets slightly different vectors, from DeepInfra's two variants or from Nebius. Measured,
  this costs no quality, but the top 10 can reorder between identical searches. The only fixes are caching the query vector by
  text (repeat searches become identical; first searches unchanged) or re-embedding the whole catalog with a provider that is
  stable to cosine 0.9999 (Nebius) and using only that provider. The second option gives up the fallback this recommendation
  relies on.
- An episode affecting both providers at once (1 of 180 slots in round 1), and OpenRouter-wide incidents.

**Alternatives measured and rejected.**
- Pinning Nebius or SiliconFlow: no fewer failures (11.7 % and 16.1 % in round 1) and three times slower at the median.
- Shorter attempts or retries on DeepInfra alone (A3): no gain during an episode.
- Hedging on DeepInfra alone (A4): halves the failures, but still 5.6 %.
- OpenRouter fallbacks (A5): never triggered by slowness.
- A longer single deadline (B1): would trade fallbacks for searches of up to 5-10 s, and episode-time latencies were not
  observed uncensored.

## Spend, files and what was not verified

**Spend.** Real cost reported by OpenRouter: 0.0018 USD over 4,899 logged calls (`ledger.jsonl`; the cap was 1.50
USD). The 419 calls without a reported cost are failed calls (timeouts and HTTP errors), for which OpenRouter returns no
usage. Neon: read-only throughout, one connection, sequential statements:
- 200 exact-search statements (title self-retrieval);
- 120 exact-search statements (Nebius and SiliconFlow quality);
- 9 SELECTs of the stored vectors of the 4,018 pooled works.

**Files in this folder.**

| Kind | Files |
|---|---|
| Scripts | `orclient.py` (client + ledger), `dump_texts.py`, `check_e2_index.py`, `q1_batch_vs_single.py`, `q1_reference_check.py`, `q1_self_retrieval.py`, `q1_length_bias.py`, `q1_known_items.py`, `q1_pool_and_splits.py`, `fetch_pool_qwen_vectors.py`, `q2_analyse_operational.py`, `q2_determinism.py`, `q2_doc_variant.py`, `q2_variant_quality.py`, `q2_nebius_quality.py`, `q2_mitigations.py`, `q2_mitigations_analyse.py`, `q2_round2.py`, `q2_round2_analyse.py`, `q2_round3.py`, `q2_round3_analyse.py` |
| Outputs | `*.json`, `*.out`, `*_calls.jsonl` (every trial call with its vector), `ledger.jsonl`, `endpoints_*.json` |
| Git-ignored | `work_texts.jsonl` (44 MB, catalog texts) and `pool_qwen_doc_vectors.*` (16 MB, vectors read from Neon). Both can be regenerated, and the folder's own `.gitignore` excludes them. |

**Not verified.**
- A local BAAI/bge-m3 comparison (download limit).
- The cause inside DeepInfra of the episodes and of the two variants. What these measurements show is consistent with several
  replicas behind one endpoint; DeepInfra's side was not observable.
- The recommended hedge during a full DeepInfra episode, measured directly: rounds 2 and 3 were calm, so the episode figure is estimated from round 1.
- Behaviour from the deployed function's region: all latencies were measured from this laptop.
- Whether Nebius's data policy suits the public site.
