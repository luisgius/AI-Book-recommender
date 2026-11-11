# Profile parity: local profile versus Postgres (V12)

- Date of the measurement: 2026-11-11T17:46:53+00:00 (git cbd6630)
- Database: Neon (aws-eu-central-1, free plan), PostgreSQL 18.6 (ac70818); extensions: vector 0.8.6, lakebase_text 0.1.3, lakebase_vector 1.1.2, unaccent 1.1
- Catalog: the 177-book evaluation collection in schema `eval177` (and `eval177_hnsw` with an HNSW index)
- Queries: 44 test queries (en 24, es 20), depth 10
- Judgments: `app/evaluation/relevance_judgments_extended_cloud.json` = `app/evaluation/relevance_judgments_extended_z-ai-glm-5.3-flash.json` (sha256 e92185379347...) plus 68 pairs graded by z-ai/glm-5.3-flash (68 holes at the start); every returned top-10 result of every system has a judgment

## Design (fixed before the run)

Same documents (the 177 books of `data/catalog.db`), same document vectors (copied from the local `multilingual-e5-large-instruct` FAISS index of the V06 `vec_e5` arm), same query vectors (each query encoded once and passed to both profiles), same fusion (RRF, k = 60, 20 candidates per leg, in the unchanged `SearchService`). Only the storage and the retrieval engines differ.

| System | Description |
|---|---|
| `bm25_normalized` | BM25, normalizing tokenizer (accents folded, punctuation split, stopwords removed) |
| `vec_e5` | FAISS, intfloat/multilingual-e5-large-instruct (cosine index) |
| `hybrid_norm_e5` | RRF(bm25_normalized, vec_e5) |
| `pg_bm25` | Postgres BM25 (lakebase_text index, k1 1.5, b 0.75), unaccent + per-language stemming |
| `pg_fts_tsrank` | Postgres full-text search ranked with ts_rank_cd (no IDF), same documents and query |
| `pg_vec_exact` | pgvector halfvec, cosine, index scans disabled (exact) |
| `pg_vec_ann` | pgvector halfvec, cosine, the schema's ANN index (approximate) |
| `pg_vec_hnsw` | pgvector halfvec, cosine, pgvector HNSW index (approximate; second schema) |
| `pg_hybrid` | RRF(pg_bm25, pg_vec_exact) through the unchanged SearchService |
| `bm25_normalized_matches` | derived, descriptive: `bm25_normalized` without its zero-score results (rank_bm25 fills the list with books that share no term with the query) |
| `hybrid_norm_e5_matches` | descriptive: RRF(bm25_normalized without zero-score books, vec_e5) through SearchService |

Pre-declared contrasts (nDCG@10, system minus reference): lexical: `pg_bm25` vs `bm25_normalized`; vector_exact: `pg_vec_exact` vs `vec_e5`; hybrid: `pg_hybrid` vs `hybrid_norm_e5`. Two-sided paired randomisation test (sign flips), Holm correction across exactly these three; paired percentile bootstrap 95% interval (10,000 resamples, seed 0). Everything else is descriptive.

## Results

### Pre-declared contrasts

| Contrast | System | Reference | nDCG@10 system [95% CI] | nDCG@10 reference [95% CI] | Difference [95% CI] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| lexical | `pg_bm25` | `bm25_normalized` | 0.492 [0.394, 0.591] | 0.489 [0.395, 0.584] | +0.003 [-0.032, +0.039] | 0.878 | 1.000 |
| vector_exact | `pg_vec_exact` | `vec_e5` | 0.662 [0.586, 0.732] | 0.662 [0.586, 0.732] | +0.000 [+0.000, +0.000] | 1.000 | 1.000 |
| hybrid | `pg_hybrid` | `hybrid_norm_e5` | 0.624 [0.540, 0.705] | 0.592 [0.504, 0.679] | +0.032 [-0.007, +0.074] | 0.139 | 0.417 |

### All systems (descriptive)

| System | nDCG@10 [95% CI] | Recall@10 | P@5 | MRR | Judged@10 | Judged share of returned | Results per query (queries < 10) | nDCG@10 en | nDCG@10 es |
|---|---|---|---|---|---|---|---|---|---|
| `bm25_normalized` | 0.489 [0.395, 0.584] | 0.282 | 0.582 | 0.735 | 1.000 | 1.000 | 10.0 (0) | 0.428 | 0.562 |
| `vec_e5` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 1.000 | 10.0 (0) | 0.592 | 0.745 |
| `hybrid_norm_e5` | 0.592 [0.504, 0.679] | 0.413 | 0.700 | 0.804 | 1.000 | 1.000 | 10.0 (0) | 0.532 | 0.665 |
| `hybrid_norm_e5_matches` | 0.622 [0.540, 0.700] | 0.433 | 0.727 | 0.836 | 1.000 | 1.000 | 10.0 (0) | 0.552 | 0.707 |
| `pg_bm25` | 0.492 [0.394, 0.591] | 0.274 | 0.577 | 0.695 | 0.802 | 1.000 | 8.0 (14) | 0.415 | 0.585 |
| `pg_fts_tsrank` | 0.451 [0.361, 0.539] | 0.264 | 0.605 | 0.740 | 0.802 | 1.000 | 8.0 (14) | 0.342 | 0.583 |
| `pg_vec_exact` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 1.000 | 10.0 (0) | 0.592 | 0.745 |
| `pg_vec_ann` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 1.000 | 10.0 (0) | 0.592 | 0.745 |
| `pg_vec_hnsw` | 0.662 [0.586, 0.732] | 0.492 | 0.750 | 0.849 | 1.000 | 1.000 | 10.0 (0) | 0.592 | 0.745 |
| `pg_hybrid` | 0.624 [0.540, 0.705] | 0.429 | 0.709 | 0.809 | 1.000 | 1.000 | 10.0 (0) | 0.550 | 0.714 |
| `bm25_normalized_matches` | 0.474 [0.376, 0.572] | 0.259 | 0.559 | 0.708 | 0.727 | 1.000 | 7.3 (18) | 0.410 | 0.549 |

Judged@10 divides by 10 (as in ir-measures). Every returned result has a judgment (judged share of returned = 1.000); Judged@10 is below 1.000 only where a system returned fewer than 10 works. Postgres full-text search returns only works that contain a query lexeme; rank_bm25 scores every book, so the local list is always 10 long.

Descriptive differences (no correction, not part of the tested family):

| Pair | nDCG@10 system | nDCG@10 reference | Difference [95% CI] | p (unadjusted) |
|---|---|---|---|---|
| pg_bm25 vs bm25_normalized_matches | 0.492 | 0.474 | +0.019 [-0.013, +0.052] | 0.282 |
| pg_fts_tsrank vs bm25_normalized | 0.451 | 0.489 | -0.038 [-0.097, +0.019] | 0.222 |
| pg_vec_ann vs vec_e5 | 0.662 | 0.662 | +0.000 [+0.000, +0.000] | 1.000 |
| pg_vec_hnsw vs vec_e5 | 0.662 | 0.662 | +0.000 [+0.000, +0.000] | 1.000 |
| pg_hybrid vs hybrid_norm_e5_matches | 0.624 | 0.622 | +0.002 [-0.032, +0.038] | 0.929 |
| hybrid_norm_e5_matches vs hybrid_norm_e5 | 0.622 | 0.592 | +0.030 [+0.007, +0.059] | 0.020 |

### Agreement of the rankings (top 10)

| Pair | Mean overlap | Recall of the reference's top 10 | Identical lists | Identical sets |
|---|---|---|---|---|
| pg_bm25 vs bm25_normalized | 0.518 | 0.518 | 0 of 44 | 2 of 44 |
| pg_bm25 vs bm25_normalized_matches | 0.617 | 0.751 | 4 of 44 | 7 of 44 |
| pg_fts_tsrank vs bm25_normalized | 0.473 | 0.473 | 0 of 44 | 3 of 44 |
| pg_bm25 vs pg_fts_tsrank | 0.782 | 0.782 | 10 of 44 | 20 of 44 |
| pg_vec_exact vs vec_e5 | 1.000 | 1.000 | 44 of 44 | 44 of 44 |
| pg_vec_ann vs pg_vec_exact | 0.998 | 0.998 | 43 of 44 | 43 of 44 |
| pg_vec_hnsw vs pg_vec_exact | 1.000 | 1.000 | 44 of 44 | 44 of 44 |
| pg_hybrid vs hybrid_norm_e5 | 0.725 | 0.725 | 1 of 44 | 4 of 44 |
| pg_hybrid vs hybrid_norm_e5_matches | 0.807 | 0.807 | 6 of 44 | 10 of 44 |

Cosine similarity of the same (query, book) pairs, Postgres halfvec versus FAISS float32: mean absolute difference 4.2e-06, largest 1.9e-05 over 440 pairs.

### Turning the query into a tsquery

| Query form | Queries with no match | Queries with fewer than 10 matches | Median matching works |
|---|---|---|---|
| AND (websearch_to_tsquery in each configuration) | 32 | 39 | 0.0 |
| OR of the query lexemes (shipped) | 2 | 14 | 25.5 |

### Latency from the measuring machine (warm, per query)

Postgres searches go through the pooled connection string over the Internet, from the measuring machine to aws-eu-central-1 (Frankfurt), one round trip per search leg; local searches run in process. The query vector is precomputed in both, so the model is not timed. The last row is a trivial statement through the same pool: the network floor.

| System | Median ms | p95 ms | Warm-up query ms |
|---|---|---|---|
| `bm25_normalized` | 0.07 | 0.09 | 0.13 |
| `vec_e5` | 0.04 | 0.04 | 0.82 |
| `hybrid_norm_e5` | 0.17 | 0.19 | 0.25 |
| `hybrid_norm_e5_matches` | 0.17 | 0.25 | 0.17 |
| `pg_bm25` | 54.54 | 156.70 | 1177.27 |
| `pg_fts_tsrank` | 56.07 | 67.86 | 145.68 |
| `pg_vec_exact` | 66.05 | 75.91 | 199.10 |
| `pg_vec_ann` | 66.29 | 109.21 | 170.18 |
| `pg_vec_hnsw` | 95.50 | 157.77 | 393.41 |
| `pg_hybrid` | 131.47 | 303.80 | 221.18 |
| SELECT 1 (network round trip) | 47.39 | 56.21 | n/a |

### Cold start

pg_hybrid as the API wires it (BM25 + exact vector, RRF), one query, query vector precomputed. 'first search' builds fresh adapters, so it includes reading the catalog facts once (three small statements) besides the two search statements.

| When (UTC) | How the suspension was ensured | Suspended compute: connect + first search = ms | Awake compute: connect + first search = ms | Awake, connection and adapters reused: search ms (median of 5) |
|---|---|---|---|---|
| 2026-11-11T17:53:05+00:00 | Neon scale-to-zero after 5 min idle; Neon API reported state idle (suspended_at 2026-11-11T17:52:47Z) just before | 966 + 825 = 1790 | 324 + 438 = 762 | 140 |

### Build times and sizes

Schema `eval177` (177 works, 177 texts; vector index lakebase_ann). Index builds: work_texts_bm25 0.10 s, work_texts_tsv_gin 0.06 s, works_embedding_ann (lakebase_ann) 0.13 s; VACUUM ANALYZE 0.22 s.

| Relation | Rows | MiB | Note |
|---|---|---|---|
| table `works` | 177 | 1.96 | heap + TOAST + indexes |
| table `work_texts` | 177 | 1.35 | heap + TOAST + indexes |
| table `catalog_info` | 18 | 0.03 | heap + TOAST + indexes |
| index `catalog_info_pkey` |  | 0.02 | btree |
| index `work_texts_bm25` |  | 0.48 | lakebase_bm25 |
| index `work_texts_pkey` |  | 0.03 | btree |
| index `work_texts_tsv_gin` |  | 0.27 | gin |
| index `works_embedding_ann` |  | 0.83 | lakebase_ann |
| index `works_language_idx` |  | 0.02 | btree |
| index `works_pkey` |  | 0.02 | btree |
| index `works_source_source_id_key` |  | 0.04 | btree |
| index `works_year_idx` |  | 0.02 | btree |
| schema total |  | 3.34 |  |

Schema `eval177_hnsw` (177 works, 177 texts; vector index hnsw). Index builds: work_texts_bm25 0.06 s, work_texts_tsv_gin 0.05 s, works_embedding_ann (hnsw) 0.08 s; VACUUM ANALYZE 0.10 s.

| Relation | Rows | MiB | Note |
|---|---|---|---|
| table `works` | 177 | 1.58 | heap + TOAST + indexes |
| table `work_texts` | 177 | 1.36 | heap + TOAST + indexes |
| table `catalog_info` | 18 | 0.03 | heap + TOAST + indexes |
| index `catalog_info_pkey` |  | 0.02 | btree |
| index `work_texts_bm25` |  | 0.48 | lakebase_bm25 |
| index `work_texts_pkey` |  | 0.03 | btree |
| index `work_texts_tsv_gin` |  | 0.27 | gin |
| index `works_embedding_ann` |  | 0.47 | hnsw |
| index `works_language_idx` |  | 0.02 | btree |
| index `works_pkey` |  | 0.02 | btree |
| index `works_source_source_id_key` |  | 0.04 | btree |
| index `works_year_idx` |  | 0.02 | btree |
| schema total |  | 2.97 |  |

Database size (every schema and the system catalogs): 184.67 MiB at the time of the run.

### Larger catalog: `catalog_small` with lakebase_ann (2026-11-11T18:07:04+00:00)

Source `data/catalog_small/catalog.db` (sha256 3569c106424b...).

5000 works, 24314 texts, text-search configurations in use: 11. Load: upsert 41.8 s, 5000 vectors encoded in 300.5 s. Index builds: work_texts_bm25 1.39 s, work_texts_tsv_gin 2.53 s, works_embedding_ann (lakebase_ann) 0.49 s.

ANN recall@10 against exact search (44 test queries): 0.998 (43 of 44 queries at 1.000). Latency (median / p95 ms): exact 97.3 / 178.1; approximate 75.8 / 155.7; bm25 70.8 / 167.0.

| Relation | Rows | MiB | Note |
|---|---|---|---|
| table `works` | 5000 | 54.51 | heap + TOAST + indexes |
| table `work_texts` | 24314 | 105.25 | heap + TOAST + indexes |
| table `catalog_info` | 18 | 0.03 | heap + TOAST + indexes |
| index `catalog_info_pkey` |  | 0.02 | btree |
| index `work_texts_bm25` |  | 24.06 | lakebase_bm25 |
| index `work_texts_pkey` |  | 1.88 | btree |
| index `work_texts_tsv_gin` |  | 21.46 | gin |
| index `works_embedding_ann` |  | 15.00 | lakebase_ann |
| index `works_language_idx` |  | 0.09 | btree |
| index `works_pkey` |  | 0.32 | btree |
| index `works_source_source_id_key` |  | 0.30 | btree |
| index `works_year_idx` |  | 0.10 | btree |
| schema total |  | 159.79 |  |

Stored bytes by column (pg_column_size, after compression): works_embedding 9.78 MiB, works_metadata 15.24 MiB, works_description 3.45 MiB, work_texts_body 16.76 MiB, work_texts_tsv 26.22 MiB.

Free-plan capacity (1 GiB per project, linear extrapolation from 5000 works): 32.7 KiB per work with the ANN index, 29.7 KiB without it; about 31,562 works fit with the ANN index and 34,832 without it (after what the database already holds).

### Larger catalog: `catalog_small` with hnsw (2026-11-11T18:07:42+00:00)

Source `data/catalog_small/catalog.db` (sha256 3569c106424b...).

5000 works, 24314 texts, text-search configurations in use: 11. Load: upsert 41.8 s, 5000 vectors encoded in 300.5 s. Index builds: work_texts_bm25 1.46 s, work_texts_tsv_gin 2.50 s, works_embedding_ann (hnsw) 2.70 s, works_embedding_ann (lakebase_ann) 0.49 s.

ANN recall@10 against exact search (44 test queries): 0.977 (36 of 44 queries at 1.000). Latency (median / p95 ms): exact 101.8 / 170.4; approximate 82.6 / 190.6; bm25 73.6 / 157.4.

| Relation | Rows | MiB | Note |
|---|---|---|---|
| table `works` | 5000 | 52.52 | heap + TOAST + indexes |
| table `work_texts` | 24314 | 105.25 | heap + TOAST + indexes |
| table `catalog_info` | 18 | 0.03 | heap + TOAST + indexes |
| index `catalog_info_pkey` |  | 0.02 | btree |
| index `work_texts_bm25` |  | 24.06 | lakebase_bm25 |
| index `work_texts_pkey` |  | 1.88 | btree |
| index `work_texts_tsv_gin` |  | 21.46 | gin |
| index `works_embedding_ann` |  | 13.03 | hnsw |
| index `works_language_idx` |  | 0.09 | btree |
| index `works_pkey` |  | 0.32 | btree |
| index `works_source_source_id_key` |  | 0.30 | btree |
| index `works_year_idx` |  | 0.10 | btree |
| schema total |  | 157.80 |  |

Stored bytes by column (pg_column_size, after compression): works_embedding 9.78 MiB, works_metadata 15.24 MiB, works_description 3.45 MiB, work_texts_body 16.76 MiB, work_texts_tsv 26.22 MiB.

Free-plan capacity (1 GiB per project, linear extrapolation from 5000 works): 32.3 KiB per work with the ANN index, 29.6 KiB without it; about 31,960 works fit with the ANN index and 34,837 without it (after what the database already holds).

## Reproduce

```
python -m scripts.load_postgres --from data/catalog.db --schema eval177 --embedding-model intfloat/multilingual-e5-large-instruct
python -m scripts.load_postgres --from data/catalog.db --schema eval177_hnsw --embedding-model intfloat/multilingual-e5-large-instruct --vector-index hnsw
python -m app.evaluation.profile_parity run --schema eval177 --hnsw-schema eval177_hnsw
python -m app.evaluation.profile_parity report   # no database, no model, no judge
python -m scripts.load_postgres --from data/catalog_small/catalog.db --schema catalog_small --embedding-model intfloat/multilingual-e5-large-instruct
python -m app.evaluation.profile_parity scale --schema catalog_small
```
