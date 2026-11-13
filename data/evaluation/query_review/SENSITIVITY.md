# Sensitivity of the evaluations to the doubtful generated queries (V27)

The author reviewed by hand the 254 queries and requests that an LLM wrote for the catalog evaluation (V14) and the assistant evaluation (V22 and its re-measurement V25) (`data/evaluation/query_review/review_sheet.csv`, summary in `data/evaluation/query_review/query_review.json`, task note `docs/notes/tareas/V27-hoja-revision-consultas.md`). This report recomputes, from the saved files only (no network, no LLM, no database), the figures of the published reports that are computed over a set holding an item marked `dudosa` (doubtful) or `mal` (wrong), without those items, to see whether the conclusions of the evaluations depend on them.

- **Decided after the results.** Neither protocol (`docs/notes/2026-11-12-protocolo-evaluacion-catalogo.md`, `docs/notes/2026-11-12-protocolo-evaluacion-asistente.md` and its addendum) declares this analysis; it was decided once the published results were known. It is exploratory and is reported whatever its direction. It does not replace or modify any published result: the reports and their `results.json` are unchanged and remain the results of the evaluations.
- **Excluded items**: every item of the sheet whose verdict is `dudosa` or `mal`: 16 (16 dudosa). The author's reason for each is in the table below.
- **Same method as the published reports**: the same functions, resamples (10,000), seed (0), level (0.05) and Holm families. Before reporting, each evaluation was recomputed with no exclusion and compared with its published `results.json` (catalog: the metrics of A and B, the contrasts and stratum C under both judges, the conclusions and the two rules of section 8; assistant: the whole file of each measurement). Every compared figure is identical; the command stops on the first one that is not, and the test `test_no_exclusion_reproduces_the_published_figures_exactly` checks the same.
- **Choices made here.** (1) The assistant's sets B and D are V14's strata B and D (same ids), reviewed once as catalog rows, so a doubtful query of those strata is also left out of the assistant's set (`b-zh-6`): it changes H7, the route thresholds on A+B and on B, the route stability and, if it is in it, the direct-agent sample. (2) Stratum C: the doubtful version is dropped and its need keeps its other versions. (3) Condition 2 of the embedding rule uses the published operational measurements (all 168 queries).
- **Regenerate**: `.venv/bin/python -m app.evaluation.query_review sensitivity` (reads the sheet; writes only `sensitivity.json` and this file in `data/evaluation/query_review/`).

## Excluded items

| Item | Sheet | Used in | Verdict | Author's comment |
|---|---|---|---|---|
| `b-zh-6` | catalogo B | V14 stratum B; V22 set B; V25 set B | dudosa | Suena a reseña literaria más que a búsqueda: quien cita a Cao Xueqin, Jia Baoyu, Lin Daiyu y las dos mansiones ya conoce el título. La necesidad sí coincide. |
| `c09-fr` | catalogo C | V14 stratum C | dudosa | «pièce de théâtre comique sentimentale» suena a traducción literal; un lector francés escribiría «comédie romantique». La necesidad se entiende. |
| `e-b-ca-1` | asistente E | V22 set E; V25 set E | dudosa | «antologia de versos» es poco natural (lo habitual es «antologia de poesia» o «antologia poètica»). Como versión vaga de la necesidad sí vale. |
| `e-b-ca-4` | asistente E | V22 set E; V25 set E | dudosa | Añade la restricción «en català», que no está en need (need pide ambientación en la Barcelona de posguerra, no idioma). |
| `e-b-de-2` | asistente E | V22 set E; V25 set E | dudosa | Añade «deutsche», que no está en need (need pide ambientación en el Berlín de Weimar, no literatura alemana), y pierde lo histórico. |
| `e-b-fr-5` | asistente E | V22 set E; V25 set E | dudosa | «documentaire» en francés suele entenderse como película documental; para un libro sería «essai» o «livre d'histoire». Ambigua. |
| `e-b-ja-4` | asistente E | V22 set E; V25 set E | dudosa | Añade la restricción «海外文学» (literatura extranjera), que no está en need. |
| `e-b-pt-4` | asistente E | V22 set E; V25 set E | dudosa | «romance de época» suele entenderse como novela romántica de época, no como novela histórica sobre los descubrimientos; puede llevar a otro género. |
| `e-b-ru-5` | asistente E | V22 set E; V25 set E | dudosa | «этнография» no es la categoría general de la necesidad (mitología y creencias eslavas); encajaría mejor «мифология» o «история религии». |
| `f03` | asistente F | V22 set F; V25 set F | dudosa | Mezcla idiomas: en una consulta en francés se esperaría «Le Sursis», no el título inglés «The Reprieve». Tampoco nombra a Sartre («l'auteur»). |
| `f04` | asistente F | V22 set F; V25 set F | dudosa | Mezcla idiomas: un lector japonés escribiría «十二国記», no «The Twelve Kingdoms». |
| `f07` | asistente F | V22 set F; V25 set F | dudosa | «desarrollo inicial como obras del autor» es una expresión vaga y poco natural; no queda claro qué se compara. No nombra al autor. |
| `f20` | asistente F | V22 set F; V25 set F | dudosa | La pregunta es natural, pero «كتاب الفلاحة» es título de varias obras andalusíes (Ibn al-'Awwam, Ibn Bassal, Abu l-Jayr), y al preguntar además por el autor queda ambigua. |
| `g18` | asistente G | V25 set G | dudosa | Cadena de palabras poco natural, calco de «Thai curry cooking»; además «caril» es portugués europeo y el resto de filas pt son brasileñas. |
| `g50` | asistente G | V25 set G | dudosa | Poco natural en inglés: lo habitual es «Camino de Santiago»; «Santiago» solo es ambiguo. |
| `g54` | asistente G | V25 set G | dudosa | Cadena poco natural: «lidar» suelto, calco de «anxiety coping strategies». Lo habitual sería «como lidar com a ansiedade». |

## Outcome

- **No declared decision changes** (57 checked): every declared contrast keeps its significance under each judge and its conclusion, every route threshold keeps its outcome, and the two rules of the catalog protocol's section 8 keep theirs.
- Mechanical flags of a report that change (they are not declared decisions):
  - Assistant evaluation, re-measurement (V25): agent turn seconds, median: second measurement worse: yes (10.4 -> 11.9) -> no (10.3 -> 11.0)
- Catalog evaluation (V14): 2 items excluded; 253 of 435 recomputed figures change; the contrast that moves most is H4, primary judge, +0.347 -> +0.342.
- Assistant evaluation, first measurement (V22): 12 items excluded; 471 of 869 recomputed figures change; the contrast that moves most is H9, second judge, +0.095 -> +0.144.
- Assistant evaluation, re-measurement (V25): 15 items excluded; 514 of 1010 recomputed figures change; the contrast that moves most is H9, second judge, +0.087 -> +0.133.

## Catalog evaluation (V14)

Published report: `data/evaluation/CATALOG_EVALUATION.md`. Excluded here: `b-zh-6` (stratum B), `c09-fr` (stratum C). Strata A and B, the statistical unit: 116 -> 115 queries. 435 figures recomputed, 253 of them change.

### Strata A and B, primary judge `z-ai/glm-5.3-flash`

Each cell: published -> without the doubtful items (a single value when equal).

| System | Queries | nDCG@10 [95 % CI] | P@10 | MRR@10 | Recall@100 (pool) | Judged@10 |
|---|---|---|---|---|---|---|
| S1 | 116 -> 115 | 0.336 [0.287, 0.384] -> 0.339 [0.292, 0.388] | 0.527 -> 0.531 | 0.737 -> 0.743 | 0.464 -> 0.468 | 0.916 -> 0.923 |
| S2 | 116 -> 115 | 0.683 [0.647, 0.717] -> 0.681 [0.645, 0.714] | 0.856 -> 0.857 | 0.966 | 0.837 -> 0.836 | 1.000 |
| S3 | 116 -> 115 | 0.592 [0.554, 0.628] -> 0.589 [0.552, 0.624] | 0.778 | 0.956 | 0.902 | 1.000 |
| S4 | 116 -> 115 | 0.798 [0.772, 0.823] -> 0.797 [0.771, 0.822] | 0.897 -> 0.898 | 0.990 | 0.837 -> 0.836 | 1.000 |
| S5 | 116 -> 115 | 0.769 [0.740, 0.797] -> 0.768 [0.738, 0.796] | 0.883 -> 0.884 | 0.989 | 0.902 | 1.000 |
| S6 | 116 -> 115 | 0.811 [0.782, 0.838] -> 0.810 [0.781, 0.836] | 0.902 | 0.986 | 0.902 | 1.000 |
| S6b | 116 -> 115 | 0.812 [0.783, 0.840] -> 0.811 [0.781, 0.839] | 0.906 | 0.981 | 0.902 | 1.000 |
| E1 | 116 -> 115 | 0.683 [0.647, 0.717] -> 0.681 [0.645, 0.714] | 0.856 -> 0.857 | 0.966 | 0.838 -> 0.837 | 1.000 |
| E2 | 116 -> 115 | 0.426 [0.379, 0.470] -> 0.424 [0.377, 0.469] | 0.685 | 0.822 -> 0.820 | 0.554 -> 0.553 | 1.000 |

### Strata A and B, second judge `minimax/minimax-m3.1-flash`

Each cell: published -> without the doubtful items (a single value when equal).

| System | Queries | nDCG@10 [95 % CI] | P@10 | MRR@10 | Recall@100 (pool) | Judged@10 |
|---|---|---|---|---|---|---|
| S1 | 116 -> 115 | 0.344 [0.294, 0.393] -> 0.347 [0.298, 0.397] | 0.524 -> 0.529 | 0.709 -> 0.715 | 0.455 -> 0.459 | 0.916 -> 0.923 |
| S2 | 116 -> 115 | 0.700 [0.662, 0.737] -> 0.698 [0.660, 0.734] | 0.850 -> 0.851 | 0.953 | 0.829 -> 0.828 | 1.000 |
| S3 | 116 -> 115 | 0.601 [0.562, 0.640] -> 0.598 [0.559, 0.636] | 0.777 | 0.941 | 0.896 | 1.000 |
| S4 | 116 -> 115 | 0.806 [0.778, 0.832] -> 0.805 [0.777, 0.831] | 0.895 -> 0.897 | 0.972 -> 0.971 | 0.829 -> 0.828 | 1.000 |
| S5 | 116 -> 115 | 0.772 [0.744, 0.798] -> 0.770 [0.743, 0.796] | 0.878 -> 0.879 | 0.973 -> 0.972 | 0.896 | 1.000 |
| S6 | 116 -> 115 | 0.803 [0.772, 0.833] -> 0.802 [0.771, 0.831] | 0.900 -> 0.902 | 0.972 | 0.896 | 1.000 |
| S6b | 116 -> 115 | 0.802 [0.769, 0.832] -> 0.801 [0.768, 0.831] | 0.892 -> 0.895 | 0.981 -> 0.980 | 0.896 | 1.000 |
| E1 | 116 -> 115 | 0.700 [0.662, 0.737] -> 0.698 [0.660, 0.734] | 0.850 -> 0.851 | 0.953 | 0.830 -> 0.829 | 1.000 |
| E2 | 116 -> 115 | 0.460 [0.412, 0.507] -> 0.458 [0.409, 0.505] | 0.715 | 0.864 -> 0.863 | 0.558 -> 0.556 | 1.000 |

### Declared contrasts H1-H6 and H5-rep

nDCG@10 difference (system - baseline) on A and B, 95 % paired bootstrap interval, paired randomisation test (10,000 resamples, seed 0), Holm over H1-H6; H5-rep is outside the family (raw p).

| Id | Contrast | Judge | Queries | Difference [95 % CI] | p Holm |
|---|---|---|---|---|---|
| H1 | S3 - S2 | primary | 116 -> 115 | -0.092 [-0.128, -0.055] -> -0.092 [-0.129, -0.056] | <0.001 |
| H1 | S3 - S2 | second | 116 -> 115 | -0.099 [-0.130, -0.068] -> -0.100 [-0.132, -0.069] | <0.001 |
| H2 | S5 - S3 | primary | 116 -> 115 | +0.178 [+0.146, +0.210] -> +0.179 [+0.148, +0.211] | <0.001 |
| H2 | S5 - S3 | second | 116 -> 115 | +0.170 [+0.140, +0.201] -> +0.172 [+0.142, +0.203] | <0.001 |
| H3 | S5 - S4 | primary | 116 -> 115 | -0.029 [-0.045, -0.014] | <0.001 |
| H3 | S5 - S4 | second | 116 -> 115 | -0.035 [-0.054, -0.014] | <0.001 -> 0.001 |
| H4 | S2 - S1 | primary | 116 -> 115 | +0.347 [+0.293, +0.401] -> +0.342 [+0.288, +0.395] | <0.001 |
| H4 | S2 - S1 | second | 116 -> 115 | +0.357 [+0.304, +0.408] -> +0.351 [+0.298, +0.404] | <0.001 |
| H5 | S6 - S5 | primary | 116 -> 115 | +0.042 [+0.012, +0.072] -> +0.042 [+0.011, +0.071] | 0.007 -> 0.006 |
| H5 | S6 - S5 | second | 116 -> 115 | +0.032 [+0.007, +0.057] -> +0.032 [+0.007, +0.056] | 0.013 -> 0.010 |
| H6 | E2 - E1 | primary | 116 -> 115 | -0.258 [-0.302, -0.214] -> -0.257 [-0.302, -0.213] | <0.001 |
| H6 | E2 - E1 | second | 116 -> 115 | -0.241 [-0.287, -0.195] -> -0.240 [-0.288, -0.195] | <0.001 |
| H5-rep | S6b - S5 | primary | 116 -> 115 | +0.043 [+0.013, +0.073] -> +0.043 [+0.013, +0.072] | 0.004 -> 0.005 (raw) |
| H5-rep | S6b - S5 | second | 116 -> 115 | +0.030 [+0.004, +0.056] -> +0.031 [+0.004, +0.056] | 0.024 -> 0.020 (raw) |

### Stratum C (descriptive)

10 needs; the need of each excluded version keeps its other versions (the choice made here), so the pairs of versions whose top 10 overlap is averaged go from 60 to 57. Only the columns that change are shown (primary judge).

| System | fr nDCG@10 (10 -> 9 versions) | Top-10 overlap (60 -> 57 pairs) |
|---|---|---|
| S1 | 0.319 -> 0.345 | 0.065 -> 0.067 |
| S2 | 0.618 -> 0.608 | 0.488 -> 0.505 |
| S3 | 0.459 -> 0.474 | 0.215 -> 0.226 |
| S4 | 0.688 -> 0.693 | 0.567 -> 0.586 |
| S5 | 0.765 -> 0.780 | 0.480 -> 0.498 |
| S6 | 0.746 -> 0.757 | 0.455 -> 0.468 |
| S6b | 0.753 -> 0.765 | 0.492 -> 0.507 |
| E1 | 0.618 -> 0.608 | 0.488 -> 0.505 |
| E2 | 0.463 -> 0.449 | 0.402 -> 0.411 |

### Declared decisions

| Decision | Published | Without the doubtful items | Changes |
|---|---|---|---|
| H1 significant at 0.05, primary judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H1 significant at 0.05, second judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H1 conclusion (both judges) | sustained | sustained | no |
| H2 significant at 0.05, primary judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H2 significant at 0.05, second judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H2 conclusion (both judges) | sustained | sustained | no |
| H3 significant at 0.05, primary judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H3 significant at 0.05, second judge | yes (p Holm <0.001, n 116) | yes (p Holm 0.001, n 115) | no |
| H3 conclusion (both judges) | sustained | sustained | no |
| H4 significant at 0.05, primary judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H4 significant at 0.05, second judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H4 conclusion (both judges) | sustained | sustained | no |
| H5 significant at 0.05, primary judge | yes (p Holm 0.007, n 116) | yes (p Holm 0.006, n 115) | no |
| H5 significant at 0.05, second judge | yes (p Holm 0.013, n 116) | yes (p Holm 0.010, n 115) | no |
| H5 conclusion (both judges) | sustained | sustained | no |
| H6 significant at 0.05, primary judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H6 significant at 0.05, second judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H6 conclusion (both judges) | sustained | sustained | no |
| H5-rep significant at 0.05, primary judge | yes (p 0.004, n 116) | yes (p 0.005, n 115) | no |
| H5-rep significant at 0.05, second judge | yes (p 0.024, n 116) | yes (p 0.020, n 115) | no |
| H5-rep conclusion (both judges) | sustained | sustained | no |
| Hybrid rule: S4 significantly better than S5 under both judges | yes | yes | no |
| Embedding rule, condition 1: H6 does not show qwen significantly better | no | no | no |
| Embedding rule, outcome | keep qwen/qwen3-embedding-8b | keep qwen/qwen3-embedding-8b | no |

## Assistant evaluation, first measurement (V22)

Published report: `data/evaluation/ASSISTANT_EVALUATION.md`. Excluded here: 12 requests (B 1, E 7, F 4: `b-zh-6`, `e-b-ca-1`, `e-b-ca-4`, `e-b-de-2`, `e-b-fr-5`, `e-b-ja-4`, `e-b-pt-4`, `e-b-ru-5`, `f03`, `f04`, `f07`, `f20`). 869 figures recomputed, 471 of them change.

### Contrasts H7-H9

Paired randomisation test (10,000 sign flips, seed 0), 95 % paired bootstrap interval, Holm over H7-H9, under each judge. Each cell: published -> without the doubtful items.

| Id | Judge | n | System / P | Difference [95 % CI] | p | p Holm |
|---|---|---|---|---|---|---|
| H7 | primary | 116 -> 115 | 0.653 / 0.790 -> 0.650 / 0.788 | -0.137 [-0.194, -0.084] -> -0.138 [-0.198, -0.085] | <0.001 | <0.001 |
| H8 | primary | 30 -> 23 | 0.338 / 0.129 -> 0.341 / 0.139 | +0.209 [+0.104, +0.326] -> +0.202 [+0.088, +0.328] | <0.001 -> 0.008 | 0.002 -> 0.016 |
| H9 | primary | 20 -> 16 | 0.925 / 0.795 -> 0.944 / 0.781 | +0.130 [+0.025, +0.260] -> +0.163 [+0.037, +0.319] | 0.059 -> 0.060 | 0.059 -> 0.060 |
| H7 | second | 116 -> 115 | 0.674 / 0.801 -> 0.671 / 0.800 | -0.127 [-0.182, -0.076] -> -0.128 [-0.185, -0.077] | <0.001 | <0.001 |
| H8 | second | 30 -> 23 | 0.351 / 0.153 -> 0.363 / 0.170 | +0.198 [+0.095, +0.311] -> +0.194 [+0.081, +0.320] | <0.001 -> 0.008 | 0.002 -> 0.016 |
| H9 | second | 20 -> 16 | 0.948 / 0.853 -> 0.959 / 0.816 | +0.095 [-0.020, +0.230] -> +0.144 [+0.025, +0.300] | 0.186 -> 0.125 | 0.186 -> 0.125 |

### Declared decisions

| Decision | Published | Without the doubtful items | Changes |
|---|---|---|---|
| H7 significant at 0.05, primary judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H7 significant at 0.05, second judge | yes (p Holm <0.001, n 116) | yes (p Holm <0.001, n 115) | no |
| H7 conclusion (both judges) | sustained | sustained | no |
| H8 significant at 0.05, primary judge | yes (p Holm 0.002, n 30) | yes (p Holm 0.016, n 23) | no |
| H8 significant at 0.05, second judge | yes (p Holm 0.002, n 30) | yes (p Holm 0.016, n 23) | no |
| H8 conclusion (both judges) | sustained | sustained | no |
| H9 significant at 0.05, primary judge | no (p Holm 0.059, n 20) | no (p Holm 0.060, n 16) | no |
| H9 significant at 0.05, second judge | no (p Holm 0.186, n 20) | no (p Holm 0.125, n 16) | no |
| H9 conclusion (both judges) | not significant | not significant | no |
| Protocol section 7, run1: out of scope on A+B at most 2.0 % | no (11/116 = 9.5 %) | no (11/115 = 9.6 %) | no |
| Protocol section 7, run1: questions on B at most 10.0 % | yes (1/72 = 1.4 %) | yes (1/71 = 1.4 %) | no |
| Protocol section 7, run1: all met | no | no | no |
| Protocol section 7, run2: out of scope on A+B at most 2.0 % | no (9/116 = 7.8 %) | no (9/115 = 7.8 %) | no |
| Protocol section 7, run2: questions on B at most 10.0 % | yes (0/72 = 0.0 %) | yes (0/71 = 0.0 %) | no |
| Protocol section 7, run2: all met | no | no | no |

### Other figures of the report

The report's figures over sets that hold an excluded item; every one of the 471 figures that change is listed in `sensitivity.json`.

| Figure | Items | Published | Without the doubtful items |
|---|---|---|---|
| Routes of set B, run 1 | 72 -> 71 | clarify 1, search 70, out_of_scope 1 | clarify 1, search 69, out_of_scope 1 |
| Routes of set B, run 2 | 72 -> 71 | search 72 | search 71 |
| Routes of set E (first turn) | 30 -> 23 | clarify 14, search 16 | clarify 9, search 14 |
| Routes of set F | 20 -> 16 | search 1, agent 19 | search 1, agent 15 |
| Questions on A+B, run 1 | 116 -> 115 | 9/116 = 7.8 % | 9/115 = 7.8 % |
| Route stability: same route in runs 1 and 2 | 116 -> 115 | 106 | 105 |
| H7, group B: assistant / P nDCG@10 (primary judge) | 72 -> 71 | 0.774 / 0.800 | 0.771 / 0.798 |
| H7, group B: assistant / P nDCG@10 (second judge) | 72 -> 71 | 0.788 / 0.811 | 0.786 / 0.810 |
| H7, requests it searched: assistant / P (primary judge) | 96 -> 95 | 0.789 / 0.789 | 0.787 / 0.787 |
| H7, requests it searched: assistant / P (second judge) | 96 -> 95 | 0.814 / 0.814 | 0.813 / 0.812 |
| A+B recommendations: precision / P at the same k (primary judge) | 83 -> 82 | 0.898 / 0.793 | 0.896 / 0.791 |
| A+B recommendations: precision / P at the same k (second judge) | 83 -> 82 | 0.942 / 0.865 | 0.941 / 0.864 |
| E: conversations where it asked | 30 -> 23 | 14 | 9 |
| E: assistant after the conversation / P on the vague request (primary judge) | 30 -> 23 | 0.338 [0.229, 0.453] / 0.129 [0.094, 0.162] | 0.341 [0.228, 0.465] / 0.139 [0.104, 0.174] |
| E: assistant after the conversation / P on the vague request (second judge) | 30 -> 23 | 0.351 [0.241, 0.466] / 0.153 [0.111, 0.197] | 0.363 [0.244, 0.493] / 0.170 [0.124, 0.223] |
| E: P on the original query (primary judge) | 30 -> 23 | 0.775 [0.724, 0.824] | 0.766 [0.706, 0.825] |
| E: P on the original query (second judge) | 30 -> 23 | 0.790 [0.741, 0.837] | 0.791 [0.736, 0.845] |
| E, asked: assistant / P on the vague request (primary judge) | 14 -> 9 | 0.540 [0.356, 0.716] / 0.091 [0.043, 0.146] | 0.605 [0.387, 0.799] / 0.088 [0.034, 0.149] |
| E, asked: assistant / P on the vague request (second judge) | 14 -> 9 | 0.537 [0.351, 0.714] / 0.112 [0.063, 0.161] | 0.612 [0.383, 0.824] / 0.118 [0.054, 0.180] |
| E, did not ask: assistant / P on the vague request (primary judge) | 16 -> 14 | 0.161 [0.122, 0.195] / 0.161 [0.123, 0.196] | 0.172 [0.134, 0.206] / 0.172 [0.134, 0.206] |
| E, did not ask: assistant / P on the vague request (second judge) | 16 -> 14 | 0.188 [0.133, 0.255] / 0.189 [0.135, 0.256] | 0.204 [0.144, 0.274] / 0.204 [0.144, 0.274] |
| F: routes | 20 -> 16 | agent 19, search 1 | agent 15, search 1 |
| F: replies with recommendations | 20 -> 16 | 20 | 16 |
| F: named works recommended / in P's top k | 27 -> 20 | 20 / 24 | 13 / 17 |
| F: nDCG@10 of the reply's list / P (primary judge) | 20 -> 16 | 0.675 [0.594, 0.753] / 0.856 [0.792, 0.913] | 0.682 [0.607, 0.753] / 0.838 [0.763, 0.906] |
| F: nDCG@10 of the reply's list / P (second judge) | 20 -> 16 | 0.633 [0.568, 0.694] / 0.848 [0.763, 0.922] | 0.647 [0.594, 0.697] / 0.820 [0.720, 0.908] |
| F, compare: precision of the recommendations / P (primary judge) | 7 -> 4 | 0.914 / 0.771 | 1.000 / 0.750 |
| F, compare: precision of the recommendations / P (second judge) | 7 -> 4 | 0.943 / 0.929 | 1.000 / 0.875 |
| F, like_but: precision of the recommendations / P (primary judge) | 7 | 0.871 / 0.643 | 0.871 / 0.643 |
| F, like_but: precision of the recommendations / P (second judge) | 7 | 0.907 / 0.650 | 0.907 / 0.650 |
| F, facts: precision of the recommendations / P (primary judge) | 6 -> 5 | 1.000 / 1.000 | 1.000 / 1.000 |
| F, facts: precision of the recommendations / P (second judge) | 6 -> 5 | 1.000 / 1.000 | 1.000 / 1.000 |
| Direct agent: precision / P at the same k (primary judge) | 39 -> 38 | 0.795 [0.687, 0.887] / 0.698 [0.587, 0.799] | 0.789 [0.679, 0.884] / 0.690 [0.582, 0.792] |
| Direct agent: precision / P at the same k (second judge) | 39 -> 38 | 0.781 [0.670, 0.879] / 0.753 [0.638, 0.856] | 0.775 [0.662, 0.875] / 0.746 [0.631, 0.853] |
| Search route, all sets: picks proposed / kept / dropped | 127 -> 119 | 312 / 303 / 9 | 292 / 284 / 8 |
| Agent route in the assistant: runs / recommendations kept | n/a | 19 / 52 | 15 / 42 |
| Language of the reply: ok / mismatch / undecided | n/a | 143 / 21 / 28 | 132 / 18 / 25 |
| All turns: seconds, median / p95 | 308 -> 290 | 5.073 / 10.259 | 5.070 / 10.137 |
| All turns: USD, median / p95 | 308 -> 290 | 0.00248 / 0.00399 | 0.00247 / 0.00377 |
| LLM calls / failed | n/a | 768 / 5 | 722 / 5 |
| Answer calls without output | n/a | 5 | 5 |

## Assistant evaluation, re-measurement (V25)

Published report: `data/evaluation/ASSISTANT_EVALUATION_V25.md`. Excluded here: 15 requests (B 1, E 7, F 4, G 3: `b-zh-6`, `e-b-ca-1`, `e-b-ca-4`, `e-b-de-2`, `e-b-fr-5`, `e-b-ja-4`, `e-b-pt-4`, `e-b-ru-5`, `f03`, `f04`, `f07`, `f20`, `g18`, `g50`, `g54`). 1010 figures recomputed, 514 of them change.

### Contrasts H7-H9

Paired randomisation test (10,000 sign flips, seed 0), 95 % paired bootstrap interval, Holm over H7-H9, under each judge. Each cell: published -> without the doubtful items.

| Id | Judge | n | System / P | Difference [95 % CI] | p | p Holm |
|---|---|---|---|---|---|---|
| H7 | primary | 116 -> 115 | 0.756 / 0.790 -> 0.755 / 0.789 | -0.034 [-0.068, -0.006] | 0.048 -> 0.043 | 0.145 -> 0.130 |
| H8 | primary | 30 -> 23 | 0.206 / 0.128 -> 0.187 / 0.139 | +0.078 [+0.008, +0.167] -> +0.048 [-0.001, +0.124] | 0.094 -> 0.248 | 0.145 -> 0.248 |
| H9 | primary | 20 -> 16 | 0.890 / 0.781 -> 0.900 / 0.764 | +0.109 [+0.013, +0.233] -> +0.136 [+0.019, +0.290] | 0.062 -> 0.063 | 0.145 -> 0.130 |
| H7 | second | 116 -> 115 | 0.768 / 0.802 -> 0.767 / 0.801 | -0.034 [-0.068, -0.006] | 0.048 -> 0.044 | 0.145 -> 0.133 |
| H8 | second | 30 -> 23 | 0.235 / 0.154 -> 0.223 / 0.170 | +0.082 [+0.005, +0.177] -> +0.054 [+0.001, +0.134] | 0.097 -> 0.127 | 0.193 -> 0.255 |
| H9 | second | 20 -> 16 | 0.927 / 0.841 -> 0.934 / 0.801 | +0.087 [-0.033, +0.230] -> +0.133 [-0.004, +0.300] | 0.282 -> 0.158 | 0.282 -> 0.255 |

### Declared decisions

The last rows are the report's "What changed" table (worse is applied mechanically; not a decision of the addendum).

| Decision | Published | Without the doubtful items | Changes |
|---|---|---|---|
| H7 significant at 0.05, primary judge | no (p Holm 0.145, n 116) | no (p Holm 0.130, n 115) | no |
| H7 significant at 0.05, second judge | no (p Holm 0.145, n 116) | no (p Holm 0.133, n 115) | no |
| H7 conclusion (both judges) | not significant | not significant | no |
| H8 significant at 0.05, primary judge | no (p Holm 0.145, n 30) | no (p Holm 0.248, n 23) | no |
| H8 significant at 0.05, second judge | no (p Holm 0.193, n 30) | no (p Holm 0.255, n 23) | no |
| H8 conclusion (both judges) | not significant | not significant | no |
| H9 significant at 0.05, primary judge | no (p Holm 0.145, n 20) | no (p Holm 0.130, n 16) | no |
| H9 significant at 0.05, second judge | no (p Holm 0.282, n 20) | no (p Holm 0.255, n 16) | no |
| H9 conclusion (both judges) | not significant | not significant | no |
| Addendum section 4, secondary (A+B), run1: out of scope on A+B at most 2.0 % | yes (1/116 = 0.9 %) | yes (1/115 = 0.9 %) | no |
| Addendum section 4, secondary (A+B), run1: questions on B at most 10.0 % | yes (0/72 = 0.0 %) | yes (0/71 = 0.0 %) | no |
| Addendum section 4, secondary (A+B), run1: all met | yes | yes | no |
| Addendum section 4, secondary (A+B), run2: out of scope on A+B at most 2.0 % | yes (1/116 = 0.9 %) | yes (1/115 = 0.9 %) | no |
| Addendum section 4, secondary (A+B), run2: questions on B at most 10.0 % | yes (0/72 = 0.0 %) | yes (0/71 = 0.0 %) | no |
| Addendum section 4, secondary (A+B), run2: all met | yes | yes | no |
| Addendum section 4, required: out of scope on the valid G requests at most 2.0 % | no (2/60 = 3.3 %) | no (2/57 = 3.5 %) | no |
| Addendum section 4, required: questions on B at most 10.0 % | yes (0/72 = 0.0 %) | yes (0/71 = 0.0 %) | no |
| Addendum section 4, required: all met | no | no | no |
| run 1 out_of_scope_ab: second measurement worse | no (11/116 = 9.5 % -> 1/116 = 0.9 %) | no (11/115 = 9.6 % -> 1/115 = 0.9 %) | no |
| run 1 clarify_ab: second measurement worse | no (9/116 = 7.8 % -> 4/116 = 3.4 %) | no (9/115 = 7.8 % -> 4/115 = 3.5 %) | no |
| run 1 clarify_b: second measurement worse | no (1/72 = 1.4 % -> 0/72 = 0.0 %) | no (1/71 = 1.4 % -> 0/71 = 0.0 %) | no |
| run 1 d_detection: second measurement worse | no (12/12 = 100.0 % -> 12/12 = 100.0 %) | no (12/12 = 100.0 % -> 12/12 = 100.0 %) | no |
| H7 system mean (primary): second measurement worse | no (0.653 -> 0.756) | no (0.650 -> 0.755) | no |
| H8 system mean (primary): second measurement worse | yes (0.338 -> 0.206) | yes (0.341 -> 0.187) | no |
| H9 system mean (primary): second measurement worse | yes (0.925 -> 0.890) | yes (0.944 -> 0.900) | no |
| H7 system mean (second): second measurement worse | no (0.674 -> 0.768) | no (0.671 -> 0.767) | no |
| H8 system mean (second): second measurement worse | yes (0.351 -> 0.235) | yes (0.363 -> 0.223) | no |
| H9 system mean (second): second measurement worse | yes (0.948 -> 0.927) | yes (0.959 -> 0.934) | no |
| E asked (primary): second measurement worse | no (14/30 -> 8/30) | no (9/23 -> 5/23) | no |
| F agent route: second measurement worse | no (19 -> 20) | no (15 -> 16) | no |
| search picks dropped by verify: second measurement worse | yes (9/312 -> 13/322) | yes (8/292 -> 12/303) | no |
| answer calls without output: second measurement worse | no (5 -> 1) | no (5 -> 1) | no |
| search turn seconds, median: second measurement worse | no (5.4 -> 5.1) | no (5.4 -> 5.1) | no |
| agent turn seconds, median: second measurement worse | yes (10.4 -> 11.9) | no (10.3 -> 11.0) | **yes** |

### Other figures of the report

The report's figures over sets that hold an excluded item; every one of the 514 figures that change is listed in `sensitivity.json`.

| Figure | Items | Published | Without the doubtful items |
|---|---|---|---|
| Routes of set B, run 1 | 72 -> 71 | search 72 | search 71 |
| Routes of set B, run 2 | 72 -> 71 | search 72 | search 71 |
| Routes of set E (first turn) | 30 -> 23 | clarify 8, search 21, out_of_scope 1 | clarify 5, search 18 |
| Routes of set F | 20 -> 16 | agent 20 | agent 16 |
| Questions on A+B, run 1 | 116 -> 115 | 4/116 = 3.4 % | 4/115 = 3.5 % |
| Route stability: same route in runs 1 and 2 | 116 -> 115 | 114 | 113 |
| H7, group B: assistant / P nDCG@10 (primary judge) | 72 -> 71 | 0.801 / 0.800 | 0.799 / 0.798 |
| H7, group B: assistant / P nDCG@10 (second judge) | 72 -> 71 | 0.814 / 0.813 | 0.812 / 0.811 |
| H7, requests it searched: assistant / P (primary judge) | 111 -> 110 | 0.790 / 0.789 | 0.789 / 0.788 |
| H7, requests it searched: assistant / P (second judge) | 111 -> 110 | 0.803 / 0.801 | 0.801 / 0.800 |
| A+B recommendations: precision / P at the same k (primary judge) | 92 -> 91 | 0.857 / 0.774 | 0.855 / 0.771 |
| A+B recommendations: precision / P at the same k (second judge) | 92 -> 91 | 0.893 / 0.828 | 0.892 / 0.826 |
| E: conversations where it asked | 30 -> 23 | 8 | 5 |
| E: assistant after the conversation / P on the vague request (primary judge) | 30 -> 23 | 0.206 [0.128, 0.297] / 0.128 [0.093, 0.161] | 0.187 [0.121, 0.273] / 0.139 [0.103, 0.174] |
| E: assistant after the conversation / P on the vague request (second judge) | 30 -> 23 | 0.235 [0.149, 0.333] / 0.154 [0.112, 0.198] | 0.223 [0.145, 0.317] / 0.170 [0.124, 0.222] |
| E: P on the original query (primary judge) | 30 -> 23 | 0.776 [0.728, 0.824] | 0.768 [0.710, 0.825] |
| E: P on the original query (second judge) | 30 -> 23 | 0.793 [0.747, 0.838] | 0.794 [0.739, 0.847] |
| E, asked: assistant / P on the vague request (primary judge) | 8 -> 5 | 0.420 [0.193, 0.670] / 0.109 [0.052, 0.166] | 0.361 [0.128, 0.646] / 0.133 [0.059, 0.201] |
| E, asked: assistant / P on the vague request (second judge) | 8 -> 5 | 0.462 [0.221, 0.723] / 0.133 [0.081, 0.184] | 0.410 [0.166, 0.714] / 0.169 [0.103, 0.220] |
| E, did not ask: assistant / P on the vague request (primary judge) | 22 -> 18 | 0.128 [0.086, 0.170] / 0.135 [0.095, 0.175] | 0.139 [0.098, 0.180] / 0.141 [0.099, 0.181] |
| E, did not ask: assistant / P on the vague request (second judge) | 22 -> 18 | 0.153 [0.101, 0.211] / 0.161 [0.110, 0.218] | 0.172 [0.115, 0.236] / 0.170 [0.113, 0.236] |
| F: routes | 20 -> 16 | agent 20 | agent 16 |
| F: replies with recommendations | 20 -> 16 | 20 | 16 |
| F: named works recommended / in P's top k | 27 -> 20 | 20 / 26 | 13 / 19 |
| F: nDCG@10 of the reply's list / P (primary judge) | 20 -> 16 | 0.704 [0.606, 0.789] / 0.842 [0.769, 0.907] | 0.723 [0.623, 0.800] / 0.821 [0.734, 0.898] |
| F: nDCG@10 of the reply's list / P (second judge) | 20 -> 16 | 0.658 [0.571, 0.740] / 0.843 [0.757, 0.919] | 0.680 [0.595, 0.763] / 0.814 [0.713, 0.905] |
| F, compare: precision of the recommendations / P (primary judge) | 7 -> 4 | 0.829 / 0.781 | 0.850 / 0.767 |
| F, compare: precision of the recommendations / P (second judge) | 7 -> 4 | 0.886 / 0.952 | 0.900 / 0.917 |
| F, like_but: precision of the recommendations / P (primary judge) | 7 | 0.857 / 0.593 | 0.857 / 0.593 |
| F, like_but: precision of the recommendations / P (second judge) | 7 | 0.907 / 0.593 | 0.907 / 0.593 |
| F, facts: precision of the recommendations / P (primary judge) | 6 -> 5 | 1.000 / 1.000 | 1.000 / 1.000 |
| F, facts: precision of the recommendations / P (second judge) | 6 -> 5 | 1.000 / 1.000 | 1.000 / 1.000 |
| Direct agent: precision / P at the same k (primary judge) | 38 -> 37 | 0.805 [0.705, 0.895] / 0.712 [0.609, 0.811] | 0.811 [0.703, 0.903] / 0.715 [0.607, 0.816] |
| Direct agent: precision / P at the same k (second judge) | 38 -> 37 | 0.833 [0.732, 0.921] / 0.775 [0.663, 0.875] | 0.834 [0.732, 0.926] / 0.775 [0.659, 0.878] |
| Search route, all sets: picks proposed / kept / dropped | 140 -> 133 | 322 / 309 / 13 | 303 / 291 / 12 |
| Agent route in the assistant: runs / recommendations kept | n/a | 20 / 58 | 16 / 48 |
| Language of the reply: ok / mismatch / undecided | n/a | 143 / 29 / 14 | 133 / 26 / 12 |
| All turns: seconds, median / p95 | 302 -> 286 | 5.049 / 11.645 | 4.933 / 10.128 |
| All turns: USD, median / p95 | 302 -> 286 | 0.00256 / 0.00410 | 0.00255 / 0.00393 |
| LLM calls / failed | n/a | 793 / 1 | 749 / 1 |
| Answer calls without output | n/a | 1 | 1 |
| G valid: routed to search | 60 -> 57 | 58/60 = 96.7 % | 55/57 = 96.5 % |
| G valid, routes by form | 60 -> 57 | keywords (search 29, out_of_scope 1); sentence (search 29, out_of_scope 1) | keywords (search 26, out_of_scope 1); sentence (search 29, out_of_scope 1) |
| G valid, routes by language | 60 -> 57 | de (search 10); en (search 10); es (search 9, out_of_scope 1); fr (search 10); it (search 9, out_of_scope 1); pt (search 10) | de (search 10); en (search 9); es (search 9, out_of_scope 1); fr (search 10); it (search 9, out_of_scope 1); pt (search 8) |
| G out of scope: detected | 20 | 20/20 = 100.0 % | 20/20 = 100.0 % |
| G turns on the search route: seconds, median / p95 | 58 -> 55 | 4.810 / 7.164 | 4.812 / 7.040 |

## Figures not recomputed

- Catalog evaluation (V14): Per-language and per-type tables, latency per stage, the S6 replication counts, section 9 (unanswerable queries), section 10 (explanations), the human annotation sheet and the agreement between the judges: descriptive or exploratory figures outside the scope of this analysis (strata A and B, and stratum C); no declared decision rests on them.
- Catalog evaluation (V14): Section 8, operational measurements of the two embedding models (all 168 queries): kept as published; condition 2 of the embedding rule is re-applied with them.
- Assistant evaluation (V22 and V25): Judgment counts (measured pairs, from V14, judged here): they describe the judging work per need, and E's needs are B's needs, so a pair cannot be assigned to one request.
- Assistant evaluation (V22 and V25): Spend and calls by stage and model, the key-usage cross-check, wall-clock per stage and the ledger's list of failed calls: they come from the spend ledger, whose entries do not name the request they served.
- Assistant evaluation (V22 and V25): V25, answer calls that reached their token cap: read from the stage log, whose lines do not name the request.
- Assistant evaluation (V22 and V25): Inventory of the runs (P texts, turns, query vectors, depth check, commits, dates) and the generation checks of E, F and G: they describe what was run and how the sets were written, not the system; P and the depth check are stored by request text.

## Files

- Code: `app/evaluation/review_sensitivity.py` (this analysis), `app/evaluation/query_review.py` (the `sensitivity` command); `assistant_evaluation/results.build_results` takes the excluded requests.
- Tests: `tests/evaluation/test_review_sensitivity.py`.
- `sensitivity.json`: per evaluation the excluded items, the decisions, and every recomputed figure that changes (published and without the doubtful items).
