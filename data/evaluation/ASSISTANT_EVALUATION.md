# Evaluation of the assistant and the agent (V22)

- Protocol: `docs/notes/2026-11-12-protocolo-evaluacion-asistente.md`, committed in `0ccd5f2` before any result. This report implements it; every deviation is listed in [Deviations from the protocol](#deviations-from-the-protocol).
- System evaluated: the branch at commit `36802ee` (commits recorded per run: `36802ee` x514), cloud profile, Neon schema `catalog_deploy` (50,000 works), called through the `ConversationalAssistant` port with the cloud adapters: no HTTP, no public URL, no daily budget.
- **Relevance judgments come from LLM judges only** (V14's two judges); no person graded anything.
- **The user in set E is simulated** by an LLM; it answers every question briefly and coherently with the hidden need.
- Date of the runs: 2026-11-12 (from 2026-11-12T15:45:26 to 2026-11-12T15:55:05). Report regenerated offline from the saved files (`.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`).
- Models and providers (all through OpenRouter): system LLM `deepseek/deepseek-v4.1-flash` (temperature 0.0, reasoning False); embeddings `qwen/qwen3-embedding-8b` (1024 d) on deepinfra with hedge to nebius; reranker `qwen/qwen3-reranker-8b` on primeintellect; request generator `google/gemini-3.8-flash`; simulated user `mistralai/mistral-small-2603`; judges `z-ai/glm-5.3-flash` (primary) and `minimax/minimax-m3.1-flash` (second), reasoning effort low.
- Real spend reported by OpenRouter: **1.3980 USD** over 2918 calls (cap 4.00 USD); the key's own usage counter moved 1.4033 USD between the first and the last snapshot (it also counts any other use of the key in that window). By stage in [Cost and time](#cost-and-time).

## Summary

- Section 7 thresholds (run 1): out of scope on A+B 11/116 = 9.5 % (max 2.0 %): **not met**; questions on B 1/72 = 1.4 % (max 10.0 %): **met**. Both: **not met**.
- Out-of-scope detection on D: 12/12; questions on A+B: 9/116; agent route on A+B: 0.
- Route stability (A+B run twice): same route on 106/116 (91.4 %).
- H7 (assistant vs P, A+B, nDCG@10): -0.137 (Holm p <0.001) under the primary judge, -0.127 (Holm p <0.001) under the second: **sustained**.
- H8 (assistant after asking vs P on the vague request, E, nDCG@10): +0.209 (Holm p 0.002) under the primary judge, +0.198 (Holm p 0.002) under the second: **sustained**.
- H9 (recommendation precision vs P at the same k, F): +0.130 (Holm p 0.059) under the primary judge, +0.095 (Holm p 0.186) under the second: **not significant**.
- Where H7 comes from (primary judge): on the 96 A+B requests the assistant searched, its nDCG@10 equals P's (0.789 against 0.789; the query was rewritten on 14 of them, changing nDCG@10 on 12). The whole loss is in the requests routed to a question (9) or out of scope (11), which return no books. The follow-up section 7 attaches to H7 (search with the user's text instead of the rewritten query) would therefore not remove the loss; the routing follow-up of the thresholds addresses it.
- E: the assistant asked in 14/30 conversations; nDCG@10 (primary) after the conversation 0.338, P on the vague request 0.129, P on the original B query 0.775.

## Request sets

| Set | Requests | Origin | Expected route |
|---|---|---|---|
| A | 44 | V14 stratum A (10 `ambiguous`) | search |
| B | 72 | V14 stratum B | search |
| D | 12 | V14 stratum D | out of scope |
| E | 30 | generated; needs of B, seed 0 | clarify |
| F | 20 | generated from catalog records | agent |

E and F were written by `google/gemini-3.8-flash` (temperature 0.7, reasoning {'effort': 'low'}, prompt versions {'E': 'v1', 'F': 'v1'}); prompts, raw outputs and dates are in `data/evaluation/assistant/sets/generation/`.

Set E: 10 problems after the first pass, 12 replacements, 0 problems left; language verdicts {'undecided': 14, 'ok': 16} ("undecided": a short request without function words).

- round 1, `b-ca-1`: gives need away: shares ['catalana', 'poesia'] with the precise query. `poesia catalana` -> `antologia de versos`
- round 1, `b-de-2`: gives need away: shares ['historischer', 'roman'] with the precise query. `historischer Roman` -> `deutsche Belletristik`
- round 1, `b-fr-3`: language: not fr: ca 1.00. `récits d'aventure` -> `carnets d'expédition`
- round 1, `b-fr-4`: language: not fr: ca 1.00; gives need away: shares ['anticipation', 'roman'] with the precise query. `roman d'anticipation` -> `de la science-fiction`
- round 1, `b-fr-5`: language: not fr: ca 1.00. `livre d'histoire` -> `documentaire historique`
- round 1, `b-it-3`: gives need away: shares ['fantastici', 'racconti'] with the precise query. `racconti fantastici` -> `antologie narrative`
- round 1, `b-nl-4`: gives need away: shares ['psychologische', 'roman'] with the precise query. `een psychologische roman` -> `aangrijpende literatuur`
- round 1, `b-pt-5`: gives need away: shares ['brasil', 'historia'] with the precise query. `história do Brasil` -> `ensaios sobre sociedade`
- round 1, `b-ru-5`: gives need away: shares ['мифология', 'славянская'] with the precise query. `славянская мифология` -> `книги по этнографии`
- round 2, `b-fr-3`: language: not fr: ca 1.00. `carnets d'expédition` -> `livre de voyage`
- round 2, `b-pt-5`: language: not pt: es 1.00. `ensaios sobre sociedade` -> `livro sobre história`
- round 3, `b-pt-5`: language: not pt: es 1.00; gives need away: shares ['historia', 'livro'] with the precise query. `livro sobre história` -> `obras de não-ficção`

Set F: 0 problems after the first pass, 0 replacements, 0 problems left; language verdicts {'ok': 20} ("undecided": a short request without function words).


Set E (vague request, original B query whose need is hidden):

| Request | Lang | Vague request | Original query |
|---|---|---|---|
| e-b-nl-2 | nl | hedendaagse filosofie | filosofie ethiek moderne technologie |
| e-b-nl-6 | nl | Nederlandse literatuur | roman van willem frederik hermans over geoloog alfred issendorf op expeditie naar meteorieten in noorwegen |
| e-b-ar-6 | ar | سير ذاتية | سيرة ذاتية لعميد الأدب العربي يروي فيها معاناته مع فقدان البصر في طفولته ودراسته في الأزهر وفرنسا |
| e-b-fr-4 | fr | de la science-fiction | un roman d'anticipation dystopique sur le contrôle des esprits et la surveillance de masse |
| e-b-ru-6 | ru | русская классика | роман Булгакова где сатана со свитой появляется в советской Москве |
| e-b-ru-3 | ru | книги по философии | философские эссе экзистенциализм двадцатый век |
| e-b-nl-4 | nl | aangrijpende literatuur | een psychologische roman over het verlies van een kind en rouwverwerking binnen een huwelijk |
| e-b-it-3 | it | antologie narrative | racconti di fantascienza distopica |
| e-b-ru-2 | ru | историческая проза | исторические романы эпоха наполеоновских войн |
| e-b-en-5 | en | ancient history | a book that explains the fall of the Western Roman Empire for general readers |
| e-b-it-2 | it | saggi di filosofia | filosofia morale illuminismo |
| e-b-pt-4 | pt | romance de época | um romance histórico sobre as viagens marítimas e descobrimentos portugueses |
| e-b-pt-5 | pt | obras de não-ficção | livro que explica a história do Brasil colonial e a escravidão |
| e-b-de-2 | de | deutsche Belletristik | Historische Romane Weimarer Republik Berlin |
| e-b-fr-3 | fr | livre de voyage | récits de voyage exploration arctique |
| e-b-ca-3 | ca | llibres de filosofia | filosofia medieval i teologia |
| e-b-en-1 | en | gothic fiction | early Victorian gothic horror novels |
| e-b-pt-3 | pt | livros de poesia | poesia lírica barroca século XVII |
| e-b-nl-1 | nl | historische fictie | historische romans gouden eeuw amsterdam |
| e-b-ca-1 | ca | antologia de versos | poesia catalana contemporània segle XX |
| e-b-it-4 | it | storia antica | un saggio che spiega le cause della caduta dell'impero romano d'occidente |
| e-b-de-5 | de | Philosophie Sachbuch | Ein philosophischer Essay über die moralische Verantwortung der Wissenschaft für künftige Generationen |
| e-b-fr-5 | fr | documentaire historique | un essai historique qui analyse les causes économiques et sociales de la Révolution française |
| e-b-ja-4 | ja | 海外文学の小説 | 孤島に取り残された少年たちのサバイバルと心理的な葛藤を描いた長編小説 |
| e-b-ru-5 | ru | книги по этнографии | книга которая объясняет мифологию и верования древних славян доступным языком |
| e-b-ca-4 | ca | novel·la en català | una novel·la sobre la vida quotidiana d'una dona a la Barcelona de postguerra |
| e-b-en-2 | en | philosophy books | philosophy of mind consciousness essays |
| e-b-nl-5 | nl | oorlogsgeschiedenis | een historisch boek dat uitlegt hoe het dagelijks leven eruitzag tijdens de bezetting in nederland |
| e-b-ar-5 | ar | كتب تاريخ | كتاب تاريخي يتناول رحلات الاستكشاف الجغرافي وحياة الرحالة العرب في العصور الوسطى |
| e-b-en-4 | en | science fiction | a novel about an astronomer discovering signs of alien life |

Set F:

| Request | Kind | Lang | Text |
|---|---|---|---|
| f01 | compare | en | How do Poems by Emily Dickinson, Three Series, Complete and Poems by Emily Dickinson, Series Two differ in scope, and which collection would you recommend for a beginner? |
| f02 | compare | en | Could you compare Justine and Clea in terms of their narrative perspective and place within the sequence? |
| f03 | compare | fr | Pourriez-vous comparer Réflexions sur la question juive et The Reprieve afin de m'expliquer comment la pensée de l'auteur s'exprime différemment à travers l'essai et la fiction ? |
| f04 | compare | ja | 小野不由美の「The Twelve Kingdoms」と「悪霊シリーズ」では、作風や世界観にどのような違いがありますか？ |
| f05 | compare | en | Which would be better suited for a middle-school reader interested in historical fiction, Amos Fortune, Free Man or Before Green Gables? |
| f06 | compare | en | How do William Gibson's short stories The Gernsback Continuum and Skinner's Room compare in their themes and visions of the future? |
| f07 | compare | es | ¿Qué diferencias existen entre El inquilino y El móvil en cuanto a su formato narrativo y desarrollo inicial como obras del autor? |
| f08 | like_but | en | Can you recommend sci-fi books featuring megastructures like Ringworld, but written more recently? |
| f09 | like_but | en | Are there any historical studies similar to Bloodlands that focus on Western Europe or the Pacific theater instead? |
| f10 | like_but | en | I enjoyed the theatrical absurdity of Hurlothrumbo; could you suggest modern comedic plays with a similar nonsense style? |
| f11 | like_but | es | ¿Podrías recomendarme estudios antropológicos similares a Los retos de la diferencia : Los actores de la multiculturalidad entre México y Colombia, pero que se centren en países del Cono Sur? |
| f12 | like_but | en | Could you suggest books similar to Seventh Son that explore magical alternate histories, but set in Europe instead of frontier America? |
| f13 | like_but | de | Können Sie mir Gedichtbände empfehlen, die wie West-östlicher Divan einen interkulturellen Dialog suchen, aber aus dem 20. oder 21. Jahrhundert stammen? |
| f14 | like_but | en | Can you recommend books like The Japanese in Latin America that focus specifically on Asian migrations to North America instead? |
| f15 | facts | fr | De quoi traite exactement l'ouvrage Les Ombres errantes et à quel genre littéraire appartient-il ? |
| f16 | facts | en | What is the central story of Alfred Tennyson's poem Enoch Arden, and when was it originally published? |
| f17 | facts | en | Who is the author of Beat the Reaper, and what kind of plot does this novel follow? |
| f18 | facts | en | What is the background and subject matter of Bond Girls Are Forever? |
| f19 | facts | en | What topics and period of journalism are covered in The Great Shark Hunt? |
| f20 | facts | ar | ما هي الموضوعات الرئيسية التي يناقشها كتاب الفلاحة ومن هو مؤلفه؟ |

## Runs

- P: 166 request texts, 100 results each. Assistant turns: 308 ({'run1/A': 44, 'run1/D': 12, 'run1/F': 20, 'run1/E': 44, 'run1/B': 72, 'run2/A': 44, 'run2/B': 72}); E conversations: 30; direct agent: 40.
- Query vectors stored by text: 327 (0 needed more than one attempt).
- Depth check: for 166/166 P texts the 30 vector candidates the reranker sees are the same set at depth 60 and 200 (166 in the same order; smallest overlap 30/30).

Configuration evaluated (from the application's settings; no secret):

```json
{
 "commit": "36802ee",
 "profile": {
  "name": "cloud",
  "candidates": "vector",
  "schema": "catalog_deploy",
  "lexical_ranking": "bm25",
  "vector_search": "approximate"
 },
 "embedding": {
  "model": "qwen/qwen3-embedding-8b",
  "dimensions": 1024,
  "provider": "deepinfra",
  "zdr": true,
  "hedge_provider": "nebius",
  "hedge_after_s": 1.0,
  "hedge_primary_again_s": 2.5,
  "query_cache_size": 256,
  "query_deadlines": {
   "timeout_s": 3.0,
   "max_retries": 1,
   "deadline_s": 5.0
  },
  "evaluation_attempts": 3
 },
 "reranker": {
  "model": "qwen/qwen3-reranker-8b",
  "provider": "primeintellect",
  "depth": 30,
  "evaluation_deadlines": {
   "timeout_s": 30.0,
   "max_retries": 3
  }
 },
 "llm": {
  "provider": "openrouter",
  "model": "deepseek/deepseek-v4.1-flash",
  "temperature": 0.0,
  "reasoning": false,
  "max_tokens": 1024,
  "timeout_s": 180.0,
  "max_retries": 1,
  "client_max_retries": null
 },
 "assistant": {
  "router_max_tokens": 400,
  "answer_context_books": 6,
  "max_clarifying_turns": 2,
  "max_results": 10
 },
 "agent": {
  "max_tool_rounds": 4,
  "max_reflections": 1,
  "max_recommendations": 5,
  "tool_results": 8
 },
 "usage_budget": null
}
```

## Routes

Route matrix, run1:

| Set | clarify | search | agent | out_of_scope | error | total |
|---|---|---|---|---|---|---|
| A ambiguous | 8 | 1 | 0 | 1 | 0 | 10 |
| A other | 0 | 25 | 0 | 9 | 0 | 34 |
| B | 1 | 70 | 0 | 1 | 0 | 72 |
| D | 0 | 0 | 0 | 12 | 0 | 12 |
| E first turn | 14 | 16 | 0 | 0 | 0 | 30 |
| F | 0 | 1 | 19 | 0 | 0 | 20 |

Route matrix, run2:

| Set | clarify | search | agent | out_of_scope | error | total |
|---|---|---|---|---|---|---|
| A ambiguous | 6 | 3 | 0 | 1 | 0 | 10 |
| A other | 1 | 25 | 0 | 8 | 0 | 34 |
| B | 0 | 72 | 0 | 0 | 0 | 72 |

Route rates:

| Run | Measure | Count | Rate |
|---|---|---|---|
| run1 | clarify_ab | 9/116 | 7.8 % |
| run1 | clarify_a | 8/44 | 18.2 % |
| run1 | clarify_b | 1/72 | 1.4 % |
| run1 | out_of_scope_ab | 11/116 | 9.5 % |
| run1 | out_of_scope_b | 1/72 | 1.4 % |
| run1 | agent_ab | 0/116 | 0.0 % |
| run1 | d_detection | 12/12 | 100.0 % |
| run2 | clarify_ab | 7/116 | 6.0 % |
| run2 | clarify_a | 7/44 | 15.9 % |
| run2 | clarify_b | 0/72 | 0.0 % |
| run2 | out_of_scope_ab | 9/116 | 7.8 % |
| run2 | out_of_scope_b | 0/72 | 0.0 % |
| run2 | agent_ab | 0/116 | 0.0 % |
| run2 | d_detection | 0/0 | n/a |

### Section 7 thresholds

| Run | Measure | Value | Maximum | Outcome |
|---|---|---|---|---|
| run1 | out_of_scope_ab | 11/116 = 9.5 % | 2.0 % | **not met** |
| run1 | clarify_b | 1/72 = 1.4 % | 10.0 % | met |

run1: both thresholds **not met**. By the protocol, the routing node's instructions are to be revised and the route measure repeated, declaring the change, before the assistant becomes the public page's main entry point. That is a separate task; nothing was changed here.

| Run | Measure | Value | Maximum | Outcome |
|---|---|---|---|---|
| run2 | out_of_scope_ab | 9/116 = 7.8 % | 2.0 % | **not met** |
| run2 | clarify_b | 0/72 = 0.0 % | 10.0 % | met |

run2: both thresholds **not met**. By the protocol, the routing node's instructions are to be revised and the route measure repeated, declaring the change, before the assistant becomes the public page's main entry point. That is a separate task; nothing was changed here.

A and B requests that got a question or an out-of-scope reply (run 1):

| Request | Set | Request text | Reply |
|---|---|---|---|
| q03 | A | recetas cocina rumana | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros parecidos a uno que te haya gustado. ¿Qué te |
| q11 | A | computer vision opencv beginner guide | I can only help with books from this catalog: searching by topic, author or period, comparing books, or finding books similar to one you liked. What would you l |
| q14 | A | GPT large language models explained | I can only help with books from this catalog: searching by topic, author or period, comparing books, or finding books similar to one you liked. What would you l |
| q25 | A | recetas tradicionales cocina mediterranea | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros parecidos a uno que te haya gustado. ¿Qué te |
| q27 | A | postres caseros faciles sin horno | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros parecidos a uno que te haya gustado. ¿Qué te |
| q28 | A | Italian pasta recipes homemade sauce | I can only help with books from this catalog: searching by topic, author or period, comparing books, or finding books similar to one you liked. What would you l |
| q31 | A | recetas de mariscos y pescados frescos | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros parecidos a uno que te haya gustado. ¿Qué te |
| q32 | A | Asian cuisine wok stir fry techniques | I can only help with books from this catalog: searching by topic, author or period, comparing books, or finding books similar to one you liked. What would you l |
| q33 | A | comida mexicana autentica tacos enchiladas | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros parecidos a uno que te haya gustado. ¿Qué te |
| q35 | A | python | What kind of Python book are you looking for? |
| q36 | A | historia | ¿Qué tipo de historia buscas? |
| q37 | A | recetas faciles | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros parecidos a uno que te haya gustado. ¿Qué te |
| q38 | A | java | What kind of book are you looking for related to Java? |
| q39 | A | apple | What kind of book are you looking for? |
| q40 | A | networks | What kind of networks are you interested in? |
| q41 | A | learning | What kind of learning books are you looking for? |
| q43 | A | classic | What kind of classic books are you looking for? |
| q44 | A | guerra | ¿Qué tipo de libro sobre la guerra buscas? |
| b-fr-6 | B | Meursault tue un Arabe sur une plage sous le soleil d'Alger | I can only help with books from this catalog: searching by topic, author or period, comparing books, or finding books similar to one you liked. What would you l |
| b-de-5 | B | Ein philosophischer Essay über die moralische Verantwortung der Wissenschaft für künftige Generationen | Suchst du einen philosophischen Essay (Sachbuch) zu diesem Thema oder eher einen Roman mit dieser Thematik? |

D requests (run 1):

| Request | Text | Route |
|---|---|---|
| d-en | train schedule from London to Edinburgh tomorrow morning | out_of_scope |
| d-es | pronóstico del tiempo para Madrid este fin de semana | out_of_scope |
| d-fr | vols pas chers Paris New York départ vendredi | out_of_scope |
| d-de | Öffnungszeiten Bürgeramt Berlin Neukölln heute | out_of_scope |
| d-it | numero di telefono farmacia di turno più vicina | out_of_scope |
| d-pt | cotação do dólar hoje em reais | out_of_scope |
| d-ca | com arribar a l'hospital Clínic amb metro | out_of_scope |
| d-nl | uitslagen voetbal eredivisie gisteravond | out_of_scope |
| d-ru | доставка пиццы круглосуточно рядом со мной | out_of_scope |
| d-zh | 北京今天空气质量指数实时查询 | out_of_scope |
| d-ja | 渋谷駅周辺の現在空いているコインパーキング | out_of_scope |
| d-ar | مواقيت صلاة الفجر والظهر اليوم في القاهرة | out_of_scope |

Routes forced by code, with their reason:

- run1: {'E: invalid_clarification -> search': 3}
- run2: {'A: invalid_clarification -> search': 2}
- E, later turns: {'turn 2: search': 14}

### Route stability

A and B were run twice. Same route on 106/116 requests (91.4 %); by set: A 36/44, B 70/72. Changes: {'search -> clarify': 1, 'out_of_scope -> search': 4, 'search -> out_of_scope': 2, 'clarify -> search': 3} (b-de-5, b-fr-6, q02, q11, q12, q13, q25, q32, q40, q41).

## Contrasts H7-H9

One family, two-sided, paired randomisation test (10,000 sign flips, seed 0), 95 % paired bootstrap interval, Holm correction over the three contrasts, under each judge. A conclusion holds only with the same sign and significance under both judges.

| Id | Judge | n | System | Reference | Difference | 95 % CI | p | p Holm |
|---|---|---|---|---|---|---|---|---|
| H7 | primary `z-ai/glm-5.3-flash` | 116 | 0.653 | 0.790 | -0.137 | [-0.194, -0.084] | <0.001 | <0.001 |
| H7 | second `minimax/minimax-m3.1-flash` | 116 | 0.674 | 0.801 | -0.127 | [-0.182, -0.076] | <0.001 | <0.001 |
| H7 | **outcome** |  |  |  |  |  |  | **sustained** |
| H8 | primary `z-ai/glm-5.3-flash` | 30 | 0.338 | 0.129 | +0.209 | [+0.104, +0.326] | <0.001 | 0.002 |
| H8 | second `minimax/minimax-m3.1-flash` | 30 | 0.351 | 0.153 | +0.198 | [+0.095, +0.311] | <0.001 | 0.002 |
| H8 | **outcome** |  |  |  |  |  |  | **sustained** |
| H9 | primary `z-ai/glm-5.3-flash` | 20 | 0.925 | 0.795 | +0.130 | [+0.025, +0.260] | 0.059 | 0.059 |
| H9 | second `minimax/minimax-m3.1-flash` | 20 | 0.948 | 0.853 | +0.095 | [-0.020, +0.230] | 0.186 | 0.186 |
| H9 | **outcome** |  |  |  |  |  |  | **not significant** |

What follows from each (protocol section 7): H7 negative and sustained -> the search route would use the user's text instead of the rewritten query on single-turn conversations (separate task); H8 not sustained -> the thesis says the clarifying question showed no measurable gain with this method; H9 informs when the agent's cost pays off and changes nothing.

### H7 in detail (primary `z-ai/glm-5.3-flash`)

| Group | Requests | Assistant nDCG@10 | P nDCG@10 |
|---|---|---|---|
| A ambiguous | 10 | 0.091 | 0.769 |
| A other | 34 | 0.562 | 0.775 |
| B | 72 | 0.774 | 0.800 |
| route clarify | 9 | 0.000 | 0.761 |
| route out_of_scope | 11 | 0.000 | 0.825 |
| route search | 96 | 0.789 | 0.789 |

Only the requests the assistant searched (descriptive): 96 requests, assistant 0.789, P 0.789, difference +0.000 [-0.003, +0.003], raw p 0.915. The condensed query differed from the request text on 14 of them (nDCG@10 changed on 12, mean difference +0.001) and was the request text itself on 82.
Recommendations on A+B: 83 replies with picks; precision (grade >= 2) 0.898 against P 0.793 at the same k.

### H7 in detail (second `minimax/minimax-m3.1-flash`)

| Group | Requests | Assistant nDCG@10 | P nDCG@10 |
|---|---|---|---|
| A ambiguous | 10 | 0.090 | 0.759 |
| A other | 34 | 0.603 | 0.792 |
| B | 72 | 0.788 | 0.811 |
| route clarify | 9 | 0.000 | 0.751 |
| route out_of_scope | 11 | 0.000 | 0.734 |
| route search | 96 | 0.814 | 0.814 |

Only the requests the assistant searched (descriptive): 96 requests, assistant 0.814, P 0.814, difference +0.001 [-0.003, +0.005], raw p 0.778. The condensed query differed from the request text on 14 of them (nDCG@10 changed on 12, mean difference +0.004) and was the request text itself on 82.
Recommendations on A+B: 83 replies with picks; precision (grade >= 2) 0.942 against P 0.865 at the same k.

## Set E: vague requests and the simulated user

primary `z-ai/glm-5.3-flash`: asked in 14/30 (46.7 %); questions per conversation {0: 16, 1: 14}; assistant turns {1: 16, 2: 14}; stop {'answered': 30}; final route {'search': 30}.

| Subset | n | Assistant after the conversation | P on the vague request | P on the original query (reference) |
|---|---|---|---|---|
| all | 30 | 0.338 | 0.129 | 0.775 |
| asked | 14 | 0.540 | 0.091 | 0.787 |
| did not ask | 16 | 0.161 | 0.161 | 0.764 |

| Request | Lang | Vague | Questions | Final route | Final query | nDCG final | P vague | P original |
|---|---|---|---|---|---|---|---|---|
| e-b-nl-2 | nl | hedendaagse filosofie | 0 | search | hedendaagse filosofie | 0.183 | 0.183 | 0.697 |
| e-b-nl-6 | nl | Nederlandse literatuur | 1 | search | Nederlandse literatuur roman | 0.200 | 0.000 | 0.933 |
| e-b-ar-6 | ar | سير ذاتية | 1 | search | سير ذاتية لكتّاب | 0.000 | 0.000 | 0.433 |
| e-b-fr-4 | fr | de la science-fiction | 0 | search | de la science-fiction | 0.137 | 0.137 | 0.766 |
| e-b-ru-6 | ru | русская классика | 0 | search | русская классика | 0.000 | 0.000 | 0.761 |
| e-b-ru-3 | ru | книги по философии | 0 | search | книги по философии | 0.170 | 0.170 | 0.777 |
| e-b-nl-4 | nl | aangrijpende literatuur | 0 | search | aangrijpende literatuur | 0.076 | 0.076 | 0.531 |
| e-b-it-3 | it | antologie narrative | 0 | search | antologie narrative | 0.230 | 0.230 | 0.614 |
| e-b-ru-2 | ru | историческая проза | 1 | search | историческая проза о периоде наполеоновских войн в Европе и России | 0.933 | 0.206 | 1.000 |
| e-b-en-5 | en | ancient history | 0 | search | ancient history | 0.194 | 0.194 | 0.655 |
| e-b-it-2 | it | saggi di filosofia | 0 | search | saggi di filosofia | 0.131 | 0.131 | 0.771 |
| e-b-pt-4 | pt | romance de época | 1 | search | romance de época ambientado na Idade dos Descobrimentos e expedições marítimas portuguesas | 0.899 | 0.133 | 0.874 |
| e-b-pt-5 | pt | obras de não-ficção | 1 | search | obras de não-ficção sobre História, com foco na escravidão e comércio transatlântico no Brasil colonial | 0.659 | 0.000 | 0.612 |
| e-b-de-2 | de | deutsche Belletristik | 0 | search | deutsche Belletristik | 0.011 | 0.011 | 0.949 |
| e-b-fr-3 | fr | livre de voyage | 1 | search | récit de voyage sur les expéditions polaires et l'exploration arctique | 0.865 | 0.132 | 0.795 |
| e-b-ca-3 | ca | llibres de filosofia | 0 | search | llibres de filosofia | 0.215 | 0.215 | 1.000 |
| e-b-en-1 | en | gothic fiction | 0 | search | gothic fiction | 0.229 | 0.229 | 0.641 |
| e-b-pt-3 | pt | livros de poesia | 0 | search | livros de poesia | 0.229 | 0.229 | 0.893 |
| e-b-nl-1 | nl | historische fictie | 1 | search | historische fictie over de Nederlandse geschiedenis, specifiek de Gouden Eeuw in Amsterdam | 0.887 | 0.250 | 0.787 |
| e-b-ca-1 | ca | antologia de versos | 0 | search | antología de versos | 0.156 | 0.164 | 0.639 |
| e-b-it-4 | it | storia antica | 1 | search | storia antica, Roma antica, in particolare il declino e la caduta dell'Impero Romano d'Occidente | 0.652 | 0.079 | 0.864 |
| e-b-de-5 | de | Philosophie Sachbuch | 0 | search | Philosophie Sachbuch | 0.254 | 0.254 | 0.903 |
| e-b-fr-5 | fr | documentaire historique | 1 | search | documentaire historique sur la Seconde Guerre mondiale | 0.000 | 0.017 | 0.795 |
| e-b-ja-4 | ja | 海外文学の小説 | 1 | search | 20世紀の海外文学の小説 | 0.000 | 0.024 | 0.849 |
| e-b-ru-5 | ru | книги по этнографии | 1 | search | книги по этнографии о культуре древних славян | 0.764 | 0.000 | 0.727 |
| e-b-ca-4 | ca | novel·la en català | 1 | search | novel·la històrica en català sobre la postguerra a Barcelona | 0.456 | 0.308 | 0.797 |
| e-b-en-2 | en | philosophy books | 0 | search | philosophy books | 0.242 | 0.242 | 1.000 |
| e-b-nl-5 | nl | oorlogsgeschiedenis | 1 | search | boeken over oorlogsgeschiedenis, Tweede Wereldoorlog | 0.403 | 0.031 | 0.797 |
| e-b-ar-5 | ar | كتب تاريخ | 1 | search | كتب تاريخ إسلامي عن الرحلات الجغرافية والاستكشاف في العصور الوسطى | 0.849 | 0.098 | 0.756 |
| e-b-en-4 | en | science fiction | 0 | search | science fiction | 0.115 | 0.115 | 0.629 |

second `minimax/minimax-m3.1-flash`: asked in 14/30 (46.7 %); questions per conversation {0: 16, 1: 14}; assistant turns {1: 16, 2: 14}; stop {'answered': 30}; final route {'search': 30}.

| Subset | n | Assistant after the conversation | P on the vague request | P on the original query (reference) |
|---|---|---|---|---|
| all | 30 | 0.351 | 0.153 | 0.790 |
| asked | 14 | 0.537 | 0.112 | 0.768 |
| did not ask | 16 | 0.188 | 0.189 | 0.809 |

## Set F and the agent

primary `z-ai/glm-5.3-flash`: routes {'agent': 19, 'search': 1}; 20/20 replies with recommendations (without: none); works named in the requests that were recommended: 20/27 (in P's top k: 24); nDCG@10 of the reply's list 0.675 against P 0.856.

| By | Value | Requests | Precision of the recommendations | P at the same k |
|---|---|---|---|---|
| route | agent | 19 | 0.921 | 0.837 |
| route | search | 1 | 1.000 | 0.000 |
| kind | compare | 7 | 0.914 | 0.771 |
| kind | like_but | 7 | 0.871 | 0.643 |
| kind | facts | 6 | 1.000 | 1.000 |

second `minimax/minimax-m3.1-flash`: routes {'agent': 19, 'search': 1}; 20/20 replies with recommendations (without: none); works named in the requests that were recommended: 20/27 (in P's top k: 24); nDCG@10 of the reply's list 0.633 against P 0.848.

| By | Value | Requests | Precision of the recommendations | P at the same k |
|---|---|---|---|---|
| route | agent | 19 | 0.945 | 0.897 |
| route | search | 1 | 1.000 | 0.000 |
| kind | compare | 7 | 0.943 | 0.929 |
| kind | like_but | 7 | 0.907 | 0.650 |
| kind | facts | 6 | 1.000 | 1.000 |

### The agent called directly (40 requests of A and B, seed 0)

primary `z-ai/glm-5.3-flash`: 39/40 runs with recommendations (0 failed); precision of the agent's recommendations 0.795 [0.687, 0.887] against P at the same k 0.698 [0.587, 0.799]; on the 29 requests where the assistant (run 1) also recommended: agent 0.890, assistant 0.908. Routes the assistant took on these requests: {'search': 35, 'out_of_scope': 5}; number of recommendations: {5: 34, 4: 1, 3: 2, 1: 2, 0: 1}.

second `minimax/minimax-m3.1-flash`: 39/40 runs with recommendations (0 failed); precision of the agent's recommendations 0.781 [0.670, 0.879] against P at the same k 0.753 [0.638, 0.856]; on the 29 requests where the assistant (run 1) also recommended: agent 0.903, assistant 0.948. Routes the assistant took on these requests: {'search': 35, 'out_of_scope': 5}; number of recommendations: {5: 34, 4: 1, 3: 2, 1: 2, 0: 1}.

## Grounding

Search route (run 1 and every E turn): picks the answer call proposed and the verify node kept, why the others were dropped, and citations.

| Set | Replies | No answer output | Every pick dropped | Picks proposed | Kept | Dropped | Reasons | Citations proposed | Citations of kept picks |
|---|---|---|---|---|---|---|---|---|---|
| A | 26 | 0 | 0 | 54 | 54 | 0 | - | 110 | 107 |
| B | 70 | 1 | 3 | 174 | 167 | 7 | {'no_valid_citation': 7} | 345 | 330 |
| D | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 | 0 |
| E | 30 | 0 | 1 | 83 | 81 | 2 | {'no_valid_citation': 2} | 161 | 158 |
| F | 1 | 0 | 0 | 1 | 1 | 0 | - | 1 | 1 |
| all | 127 | 1 | 4 | 312 | 303 | 9 | {'no_valid_citation': 9} | 617 | 596 |

- Agent route inside the assistant: 19 runs, 52 recommendations kept, 0 recommended works that no tool had returned (dropped), 0 runs needed the reflection step; tool calls {'search_books': 38, 'get_book_details': 41, 'find_similar_books': 2}, per run 4.0 / 7.2.
- Agent called directly: 40 runs, 182 recommendations kept, 0 recommended works that no tool had returned (dropped), 4 runs needed the reflection step; tool calls {'search_books': 97, 'get_book_details': 60}, per run 4.0 / 6.0.

## Language of the reply

Run 1 and every E turn. The catalog's language detector decides (twelve languages, 40 letters or more); a reply it cannot decide is judged by its writing system for Russian, Chinese, Japanese and Arabic and is "undecided" otherwise. The fixed replies (out of scope, no results, results without picks) exist only in Spanish and English.

| Request language | ok | mismatch | undecided |
|---|---|---|---|
| ar | 10 | 2 | 0 |
| ca | 5 | 3 | 3 |
| de | 7 | 2 | 1 |
| en | 43 | 0 | 5 |
| es | 19 | 0 | 10 |
| fr | 8 | 2 | 4 |
| it | 8 | 2 | 1 |
| ja | 9 | 1 | 0 |
| nl | 10 | 4 | 1 |
| pt | 6 | 3 | 3 |
| ru | 12 | 1 | 0 |
| zh | 6 | 1 | 0 |

| Route | ok | mismatch | undecided |
|---|---|---|---|
| agent | 18 | 0 | 1 |
| clarify | 8 | 3 | 12 |
| out_of_scope | 5 | 11 | 7 |
| search | 112 | 7 | 8 |

| Text of the reply | ok | mismatch | undecided |
|---|---|---|---|
| fixed reply | 15 | 14 | 9 |
| written by the LLM | 128 | 7 | 19 |

Replies written by the LLM only, by request language:

| Request language | ok | mismatch | undecided |
|---|---|---|---|
| ar | 9 | 1 | 0 |
| ca | 5 | 2 | 2 |
| de | 7 | 0 | 0 |
| en | 33 | 0 | 5 |
| es | 16 | 0 | 3 |
| fr | 8 | 0 | 4 |
| it | 8 | 1 | 1 |
| ja | 9 | 0 | 0 |
| nl | 9 | 2 | 1 |
| pt | 6 | 1 | 3 |
| ru | 12 | 0 | 0 |
| zh | 6 | 0 | 0 |

Mismatches (request -> detected): {'nl->en': 4, 'fr->en': 2, 'pt->en': 2, 'ca->es': 2, 'ar->en': 2, 'de->en': 2, 'ca->en': 1, 'it->en': 2, 'ru->en': 1, 'zh->en': 1, 'ja->en': 1, 'pt->nl': 1}.

## Latency and cost per turn

Wall time of `reply()` from this laptop, four requests in flight; cost as reported by OpenRouter for the calls recorded in the request's thread.

| Route | Turns | Seconds median / p95 | USD median / p95 | LLM calls median / p95 |
|---|---|---|---|---|
| agent | 19 | 10.4 / 16.3 | 0.00516 / 0.00859 | 5.0 / 5.1 |
| clarify | 30 | 1.3 / 2.3 | 0.00015 / 0.00024 | 1.0 / 1.0 |
| out_of_scope | 32 | 0.9 / 1.5 | 0.00008 / 0.00047 | 1.0 / 1.0 |
| search | 227 | 5.4 / 9.5 | 0.00264 / 0.00348 | 2.0 / 2.0 |
| all turns | 308 | 5.1 / 10.3 | 0.00248 / 0.00399 | 2.0 / 5.0 |
| direct agent | 40 | 11.5 / 18.6 | 0.00559 / 0.01138 | 4.0 / 6.0 |

| Set | Turns | Seconds median / p95 | USD median / p95 |
|---|---|---|---|
| A | 88 | 3.6 / 7.9 | 0.00198 / 0.00322 |
| B | 144 | 5.4 / 9.4 | 0.00270 / 0.00357 |
| D | 12 | 0.8 / 1.4 | 0.00008 / 0.00037 |
| E | 44 | 5.1 / 8.9 | 0.00221 / 0.00330 |
| F | 20 | 10.3 / 16.2 | 0.00507 / 0.00856 |

Median seconds per node, by route:

- search: route 1.06, retrieve 1.13, answer 2.99, verify 0.00
- out_of_scope: route 0.94, out_of_scope 0.00
- agent: route 1.08, agent 9.22
- clarify: route 1.33, clarify 0.00

LLM calls: 768, failed 5; embedding and rerank calls failed: 0; hedged embedding requests cancelled because the other provider answered first: 26 (in the request's thread). Every call the ledger records as not completed, by stage: {'2-reference qwen/qwen3-embedding-8b: cancelled': 25, '3-assistant qwen/qwen3-embedding-8b: cancelled': 26, '3-assistant deepseek/deepseek-v4.1-flash: LengthFinishReasonError': 5, '4-judge z-ai/glm-5.3-flash: LengthFinishReasonError': 4}. `cancelled` is the losing leg of a hedged query embedding, not a failure; `LengthFinishReasonError` of the system LLM is the answer call of the search route reaching its 1024-token cap (in 4 of the 5 the provider counted 773 to 1024 of them as reasoning tokens although reasoning is off; read from `assistant.log`), after which the reply keeps the results and drops the picks (`assistant_answer_failed`); of the primary judge, a judgment retried in a later pass.

## Failures

- turns without reply: none
- turns retried: none
- notices: {'assistant_answer_failed': 5}
- degraded searches: none
- searches not reranked: none
- router errors: none
- answer errors: 5
- agent direct errors: none
- P not reranked or degraded: none
- P retried: none

## Judgments

Needs with measured pairs: 136.

| Judge | Measured pairs | From V14 | Judged here | Unjudged |
|---|---|---|---|---|
| z-ai/glm-5.3-flash | 1826 | 1220 | 606 | 0 |
| minimax/minimax-m3.1-flash | 1826 | 1220 | 606 | 0 |

- minimax/minimax-m3.1-flash: 606 pairs to judge, 606 judged in the run, 0 failed calls, 0 left, 2026-11-12T15:56:45
- z-ai/glm-5.3-flash: 606 pairs to judge, 606 judged in the run, 4 failed calls, 0 left, 2026-11-12T15:57:21

## Cost and time

| Stage | Calls | Calls without cost | Failed | USD | Wall-clock s |
|---|---|---|---|---|---|
| 1-queries | 14 | 0 | 0 | 0.0428 | n/a |
| 2-reference | 357 | 25 | 25 | 0.2824 | 126 |
| 3-assistant | 1331 | 31 | 31 | 0.9655 | 577 |
| 4-judge | 1216 | 4 | 4 | 0.1072 | 124 |

| Model | Calls | USD |
|---|---|---|
| deepseek/deepseek-v4.1-flash | 768 | 0.3488 |
| google/gemini-3.8-flash | 14 | 0.0428 |
| minimax/minimax-m3.1-flash | 606 | 0.0579 |
| mistralai/mistral-small-2603 | 14 | 0.0009 |
| qwen/qwen3-embedding-8b | 378 | 0.0001 |
| qwen/qwen3-reranker-8b | 528 | 0.8981 |
| z-ai/glm-5.3-flash | 610 | 0.0494 |

Stage wall-clock (seconds, every run of the stage summed): {'1-sets': 55.599999999999994, '2-reference': 125.6, '3-assistant': 576.8000000000001, '4-judge': 123.8}.

## Deviations from the protocol

- Simulated user (set E): the protocol asks for an LLM of another vendor than the system LLM; `mistralai/mistral-small-2603` (Mistral, a vendor used nowhere else in the evaluation) was used at temperature 0.0, with the prompt in `simulated_user.py` (it sees the hidden need, the conversation so far and the latest question with its options, never a result). An empty answer would have been replaced by the chosen option or the first option (recorded per call).
- Set E checks. The protocol names the generator and the size of the requests but not the checks of "does not give its need away"; they were fixed here as: at most one content word shared with the original B query, at most 3 words (8 characters in Chinese and Japanese), not the original query, and for known-item needs no title word and not the author's surname. After the first generation pass, and before any system run, the shared-word rule was widened to count inflected forms (same first five letters), because `славянская мифология` passed for a need about Slavic mythology; the generator calls were reused from the cache and only the new replacements were requested. The checks of that first pass are kept in `sets/checks_before_prefix_rule.json`.
- V14's language check of queries reads the elided French article (`d'`) as Catalan; three correct French E requests (`récits d'aventure`, `roman d'anticipation`, `livre d'histoire`) were therefore sent back and replaced. The replacements are also French and are what was evaluated.
- Set F design, not fixed by the protocol: 7 comparisons of two works, 7 "like X but ..." and 6 questions about one work; the first work of each task drawn with seed 0 from the catalog works in the twelve B languages with a description of at least 300 characters, the second work of a comparison sharing the first one's author (else subject, else language); each request written in the first work's catalog language.
- Definition of the list a reply gives (needed for nDCG@10, not spelled out by the protocol): the search results on the search route, the agent's recommendations in order on the agent route, and no list (nDCG@10 = 0) on the clarify and out-of-scope routes, because the user got no books in that turn. H7 is also shown on the requests the assistant searched, descriptively.
- H9 uses the assistant's turn on each F request whatever route it took (the system as it runs; set F expects the agent route), and compares the precision of its recommendations (grade 2 or 3) with P's precision at the same number of top positions. Requests without any recommendation have no number of positions and are left out of H9; they are listed.
- The two thresholds of section 7 are applied to run 1 of A and B; run 2 is shown next to them.
- Depth. The assistant was called with the endpoint's default of 10 results (vector candidate depth 60), P with 100 results (depth 200) so that its top-100 list could be saved. The 30 candidates the reranker sees were compared at both depths for every P text (database only); the result is in the runs section.
- Deadlines. The reranker calls used the evaluation's long deadlines (30 s per attempt, 3 retries), as in V14, not the deployed 5 s, so that a slow provider does not turn a quality measure into a fallback. Query embeddings used the application's encoder with its deployed settings (DeepInfra, hedge to Nebius after 1 s, DeepInfra again at 2.5 s, 5 s deadline, Zero Data Retention, in-process cache), behind an evaluation cache of vectors by text with up to 3 attempts per text; a P search that came back without reranking or degraded was repeated (up to 3 searches). A turn of the assistant that raised was repeated up to 3 times.
- LLM settings: those `LLMSettings.from_env` gives with the project's `.env` (OpenRouter, `deepseek/deepseek-v4.1-flash`, temperature 0, reasoning off, 1024 output tokens, 180 s timeout). The deployed function's own environment was not read (V17 records `LLM_TIMEOUT_S=15` there); a slow call that the deployment would cut at 15 s completed here.
- No daily budget: the search service was built without the usage budget, so the live site's counters were neither read nor spent. The database connection ran every statement with `transaction_read_only=on`, and its write methods raise.
- Judgments: a pair already judged in V14 (same need id, so the same need text, prompt and judge) was reused; nDCG@10 is computed against the union of V14's pool and the pairs judged here, so its values are not comparable with V14's report, only within this one.
- The agent called directly on the 40-request sample receives the request text, as the assistant would hand it a condensed request.
- Per-turn costs add the LLM calls and the embedding and rerank calls recorded in the request's thread; hedged embedding requests run on other threads and are only in the ledger. The ledger, not the per-turn sums, gives the spend.
- V14's generator class labels its ledger entries with the stage `1-queries`; in this ledger that label is stage 1.
- Before the full runs, each part of stage 3 was run on its first two requests as a smoke test with the same code; those turns are kept and are part of the data (a re-run skips what is stored). A separate check of the composition (one P search and one assistant turn, about 0.004 USD) was made outside the evaluation's data directory, so it is not in the ledger; it falls inside the key-usage window reported on the first page.
- Commit: HEAD moved during the task from `e964682` to `36802ee` (another agent's documentation-only commit); every run of this evaluation was made at `36802ee`, whose application code is that of `e964682`.

## Limits

- Judgments are from two LLM judges only; their agreement with people on this catalog is pending V14's human sheet.
- The user of set E is simulated: it always answers, briefly and coherently with the need, which a person does not guarantee. Any gain measured in H8 is a favourable bound.
- Sets E and F were written by an LLM; F requests name catalog works exactly by their catalog titles.
- The assistant is not deterministic. Its route was repeated once on A and B; quality was measured once (run 1).
- E and F are small (30 and 20 requests); per-language, per-kind and per-route tables are descriptive.
- Latencies were measured from a laptop over the internet, four requests in flight, against a Neon compute shared with the live site, with the evaluation's reranker deadlines; they describe this setting, not the deployed function.
- The judges grade a work against a need; they do not grade the reply's text, the comparison an F request asks for, or whether a clarifying question was useful in itself.

## Files

- Code: `app/evaluation/assistant_evaluation/`; tests: `tests/evaluation/test_assistant_evaluation.py`.
- Data under `data/evaluation/assistant/`: `sets/` (requests, checks, generator calls), `runs/` (`P.jsonl`, `depth_check.jsonl`, `turns.jsonl`, `conversations.jsonl`, `agent_direct.jsonl`, `simulated_user/`), `query_vectors/`, `judgments/` (`pairs.json`, the two caches of this task, `judging_runs.jsonl`), `system.json`, `spend_ledger.jsonl`, `stage_times.jsonl`, `results.json`, `per_request.json`, the stage logs.
- Regenerate this report: `.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`.
