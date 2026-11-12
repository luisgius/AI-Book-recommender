# Re-measurement of the assistant after revising the routing instruction (V25)

- Contract: the addendum `docs/notes/2026-11-12-protocolo-evaluacion-asistente-adenda.md`, committed in `0a63751` before the instruction changed, completing the protocol `docs/notes/2026-11-12-protocolo-evaluacion-asistente.md` (`0ccd5f2`).
- first (V22): `data/evaluation/assistant/`, report `data/evaluation/ASSISTANT_EVALUATION.md`, commits recorded per run `36802ee` x514; its files are unchanged.
- second (V25): `data/evaluation/assistant_v25/`, commits recorded per run `a09c517` x342, `36802ee` x166 (the 166 P searches are the first measurement's, copied); dates 2026-11-12T16:19:05 to 2026-11-12T16:30:23.
- **Relevance judgments come from LLM judges only**; the user of set E is simulated; sets E, F and G were written by an LLM.
- Real spend of the second measurement reported by OpenRouter: **1.3038 USD** (cap 2.50 USD); the key's usage counter moved 1.2301 USD between its first and last snapshot.
- Regenerate both reports offline: `.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`.

## The two instructions

- first (V22): prompts `v1.1` at commit `36802ee` (routing instruction sha256 `2e2cf1bf4920`). The `out_of_scope` line:

  > - "out_of_scope": not something a book catalog can help with (weather, maths, code, chit-chat with no book request).

- second (V25): prompts `v1.2` at commit `5984e71` (routing instruction sha256 `b8d80f3f6940`). The `out_of_scope` line:

  > - "out_of_scope": the message asks for something other than books or information about books: a fact about the present moment, a calculation, that a program be written, or conversation with no request. Decide by what the message asks for, not by its topic: any topic can be the subject of a book, so a few words or a topic typed into this book search are a request for books about that topic.

Lines of the routing instruction that differ: 1 (the `out_of_scope` line only); everything else in the prompts, the graph and the code rules is the same.

LLM request fields: first (V22) reasoning `False`, request body not recorded; second (V25) reasoning `False`, request body `{'provider': {'require_parameters': True, 'ignore': ['novita']}, 'reasoning': {'enabled': False}}` (the answer-call defect of the addendum's section 5, below).

## Outcome

| Threshold (addendum, section 4) | Value | Maximum | Outcome |
|---|---|---|---|
| out of scope on the 60 valid G requests | 2/60 = 3.3 % | 2.0 % | **not met** |
| questions on B (run 1) | 0/72 = 0.0 % | 10.0 % | met |

Detection on the 20 out-of-scope G requests (reported, no threshold): 20/20 = 100.0 %.

The required thresholds are **not met**. By the addendum, the instruction is not touched again in this task and the assistant is not presented as the public page's main entry point.

Secondary (A and B; the revised instruction was written knowing the first measurement's errors on these requests, so they are not an independent test):

| Run | Measure | first (V22) | second (V25) | Maximum |
|---|---|---|---|---|
| run1 | out_of_scope_ab | 11/116 = 9.5 % (not met) | 1/116 = 0.9 % (met) | 2.0 % |
| run1 | clarify_b | 1/72 = 1.4 % (met) | 0/72 = 0.0 % (met) | 10.0 % |
| run2 | out_of_scope_ab | 9/116 = 7.8 % (not met) | 1/116 = 0.9 % (met) | 2.0 % |
| run2 | clarify_b | 0/72 = 0.0 % (met) | 0/72 = 0.0 % (met) | 10.0 % |

## Set G (held-out)

Written by `google/gemini-3.8-flash` (temperature 0.7, reasoning {'effort': 'low'}, prompt `v1`, provider Google, 2026-11-12T15:17:20+00:00), after the instruction's commit `5984e71`, from the topic list in `sets/g_topics.json` (also in `set_g.py`); the generator saw only that list and the task's rules, never the routing instruction or a system output. Prompt and raw outputs: `sets/generation/G*.json`; checks: `sets/g_checks.json`.

Checks: 8 problems after the first pass, 11 replacements, 0 problems left; language verdicts {'ok': 56, 'undecided': 24} ("undecided": keywords without function words).

- round 1, `g13`: language: not es: pt 1.00. `viajar a Marruecos` -> `turismo en Marruecos`
- round 1, `g21`: asks for reading: the translation 'Do you have a work that offers effective exercises against back pain?' does not ask for something to read. `Avez-vous un ouvrage qui propose des exercices efficaces contre le mal de dos?` -> `Je cherche un bon livre illustré d'exercices pratiques pour soulager efficacement le mal de dos.`
- round 1, `g34`: asks for reading: the translation 'Do you have good instructions on fermenting vegetables such as kimchi and sauerkraut?' does not ask for something to read. `Haben Sie eine gute Anleitung zum Fermentieren von Gemüse wie Kimchi und Sauerkraut?` -> `Ich möchte ein Buch über das Fermentieren von Gemüse wie Kimchi und Sauerkraut lesen.`
- round 1, `g35`: asks for reading: the translation 'I am looking for a clear text on automotive mechanics and routine maintenance.' does not ask for something to read. `Sto cercando un testo chiaro sulla meccanica automobilistica e la manutenzione ordinaria.` -> `Sto cercando un libro chiaro sulla meccanica automobilistica per imparare la manutenzione di base.`
- round 1, `g45`: asks for reading: the translation 'I would like to borrow a good theoretical and practical introduction to the principles of permaculture.' does not ask for something to read. `J'aimerais emprunter une bonne introduction théorique et pratique aux principes de la permaculture.` -> `Je cherche un bon manuel à lire pour découvrir les bases et principes de la permaculture.`
- round 1, `g47`: language: not it: nl 1.00, it 0.75, pt 0.50. `Vorrei trovare un pratico manuale fai-da-te dedicato alle riparazioni idrauliche domestiche.` -> `Potreste indicarmi un testo introduttivo dedicato alla riparazione dei guasti idraulici negli appartamenti?`
- round 1, `g50`: language: not en: es 0.17, fr 0.17, pt 0.17. `Camino de Santiago walking` -> `St James Way pilgrimage`
- round 1, `g60`: asks for reading: the translation 'I would like a publication specialized in positive reinforcement training and essential puppy care.' does not ask for something to read. `Gostaria de uma publicação especializada em adestramento positivo e cuidados essenciais com filhotes de cachorro.` -> `Quero ler um livro prático sobre técnicas eficazes para adestrar e educar um filhote de cachorro.`
- round 2, `g47`: asks for reading: the translation 'Could you recommend an introductory text dedicated to repairing plumbing faults in apartments?' does not ask for something to read. `Potreste indicarmi un testo introduttivo dedicato alla riparazione dei guasti idraulici negli appartamenti?` -> `Vorrei trovare un pratico manuale fai-da-te da leggere sulle riparazioni idrauliche domestiche.`
- round 2, `g50`: topic: the translation 'St James Way pilgrimage' does not name the topic 'walking the Camino de Santiago'. `St James Way pilgrimage` -> `walking Camino de Santiago`
- round 3, `g50`: language: not en: es 0.17, fr 0.17, pt 0.17. `walking Camino de Santiago` -> `Santiago pilgrimage route walking`

| Valid G requests (60) | Count | Rate |
|---|---|---|
| clarify | 0 | 0.0 % |
| search | 58 | 96.7 % |
| agent | 0 | 0.0 % |
| out_of_scope | 2 | 3.3 % |
| error | 0 | 0.0 % |

| By form | clarify | search | agent | out_of_scope |
|---|---|---|---|---|
| keywords | 0 | 29 | 0 | 1 |
| sentence | 0 | 29 | 0 | 1 |

| By language | clarify | search | agent | out_of_scope |
|---|---|---|---|---|
| de | 0 | 10 | 0 | 0 |
| en | 0 | 10 | 0 | 0 |
| es | 0 | 9 | 0 | 1 |
| fr | 0 | 10 | 0 | 0 |
| it | 0 | 9 | 0 | 1 |
| pt | 0 | 10 | 0 | 0 |

| By field | clarify | search | agent | out_of_scope |
|---|---|---|---|---|
| computing | 0 | 6 | 0 | 0 |
| cooking | 0 | 6 | 0 | 0 |
| economics | 0 | 6 | 0 | 0 |
| gardening | 0 | 6 | 0 | 0 |
| health | 0 | 6 | 0 | 0 |
| home and family | 0 | 5 | 0 | 1 |
| science | 0 | 6 | 0 | 0 |
| sport | 0 | 6 | 0 | 0 |
| trades | 0 | 6 | 0 | 0 |
| travel | 0 | 5 | 0 | 1 |

| Out-of-scope G requests, by kind | clarify | search | agent | out_of_scope |
|---|---|---|---|---|
| calculation | 0 | 0 | 0 | 5 |
| conversation | 0 | 0 | 0 | 5 |
| present_fact | 0 | 0 | 0 | 5 |
| program | 0 | 0 | 0 | 5 |

Valid requests routed out of scope: 2.

| Request | Lang | Field or kind | Text | Route | Router's reason |
|---|---|---|---|---|---|
| g11 | it | home and family | Vorrei un manuale introduttivo per allevare galline in giardino nel modo corretto. | out_of_scope | The request is for a practical guide on raising chickens, which is outside the scope of a book catalog. |
| g25 | es | travel | ruta Islandia coche | out_of_scope | The user's message is about a travel itinerary (Iceland by car), not a request for books or book information, so it is out of scope. |

Valid requests that got a question: 0.

Out-of-scope requests not detected: 0.

All G requests:

| Request | Group | Lang | Field or kind | Form | Text | Route |
|---|---|---|---|---|---|---|
| g01 | valid | es | health | keywords | diabetes tipo 2 | search |
| g02 | valid | en | science | keywords | quantum physics beginners | search |
| g03 | valid | fr | economics | keywords | histoire de la monnaie | search |
| g04 | valid | de | trades | keywords | Holzbearbeitung Schreinerei | search |
| g05 | valid | it | sport | keywords | tattica calcistica | search |
| g06 | valid | pt | sport | keywords | treino de maratona | search |
| g07 | valid | es | sport | sentence | Quisiera encontrar un manual práctico para mejorar mi técnica de natación. | search |
| g08 | valid | en | cooking | sentence | Can you recommend a comprehensive guide on meat smoking and outdoor barbecue? | search |
| g09 | valid | fr | computing | sentence | Je cherche un bon livre sur la cybersécurité adapté aux petites entreprises. | search |
| g10 | valid | de | gardening | sentence | Ich suche einen praktischen Ratgeber über das richtige Kompostieren im eigenen Garten. | search |
| g11 | valid | it | home and family | sentence | Vorrei un manuale introduttivo per allevare galline in giardino nel modo corretto. | out_of_scope |
| g12 | valid | pt | home and family | sentence | Procuro um guia completo passo a passo sobre como planejar um casamento. | search |
| g13 | valid | es | travel | keywords | turismo en Marruecos | search |
| g14 | valid | en | economics | keywords | starting small business | search |
| g15 | valid | fr | economics | keywords | économie comportementale | search |
| g16 | valid | de | home and family | keywords | Katzenverhalten verstehen | search |
| g17 | valid | it | cooking | keywords | pane con lievito madre | search |
| g18 | valid | pt | cooking | keywords | culinária caril tailandês | search |
| g19 | valid | es | trades | sentence | Necesito un libro de costura y confección básica para empezar a diseñar ropa. | search |
| g20 | valid | en | trades | sentence | I am looking for an introductory handbook on basic household electrical wiring. | search |
| g21 | valid | fr | health | sentence | Je cherche un bon livre illustré d'exercices pratiques pour soulager efficacement le mal de dos. | search |
| g22 | valid | de | health | sentence | Gibt es hier ein Fachbuch über gesunde Ernährung während der Schwangerschaft? | search |
| g23 | valid | it | gardening | sentence | Cerco una guida illustrata dedicata alla cura e coltivazione delle piante da appartamento. | search |
| g24 | valid | pt | sport | sentence | Gostaria de encontrar um livro introdutório sobre técnicas fundamentais de escalada em rocha. | search |
| g25 | valid | es | travel | keywords | ruta Islandia coche | out_of_scope |
| g26 | valid | en | science | keywords | volcanoes earthquakes | search |
| g27 | valid | fr | computing | keywords | algorithmes structures données | search |
| g28 | valid | de | sport | keywords | Yoga für Anfänger | search |
| g29 | valid | it | computing | keywords | progetti Raspberry Pi elettronica | search |
| g30 | valid | pt | cooking | keywords | decoração de bolos | search |
| g31 | valid | es | health | sentence | Me gustaría consultar un manual actualizado de primeros auxilios y atención de urgencias. | search |
| g32 | valid | en | home and family | sentence | Could you suggest an inspiring book about decluttering and organizing the home? | search |
| g33 | valid | fr | travel | sentence | Je souhaite lire un guide pratique pour voyager au Japon avec un petit budget. | search |
| g34 | valid | de | cooking | sentence | Ich möchte ein Buch über das Fermentieren von Gemüse wie Kimchi und Sauerkraut lesen. | search |
| g35 | valid | it | trades | sentence | Sto cercando un libro chiaro sulla meccanica automobilistica per imparare la manutenzione di base. | search |
| g36 | valid | pt | gardening | sentence | Preciso de um manual prático para aprender a cultivar e podar árvores bonsai. | search |
| g37 | valid | es | computing | keywords | administración sistemas Linux | search |
| g38 | valid | en | science | keywords | coral reef marine biology | search |
| g39 | valid | fr | gardening | keywords | taille arbres fruitiers | search |
| g40 | valid | de | trades | keywords | Schweißen Grundkurs | search |
| g41 | valid | it | computing | keywords | database relazionali SQL | search |
| g42 | valid | pt | sport | keywords | aberturas de xadrez | search |
| g43 | valid | es | travel | sentence | Busco una guía detallada para mochileros que recorren el sudeste asiático por libre. | search |
| g44 | valid | en | computing | sentence | Please recommend a modern tutorial book for building dynamic websites with JavaScript. | search |
| g45 | valid | fr | gardening | sentence | Je cherche un bon manuel à lire pour découvrir les bases et principes de la permaculture. | search |
| g46 | valid | de | health | sentence | Können Sie mir ein Buch über Schlafstörungen und Wege zur Überwindung von Schlaflosigkeit empfehlen? | search |
| g47 | valid | it | trades | sentence | Vorrei trovare un pratico manuale fai-da-te da leggere sulle riparazioni idrauliche domestiche. | search |
| g48 | valid | pt | travel | sentence | Quero um guia confiável sobre trilhas e caminhadas deslumbrantes nos Alpes. | search |
| g49 | valid | es | science | keywords | agujeros negros astrofísica | search |
| g50 | valid | en | travel | keywords | Santiago pilgrimage route walking | search |
| g51 | valid | fr | economics | keywords | bourse débutants investissement | search |
| g52 | valid | de | science | keywords | Klimawandel Wissenschaft | search |
| g53 | valid | it | science | keywords | genetica umana DNA | search |
| g54 | valid | pt | health | keywords | ansiedade estratégias lidar | search |
| g55 | valid | es | cooking | sentence | Busco un recetario práctico de meal prep con menús vegetarianos semanales equilibrados. | search |
| g56 | valid | en | economics | sentence | I want to read an explanatory text on inflation and the role of central banks. | search |
| g57 | valid | fr | home and family | sentence | Je cherche un guide bienveillant sur l'accompagnement du sommeil chez le jeune enfant. | search |
| g58 | valid | de | gardening | sentence | Ich benötige ein leicht verständliches Buch über das Anlegen und Pflegen eines Gemüsegartens. | search |
| g59 | valid | it | economics | sentence | Desidero trovare un valido manuale di finanza personale incentrato sul risparmio e la gestione del budget. | search |
| g60 | valid | pt | home and family | sentence | Quero ler um livro prático sobre técnicas eficazes para adestrar e educar um filhote de cachorro. | search |
| g-oos01 | out_of_scope | es | program | any | Escribe una función en Python para eliminar duplicados de una lista manteniendo el orden | out_of_scope |
| g-oos02 | out_of_scope | en | calculation | any | Convert 72 degrees Fahrenheit to Celsius | out_of_scope |
| g-oos03 | out_of_scope | fr | present_fact | any | Comment est la circulation sur le périphérique en ce moment? | out_of_scope |
| g-oos04 | out_of_scope | de | conversation | any | Hallo, wie geht es dir heute? | out_of_scope |
| g-oos05 | out_of_scope | it | program | any | Scrivi una query SQL per contare il numero di ordini per ciascun cliente | out_of_scope |
| g-oos06 | out_of_scope | pt | present_fact | any | Qual é a cotação atual do bitcoin agora? | out_of_scope |
| g-oos07 | out_of_scope | es | program | any | Genera una expresión regular para validar una dirección de correo electrónico | out_of_scope |
| g-oos08 | out_of_scope | en | calculation | any | How many days are there between March 3 and August 19? | out_of_scope |
| g-oos09 | out_of_scope | fr | program | any | Écris un script bash qui renomme automatiquement toutes les photos selon leur date | out_of_scope |
| g-oos10 | out_of_scope | de | conversation | any | Was ist deine Lieblingsfarbe? | out_of_scope |
| g-oos11 | out_of_scope | it | conversation | any | Raccontami una barzelletta divertente | out_of_scope |
| g-oos12 | out_of_scope | pt | conversation | any | Estou muito entediado, podemos bater um papo? | out_of_scope |
| g-oos13 | out_of_scope | es | calculation | any | Calcula la cuota mensual de una hipoteca de 200000 euros al 3 por ciento durante 25 años | out_of_scope |
| g-oos14 | out_of_scope | en | calculation | any | What is 15 percent of 240? | out_of_scope |
| g-oos15 | out_of_scope | fr | present_fact | any | Quel est le score en direct de la finale de tennis ce soir? | out_of_scope |
| g-oos16 | out_of_scope | de | present_fact | any | Hat die nächste Postfiliale jetzt im Moment geöffnet? | out_of_scope |
| g-oos17 | out_of_scope | it | program | any | Crea una pagina HTML con un modulo di contatto completo | out_of_scope |
| g-oos18 | out_of_scope | pt | conversation | any | Muito obrigado pela ajuda e até logo! | out_of_scope |
| g-oos19 | out_of_scope | es | present_fact | any | Cuáles son los resultados de la lotería de hoy | out_of_scope |
| g-oos20 | out_of_scope | en | calculation | any | Solve 3x + 7 = 22 | out_of_scope |

## Routes, both measurements

Each cell: first (V22) -> second (V25) (a single number when equal). Run 1:

| Set | clarify | search | agent | out_of_scope | error | total |
|---|---|---|---|---|---|---|
| A ambiguous | 8 -> 4 | 1 -> 6 | 0 | 1 -> 0 | 0 | 10 |
| A other | 0 | 25 -> 33 | 0 | 9 -> 1 | 0 | 34 |
| B | 1 -> 0 | 70 -> 72 | 0 | 1 -> 0 | 0 | 72 |
| D | 0 | 0 | 0 | 12 | 0 | 12 |
| E first turn | 14 -> 8 | 16 -> 21 | 0 | 0 -> 1 | 0 | 30 |
| F | 0 | 1 -> 0 | 19 -> 20 | 0 | 0 | 20 |

Run 2 (A and B only):

| Set | clarify | search | agent | out_of_scope | error | total |
|---|---|---|---|---|---|---|
| A ambiguous | 6 -> 4 | 3 -> 5 | 0 | 1 | 0 | 10 |
| A other | 1 -> 0 | 25 -> 34 | 0 | 8 -> 0 | 0 | 34 |
| B | 0 | 72 | 0 | 0 | 0 | 72 |

| Run | Measure | first (V22) | second (V25) |
|---|---|---|---|
| run1 | clarify_ab | 9/116 = 7.8 % | 4/116 = 3.4 % |
| run1 | clarify_a | 8/44 = 18.2 % | 4/44 = 9.1 % |
| run1 | clarify_b | 1/72 = 1.4 % | 0/72 = 0.0 % |
| run1 | out_of_scope_ab | 11/116 = 9.5 % | 1/116 = 0.9 % |
| run1 | out_of_scope_b | 1/72 = 1.4 % | 0/72 = 0.0 % |
| run1 | agent_ab | 0/116 = 0.0 % | 0/116 = 0.0 % |
| run1 | d_detection | 12/12 = 100.0 % | 12/12 = 100.0 % |
| run2 | clarify_ab | 7/116 = 6.0 % | 4/116 = 3.4 % |
| run2 | clarify_a | 7/44 = 15.9 % | 4/44 = 9.1 % |
| run2 | clarify_b | 0/72 = 0.0 % | 0/72 = 0.0 % |
| run2 | out_of_scope_ab | 9/116 = 7.8 % | 1/116 = 0.9 % |
| run2 | out_of_scope_b | 0/72 = 0.0 % | 0/72 = 0.0 % |
| run2 | agent_ab | 0/116 = 0.0 % | 0/116 = 0.0 % |

- Route stability, first (V22): same route on 106/116 (91.4 %); changes {'search -> clarify': 1, 'out_of_scope -> search': 4, 'search -> out_of_scope': 2, 'clarify -> search': 3}.
- Routes forced by code, first (V22): {'run1': {'E: invalid_clarification -> search': 3}, 'run2': {'A: invalid_clarification -> search': 2}}; E later turns {'turn 2: search': 14}.
- Route stability, second (V25): same route on 114/116 (98.3 %); changes {'out_of_scope -> search': 1, 'search -> out_of_scope': 1}.
- Routes forced by code, second (V25): {'run1': {}, 'run2': {}}; E later turns {'turn 2: search': 8}.

A and B requests that got a question or an out-of-scope reply (second (V25), run 1):

| Request | Set | Request text | Reply |
|---|---|---|---|
| q31 | A | recetas de mariscos y pescados frescos | Solo puedo ayudarte con libros de este catálogo: buscar por tema, autor o época, comparar libros o encontrar libros pare |
| q36 | A | historia | ¿Sobre qué tipo de historia quieres libros? |
| q39 | A | apple | What kind of books about "apple" are you looking for? |
| q41 | A | learning | What kind of learning are you interested in? |
| q43 | A | classic | What kind of classic books are you looking for? |

D requests, second (V25): d-en out_of_scope, d-es out_of_scope, d-fr out_of_scope, d-de out_of_scope, d-it out_of_scope, d-pt out_of_scope, d-ca out_of_scope, d-nl out_of_scope, d-ru out_of_scope, d-zh out_of_scope, d-ja out_of_scope, d-ar out_of_scope

## Contrasts H7-H9, both measurements

Same test as the first report (paired randomisation test, 10,000 sign flips, seed 0, 95 % paired bootstrap interval, Holm over the three contrasts, under each judge). Cell: difference [CI] Holm p.

| Id | Judge | first (V22) system / P | first (V22) difference | second (V25) system / P | second (V25) difference |
|---|---|---|---|---|---|
| H7 | primary `z-ai/glm-5.3-flash` | 0.653 / 0.790 | -0.137 [-0.194, -0.084] p <0.001 | 0.756 / 0.790 | -0.034 [-0.068, -0.006] p 0.145 |
| H7 | second `minimax/minimax-m3.1-flash` | 0.674 / 0.801 | -0.127 [-0.182, -0.076] p <0.001 | 0.768 / 0.802 | -0.034 [-0.068, -0.006] p 0.145 |
| H7 | **outcome** |  | **sustained** |  | **not significant** |
| H8 | primary `z-ai/glm-5.3-flash` | 0.338 / 0.129 | +0.209 [+0.104, +0.326] p 0.002 | 0.206 / 0.128 | +0.078 [+0.008, +0.167] p 0.145 |
| H8 | second `minimax/minimax-m3.1-flash` | 0.351 / 0.153 | +0.198 [+0.095, +0.311] p 0.002 | 0.235 / 0.154 | +0.082 [+0.005, +0.177] p 0.193 |
| H8 | **outcome** |  | **sustained** |  | **not significant** |
| H9 | primary `z-ai/glm-5.3-flash` | 0.925 / 0.795 | +0.130 [+0.025, +0.260] p 0.059 | 0.890 / 0.781 | +0.109 [+0.013, +0.233] p 0.145 |
| H9 | second `minimax/minimax-m3.1-flash` | 0.948 / 0.853 | +0.095 [-0.020, +0.230] p 0.186 | 0.927 / 0.841 | +0.087 [-0.033, +0.230] p 0.282 |
| H9 | **outcome** |  | **not significant** |  | **not significant** |

H7 by group, primary `z-ai/glm-5.3-flash` (assistant / P):

| Group | first (V22) n | first (V22) | second (V25) n | second (V25) |
|---|---|---|---|---|
| A ambiguous | 10 | 0.091 / 0.769 | 10 | 0.462 / 0.769 |
| A other | 34 | 0.562 / 0.775 | 34 | 0.749 / 0.776 |
| B | 72 | 0.774 / 0.800 | 72 | 0.801 / 0.800 |
| route clarify | 9 | 0.000 / 0.761 | 4 | 0.000 / 0.767 |
| route out_of_scope | 11 | 0.000 / 0.825 | 1 | 0.000 / 1.000 |
| route search | 96 | 0.789 / 0.789 | 111 | 0.790 / 0.789 |

- first (V22): on the 96 requests the assistant searched, assistant 0.789, P 0.789; recommendations on A+B: 83 replies, precision 0.898 against P 0.793.
- second (V25): on the 111 requests the assistant searched, assistant 0.790, P 0.789; recommendations on A+B: 92 replies, precision 0.857 against P 0.774.

H7 by group, second `minimax/minimax-m3.1-flash` (assistant / P):

| Group | first (V22) n | first (V22) | second (V25) n | second (V25) |
|---|---|---|---|---|
| A ambiguous | 10 | 0.090 / 0.759 | 10 | 0.408 / 0.759 |
| A other | 34 | 0.603 / 0.792 | 34 | 0.776 / 0.792 |
| B | 72 | 0.788 / 0.811 | 72 | 0.814 / 0.813 |
| route clarify | 9 | 0.000 / 0.751 | 4 | 0.000 / 0.878 |
| route out_of_scope | 11 | 0.000 / 0.734 | 1 | 0.000 / 0.556 |
| route search | 96 | 0.814 / 0.814 | 111 | 0.803 / 0.801 |

- first (V22): on the 96 requests the assistant searched, assistant 0.814, P 0.814; recommendations on A+B: 83 replies, precision 0.942 against P 0.865.
- second (V25): on the 111 requests the assistant searched, assistant 0.803, P 0.801; recommendations on A+B: 92 replies, precision 0.893 against P 0.828.

## Set E: how often it asked, and quality with and without asking

| Judge | Measurement | Asked | All: assistant / P vague | Asked: n: assistant / P vague | Did not ask: n: assistant / P vague | P on the original query |
|---|---|---|---|---|---|---|
| primary `z-ai/glm-5.3-flash` | first (V22) | 14/30 | 0.338 / 0.129 | 14: 0.540 / 0.091 | 16: 0.161 / 0.161 | 0.775 |
| primary `z-ai/glm-5.3-flash` | second (V25) | 8/30 | 0.206 / 0.128 | 8: 0.420 / 0.109 | 22: 0.128 / 0.135 | 0.776 |
| second `minimax/minimax-m3.1-flash` | first (V22) | 14/30 | 0.351 / 0.153 | 14: 0.537 / 0.112 | 16: 0.188 / 0.189 | 0.790 |
| second `minimax/minimax-m3.1-flash` | second (V25) | 8/30 | 0.235 / 0.154 | 8: 0.462 / 0.133 | 22: 0.153 / 0.161 | 0.793 |

- first (V22): questions per conversation {0: 16, 1: 14}; stop {'answered': 30}; final route {'search': 30}.
- second (V25): questions per conversation {0: 22, 1: 8}; stop {'answered': 30}; final route {'search': 29, 'out_of_scope': 1}.

## Set F and the agent

| Judge | Measurement | F routes | F with picks | Named works recommended | F precision by kind (picks / P) | Direct agent: precision / P |
|---|---|---|---|---|---|---|
| primary `z-ai/glm-5.3-flash` | first (V22) | {'agent': 19, 'search': 1} | 20/20 | 20/27 | compare 0.914/0.771 facts 1.000/1.000 like_but 0.871/0.643 | 39/40: 0.795 / 0.698 |
| primary `z-ai/glm-5.3-flash` | second (V25) | {'agent': 20} | 20/20 | 20/27 | compare 0.829/0.781 facts 1.000/1.000 like_but 0.857/0.593 | 38/40: 0.805 / 0.712 |
| second `minimax/minimax-m3.1-flash` | first (V22) | {'agent': 19, 'search': 1} | 20/20 | 20/27 | compare 0.943/0.929 facts 1.000/1.000 like_but 0.907/0.650 | 39/40: 0.781 / 0.753 |
| second `minimax/minimax-m3.1-flash` | second (V25) | {'agent': 20} | 20/20 | 20/27 | compare 0.886/0.952 facts 1.000/1.000 like_but 0.907/0.593 | 38/40: 0.833 / 0.775 |

## Grounding

| Measurement | Set | Replies | No answer output | Every pick dropped | Picks proposed | Kept | Dropped | Reasons |
|---|---|---|---|---|---|---|---|---|
| first (V22) | A | 26 | 0 | 0 | 54 | 54 | 0 | - |
| first (V22) | B | 70 | 1 | 3 | 174 | 167 | 7 | {'no_valid_citation': 7} |
| first (V22) | D | 0 | 0 | 0 | 0 | 0 | 0 | - |
| first (V22) | E | 30 | 0 | 1 | 83 | 81 | 2 | {'no_valid_citation': 2} |
| first (V22) | F | 1 | 0 | 0 | 1 | 1 | 0 | - |
| first (V22) | all | 127 | 1 | 4 | 312 | 303 | 9 | {'no_valid_citation': 9} |
| second (V25) | A | 39 | 0 | 3 | 68 | 63 | 5 | {'no_valid_citation': 5} |
| second (V25) | B | 72 | 0 | 2 | 181 | 175 | 6 | {'no_valid_citation': 6} |
| second (V25) | D | 0 | 0 | 0 | 0 | 0 | 0 | - |
| second (V25) | E | 29 | 1 | 2 | 73 | 71 | 2 | {'no_valid_citation': 2} |
| second (V25) | F | 0 | 0 | 0 | 0 | 0 | 0 | - |
| second (V25) | all | 140 | 1 | 7 | 322 | 309 | 13 | {'no_valid_citation': 13} |

- first (V22), agent route: 19 runs, 52 recommendations kept, 0 recommended works no tool returned (dropped), 0 runs with the reflection step; tool calls {'search_books': 38, 'get_book_details': 41, 'find_similar_books': 2}.
- first (V22), agent called directly: 40 runs, 182 recommendations kept, 0 recommended works no tool returned (dropped), 4 runs with the reflection step; tool calls {'search_books': 97, 'get_book_details': 60}.
- second (V25), agent route: 20 runs, 58 recommendations kept, 1 recommended works no tool returned (dropped), 1 runs with the reflection step; tool calls {'search_books': 48, 'get_book_details': 37, 'find_similar_books': 4}.
- second (V25), agent called directly: 40 runs, 172 recommendations kept, 0 recommended works no tool returned (dropped), 3 runs with the reflection step; tool calls {'search_books': 96, 'get_book_details': 61}.

## Answer calls that reached their token cap (addendum, section 5)

Cause, found before the instruction changed: the request asked for reasoning off with `{"enabled": false}`, which OpenRouter forwards, but one provider of `deepseek/deepseek-v4.1-flash`, Novita, keeps reasoning whatever the request says (also under `{"effort": "none"}`, both together, `{"max_tokens": 0}` and the top-level `reasoning_effort`), and OpenRouter routes some calls to it. Fix, in the client: with reasoning off, `provider.ignore = ["novita"]`. The 1024-token cap is unchanged.

| Measurement | Answer calls at the cap | with reasoning tokens | reasoning tokens | answer errors |
|---|---|---|---|---|
| first (V22) | 5 | 4 | [1024, 1024, 773, 0, 1024] | 5 |
| second (V25) | 1 | 0 | [0] | 1 |

## Latency and cost per route

Wall time of `reply()` from this laptop, four requests in flight; cost reported by OpenRouter for the calls in the request's thread. Set G is shown apart.

| Route | first (V22) turns | first (V22) s median / p95 | first (V22) USD median / p95 | second (V25) turns | second (V25) s median / p95 | second (V25) USD median / p95 |
|---|---|---|---|---|---|---|
| agent | 19 | 10.4 / 16.3 | 0.00516 / 0.00859 | 20 | 11.9 / 34.5 | 0.00497 / 0.01423 |
| clarify | 30 | 1.3 / 2.3 | 0.00015 / 0.00024 | 16 | 1.4 / 1.9 | 0.00013 / 0.00025 |
| out_of_scope | 32 | 0.9 / 1.5 | 0.00008 / 0.00047 | 15 | 1.1 / 1.8 | 0.00011 / 0.00019 |
| search | 227 | 5.4 / 9.5 | 0.00264 / 0.00348 | 251 | 5.1 / 8.4 | 0.00258 / 0.00356 |
| all turns | 308 | 5.1 / 10.3 | 0.00248 / 0.00399 | 302 | 5.0 / 11.6 | 0.00256 / 0.00410 |
| direct agent | 40 | 11.5 / 18.6 | 0.00559 / 0.01138 | 40 | 10.1 / 19.3 | 0.00545 / 0.01168 |

| G route | Turns | s median / p95 | USD median / p95 |
|---|---|---|---|
| out_of_scope | 22 | 1.0 / 2.7 | 0.00009 / 0.00016 |
| search | 58 | 4.8 / 7.2 | 0.00236 / 0.00310 |

- first (V22): LLM calls 768, failed 5; embedding and rerank calls failed 0; ledger errors {'2-reference qwen/qwen3-embedding-8b: cancelled': 25, '3-assistant qwen/qwen3-embedding-8b: cancelled': 26, '3-assistant deepseek/deepseek-v4.1-flash: LengthFinishReasonError': 5, '4-judge z-ai/glm-5.3-flash: LengthFinishReasonError': 4}.
- second (V25): LLM calls 793, failed 1; embedding and rerank calls failed 0; ledger errors {'3-assistant qwen/qwen3-embedding-8b: cancelled': 25, '3-assistant deepseek/deepseek-v4.1-flash: LengthFinishReasonError': 1}.
- second (V25), set G: LLM calls failed 0; turns without reply none.

## Failures

| Failure | first (V22) | second (V25) |
|---|---|---|
| turns_without_reply | none | none |
| turns_retried | none | none |
| notices | {'assistant_answer_failed': 5} | {'assistant_answer_failed': 1} |
| degraded_searches | none | none |
| searches_not_reranked | none | none |
| router_errors | none | none |
| answer_errors | 5 | 1 |
| agent_direct_errors | none | none |
| P_not_reranked_or_degraded | none | none |
| P_retried | none | none |

## What changed

Measures with a direction, run 1; "worse" is applied mechanically (a higher out-of-scope or question rate, a lower detection rate or system mean, more dropped picks or answer errors, a median turn more than 10 % slower).

| Measure | first (V22) | second (V25) | Worse |
|---|---|---|---|
| run 1 out_of_scope_ab | 11/116 = 9.5 % | 1/116 = 0.9 % | no |
| run 1 clarify_ab | 9/116 = 7.8 % | 4/116 = 3.4 % | no |
| run 1 clarify_b | 1/72 = 1.4 % | 0/72 = 0.0 % | no |
| run 1 d_detection | 12/12 = 100.0 % | 12/12 = 100.0 % | no |
| H7 system mean (primary) | 0.653 | 0.756 | no |
| H8 system mean (primary) | 0.338 | 0.206 | **yes** |
| H9 system mean (primary) | 0.925 | 0.890 | **yes** |
| H7 system mean (second) | 0.674 | 0.768 | no |
| H8 system mean (second) | 0.351 | 0.235 | **yes** |
| H9 system mean (second) | 0.948 | 0.927 | **yes** |
| E asked (primary) | 14/30 | 8/30 | no |
| F agent route | 19 | 20 | no |
| search picks dropped by verify | 9/312 | 13/322 | **yes** |
| answer calls without output | 5 | 1 | no |
| search turn seconds, median | 5.4 | 5.1 | no |
| agent turn seconds, median | 10.4 | 11.9 | **yes** |

## Judgments

| Measurement | Judge | Measured pairs | From V14 | From the first measurement | Judged here | Unjudged |
|---|---|---|---|---|---|---|
| first (V22) | z-ai/glm-5.3-flash | 1826 | 1220 | - | 606 | 0 |
| first (V22) | minimax/minimax-m3.1-flash | 1826 | 1220 | - | 606 | 0 |
| second (V25) | z-ai/glm-5.3-flash | 1795 | 1211 | 521 | 63 | 0 |
| second (V25) | minimax/minimax-m3.1-flash | 1795 | 1211 | 521 | 63 | 0 |

## Spend of the second measurement

| Stage | Calls | Calls without cost | Failed | USD | Wall-clock s |
|---|---|---|---|---|---|
| 0-diagnosis | 92 | 0 | 0 | 0.0649 | n/a |
| 1-queries | 12 | 0 | 0 | 0.0628 | n/a |
| 3-assistant | 1570 | 26 | 26 | 1.1660 | 675 |
| 4-judge | 126 | 0 | 0 | 0.0101 | 19 |

| Model | Calls | USD |
|---|---|---|
| deepseek/deepseek-v4.1-flash | 1023 | 0.4677 |
| google/gemini-3.8-flash | 12 | 0.0628 |
| minimax/minimax-m3.1-flash | 63 | 0.0059 |
| mistralai/mistral-small-2603 | 8 | 0.0005 |
| qwen/qwen3-embedding-8b | 178 | 0.0000 |
| qwen/qwen3-reranker-8b | 453 | 0.7626 |
| z-ai/glm-5.3-flash | 63 | 0.0042 |

First measurement, for reference: 1.3980 USD.

## Deviations from the addendum

- Where the reasoning defect came from was found by replaying the five answer calls of V22 that hit their cap (their books read from the database, read only) and then one of them on each OpenRouter provider of the model in turn; the diagnostic calls that answered are in the ledger as stage `0-diagnosis` (calls refused with HTTP 404 or 429 by a provider cost nothing and are not listed). They were made before the revised instruction was committed and do not involve the routing instruction. The key's usage counter is read only around the stages run through the command line, so it does not cover them.
- Reuse. The second measurement copies the request sets A-F, P (no LLM; the retrieval code is the same at both commits), the depth check and the query vectors from the first one, and reads the first measurement's judgment caches; only pairs not judged before were judged. The simulated user's calls were not reused (they depend on the assistant's question), so set E was played again with the same simulated user and prompt.
- Set G is measured by its route only, as the addendum's thresholds need; no P, judgments or quality measure was computed for it. G was run once, not twice.
- Set G checks, not fixed by the addendum: language (V14's rule), form (keywords: at most 4 words and no final punctuation; sentence: at least 5 words), no duplicate within G or with A-F, and, read on the English translation the generator gives with each request: a valid request shares a content word with its topic and a sentence asks for something to read (book, guide, manual, ...); an out-of-scope request mentions no book or reading word and reads as its kind (a time word for present facts, a digit for calculations, a programming word for programs). Languages and forms were assigned to topics in code with seed 0 (every language: five keyword and five sentence requests). Some flags were stricter than needed ("ouvrage", "Anleitung", "testo", "publicação" are not in the reading-word list; `viajar a Marruecos` read as Portuguese); every replacement is listed.
- The required thresholds are those of the addendum's section 4: out of scope on the 60 valid G requests (at most 2%) and questions on B (at most 10%), both on run 1 of the second measurement. A and B rates are shown for both runs as secondary data.
- The evaluation code that runs and reports the second measurement (set G, the re-measurement set-up, this report) is not committed; the system it measures is at the commits recorded per turn, whose application code is that of the two V25 commits.
- V14's generator class labels its ledger entries with the stage `1-queries`; for set G that label is stage 1 too.
- nDCG@10 is computed in each measurement against the union of V14's pool and the pairs judged in that measurement, so P's value on the same request can differ slightly between the two columns when the second measurement judged new works for its need.
- Before the full run, every part of stage 3 was run on its first two requests as a smoke test with the same code; those turns are part of the data.

## Files

- Code: `app/evaluation/assistant_evaluation/` (`set_g.py`, `remeasure.py`, `report_v25.py` and the V25 parts of the other modules); tests: `tests/evaluation/test_assistant_evaluation.py`.
- Data under `data/evaluation/assistant_v25/`: `instructions.json`, `sets/` (requests with G, `g_topics.json`, `g_checks.json`, `generation/`), `runs/`, `query_vectors/`, `judgments/`, `system.json`, `spend_ledger.jsonl`, `stage_times.jsonl`, `results.json`, `per_request.json`, `assistant.log`.
- Regenerate: `.venv/bin/python -m app.evaluation.assistant_evaluation report --offline`.
