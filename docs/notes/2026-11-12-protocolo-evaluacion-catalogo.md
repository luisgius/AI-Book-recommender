# Protocolo de la evaluación sobre el catálogo desplegado (V14)

Fecha: 2026-11-12. Este documento se escribe y se guarda en git **antes** de
ejecutar la evaluación. Fija qué se va a medir, con qué datos, qué contrastes
se van a probar y qué decisiones dependen de cada resultado, de modo que los
resultados no puedan elegir las preguntas. Cualquier desviación posterior se
anotará en el informe (`data/evaluation/CATALOG_EVALUATION.md`) con su motivo.

## 1. Por qué hace falta

Todo lo medido hasta hoy (V03 a V16) usa la colección de 177 libros y 44
consultas en inglés y español, sobre el perfil local. El sistema desplegado es
otro perfil (Postgres en Neon, modelos por API) y otro catálogo (50 000 obras
en más de doce idiomas, commit `04569cf`). La paridad entre perfiles se midió
(V12, V13), pero tres conclusiones siguen abiertas:

1. La fusión híbrida no superó a la búsqueda vectorial sola en 177 libros
   (V06, V13). Con 50 000 obras puede cambiar, en cualquier sentido.
2. La ganancia del LLM como recomendador no se confirmó en la segunda
   ejecución (V07).
3. El modelo de embeddings elegido para la nube (`qwen/qwen3-embedding-8b`)
   no es repetible entre llamadas y tiene una cola de latencia larga (V16);
   el 2026-11-12 se observaron además dos búsquedas degradadas por tiempo
   agotado del proveedor. `baai/bge-m3` es la alternativa estable.

## 2. Colección y sistema evaluado

- Colección: el catálogo desplegado, esquema `catalog_deploy` de Neon, 50 000
  obras. No se modifica durante la evaluación.
- Sistema: el perfil nube, llamando a sus adaptadores directamente. No se usa
  la URL pública, para no gastar el presupuesto diario de los visitantes ni
  medir la red del cliente.
- Embedding de la consulta congelado: cada consulta se codifica una sola vez
  con cada modelo y ese vector se guarda y se reutiliza en todos los sistemas.
  Así los contrastes entre sistemas no se mezclan con el ruido del modelo, que
  se estudia aparte (sección 8).

## 3. Consultas

Cuatro estratos. Cada consulta lleva identificador, texto, idioma, tipo,
estrato y una descripción de la necesidad de información en inglés; el juez
valora contra la necesidad, no contra las palabras de la consulta.

| Estrato | Consultas | Contenido |
|---|---|---|
| A. Originales | 44 | Las de `app/evaluation/test_queries.json`, sin cambios (24 en inglés, 20 en español). Su necesidad es el propio texto. |
| B. Nuevas, un idioma | 72 | Doce idiomas con cuota en el catálogo (en, es, fr, de, it, pt, ca, nl, ru, zh, ja, ar), seis por idioma: tres temáticas, dos descriptivas («una novela sobre...») y una de obra conocida descrita sin palabras de su título. |
| C. Paralelas | 40 | Diez necesidades, cada una escrita en español, inglés, francés y alemán. Los juicios se comparten entre las cuatro versiones. |
| D. Sin respuesta | 12 | Peticiones que un catálogo de libros no puede atender (por ejemplo, el horario de un tren). Una consulta de D se mantiene en el estrato solo si ninguna obra del pool recibe nota 2 o 3; si no, se reclasifica y se dice. |

Las consultas de B, C y D las genera un LLM de un proveedor distinto del juez,
del segundo juez, del LLM del sistema y de los modelos de recuperación, a
partir de la lista de materias e idiomas del catálogo y sin ver ningún
resultado del sistema. La instrucción y la salida se guardan. Es una amenaza a
la validez declarada (sección 11): el autor debe revisar la lista.

Unidad estadística: los estratos A y B (116 consultas independientes). C se
informa aparte, porque sus cuatro versiones no son independientes. D solo se
usa en el análisis de la sección 9.

## 4. Sistemas

| Id | Sistema |
|---|---|
| S1 | BM25 en Postgres |
| S2 | Vectorial, `qwen/qwen3-embedding-8b`, índice aproximado (HNSW) |
| S3 | Híbrido: fusión RRF de S1 y S2 |
| S4 | S2 más reranker por API (`qwen/qwen3-reranker-8b`) |
| S5 | S3 más reranker por API. Es la configuración desplegada. |
| S6 | S3 más el LLM del sistema (`deepseek/deepseek-v4.1-flash`) como reordenador de la lista (V07, variante R1). Se ejecuta dos veces. |
| E1 | Vectorial exacto con `qwen/qwen3-embedding-8b` (Postgres sin el índice aproximado) |
| E2 | Vectorial exacto con `baai/bge-m3` sobre las mismas obras y el mismo texto por obra (índice local) |

De cada sistema se guarda la lista de 100 resultados por consulta, para que
las métricas se recalculen sin llamar a ninguna API.

## 5. Juicios de relevancia

- Pool: unión de los diez primeros de S1 a S6 (las dos ejecuciones de S6), E1
  y E2, por necesidad.
- Escala y texto de la instrucción: los de V05 (0 a 3).
- Juez principal: `z-ai/glm-5.3-flash` (kappa 0,765 frente a los juicios
  originales en la colección de 177 libros, V10).
- Segundo juez: `minimax/minimax-m3.1-flash` (kappa 0,739), de otro proveedor
  que el juez principal, el LLM del sistema y los modelos de recuperación.
  Juzga el pool entero. Se informa del acuerdo entre jueces.
- Hoja de anotación humana: 200 pares elegidos al azar con semilla 0,
  estratificados por idioma y por nota del juez principal, sin mostrar las
  notas de los jueces. Un script calcula el acuerdo cuando el autor la
  rellene. Hasta entonces los juicios son solo de LLM, y el informe lo dice
  en su primera página.

## 6. Métricas y estadística

- Principal: nDCG@10. Secundarias: P@10, MRR@10, Recall@100 relativo al pool y
  Judged@10.
- Intervalos por bootstrap sobre consultas y contrastes por prueba de
  aleatorización pareada, con el módulo de V03 y sus semillas.
- Corrección de Holm sobre la familia de la sección 7.
- Una conclusión se da por sostenida solo si tiene el mismo signo y es
  significativa con los dos jueces. Si los jueces discrepan, se informa como
  no concluyente.

## 7. Contrastes declarados

Familia única, bilateral, sobre A y B, con el juez principal; se repite con el
segundo juez.

| Id | Contraste | Pregunta |
|---|---|---|
| H1 | S3 frente a S2 | ¿Aporta la fusión híbrida sobre la vectorial sola? |
| H2 | S5 frente a S3 | ¿Aporta el reranker sobre el híbrido? |
| H3 | S5 frente a S4 | Con reranker, ¿aportan los candidatos de BM25? |
| H4 | S2 frente a S1 | ¿Supera la vectorial a BM25? |
| H5 | S6 frente a S5 | ¿Supera el LLM como reordenador al cross-encoder? Se prueba con la primera ejecución; la segunda es la réplica. |
| H6 | E2 frente a E1 | ¿Difieren los dos modelos de embeddings en calidad? |

Descriptivo, sin contraste: resultados por idioma y por tipo de consulta;
solapamiento de S2 con E1 (coste del índice aproximado); en el estrato C,
nDCG@10 por idioma y solapamiento de los diez primeros entre las cuatro
versiones de cada necesidad.

## 8. Regla de decisión del modelo de embeddings

Para cada modelo se codifican las 168 consultas tres veces, con el tiempo
límite desplegado, y se anotan: latencia mediana y p95, llamadas fallidas o
agotadas, y número de consultas cuya lista de diez primeros cambia entre
repeticiones.

Se propondrá cambiar el modelo desplegado a `baai/bge-m3` si se cumplen las
dos condiciones:

1. H6 no muestra a `qwen/qwen3-embedding-8b` significativamente mejor con
   ninguno de los dos jueces.
2. `baai/bge-m3` es mejor en al menos dos de los tres criterios operativos y
   no es peor en el tercero en más de un 10 %.

En cualquier otro caso se mantiene el actual. El cambio en sí (volver a
calcular 50 000 vectores en Neon) es una tarea aparte.

Regla para la fusión híbrida: el texto del TFG dirá lo que diga H1. La
configuración desplegada mantiene BM25 aunque H1 y H3 no sean significativos,
porque BM25 es el camino de degradación cuando falla el proveedor de
embeddings; se cambiaría solo si S4 superase a S5 de forma significativa.

## 9. Consultas sin respuesta (exploratorio)

Se compara la puntuación del primer resultado del reranker en S5 entre las
consultas con respuesta (A y B con alguna obra de nota 2 o 3) y las del
estrato D: distribución, área bajo la curva y el umbral que mejor las separa.
Es exploratorio: el umbral se elige y se mide con los mismos datos, de modo
que no se afirmará nada sobre su rendimiento fuera de ellos.

## 10. Explicaciones

Cuarenta consultas de A y B elegidas al azar (semilla 0), tres primeros
resultados de S5, explicación generada por el LLM del sistema. Se mide la
precisión de citas antes del guardarraíl y si la explicación está en el
idioma de la consulta, con el procedimiento de V10.

## 11. Amenazas a la validez

- Las consultas nuevas las escribe un LLM; pueden ser más regulares que las
  de usuarios reales.
- Los juicios son de LLM. El acuerdo con personas se midió en otra colección
  y en dos idiomas; en este catálogo queda pendiente de la hoja humana.
- El pool favorece a los sistemas que lo forman. Un sistema futuro no puede
  compararse con estos juicios sin juzgar sus resultados nuevos.
- Los modelos se sirven por API y pueden cambiar; se anota el proveedor y la
  fecha de cada llamada.
- Una sola colección y una sola ejecución de los sistemas deterministas; S6 se
  repite por ser el único con salida variable.

## 12. Coste y límites

Tope de gasto en OpenRouter para toda la tarea: 5 USD. Estimación: unos 5 000
juicios por juez a 0,00007 USD, los vectores de `baai/bge-m3` para 50 000
obras y unas 500 llamadas a LLM. Si el gasto acumulado llega a 5 USD, se para
y se informa. No se añade nada a la base de datos de Neon salvo tablas
temporales pequeñas; los vectores de `baai/bge-m3` se guardan en local
(unos 200 MB, fuera de git).
