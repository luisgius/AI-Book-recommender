# V01 · Normalización léxica para BM25

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** infraestructura (`app/infrastructure/search/`)

## Qué se ha hecho

Se ha sustituido el tokenizador de BM25, que era `text.lower().split()`, por un
módulo propio, `text_normalization.py`, que usa solo la biblioteca estándar
(`unicodedata` y `re`). El módulo contiene:

- `fold_accents(text, keep_enye=False)`: elimina diacríticos mediante la
  descomposición NFKD. Por defecto la `ñ` pasa a `n`; con `keep_enye=True` se
  conserva.
- Un conjunto de palabras vacías en español e inglés, almacenado ya sin acentos.
- Dos tokenizadores con nombre, accesibles mediante `get_tokenizer(name)`:
  - `"whitespace"`: el comportamiento anterior, sin cambios.
  - `"normalized"`: minúsculas, plegado de acentos (incluida la `ñ`), separación
    por `\w+` y eliminación de palabras vacías. Si un texto está formado solo
    por palabras vacías, se conservan sus tokens.

`BM25SearchRepository` recibe ahora el parámetro `tokenizer` (por defecto
`"normalized"`). El nombre del tokenizador se guarda dentro del fichero del
índice y se restaura al cargarlo, de modo que documentos y consultas usan
siempre el mismo. Un índice sin ese nombre se carga como `"whitespace"`.

El índice distribuido, `data/indexes/bm25_index.pkl`, no se ha reconstruido. El
comportamiento actual de la búsqueda no cambia hasta que la tarea V06 mida
ambos tokenizadores y decida cuál queda por defecto.

## Por qué

El catálogo es bilingüe: 177 libros, 116 en inglés y 61 en español. El
tokenizador anterior trataba como tokens distintos `programación` y
`programacion`, y también `python,` y `python`. Las consultas en español se
escriben a menudo sin tildes, y las consultas de evaluación se almacenan así.
Además, en consultas cortas palabras como `de`, `en` o `the` pesan lo mismo que
las palabras con contenido. La nota `docs/notes/2026-11-11-validacion-proyecto-2026.md`
identifica este tokenizador como una de las piezas envejecidas del proyecto.

## Objetivo

Que una consulta y un documento coincidan aunque difieran en acentos,
puntuación o mayúsculas, y que las palabras vacías no aporten coincidencias
espurias. Todo ello sin añadir dependencias y manteniendo el comportamiento
anterior disponible como línea base para la ablación de V06.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Mantener el tokenizador por espacios | Falla con acentos y puntuación, como muestra la medición descrita más abajo |
| Stemming con Snowball (PyStemmer) | Añade una dependencia, exige conocer el idioma de cada consulta y la mejora esperada es pequeña; queda como posible ablación posterior |
| Sustituir `rank-bm25` por `bm25s` | Su ventaja es la velocidad, irrelevante con 177 libros; la mejora está en el tokenizador |
| Tokenizadores y listas de palabras vacías de spaCy o NLTK | Dependencias pesadas y descarga de modelos para una función de unas veinte líneas; una lista corta incrustada se puede auditar línea a línea |
| `Unidecode` para plegar acentos | Licencia GPL, y `unicodedata` de la biblioteca estándar es suficiente |
| Conservar la `ñ` como letra distinta | Los usuarios escriben `espana`; la opción existe en `fold_accents(keep_enye=True)` para una ablación futura |

## Cómo se ha validado

**Tests.** Se han añadido 41 tests: 25 en `tests/infrastructure/test_text_normalization.py`
y 16 en `tests/infrastructure/test_bm25_search_repository.py`. La suite
completa tras la tarea da 704 tests correctos y 0 fallos.

**Medición de sanidad sobre el catálogo real.** No es la evaluación formal. Las
consultas en español de la colección de prueba están almacenadas sin tildes, de
modo que 20 de ellas se reescribieron a mano con acentos y se compararon con su
forma sin acentuar. La cifra es cuántos de los 10 primeros resultados coinciden.

| Tokenizador | Coincidencia media (de 10) | Listas idénticas |
|---|---|---|
| `whitespace` | 9,00 | 11 de 20 |
| `whitespace`, solo las 14 consultas cuyo texto cambia | 8,57 | 5 de 14 |
| `normalized` | 10 | todas |

Los peores casos con `whitespace` son q19 y q29 (5 de 10). Con `normalized` el
resultado de 10 sobre 10 es así por construcción, porque ambas formas de la
consulta producen los mismos tokens; la medición confirma la propiedad, no
demuestra una mejora de calidad.

**Efecto secundario observado.** Para q07, q23 y q31 el tokenizador
`normalized` devuelve solo entre 3 y 4 libros con puntuación mayor que cero.
El tokenizador `whitespace` devolvía 10 porque casaba palabras vacías como
`de`, `en` o `y`. La consulta q12 no coincide con ningún libro con ninguno de
los dos tokenizadores.

## Cómo usarlo

```python
from app.infrastructure.search.bm25_search_repository import BM25SearchRepository

repo = BM25SearchRepository()                       # "normalized"
legacy = BM25SearchRepository(tokenizer="whitespace")
repo.build_index(books)
repo.save_index("data/indexes/bm25_index.pkl")      # guarda también el nombre
```

Al llamar a `load_index`, el nombre guardado en el fichero sustituye al del
constructor. `repo.tokenizer_name` indica el tokenizador activo. Un nombre
desconocido provoca `ValueError`.

## Límites y trabajo pendiente

- El efecto formal sobre nDCG@10 no está medido; se mide en V06. La medición de
  sanidad solo comprueba la coherencia entre consultas con y sin tildes.
- `scripts/ingest_and_build.py` construye con el tokenizador por defecto, así que
  la próxima reconstrucción del índice será `normalized`.
- No se tratan como palabras vacías `sin`, `son`, `era` (palabras con contenido
  en inglés), `no` ni `not`.
- Los dígitos se conservan (`xix`, `1984`). `data-science` pasa a dos tokens, y
  `c++` y `c#` quedan reducidos a `c`.
- Una consulta formada solo por palabras vacías conserva sus tokens, pero
  puntúa 0 en todos los libros.
- No hay stemming ni lematización; queda como posible ablación.

## Ficheros

- `app/infrastructure/search/text_normalization.py` (nuevo)
- `app/infrastructure/search/bm25_search_repository.py` (modificado)
- `app/infrastructure/search/__init__.py` (modificado)
- `tests/infrastructure/test_text_normalization.py` (nuevo)
- `tests/infrastructure/test_bm25_search_repository.py` (modificado)
