# V11 · Catálogo nuevo, multilingüe y de varias fuentes

- **Fecha:** 2026-11-11
- **Estado:** hecha y validada (actualizada el mismo día: resúmenes de Wikipedia completados)
- **Capa:** dominio (`app/domain/`), ingesta (`app/ingestion/`), infraestructura (`app/infrastructure/external/`, `app/infrastructure/db/`), scripts

## Qué se ha hecho

- Dominio: el puerto `BulkBooksSource` (recorrer una fuente masiva, con su licencia y el recuento de registros rechazados) junto al puerto `ExternalBooksProvider`, que no se toca; el puerto `DuplicateKeyIndex` (consultar y añadir claves de deduplicación); y `get_many`, `iter_all` y `save_many` en `BookCatalogRepository`.
- Modelo de libro multilingüe en `app/domain/value_objects.py`: `BookDescription` (texto, idioma, fuente y licencia), `SourceRecord` (procedencia de cada registro) y `SourceLicense`, y los campos nuevos de `BookMetadata`: identificadores (Wikidata, Open Library, Gutenberg, ISBN-13, DOI) y señal de popularidad. Es retrocompatible: el JSON guardado de un libro antiguo es idéntico byte a byte y `Book.get_searchable_text()` no cambia.
- `app/domain/book_normalization.py`: funciones puras con la biblioteca estándar para códigos de idioma, años, nombres de autor, materias, descripciones (se quita el marcado y se trunca en un final de frase) y claves de comparación.
- Un adaptador por fuente en `app/infrastructure/external/` (Wikidata y Wikipedia, Project Gutenberg, DOAB, volcado de Open Library). Todos leen ficheros locales, de modo que los tests no necesitan red.
- `app/ingestion/catalog_builder.py`: normalización, filtro de calidad, descripción principal, deduplicación entre fuentes y fusión. `app/ingestion/language_detection.py` detecta el idioma de las descripciones; `catalog_report.py` y `catalog_selection.py` generan el informe y el catálogo pequeño.
- `python -m scripts.build_catalog` con cuatro pasos reanudables: `download`, `ingest`, `report` y `select`.

## Por qué

Antes de la tarea el catálogo tenía 177 libros obtenidos de la API de Google Books en 2024 (116 en inglés, 61 en español, el 31 % sin descripción). Con tan pocos libros no se puede medir cómo se comporta la búsqueda híbrida a escala ni atender consultas en varios idiomas. El autor decidió construir la mayor biblioteca que el sistema pueda mantener con sensatez, multilingüe, con las mejores fuentes de noviembre de 2026 y sin coste o con uno muy bajo (decisión 2 de la sección 7 de `docs/notes/2026-11-11-validacion-proyecto-2026.md`).

## Objetivo

Un catálogo grande, reproducible con un comando, donde cada libro conserve de dónde procede cada dato y bajo qué licencia, más una versión pequeña (5 000 obras) que cabe en el repositorio y sirve al perfil local. La colección de 177 libros (`data/catalog.db`) debe quedar intacta.

## Fuentes: qué se eligió y qué se descartó

La investigación se hizo el mismo día con fuentes primarias y llamadas reales.

| Fuente | Decisión | Motivo |
|---|---|---|
| Wikidata + Wikipedia | Elegida | Gratuita; Wikidata es CC0 y da identificadores y popularidad; Wikipedia aporta resúmenes en varios idiomas |
| Project Gutenberg | Elegida | Gratuita; catálogo sin conexión recomendado por la propia fuente |
| DOAB (Directory of Open Access Books) | Elegida | Gratuita; metadatos CC0 |
| Volcado de Open Library | Elegida | Gratuita; la fuente que más obras aporta; pide usar los volcados para el uso masivo |
| Google Books | Descartada como base | Sin clave las llamadas devuelven HTTP 429 con cuota 0 y sus condiciones prohíben construir una base de datos permanente |
| ISBNdb | Descartada | De pago y exige borrar los datos al terminar la suscripción |
| API de Goodreads | Descartada | Cerrada a claves nuevas |
| Conjunto de Goodreads de UCSD | Descartado | "Solo uso académico, no redistribuir" |
| Amazon Reviews 2023 | Descartado | No declara licencia |
| WorldCat | Descartada | Requiere pertenecer a una biblioteca |

## Alternativas de diseño consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Mantener Google Books como base | Cuota 0 sin clave y sus condiciones prohíben una base de datos permanente |
| Un agregador de pago (ISBNdb) | Los datos hay que borrarlos al acabar la suscripción, así que el sistema dejaría de funcionar |
| Consultar las fuentes en vivo en el momento de la búsqueda | Ofrecen búsqueda por palabras sobre metadatos, casi siempre sin descripciones, y no hay búsqueda semántica entre idiomas; la ingesta es lo que permite trabajar sin conexión y hace reproducible la evaluación |
| Un registro por edición | El mismo libro saldría muchas veces en los resultados; la unidad es la obra y las ediciones solo aportan identificadores |
| Reutilizar `ExternalBooksProvider` para las fuentes masivas | Modela "buscar una consulta en una API", no "recorrer un volcado"; un puerto pequeño aparte mantiene ambos con un cometido claro |
| Una biblioteca de detección de idioma | Sería una dependencia nueva para una decisión que las palabras funcionales y el sistema de escritura resuelven con un 98,4 % de aciertos sobre datos reales |
| Mantener todos los registros en memoria para deduplicar | Un índice de claves en SQLite mantiene la memoria en 646 MB para más de un millón de obras |
| Descartar los registros con año implausible o sin autor | Se descarta el año y se conserva el registro; las obras de Wikidata sin autor usan el creador y, si no, "Unknown" |
| Enviar datos de contacto en el User-Agent para obtener el límite mayor de Wikimedia | Es una decisión del autor, no algo que deba hacerse por defecto (resuelto el mismo día; véase la sección de actualización) |

## Cómo se ha validado

- 152 tests nuevos, todos sin conexión y con ficheros de muestra (por ejemplo `tests/external/test_bulk_sources.py`, `tests/ingestion/test_catalog_builder.py`, `tests/domain/test_book_normalization.py`). Suite completa: 1434 superados, 12 omitidos, 0 fallidos.
- Detección de idioma de las descripciones: sobre 12 000 introducciones reales de Wikipedia, 98,4 % correctas, 1,3 % indecisas y 0,3 % erróneas.
- `data/catalog.db` (los 177 libros) es idéntico byte a byte al de antes de la tarea.
- Ejecución completa: la ingesta tardó 531 segundos con un pico de memoria de 646 MB; el informe, 20 s; la selección, 14 s.
- El informe `data/catalog_v2/CATALOG_REPORT.md` se genera desde la base de datos y los contadores por fuente guardados en ella.

## Resultado: el catálogo

El catálogo final tiene **1 147 627 obras**. El fichero ocupa 2,6 GB, no se versiona y se reconstruye con el comando anterior. Filtro de calidad: título, autor y una descripción de al menos 200 caracteres tras quitar el marcado (truncada hacia 1 200) en uno de los doce idiomas configurados.

Registros por fuente (leídos, conservados tras el filtro, obras nuevas, fusionadas en una obra existente):

| Fuente | Leídos | Conservados | Obras nuevas | Fusionados |
|---|---|---|---|---|
| `wikidata` | 156 724 | 74 360 | 74 360 | 0 |
| `gutenberg` | 79 785 | 74 318 | 66 146 | 8 172 |
| `doab` | 102 078 | 76 121 | 69 260 | 6 861 |
| `open_library` | 41 731 960 | 1 015 426 | 937 861 | 77 565 |

34 737 obras proceden de más de una fuente. La descripción principal es la del idioma propio de la obra; si no existe, la inglesa; si no, la primera según el orden de prioridad configurado. Se guarda una descripción por idioma, con preferencia por la escrita por personas sobre la generada automáticamente y después por Wikipedia, DOAB, Open Library y Gutenberg.

Idioma de la obra (los cinco primeros): inglés 975 897 (85,0 %), francés 70 626 (6,2 %), alemán 33 193 (2,9 %), español 23 214 (2,0 %), neerlandés 15 866 (1,4 %). El idioma lo declara una fuente en 190 704 obras y se toma de la descripción principal en 956 923.

Descripciones: 25 054 obras (2,2 %) tienen descripciones en más de un idioma. Por fuente: Open Library 944 249, Wikipedia 131 818, DOAB 69 401, Gutenberg 64 107. Longitud de la descripción principal: mínimo 200, mediana 710, máximo 1 200 caracteres.

Cobertura de campos: año 21,9 %, materias 89,8 %, ISBN-13 5,7 %, identificador de Wikidata 6,5 %, de Open Library 85,5 %, de Gutenberg 6,1 %, DOI 5,3 %.

## El catálogo pequeño

`python -m scripts.build_catalog select --size 5000 --out data/catalog_small/catalog.db` elige 5 000 obras de forma determinista, para el repositorio y el perfil local. Cuotas por idioma: inglés 2 000, español 1 250 y 175 cada uno para francés, alemán, italiano, portugués, catalán, neerlandés, ruso, chino, japonés y árabe.

- Puntuación: número de ediciones de Wikipedia + descargas de Gutenberg / 1 000.
- Dentro de cada idioma van primero las obras cuyo idioma declara una fuente y después las inferidas. Esta regla se añadió porque obras inglesas descritas solo en español habían caído en la cuota española.
- Fuentes: 4 648 de Wikidata, 324 de Gutenberg y 28 de DOAB; 3 870 tienen descripciones en más de un idioma.
- Fichero: 36 MB. La regla y las cuotas quedan en `data/catalog_small/selection.json`.

## Licencias y atribución

Verificadas el 2026-11-11 y listadas por fuente en el informe del catálogo.

| Fuente | Licencia declarada | Atribución |
|---|---|---|
| Wikidata | CC0 | No necesaria |
| Texto de Wikipedia | CC BY-SA (3.0 vía DBpedia, 4.0 vía API) | Se conserva la URL del artículo en cada registro; la compartición igual alcanza al texto |
| DOAB | Metadatos CC0 ("All metadata is openly available under a CC0 license") | No necesaria; se guarda la licencia propia de cada libro |
| Project Gutenberg | No declara licencia explícita para los metadatos; pide usar los catálogos sin conexión y no rastrear la web | Project Gutenberg, con el enlace al libro |
| Open Library | "Does not assert any new copyright"; pide usar los volcados para el uso masivo | Open Library, con el enlace a la obra |

Descripciones por licencia: Open Library (sin licencia explícita) 944 249, CC BY-SA 3.0 118 527, CC0 69 401, Gutenberg (sin licencia explícita) 64 107, CC BY-SA 4.0 13 291. Antes de publicar el catálogo grande conviene que el autor revise las condiciones de redistribución de las dos fuentes sin licencia explícita.

Nota: la API de Wikipedia descargó 16 551 introducciones, pero en el catálogo final figuran 13 291 descripciones con licencia CC BY-SA 4.0. La diferencia (3 260) no se ha desglosado; es compatible con introducciones que no superaron el filtro de calidad o que perdieron frente a otra descripción del mismo idioma, pero no se ha comprobado. Tras completar las introducciones (resuelto el mismo día; véase la sección de actualización), el catálogo sin Open Library contiene 82 186 descripciones CC BY-SA 4.0 y 118 527 CC BY-SA 3.0; la diferencia anterior no se investigó más.

## Actualización: resúmenes de Wikipedia completados y catálogo sin Open Library

### Qué ocurrió

El 2026-11-11 el autor aprobó identificar al cliente ante Wikimedia. El User-Agent llevaba solo la URL pública del repositorio (`https://github.com/luisgius/AI-Book-recommender`), sin nombre ni correo electrónico. Con ella el límite es de 200 peticiones por minuto; la ejecución usó 150 por minuto, hasta 20 títulos por petición, y duró unos 56 minutos en dos tandas (la primera la cortó un tiempo máximo de 30 minutos de la herramienta que la lanzó y la segunda se reanudó desde disco). No hubo ningún HTTP 429 ni 403. Se obtuvieron 99 265 introducciones y ya no falta ningún par (idioma, artículo).

Introducciones obtenidas en esta ejecución, por idioma:

| Idioma | Introducciones |
|---|---|
| en | 15 724 |
| fr | 17 446 |
| it | 12 087 |
| de | 11 889 |
| ar | 8 973 |
| ja | 8 440 |
| ru | 7 972 |
| zh | 4 226 |
| pt | 4 290 |
| nl | 3 579 |
| es | 2 448 |
| ca | 2 191 |

### Segundo catálogo, sin Open Library

Se construyó `data/catalog_core/catalog.db` (no versionado, 808 MB) con Wikidata, Gutenberg y DOAB. Tiene **254 909 obras**. Su informe es `data/catalog_core/CATALOG_REPORT.md`.

| Fuente | Leídos | Conservados | Obras nuevas | Fusionados |
|---|---|---|---|---|
| `wikidata` | 156 724 | 120 143 | 120 143 | 0 |
| `gutenberg` | 79 785 | 74 318 | 65 508 | 8 810 |
| `doab` | 102 078 | 76 121 | 69 258 | 6 863 |

4 066 obras proceden de más de una fuente. Los registros de Wikidata rechazados bajan de 82 364 a: 36 513 por descripción corta, 67 sin descripción y 1 sin autor.

Idioma de la obra (los cinco primeros): inglés 165 664 (65,0 %), alemán 20 954 (8,2 %), francés 20 384 (8,0 %), español 7 231 (2,8 %), italiano 7 019 (2,8 %). El idioma lo declara una fuente en 219 946 obras y se toma de la descripción principal en 34 963.

Descripciones: 33 626 obras (13,2 %) las tienen en más de un idioma. Resúmenes generados automáticamente por Gutenberg como descripción principal: 65 748. Longitud de la descripción principal: mínimo 200, mediana 809, máximo 1 200 caracteres.

| Idioma | Obras con descripción en él | Obras cuya descripción principal está en él |
|---|---|---|
| en | 194 047 | 188 359 |
| es | 17 542 | 7 416 |
| fr | 27 564 | 16 552 |
| de | 24 276 | 16 114 |
| it | 17 862 | 6 781 |
| pt | 10 071 | 3 295 |
| ca | 4 693 | 999 |
| nl | 6 272 | 2 138 |
| ru | 13 581 | 4 934 |
| zh | 3 458 | 840 |
| ja | 5 181 | 2 636 |
| ar | 11 815 | 4 845 |

Cobertura de campos: año 55,1 %, materias 70,1 %, ISBN-13 25,8 %, identificador de Wikidata 47,1 %, de Open Library 11,7 %, de Gutenberg 27,3 %, DOI 23,9 %. Descripciones por fuente: Wikipedia 200 713, DOAB 69 399, Gutenberg 66 250. Descripciones por licencia: CC BY-SA 3.0 118 527, CC BY-SA 4.0 82 186, CC0 69 399, Gutenberg (sin licencia explícita) 66 250.

### Comparación de los dos catálogos

| Magnitud | Con Open Library | Sin Open Library |
|---|---|---|
| Obras | 1 147 627 | 254 909 |
| Idioma de la obra: inglés | 85,0 % | 65,0 % |
| Obras en español | 23 214 | 7 231 |
| Obras con año de publicación | 21,9 % | 55,1 % |
| Obras con descripciones en más de un idioma | 2,2 % | 13,2 % |
| Tamaño en disco | 2,6 GB | 808 MB |

### El nuevo catálogo pequeño

`data/catalog_small/catalog.next.db` (5 000 obras, 39 MB) mantiene las mismas cuotas por idioma. Fuentes: Wikidata 4 805 y Gutenberg 195; ya no se selecciona ninguna obra de DOAB (la selección anterior tenía 28). 4 445 obras tienen descripciones en más de un idioma (antes 3 870). 4 668 de las 5 000 obras coinciden con la selección anterior. Se escribe junto al fichero anterior, que se conserva hasta que la tarea de despliegue cambie a este.

### Cambios de código de esta actualización

- La fuente de Open Library se considera disponible si existe el volcado de obras o su fichero de candidatos en caché.
- Sin el volcado de autores, ahora se detiene con un error claro en lugar de descartar en silencio todas las obras: el fichero de candidatos guarda solo claves de autor, de modo que los nombres no se pueden resolver sin ese volcado.
- El paso `select` ya no sobrescribe el resumen de selección de otro catálogo.
- Tres tests nuevos en `tests/external/test_bulk_sources.py`.

## Límites que debe conocer el tribunal

1. **Wikidata no se pudo consultar por el servicio oficial.** Contar las obras tardó 59,7 s, las consultas paginadas agotaron el tiempo y una devolvió 502. Se usó QLever (Universidad de Friburgo; volcado de Wikidata del 2026-08-10 con actualizaciones en vivo), que no publica condiciones de uso.
2. **Wikimedia limita a 10 peticiones por minuto** a los clientes cuyo User-Agent no lleva datos de contacto (200 con ellos). El proyecto no envía datos personales en las peticiones, así que la mayoría de las introducciones salen de los resúmenes de DBpedia: 118 527 descripciones, de revisiones de Wikipedia de hacia noviembre de 2022, CC BY-SA 3.0, cada una con la URL del artículo. La API en vivo añadió 16 551 a unas 9 peticiones por minuto (primero las obras en español sin descripción, 7 171, completas; después 9 380 en inglés). El informe del catálogo contabiliza 13 291 descripciones CC BY-SA 4.0; esta nota no explica la diferencia con las 16 551. Quedan sin descripción y fuera del catálogo 67 532 obras de Wikidata, y faltan 99 265 pares (idioma, artículo) (resuelto el mismo día; véase la sección de actualización).
3. **Open Library no declara idioma** en las obras. En 956 923 el idioma se infiere de la descripción principal, con el error que eso conlleva (0,3 % sobre la muestra de Wikipedia, no medido sobre Open Library).
4. **Sesgo de idioma.** El catálogo es un 85,0 % inglés; el español son 23 214 obras (2,0 %). El catálogo pequeño corrige la mezcla con cuotas, no el grande.
5. **63 677 descripciones principales son resúmenes que Project Gutenberg genera automáticamente**; cada registro lo indica. Los registros de Gutenberg no tienen año.
6. **Solo el 21,9 % de las obras tiene año**, de modo que el filtro por año deja fuera a la mayoría.
7. **Texto de 2022.** No se ha verificado cuán actual es el texto de DBpedia ni las condiciones de QLever.
8. **Obras con varias descripciones.** Solo 25 054 obras tienen más de una; las descripciones en idiomas secundarios aún no forman parte del texto de búsqueda.
9. **Memoria.** Un `Book` con todas sus descripciones ocupa unos 10,5 KB, de modo que el catálogo completo no cabe en la memoria de los adaptadores de búsqueda actuales; `get_many` permite rellenar los resultados desde la base de datos. Es trabajo de las tareas siguientes.
10. **Datos locales.** Tras la ejecución se borraron del equipo del autor, a petición suya, los dos ficheros brutos del volcado de Open Library (4,6 GB); se conserva el fichero de candidatos filtrados (0,4 GB). Reingerir Open Library desde cero exige volver a descargar el volcado de obras.

## Trabajo pendiente y cómo completarlo

La compleción de los resúmenes de Wikipedia ya está hecha (véase la sección de actualización). Queda lo siguiente.

- **Decisión abierta del autor: si Open Library forma parte del catálogo que se desplegará.** A favor y en contra, anotados el 2026-11-11: multiplica el catálogo por unas seis, pero sus descripciones son casi todas en inglés y sus obras no declaran idioma ni año. La base de datos en la nube está en un plan gratuito de 1 GB; el mismo día, en la tarea V12, se midió que ese plan admite unas 31 600 obras de este tipo multilingüe, de modo que el catálogo desplegado será una selección de unas 30 000 obras tomadas del catálogo sin Open Library, salvo que el autor pase al plan de pago por uso. El `--sources` de `ingest` permite construir el catálogo con o sin ella.
- **Reingerir Open Library en este equipo** exige volver a descargar los volcados de obras y de autores (unos 4,6 GB), porque ambos se borraron para liberar disco.
- **Sin verificar:** las condiciones de uso de QLever y la antigüedad del texto de DBpedia.
- **Para las tareas de búsqueda:** usar `get_many` en lugar de un diccionario en memoria, y decidir si las descripciones de idiomas secundarios entran en el texto de búsqueda.

## Ficheros

- Dominio: `app/domain/ports.py`, `app/domain/value_objects.py`, `app/domain/book_normalization.py`.
- Ingesta: `app/ingestion/catalog_builder.py`, `catalog_report.py`, `catalog_selection.py`, `language_detection.py`.
- Infraestructura: `app/infrastructure/external/` (`wikidata_wikipedia_source.py`, `gutenberg_source.py`, `doab_source.py`, `openlibrary_dump_source.py`, `bulk_download.py`) y `app/infrastructure/db/` (`sqlite_book_catalog_repository.py`, `sqlite_catalog_build_state.py`).
- Script: `scripts/build_catalog.py`.
- Datos: `data/catalog_v2/CATALOG_REPORT.md` (el catálogo grande no se versiona), `data/catalog_core/CATALOG_REPORT.md` (generado, no versionado), `data/catalog_small/catalog.db`, `data/catalog_small/selection.json`.
- Tests: `tests/external/test_bulk_sources.py` (incluye los tres tests nuevos de la actualización), `tests/ingestion/`, `tests/domain/test_book_normalization.py`, `tests/domain/test_value_objects.py`.
