# V15 · Servidor MCP de solo lectura sobre el buscador
- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** aplicación (`app/api/mcp/`, `app/main.py`), configuración (`app/infrastructure/config/settings.py`)

## Qué se ha hecho

- Qué es MCP. El Model Context Protocol es la forma común en que los asistentes de IA llaman a herramientas externas. Su revisión vigente es la 2026-07-28 (sin estado), lo gobierna la Agentic AI Foundation (Linux Foundation) desde diciembre de 2025 y Claude, incluido el plan gratuito con un conector, permite añadir un servidor MCP remoto sin inicio de sesión.
- Un punto de acceso `/mcp` en la aplicación FastAPI existente (HTTP en flujo continuo, sin estado, respuestas JSON). Es un adaptador de entrada situado junto a la API REST: llama al mismo `SearchService` y al mismo repositorio de catálogo, con el cableado de `app/api/v1/dependencies.py`. El dominio no cambia.
- Tres herramientas, todas anotadas como de solo lectura, con esquemas tipados de entrada y de salida y errores legibles (por ejemplo, un identificador desconocido):
  - `search_books`: búsqueda híbrida, con reordenación si el servidor tiene un reordenador. Filtros por idioma, categoría e intervalo de años. De 1 a 20 resultados (10 por defecto). La descripción se trunca a 300 caracteres.
  - `get_book`: ficha completa por identificador. Descripción de hasta 8 000 caracteres y, en los registros de varias fuentes, todas las descripciones con su fuente y su licencia.
  - `similar_books`: recomendación de libro a libro por similitud de embeddings. De 1 a 20 resultados (10 por defecto).
- Ninguna herramienta llama al LLM. Solo usan `search_with_fallback(use_explanations=False)` y `find_similar_books`.
- Biblioteca: el SDK oficial de Python `mcp` 2.3.0. Obligó a subir `pydantic` de 2.10.4 a 2.14.3 (y con él `pydantic-core` y `typing-extensions`). FastAPI, Starlette y uvicorn no se movieron. La batería completa pasó tras la actualización y antes de escribir código nuevo.
- Protecciones: validación y límites de entrada, comprobación de `Host` y `Origin`, límite de peticiones por cliente, tope de tamaño de respuesta, cuerpo de petición limitado a 64 KiB e interruptor `MCP_ENABLED`. Detalle en la sección de seguridad.
- `MCPSettings` agrupa la configuración (`MCP_ENABLED`, `MCP_ALLOWED_HOSTS`, `MCP_ALLOWED_ORIGINS`, `MCP_RATE_LIMIT_PER_MINUTE`, `MCP_MAX_RESPONSE_BYTES`), documentada en `.env.example`.
- `scripts/mcp_demo.py`: cliente de demostración que lista las herramientas y llama a cada una una vez, midiendo la latencia en el lado del cliente.

## Por qué

- El autor preguntó cómo podría usarse MCP en el proyecto. La decisión es exponer la búsqueda del sistema como un servidor MCP pequeño y de solo lectura, de modo que cualquier asistente pueda consultar el catálogo, y dejar la canalización propia tal como está.
- La opción corresponde a la variante (b) de la sección 4 de `docs/notes/2026-11-11-propuesta-nube-y-mcp.md`: un módulo montado en la aplicación, una dependencia y ningún coste de LLM en el servidor.

## Objetivo

Permitir que un asistente externo busque libros, consulte una ficha y pida libros parecidos sobre el catálogo del sistema, con un coste de despliegue mínimo, sin gasto de LLM en el servidor y sin superficie de escritura. No se pretende que el servidor genere explicaciones ni que sustituya a la API REST.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Que el LLM del proyecto busque en servidores MCP de terceros en lugar de en el índice propio | Las API de libros en vivo hacen búsqueda por palabra clave sobre metadatos y casi siempre sin descripciones. Un bucle de agente necesita al menos dos turnos de LLM (unos 15 s con el modelo medido en V10). Un estudio de 2026 sitúa el refinamiento de la lista por el propio agente por debajo de un reordenador explícito (nDCG@10 de 43,9 frente a 48,0, arXiv 2601.07711). Y la evaluación con fuentes en vivo no es reproducible (arXiv 2508.06600). |
| Una herramienta que genere explicaciones con el LLM | Un servidor público gastaría el presupuesto de LLM del propietario en cada llamada. El asistente que llama a las herramientas puede explicar a partir de los campos devueltos. |
| FastMCP 4 | Habría obligado a actualizar FastAPI y Starlette. El SDK oficial solo necesitó subir Pydantic. |
| Un servicio MCP separado | Una pieza más que desplegar. Montarlo en la misma aplicación reutiliza el cableado y el mismo servicio de búsqueda. |
| Solo el transporte local `stdio` | No se puede añadir a un asistente alojado. El HTTP en flujo continuo sí. |
| Autenticación con OAuth | Innecesaria para datos públicos de solo lectura, y algunos asistentes manejan con más sencillez los servidores sin autenticación. Queda documentada para cuando eso cambie. |
| Un servidor MCP de base de datos (Supabase, Postgres) para atender a usuarios | El de Supabase es una herramienta de desarrollo que su documentación dice que no debe darse a usuarios finales, y los servidores de referencia de Postgres y SQLite están archivados. |
| Una herramienta de SQL libre | Insegura en un punto de acceso público. Tres herramientas tipadas con entradas acotadas cubren el uso. |

La razón de fondo para no usar FastMCP: requiere Starlette 1.0.1 o posterior y uvicorn 0.35 o posterior, y FastAPI 0.115.6 solo funciona con Starlette anterior a 0.42. El SDK oficial solo exigía Pydantic 2.12 o posterior.

## Cómo se ha validado

- **Tests nuevos:** 38. Treinta y seis en `tests/api/test_mcp_server.py` y dos de extremo a extremo en `tests/integration/test_mcp_endpoint.py`, que arrancan la aplicación en el mismo proceso y hablan con `/mcp` mediante un cliente MCP real, uno por revisión del protocolo (2026-07-28 y 2025-11-25), contra el catálogo real de 177 libros. Batería completa: 1278 pasan, 0 fallan.
- **Ausencia de llamadas al LLM.** Un cliente LLM espía registra cero llamadas en todos los caminos MCP. Una prueba de control demuestra que el mismo espía sí detecta el uso del LLM en el camino de explicaciones de la API REST, de modo que la prueba no pasa en vacío.
- **Comprobación en vivo:** `LLM_PROVIDER=none uvicorn app.main:app --port 8010` y después `python -m scripts.mcp_demo --url http://127.0.0.1:8010/mcp` (el puerto 8000 lo ocupaba un servicio ajeno al proyecto). Sin reordenador configurado en esa ejecución. Latencia medida en el cliente, con el sistema caliente:

| Operación | Latencia |
|---|---|
| Conexión | 14 ms |
| `tools/list` | 2 ms |
| `search_books` | 28 ms |
| `get_book` | 7 ms |
| `similar_books` | 4 ms |
| Llamada con error | 1 ms |

  La primera llamada a `search_books` tras el arranque tardó 2625 ms, porque carga el índice BM25, el índice FAISS y el modelo de embeddings.
- **MCP Inspector 2.14.0 en modo CLI** contra el servidor en marcha, con `tools/list` y con `tools/call` sobre `search_books`. El modo con interfaz web procede de la documentación y no se abrió.

## Las tres herramientas

| Herramienta | Entrada | Salida |
|---|---|---|
| `search_books` | `query` (1 a 500 caracteres), `max_results` (1 a 20, por defecto 10), `language` (código de dos letras), `min_year`, `max_year`, `category` | Lista de libros con identificador, título, autores, año, idioma, categorías, descripción de hasta 300 caracteres, fuente y puntuaciones. Indicador `truncated`. |
| `get_book` | `book_id` (UUID) | Ficha completa: descripción de hasta 8 000 caracteres, ISBN, editorial y, si hay varias fuentes, cada descripción con su fuente y licencia, y la procedencia del registro. |
| `similar_books` | `book_id` (UUID), `max_results` (1 a 20, por defecto 10) | Lista en el mismo formato compacto, con puntuación de similitud, sin incluir el libro de partida. Indicador `truncated`. |

Los esquemas completos están en `app/api/mcp/schemas.py` y las descripciones que lee el modelo, en `app/api/mcp/server.py`.

## Cómo usarlo y cómo enseñarlo al tribunal

Arranque local y demostración con el cliente propio:

```bash
LLM_PROVIDER=none uvicorn app.main:app --port 8010
python -m scripts.mcp_demo --url http://127.0.0.1:8010/mcp
```

Con MCP Inspector en modo CLI (el servidor debe estar en marcha en el puerto indicado):

```bash
npx @modelcontextprotocol/inspector --cli --server-url http://127.0.0.1:8000/mcp --transport http --method tools/list
npx @modelcontextprotocol/inspector --cli --server-url http://127.0.0.1:8000/mcp --transport http --method tools/call --tool-name search_books --tool-args-json '{"query":"machine learning","max_results":2}'
```

La interfaz web del Inspector se abre con el mismo comando sin `--cli` ni `--method`. Eso procede de la documentación y no se ha probado. El Inspector 2.14.0 pide Node 26.12.1 o posterior en la línea 26. La máquina tiene 26.8.1 y funcionó con un aviso: conviene actualizar Node antes de la defensa.

Para añadirlo a Claude cuando el servidor tenga una URL pública (pasos del artículo de ayuda de Claude, comprobados el mismo día):

1. Abrir Customize, Connectors, Add custom connector.
2. Escribir un nombre y `https://<host>/mcp`.
3. Elegir la autenticación "No sign in".

Condiciones: el servidor debe ser accesible desde internet público y `MCP_ALLOWED_HOSTS` debe incluir su nombre de host, porque en otro caso cada petición recibe un 421. Detrás de un proxy TLS hay que arrancar uvicorn con `--proxy-headers`. Los pasos para ChatGPT no se han verificado.

Guion sugerido para el tribunal: mostrar `tools/list` en el Inspector, llamar a `search_books`, copiar un `book_id` a `get_book` y a `similar_books` y, cuando haya URL pública, repetirlo desde un asistente. Conviene lanzar una búsqueda antes de empezar para que el primer uso no pague los 2625 ms de carga.

## Seguridad y límites que debe conocer el tribunal

Protecciones activas:

- Validación de entrada y límites, derivados de las firmas de las herramientas: tipos, longitudes, patrón de idioma y rango de años.
- Comprobación de `Host` y `Origin`. Por defecto solo se responde a peticiones dirigidas a la propia máquina. Un despliegue público define `MCP_ALLOWED_HOSTS`.
- Límite de peticiones por cliente con ventana deslizante en memoria (`MCP_RATE_LIMIT_PER_MINUTE`, 60 por defecto; 0 lo desactiva). Responde 429 con `Retry-After`.
- Tope de tamaño de respuesta (`MCP_MAX_RESPONSE_BYTES`, 40 000 por defecto): los resultados pierden elementos del final y se marca `truncated=true`.
- Cuerpos de petición limitados a 64 KiB por el SDK.
- `MCP_ENABLED=false` elimina el punto de acceso.

No hay autenticación, porque las herramientas son de solo lectura sobre datos públicos del catálogo.

Inyección de instrucciones. Las descripciones de los libros proceden de fuentes externas y algunas son editables por cualquiera (Wikipedia). Se devuelven solo como campos de datos, nunca dentro de las descripciones de las herramientas ni de las instrucciones del servidor, y estas últimas piden al modelo cliente que las trate como datos.

Límites:

- El límite de peticiones cuenta por dirección de cliente y en memoria. Todo el tráfico de Claude llega desde direcciones del proveedor, así que esos usuarios comparten un contador, y cada proceso trabajador cuenta por separado.
- No hay CORS: un cliente que funcione directamente desde el navegador lo necesitaría.
- Lo que el asistente externo diga después de recibir los resultados queda fuera del control del sistema. El servidor solo devuelve libros del catálogo con sus identificadores, y así debe constar en la memoria.
- La especificación MCP cambió de forma sustancial hace tres meses, así que el punto de acceso depende de un SDK joven.
- Un test de extremo a extremo modifica variables globales privadas de `app/api/v1/dependencies.py` y fallará de forma visible si se renombran.

## Trabajo pendiente

- Obtener una URL pública (tarea de despliegue) y hacer después la demostración en vivo con un asistente.
- Subir o compartir el límite de peticiones en un despliegue público.
- Añadir OAuth 2.1 mediante el SDK si el servidor llega a exponer datos privados o herramientas con coste.
- Actualizar Node antes de la defensa si se quiere usar el Inspector sin aviso.

## Ficheros

- `app/api/mcp/server.py`: servidor MCP, descripciones de herramientas, transporte, `MCPEndpoint` y `mount_mcp_server`.
- `app/api/mcp/tools.py`: lógica de las tres herramientas, límites y recorte por tamaño.
- `app/api/mcp/schemas.py`: modelos de salida.
- `app/api/mcp/rate_limit.py`: límite de peticiones de ventana deslizante y su middleware.
- `app/api/mcp/__init__.py`: exportaciones del paquete.
- `app/main.py`: montaje del punto de acceso y ciclo de vida del transporte.
- `app/infrastructure/config/settings.py`: `MCPSettings`.
- `.env.example`: variables `MCP_*` documentadas.
- `scripts/mcp_demo.py`: cliente de demostración.
- `requirements.txt`: `mcp` 2.3.0, `mcp-types` 2.3.0 y `pydantic` 2.14.3.
- `tests/api/test_mcp_server.py`, `tests/integration/test_mcp_endpoint.py`: tests nuevos.
