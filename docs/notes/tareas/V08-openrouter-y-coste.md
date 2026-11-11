# V08 · Proveedor OpenRouter, modelos por defecto y coste por llamada
- **Fecha:** 2026-11-11
- **Estado:** hecha y validada
- **Capa:** infraestructura (`app/infrastructure/llm/`, `app/infrastructure/config/`)

## Qué se ha hecho

- Nuevo proveedor `openrouter` para el LLM del sistema y para el juez de relevancia (`LLM_PROVIDER=openrouter`, `JUDGE_PROVIDER=openrouter`).
  - Usa el paquete `langchain-openai`, ya instalado, contra `https://openrouter.ai/api/v1`. La clave se lee de `OPENROUTER_API_KEY` solo al construir el modelo de chat y no se guarda en la configuración. Si falta, el error nombra la variable.
  - No se envía ninguna cabecera identificativa.
  - Cada petición pide a OpenRouter (`provider.require_parameters`) que enrute solo a endpoints que admitan todos los parámetros enviados, en particular el esquema JSON y el interruptor de razonamiento.
- Modelos por defecto: `qwen/qwen3.8-flash` para el LLM del sistema y `minimax/minimax-m3.1-flash` para el juez, de otra familia de modelos. El valor por defecto del proveedor `openai` pasa de `gpt-4o-mini` a `gpt-6-luna`, sin probar en vivo. El proveedor por defecto del código no cambia (`ollama`); `.env.example` presenta OpenRouter como configuración recomendada, con Ollama y `LLM_PROVIDER=none` como alternativas.
- Salida estructurada (`app/infrastructure/llm/structured_output.py`): el método se fija de forma explícita por clase de cliente, `json_schema` para Ollama y para los clientes compatibles con OpenAI y `function_calling` para Anthropic. Así una actualización de LangChain no cambia en silencio cómo se pide el JSON. Se encontraron y corrigieron dos problemas con modelos reales:
  - La palabra clave `example` del esquema JSON (procedente de `json_schema_extra`) fue rechazada por un proveedor con HTTP 400. Se eliminaron los ejemplos.
  - `qwen/qwen3.8-flash` acepta `maxLength` pero no lo aplica: un campo de texto libre llegó más largo que su límite y falló la validación. Los campos de texto libre se truncan ahora en su límite mediante un validador (`soft_max_length` en `schemas.py`). Los rangos numéricos, las enumeraciones y las longitudes mínimas siguen siendo errores.
- Modo de razonamiento: Qwen y DeepSeek lo aceptan desactivado. MiniMax y Gemini lo rechazan ("Reasoning is mandatory") y se usan con esfuerzo bajo. `LLM_REASONING` admite niveles de esfuerzo (`minimal` a `max`) y `JUDGE_REASONING` es nueva; el juez por defecto funciona con `low`.
- Registro de uso: `LLMUsageRecorder` (`app/infrastructure/llm/usage.py`), un callback de LangChain que `build_chat_model` adjunta al modelo. Registra por llamada la operación (`query_understanding`, `explanation`, `explanation_judge`, `relevance_judge`, `agent`), el modelo, la latencia, los tokens de entrada y salida (`None` si el proveedor no los informa, nunca se estiman) y los errores. `summary()` agrega por operación y estima el coste con `LLM_PRICE_INPUT_PER_MTOK` y `LLM_PRICE_OUTPUT_PER_MTOK` (USD por millón de tokens). No hay precios en el código.

## Por qué

- El proyecto se diseñó en 2024 en torno a `gpt-4o-mini`. OpenAI retira el snapshot `gpt-4o-mini-2024-07-18` el 2027-01-14 (página de deprecaciones de OpenAI, leída el 2026-11-03 para la revisión interna del proyecto).
- El modelo local de Ollama era el único LLM disponible. Es gratuito pero lento (ver resultados) y, al ser el único, el juez de la evaluación era el mismo modelo que el LLM del sistema.
- El proyecto no registraba ni tokens ni latencia por llamada al LLM, aunque el coste por consulta es un argumento central en 2026.
- Decisión del autor el 2026-11-11: usar por ahora modelos alojados baratos a través de OpenRouter, preferir modelos chinos o Gemini Flash, no usar modelos GPT y dejar de usar Ollama por el momento.

## Objetivo

Poder ejecutar el sistema y la evaluación con varias familias de modelos alojados mediante una sola clave, con salida estructurada fiable, y medir tokens, latencia y coste estimado por operación. No se pretende elegir aquí el mejor modelo para cada función.

## Alternativas consideradas y por qué no

| Alternativa | Por qué no |
|---|---|
| Mantener `gpt-4o-mini` como valor por defecto | Se retira. |
| Mantener Ollama local como modelo de trabajo de la evaluación | Es entre 3,7 y 13,4 veces más lento por llamada en las mediciones de abajo (según modelo y operación) y en local solo hay una familia de modelos. |
| Modelos GPT o Claude a través de OpenRouter | Excluidos por la decisión del autor del 2026-11-11. |
| Un SDK y una clave por proveedor (Alibaba, DeepSeek, MiniMax, Google) | OpenRouter expone todos tras un único endpoint compatible con OpenAI y una sola clave, lo que además simplifica la ejecución a un profesor que quiera probar el sistema. |
| Un proxy como LiteLLM | Un servicio y una dependencia más para algo que `langchain-openai` ya hace con una URL base. |
| LangSmith o Langfuse para tokens y latencia | Exigen una cuenta externa o un servidor propio. Un callback en proceso de unas cien líneas no tiene dependencias y el código de evaluación puede leerlo. |
| Tabla de precios fija en el código | Los precios cambian a menudo y difieren entre endpoints del mismo modelo. Se leen del entorno. |
| JSON mode o function calling en todos los clientes | El esquema JSON estricto funcionó en los cuatro modelos y garantiza una salida válida respecto al esquema. |
| Fallar la petición cuando un campo de texto libre supera su longitud | Los modelos no aplican el límite, y es preferible truncar una justificación a que falle una búsqueda. |

## Cómo se ha validado

- **Tests:** la batería pasa de 885 a 982 tests que pasan, 0 fallan, sin red. Los nuevos están en `tests/infrastructure/llm/test_openrouter_provider.py`, `test_usage_recorder.py` y `test_llm_schemas.py`.
- **Verificación en vivo sobre el catálogo real:** `search_with_understanding` con tres consultas y explicaciones para los tres primeros resultados, con cuatro modelos a través de OpenRouter. Las 72 llamadas se interpretaron correctamente, sin fallos y sin explicaciones de respaldo. El esquema JSON estricto funcionó con los siete esquemas de la aplicación en los cuatro modelos.
- **Juez de relevancia** sobre cinco pares: las cuatro configuraciones dieron las mismas notas.
- **Precisión de la estimación de coste** frente al coste que informa OpenRouter: en Qwen, MiniMax y Gemini el coste real fue entre un 9 % y un 23 % menor (los descuentos por caché de prompt no se modelan). En DeepSeek el coste real fue unas 1,8 veces la estimación: su precio publicado es el de uno de 36 endpoints, con precios entre 0,006 y 0,45 USD por millón de tokens de entrada, y el precio real depende del enrutamiento.
- Gasto total de la tarea: unos 0,08 USD.

## Resultados

Resumen del registro de uso con tres consultas. Precios del 2026-11-11 en USD por millón de tokens, entrada/salida: `qwen3.8-flash` 0,15/0,47; `minimax-m3.1-flash` 0,15/0,60; `deepseek-v4.1-flash` 0,051/0,567; `gemini-3.8-flash` 0,825/4,125.

| Modelo | Operación | Llamadas | Tokens entrada | Tokens salida | Latencia mediana (s) | Coste est. (USD) |
|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` | `query_understanding` | 9 | 6819 | 536 | 2,01 | 0,00127 |
| `qwen/qwen3.8-flash` | `explanation` | 9 | 11136 | 3438 | 6,19 | 0,00329 |
| `qwen/qwen3.8-flash` | `relevance_judge` | 5 | 2808 | 192 | 1,99 | 0,00051 |
| `minimax/minimax-m3.1-flash` | `query_understanding` | 9 | 7665 | 531 | 1,27 | 0,00147 |
| `minimax/minimax-m3.1-flash` | `explanation` | 9 | 12078 | 4724 | 1,95 | 0,00465 |
| `minimax/minimax-m3.1-flash` | `relevance_judge` | 5 | 3515 | 207 | 0,69 | 0,00065 |
| `deepseek/deepseek-v4.1-flash` | `query_understanding` | 9 | 7059 | 529 | 0,84 | 0,00066 |
| `deepseek/deepseek-v4.1-flash` | `explanation` | 9 | 12978 | 3450 | 2,57 | 0,00262 |
| `deepseek/deepseek-v4.1-flash` | `relevance_judge` | 5 | 3197 | 195 | 0,88 | 0,00027 |
| `google/gemini-3.8-flash` | `query_understanding` | 9 | 6537 | 1280 | 2,97 | 0,01067 |
| `google/gemini-3.8-flash` | `explanation` | 9 | 9779 | 3672 | 4,49 | 0,02321 |
| `google/gemini-3.8-flash` | `relevance_judge` | 5 | 2633 | 258 | 2,12 | 0,00324 |
| `qwen3.8:27b-mlx` (Ollama, una consulta) | `query_understanding` | 3 | 2242 | 220 | 11,11 | n/d |
| `qwen3.8:27b-mlx` (Ollama, una consulta) | `explanation` | 3 | 2915 | 1015 | 26,22 | n/d |

- **Coste de una búsqueda** con comprensión de la consulta y tres explicaciones con `qwen/qwen3.8-flash`: (0,00127 + 0,00329) / 3 = 0,00456 / 3 = 0,00152 USD, es decir, unas 1,5 milésimas de dólar por búsqueda. Es una estimación a partir de tres consultas y de los precios de esa fecha.
- **Latencia frente a Ollama:** las medianas de Ollama son entre 3,7 y 13,4 veces las de los modelos alojados (11,11 s frente a 0,84-2,97 s en comprensión; 26,22 s frente a 1,95-6,19 s en explicación). Las medidas de Ollama se tomaron mientras otra evaluación compartía el modelo local, así que son una cota superior, y proceden de una sola consulta.
- **Diferencias de calidad observadas en esta muestra pequeña:** MiniMax escribió una explicación en español para una consulta en inglés. Los modelos discreparon en los filtros de idioma y categoría extraídos, y las protecciones de filtros ya existentes los descartaron todos, por lo que no se aplicó ningún filtro.

## Cómo usarlo

```bash
# .env (la clave no se versiona)
LLM_PROVIDER=openrouter
LLM_MODEL=qwen/qwen3.8-flash
OPENROUTER_API_KEY=<clave>
# Estimación de coste (opcional, ambas o ninguna)
LLM_PRICE_INPUT_PER_MTOK=0.15
LLM_PRICE_OUTPUT_PER_MTOK=0.47
# Modelos con razonamiento obligatorio
LLM_REASONING=low
```

Sin LLM, `LLM_PROVIDER=none`. Con Ollama, `LLM_PROVIDER=ollama`. El juez usa `JUDGE_PROVIDER`, `JUDGE_MODEL` y `JUDGE_REASONING`, y por defecto `minimax/minimax-m3.1-flash` con esfuerzo `low`. Las variables están documentadas en `.env.example` y en el docstring de `settings.py`. Para leer el coste: `recorder.summary(TokenPrices.from_env())`.

## Límites y trabajo pendiente

- Los proveedores `openai` y `anthropic` no se han probado en vivo; `langchain-anthropic` no está instalado.
- El bucle del agente de búsqueda y `judge_explanation` no se han ejecutado a través de OpenRouter.
- El registro no captura los tokens leídos de caché ni el coste que informa el proveedor, de ahí la desviación de la estimación respecto al coste real.
- El coste de DeepSeek depende del enrutamiento de OpenRouter y la estimación puede quedar muy por debajo del real.
- Qué modelo es mejor para cada función se decidirá en una comparación aparte sobre las tareas del proyecto (pendiente). La muestra de esta tarea (tres consultas, cinco pares) solo comprueba que todo funciona y mide coste y latencia.
- Los precios y los modelos disponibles son los del 2026-11-11 y pueden cambiar.

## Ficheros

- `app/infrastructure/config/settings.py`: proveedor `openrouter`, modelos por defecto, `JudgeSettings`, `TokenPrices`, razonamiento por niveles de esfuerzo.
- `app/infrastructure/llm/langchain_llm_client.py`: `build_chat_model`, opciones de OpenRouter y adjunto del registro.
- `app/infrastructure/llm/structured_output.py`: método de salida estructurada por clase de cliente.
- `app/infrastructure/llm/usage.py`: `LLMUsageRecorder`, `operation_config`, resumen y coste.
- `app/infrastructure/llm/schemas.py`, `schemas_query.py`, `schemas_judge.py`: sin ejemplos y con truncado de texto libre.
- `app/infrastructure/llm/chains.py`, `graphs/query_understanding.py`, `agents/search_agent.py`, `relevance_judge.py`: etiquetado de operaciones y uso del método de salida estructurada.
- `.env.example`: configuración recomendada y variables nuevas.
- `tests/infrastructure/llm/test_openrouter_provider.py`, `test_usage_recorder.py`, `test_llm_schemas.py`: tests nuevos.
