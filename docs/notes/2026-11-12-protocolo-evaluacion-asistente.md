# Protocolo de la evaluación del asistente y del agente (V22)

Fecha: 2026-11-12. Como el protocolo de V14
(`2026-11-12-protocolo-evaluacion-catalogo.md`), este documento se guarda en
git **antes** de ejecutar nada. Fija qué se mide, con qué datos, qué contrastes
se prueban y qué se hará con cada resultado. Las desviaciones se anotarán en el
informe (`data/evaluation/ASSISTANT_EVALUATION.md`) con su motivo.

## 1. Por qué hace falta

El asistente (V20) y el agente al que delega existen, tienen pruebas unitarias
y una comprobación en vivo de diez turnos, pero no se ha medido cómo de bien
funcionan. El autor pidió medirlo el 2026-11-12 («lets measure so»). Tres cosas
no se saben:

1. Si pasar por el asistente empeora o mejora los resultados de una petición
   que ya era clara, puesto que el asistente reescribe la consulta antes de
   buscar.
2. Si preguntar antes de recomendar mejora los resultados cuando la petición
   es vaga, que es la razón de ser de la pregunta aclaratoria.
3. Si el asistente elige bien la ruta: cuándo pregunta sin necesidad, cuándo
   rechaza una petición válida y cuándo reconoce una que no puede atender.

## 2. Qué se evalúa

- Sistema: el asistente del commit que esté en la rama al ejecutar, sobre el
  catálogo desplegado (50 000 obras, esquema `catalog_deploy`), llamando a los
  adaptadores del perfil nube sin pasar por la URL pública ni gastar el
  presupuesto diario.
- Referencia P: la configuración de recuperación desplegada en ese momento
  (tras aplicar la regla de V14, vectorial más reranker), ejecutada sobre el
  texto de la petición sin LLM.
- Los vectores de consulta se guardan por texto, de modo que una misma
  consulta reescrita no cambie de resultados entre ejecuciones.

## 3. Conjuntos de peticiones

| Conjunto | Peticiones | Origen | Ruta esperada |
|---|---|---|---|
| A y B | 116 | Los estratos A y B de V14, sin cambios | Búsqueda. Las 10 de la categoría «ambiguous» de A se informan aparte. |
| D | 12 | El estrato D de V14 (sin respuesta) | Fuera de alcance |
| E | 30 | Nuevo. Treinta necesidades del estrato B elegidas al azar (semilla 0); de cada una, el generador de V14 escribe una petición vaga de una a tres palabras en el mismo idioma. La necesidad original queda oculta. | Pregunta aclaratoria |
| F | 20 | Nuevo. Peticiones de varios pasos sobre obras que están en el catálogo (comparar dos obras, «algo como X pero...», datos de una obra concreta), escritas por el generador a partir de fichas del catálogo y sin ver resultados del sistema. | Agente |

Usuario simulado para E: un LLM de otro proveedor que el LLM del sistema
recibe la necesidad oculta, la pregunta del asistente y sus opciones, y
responde con una frase corta coherente con la necesidad (elige una opción si
alguna encaja; si no, escribe la suya). No ve resultados. La conversación
sigue hasta que el asistente devuelve libros.

## 4. Juicios

Los de V14 para las necesidades de A, B y D. Los pares nuevos (resultados y
libros recomendados que no estén juzgados, y todo E y F) los juzgan los dos
jueces de V14 con la misma instrucción, en ficheros aparte. En E se juzga
contra la necesidad oculta; en F, contra la petición.

## 5. Medidas

- Ruta: matriz de rutas por conjunto; tasa de preguntas innecesarias en A y B;
  tasa de «fuera de alcance» indebido en A y B; tasa de detección en D; rutas
  forzadas por código y su motivo; coincidencia de la ruta al repetir A y B una
  segunda vez.
- Calidad de la lista: nDCG@10 de los resultados que devuelve el asistente.
- Calidad de las recomendaciones: proporción de libros recomendados con nota 2
  o 3, comparada con la precisión de P en el mismo número de primeros puestos.
- Anclaje: recomendaciones y citas descartadas por el nodo de verificación;
  libros recomendados por el agente que no salieron de sus herramientas.
- Idioma: si la respuesta está en el idioma de la petición, por idioma.
- Coste y tiempo: llamadas al LLM, coste real y latencia por turno y por ruta
  (mediana y p95); llamadas fallidas.

## 6. Contrastes declarados

Familia única, bilateral, con corrección de Holm, con el juez principal y
repetida con el segundo. Una conclusión se sostiene solo si tiene el mismo
signo y es significativa con los dos.

| Id | Contraste | Conjunto | Pregunta |
|---|---|---|---|
| H7 | nDCG@10 del asistente frente a P | A y B | ¿Cambia la calidad al pasar por el asistente una petición clara? |
| H8 | nDCG@10 del asistente con aclaración frente a P sobre la petición vaga | E | ¿Mejora preguntar? |
| H9 | Precisión de las recomendaciones del agente frente a P en los mismos puestos | F | ¿Recomienda el agente mejor que la recuperación sola? |

Descriptivo: en E, además, P sobre la petición original de B como techo de
referencia; número de turnos; en una muestra de 40 peticiones de A y B
(semilla 0), el agente llamado directamente.

## 7. Qué se hará con cada resultado

- Si H7 es negativo y se sostiene, la ruta de búsqueda usará el texto del
  usuario, y no la consulta reescrita, cuando la conversación tenga un solo
  turno. Es una tarea aparte.
- Si H8 no se sostiene, el TFG dirá que la pregunta aclaratoria no mostró
  mejora medible con este método, sin más.
- Antes de presentar el asistente como entrada principal de la página pública
  debe cumplirse: «fuera de alcance» indebido en A y B no superior al 2 %,
  y preguntas innecesarias en las peticiones de B no superiores al 10 %. Si no
  se cumple, se revisan las instrucciones del nodo de ruta y se repite la
  medida de ruta, declarando el cambio.
- H9 no cambia el sistema: informa de cuándo compensa el coste del agente.

## 8. Amenazas a la validez

- El usuario de E es simulado: responde siempre de forma coherente y breve,
  cosa que una persona no garantiza. La mejora medida en H8 es un límite
  favorable.
- Las peticiones de E y F las escribe un LLM.
- Los jueces son LLM; su acuerdo con personas en este catálogo depende de la
  hoja de anotación pendiente.
- El asistente no es determinista; la ruta se repite una vez para medirlo, la
  calidad no.

## 9. Coste y límites

Tope de gasto en OpenRouter: 4 USD. La base de datos de Neon se usa solo para
leer. Si el gasto llega al tope, se para y se informa.
