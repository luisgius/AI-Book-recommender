# Adenda al protocolo de evaluación del asistente (V25)

Fecha: 2026-11-12. Se guarda en git antes de cambiar las instrucciones del
nodo de ruta y antes de repetir ninguna medida. Completa
`2026-11-12-protocolo-evaluacion-asistente.md`, cuya sección 7 preveía este
caso.

## 1. Qué ha pasado

La evaluación V22 (`data/evaluation/ASSISTANT_EVALUATION.md`, commit `7e91bf1`)
dio este resultado en los dos umbrales declarados:

- «Fuera de alcance» indebido en A y B: 11 de 116 (9,5 %; en la segunda
  ejecución, 7,8 %), frente a un máximo del 2 %. **No se cumple.**
- Preguntas innecesarias en B: 1 de 72 (1,4 %), frente a un máximo del 10 %.
  Se cumple.

Las peticiones rechazadas por error son sobre todo de cocina y recetas, más
algunas de informática. La instrucción del nodo de ruta define «fuera de
alcance» con ejemplos de tema («tiempo, matemáticas, código»), y el modelo
toma el tema de la petición por su tipo. La pérdida medida en H7 (−0,137)
viene entera de la ruta: en las 96 peticiones que el asistente buscó, su
resultado es igual al de la referencia.

## 2. Qué se cambia

Solo las instrucciones del nodo de ruta, con un criterio y no con una lista
de casos: una petición está fuera de alcance por lo que pide, no por su tema.
Cualquier tema puede ser el de un libro, de modo que unas palabras sueltas o
un tema en la caja de un buscador de libros se tratan como petición de libros
sobre ese tema. Queda fuera de alcance lo que pide otra cosa que libros o
información sobre libros: un dato del momento, un cálculo, que se escriba un
programa o una conversación sin petición.

No se añaden a la instrucción ejemplos tomados de las peticiones de A, B, D,
E o F. No se cambia el grafo, ni las reglas del código, ni el resto de las
instrucciones, salvo el defecto de la sección 5.

## 3. Qué se vuelve a medir y con qué datos

La instrucción nueva se escribe conociendo los errores de la primera medida,
así que repetir la medida sobre las mismas peticiones no es una prueba
independiente. Por eso se hacen dos cosas, en este orden:

1. Se guarda en git la instrucción nueva. Después, y no antes, se genera un
   conjunto nuevo G que nadie ha visto al escribirla: 60 peticiones válidas
   de temas no literarios (cocina, informática, salud, viajes, economía,
   ciencia, oficios, deporte y otros), en seis idiomas (es, en, fr, de, it,
   pt), la mitad como palabras sueltas y la mitad como frases, y 20 peticiones
   fuera de alcance del tipo descrito en la sección 2. Las escribe el
   generador de V14 a partir de una lista de temas, sin ver ni la instrucción
   ni ningún resultado.
2. Se repite completa la medida de V22 (rutas, H7 a H9, conjuntos A, B, D, E y
   F) con la instrucción nueva, y se mide G.

## 4. Umbrales

Los mismos de la sección 7 del protocolo, y se exigen sobre G, que es el
conjunto independiente:

- «Fuera de alcance» indebido en las 60 válidas de G: no más del 2 % (como
  máximo una).
- Detección en las 20 fuera de alcance de G: se informa, sin umbral.
- Preguntas innecesarias en B: no más del 10 %, como antes.

Sobre A y B se informan los mismos porcentajes como dato secundario, diciendo
que la instrucción se escribió conociéndolos. Si G no cumple, no se vuelve a
tocar la instrucción en esta tarea: se informa, y el asistente no se presenta
como entrada principal de la página pública.

La regla sobre H7 de la sección 7 del protocolo (usar el texto del usuario en
conversaciones de un turno) no se aplica: V22 mostró que la reescritura no
causa la pérdida.

## 5. Un defecto que se corrige a la vez

En 5 de 768 llamadas, la respuesta del nodo que redacta la recomendación
agotó su límite de 1 024 tokens; en 4, el modelo gastó entre 773 y 1 024 en
razonamiento aunque el razonamiento está desactivado. Se corrige en el
cliente (cómo se desactiva el razonamiento para este proveedor, o el límite),
y se informa de cuántas veces ocurre en la repetición.

## 6. Informe y coste

Los resultados de la primera medida no se tocan. La repetición se guarda
aparte y el informe muestra las dos, una junto a otra, con la instrucción de
cada una. Tope de gasto: 2,5 USD.
