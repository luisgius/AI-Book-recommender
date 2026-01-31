# Preparación Entrevista Dawnguard - ML/AI Engineer

> **Fecha**: Mañana
> **Empresa**: Dawnguard (startup seguridad cloud con agentes IA)
> **Posición**: ML/AI Engineer
> **Tu proyecto**: Sistema de recomendación de libros con RAG + Agentes

---

## Tu Pitch del Proyecto (2 minutos - MEMORIZA ESTO)

> "Para mi TFG construí un **sistema de recomendación de libros inteligente** que va más allá de búsqueda tradicional.
>
> El usuario describe en lenguaje natural qué tipo de libro quiere - por ejemplo, *'quiero algo como El Señor de los Anillos pero más oscuro'* - y el sistema encuentra y recomienda libros relevantes **con explicaciones de por qué cada uno encaja**.
>
> Lo interesante es que no es solo RAG básico. Implementé un **agente con LangGraph** que primero decide si la consulta requiere buscar en la base de datos o si puede responder directamente. Por ejemplo, si preguntas '¿quién escribió 1984?', no necesita buscar - pero si pides recomendaciones, activa la búsqueda.
>
> Para el retrieval usé **búsqueda híbrida**: BM25 para keywords exactos más FAISS para semántica, fusionados con RRF. Esto mejoró el **recall@10 en un 20%** comparado con solo búsqueda semántica.
>
> La arquitectura es **hexagonal** - separé dominio, infraestructura y aplicación - lo que me permitió cambiar de proveedor de embeddings sin tocar la lógica de negocio.
>
> Para evaluar usé **LLM-as-judge** porque métricas tradicionales no capturan bien la calidad de recomendaciones. El juez evalúa relevancia, coherencia y si las explicaciones están fundamentadas en los datos."

### Por qué este pitch funciona:
- ✅ Empieza con el **problema** (recomendaciones en lenguaje natural)
- ✅ Menciona **decisión técnica interesante** (router del agente)
- ✅ Da un **número concreto** (20% mejora)
- ✅ Muestra **buenas prácticas** (arquitectura hexagonal, evaluación seria)
- ✅ Dura ~2 minutos hablando normal

---

## PREGUNTAS TÉCNICAS SEGURAS (te las van a hacer)

---

### 1. ¿Qué es un AI Agent?

**Respuesta (~90 segundos):**

> "Un AI agent es un sistema que puede **razonar, planificar y ejecutar acciones de forma autónoma** para lograr un objetivo.
>
> La diferencia clave con un LLM simple es el **loop**:
> - Un LLM es **one-shot**: pregunta → respuesta
> - Un agent es **iterativo**: observa → razona → actúa → observa resultado → repite
>
> El patrón más común es **ReAct** (Reasoning + Acting):
> 1. **Thought**: El agente razona sobre qué hacer
> 2. **Action**: Decide qué herramienta usar
> 3. **Observation**: Recibe el resultado
> 4. **Repeat**: Hasta completar la tarea
>
> En mi proyecto, el agente decide si buscar en la base de datos o responder directamente. Analiza la consulta, elige la acción, observa los resultados del retrieval, y genera una respuesta fundamentada. Es un loop, no un one-shot.
>
> Para seguridad cloud, un agente podría detectar una anomalía, investigar logs, correlacionar eventos, y decidir si escalar o remediar automáticamente."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿Por qué ReAct vs otros patrones?" | "ReAct es más interpretable - puedes ver el razonamiento paso a paso, crítico para debugging. Plan-and-Execute planifica todo antes, pero ReAct se adapta mejor cuando el contexto cambia durante la ejecución." |
| "¿Limitaciones de agentes?" | "Tres: 1) Loops infinitos - necesitas max_iterations, 2) Costo - cada iteración es una llamada al LLM, 3) Latencia - múltiples llamadas = más tiempo." |
| "¿Cómo evitar acciones peligrosas?" | "Guardrails: human-in-the-loop para acciones destructivas, whitelist de herramientas permitidas, validación antes de ejecutar, logging de todo para auditoría." |

**NO digas:**
- ❌ "Es como ChatGPT pero más inteligente" (muy vago)
- ❌ Confundir agent con chatbot o con fine-tuning
- ❌ Olvidar mencionar el loop - es LA diferencia clave

---

### 2. Explica RAG (Retrieval-Augmented Generation)

**Respuesta (~90 segundos):**

> "RAG es una técnica que **combina retrieval con generación** para que el LLM responda basándose en datos externos en lugar de solo su conocimiento de entrenamiento.
>
> El flujo es:
> 1. **Query** → El usuario hace una pregunta
> 2. **Retrieve** → Buscas documentos relevantes en tu base de datos
> 3. **Augment** → Inyectas esos documentos en el contexto del LLM
> 4. **Generate** → El LLM genera una respuesta fundamentada en esos documentos
>
> **¿Por qué reduce alucinaciones?** Porque el LLM tiene los datos reales delante. En vez de inventar, puede citar. Si le pides recomendar un libro y le das la sinopsis real, no puede inventarse el argumento.
>
> **¿Cuándo usarlo?**
> - Datos privados o recientes (después del cutoff del modelo)
> - Cuando necesitas citar fuentes
> - Cuando la precisión es crítica
>
> En mi proyecto, el RAG recibe los libros del retrieval híbrido y genera explicaciones de por qué cada libro encaja con lo que pidió el usuario. Sin RAG, el LLM inventaría libros que no existen.
>
> Para seguridad cloud: un agente RAG podría responder preguntas sobre tu infraestructura buscando en tus propios runbooks, logs, y documentación interna."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿RAG vs fine-tuning?" | "RAG para datos que cambian frecuentemente o son muy específicos. Fine-tuning para cambiar el estilo o comportamiento del modelo. RAG es más barato y no requiere re-entrenar." |
| "¿Y si el retrieval devuelve 0 resultados?" | "Tres opciones: 1) El agente responde que no tiene información, 2) Fallback a conocimiento general del LLM con disclaimer, 3) Pide al usuario reformular. En mi proyecto uso la opción 1 para evitar alucinaciones." |
| "¿Cómo mides si RAG funciona bien?" | "Dos dimensiones: retrieval (recall@k, precision@k) y generación (LLM-as-judge para relevancia y groundedness). Si el retrieval falla, la generación no puede salvarlo." |

**NO digas:**
- ❌ "RAG elimina las alucinaciones" (las reduce, no las elimina)
- ❌ Olvidar que tiene DOS partes: retrieval Y generation
- ❌ No mencionar que depende de la calidad del retrieval

---

### 3. ¿Qué es Cosine Similarity?

**Respuesta (~60 segundos):**

> "Cosine similarity mide el **ángulo entre dos vectores**, ignorando su magnitud. La fórmula es:
>
> ```
> cos(θ) = (A · B) / (||A|| × ||B||)
> ```
>
> El rango es **-1 a 1**:
> - **1** = vectores idénticos en dirección (mismo significado)
> - **0** = ortogonales (sin relación)
> - **-1** = opuestos (significado contrario)
>
> **¿Por qué para embeddings?** Porque los embeddings capturan significado semántico en la dirección del vector, no en su magnitud. Dos textos pueden tener embeddings de diferente longitud pero apuntar en la misma dirección si significan lo mismo.
>
> En mi proyecto, cuando el usuario pide 'libros de fantasía épica', convierto eso a un embedding y busco los libros cuyo embedding tenga mayor cosine similarity. Un libro de Tolkien tendrá similarity alta aunque las palabras sean diferentes.
>
> Es más estable que distancia euclidiana para espacios de alta dimensión porque normaliza por magnitud."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿Por qué no distancia euclidiana?" | "Euclidiana depende de magnitud. Dos documentos largos podrían parecer más similares solo por tener embeddings más grandes. Cosine normaliza eso." |
| "¿Cuándo usarías euclidiana?" | "Cuando la magnitud importa, por ejemplo si el tamaño del vector representa intensidad o cantidad, no solo dirección semántica." |
| "¿Cómo manejas que sea -1 a 1?" | "En la práctica, embeddings de texto casi nunca dan negativos porque el espacio está diseñado así. Normalmente trabajo con 0 a 1." |

**NO digas:**
- ❌ "Es como distancia pero al revés" (impreciso)
- ❌ Olvidar mencionar que ignora magnitud
- ❌ Confundir con dot product (que no normaliza)

---

### 4. ¿Cómo hacer que un LLM devuelva JSON válido?

**Respuesta (~90 segundos):**

> "Hay varias capas de defensa, de más simple a más robusto:
>
> **1. System prompt + few-shot**
> ```
> 'Responde SOLO con JSON válido. Formato: {"title": "...", "reason": "..."}'
> ```
> Funciona ~80% del tiempo pero no es confiable.
>
> **2. JSON mode nativo**
> OpenAI y Anthropic tienen `response_format: json_object`. Garantiza JSON válido pero no garantiza el schema.
>
> **3. Structured outputs con schema**
> Defines el schema exacto y el modelo está forzado a seguirlo. OpenAI lo soporta nativamente.
>
> **4. Pydantic + validación**
> Es lo que uso en mi proyecto con LangChain:
> ```python
> class BookRecommendation(BaseModel):
>     title: str
>     author: str
>     reason: str = Field(description="Por qué este libro encaja")
> ```
> LangChain inyecta el schema en el prompt y parsea la respuesta. Si falla el parsing, puedes hacer retry.
>
> **5. Fallback y retry**
> Si falla la validación, reintento con el error en el prompt: 'Tu respuesta no fue JSON válido. Error: X. Intenta de nuevo.'
>
> En producción uso capas 3+4+5 juntas. Para seguridad cloud esto es crítico - si el agente devuelve una acción mal formada, no puede ejecutarla."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿Qué pasa si sigue fallando después del retry?" | "Logging del error, fallback a respuesta por defecto o escalado a humano. Nunca ejecutar acciones con datos mal parseados." |
| "¿Por qué Pydantic específicamente?" | "Validación de tipos en runtime, mensajes de error claros, y se integra nativo con LangChain y FastAPI. Es el estándar en Python para esto." |
| "¿Y para respuestas muy largas?" | "El JSON mode puede fallar con respuestas largas que se truncan. Solución: limitar el tamaño esperado o usar streaming con validación incremental." |

**NO digas:**
- ❌ "Solo pon 'responde en JSON' en el prompt" (muy naive)
- ❌ Asumir que JSON mode = schema correcto
- ❌ Olvidar mencionar retry/fallback

---

## PREGUNTAS TÉCNICAS PROBABLES

---

### 5. ¿Qué es Hybrid Search?

**Respuesta (~90 segundos):**

> "Hybrid search **combina búsqueda léxica (keywords) con búsqueda semántica (embeddings)** para obtener lo mejor de ambos mundos.
>
> **Búsqueda léxica (BM25):**
> - Busca coincidencias exactas de palabras
> - Excelente para términos específicos, nombres propios, códigos
> - Falla si el usuario usa sinónimos
>
> **Búsqueda semántica (embeddings):**
> - Busca por significado, no palabras exactas
> - Excelente para consultas en lenguaje natural
> - Puede fallar con términos muy específicos o raros
>
> **¿Por qué combinarlas?** Porque se complementan. Si buscas 'error AWS S3 bucket permissions', quieres que encuentre documentos con esas palabras exactas (léxica) pero también documentos que hablen de 'problemas de acceso en almacenamiento cloud' (semántica).
>
> **RRF (Reciprocal Rank Fusion):**
> Es el algoritmo para fusionar los rankings:
> ```
> score = Σ 1/(k + rank)
> ```
> Pondera más los documentos que aparecen arriba en ambas búsquedas.
>
> En mi proyecto, hybrid search mejoró recall@10 en **20%** vs solo semántico. Los libros con títulos específicos que la búsqueda semántica perdía, BM25 los encontraba."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿Cómo elegiste los pesos de RRF?" | "Experimentación con el evaluation set. Empecé 50/50 y ajusté. En mi caso 60% semántico / 40% léxico funcionó mejor porque las consultas eran principalmente en lenguaje natural." |
| "¿Y si un método devuelve 0 resultados?" | "Uso solo el otro. RRF maneja esto naturalmente - si solo hay rankings de un método, esos dominan." |
| "¿Por qué no solo semántico con mejor modelo?" | "Incluso el mejor modelo semántico pierde keywords exactos. Es un problema fundamental, no de calidad del modelo." |

**NO digas:**
- ❌ "Semántico es siempre mejor" (no lo es para términos específicos)
- ❌ Olvidar explicar RRF
- ❌ No dar el número concreto de mejora

---

### 6. ¿Qué son Embeddings?

**Respuesta (~60 segundos):**

> "Los embeddings son **representaciones vectoriales densas** que capturan el significado semántico de texto (o imágenes, audio, etc.) en un espacio de alta dimensión.
>
> **Características clave:**
> - **Densos**: Cada dimensión tiene un valor (vs sparse como bag-of-words)
> - **Dimensión fija**: Típicamente 384, 768, o 1536 dimensiones
> - **Semánticos**: Textos similares en significado → vectores cercanos en el espacio
>
> **¿Cómo se generan?**
> Un modelo de embeddings (como `text-embedding-3-small` de OpenAI o sentence-transformers) procesa el texto y produce el vector.
>
> **¿Por qué son útiles?**
> Permiten buscar por significado, no por palabras exactas. 'Coche rojo' y 'automóvil escarlata' tendrán embeddings cercanos aunque no compartan palabras.
>
> En mi proyecto, cada libro tiene un embedding de su descripción. Cuando el usuario busca, convierto su query a embedding y busco los más cercanos con FAISS.
>
> Para seguridad: podrías embeddear logs de seguridad y buscar patrones similares a ataques conocidos sin necesitar reglas exactas."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿Cómo elegir el modelo de embeddings?" | "Depende del tradeoff: modelos más grandes = mejor calidad pero más lentos y caros. Para mi proyecto usé uno mediano porque el volumen era manejable. En producción con millones de documentos, optimizaría más." |
| "¿Los embeddings se pueden actualizar?" | "Hay que regenerarlos si cambias de modelo o si el contenido cambia. Por eso es importante tener el pipeline de indexación automatizado." |

**NO digas:**
- ❌ Confundir embeddings con one-hot encoding
- ❌ "Son como palabras convertidas a números" (muy simplificado)

---

### 7. Cuéntame de tu Thesis (2 MIN MAX)

**USA EL PITCH DE ARRIBA - está al inicio del documento.**

**Tips adicionales:**
- Haz contacto visual, no recites
- Si ves que se aburren, salta al resultado (20% mejora)
- Prepárate para que interrumpan - es buena señal, significa que les interesa algo específico

---

### 8. ¿Cómo debuggear un agente que falla?

**Respuesta (~90 segundos):**

> "El debugging de agentes es más complejo que software tradicional porque hay no-determinismo y múltiples componentes. Mi approach:
>
> **1. Logging estructurado**
> Loggeo cada paso del agente: input, thought, action, observation, output. En formato JSON para poder filtrar y analizar.
>
> **2. Tracing**
> Uso LangSmith (o similar) para ver el trace completo: qué prompts se enviaron, qué respondió el LLM, cuánto tardó cada paso. Es como un debugger visual para agentes.
>
> **3. Identificar dónde falla**
> - ¿El retrieval devuelve documentos irrelevantes? → Problema de indexación o query
> - ¿El LLM ignora los documentos? → Problema de prompt
> - ¿El agente entra en loop? → Problema de condición de salida
> - ¿El JSON es inválido? → Problema de parsing/validación
>
> **4. Reproducibilidad**
> Guardo las queries que fallan como test cases. Configuro temperature=0 para debugging (elimina variabilidad).
>
> **5. Validación en cada paso**
> No espero al final para validar. Valido el output del retrieval antes de pasarlo al LLM, valido el JSON antes de ejecutar acciones.
>
> En seguridad cloud esto es crítico - un agente que falla silenciosamente y toma acciones incorrectas es peor que uno que falla ruidosamente."

**Follow-ups probables:**

| Pregunta | Respuesta |
|----------|-----------|
| "¿Qué herramienta de tracing usas?" | "LangSmith es la integrada con LangChain. También he visto Weights & Biases y Phoenix. Lo importante es tener visibilidad del trace completo." |
| "¿Cómo manejas errores en producción?" | "Circuit breaker pattern: si un componente falla N veces, lo deshabilito temporalmente. Alertas a Slack/PagerDuty. Fallback a respuesta segura." |

**NO digas:**
- ❌ "Uso print statements" (suena amateur)
- ❌ Olvidar mencionar el tracing - es esencial para agentes

---

## PREGUNTAS TÉCNICAS POSIBLES

---

### 9. FAISS vs Pinecone

**Respuesta (~60 segundos):**

> "FAISS y Pinecone resuelven el mismo problema - búsqueda de vectores similares - pero con tradeoffs diferentes:
>
> | | FAISS | Pinecone |
> |---|-------|----------|
> | **Tipo** | Librería (self-hosted) | Servicio managed |
> | **Costo** | Gratis, pagas infra | Pay-per-use |
> | **Escala** | Tú manejas sharding | Escala automático |
> | **Latencia** | Muy baja (local) | Red overhead |
> | **Setup** | Más trabajo | Plug and play |
>
> **¿Cuándo FAISS?**
> - Volumen pequeño-mediano (<1M vectores)
> - Latencia crítica
> - Quieres control total
> - Presupuesto limitado
>
> **¿Cuándo Pinecone?**
> - Escala grande o variable
> - No quieres mantener infra
> - Necesitas features como filtering, namespaces
>
> En mi proyecto usé FAISS porque tenía ~10K libros y no necesitaba escala. Para un producto de seguridad con millones de eventos, probablemente elegiría Pinecone o similar para no preocuparme por la infra."

**NO digas:**
- ❌ "FAISS es mejor porque es gratis" (depende del contexto)
- ❌ Olvidar mencionar que FAISS requiere mantener la infra

---

### 10. ¿Qué es LangGraph?

**Respuesta (~60 segundos):**

> "LangGraph es un framework para construir **agentes y workflows con estado** como grafos.
>
> **Conceptos clave:**
> - **StateGraph**: Define el estado que se comparte entre nodos
> - **Nodes**: Funciones que transforman el estado (ej: 'retrieve', 'generate', 'route')
> - **Edges**: Conexiones entre nodos, pueden ser condicionales
> - **Checkpointing**: Guarda el estado para poder resumir o debuggear
>
> **¿Por qué no solo LangChain?**
> LangChain es más lineal (chains). LangGraph permite **ciclos y branching** - esencial para agentes que iteran.
>
> En mi proyecto, el grafo tiene:
> ```
> START → Router → [Search Node | Direct Response Node] → Generate → END
> ```
> El Router decide qué camino tomar basándose en la query. Esto es difícil de hacer limpiamente con chains lineales.
>
> Para seguridad: podrías tener un grafo que va de 'Detectar Anomalía' → 'Investigar' → 'Decidir Severidad' → ['Escalar' | 'Auto-remediar' | 'Ignorar']. El estado mantiene todo el contexto de la investigación."

**NO digas:**
- ❌ "Es LangChain pero mejor" (son complementarios)
- ❌ Confundir con LangChain Expression Language (LCEL)

---

## PREGUNTAS BEHAVIORAL (Formato STAR)

---

### 11. Un problema técnico difícil que resolviste

**Respuesta STAR (~2 minutos):**

> **Situation:**
> "En mi proyecto de recomendación de libros, la búsqueda puramente semántica tenía un problema: perdía libros cuando el usuario mencionaba títulos o autores específicos. Si buscabas 'algo como los libros de Brandon Sanderson', a veces no aparecían sus libros porque el embedding de la query no capturaba bien el nombre propio."
>
> **Task:**
> "Necesitaba mejorar el recall sin sacrificar la capacidad de entender consultas en lenguaje natural."
>
> **Action:**
> "Investigué y decidí implementar búsqueda híbrida: BM25 para capturar keywords exactos + FAISS para semántica. El reto fue cómo fusionar los rankings. Probé varias estrategias:
> 1. Promedio simple de scores (no funcionó porque las escalas son diferentes)
> 2. Normalización + promedio (mejor pero perdía información de ranking)
> 3. RRF (Reciprocal Rank Fusion) - funcionó mejor porque pondera por posición, no por score absoluto
>
> También tuve que tunear el parámetro k de RRF y los pesos relativos. Construí un evaluation set de ~50 queries con ground truth y optimicé contra recall@10."
>
> **Result:**
> "La búsqueda híbrida mejoró recall@10 en un **20%** comparado con solo semántico. Los casos que antes fallaban (búsquedas con nombres propios) ahora funcionaban, sin perder la capacidad de entender 'quiero algo épico y oscuro'."

---

### 12. ¿Por qué Dawnguard?

**Respuesta (~90 segundos):**

> "Tres razones principales:
>
> **1. El problema me interesa genuinamente**
> La intersección de AI y seguridad es donde veo más potencial de impacto. Los atacantes ya usan AI - la defensa necesita ponerse al día. Los agentes de seguridad que pueden investigar y responder automáticamente son el futuro.
>
> **2. Mi experiencia es directamente aplicable**
> Mi thesis es literalmente sobre agentes con RAG. El patrón de 'recibir señal → investigar contexto → decidir acción' que implementé para libros es el mismo que necesita un agente de seguridad. La diferencia es el dominio, no la arquitectura.
>
> **3. Stage de la empresa**
> En una startup early-stage puedo tener impacto real y aprender rápido. Prefiero construir el producto que mantener sistemas legacy. Y en seguridad cloud hay mucho espacio para innovar - no está todo resuelto.
>
> [Si preguntan más sobre la empresa, muestra que investigaste:]
> He visto que están trabajando en [menciona algo específico de su web/blog/LinkedIn]. Me interesa especialmente [aspecto concreto]."

**Personaliza esta respuesta:**
- Investiga Dawnguard antes de la entrevista
- Menciona algo específico de su producto o blog
- Conecta tu interés con algo concreto que hacen

---

## ERRORES COMUNES A EVITAR

### Durante toda la entrevista:
- ❌ Hablar más de 2 minutos sin pausa (deja que pregunten)
- ❌ Decir "no sé" sin intentar razonar (di "no lo he usado directamente pero imagino que...")
- ❌ Criticar tecnologías sin dar alternativas ("X es malo" → "X tiene limitaciones en Y, por eso preferí Z")
- ❌ Exagerar ("mi sistema es perfecto") - siempre menciona limitaciones
- ❌ Ser demasiado técnico sin context business

### En preguntas de tu proyecto:
- ❌ No saber los números (memoriza: 20% mejora recall@10, ~10K libros, arquitectura hexagonal)
- ❌ No poder explicar decisiones ("usé FAISS porque..." no "usé FAISS porque lo vi en un tutorial")
- ❌ No mencionar tradeoffs

### En behavioral:
- ❌ Respuestas sin estructura (usa STAR)
- ❌ No dar números o resultados concretos
- ❌ Hablar mal de otros (equipos, profesores, tecnologías)

---

## PREGUNTAS PARA HACERLES TÚ

Al final siempre preguntan "¿tienes preguntas para nosotros?". Prepara 2-3:

1. **Sobre el equipo:**
   "¿Cómo es el equipo de ML/AI actualmente? ¿Cuántas personas y qué roles?"

2. **Sobre el producto:**
   "¿Cuál es el mayor reto técnico que están enfrentando ahora con los agentes?"

3. **Sobre la cultura:**
   "¿Cómo es el proceso de desarrollo? ¿Iteran rápido con usuarios o hacen releases más grandes?"

4. **Sobre el rol:**
   "¿Qué esperarían que lograra alguien en este rol en los primeros 3 meses?"

**NO preguntes:**
- ❌ Cosas que puedes googlear (funding, número de empleados si está en LinkedIn)
- ❌ Salario en la primera entrevista (espera a que ellos lo mencionen)
- ❌ "¿Qué hace la empresa?" (demuestra que no investigaste)

---

## CHECKLIST ANTES DE LA ENTREVISTA

- [ ] Releer este documento
- [ ] Practicar el pitch en voz alta (cronometra los 2 min)
- [ ] Investigar Dawnguard: web, blog, LinkedIn de fundadores
- [ ] Tener agua cerca
- [ ] Probar cámara y micro si es remota
- [ ] Tener el proyecto abierto por si piden compartir pantalla
- [ ] Preparar preguntas para ellos
- [ ] Dormir bien

---

**Mucha suerte. Ya sabes esto - solo necesitas comunicarlo con confianza.**
