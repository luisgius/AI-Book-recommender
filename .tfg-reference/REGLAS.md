# Reglas de redacción del TFG (ETSINF, UPV)

Fuente única de verdad para escribir y **verificar** cualquier texto de la memoria del TFG: capítulos en LaTeX, bibliografía, resúmenes, anexo ODS, documentos Word y diapositivas de la defensa. Cada regla lleva un identificador y la fuente de la que sale. Los documentos originales están en `originals/` y su texto plano, apto para `grep`, en `text/`.

## 0. Cómo verificar

1. Identifica qué se entrega: memoria (LaTeX o Word), un capítulo suelto, bibliografía, resumen, anexo ODS o diapositivas.
2. Aplica **todas** las reglas de las secciones 3 a 9 que afecten a ese entregable. Si es fuente LaTeX (`.tex`, `.bib`), aplica además [`latex.md`](latex.md). Si es la defensa o las diapositivas, aplica [`defensa.md`](defensa.md). La sección 10 (CCII) solo se aplica a los apartados de gestión y análisis que la memoria adopte.
3. Contrasta el texto con las decisiones abiertas (sección 2). Si un texto depende de una decisión no tomada, señálalo; no la decidas en silencio.
4. Entrega un informe con una línea por regla aplicada: `ID | CUMPLE / INCUMPLE / NO APLICA | ubicación (fichero:línea o página) | corrección propuesta`.

**Criterio de finalización:** cada regla aplicable tiene un veredicto explícito y cada incumplimiento tiene ubicación y corrección. Una verificación que omite reglas sin declararlo está incompleta.

## 1. Fuentes y precedencia

| Etiqueta | Documento | Fichero en `originals/` |
|---|---|---|
| NOR | Normativa TFG y TFM de la UPV (BOUPV 2025/0150, 17-09-2025; modificada por Consejo de Gobierno 28-07-2025) | `U0981343.pdf` |
| EST | ETSINF, *Trabajo Fin de Grado: Estructura y Contenidos Recomendados* (versión 6-3-2019) | `EstucturayContenidodeunTFGUltimaversion6_3_19.pdf` |
| REC | ETSINF, *Recomendaciones para la escritura* y *para la defensa* (7-11-2018) | `Recomendaciones-Escritura-Defensa-TFG-7_11_18.pdf` |
| MAN | Fuster y Molero, *La classe de document tfgetsinf: manual de l'usuari i guia de composició* (2015) | `U0724302.pdf` (igual a `tfgetsinf/tfgetsinfManual.pdf`) |
| CLS | Clase `tfgetsinf.cls` y plantilla `plantillatfg.tex` (2015); ejemplo compilado en `U0724309.pdf` | `tfgetsinf/` |
| ODS | Instrucciones y plantillas del anexo ODS de la ETSINF; Anexo I de NOR | `ODS-en-TFG-TFM-*`, `ods_etsinf_latex.zip` |
| CCII | Norma CCII-N2016-02 y su plantilla básica (2016) | `norma.pdf`, `plantillaNorma.pdf` |
| LTX | Oetiker et al., *La introducción no-tan-corta a LaTeX 2ε* v5.03 (2014) | `U0724513.pdf` |

Ante un conflicto, prevalece la primera de esta lista:

1. NOR (obligatoria).
2. Lo marcado como *Obligatorio* por la ETSINF: plantilla, anexo ODS, apartados obligatorios de EST.
3. Indicaciones del tutor (NOR art. 5.1.a; REC: «salvo que el tutor indique lo contrario»). No anulan los niveles 1 y 2.
4. Estilo de la plantilla: MAN y CLS.
5. Recomendaciones: EST y REC.
6. CCII (norma profesional voluntaria; complementa la estructura en análisis, riesgos, planificación y presupuesto).
7. LTX (tipografía LaTeX general).

Conflictos ya resueltos con esta regla:

- **Formato de página:** REC propone Arial o Times, cuerpo de 10 a 11 pt, títulos de 16 pt y márgenes de 3 cm para procesadores de texto. En LaTeX manda CLS: Palatino, cuerpo de 11 o 12 pt (nunca 10 pt), márgenes de 3 cm a los lados y 2,35 cm arriba y abajo, y la separación entre párrafos de la clase.
- **Primera persona:** REC excluye la primera persona del singular. EST la admite solo en el subapartado de motivación personal. Regla: impersonal o plural de modestia en todo el texto; singular únicamente en «Motivación personal», si existe.
- **Webs en la bibliografía:** EST manda las webs de fabricantes, productos y vídeos a notas al pie. MAN admite en la bibliografía los documentos accesibles en la red. Regla: producto, herramienta, web comercial o vídeo van en nota al pie la primera vez que aparecen; un documento formal en línea (artículo, informe, estándar) va a la bibliografía con su URL.

## 2. Decisiones abiertas

Están pendientes de cerrar con el tutor. Hasta entonces, cualquier verificación que toque estos puntos debe señalarlos.

- **DEC-1 Título.** El título de trabajo, «Sistema inteligente de recomendación de libros mediante PLN y LLMs», contiene dos acrónimos (EST lo prohíbe: E-03) y tiene 10 palabras (EST: los títulos más efectivos tienen menos de 10). El título debe coincidir con el aprobado por la CAT (E-04): primero hay que comprobar cuál es el aprobado en la aplicación institucional, y para cambiarlo hay que pedir una modificación.
- **DEC-2 Lengua principal.** Castellano, valenciano o inglés (N-02). Fija la opción `spanish`, `catalan` o `english` de `\documentclass`.
- **DEC-3 Resúmenes.** EST exige al menos inglés más castellano o valenciano; la plantilla trae tres huecos (valenciano, castellano, inglés). Decidir si se incluye el valenciano.
- **DEC-4 Estilo bibliográfico.** MAN define un modelo propio (orden alfabético, tipografía por tipo de documento: B-04 a B-06); EST recomienda ISO 690-2010. Elegir uno y aplicarlo en todas las entradas.
- **DEC-5 Siglas en su primera aparición.** MAN pone la sigla y el desarrollo detrás («USB (*Universal serial bus*)»); CCII pone el desarrollo y la sigla detrás («forma completa (SIGLA)»). Elegir un patrón y usarlo siempre.
- **DEC-6 IA generativa.** Ninguna de las fuentes regula el uso de IA generativa en la redacción. Confirmar con el tutor y con la política vigente de la UPV si debe declararse y cómo, antes de la entrega.
- **DEC-7 Alcance de CCII.** Qué apartados de CCII (riesgos, planificación, presupuesto, plan de seguridad) se incorporan a la memoria.

## 3. Requisitos normativos (NOR)

- **N-01** El TFG es un trabajo original que demuestra competencias del título; la defensa es individual (art. 2.3-2.4).
- **N-02** La memoria se presenta en formato electrónico por la aplicación institucional, en castellano, valenciano o inglés (art. 10.1).
- **N-03** La portada incluye como mínimo autor, título, tutor y cotutor (si lo hay), ERT (ETSINF), titulación, curso académico y logotipo de la UPV (art. 10.2).
- **N-04** La memoria incorpora la relación del trabajo con los ODS según el Anexo I (art. 10.3). Detalle en la sección 9.
- **N-05** Antes de pedir la defensa, el trabajo pasa obligatoriamente el análisis de originalidad (antiplagio) de la herramienta institucional (art. 14.1). Un porcentaje de similitud relevante puede llevar a un «no apto» (art. 14.5).
- **N-06** Cualquier trabajo con empresa exige convenio y cotutor externo (art. 3.3). Las restricciones por confidencialidad se solicitan a la CAT (art. 15).
- **N-07** La defensa es pública y dura como máximo 45 minutos, preguntas incluidas (art. 13.6). Detalle en `defensa.md`.
- **N-08** El tribunal puede pedir cambios menores antes del «apto» sin nueva defensa (art. 13.12); un «no apto» obliga a modificar y volver a defender (art. 13.11).

## 4. Estructura de la memoria (EST, MAN, CLS)

Parte inicial:

- **E-01** La memoria usa la plantilla oficial de la ETSINF (EST) y tiene entre 50 y 100 páginas sin contar anexos (EST).
- **E-02** La portada la genera la clase. Las páginas iniciales van sin numerar o en romanos (EST; CLS usa `\frontmatter`).
- **E-03** El título identifica el trabajo sin resumirlo, no lleva abreviaturas ni acrónimos y conviene que tenga menos de 10 palabras (EST). Va en minúsculas salvo la inicial y sin punto final (MAN).
- **E-04** El título coincide con el aprobado por la CAT; si no, se solicita la modificación (EST).
- **E-05** Constan el nombre completo del autor, el curso de la convocatoria de defensa con formato `2025-2026` (MAN) y el nombre y apellidos del tutor y de los cotutores (EST).
- **E-06** Los agradecimientos son opcionales, pero si el trabajo está financiado (proyecto, beca o similar) la mención es obligatoria (EST).
- **E-07** El resumen es obligatorio, sintetiza el problema, la metodología, las herramientas, los resultados y las conclusiones, es autocontenido (sin remitir a páginas ni capítulos), normalmente sin citas, y tiene entre 200 y 500 palabras (EST). Se escribe con el trabajo casi terminado (EST).
- **E-08** Hay resumen al menos en inglés y en castellano o valenciano (EST; ver DEC-3).
- **E-09** Las palabras clave están en cada idioma del resumen, son entre 3 y 10 y pueden ser compuestas (EST, MAN).
- **E-10** El índice general es obligatorio, con profundidad 3 y como mucho 4 (EST). Los índices de figuras, tablas, cuadros y algoritmos se activan con las opciones de la clase cuando la memoria contiene esos objetos (MAN).

Capítulos. Cada capítulo empieza con un párrafo que presenta lo que trata y se divide en secciones (MAN).

- **E-11** El capítulo 1, «Introducción», expone el problema de forma sencilla y global sin suponer que el lector conoce el tema, y motiva a seguir leyendo (EST, MAN).
- **E-12** La motivación explica por qué se hace el trabajo y valora las alternativas de tema, y sirve de base a los objetivos (EST).
- **E-13** Los objetivos están en una enumeración (`enumerate`; MAN), redactados con verbos en infinitivo, concretos, alcanzables, medibles y ordenados por relevancia (un objetivo general y varios específicos) (EST).
- **E-14** El impacto esperado describe las mejoras para cada tipo de usuario y puede enlazar con los ODS (EST).
- **E-15** La metodología explica los pasos para cumplir los objetivos y justifica que se ajusta al tipo de trabajo (EST).
- **E-16** La estructura de la memoria es un índice comentado de los capítulos (hasta el segundo nivel), describe también los anexos y anuncia el glosario si existe (EST, MAN).
- **E-17** Si el trabajo es colaborativo, se identifica a todo el equipo y se delimita qué hizo el autor (obligatorio si es el caso; EST).
- **E-18** El apartado de convenciones declara cualquier marcado con significado propio (código en monoespaciada, extranjerismos en cursiva, citas textuales entre comillas) (EST).
- **E-19** El estado del arte documenta soluciones existentes, apoya las afirmaciones en referencias fiables y termina con una crítica que identifica el hueco y una propuesta que explica qué aporta el trabajo (EST).
- **E-20** El análisis del problema especifica requisitos con técnicas estándar (plantillas, casos de uso, UML) y valora, según proceda, seguridad, eficiencia o energía, marco legal y ético (protección de datos, licencias, propiedad intelectual) y riesgos (tipo, impacto y medidas) (EST).
- **E-21** Se analizan las soluciones posibles con pros y contras y un criterio de selección explícito, y se justifica la elegida (EST).
- **E-22** El plan de trabajo indica las fases con horas-persona y compara lo planificado con lo real, con una reflexión crítica sobre las desviaciones; el presupuesto recoge los recursos humanos y materiales (hardware, software, horas) (EST).
- **E-23** El diseño presenta la arquitectura con un diagrama de bloques o UML de componentes o despliegue, el diseño detallado (módulos, clases, base de datos, estructura de directorios) y la tecnología usada con su justificación (EST).
- **E-24** El desarrollo cuenta cómo se pasó de la propuesta a la solución, las dificultades y las decisiones, con el mínimo código (ver R-14) (EST).
- **E-25** Hay apartados de implantación y de pruebas: verificación, validación con usuarios y, si procede, pruebas de carga (EST).
- **E-26** Las conclusiones retoman cada objetivo de la introducción y dicen si se alcanzó; no repiten resultados ni introducen objetivos nuevos; recogen los problemas encontrados, los errores y lo aprendido (EST, MAN). REC recomienda copiar la lista de objetivos en las conclusiones para vincularlos.
- **E-27** La «Relación del trabajo desarrollado con los estudios cursados» es obligatoria e incluye las competencias transversales aplicadas (EST).
- **E-28** Los trabajos futuros son opcionales: flecos pendientes, ampliaciones y caminos que conviene no seguir, con sus motivos (EST).
- **E-29** La bibliografía es obligatoria (EST). Ver sección 8.
- **E-30** Los anexos empiezan con `\APPENDIX` y se escriben como capítulos (MAN). Contienen lo que no hace falta para entender la memoria (detalle técnico, manual de usuario, código, material descartado) (EST).
- **E-31** El glosario es recomendable: recoge términos y acrónimos que un informático no especialista no tiene por qué conocer y se anuncia en la introducción (EST, REC). Se va completando mientras se escribe (REC).
- **E-32** El anexo ODS va al final de la memoria (ODS).

## 5. Redacción y estilo (REC, EST, MAN, CCII)

- **R-01** Público objetivo: un profesional con cultura informática general que no es especialista en el tema. Los tecnicismos se explican (REC, EST).
- **R-02** El lenguaje es denotativo y unívoco: palabras exactas, una sola lectura posible y sin uso creativo (REC). La prosa solo se permite en los primeros párrafos de la introducción (EST).
- **R-03** Redacción impersonal o plural de modestia; la primera persona del singular solo en la motivación personal (REC, EST; ver sección 1).
- **R-04** Vocabulario amplio y técnicamente preciso, sin coloquialismos, sin estilo directo o personal y sin abusar de abreviaturas (REC).
- **R-05** Frases bien construidas y puntuadas, sin oraciones largas y barrocas (REC).
- **R-06** Cero faltas de ortografía: corrector ortográfico más revisión manual completa (REC).
- **R-07** Sin copiar y pegar de ninguna fuente. Toda cita literal va entrecomillada y con su origen (MAN, NOR art. 14).
- **R-08** Citas textuales: en castellano y valenciano con comillas latinas «…» (en LaTeX `<<…>>`); en inglés con comillas inglesas (`` ``…'' ``) (MAN).
- **R-09** Las expresiones en otra lengua van en cursiva y con su lengua declarada: `\foreign{english}{...}` en línea y el entorno `Foreign` para párrafos (MAN; EST, convenciones).
- **R-10** Las siglas van en mayúsculas sin puntos (o en versalitas con `\textsc{}`) y se desarrollan la primera vez que aparecen (MAN, CCII; patrón en DEC-5).
- **R-11** Los requisitos se expresan con «debe» y las recomendaciones con «debería» o en condicional (CCII). Cada requisito tiene una sola interpretación y se puede verificar.
- **R-12** El aspecto visual (espacios, saltos de página, posición de los flotantes) lo decide LaTeX. Solo se fuerza algo a mano con el documento terminado y si no hay otra salida (MAN, REC). No se añaden espacios ni párrafos vacíos para alargar el texto (REC).
- **R-13** Notas al pie para definiciones puntuales, webs de productos o herramientas en su primera mención, URL de vídeos y la localización exacta de un dato (EST, MAN). Ver P-07 para su sintaxis.
- **R-14** Código fuente: lo mínimo imprescindible en el cuerpo, solo si es esencial, original o complejo, y siempre comentado. Lo prescindible va a un anexo (REC, EST).
- **R-15** Las figuras solo están si aclaran algo; el texto comenta lo que no se ve a simple vista (REC).

## 6. Plantilla `tfgetsinf` y composición (CLS, MAN)

- **P-01** Clase: `\documentclass[11pt|12pt, spanish|catalan|english, list...]{tfgetsinf}`. Sin `10pt` y sin opciones de tamaño de papel: siempre A4 (MAN, CLS). Lengua principal según DEC-2.
- **P-02** La plantilla define `\title`, `\author`, `\tutor` (con `\\` si hay varios tutores) y `\curs{AAAA-AAAA}` (MAN).
- **P-03** Revisa el texto fijo de la portada. CLS escribe «Grado en Ingeniería Informática» (`\titulacioname`) y «Escuela Técnica Superior de Ingeniería Informática / Universidad Politécnica de Valencia». Si la titulación es otra, redefine `\titulacioname` en el preámbulo (N-03).
- **P-04** `\keywords{valenciano}{castellano}{inglés}` siempre en ese orden, sea cual sea la lengua principal (MAN).
- **P-05** Mecánica de los resúmenes en CLS: el entorno `abstract` toma la lengua del argumento opcional (por defecto `catalan`), y **al cerrar el resumen en inglés la clase genera los índices**. Orden obligatorio: `abstract` (valenciano, si existe), `abstract[spanish]`, `abstract[english]` en último lugar, y después `\mainmatter`.
- **P-06** Los números, variables y unidades van en modo matemático; las unidades y constantes en redonda (`\mathrm{}`); los miles se separan con espacio fino (`\,`); entre número y unidad, espacio irrompible `~` (por ejemplo `$2\,048~\mathrm{GB}$`) (MAN).
- **P-07** `\footnote{}` va pegada a la palabra, sin espacio, y detrás de la puntuación: `palabra.\footnote{...}` (MAN).
- **P-08** Código con `lstlisting` y el lenguaje como opción (`[language=Python]`); la clase lo configura para C++ por defecto (MAN, CLS).
- **P-09** Flotantes: `figure` (Figura), `table` (Tabla), `quadre` (Cuadro; no en inglés) y `algorithm` (Algoritmo, con `algorithmic`). Tabla para datos tabulares, cuadro para información esquemática (MAN).
- **P-10** Compilación con `pdflatex` e imágenes en PDF, PNG o JPEG; con `latex`, `dvips` y `ps2pdf`, EPS (MAN).
- **P-11** La bibliografía se cierra con `\cleardoublepage` y los anexos empiezan con `\APPENDIX`, en mayúsculas (MAN, CLS).

## 7. Figuras, tablas, cuadros y algoritmos

- **F-01** Todo flotante está numerado y tiene leyenda (`\caption`) y etiqueta (`\label`) (REC, MAN).
- **F-02** Todo flotante se cita en el texto con espacio irrompible: `la figura~\ref{fig:arquitectura}`. Ningún flotante queda sin citar (MAN, EST, REC).
- **F-03** Una figura que no es del autor indica su fuente. Se prefieren imágenes con licencia Creative Commons o similar (MAN, REC).
- **F-04** Si la leyenda es larga, se da una versión corta para el índice: `\caption[corta]{larga}` (MAN).
- **F-05** La posición de los flotantes se ajusta solo al final; `[H]` solo si no hay alternativa (MAN).

## 8. Bibliografía y citas

- **B-01** Toda entrada de la bibliografía se cita al menos una vez en el texto, y toda cita del texto existe en la bibliografía (EST, MAN). Una misma fuente puede citarse varias veces.
- **B-02** Hay entre 15 y 25 fuentes aproximadamente, fiables (libros y artículos con revisión antes que webs) y pertinentes (MAN, EST).
- **B-03** Los algoritmos y tecnologías específicas se respaldan con artículos científicos o fuentes serias. La Wikipedia no sirve para conceptos de primer curso (EST).
- **B-04** Con el modelo MAN (DEC-4), el orden es alfabético por el apellido del primer autor (MAN).
- **B-05** Con el modelo MAN, un artículo se escribe: Autores. Título (redonda). *Revista* (cursiva), volumen:número:páginas, mes, año.
- **B-06** Con el modelo MAN, un libro se escribe: Autores. *Título* (cursiva). Editorial, ciudad, edición, año. Un documento en línea lleva descripción, autor o entidad, fecha y URL.
- **B-07** Con ISO 690-2010 (DEC-4), se aplica la norma completa y con coherencia en todas las entradas (EST).
- **B-08** Las webs de productos, herramientas o fabricantes y los vídeos van en nota al pie en su primera mención (EST; sección 1). Si una fuente secundaria remite a una web original, se cita y se comprueba la original (EST).
- **B-09** Las citas en el texto usan `~\cite{clave}` con espacio irrompible, y varias claves van en un solo `\cite{a,b,c}` (MAN).

## 9. Anexo ODS (NOR Anexo I, ODS)

- **O-01** Es un anexo obligatorio, al final de la memoria, y además se entrega como fichero aparte en la aplicación de depósito (ODS).
- **O-02** Tabla con los 17 ODS y su grado de relación (Alto, Medio, Bajo, No procede), con los nombres exactos del Anexo I de NOR. Plantilla LaTeX en `unzipped/ods_etsinf_latex/latex/`.
- **O-03** Una reflexión de entre 500 y 1500 palabras sobre los ODS más relacionados. Si no hay relación, se justifica por qué y se señalan los ODS con una relación menor (ODS).
- **O-04** Se borra el texto de instrucciones de la plantilla («Empieza a escribir aquí…») (ODS).

## 10. Complementos CCII (si la memoria los adopta; DEC-7)

- **C-01** La primera vez que aparece una sigla o abreviatura, se presenta junto al texto completo (CCII; patrón en DEC-5).
- **C-02** El texto se entiende sin los autores: está redactado para terceros no especialistas (CCII).
- **C-03** Estudio de alternativas: alternativas consideradas, criterios de valoración y justificación de la elegida y de las descartadas (CCII 10).
- **C-04** Análisis de riesgos: lista clasificada con impacto y acciones de mitigación (CCII 12).
- **C-05** Planificación temporal: cronograma con hitos, entregas parciales y duración, con criterios para actualizarlo (CCII 14).
- **C-06** Presupuesto: cifras sin ambigüedad, completas, sin costes ocultos, con un total desglosado por partidas y las bases de cálculo explícitas (CCII 15, 19).
- **C-07** Normas y referencias: legislación aplicable, sobre todo protección de datos y propiedad intelectual, más los métodos, herramientas y métricas usados (CCII 5). La plantilla de 2016 cita LOPD, RDLOPD, LSSI y el Reglamento UE de 2016; hoy se aplican el RGPD y la LOPDGDD.
- **C-08** Mecanismos de calidad: indicar cómo se verificó la integridad y completitud de la documentación, por ejemplo con la trazabilidad entre requisitos y especificaciones (CCII 5.4).
