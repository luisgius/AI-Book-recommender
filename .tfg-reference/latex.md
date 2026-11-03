# Reglas de fuente LaTeX (memoria con `tfgetsinf`)

Se aplica al verificar ficheros `.tex` y `.bib` y el `.log` de compilación. Forma parte de la verificación descrita en [`REGLAS.md`](REGLAS.md) (sección 0); se usa el mismo formato de informe. Las reglas P-xx de `REGLAS.md` (mecánica de la clase) tienen prioridad sobre estas.

Fuente: LTX, *La introducción no-tan-corta a LaTeX 2ε* v5.03 (`originals/U0724513.pdf`). «p. N» es la página N del PDF, que en `text/U0724513_paged.txt` es el bloque tras el (N-1)-ésimo salto de página `\f`.

## Interacción con la clase (CLS prevalece sobre LTX)

- **L-01** La clase ya fija A4, Palatino, T1, babel, hyperref, listings, geometry y caption. El preámbulo no los vuelve a cargar ni cambia márgenes o fuentes.
- **L-02** La clase ejecuta `\frontmatter` y la portada al empezar el documento. El fuente no escribe `\frontmatter` ni `\maketitle`; escribe `\mainmatter` antes del primer capítulo (p. 53).
- **L-03** Sin `\backmatter`: en la plantilla, la bibliografía va tras las conclusiones y los anexos tras `\APPENDIX`.
- **L-04** La clase fija `\parskip` de 1ex y los enlaces (texto en negro, URL en cian). LTX desaconseja `\parskip` (p. 136) y los enlaces en color para imprimir (p. 100-101); aquí manda la clase, porque la entrega es electrónica (N-02). Si se imprime un ejemplar, conviene poner `urlcolor=black` en esa versión.
- **L-05** La clase carga hyperref, así que todo `\usepackage` del preámbulo queda detrás de él, en contra de la recomendación de LTX (p. 50, 99). Se añaden solo los paquetes imprescindibles y se revisa el `.log` tras cada uno.
- **L-06** Compilación con `pdflatex`. Las imágenes van en PDF (vectoriales), PNG (capturas) o JPEG (fotos), sin EPS, y `\includegraphics` se escribe sin extensión (p. 99, 103).

## Castellano con babel

- **L-07** Con `spanish`, en modo matemático un punto seguido de dígito sale como coma decimal (`$0.54$` da 0,54) (p. 44). El signo decimal es el mismo en texto, tablas y fórmulas (p. 60).
- **L-08** Comillas: «…» con `<<…>>` y, anidadas, “…” con ``` ``…'' ```. Nunca la comilla recta `"`, que además es carácter activo con spanish (p. 35-36, 43).
- **L-09** Puntos suspensivos con `\dots` en castellano y `\ldots` en pasajes en inglés (p. 37, 46).
- **L-10** Guiones: `-` en compuestos, `---` como raya de inciso, pegada a él (`texto ---inciso--- texto`), y `$-1$` como signo menos. En tipografía española tradicional no existe la raya corta: los intervalos se escriben con guion o con `--`, pero siempre del mismo modo en todo el documento (p. 36).
- **L-11** Las funciones en castellano son `\sen`, `\tg` y `\arcsen` (p. 69). Babel acentúa lím, máx y mín (p. 45).
- **L-12** `\hyphenation{}` va después de `\begin{document}` y solo con caracteres ASCII (p. 34).
- **L-13** En una figura TikZ, `\shorthandoff{...}` evita choques con los atajos de spanish (p. 123).

## Espacios y caracteres

- **L-14** `~` delante de `\ref`, `\pageref`, `\eqref` y `\cite`, y entre número y unidad (p. 22, 51, 54, 90).
- **L-15** Tras una orden sin argumento se pone `{}` o `\ ` para no perder el espacio (`\LaTeX{} es`) (p. 19).
- **L-16** Los caracteres reservados se escapan en el texto: `\# \$ \% \& \_ \{ \}`. La antibarra se escribe `$\backslash$` o `\textbackslash` (p. 19).
- **L-17** Barra entre palabras con `\slash`; la `/` sola queda para unidades y cocientes (p. 36).
- **L-18** El euro con `\euro` (paquete eurosym) o `\texteuro` (textcomp) (p. 37).

## Párrafos y estructura

- **L-19** Un párrafo por idea. Se termina con una línea en blanco, no con `\\` (p. 18, 31, 33).
- **L-20** Las secciones siguen la jerarquía sin saltarse niveles. Un título largo lleva versión corta: `\section[corto]{largo}` (p. 51-53).
- **L-21** Nada de `\sloppy` ni de `\linespread` (p. 34, 135).
- **L-22** Los espacios y anchos manuales son relativos (`\textwidth`, `em`, `ex`, `\bigskip`), nunca cm, mm o pt absolutos (p. 136-140). Ver R-12.
- **L-23** Un formato que se repite con significado propio (nombres de clases, rutas, métricas) se define una vez con `\newcommand` (p. 127-128, 134).

## Énfasis y listas

- **L-24** El énfasis se marca con `\emph{}`; sin `\underline` (p. 55).
- **L-25** Las listas usan itemize, enumerate o description; las citas largas, `quote` o `quotation` (p. 56-57).
- **L-26** Ni `\verb` ni `verbatim` dentro del argumento de otra orden (`\section`, `\caption`, `\footnote`) (p. 58).

## Referencias, notas y flotantes

- **L-27** `\label` va justo después de `\caption`, dentro del flotante (p. 54, 63).
- **L-28** Las etiquetas llevan prefijo de tipo: `fig:`, `tab:`, `cua:`, `alg:`, `sec:`, `cap:`, `eq:` (p. 54, 66).
- **L-29** Colocador de varias letras (`[ht]`, `[tb]`, `[tbp]`). Nunca `[h]` solo; `[H]` solo como último recurso (p. 61-62; MAN).
- **L-30** Las órdenes frágiles en argumentos móviles llevan `\protect` (`\caption{...\protect\footnote{...}}`). Los títulos con matemáticas o color usan `\texorpdfstring` (p. 63, 102-103).
- **L-31** Tablas: texto largo en columnas `p{ancho}`; cabeceras compartidas con `\multicolumn`; longtable si la tabla ocupa más de una página; anchura total de la tabla como máximo `\textwidth` (p. 59-60, 141).

## Matemáticas

- **L-32** Las fórmulas en línea van con `$…$` y las destacadas con `\[…\]` o `equation`; las de varias líneas con align, gather o multline de amsmath, no con `eqnarray` (p. 65-66, 74).
- **L-33** Sin líneas en blanco dentro de una fórmula ni alrededor de una ecuación destacada que continúa el párrafo (p. 31-32, 67).
- **L-34** La puntuación tras una fórmula destacada va dentro de ella, separada con `\;` (p. 32).
- **L-35** Los subíndices que son palabras van en redonda con `\mathrm{}` o `\textrm{}` (`\mathrm{NDCG}_{10}`, `s_{\mathrm{RRF}}`); `\mathrm` sin espacios ni acentos (p. 67, 75).
- **L-36** `\ldots` en listas con comas y `\cdots` entre operadores. Diferencial con `\,\mathrm{d}x` (p. 71-72).
- **L-37** Ecuaciones citadas con `\eqref{}` (amsmath) o `(\ref{})` (p. 66).

## Bibliografía (complementa la sección 8 de `REGLAS.md`)

- **L-38** BibTeX con base de datos `.bib` o `thebibliography` manual con `\bibitem{clave}` (p. 89-90). El estilo lo fija DEC-4.

## Comprobaciones del `.log` y el PDF

- **L-39** Sin `Overfull \hbox` (p. 34): `rg -n 'Overfull \\hbox' *.log`.
- **L-40** Sin `Font ... not found` (p. 98), sin `Token not allowed in a PDFDocEncoded string` (p. 102) y sin avisos de volver a compilar (p. 52).
- **L-41** Sin referencias ni citas sin resolver: `pdftotext memoria.pdf - | rg -n '\?\?'` (derivado de p. 52 y 54).

## Patrones de búsqueda de errores frecuentes

Se ejecutan con `rg -n` (con `-P` si hay lookbehind y `-U` si abarcan varias líneas) sobre los `.tex`, fuera de bloques `verbatim` y `lstlisting`. Un acierto no siempre es un error: hay que revisarlo y dar un veredicto.

- `"[^"\n]{2,}"`: comillas rectas (L-08).
- `(?<!\\)\.\.\.`: tres puntos en lugar de `\dots` (L-09).
- `\\\\\s*\n\s*\n` (`-U`): `\\` al final de párrafo (L-19).
- `[^~(\s]\s+\\(ref|pageref|eqref|cite)\{`: falta `~` antes de la referencia o cita (L-14).
- `\s\\footnote\{`: espacio antes de `\footnote` (P-07).
- `\\footnote\{(?:[^{}]|\{[^{}]*\})*\}[.,;:]`: nota antes de la puntuación (P-07).
- `\\underline\{`: subrayado como énfasis (L-24).
- `\\begin\{(figure|table|quadre|algorithm)\}\[[hH]?\]`: colocador `[h]` o vacío (L-29).
- `\$\$`: fórmula destacada al estilo TeX (L-32).
- `\\begin\{eqnarray`: eqnarray (L-32).
- `\\(sloppy|linespread)\b`: relajación del ajuste o del interlineado (L-21).
- `\\(hspace|vspace)\*?\{[\d.]+(cm|mm|pt|in)\}`: espaciado absoluto (L-22).
- `width=[\d.]+(cm|mm|pt|in)`: ancho absoluto (L-22).
- `\\includegraphics(\[[^]]*\])?\{[^}]*\.(eps|png|jpg|pdf)\}`: extensión explícita (L-06).
- `(?<!\\)[#&_](?![^$]*\$)` en prosa: carácter reservado sin escapar (L-16).
- `\\(LaTeX|TeX)\s+\p{L}`: orden que se come el espacio (L-15).
- `\\(sin|tan|arcsin|arctan)\b`: función en inglés en texto castellano (L-11).
- `\\mathrm\{[^}]*[ áéíóúñ][^}]*\}`: `\mathrm` con espacio o acento (L-35).
- `\\(frontmatter|maketitle|backmatter)\b`: órdenes que ya pone la clase o que no van (L-02, L-03).
- `\\usepackage(\[[^]]*\])?\{(geometry|babel|fontenc|hyperref|graphicx|listings|caption)\}`: paquete ya cargado por la clase (L-01).
