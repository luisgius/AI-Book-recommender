// Client for the book search API (no framework, no external assets).
// Every value from the API is rendered with textContent (never parsed as
// HTML), so catalog text cannot inject markup.
//
// Sections: constants, API access, waiting states and messages, rendering of
// a book (card, why-row, detail dialog), search mode, assistant mode, start-up.

const API = "/api/v1";
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Example queries: several languages, general works the public catalog holds.
const SEARCH_EXAMPLES = [
  { text: "novelas sobre la guerra civil española", lang: "es" },
  { text: "introduction to machine learning for beginners", lang: "en" },
  { text: "roman d'aventures en mer", lang: "fr" },
  { text: "Geschichte Europas im Mittelalter", lang: "de" },
  { text: "recetas de cocina tradicional", lang: "es" },
  { text: "poesia d'amore", lang: "it" },
];
// Assistant openers: a specific request (searches at once), a vague one (should
// get a clarifying question) and a multi-step one (should go to the agent).
const ASSISTANT_EXAMPLES = [
  { text: "Recomiéndame un libro de cocina mexicana", lang: "es" },
  { text: "I'd like something good to read", lang: "en" },
  { text: "Trouve un livre d'introduction au machine learning, puis des livres similaires à celui-là", lang: "fr" },
];

const LANGUAGE_NAMES = {
  es: "español", en: "inglés", fr: "francés", de: "alemán", it: "italiano", pt: "portugués",
  ca: "catalán", nl: "neerlandés", ru: "ruso", zh: "chino", ja: "japonés", ar: "árabe",
  la: "latín", sv: "sueco", pl: "polaco",
};
const FIELD_NAMES = { title: "título", description: "descripción", categories: "categorías", authors: "autores" };
const INTENT_NAMES = { recommendation: "recomendación", factual: "dato concreto", exploratory: "exploración" };
const FILTER_NAMES = { language: "idioma", category: "categoría", min_year: "desde", max_year: "hasta" };

// Agent tools (app/infrastructure/llm/agents/search_agent.py) and their arguments.
const TOOL_NAMES = {
  search_books: "Búsqueda híbrida",
  get_book_details: "Ficha completa de un libro",
  find_similar_books: "Libros similares",
};
const ARGUMENT_NAMES = { query: "consulta", language: "idioma", min_year: "desde", max_year: "hasta", book_id: "libro" };

// Names of the catalog sources and the licence deeds the attribution links to.
const SOURCE_NAMES = {
  wikipedia: "Wikipedia", wikidata: "Wikidata", gutenberg: "Project Gutenberg",
  doab: "DOAB", open_library: "Open Library", google_books: "Google Books",
};
const LICENSE_URLS = {
  "CC BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
  "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
  "CC0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
};

// Notices the API returns when a daily budget of the public demo is spent,
// or when the request asked for the LLM and this server has none.
const NOTICE_TEXT = {
  search_budget_spent: "Se ha alcanzado el límite diario de búsquedas con modelos de esta demostración. "
    + "Hasta las 00:00 UTC la búsqueda usa solo palabras clave (BM25), sin búsqueda por significado ni reordenador. "
    + "Los resultados siguen siendo válidos, pero no entienden sinónimos ni otros idiomas.",
  llm_budget_spent: "Se ha alcanzado el límite diario de peticiones al LLM de esta demostración. "
    + "Hasta las 00:00 UTC no hay interpretación de la consulta, explicaciones ni agente; la búsqueda funciona igual.",
  llm_unavailable: "Este servidor no tiene un LLM configurado (falta OPENROUTER_API_KEY), así que no se ha "
    + "interpretado la consulta ni se han generado explicaciones. La búsqueda funciona igual.",
};
// The same notices, as the assistant explains them (it degrades to a plain search),
// and the assistant's own codes (app/infrastructure/llm/graphs/assistant.py).
const ASSISTANT_NOTICE_TEXT = {
  llm_unavailable: "Este servidor no tiene un LLM configurado (falta OPENROUTER_API_KEY): el asistente no puede "
    + "preguntar ni redactar una respuesta, así que ha hecho una búsqueda simple de tu último mensaje.",
  llm_budget_spent: "Se ha alcanzado el límite diario de peticiones al LLM de esta demostración: hasta las 00:00 UTC "
    + "el asistente hace una búsqueda simple de tu último mensaje, sin preguntar ni redactar una respuesta.",
  assistant_answer_failed: "El LLM no ha podido redactar la respuesta: se muestran los resultados de la búsqueda sin recomendación.",
  agent_failed: "El agente ha fallado y el asistente ha usado el camino de búsqueda.",
};
const NO_LLM_NOTE = "Este servidor no tiene un LLM configurado (falta OPENROUTER_API_KEY, ver .env.example): "
  + "las opciones con LLM y el agente no están disponibles. La búsqueda funciona sin ellas.";

// Waiting texts: the public deployment sleeps when idle (cold start).
const COLD_START_HINT = "Si el servidor estaba inactivo, la primera petición puede tardar entre 10 y 20 segundos "
  + "mientras arranca. La página no se ha quedado colgada.";

// Health of the server, filled at start-up (null until known).
let health = null;
// The last search response, to go back to it from the similar-books view.
let lastSearch = null;

// ---------------------------------------------------------------------------
// API access
// ---------------------------------------------------------------------------

/** An API failure with the HTTP status (0: the request did not reach the server). */
class ApiError extends Error {
  constructor(status, detail, retryAfter = null) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.retryAfter = retryAfter;
  }
}

async function callApi(path, options = {}) {
  let response;
  try {
    response = await fetch(API + path, { headers: { "Content-Type": "application/json" }, ...options });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(0, String(error.message || error));
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new ApiError(response.status, detail, response.headers.get("Retry-After"));
  }
  return body;
}

/**
 * Plain-Spanish text for a failed request: what happened and what still works.
 * The API's own detail (in English) is kept as a technical line.
 */
function describeError(error, what) {
  const s = error.status;
  if (s === 0) {
    return { title: `No se ha podido contactar con el servidor para ${what}.`,
      body: "Comprueba la conexión y vuelve a intentarlo; si el servidor estaba arrancando, el segundo intento suele funcionar." };
  }
  if (s === 429 && /budget/i.test(error.detail)) {
    return { title: "Se ha alcanzado el límite diario de uso del LLM de esta demostración.",
      body: "Hasta las 00:00 UTC no están disponibles el agente ni las opciones con LLM. La búsqueda de libros sigue funcionando." };
  }
  if (s === 429) {
    const wait = error.retryAfter ? ` Espera unos ${error.retryAfter} segundos` : " Espera un minuto";
    return { title: "Demasiadas peticiones seguidas desde este equipo.",
      body: `${wait} y vuelve a intentarlo. Es un límite por minuto de la demostración pública, no un fallo.` };
  }
  if (s === 503 && /OPENROUTER_API_KEY|no LLM|needs an LLM/i.test(error.detail)) {
    return { title: "Este servidor no tiene un LLM configurado.",
      body: "Sin LLM no hay agente ni explicaciones (falta OPENROUTER_API_KEY). La búsqueda de libros funciona sin él." };
  }
  if (s === 503) {
    return { title: `El servicio necesario para ${what} no responde ahora mismo.`,
      body: "Puede ser el LLM o un índice que no está cargado. Vuelve a intentarlo en unos segundos; la búsqueda simple suele seguir disponible." };
  }
  if (s === 400 || s === 422) {
    return { title: "La petición no es válida.",
      body: "Revisa el texto y los filtros (por ejemplo, que el año inicial no sea mayor que el final, o que la pregunta tenga al menos 3 caracteres)." };
  }
  if (s === 404) {
    return { title: "No se ha encontrado ese libro en el catálogo.", body: "Puede que el catálogo haya cambiado; repite la búsqueda." };
  }
  return { title: `Ha fallado la petición para ${what} (error ${s}).`, body: "Vuelve a intentarlo en unos segundos." };
}

function showMessage(id, error, what, overrides = {}) {
  const el = $(id);
  const { title, body } = overrides[error.status] || describeError(error, what);
  const strong = document.createElement("p");
  strong.className = "message-title";
  strong.textContent = title;
  const p = document.createElement("p");
  p.textContent = body;
  el.replaceChildren(strong, p);
  if (error.detail) {
    const tech = document.createElement("p");
    tech.className = "message-tech";
    tech.textContent = `Detalle técnico: ${error.status ? `HTTP ${error.status}, ` : ""}${error.detail}`;
    el.append(tech);
  }
  el.className = error.status === 429 || error.status === 503 ? "message notice" : "message error";
  el.hidden = false;
}

// ---------------------------------------------------------------------------
// Waiting state: text that changes with time, a cancel button, form usable
// ---------------------------------------------------------------------------

/**
 * Show a waiting state that says, in plain words, how long the request may
 * take. Returns {signal, stop}: abort signal for fetch and a stop function.
 */
function startWaiting(waitId, buttonId, steps, { keepFocus = false } = {}) {
  const box = $(waitId);
  const main = box.querySelector(".wait-main");
  const hint = box.querySelector(".wait-hint");
  const button = $(buttonId);
  const controller = new AbortController();
  const started = Date.now();

  main.textContent = steps[0].text;
  hint.textContent = steps[0].hint || "";
  box.hidden = false;
  // keepFocus: the button stays focusable (aria-disabled), so focus is not lost
  // when the user sent with it; the caller ignores clicks while waiting.
  if (keepFocus) button.setAttribute("aria-disabled", "true");
  else button.disabled = true;
  button.setAttribute("aria-busy", "true");

  let shown = 0;
  const timer = setInterval(() => {
    const seconds = (Date.now() - started) / 1000;
    const next = steps.findLastIndex((step) => seconds >= (step.after || 0));
    if (next !== shown) {
      shown = next;
      main.textContent = steps[next].text;
      hint.textContent = steps[next].hint || "";
    }
  }, 1000);

  const stop = () => {
    clearInterval(timer);
    box.hidden = true;
    button.disabled = false;
    button.removeAttribute("aria-disabled");
    button.removeAttribute("aria-busy");
  };
  box.querySelector(".link-button").onclick = () => controller.abort();
  return { signal: controller.signal, stop, abort: () => controller.abort() };
}

// ---------------------------------------------------------------------------
// Rendering a book
// ---------------------------------------------------------------------------

// A link built with DOM properties (never HTML); only http(s) URLs are linked.
function link(text, url) {
  if (!url || !/^https?:\/\//.test(url)) return document.createTextNode(text);
  const a = document.createElement("a");
  a.href = url;
  a.textContent = text;
  a.rel = "noopener";
  a.target = "_blank";
  return a;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Where the description and the record come from, with the licence and the
// link each source asks for (Wikipedia text is CC BY-SA: article and licence).
function renderAttribution(book) {
  const parts = [];
  const desc = book.description_source;
  if (desc) {
    const name = SOURCE_NAMES[desc.source] || desc.source;
    const label = desc.source === "wikipedia" ? `Extracto del artículo de ${name}` : `Descripción: ${name}`;
    parts.push(link(desc.language ? `${label} (${desc.language})` : label, desc.attribution_url));
    if (desc.license) {
      parts.push(document.createTextNode(", "));
      parts.push(link(desc.license, LICENSE_URLS[desc.license]));
    }
    if (desc.auto_generated) parts.push(document.createTextNode(", resumen generado automáticamente"));
  }
  const records = (book.provenance || []).filter((r) => r.source !== "wikipedia");
  if (records.length) {
    parts.push(document.createTextNode(parts.length ? ". Ficha: " : "Ficha: "));
    records.forEach((record, i) => {
      if (i) parts.push(document.createTextNode(", "));
      const name = SOURCE_NAMES[record.source] || record.source;
      parts.push(link(record.source === "doab" ? name : `${name} ${record.record_id}`, record.url));
    });
  }
  if (!parts.length) parts.push(document.createTextNode(`Fuente: ${SOURCE_NAMES[book.source] || book.source}`));
  return parts;
}

function bookYear(book) {
  return book.published_date ? book.published_date.slice(0, 4) : null;
}

function languageName(code) {
  return code ? LANGUAGE_NAMES[code] || code : null;
}

/** Year, language and categories as separate short items. */
function bookFacts(book) {
  const items = [];
  const year = bookYear(book);
  items.push(year || "sin fecha");
  if (book.language) items.push(languageName(book.language));
  if (book.categories && book.categories.length) items.push(book.categories.slice(0, 3).join(", "));
  return items;
}

function fillFacts(container, book) {
  container.replaceChildren(...bookFacts(book).map((text) => el("span", "fact", text)));
}

function shorten(text, max) {
  if (!text) return "";
  return text.length > max ? text.slice(0, max - 1).trimEnd() + "…" : text;
}

function ordinal(n) {
  return `${n}.º`;
}

/**
 * Position of each result in each stage.
 * The API returns the true positions (lexical_rank, vector_rank, rerank_rank:
 * the place of the book in each stage's full list). A server without them
 * only gives the stage scores; then the positions are computed here among
 * the results shown and the page says they are approximate.
 */
function stageRanks(results) {
  const ranksBy = (key) => {
    const scored = results.filter((r) => r[key] !== null && r[key] !== undefined);
    scored.sort((a, b) => b[key] - a[key]);
    return new Map(scored.map((r, i) => [r.book.id, i + 1]));
  };
  const given = (key) => new Map(results.filter((r) => isValue(r[key])).map((r) => [r.book.id, r[key]]));
  const exact = results.some((r) => "lexical_rank" in r);
  return {
    exact,
    lexical: exact ? given("lexical_rank") : ranksBy("lexical_score"),
    vector: exact ? given("vector_rank") : ranksBy("vector_score"),
    rerank: exact ? given("rerank_rank") : ranksBy("rerank_score"),
    anyRerank: results.some((r) => isValue(r.rerank_score) || isValue(r.rerank_rank)),
    total: results.length,
  };
}

function isValue(value) {
  return value !== null && value !== undefined;
}

function trailStep(name, detail, value, found) {
  const li = el("li", found ? "step found" : "step missed");
  li.append(el("span", "step-name", name));
  if (detail) li.append(el("span", "step-detail", detail));
  li.append(el("span", "step-value", value));
  return li;
}

function formatRaw(value, digits = 4) {
  return value === null || value === undefined ? "sin valor" : value.toFixed(digits);
}

/** The why-row of a search result: which stages found it and where it ranked. */
function renderWhy(node, result, ranks, mode, searchMode) {
  const trail = node.querySelector(".trail");
  const raw = node.querySelector(".raw-values");
  const id = result.book.id;
  const pairs = [];

  if (mode === "similar") {
    node.querySelector(".why-label").textContent = "Por qué aparece";
    trail.append(trailStep("Similitud de significado", "con el libro de partida",
      result.vector_score === null ? "sin valor" : `${Math.round(result.vector_score * 100)} %`, true));
    trail.append(trailStep("Puesto", null, ordinal(result.rank), true));
    pairs.push(["Similitud coseno (vector_score)", formatRaw(result.vector_score)]);
  } else {
    const lex = ranks.lexical.get(id);
    const vec = ranks.vector.get(id);
    const rr = ranks.rerank.get(id);
    trail.append(trailStep("Palabras clave", "BM25",
      lex ? ordinal(lex) : "no lo encontró", Boolean(lex)));
    trail.append(trailStep("Significado", "embeddings",
      searchMode === "lexical_only" ? "no se usó" : vec ? ordinal(vec) : "no lo encontró", Boolean(vec)));
    let rerankText = "no se usó";
    if (ranks.anyRerank) rerankText = rr ? ordinal(rr) : "fuera de su lista";
    trail.append(trailStep("Reordenador", null, rerankText, Boolean(rr)));
    trail.append(trailStep("Resultado", null, ordinal(result.rank), true));
    pairs.push(
      ["Origen (source)", { lexical: "solo palabras clave", vector: "solo significado", hybrid: "fusión RRF de las dos listas" }[result.source] || result.source],
      ["BM25 (lexical_score)", formatRaw(result.lexical_score)],
      ["Similitud coseno (vector_score)", formatRaw(result.vector_score)],
      ["Reordenador (rerank_score)", formatRaw(result.rerank_score)],
      ["Fusión RRF (final_score)", formatRaw(result.final_score, 6)],
      ["Puesto final (rank)", String(result.rank)],
    );
    if (ranks.exact) {
      pairs.push(
        ["Puesto en BM25 (lexical_rank)", isValue(result.lexical_rank) ? String(result.lexical_rank) : "sin valor"],
        ["Puesto por significado (vector_rank)", isValue(result.vector_rank) ? String(result.vector_rank) : "sin valor"],
        ["Puesto tras reordenar (rerank_rank)", isValue(result.rerank_rank) ? String(result.rerank_rank) : "sin valor"],
      );
      node.querySelector(".raw").append(el("p", "field-note",
        "Cada puesto por etapa es la posición del libro en la lista completa de esa etapa, antes de quedarse con los resultados mostrados."));
    } else {
      node.querySelector(".raw").append(el("p", "field-note",
        `Puestos aproximados: este servidor no da la posición en cada etapa, así que se calcula entre los ${ranks.total} resultados mostrados, a partir de estas puntuaciones.`));
    }
  }
  for (const [label, value] of pairs) raw.append(el("dt", null, label), el("dd", null, value));
}

/**
 * The box under a book with a reason and its verified citations: the LLM
 * explanation of a search result, or why the assistant recommends the book.
 * The server returns the text without the model's inline citation markers.
 */
function fillReasonBox(node, { label, text, citations, warning, note }) {
  const box = node.querySelector(".explanation");
  box.hidden = false;
  box.classList.toggle("ungrounded", Boolean(warning));
  box.querySelector(".explanation-label").textContent = label;
  box.querySelector(".explanation-text").textContent = text;
  const list = box.querySelector(".citations");
  list.replaceChildren();
  for (const citation of citations) {
    const item = document.createElement("li");
    item.append(el("span", "field", FIELD_NAMES[citation.field] || citation.field),
      document.createTextNode(`«${citation.snippet}»`));
    list.append(item);
  }
  box.querySelector(".explanation-model").textContent = note;
}

function renderExplanation(node, explanation) {
  if (!explanation) return;
  fillReasonBox(node, {
    label: "Por qué encaja, según el LLM",
    text: explanation.text,
    citations: explanation.citations,
    warning: !explanation.grounded,
    note: explanation.grounded
      ? `Citas comprobadas contra los datos del libro. Modelo: ${explanation.model}`
      : `Ninguna cita se pudo comprobar en los datos del libro: léase con cautela. Modelo: ${explanation.model}`,
  });
}

function renderResult(result, ranks, mode, searchMode) {
  const node = $("result-template").content.firstElementChild.cloneNode(true);
  const book = result.book;
  node.querySelector(".rank").textContent = result.rank;
  const title = node.querySelector(".title-button");
  title.textContent = book.title;
  title.setAttribute("aria-label", `${result.rank}. ${book.title}: ver ficha`);
  title.addEventListener("click", () => openBook(book));
  node.querySelector(".authors").textContent = book.authors.length ? book.authors.join(", ") : "Autor desconocido";
  fillFacts(node.querySelector(".facts"), book);
  node.querySelector(".description").textContent = shorten(book.description, 300);
  node.querySelector(".attribution").replaceChildren(...renderAttribution(book));
  renderExplanation(node, result.explanation);
  // A book recommended by the agent was not ranked by a search: no why-row.
  if (mode === "agent") node.querySelector(".why").remove();
  else renderWhy(node, result, ranks, mode, searchMode);
  const similar = node.querySelector(".similar");
  similar.setAttribute("aria-label", `Libros similares a ${book.title}`);
  similar.addEventListener("click", () => showSimilar(book));
  return node;
}

// ---------------------------------------------------------------------------
// Book detail dialog (shared by search results and the assistant's books)
// ---------------------------------------------------------------------------

let dialogBook = null;

function openBook(book) {
  dialogBook = book;
  const dialog = $("book-dialog");
  $("dialog-title").textContent = book.title;
  dialog.querySelector(".dialog-authors").textContent = book.authors.length ? book.authors.join(", ") : "Autor desconocido";
  const facts = dialog.querySelector(".dialog-facts");
  facts.replaceChildren();
  const rows = [
    ["Año", bookYear(book) || "sin fecha"],
    ["Idioma", languageName(book.language) || "desconocido"],
    ["Categorías", book.categories.length ? book.categories.join(", ") : "sin categoría"],
  ];
  if (book.metadata && book.metadata.publisher) rows.push(["Editorial", book.metadata.publisher]);
  if (book.metadata && book.metadata.isbn) rows.push(["ISBN", book.metadata.isbn]);
  if (book.metadata && book.metadata.page_count) rows.push(["Páginas", String(book.metadata.page_count)]);
  for (const [label, value] of rows) facts.append(el("dt", null, label), el("dd", null, value));
  dialog.querySelector(".dialog-description").textContent = book.description || "Este libro no tiene descripción en el catálogo.";
  dialog.querySelector(".dialog-attribution").replaceChildren(...renderAttribution(book));
  dialog.showModal();
}

function setupDialog() {
  const dialog = $("book-dialog");
  $("dialog-close").addEventListener("click", () => dialog.close());
  // Click on the backdrop closes it too.
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  $("dialog-similar").addEventListener("click", () => {
    dialog.close();
    selectTab("search");
    showSimilar(dialogBook);
  });
}

// ---------------------------------------------------------------------------
// Search mode
// ---------------------------------------------------------------------------

function intOrNull(value) {
  const n = parseInt(value, 10);
  return Number.isFinite(n) ? n : null;
}

function buildRequest() {
  const filters = {
    language: $("language").value || null,
    category: $("category").value.trim() || null,
    min_year: intOrNull($("min-year").value),
    max_year: intOrNull($("max-year").value),
  };
  return {
    text: $("query").value.trim(),
    filters,
    max_results: intOrNull($("max-results").value) || 10,
    use_understanding: $("use-understanding").checked,
    use_explanations: $("use-explanations").checked,
    use_diversification: $("use-diversification").checked,
    include_metadata: true,
  };
}

function setStatus(text) {
  $("status").textContent = text;
}

/** Plain-Spanish lines for the notice codes of a response (texts: first table that has the code). */
function noticeLines(response, ...tables) {
  const codes = response.notices || [];
  const lines = codes.map((code) => tables.map((t) => t[code]).find(Boolean) || code);
  // A degraded search without a budget notice (e.g. the vector index failed).
  if (response.degraded && !codes.includes("search_budget_spent")) {
    lines.push("La búsqueda por significado no estaba disponible y se ha usado solo la búsqueda por palabras clave "
      + "(BM25). Los resultados son válidos, pero no tienen en cuenta sinónimos ni otros idiomas.");
  }
  return lines;
}

function showNotices(response) {
  const box = $("notice");
  const lines = noticeLines(response, NOTICE_TEXT);
  box.replaceChildren(...lines.map((line) => el("p", null, line)));
  if (response.degraded && response.degradation_reason) {
    box.append(el("p", "message-tech", `Detalle técnico: ${response.degradation_reason}`));
  }
  box.hidden = lines.length === 0;
}

function renderIntent(intent) {
  const section = $("intent");
  section.replaceChildren();
  if (!intent) {
    section.hidden = true;
    return;
  }
  const filters = Object.entries(intent.extracted_filters)
    .filter(([, value]) => value !== null)
    .map(([key, value]) => `${FILTER_NAMES[key] || key}: ${key === "language" ? languageName(value) : value}`)
    .join("; ") || "ninguno";
  const rows = [
    ["Intención", `${INTENT_NAMES[intent.intent_type] || intent.intent_type} (confianza ${Math.round(intent.confidence * 100)} %)`],
    ["Consulta reformulada", intent.reformulated_query],
    ["Variantes buscadas", intent.query_variations.join(" | ") || "ninguna"],
    ["Filtros deducidos", filters],
    ["Razonamiento", intent.reasoning],
  ];
  section.append(el("h2", "section-title", "Cómo ha entendido el LLM la consulta"));
  const list = document.createElement("dl");
  for (const [label, value] of rows) list.append(el("dt", null, label), el("dd", null, value));
  section.append(list);
  section.hidden = false;
}

function showResults(results, mode, searchMode) {
  const ranks = stageRanks(results);
  $("results").replaceChildren(...results.map((r) => renderResult(r, ranks, mode, searchMode)));
  $("empty-search").hidden = true;
  $("results").removeAttribute("aria-busy");
}

function searchStatus(response) {
  const n = response.results.length;
  if (n === 0) return "No hay resultados. Prueba con otras palabras, quita filtros o busca en otro idioma.";
  const parts = [`${n} ${n === 1 ? "resultado" : "resultados"} en ${(response.latency_ms / 1000).toFixed(1)} s`];
  const m = response.metadata;
  if (m) parts.push(`candidatos: ${m.candidates_lexical} por palabras clave y ${m.candidates_vector} por significado`);
  if (response.search_mode === "lexical_only") parts.push("solo palabras clave");
  if (m && m.diversification_enabled) parts.push("diversificados con MMR");
  return parts.join(", ") + ".";
}

async function runSearch(event) {
  event.preventDefault();
  const request = buildRequest();
  if (!request.text) return;

  const steps = [{ text: "Buscando…", hint: "" }, { after: 3, text: "Buscando… sigue en marcha.", hint: COLD_START_HINT }];
  if (request.use_understanding) {
    steps[0].text = "Interpretando la consulta con el LLM y buscando…";
    steps[0].hint = "El LLM añade unos segundos.";
  }
  if (request.use_explanations) {
    steps.push({ after: 8, text: "Generando las explicaciones con el LLM…",
      hint: "Las explicaciones tardan más que la búsqueda: entre 10 y 40 segundos. " + COLD_START_HINT });
  }
  $("search-message").hidden = true;
  $("notice").hidden = true;
  $("similar-banner").hidden = true;
  reserveSpace("search-output");
  $("empty-search").hidden = true;
  renderIntent(null);
  $("results").setAttribute("aria-busy", "true");
  $("results").classList.add("stale");
  setStatus("");
  const wait = startWaiting("search-wait", "search-button", steps);

  try {
    const response = await callApi("/search", { method: "POST", body: JSON.stringify(request), signal: wait.signal });
    lastSearch = response;
    showNotices(response);
    renderIntent(response.query_intent);
    showResults(response.results, "search", response.search_mode);
    setStatus(searchStatus(response));
  } catch (error) {
    if (error.name === "AbortError") setStatus("Búsqueda cancelada.");
    else showMessage("search-message", error, "buscar");
  } finally {
    $("results").classList.remove("stale");
    $("results").removeAttribute("aria-busy");
    wait.stop();
  }
}

async function showSimilar(book) {
  $("search-message").hidden = true;
  reserveSpace("search-output");
  $("results").classList.add("stale");
  const wait = startWaiting("search-wait", "search-button",
    [{ text: `Buscando libros similares a «${book.title}»…` }, { after: 3, text: "Sigue en marcha.", hint: COLD_START_HINT }]);
  try {
    const max = intOrNull($("max-results").value) || 10;
    const response = await callApi(`/books/${book.id}/similar?max_results=${max}`, { signal: wait.signal });
    const banner = $("similar-banner");
    banner.querySelector(".banner-text").textContent =
      `Libros parecidos por significado a «${book.title}». No hay consulta: se compara el vector de ese libro con el de los demás.`;
    $("similar-back").hidden = !lastSearch;
    banner.hidden = false;
    renderIntent(null);
    $("notice").hidden = true;
    showResults(response.results, "similar", "vector_only");
    setStatus(`${response.results.length} libros similares.`);
    $("panel-search").scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
    banner.querySelector(".link-button").focus({ preventScroll: true });
  } catch (error) {
    if (error.name !== "AbortError") showMessage("search-message", error, "buscar libros similares");
  } finally {
    $("results").classList.remove("stale");
    wait.stop();
  }
}

function backToSearch() {
  if (!lastSearch) return;
  $("similar-banner").hidden = true;
  showNotices(lastSearch);
  renderIntent(lastSearch.query_intent);
  showResults(lastSearch.results, "search", lastSearch.search_mode);
  setStatus(searchStatus(lastSearch));
  $("query").focus();
}

// ---------------------------------------------------------------------------
// Assistant mode (POST /assistant/chat, the LangGraph assistant)
// ---------------------------------------------------------------------------

// The server keeps nothing: the page holds the conversation in this variable
// only (no browser storage, no cookie) and sends all of it with each message.
// Each assistant turn carries the `route` of the reply that produced it,
// because the server counts the clarifying questions already asked from the
// routes it receives (an assistant turn without one counts as a question).
let conversation = [];
let chatWait = null;
// Bumped by "Nueva conversación": a reply to an older conversation is ignored.
let chatGeneration = 0;

// Limits of the request (app/api/v1/schemas.py: at most 50 messages of up to
// 5000 characters) and of what the assistant reads of each message
// (app/domain/value_objects.py: the first 500 characters), so the box stops
// at 500 and the conversation at 50 messages.
const MAX_CHAT_MESSAGES = 50;
const MAX_CHAT_CHARS = 500;
const MAX_CLARIFYING_TURNS = 2;

const ROUTE_NAMES = {
  clarify: "Pregunta aclaratoria", search: "Búsqueda", agent: "Agente", out_of_scope: "Fuera de ámbito",
};
// Graph nodes in the order they can run (a failed agent goes on to retrieve).
const NODE_ORDER = ["route", "clarify", "agent", "retrieve", "answer", "verify", "out_of_scope"];
const NODE_NAMES = {
  route: "Enrutar", clarify: "Preguntar", agent: "Agente", retrieve: "Buscar",
  answer: "Responder", verify: "Verificar", out_of_scope: "Fuera de ámbito",
};
// Why code overrode the model's route (forced_route).
const FORCED_ROUTE_TEXT = {
  clarification_cap: "El modelo quería hacer otra pregunta, pero ya se habían hecho las dos permitidas: "
    + "el código ha pasado a buscar con lo que se sabe.",
  invalid_clarification: "El modelo quería preguntar, pero sin pregunta o con menos de dos respuestas sugeridas: "
    + "el código ha pasado a buscar.",
  reply_to_clarification: "El modelo marcó tu respuesta a su pregunta como fuera de ámbito, pero no hablaba de otra "
    + "cosa: el código ha buscado con lo que se sabe.",
  invalid_router_output: "La llamada de enrutado falló o devolvió una salida no válida: el código ha buscado tu último mensaje.",
  agent_failed: "El agente falló: el código ha pasado al camino de búsqueda.",
};
// Errors of this endpoint, in its own terms (see assistant_endpoints.py).
const ASSISTANT_ERRORS = {
  400: { title: "El servidor no ha aceptado la conversación.",
    body: "La conversación debe terminar con un mensaje tuyo. Empieza una nueva conversación si el problema se repite." },
  422: { title: "El mensaje no es válido.",
    body: `Cada mensaje puede tener hasta ${MAX_CHAT_CHARS} caracteres y la conversación hasta ${MAX_CHAT_MESSAGES} mensajes. Tu texto sigue en el cuadro.` },
  503: { title: "La búsqueda no está disponible ahora mismo, así que el asistente no puede responder.",
    body: "Suele ser un índice que aún no ha cargado o un servicio externo que no responde. Vuelve a intentarlo en unos segundos; tu texto sigue en el cuadro." },
};

function formatMs(ms) {
  if (!isValue(ms)) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
}

/** The nodes the request went through, in graph order (from the per-node timings). */
function routePath(response) {
  return NODE_ORDER.filter((node) => node in (response.timings_ms || {}));
}

function usedLlm(response) {
  return !(response.notices || []).some((code) => code === "llm_unavailable" || code === "llm_budget_spent");
}

/** One tool call of the agent: tool, arguments in plain words, books it returned. */
function renderStep(step, titles) {
  const li = el("li", "agent-step");
  li.append(el("p", "tool-name", TOOL_NAMES[step.tool] || step.tool));
  const args = el("dl", "tool-args");
  for (const [key, value] of Object.entries(step.arguments || {})) {
    if (value === null || value === undefined || value === "") continue;
    let shown = typeof value === "object" ? JSON.stringify(value) : String(value);
    if (key === "book_id") shown = titles.get(value) ? `«${titles.get(value)}»` : `id ${shown.slice(0, 8)}…`;
    if (key === "language") shown = languageName(value) || shown;
    if (key === "query") shown = `«${shown}»`;
    args.append(el("dt", null, ARGUMENT_NAMES[key] || key), el("dd", null, shown));
  }
  if (args.children.length) li.append(args);
  const books = step.books || [];
  const count = step.result_count === 0 ? "No devolvió ningún libro."
    : step.result_count === 1 ? "Devolvió 1 libro:" : `Devolvió ${step.result_count} libros:`;
  li.append(el("p", "tool-count", books.length ? count : count.replace(/:$/, ".")));
  if (books.length) {
    const list = el("ol", "tool-books");
    list.append(...books.map((book) => el("li", null, book.title || `id ${book.book_id.slice(0, 8)}…`)));
    li.append(list);
  }
  const raw = document.createElement("details");
  raw.className = "raw";
  raw.append(el("summary", null, "Llamada exacta"), el("pre", "raw-call",
    `${step.tool}(${JSON.stringify(step.arguments, null, 1).replace(/\n\s*/g, " ")})`));
  li.append(raw);
  return li;
}

function renderUserTurn(text) {
  const li = el("li", "turn user");
  li.append(el("p", "turn-who", "Tú"), el("p", "turn-text", text));
  return li;
}

/** Suggested answers of a clarifying question: each one is sent as the next message. */
function renderOptions(options) {
  const group = el("div", "options-group");
  group.setAttribute("role", "group");
  group.setAttribute("aria-label", "Respuestas sugeridas");
  group.append(el("p", "options-label", "Respuestas sugeridas (o escribe la tuya en el cuadro):"));
  const list = el("ul", "option-list");
  for (const option of options) {
    const li = document.createElement("li");
    const button = el("button", "option", option);
    button.type = "button";
    button.addEventListener("click", () => {
      if (chatWait || button.disabled) return;
      button.classList.add("chosen");
      sendChat(option);
    });
    li.append(button);
    list.append(li);
  }
  group.append(list);
  return group;
}

/** Suggested answers can only be chosen while their question is the last reply. */
function setOptionsEnabled(enabled) {
  const turns = $("chat-thread").querySelectorAll(".turn.assistant");
  turns.forEach((turn, i) => {
    const current = enabled && i === turns.length - 1;
    for (const button of turn.querySelectorAll(".option")) {
      button.disabled = !current;
      if (current) button.classList.remove("chosen");
    }
  });
}

/** Recommended books first (reason and citations), then the rest of the search results. */
function renderAssistantBooks(response) {
  const box = el("div", "assistant-books");
  const results = response.results || [];
  const searchMode = response.degraded ? "lexical_only" : "hybrid";
  const ranks = stageRanks(results);
  const byId = new Map(results.map((r) => [r.book.id, r]));
  const recommended = new Set();

  if (response.recommendations.length) {
    const n = response.recommendations.length;
    box.append(el("h3", "section-title", n === 1 ? "Libro que recomienda" : `Libros que recomienda (${n})`));
    const list = el("ol", "results recs");
    response.recommendations.forEach((rec, i) => {
      recommended.add(rec.book.id);
      const result = byId.get(rec.book.id);
      const node = result
        ? renderResult(result, ranks, "search", searchMode)
        : renderResult({ book: rec.book, rank: i + 1 }, null, "agent", null);
      const verified = rec.citations.length > 0;
      fillReasonBox(node, {
        label: "Por qué lo recomienda el asistente",
        text: rec.reason,
        citations: rec.citations,
        warning: false,
        note: verified
          ? "Citas comprobadas contra los datos del libro."
          : "El agente solo puede recomendar libros que le ha devuelto una de sus herramientas.",
      });
      list.append(node);
    });
    box.append(list);
  }

  const rest = results.filter((r) => !recommended.has(r.book.id));
  if (rest.length) {
    const details = el("details", "more-results");
    // Without recommendations the results are the answer: shown open.
    details.open = !response.recommendations.length;
    details.append(el("summary", null, response.recommendations.length
      ? `Resto de resultados de la búsqueda (${rest.length})` : `Resultados de la búsqueda (${rest.length})`));
    const list = el("ol", "results");
    list.append(...rest.map((r) => renderResult(r, ranks, "search", searchMode)));
    details.append(list);
    box.append(details);
  }
  return box;
}

function filtersText(filters) {
  if (!filters) return "ninguno";
  return Object.entries(filters)
    .filter(([, value]) => isValue(value))
    .map(([key, value]) => `${FILTER_NAMES[key] || key}: ${key === "language" ? languageName(value) : value}`)
    .join("; ") || "ninguno";
}

/**
 * How LangGraph handled the request: the path through the graph with the time
 * of each node, a forced route and why, the condensed query and filters, the
 * agent's steps and the LLM usage. Collapsed by default.
 */
function renderRouteDetail(response) {
  const details = el("details", "route-detail");
  const path = routePath(response);
  const llm = usedLlm(response);
  const summary = document.createElement("summary");
  summary.append(el("span", "route-badge", llm ? ROUTE_NAMES[response.route] || response.route : "Búsqueda sin LLM"));
  const line = (llm ? path : ["retrieve"]).map((node) => NODE_NAMES[node] || node).join(" → ");
  summary.append(el("span", "route-line", `${line} · ${formatMs(response.latency_ms)}`));
  if (response.forced_route) summary.append(el("span", "route-forced", "ruta corregida por el código"));
  details.append(summary);

  const body = el("div", "route-body");
  body.append(el("p", "route-title", "Recorrido por el grafo del asistente"));
  const chain = el("ol", "path");
  chain.setAttribute("aria-label", "Nodos por los que pasó la petición, en orden");
  for (const node of path) {
    const li = el("li", "path-node");
    li.append(el("span", "path-name", NODE_NAMES[node] || node), el("code", "path-id", node),
      el("span", "path-time", formatMs(response.timings_ms[node])));
    chain.append(li);
  }
  body.append(chain);

  const facts = el("dl", "route-facts");
  const add = (label, value) => facts.append(el("dt", null, label), el("dd", null, value));
  if (!llm) {
    add("Por qué", "Sin LLM no hay enrutado ni respuesta redactada: el endpoint busca el último mensaje tal cual.");
  }
  if (response.forced_route) {
    add("Ruta corregida", `${FORCED_ROUTE_TEXT[response.forced_route] || response.forced_route} (${response.forced_route})`);
  }
  if (response.search_query) {
    const label = response.route === "agent" ? "Petición para el agente" : llm ? "Consulta condensada" : "Consulta buscada";
    add(label, `«${response.search_query}»`);
  }
  if (response.route === "search") {
    add("Filtros", filtersText(response.filters));
    add("Resultados", `${response.results.length}${response.degraded ? " (solo palabras clave)" : ""}`);
  }
  add("Preguntas aclaratorias", `${response.clarifications} de ${MAX_CLARIFYING_TURNS} en esta conversación`);
  const usage = response.usage || {};
  if (usage.llm_calls) {
    const parts = [`${usage.llm_calls} ${usage.llm_calls === 1 ? "llamada" : "llamadas"}`];
    if (usage.failed_calls) parts.push(`${usage.failed_calls} fallida${usage.failed_calls === 1 ? "" : "s"}`);
    if (isValue(usage.input_tokens)) parts.push(`${usage.input_tokens} tokens de entrada y ${usage.output_tokens ?? 0} de salida`);
    if (isValue(usage.cost_usd)) parts.push(`${usage.cost_usd.toFixed(4)} USD`);
    add("LLM", `${response.model || "modelo desconocido"}: ${parts.join(", ")}`);
  }
  body.append(facts);

  if (response.steps.length) {
    body.append(el("p", "route-title", "Pasos del agente"));
    body.append(el("p", "field-note", "Llamadas a herramientas en el orden en que el agente las hizo (traza ReAct)."));
    const titles = new Map();
    for (const step of response.steps) for (const b of step.books || []) titles.set(b.book_id, b.title);
    for (const rec of response.recommendations) titles.set(rec.book.id, rec.book.title);
    const steps = el("ol", "steps");
    steps.append(...response.steps.map((step) => renderStep(step, titles)));
    const verify = el("li", "agent-step verify");
    verify.append(el("p", "tool-name", "Verificación"),
      el("p", "tool-count", "Cada libro recomendado se ha comprobado contra lo que devolvieron las herramientas."));
    steps.append(verify);
    body.append(steps);
  }
  details.append(body);
  return details;
}

function renderAssistantTurn(response) {
  const li = el("li", "turn assistant");
  li.dataset.route = response.route;
  li.append(el("p", "turn-who", "Asistente"), el("p", "turn-text", response.message));
  if (response.route === "clarify" && response.options.length) li.append(renderOptions(response.options));
  // The routing line goes right under the reply, before any list of books.
  li.append(renderRouteDetail(response));
  // "No LLM on this server" is already said once above the conversation.
  const shown = { ...response, notices: (response.notices || []).filter(
    (code) => !(code === "llm_unavailable" && !$("assistant-off-note").hidden)) };
  const lines = noticeLines(shown, ASSISTANT_NOTICE_TEXT, NOTICE_TEXT);
  if (lines.length) {
    const box = el("div", "message notice");
    box.append(...lines.map((line) => el("p", null, line)));
    li.append(box);
  }
  if (response.recommendations.length || response.results.length) li.append(renderAssistantBooks(response));
  return li;
}

/** Read the new reply to assistive technology without moving focus. */
function announce(text) {
  const live = $("chat-live");
  live.textContent = "";
  setTimeout(() => { live.textContent = text; }, 50);
}

function replySummary(response) {
  const parts = [`El asistente responde: ${response.message}`];
  if (response.route === "clarify" && response.options.length) parts.push(`Respuestas sugeridas: ${response.options.join("; ")}.`);
  const n = response.recommendations.length;
  if (n) parts.push(n === 1 ? "Recomienda 1 libro." : `Recomienda ${n} libros.`);
  else if (response.results.length) parts.push(`${response.results.length} resultados de búsqueda.`);
  return parts.join(" ");
}

/** Bring the start of a new reply into view if it is below the screen (focus stays put). */
function revealTurn(node) {
  const top = node.getBoundingClientRect().top;
  if (top > window.innerHeight * 0.75 || top < 0) {
    node.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }
}

function updateChatCount() {
  const full = conversation.length >= MAX_CHAT_MESSAGES;
  $("chat-count").textContent = full
    ? `La conversación ha llegado al máximo de ${MAX_CHAT_MESSAGES} mensajes que acepta el servidor: empieza una nueva.`
    : `Hasta ${MAX_CHAT_CHARS} caracteres por mensaje. Mensajes en la conversación: ${conversation.length} de ${MAX_CHAT_MESSAGES}.`;
  $("chat-count").classList.toggle("full", full);
}

async function sendChat(rawText) {
  const text = rawText.trim().slice(0, MAX_CHAT_CHARS);
  if (!text || chatWait) return;
  if (conversation.length + 1 > MAX_CHAT_MESSAGES) {
    updateChatCount();
    announce($("chat-count").textContent);
    return;
  }
  const generation = chatGeneration;
  const hadFocusOnOption = document.activeElement && document.activeElement.classList.contains("option");
  $("assistant-message").hidden = true;
  conversation.push({ role: "user", content: text });
  const userNode = renderUserTurn(text);
  $("chat-thread").append(userNode);
  $("empty-assistant").hidden = true;
  $("chat-input").value = "";
  setOptionsEnabled(false);
  // A chosen option is now disabled: keep focus in the message box, not on <body>.
  if (hadFocusOnOption) $("chat-input").focus({ preventScroll: true });
  updateChatCount();

  chatWait = startWaiting("assistant-wait", "chat-send", [
    { text: "El asistente está pensando…", hint: "Primero decide el camino: una pregunta, una búsqueda o el agente." },
    { after: 6, text: "El asistente sigue trabajando…",
      hint: "Si ha elegido al agente puede tardar entre 15 y 60 segundos: busca varias veces y comprueba sus recomendaciones." },
    { after: 25, text: "Sigue en marcha…", hint: COLD_START_HINT },
  ], { keepFocus: true });

  try {
    const response = await callApi("/assistant/chat", {
      method: "POST", body: JSON.stringify({ messages: conversation }), signal: chatWait.signal,
    });
    if (generation !== chatGeneration) return;
    conversation.push({ role: "assistant", content: response.message || ROUTE_NAMES[response.route], route: response.route });
    const node = renderAssistantTurn(response);
    $("chat-thread").append(node);
    announce(replySummary(response));
    revealTurn(node);
  } catch (error) {
    if (generation !== chatGeneration) return;
    // The message got no reply: take it back, so the conversation is as it
    // was, and return its text to the box to send again.
    conversation.pop();
    userNode.remove();
    setOptionsEnabled(true);
    $("chat-input").value = text;
    $("empty-assistant").hidden = conversation.length > 0;
    if (error.name === "AbortError") {
      announce("Mensaje cancelado. El texto ha vuelto al cuadro.");
      $("chat-input").focus();
    } else {
      showMessage("assistant-message", error, "responder", ASSISTANT_ERRORS);
    }
  } finally {
    if (generation === chatGeneration) {
      chatWait.stop();
      chatWait = null;
      updateChatCount();
    }
  }
}

function newConversation() {
  chatGeneration += 1;
  if (chatWait) {
    chatWait.abort();
    chatWait.stop();
    chatWait = null;
  }
  conversation = [];
  $("chat-thread").replaceChildren();
  $("assistant-message").hidden = true;
  $("empty-assistant").hidden = false;
  $("chat-input").value = "";
  updateChatCount();
  announce("Conversación nueva. La anterior se ha borrado de la página.");
  $("chat-input").focus();
}

function setupAssistant() {
  renderExamples("assistant-examples", ASSISTANT_EXAMPLES, (text) => {
    // An opener starts the conversation; later it only fills the box.
    if (!conversation.length && !chatWait) sendChat(text);
    else {
      $("chat-input").value = text;
      $("chat-input").focus();
    }
  });
  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    sendChat($("chat-input").value);
  });
  // Enter sends the message; Shift+Enter adds a new line.
  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });
  $("chat-new").addEventListener("click", newConversation);
  updateChatCount();
}

// ---------------------------------------------------------------------------
// Tabs, examples, health and start-up
// ---------------------------------------------------------------------------

const TABS = ["search", "assistant"];

function selectTab(name) {
  for (const tab of TABS) {
    const active = tab === name;
    $(`tab-${tab}`).setAttribute("aria-selected", String(active));
    $(`tab-${tab}`).tabIndex = active ? 0 : -1;
    $(`panel-${tab}`).hidden = !active;
  }
}

function setupTabs() {
  const tabs = TABS.map((name) => $(`tab-${name}`));
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(TABS[i]));
    // Arrow keys move between tabs (WAI-ARIA tabs pattern).
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const last = tabs.length - 1;
      const next = event.key === "Home" ? 0 : event.key === "End" ? last
        : (i + (event.key === "ArrowRight" ? 1 : last)) % tabs.length;
      tabs[next].focus();
      selectTab(TABS[next]);
    });
  });
}

function renderExamples(listId, examples, onPick) {
  $(listId).replaceChildren(...examples.map((example) => {
    const li = document.createElement("li");
    const button = el("button", "example", example.text);
    button.type = "button";
    button.lang = example.lang;
    button.addEventListener("click", () => onPick(example.text));
    li.append(button);
    return li;
  }));
}

/**
 * Keep at least the current height (or most of a screen) for the output
 * while a request runs, so the sections below do not jump when the answer
 * arrives (no layout shift). Called right after the user's action.
 */
function reserveSpace(id) {
  const box = $(id);
  box.style.minHeight = `${Math.max(box.offsetHeight, Math.round(window.innerHeight * 0.8))}px`;
}

function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

async function loadHealth() {
  const box = $("health");
  try {
    health = await callApi("/health");
    const parts = [health.status === "ok" ? "Servidor listo" : "Servidor con avisos"];
    parts.push(health.llm.configured ? `LLM ${health.llm.model}` : "sin LLM");
    if (health.reranker && !health.reranker.configured) parts.push("sin reordenador");
    box.textContent = parts.join(", ");
    box.classList.add(health.status === "ok" ? "ok" : "degraded");

    const config = [];
    config.push(health.profile && health.profile.name ? `Perfil ${health.profile.name}.` : "");
    config.push(health.reranker && health.reranker.configured ? `Reordenador: ${health.reranker.model}.` : "Este servidor no usa reordenador.");
    config.push(health.llm.configured ? `LLM: ${health.llm.model}.` : "Este servidor no tiene LLM.");
    $("how-config").textContent = `En este servidor: ${config.filter(Boolean).join(" ")}`;

    if (!health.llm.configured) {
      for (const id of ["use-understanding", "use-explanations"]) $(id).disabled = true;
      $("llm-off-note").textContent = NO_LLM_NOTE;
      $("llm-off-note").hidden = false;
      $("assistant-off-note").textContent = "Este servidor no tiene un LLM configurado (falta OPENROUTER_API_KEY): "
        + "el asistente no puede hacer preguntas ni redactar respuestas, y cada mensaje se responde con una búsqueda "
        + "simple de su texto. La búsqueda de libros funciona igual.";
      $("assistant-off-note").hidden = false;
    }
  } catch (error) {
    box.textContent = error.status === 0 ? "No se puede contactar con el servidor" : "El servidor no responde bien";
    box.classList.add("degraded");
  }
}

function start() {
  setupTabs();
  setupDialog();
  renderExamples("search-examples", SEARCH_EXAMPLES, (text) => {
    $("query").value = text;
    $("search-form").requestSubmit();
  });
  $("search-form").addEventListener("submit", runSearch);
  $("similar-back").addEventListener("click", backToSearch);
  setupAssistant();
  loadHealth();
}

start();
