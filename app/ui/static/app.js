// Client for the book search API (no framework, no external assets).
// Every value from the API is rendered with textContent (never parsed as
// HTML), so catalog text cannot inject markup.
//
// Sections: constants, API access, waiting states and messages, rendering of
// a book (card, why-row, detail dialog), search mode, agent mode, start-up.

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
const AGENT_EXAMPLES = [
  { text: "Busco una novela clásica sobre la culpa, mejor en español", lang: "es" },
  { text: "I want to learn Python from scratch: which book should I start with?", lang: "en" },
  { text: "Quelque chose comme Frankenstein, mais plus récent", lang: "fr" },
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

function showMessage(id, error, what) {
  const el = $(id);
  const { title, body } = describeError(error, what);
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
function startWaiting(waitId, buttonId, steps) {
  const box = $(waitId);
  const main = box.querySelector(".wait-main");
  const hint = box.querySelector(".wait-hint");
  const button = $(buttonId);
  const controller = new AbortController();
  const started = Date.now();

  main.textContent = steps[0].text;
  hint.textContent = steps[0].hint || "";
  box.hidden = false;
  button.disabled = true;
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
    button.removeAttribute("aria-busy");
  };
  box.querySelector(".link-button").onclick = () => controller.abort();
  return { signal: controller.signal, stop };
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
 * Position of each result in each stage, among the results shown.
 * The API returns the stage scores, not the stage positions over the full
 * candidate lists, so positions are computed here over the visible list.
 */
function stageRanks(results) {
  const ranksBy = (key) => {
    const scored = results.filter((r) => r[key] !== null && r[key] !== undefined);
    scored.sort((a, b) => b[key] - a[key]);
    return new Map(scored.map((r, i) => [r.book.id, i + 1]));
  };
  return {
    lexical: ranksBy("lexical_score"),
    vector: ranksBy("vector_score"),
    rerank: ranksBy("rerank_score"),
    anyRerank: results.some((r) => r.rerank_score !== null && r.rerank_score !== undefined),
    total: results.length,
  };
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
      ["Origen (source)", { lexical: "solo palabras clave", vector: "solo significado", hybrid: "ambas búsquedas" }[result.source] || result.source],
      ["BM25 (lexical_score)", formatRaw(result.lexical_score)],
      ["Similitud coseno (vector_score)", formatRaw(result.vector_score)],
      ["Reordenador (rerank_score)", formatRaw(result.rerank_score)],
      ["Fusión RRF (final_score)", formatRaw(result.final_score, 6)],
      ["Puesto final (rank)", String(result.rank)],
    );
    node.querySelector(".raw").append(el("p", "field-note",
      `Los puestos por etapa se calculan entre los ${ranks.total} resultados mostrados, a partir de estas puntuaciones.`));
  }
  for (const [label, value] of pairs) raw.append(el("dt", null, label), el("dd", null, value));
}

function renderExplanation(node, explanation) {
  if (!explanation) return;
  const box = node.querySelector(".explanation");
  box.hidden = false;
  box.classList.toggle("ungrounded", !explanation.grounded);
  // The model marks its sources inline as "[CITE: field]"; the verified
  // citations are listed below, so the markers are removed from the text.
  box.querySelector(".explanation-text").textContent =
    explanation.text.replace(/\s*\[CITE:[^\]]*\]/g, "");
  const list = box.querySelector(".citations");
  for (const citation of explanation.citations) {
    const item = document.createElement("li");
    item.append(el("span", "field", FIELD_NAMES[citation.field] || citation.field),
      document.createTextNode(`«${citation.snippet}»`));
    list.append(item);
  }
  box.querySelector(".explanation-model").textContent = explanation.grounded
    ? `Citas comprobadas contra los datos del libro. Modelo: ${explanation.model}`
    : `Ninguna cita se pudo comprobar en los datos del libro: léase con cautela. Modelo: ${explanation.model}`;
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
  renderWhy(node, result, ranks, mode, searchMode);
  const similar = node.querySelector(".similar");
  similar.setAttribute("aria-label", `Libros similares a ${book.title}`);
  similar.addEventListener("click", () => showSimilar(book));
  return node;
}

// ---------------------------------------------------------------------------
// Book detail dialog (shared by search results and agent recommendations)
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

function showNotices(response) {
  const box = $("notice");
  const lines = (response.notices || []).map((code) => NOTICE_TEXT[code] || code);
  // A degraded search without a budget notice (e.g. the vector index failed).
  if (response.degraded && !(response.notices || []).includes("search_budget_spent")) {
    lines.push("La búsqueda por significado no estaba disponible y se ha usado solo la búsqueda por palabras clave "
      + "(BM25). Los resultados son válidos, pero no tienen en cuenta sinónimos ni otros idiomas.");
  }
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
// Agent mode
// ---------------------------------------------------------------------------

/** One tool call of the agent: tool, arguments in plain words, books returned. */
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
  li.append(el("p", "tool-count", step.result_count === 1 ? "Devolvió 1 libro." : `Devolvió ${step.result_count} libros.`));
  const raw = document.createElement("details");
  raw.className = "raw";
  raw.append(el("summary", null, "Llamada exacta"), el("pre", "raw-call",
    `${step.tool}(${JSON.stringify(step.arguments, null, 1).replace(/\n\s*/g, " ")})`));
  li.append(raw);
  return li;
}

function renderAgent(response) {
  $("agent-answer").textContent = response.answer;
  const meta = [`Modelo: ${response.model}`, `${(response.latency_ms / 1000).toFixed(1)} s`,
    response.steps.length === 1 ? "1 llamada a herramientas" : `${response.steps.length} llamadas a herramientas`];
  if (response.reflections) meta.push(response.reflections === 1 ? "1 nueva búsqueda tras la verificación" : `${response.reflections} nuevas búsquedas tras la verificación`);
  $("agent-meta").textContent = meta.join(". ") + ".";

  const titles = new Map(response.recommendations.map((r) => [r.book.id, r.book.title]));
  const books = $("agent-books");
  books.replaceChildren();
  for (const rec of response.recommendations) {
    const li = el("li", "agent-book");
    const h = el("h3", "book-title");
    const button = el("button", "title-button", rec.book.title);
    button.type = "button";
    button.setAttribute("aria-label", `${rec.book.title}: ver ficha`);
    button.addEventListener("click", () => openBook(rec.book));
    h.append(button);
    const facts = el("p", "facts");
    fillFacts(facts, rec.book);
    li.append(h, el("p", "authors", rec.book.authors.join(", ") || "Autor desconocido"), facts, el("p", "reason", rec.reason));
    books.append(li);
  }
  if (!response.recommendations.length) books.append(el("li", "field-note", "El agente no ha recomendado ningún libro concreto."));

  const dropped = $("agent-dropped");
  dropped.hidden = !response.dropped_recommendations;
  dropped.textContent = response.dropped_recommendations === 1
    ? "Se ha descartado 1 recomendación porque ninguna herramienta había devuelto ese libro."
    : `Se han descartado ${response.dropped_recommendations} recomendaciones porque ninguna herramienta había devuelto esos libros.`;

  const steps = $("agent-steps");
  steps.replaceChildren(...response.steps.map((step) => renderStep(step, titles)));
  const verify = el("li", "agent-step verify");
  verify.append(el("p", "tool-name", "Verificación"),
    el("p", "tool-count", "Cada libro de la respuesta se ha comprobado contra lo que devolvieron las herramientas."));
  steps.append(verify);

  $("agent-result").hidden = false;
  $("empty-agent").hidden = true;
}

async function runAgent(event) {
  event.preventDefault();
  const question = $("agent-question").value.trim();
  if (question.length < 3) return;
  $("agent-message").hidden = true;
  reserveSpace("agent-output");
  $("empty-agent").hidden = true;
  $("agent-result").classList.add("stale");
  const wait = startWaiting("agent-wait", "agent-button", [
    { text: "El agente está buscando…", hint: "Suele tardar entre 15 y 60 segundos: hace varias búsquedas y lee los resultados antes de responder." },
    { after: 10, text: "El agente sigue trabajando…", hint: "Puede estar repitiendo una búsqueda o comprobando sus recomendaciones. " + COLD_START_HINT },
  ]);
  try {
    const response = await callApi("/agent/ask", { method: "POST", body: JSON.stringify({ question }), signal: wait.signal });
    renderAgent(response);
    $("agent-answer").focus();
  } catch (error) {
    if (error.name !== "AbortError") {
      showMessage("agent-message", error, "preguntar al agente");
      // The error says the same as the start-up note: show it once.
      $("agent-off-note").hidden = true;
    }
  } finally {
    $("agent-result").classList.remove("stale");
    wait.stop();
  }
}

// ---------------------------------------------------------------------------
// Tabs, examples, health and start-up
// ---------------------------------------------------------------------------

function selectTab(name) {
  for (const [tab, panel] of [["tab-search", "panel-search"], ["tab-agent", "panel-agent"]]) {
    const active = tab === `tab-${name}`;
    $(tab).setAttribute("aria-selected", String(active));
    $(tab).tabIndex = active ? 0 : -1;
    $(panel).hidden = !active;
  }
}

function setupTabs() {
  const tabs = [$("tab-search"), $("tab-agent")];
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(i === 0 ? "search" : "agent"));
    // Arrow keys move between tabs (WAI-ARIA tabs pattern).
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - i;
      tabs[next].focus();
      selectTab(next === 0 ? "search" : "agent");
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
      $("agent-off-note").textContent = "El agente necesita un LLM y este servidor no tiene ninguno configurado "
        + "(falta OPENROUTER_API_KEY). Puedes usar la búsqueda de libros, que funciona sin él.";
      $("agent-off-note").hidden = false;
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
  renderExamples("agent-examples", AGENT_EXAMPLES, (text) => {
    $("agent-question").value = text;
    $("agent-question").focus();
  });
  $("search-form").addEventListener("submit", runSearch);
  $("agent-form").addEventListener("submit", runAgent);
  $("similar-back").addEventListener("click", backToSearch);
  // Enter sends the question; Shift+Enter adds a new line.
  $("agent-question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("agent-form").requestSubmit();
    }
  });
  loadHealth();
}

start();
