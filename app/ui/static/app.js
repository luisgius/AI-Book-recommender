// Minimal client for the book search API (no framework, no external assets).
// Every value from the API is rendered with textContent (never parsed as
// HTML), so catalog text cannot inject markup.

const API = "/api/v1";
const $ = (id) => document.getElementById(id);

const FIELD_NAMES = { title: "título", description: "descripción", categories: "categorías", authors: "autores" };
const INTENT_NAMES = { recommendation: "recomendación", factual: "factual", exploratory: "exploratoria" };

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

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

async function callApi(path, options = {}) {
  const response = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    throw new Error(`${response.status}: ${detail}`);
  }
  return body;
}

function formatScore(value) {
  return value === null || value === undefined ? "–" : value.toFixed(4);
}

function renderResult(result, { showScores = true } = {}) {
  const node = $("result-template").content.firstElementChild.cloneNode(true);
  const book = result.book;
  const year = book.published_date ? book.published_date.slice(0, 4) : "s. f.";
  const categories = book.categories.length ? book.categories.join(", ") : "sin categoría";

  node.querySelector(".rank").textContent = result.rank;
  node.querySelector(".title").textContent = book.title;
  node.querySelector(".meta").textContent =
    `${book.authors.join(", ")} · ${year} · ${book.language || "?"} · ${categories}`;
  const description = book.description || "";
  node.querySelector(".description").textContent =
    description.length > 320 ? description.slice(0, 317) + "…" : description;

  node.querySelector(".scores").textContent = showScores
    ? `final ${formatScore(result.final_score)} · BM25 ${formatScore(result.lexical_score)} · vector ${formatScore(result.vector_score)} · fuente ${result.source}`
    : `similitud ${formatScore(result.vector_score)}`;

  if (result.explanation) {
    const box = node.querySelector(".explanation");
    box.hidden = false;
    box.classList.toggle("ungrounded", !result.explanation.grounded);
    box.querySelector(".explanation-text").textContent = result.explanation.text;
    const list = box.querySelector(".citations");
    for (const citation of result.explanation.citations) {
      const item = document.createElement("li");
      const field = document.createElement("span");
      field.className = "field";
      field.textContent = FIELD_NAMES[citation.field] || citation.field;
      item.append(field, document.createTextNode(`«${citation.snippet}»`));
      list.append(item);
    }
    box.querySelector(".explanation-model").textContent =
      `${result.explanation.grounded ? "Citas verificadas contra los datos del libro" : "Sin citas verificables"} · modelo ${result.explanation.model}`;
  }

  node.querySelector(".similar").addEventListener("click", () => showSimilar(book));
  return node;
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
    .map(([key, value]) => `${key}=${value}`)
    .join(", ") || "ninguno";
  const rows = [
    ["Intención", `${INTENT_NAMES[intent.intent_type] || intent.intent_type} (confianza ${intent.confidence.toFixed(2)})`],
    ["Consulta reformulada", intent.reformulated_query],
    ["Variaciones", intent.query_variations.join(" | ") || "–"],
    ["Filtros extraídos", filters],
    ["Razonamiento", intent.reasoning],
  ];
  const list = document.createElement("dl");
  for (const [label, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
  section.append(list);
  section.hidden = false;
}

async function runSearch(event) {
  event.preventDefault();
  const request = buildRequest();
  if (!request.text) return;

  const usesLlm = request.use_understanding || request.use_explanations;
  $("search-button").disabled = true;
  $("similar-banner").hidden = true;
  setStatus(usesLlm ? "Buscando… (con LLM local puede tardar un minuto)" : "Buscando…");

  try {
    const response = await callApi("/search", { method: "POST", body: JSON.stringify(request) });
    renderIntent(response.query_intent);
    $("results").replaceChildren(...response.results.map((r) => renderResult(r)));
    const degraded = response.degraded ? ` · modo degradado: ${response.degradation_reason}` : "";
    const candidates = response.metadata
      ? ` · candidatos BM25 ${response.metadata.candidates_lexical}, vector ${response.metadata.candidates_vector}`
      : "";
    setStatus(`${response.results.length} resultados en ${Math.round(response.latency_ms)} ms (${response.search_mode})${candidates}${degraded}`);
  } catch (error) {
    setStatus(`Error en la búsqueda: ${error.message}`, true);
  } finally {
    $("search-button").disabled = false;
  }
}

async function showSimilar(book) {
  setStatus(`Buscando libros similares a «${book.title}»…`);
  try {
    const response = await callApi(`/books/${book.id}/similar?max_results=${intOrNull($("max-results").value) || 10}`);
    const banner = $("similar-banner");
    banner.textContent = `Libros similares a «${book.title}» (similitud de embeddings)`;
    banner.hidden = false;
    renderIntent(null);
    $("results").replaceChildren(...response.results.map((r) => renderResult(r, { showScores: false })));
    setStatus(`${response.results.length} libros similares`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    setStatus(`No se pudieron obtener libros similares: ${error.message}`, true);
  }
}

async function loadHealth() {
  const el = $("health");
  try {
    const health = await callApi("/health");
    const llm = health.llm.configured ? `LLM ${health.llm.model}` : "sin LLM";
    el.textContent = `${health.status === "ok" ? "Índices listos" : "Degradado"} · ${llm}`;
    el.classList.add(health.status === "ok" ? "ok" : "degraded");
    if (!health.llm.configured) {
      for (const id of ["use-understanding", "use-explanations"]) {
        $(id).disabled = true;
        $(id).parentElement.title = "No hay LLM configurado (LLM_PROVIDER)";
      }
    }
  } catch (error) {
    el.textContent = "API no disponible";
    el.classList.add("degraded");
  }
}

$("search-form").addEventListener("submit", runSearch);
loadHealth();
