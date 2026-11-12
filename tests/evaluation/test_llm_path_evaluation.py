"""Tests for the LLM-path evaluation (cached intents, ablation variants, significance)."""

import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import pytest

from app.domain.entities import Book, SearchResult
from app.domain.value_objects import QueryIntent, SearchFilters
from app.evaluation.llm_path_evaluation import (
    VARIANTS,
    collect_intents,
    evaluate_llm_path,
    intent_from_dict,
    intent_to_dict,
    variant_intent,
)
from app.evaluation.types import TestQuery


def make_book(title: str, source_id: str, language: str) -> Book:
    return Book(id=uuid4(), title=title, authors=["A"], language=language, categories=["Fiction"],
                published_date=datetime(2000, 1, 1), source="google_books", source_id=source_id)


BOOKS = [make_book("English one", "en1", "en"), make_book("Spanish one", "es1", "es"),
         make_book("English two", "en2", "en")]


class Repo:
    def __init__(self, source):
        self.source = source

    def search(self, query_text=None, max_results=10, filters=None, query_embedding=None):
        matching = [b for b in BOOKS if b.matches_filters(filters)][:max_results]
        field = "lexical_score" if self.source == "lexical" else "vector_score"
        return [SearchResult(book=b, final_score=1.0, rank=i + 1, source=self.source, **{field: 1.0})
                for i, b in enumerate(matching)]

    def is_ready(self):
        return True


class Embeddings:
    def generate_embedding(self, text: str) -> List[float]:
        return [0.1]

    generate_query_embedding = generate_embedding  # same vector for queries

    def get_embedding(self, book_id: UUID) -> Optional[List[float]]:
        return [0.1]

    def get_embeddings_batch(self, book_ids):
        return {i: v for i in book_ids if (v := self.get_embedding(i)) is not None}

    def is_ready(self):
        return True


class Catalog:
    def get_by_source_id(self, source, source_id):
        return next((b for b in BOOKS if b.source_id == source_id), None)


def intent(**overrides) -> QueryIntent:
    base = dict(intent_type="factual", original_query="libros", reformulated_query="books reformulated",
                extracted_filters=SearchFilters(language="es"), confidence=0.8, reasoning="r",
                query_variations=["books reformulated", "other wording"])
    base.update(overrides)
    return QueryIntent(**base)


class TestVariants:
    def test_each_variant_keeps_only_its_part_of_the_llm_output(self):
        full = intent()
        routing = variant_intent(full, "routing_only")
        assert (routing.reformulated_query, routing.extracted_filters, routing.query_variations) == \
            ("libros", SearchFilters(), [])
        filters = variant_intent(full, "filters")
        assert filters.reformulated_query == "libros" and filters.extracted_filters.language == "es"
        reform = variant_intent(full, "reformulation")
        assert reform.reformulated_query == "books reformulated" and reform.query_variations == []
        assert reform.extracted_filters == SearchFilters()
        multi = variant_intent(full, "reformulation_multi_query")
        assert multi.query_variations and multi.extracted_filters == SearchFilters()
        assert variant_intent(full, "full") is full

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError):
            variant_intent(intent(), "magic")

    def test_intent_round_trips_through_the_cache_format(self):
        assert intent_from_dict(json.loads(json.dumps(intent_to_dict(intent())))) == intent()


class CountingLLM:
    def __init__(self):
        self.calls = []

    def extract_query_intent(self, text):
        self.calls.append(text)
        return intent(original_query=text)


def test_collect_intents_calls_the_llm_only_for_uncached_queries(tmp_path):
    cache = tmp_path / "intents.json"
    cache.write_text(json.dumps({"intents": {"q1": intent_to_dict(intent(original_query="cached"))}}))
    llm = CountingLLM()
    queries = [TestQuery(query_id="q1", text="first"), TestQuery(query_id="q2", text="second")]

    intents = collect_intents(queries, llm, cache)

    assert llm.calls == ["second"]
    assert intents["q1"].original_query == "cached"
    assert set(json.loads(cache.read_text())["intents"]) == {"q1", "q2"}


def test_collect_intents_without_cache_or_llm_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError):
        collect_intents([TestQuery(query_id="q1", text="x")], None, tmp_path / "missing.json")


def test_evaluate_llm_path_reports_every_mode_and_significance(tmp_path):
    queries = tmp_path / "q.json"
    judgments = tmp_path / "j.json"
    queries.write_text(json.dumps([{"query_id": "q1", "text": "libros en español"}]))
    judgments.write_text(json.dumps([{"query_id": "q1", "judgments": [
        {"source": "google_books", "source_id": "es1", "relevance": 3}]}]))

    report = evaluate_llm_path(lexical_repo=Repo("lexical"), vector_repo=Repo("vector"),
                               embeddings_store=Embeddings(), catalog_repo=Catalog(),
                               intents={"q1": intent(original_query="libros en español")},
                               queries_path=str(queries),
                               judgments_path=str(judgments), max_results=10, n_resamples=200)

    assert set(report["modes"]) == {"hybrid_rrf", *VARIANTS}
    assert set(report["significance_vs_baseline"]) == set(VARIANTS)
    # the query asks for Spanish, so the language=es filter survives the guards and
    # puts the only relevant (Spanish) book first
    assert report["modes"]["filters"]["mrr"] == pytest.approx(1.0)
    assert report["modes"]["routing_only"]["mrr"] < 1.0
    assert report["per_query"][0]["ndcg_at_10"]["filters"] == pytest.approx(1.0)


def test_cached_filters_are_revalidated_with_the_current_guards(tmp_path):
    """A language the query never asked for is dropped even if the cache has it."""
    queries = tmp_path / "q.json"
    judgments = tmp_path / "j.json"
    queries.write_text(json.dumps([{"query_id": "q1", "text": "libros"}]))
    judgments.write_text(json.dumps([{"query_id": "q1", "judgments": [
        {"source": "google_books", "source_id": "es1", "relevance": 3}]}]))

    report = evaluate_llm_path(lexical_repo=Repo("lexical"), vector_repo=Repo("vector"),
                               embeddings_store=Embeddings(), catalog_repo=Catalog(),
                               intents={"q1": intent(original_query="libros")}, queries_path=str(queries),
                               judgments_path=str(judgments), max_results=10, n_resamples=200)

    assert report["per_query"][0]["extracted_filters"]["language"] is None  # what was evaluated
    assert report["modes"]["filters"]["mrr"] == report["modes"]["routing_only"]["mrr"]


# --- LLM labels of a re-run from a complete intents cache -----------------------------------


def _saved_run(tmp_path, cache, **config):
    output = tmp_path / "results.json"
    labels = {"llm_provider": "ollama", "llm_model": "qwen3.8:27b-mlx", "temperature": 0.0,
              "apply_extracted_category": False}
    output.write_text(json.dumps({"config": {**labels, "intents_cache": str(cache), **config}}))
    return output, labels


def test_labels_come_from_the_saved_results_when_every_intent_is_cached(tmp_path):
    from app.evaluation.llm_path_evaluation import cached_run_labels

    cache = tmp_path / "intents.json"
    cache.write_text(json.dumps({"intents": {"q1": {}, "q2": {}}}))
    output, labels = _saved_run(tmp_path, cache)
    assert cached_run_labels(output, cache, ["q1", "q2"]) == labels


def test_no_saved_labels_when_an_intent_is_missing_or_the_cache_differs(tmp_path):
    from app.evaluation.llm_path_evaluation import cached_run_labels

    cache = tmp_path / "intents.json"
    cache.write_text(json.dumps({"intents": {"q1": {}}}))
    output, _ = _saved_run(tmp_path, cache)
    assert cached_run_labels(output, cache, ["q1", "q2"]) is None  # the LLM is needed: label from the environment
    other, _ = _saved_run(tmp_path, tmp_path / "other.json")
    assert cached_run_labels(other, cache, ["q1"]) is None
    assert cached_run_labels(tmp_path / "missing.json", cache, ["q1"]) is None
