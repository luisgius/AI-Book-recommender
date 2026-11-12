"""
Hedged query embeddings, the Zero Data Retention flag and the query vector cache.

Every request goes to a fake httpx transport: no test reaches the network.
The fake answers each provider after a delay chosen by the test and records
which requests it received and which ones were cancelled in flight. Delays
are tens of milliseconds, with margins of several times that, so the timing
assertions do not depend on the machine's speed.
"""

import asyncio
import base64
import json
import struct
import time
from typing import Dict, List, Optional

import httpx
import pytest

from app.infrastructure.config.settings import ApiQuerySettings, EmbeddingSettings
from app.infrastructure.search.openrouter_embedding_encoder import (
    OpenRouterEmbeddingEncoder,
    QueryHedge,
    QueryVectorCache,
)
from app.infrastructure.search.openrouter_hedge import (
    CANCELLED,
    NOT_SENT,
    HedgedPoster,
    HedgeFailed,
    HedgeLeg,
)
from app.infrastructure.usage_recording import UsageRecorder

MODEL = "qwen/qwen3-embedding-8b"
DIMENSION = 1024


def vector_b64(value: float) -> str:
    return base64.b64encode(struct.pack(f"<{DIMENSION}f", *([value] * DIMENSION))).decode()


class FakeProviders:
    """
    An async transport that plays several upstream providers.

    behaviour[provider] = (delay_s, status). Each provider answers its own
    constant vector (value given in `values`), so a test can tell who won.
    """

    def __init__(self, behaviour: Dict[str, tuple], values: Optional[Dict[str, float]] = None):
        self.behaviour = behaviour
        self.values = values or {"deepinfra": 0.25, "nebius": 0.5}
        self.received: List[tuple] = []  # (provider, seconds since the fake was created)
        self.cancelled: List[str] = []
        self.completed: List[str] = []
        self.bodies: List[dict] = []
        self._t0 = time.perf_counter()

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        provider = body["provider"]["order"][0]
        self.bodies.append(body)
        self.received.append((provider, time.perf_counter() - self._t0))
        delay, status = self.behaviour[provider]
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled.append(provider)
            raise
        self.completed.append(provider)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": f"{provider} busy"}})
        return httpx.Response(200, json={
            "data": [{"index": 0, "embedding": vector_b64(self.values[provider])}],
            "model": MODEL, "provider": provider,
            "usage": {"prompt_tokens": 7, "total_tokens": 7, "cost": 7e-8},
        })

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def providers_received(self) -> List[str]:
        return [provider for provider, _ in self.received]


def encoder_with(fake: FakeProviders, *, hedge: Optional[QueryHedge] = None, cache: int = 0,
                 zdr: bool = True, recorder: Optional[UsageRecorder] = None) -> OpenRouterEmbeddingEncoder:
    hedge = hedge or QueryHedge("nebius", after_s=0.15, primary_again_s=0.4, deadline_s=1.0)
    return OpenRouterEmbeddingEncoder(
        MODEL, dimensions=DIMENSION, upstream_provider="deepinfra", api_key="test-key",
        zdr=zdr, query_hedge=hedge, query_cache_size=cache, usage_recorder=recorder,
        hedged_poster=HedgedPoster(api_key="test-key", transport=fake.transport()),
    )


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------


def test_cloud_defaults_hedge_to_nebius_with_zdr_and_a_query_cache(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    s = EmbeddingSettings.from_env({"SEARCH_PROFILE": "cloud"})
    assert (s.openrouter_provider, s.hedge_provider, s.hedge_after_s, s.hedge_primary_again_s) == (
        "deepinfra", "nebius", 1.0, 2.5)
    assert s.openrouter_zdr is True and s.query_cache_size == 256

    encoder = OpenRouterEmbeddingEncoder.from_settings(
        s, query_settings=ApiQuerySettings.from_env({"SEARCH_PROFILE": "cloud"}))
    assert encoder.query_hedge == QueryHedge("nebius", 1.0, 2.5, 5.0)
    assert encoder.query_cache is not None
    # The vectors' identity stays the pinned provider's: the stored documents are DeepInfra's
    assert encoder.get_model_name() == "openrouter:qwen/qwen3-embedding-8b?provider=deepinfra&dimensions=1024"


def test_the_previous_policy_is_available_by_configuration(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    s = EmbeddingSettings.from_env({
        "SEARCH_PROFILE": "cloud", "EMBEDDING_HEDGE_PROVIDER": "none", "EMBEDDING_QUERY_CACHE_SIZE": "0",
        "EMBEDDING_OPENROUTER_ZDR": "false",
    })
    assert s.hedge_provider is None and s.query_cache_size == 0 and s.openrouter_zdr is False
    encoder = OpenRouterEmbeddingEncoder.from_settings(s)
    assert encoder.query_hedge is None and encoder.query_cache is None
    assert encoder.request_body(["q"])["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False}


def test_hedge_parameters_are_settings():
    s = EmbeddingSettings.from_env({
        "SEARCH_PROFILE": "cloud", "EMBEDDING_HEDGE_PROVIDER": "siliconflow",
        "EMBEDDING_HEDGE_AFTER_S": "0.6", "EMBEDDING_HEDGE_PRIMARY_AGAIN_S": "none",
        "EMBEDDING_QUERY_CACHE_SIZE": "500",
    })
    assert (s.hedge_provider, s.hedge_after_s, s.hedge_primary_again_s, s.query_cache_size) == (
        "siliconflow", 0.6, None, 500)


def test_defaults_outside_the_cloud_profile_keep_the_evaluation_behaviour():
    local = EmbeddingSettings.from_env({})
    assert local.hedge_provider is None and local.query_cache_size == 0 and local.openrouter_zdr is False
    api = EmbeddingSettings.from_env({"EMBEDDING_PROVIDER": "openrouter", "EMBEDDING_MODEL": MODEL,
                                      "EMBEDDING_OPENROUTER_PROVIDER": "deepinfra"})
    assert api.hedge_provider is None and api.query_cache_size == 0
    assert api.openrouter_zdr is True  # the data policy applies wherever the application calls the API


@pytest.mark.parametrize("env, message", [
    ({"EMBEDDING_PROVIDER": "openrouter", "EMBEDDING_MODEL": MODEL, "EMBEDDING_HEDGE_PROVIDER": "nebius"},
     "needs EMBEDDING_OPENROUTER_PROVIDER"),
    ({"SEARCH_PROFILE": "cloud", "EMBEDDING_HEDGE_PROVIDER": "deepinfra"}, "must differ"),
    ({"EMBEDDING_HEDGE_PROVIDER": "nebius"}, "apply only to EMBEDDING_PROVIDER=openrouter"),
    ({"SEARCH_PROFILE": "cloud", "EMBEDDING_QUERY_CACHE_SIZE": "-1"}, "EMBEDDING_QUERY_CACHE_SIZE"),
])
def test_invalid_hedge_settings_are_rejected(env, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingSettings.from_env(env)


def test_a_hedge_must_fit_in_its_deadline():
    with pytest.raises(ValueError, match="shorter than the deadline"):
        QueryHedge("nebius", after_s=1.0, primary_again_s=6.0, deadline_s=5.0)


# -----------------------------------------------------------------------------
# Request bodies
# -----------------------------------------------------------------------------


def test_zdr_is_off_by_default_so_the_published_request_bodies_do_not_change():
    encoder = OpenRouterEmbeddingEncoder(MODEL, dimensions=DIMENSION, upstream_provider="deepinfra", api_key="k")
    assert encoder.request_body(["text"]) == {
        "model": MODEL, "input": ["text"], "encoding_format": "base64", "dimensions": DIMENSION,
        "provider": {"order": ["deepinfra"], "allow_fallbacks": False},
    }


def test_hedge_legs_pin_each_provider_and_keep_zdr():
    encoder = encoder_with(FakeProviders({}))
    legs = encoder.hedge_legs("Instruct: q")
    assert [(leg.start_s, leg.label) for leg in legs] == [(0.0, "deepinfra"), (0.15, "nebius"), (0.4, "deepinfra")]
    assert [leg.body["provider"] for leg in legs] == [
        {"order": ["deepinfra"], "allow_fallbacks": False, "zdr": True},
        {"order": ["nebius"], "allow_fallbacks": False, "zdr": True},
        {"order": ["deepinfra"], "allow_fallbacks": False, "zdr": True},
    ]
    assert {json.dumps(leg.body["input"]) for leg in legs} == {'["Instruct: q"]'}


# -----------------------------------------------------------------------------
# The race
# -----------------------------------------------------------------------------


def test_a_prompt_primary_answer_sends_no_second_request():
    fake = FakeProviders({"deepinfra": (0.01, 200), "nebius": (0.01, 200)})
    recorder = UsageRecorder()
    vector = encoder_with(fake, recorder=recorder).generate_query_embedding("q")

    assert vector[0] == pytest.approx(0.25)
    assert fake.providers_received() == ["deepinfra"]
    records = recorder.records
    assert len(records) == 1 and records[0].error is None and records[0].cost_usd == 7e-8


def test_a_slow_primary_is_hedged_and_cancelled_when_the_second_provider_wins():
    fake = FakeProviders({"deepinfra": (5.0, 200), "nebius": (0.02, 200)})
    recorder = UsageRecorder()
    started = time.perf_counter()
    vector = encoder_with(fake, recorder=recorder).generate_query_embedding("q")
    elapsed = time.perf_counter() - started

    assert vector[0] == pytest.approx(0.5)  # Nebius's vector
    assert fake.providers_received() == ["deepinfra", "nebius"]
    nebius_sent = fake.received[1][1] - fake.received[0][1]
    assert 0.12 < nebius_sent < 0.35  # at after_s (0.15 s), not before
    assert elapsed < 0.6  # did not wait for DeepInfra's 5 s
    assert fake.cancelled == ["deepinfra"] and "deepinfra" not in fake.completed
    # Both requests sent are recorded: the winner with its cost, the loser as cancelled
    records = recorder.records
    assert [(r.error, r.cost_usd) for r in records] == [(CANCELLED, None), (None, 7e-8)]


def test_an_error_from_the_primary_sends_the_second_request_at_once():
    fake = FakeProviders({"deepinfra": (0.0, 503), "nebius": (0.02, 200)})
    encoder = encoder_with(fake, hedge=QueryHedge("nebius", after_s=0.5, primary_again_s=0.8, deadline_s=1.0))
    vector = encoder.generate_query_embedding("q")

    assert vector[0] == pytest.approx(0.5)
    assert fake.providers_received() == ["deepinfra", "nebius"]
    assert fake.received[1][1] - fake.received[0][1] < 0.25  # well before after_s (0.5 s)


def test_the_primary_is_asked_again_and_can_win():
    # The first DeepInfra request hangs and Nebius is slow: the third leg
    # (DeepInfra again at 0.3 s) answers first and the other two are cancelled
    fake = FakeProviders({"deepinfra": (0.02, 200), "nebius": (5.0, 200)})
    hanging = FakeProviders({"deepinfra": (5.0, 200)})
    deepinfra_calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["provider"]["order"][0] == "deepinfra":
            deepinfra_calls.append(request)
            if len(deepinfra_calls) == 1:
                return await hanging.handle(request)
        return await fake.handle(request)

    encoder = OpenRouterEmbeddingEncoder(
        MODEL, dimensions=DIMENSION, upstream_provider="deepinfra", api_key="k", zdr=True,
        query_hedge=QueryHedge("nebius", after_s=0.1, primary_again_s=0.3, deadline_s=1.0),
        hedged_poster=HedgedPoster(api_key="k", transport=httpx.MockTransport(handle)),
    )
    vector = encoder.generate_query_embedding("q")

    assert vector[0] == pytest.approx(0.25)
    assert len(deepinfra_calls) == 2 and fake.providers_received() == ["nebius", "deepinfra"]
    assert hanging.cancelled == ["deepinfra"] and fake.cancelled == ["nebius"]


def test_when_every_provider_fails_the_call_fails_and_every_request_is_recorded():
    fake = FakeProviders({"deepinfra": (0.0, 503), "nebius": (0.0, 502)})
    recorder = UsageRecorder()
    with pytest.raises(HedgeFailed, match="every hedged request failed") as caught:
        encoder_with(fake, recorder=recorder).generate_query_embedding("q")

    assert fake.providers_received() == ["deepinfra", "nebius", "deepinfra"]
    assert [leg.sent for leg in caught.value.legs] == [True, True, True]
    assert [r.error for r in recorder.records] == ["OpenRouterError"] * 3


def test_the_deadline_cancels_every_request_in_flight():
    fake = FakeProviders({"deepinfra": (5.0, 200), "nebius": (5.0, 200)})
    hedge = QueryHedge("nebius", after_s=0.05, primary_again_s=None, deadline_s=0.3)
    started = time.perf_counter()
    with pytest.raises(HedgeFailed, match="deadline") as caught:
        encoder_with(fake, hedge=hedge).generate_query_embedding("q")

    assert time.perf_counter() - started < 0.8
    assert sorted(fake.cancelled) == ["deepinfra", "nebius"]
    assert [leg.error for leg in caught.value.legs] == [CANCELLED, CANCELLED]


def test_a_leg_never_started_is_reported_as_not_sent():
    fake = FakeProviders({"deepinfra": (0.01, 200)})
    poster = HedgedPoster(api_key="k", transport=fake.transport())
    body = {"provider": {"order": ["deepinfra"]}}
    response = poster.post_first("embeddings", [HedgeLeg(0.0, body, "a"), HedgeLeg(0.5, body, "b")], deadline_s=1.0)

    assert response.winner == "a"
    assert [(leg.sent, leg.error) for leg in response.legs] == [(True, None), (False, NOT_SENT)]


def test_the_api_key_is_sent_and_no_identifying_header():
    seen = {}

    async def handle(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": vector_b64(0.1)}]})

    poster = HedgedPoster(api_key="secret-key", transport=httpx.MockTransport(handle))
    poster.post_first("embeddings", [HedgeLeg(0.0, {"x": 1}, "a")], deadline_s=1.0)

    assert seen["authorization"] == "Bearer secret-key"
    assert "http-referer" not in seen and "x-title" not in seen


# -----------------------------------------------------------------------------
# Query cache
# -----------------------------------------------------------------------------


def test_a_repeated_query_is_served_from_the_cache_whoever_answered_it():
    fake = FakeProviders({"deepinfra": (5.0, 200), "nebius": (0.01, 200)})
    encoder = encoder_with(fake, cache=2)

    first = encoder.generate_query_embedding("novelas de la guerra civil")
    second = encoder.generate_query_embedding("novelas de la guerra civil")

    assert first == second and first[0] == pytest.approx(0.5)  # Nebius's vector, kept
    assert fake.providers_received() == ["deepinfra", "nebius"]  # no request for the repeat
    assert (encoder.query_cache.hits, encoder.query_cache.misses) == (1, 1)
    second[0] = 99.0  # a caller changing its list does not change the cache
    assert encoder.generate_query_embedding("novelas de la guerra civil")[0] == pytest.approx(0.5)


def test_the_cache_keys_on_the_exact_text_and_is_bounded():
    fake = FakeProviders({"deepinfra": (0.0, 200), "nebius": (0.0, 200)})
    encoder = encoder_with(fake, cache=2)
    for text in ("a", "A", "a ", "a"):
        encoder.generate_query_embedding(text)
    # "a", "A" and "a " are three texts; the cache holds two, so the last "a" was evicted
    assert len(fake.received) == 4 and len(encoder.query_cache) == 2


def test_the_cache_is_used_without_a_hedge_too():
    cache = QueryVectorCache(1)
    cache.put(("m", 3, "q"), [1.0, 2.0, 3.0])
    assert cache.get(("m", 3, "q")) == [1.0, 2.0, 3.0] and cache.get(("m", 3, "other")) is None
    with pytest.raises(ValueError):
        QueryVectorCache(0)


def test_documents_are_never_hedged_nor_cached():
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["provider"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": vector_b64(0.1)}]})

    fake = FakeProviders({})
    encoder = OpenRouterEmbeddingEncoder(
        MODEL, dimensions=DIMENSION, upstream_provider="deepinfra", api_key="k", zdr=True,
        transport=httpx.MockTransport(handle), query_cache_size=4,
        query_hedge=QueryHedge("nebius"), hedged_poster=HedgedPoster(api_key="k", transport=fake.transport()),
    )
    encoder.generate_embeddings_batch(["doc one"])
    encoder.generate_embeddings_batch(["doc one"])
    assert sent == [{"order": ["deepinfra"], "allow_fallbacks": False, "zdr": True}] * 2
    assert fake.received == [] and len(encoder.query_cache) == 0
