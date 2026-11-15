"""Tests for the LLM layer (stub) and the disabled-by-default tracing."""

import pytest

from llm import RealLLMAdapter, StubResponder, get_llm
from tracing import init_tracing, trace_turn


def test_classify_preference():
    result = StubResponder().classify_intent("I like science fiction.")
    assert result["intent"] == "preference"
    assert result["genre"] == "science fiction"


def test_classify_search_with_price():
    result = StubResponder().classify_intent("Show me fantasy books under $15.")
    assert result["intent"] == "search"
    assert result["genre"] == "fantasy"
    assert result["max_price"] == 15.0


def test_classify_search_with_rating_hint():
    result = StubResponder().classify_intent("Do you have any highly rated mystery books?")
    assert result["intent"] == "search"
    assert result["genre"] == "mystery"
    assert result["min_rating"] == 4.5


def test_classify_recommend_and_greeting():
    stub = StubResponder()
    assert stub.classify_intent("Can you recommend something?")["intent"] == "recommend"
    assert stub.classify_intent("Hi!")["intent"] == "greeting"
    assert stub.classify_intent("What can you do?")["intent"] == "chat"


def test_stub_respond_mentions_results_and_memory():
    stub = StubResponder()
    reply = stub.respond(
        intent="recommend",
        results=[{"title": "Orbit of Echoes", "author": "Dario Venn",
                  "price": 18.99, "rating": 4.8}],
        preferences={"genre": "science fiction"},
        user_text="Recommend something.",
    )
    assert "Orbit of Echoes" in reply
    assert "science fiction" in reply


def test_real_adapter_requires_an_explicit_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        RealLLMAdapter()


def test_get_llm_defaults_to_stub(monkeypatch):
    monkeypatch.delenv("USE_REAL_LLM", raising=False)
    assert isinstance(get_llm(), StubResponder)


def test_tracing_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_TRACING", raising=False)
    status = init_tracing()
    assert status["enabled"] is False
    assert trace_turn({"turn": 1}, status) is None


def test_tracing_without_keys_reports_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_TRACING", "1")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    status = init_tracing()
    assert status["enabled"] is False
    assert "no provider key" in status["reason"]
