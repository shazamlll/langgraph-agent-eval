"""Contract checks using httpx's in-memory transport, never paid APIs."""

import json

import httpx
import pytest

from agent import BookstoreAgent
from llm import RealLLMAdapter


CLASSIFICATION = {
    "intent": "search", "genre": "fantasy", "keywords": [],
    "max_price": 15.0, "min_rating": None,
}


def adapter_for(content, requests=None, status=200):
    def handle(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    return RealLLMAdapter(
        api_key="test-only-key", base_url="https://example.invalid/v1/",
        model="test-model", client=httpx.Client(transport=httpx.MockTransport(handle)),
    )


def test_classification_posts_json_mode_and_validates_slots():
    requests = []
    result = adapter_for(json.dumps(CLASSIFICATION), requests).classify_intent("Find fantasy under $15")
    assert result == CLASSIFICATION
    request = requests[0]
    assert str(request.url) == "https://example.invalid/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-only-key"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][-1] == {"role": "user", "content": "Find fantasy under $15"}
    assert request.extensions["timeout"]["read"] == 30.0


@pytest.mark.parametrize("change", [
    {"intent": "delete"}, {"genre": ["fantasy"]}, {"keywords": "dragons"},
    {"keywords": [7]}, {"max_price": True}, {"max_price": -1},
    {"min_rating": 6}, {"max_price": float("nan")},
])
def test_rejects_invalid_model_slots(change):
    with pytest.raises(ValueError, match="classification"):
        adapter_for(json.dumps({**CLASSIFICATION, **change})).classify_intent("Find a book")


@pytest.mark.parametrize("content", ["not json", "[]", "{}", "null"])
def test_rejects_missing_or_non_object_classification(content):
    with pytest.raises(ValueError, match="classification"):
        adapter_for(content).classify_intent("Hello")


def test_reply_receives_tool_results_and_remembered_preferences():
    requests = []
    adapter = adapter_for("Read The Clockwork Orchard.", requests)
    context = {
        "intent": "recommend", "results": [{"title": "The Clockwork Orchard", "price": 14.0}],
        "preferences": {"genre": "fantasy"}, "user_text": "Recommend something", "search_genre": None,
    }
    assert adapter.respond(**context) == "Read The Clockwork Orchard."
    payload = json.loads(requests[0].content)
    assert json.loads(payload["messages"][-1]["content"]) == context
    assert "invent" in payload["messages"][0]["content"].lower()
    assert "on search or recommend" in payload["messages"][0]["content"].lower()
    assert "response_format" not in payload


@pytest.mark.parametrize("content", ["", "  ", None, ["text"]])
def test_rejects_empty_or_non_text_reply(content):
    with pytest.raises(ValueError, match="text"):
        adapter_for(content).respond(intent="chat", results=[], preferences={}, user_text="hi")


def test_http_failure_is_clear_and_does_not_expose_response_or_key():
    with pytest.raises(RuntimeError, match="LLM request failed") as error:
        adapter_for("test-only-key", status=401).classify_intent("Hello")
    assert "test-only-key" not in str(error.value)


def test_timeout_is_reported_without_a_silent_stub_fallback():
    def timeout(request):
        raise httpx.ReadTimeout("test-only-key", request=request)
    adapter = RealLLMAdapter(api_key="test-only-key", client=httpx.Client(transport=httpx.MockTransport(timeout)))
    with pytest.raises(RuntimeError, match="LLM request failed") as error:
        adapter.classify_intent("Hello")
    assert "test-only-key" not in str(error.value)


def test_real_adapter_works_through_graph_and_remembers_preferences():
    replies = iter([
        json.dumps({**CLASSIFICATION, "intent": "preference", "genre": "science fiction"}),
        "I'll remember you like science fiction.",
        json.dumps({**CLASSIFICATION, "intent": "recommend", "genre": None}),
        "Since you like science fiction, try Orbit of Echoes.",
    ])
    def handle(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": next(replies)}}]})
    adapter = RealLLMAdapter(api_key="test-only-key", client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BookstoreAgent(llm=adapter)
    agent.chat("I like science fiction.", thread_id="mock-real")
    agent.chat("Recommend something.", thread_id="mock-real")
    assert agent.last_turn["preferences"] == {"genre": "science fiction"}
    assert agent.last_turn["tool_calls"][0]["args"] == {"genre": "science fiction"}
    assert "Orbit of Echoes" in agent.last_turn["response"]
