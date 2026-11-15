"""LangSmith calls are checked against the installed SDK, without network."""

from unittest.mock import create_autospec
from uuid import UUID
import json

from langsmith import Client

from tracing import init_tracing, trace_turn


def enabled_status(monkeypatch, client):
    monkeypatch.setenv("ENABLE_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "mock-project")
    return init_tracing(client_factory=lambda **kwargs: client)


def test_disabled_tracing_never_constructs_client(monkeypatch):
    monkeypatch.delenv("ENABLE_TRACING", raising=False)
    def unexpected(**kwargs):
        raise AssertionError("Must stay offline")
    assert init_tracing(client_factory=unexpected)["enabled"] is False


def test_turn_is_sent_as_completed_langsmith_run(monkeypatch):
    client = create_autospec(Client, instance=True)
    status = enabled_status(monkeypatch, client)
    record = {
        "turn": 2, "thread_id": "demo", "user": "Recommend something",
        "response": "Orbit of Echoes", "intent": "recommend",
        "tool_calls": [{"name": "search_catalog"}],
        "preferences": {"genre": "science fiction"},
    }
    trace_turn(record, status)
    sent = client.create_run.call_args.kwargs
    assert sent["name"] == "bookstore.turn"
    assert sent["run_type"] == "chain"
    assert sent["inputs"] == {"user": record["user"]}
    assert sent["outputs"] == {"response": record["response"]}
    assert sent["project_name"] == "mock-project"
    assert sent["extra"]["metadata"] == {key: value for key, value in record.items() if key not in ("user", "response")}
    assert UUID(str(sent["id"]))
    assert sent["start_time"] <= sent["end_time"]


def test_setup_failure_disables_tracing_without_exposing_credentials(monkeypatch):
    monkeypatch.setenv("ENABLE_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-key")
    def broken(**kwargs):
        raise RuntimeError("test-only-key")
    status = init_tracing(client_factory=broken)
    assert status["enabled"] is False
    assert "test-only-key" not in status["reason"]


def test_upload_failure_is_reported_and_does_not_break_chat(monkeypatch, caplog):
    client = create_autospec(Client, instance=True)
    client.create_run.side_effect = RuntimeError("test-only-key")
    status = enabled_status(monkeypatch, client)
    trace_turn({"user": "hi", "response": "hello"}, status)
    assert status["enabled"] is False
    assert "failed" in status["reason"].lower()
    assert "test-only-key" not in caplog.text
    assert "failed" in caplog.text.lower()


def test_installed_sdk_serializes_completed_run_without_network(monkeypatch):
    requests = []
    def capture(self, method, url, **kwargs):
        requests.append((method, url, kwargs))
    monkeypatch.setattr(Client, "request_with_retries", capture)
    monkeypatch.setenv("ENABLE_TRACING", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-only-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "mock-project")
    status = init_tracing(client_factory=lambda **kwargs: Client(api_url="https://example.invalid", **kwargs))
    trace_turn({"user": "hi", "response": "hello", "turn": 1, "thread_id": "sdk-test"}, status)
    assert status["enabled"] is True
    method, url, kwargs = requests[-1]
    assert method == "POST"
    assert url == "https://example.invalid/runs"
    payload = json.loads(kwargs["request_kwargs"]["data"])
    assert payload["session_name"] == "mock-project"
    assert payload["inputs"] == {"user": "hi"}
    assert payload["outputs"] == {"response": "hello"}
    assert payload["end_time"]
    assert payload["extra"]["metadata"]["thread_id"] == "sdk-test"
