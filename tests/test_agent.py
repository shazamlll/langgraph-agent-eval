"""Tests for the LangGraph agent flow (stub responder, offline)."""

import json

from agent import BookstoreAgent
from trajectory import TrajectoryLogger
from llm import StubResponder


def test_single_turn_search_returns_expected_book():
    agent = BookstoreAgent()
    reply = agent.chat("Find me a mystery book.")
    assert "The Quiet Witness" in reply
    assert agent.last_turn["intent"] == "search"
    assert [c["name"] for c in agent.last_turn["tool_calls"]] == ["search_catalog"]


def test_preference_is_remembered_across_turns():
    agent = BookstoreAgent()
    agent.chat("I like science fiction.", thread_id="t1")
    reply = agent.chat("Can you recommend something?", thread_id="t1")
    assert "Orbit of Echoes" in reply
    assert "science fiction" in reply.lower()
    assert agent.last_turn["preferences"]["genre"] == "science fiction"


def test_preference_override_latest_wins():
    agent = BookstoreAgent()
    agent.chat("I love fantasy.")
    agent.chat("Actually, I prefer mystery novels.")
    reply = agent.chat("Recommend something for tonight.")
    assert "The Quiet Witness" in reply
    assert "The Cartographer's Daughter" not in reply


def test_search_reply_labelled_by_searched_genre_not_preference():
    agent = BookstoreAgent()
    agent.chat("I like science fiction.")
    reply = agent.chat("Find me a mystery book under $20.")
    assert "mystery books" in reply.lower()
    assert "science fiction books" not in reply.lower()
    assert "The Quiet Witness" in reply


def test_no_results_handled_gracefully():
    agent = BookstoreAgent()
    reply = agent.chat("Do you have any books about cooking?")
    assert "don't have any" in reply.lower()
    assert agent.last_turn["tool_calls"][0]["result_count"] == 0


def test_greeting_needs_no_tool():
    agent = BookstoreAgent()
    reply = agent.chat("Hi!")
    assert "bookstore assistant" in reply.lower()
    assert agent.last_turn["tool_calls"] == []


def test_non_search_turn_does_not_receive_previous_catalog_results():
    class RecordingResponder(StubResponder):
        def respond(self, **kwargs):
            self.latest_context = kwargs
            return super().respond(**kwargs)

    llm = RecordingResponder()
    agent = BookstoreAgent(llm=llm)
    agent.chat("Find me a mystery book.")
    assert llm.latest_context["results"]
    agent.chat("I prefer fantasy.")
    assert llm.latest_context["results"] == []
    assert agent.last_turn["tool_calls"] == []


def test_threads_do_not_share_memory():
    agent = BookstoreAgent()
    agent.chat("I like fantasy.", thread_id="a")
    reply = agent.chat("Recommend something.", thread_id="b")
    assert "fantasy" not in reply.lower()
    assert "Tell me a genre you like" in reply


def test_trajectory_jsonl_written(tmp_path):
    log_path = tmp_path / "traj.jsonl"
    agent = BookstoreAgent(logger=TrajectoryLogger(log_path))
    agent.chat("I like history books.", thread_id="t")
    agent.chat("Recommend one for a gift.", thread_id="t")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["turn"] == 1 and second["turn"] == 2
    assert first["user"] == "I like history books."
    assert second["preferences"]["genre"] == "history"
    for record in (first, second):
        assert {"turn", "thread_id", "user", "intent", "tool_calls",
                "preferences", "response", "timestamp"} <= set(record)
