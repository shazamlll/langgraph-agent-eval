"""Tests for the trajectory eval rubric and the eval runner."""

from eval import PASS_THRESHOLD, evaluate_conversation, load_conversations, run_eval, score_turn


def _actual(tools=(), response=""):
    return {"tool_calls": [{"name": name} for name in tools], "response": response}


def test_score_turn_perfect():
    turn = {
        "expect_tools": ["search_catalog"],
        "expect_titles": ["The Quiet Witness"],
        "expect_memory_genre": "mystery",
    }
    actual = _actual(
        tools=("search_catalog",),
        response="Since you like mystery, I'd recommend The Quiet Witness.",
    )
    assert score_turn(turn, actual) == {
        "tool_use": 1.0,
        "memory_use": 1.0,
        "answer_quality": 1.0,
    }


def test_score_turn_wrong_tool_and_missing_title():
    turn = {
        "expect_tools": ["search_catalog"],
        "expect_titles": ["The Quiet Witness"],
    }
    actual = _actual(tools=(), response="I don't know that one.")
    assert score_turn(turn, actual) == {"tool_use": 0.0, "answer_quality": 0.0}


def test_score_turn_phrase_expectation():
    turn = {"expect_tools": ["search_catalog"], "expect_phrase": "don't have any"}
    assert score_turn(turn, _actual(("search_catalog",), "I don't have any books.")) == {
        "tool_use": 1.0,
        "answer_quality": 1.0,
    }
    assert score_turn(turn, _actual(("search_catalog",), "Here you go!"))[
        "answer_quality"
    ] == 0.0


def test_score_turn_memory_miss():
    turn = {"expect_memory_genre": "science fiction"}
    assert score_turn(turn, _actual(response="Here is a romance novel.")) == {
        "memory_use": 0.0
    }


def test_eval_set_loads_eight_conversations():
    conversations = load_conversations()
    assert len(conversations) == 8
    assert all(c["turns"] for c in conversations)


def test_full_eval_passes_threshold():
    report = run_eval(verbose=False)
    assert len(report["conversations"]) == 8
    assert report["overall"] >= PASS_THRESHOLD
    assert report["passed"] is True
    assert all(r["passed"] for r in report["conversations"])
