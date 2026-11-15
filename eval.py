"""Trajectory evaluation for the bookstore agent.

Rubric — three deterministic, offline dimensions, each scored 0.0–1.0:

1. ``tool_use`` — did the agent call exactly the tools the scripted turn
   expects (usually ``search_catalog`` when a search is needed, none when
   it is not)?
2. ``memory_use`` — on turns that should draw on a preference stated
   earlier, does the response reflect that preference (mentions the genre,
   or a book from that genre)?
3. ``answer_quality`` — does the response contain the expected book titles
   and/or expected phrases?

The test set is ``eval_data/conversations.json``: 8 scripted multi-turn
conversations with per-turn expectations. The runner executes each one
against a fresh agent thread, prints a summary table, and writes an optional
JSON report. A conversation's score is the mean of the dimensions that
apply to it; the aggregate is the mean over conversations. PASS threshold
is 0.80.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent import BookstoreAgent
from catalog import BOOKS

PASS_THRESHOLD = 0.80
EVAL_DATA_PATH = Path(__file__).parent / "eval_data" / "conversations.json"


def load_conversations(path: str | Path = EVAL_DATA_PATH) -> list[dict]:
    """Load the scripted multi-turn test set."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_turn(turn: dict, actual: dict) -> dict:
    """Score one turn against its expectations. Returns dimension scores."""
    scores: dict[str, float] = {}
    response = str(actual.get("response", ""))
    lowered = response.lower()

    if "expect_tools" in turn:
        expected = set(turn.get("expect_tools") or [])
        actual_names = {c.get("name") for c in actual.get("tool_calls", [])}
        scores["tool_use"] = 1.0 if expected == actual_names else 0.0

    if turn.get("expect_titles") or turn.get("expect_phrase"):
        hits = 0
        total = 0
        for title in turn.get("expect_titles") or []:
            total += 1
            hits += 1 if title.lower() in lowered else 0
        if turn.get("expect_phrase"):
            total += 1
            hits += 1 if turn["expect_phrase"].lower() in lowered else 0
        scores["answer_quality"] = hits / total if total else 1.0

    genre = turn.get("expect_memory_genre")
    if genre:
        genre_hit = genre.lower() in lowered
        title_hit = any(
            b["title"].lower() in lowered for b in BOOKS if b["genre"] == genre
        )
        scores["memory_use"] = 1.0 if (genre_hit or title_hit) else 0.0
    return scores


def _dimension_means(turn_scores: list[dict]) -> dict[str, float]:
    means: dict[str, float] = {}
    for dim in ("tool_use", "memory_use", "answer_quality"):
        values = [s[dim] for s in turn_scores if dim in s]
        if values:
            means[dim] = sum(values) / len(values)
    return means


def evaluate_conversation(agent: BookstoreAgent, conversation: dict) -> dict:
    """Run one scripted conversation and score it."""
    thread_id = f"eval-{conversation['id']}"
    turn_scores = []
    for turn in conversation["turns"]:
        agent.chat(turn["user"], thread_id=thread_id)
        turn_scores.append(score_turn(turn, agent.last_turn or {}))
    means = _dimension_means(turn_scores)
    overall = sum(means.values()) / len(means) if means else 0.0
    return {
        "id": conversation["id"],
        "description": conversation.get("description", ""),
        "dimensions": means,
        "overall": round(overall, 3),
        "passed": overall >= PASS_THRESHOLD,
        "turn_scores": turn_scores,
    }


def run_eval(
    conversations: list[dict] | None = None,
    agent: BookstoreAgent | None = None,
    report_path: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the full eval set. Returns the report dict (and prints a table)."""
    conversations = conversations if conversations is not None else load_conversations()
    agent = agent if agent is not None else BookstoreAgent()
    results = [evaluate_conversation(agent, c) for c in conversations]

    aggregate: dict[str, float] = {}
    for dim in ("tool_use", "memory_use", "answer_quality"):
        values = [r["dimensions"][dim] for r in results if dim in r["dimensions"]]
        if values:
            aggregate[dim] = round(sum(values) / len(values), 3)
    overall = round(sum(r["overall"] for r in results) / len(results), 3) if results else 0.0
    report = {
        "llm": {"adapter": type(agent.llm).__name__, "model": getattr(agent.llm, "model", None)},
        "threshold": PASS_THRESHOLD,
        "conversations": results,
        "aggregate": aggregate,
        "overall": overall,
        "passed": overall >= PASS_THRESHOLD,
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    if verbose:
        _print_report(report)
    return report


def _fmt(value: float | None, width: int) -> str:
    """Format a dimension score, or a dash when the dimension doesn't apply."""
    return f"{value:>{width}.2f}" if value is not None else "—".rjust(width)


def _print_report(report: dict) -> None:
    print(f"Evaluation LLM: {report['llm']['adapter']} (model={report['llm']['model']})")
    print(f"\n{'conversation':<12} {'tool':>6} {'memory':>7} {'answer':>7} {'overall':>8}  result")
    for r in report["conversations"]:
        d = r["dimensions"]
        print(
            f"{r['id']:<12} "
            f"{_fmt(d.get('tool_use'), 6)} "
            f"{_fmt(d.get('memory_use'), 7)} "
            f"{_fmt(d.get('answer_quality'), 7)} "
            f"{r['overall']:>8.2f}  {'PASS' if r['passed'] else 'FAIL'}"
        )
    agg = report["aggregate"]
    print("-" * 62)
    print(
        f"{'aggregate':<12} "
        f"{_fmt(agg.get('tool_use'), 6)} "
        f"{_fmt(agg.get('memory_use'), 7)} "
        f"{_fmt(agg.get('answer_quality'), 7)} "
        f"{report['overall']:>8.2f}  "
        f"{'PASS' if report['passed'] else 'FAIL'} (threshold {report['threshold']:.2f})"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the trajectory eval set.")
    parser.add_argument("--report", default=None, help="Write JSON report to this path.")
    args = parser.parse_args()
    run_eval(report_path=args.report)
