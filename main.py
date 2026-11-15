"""Command-line entry point for the bookstore agent demo.

Usage:
    python main.py demo    # scripted 4-turn demo conversation (default)
    python main.py eval    # run the 8-conversation trajectory eval set
    python main.py chat    # interactive REPL (type 'quit' to exit)

The default stub runs offline. USE_REAL_LLM=1 enables paid provider requests;
ENABLE_TRACING=1 with a LangSmith key enables turn exports.
"""

from __future__ import annotations

import sys

from agent import BookstoreAgent
from eval import run_eval
from trajectory import TrajectoryLogger
from llm import StubResponder

DEMO_TURNS = [
    "Hi!",
    "I like science fiction.",
    "Can you recommend something?",
    "Find me a mystery book under $20.",
]


def _mode_label(agent: BookstoreAgent) -> str:
    if isinstance(agent.llm, StubResponder):
        return "stub responder, offline inference"
    return f"{type(agent.llm).__name__}, model={getattr(agent.llm, 'model', 'unknown')}"


def run_demo() -> None:
    logger = TrajectoryLogger("trajectories/demo.jsonl")
    agent = BookstoreAgent(logger=logger)
    print(f"Bookstore assistant — scripted demo ({_mode_label(agent)})\n")
    for user_text in DEMO_TURNS:
        reply = agent.chat(user_text, thread_id="demo")
        print(f"You:   {user_text}")
        print(f"Agent: {reply}\n")
    print(f"Trajectory saved to {logger.path} ({len(logger.records)} turns)")


def run_chat() -> None:
    agent = BookstoreAgent(logger=TrajectoryLogger("trajectories/chat.jsonl"))
    print(f"Bookstore assistant — interactive chat ({_mode_label(agent)}).")
    print("Type 'quit' to exit.\n")
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_text:
            continue
        print(f"Agent: {agent.chat(user_text, thread_id='chat')}\n")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "demo"
    if command == "demo":
        run_demo()
    elif command == "eval":
        report_path = argv[2] if len(argv) > 2 else "eval_reports/latest.json"
        report = run_eval(report_path=report_path)
        return 0 if report["passed"] else 1
    elif command == "chat":
        run_chat()
    else:
        print(f"Unknown command: {command}\n\n{__doc__}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
