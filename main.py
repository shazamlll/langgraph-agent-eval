"""Command-line entry point for the bookstore agent demo.

Usage:
    python main.py demo    # scripted 4-turn demo conversation (default)
    python main.py eval    # run the 8-conversation trajectory eval set
    python main.py chat    # interactive REPL (type 'quit' to exit)

Everything runs offline with the deterministic stub responder. No API key
is needed, and no network calls are made.
"""

from __future__ import annotations

import sys

from agent import BookstoreAgent
from eval import run_eval
from trajectory import TrajectoryLogger

DEMO_TURNS = [
    "Hi!",
    "I like science fiction.",
    "Can you recommend something?",
    "Find me a mystery book under $20.",
]


def run_demo() -> None:
    logger = TrajectoryLogger("trajectories/demo.jsonl")
    agent = BookstoreAgent(logger=logger)
    print("Bookstore assistant — scripted demo (stub responder, offline)\n")
    for user_text in DEMO_TURNS:
        reply = agent.chat(user_text, thread_id="demo")
        print(f"You:   {user_text}")
        print(f"Agent: {reply}\n")
    print(f"Trajectory saved to {logger.path} ({len(logger.records)} turns)")


def run_chat() -> None:
    agent = BookstoreAgent(logger=TrajectoryLogger("trajectories/chat.jsonl"))
    print("Bookstore assistant — interactive chat (stub responder, offline).")
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
