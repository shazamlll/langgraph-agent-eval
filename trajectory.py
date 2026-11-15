"""Trajectory logging for the bookstore agent.

Every turn is recorded as one JSON line: the user message, the classified
intent, the tool calls made (name, arguments, result summary), the agent's
response, and a snapshot of the remembered preferences at that point.

The logger works in-memory by default; pass a ``path`` to also append each
turn to a JSONL file. The file is created on first write.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class TrajectoryLogger:
    """Append-only trajectory log, in-memory and optionally on disk (JSONL)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []

    def log_turn(self, record: dict) -> dict:
        """Record one turn. Adds a UTC timestamp if missing. Returns it."""
        entry = {"timestamp": _utcnow(), **record}
        self.records.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def last(self) -> dict | None:
        """The most recently logged turn, or None."""
        return self.records[-1] if self.records else None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
