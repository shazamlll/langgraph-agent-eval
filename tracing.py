"""Opt-in, best-effort LangSmith turn tracing; no SDK client by default."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

_logger = logging.getLogger(__name__)


def init_tracing(*, client_factory=None) -> dict:
    """Initialize a synchronous LangSmith client only with explicit opt-in.

    The returned status holds the client privately for reuse across turns.
    Setup failures disable tracing and never stop the offline demo.
    """
    disabled = {"enabled": False, "provider": None}
    if os.environ.get("ENABLE_TRACING") != "1":
        return {**disabled, "reason": "ENABLE_TRACING is not '1' (tracing disabled by default)"}
    key = os.environ.get("LANGSMITH_API_KEY")
    if not key:
        return {**disabled, "reason": "ENABLE_TRACING=1 but no provider key found (set LANGSMITH_API_KEY)"}
    try:
        if client_factory is None:
            from langsmith import Client
            client_factory = Client
        client = client_factory(api_key=key, auto_batch_tracing=False, timeout_ms=5000)
    except ImportError:
        return {**disabled, "reason": "langsmith package not installed (pip install langsmith)"}
    except Exception:
        return {**disabled, "reason": "LangSmith client initialization failed"}
    return {
        "enabled": True, "provider": "langsmith", "reason": "LangSmith client configured (upload unverified)",
        "_client": client, "project": os.environ.get("LANGSMITH_PROJECT", "langgraph-agent-eval"),
    }


def trace_turn(record: dict, status: dict | None = None) -> None:
    """Send one completed turn; disable further uploads after an error.

    This is a turn summary, not a detailed graph/LLM span tree. Times mark
    export time; no inference latency is claimed. User text, responses and
    synthetic catalog metadata are exported only when tracing is enabled.
    """
    if not status or not status.get("enabled"):
        return
    now = datetime.now(timezone.utc)
    try:
        status["_client"].create_run(
            name="bookstore.turn", run_type="chain", id=uuid4(),
            inputs={"user": record.get("user", "")},
            outputs={"response": record.get("response", "")},
            project_name=status.get("project", "langgraph-agent-eval"),
            start_time=now, end_time=now,
            extra={"metadata": {key: value for key, value in record.items() if key not in ("user", "response")}},
        )
    except Exception:
        status.update(enabled=False, reason="LangSmith trace upload failed; further uploads disabled")
        # External exception text may contain credentials; never echo it.
        _logger.warning("LangSmith trace upload failed; further uploads disabled")
