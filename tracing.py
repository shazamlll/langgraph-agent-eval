"""Tracing hooks for LangSmith / LangFuse. DISABLED by default.

This module is scaffolding: it defines the exact seam where observability
plugs in, but it performs no network calls and imports no tracing SDK unless
tracing is explicitly enabled.

Enablement (follow-up work only — see README "Remaining work"):

1. ``pip install langsmith`` and/or ``langfuse``
2. Set ``ENABLE_TRACING=1`` plus the provider keys, e.g.
   ``LANGSMITH_API_KEY`` / ``LANGCHAIN_TRACING_V2=true`` or
   ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``.
3. Call :func:`init_tracing` once at startup; it returns a status dict.

Until then, :func:`init_tracing` returns ``{"enabled": False, ...}`` and
:func:`trace_turn` is a no-op, so the agent and the eval runner stay fully
offline.
"""

from __future__ import annotations

import os


def init_tracing() -> dict:
    """Attempt to enable tracing. Safe to call when disabled (the default).

    Returns a status dict: ``{"enabled": bool, "provider": str | None,
    "reason": str}``. Never raises for missing SDKs or keys — it reports
    them instead, so a demo run cannot fail on tracing setup.
    """
    if os.environ.get("ENABLE_TRACING") != "1":
        return {
            "enabled": False,
            "provider": None,
            "reason": "ENABLE_TRACING is not '1' (tracing disabled by default)",
        }
    if os.environ.get("LANGSMITH_API_KEY"):
        try:
            import langsmith  # noqa: F401  (imported for its side effects)

            return {
                "enabled": True,
                "provider": "langsmith",
                "reason": "LANGSMITH_API_KEY present",
            }
        except ImportError:
            return {
                "enabled": False,
                "provider": None,
                "reason": "langsmith package not installed (pip install langsmith)",
            }
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        try:
            import langfuse  # noqa: F401  (imported for its side effects)

            return {
                "enabled": True,
                "provider": "langfuse",
                "reason": "LANGFUSE_PUBLIC_KEY present",
            }
        except ImportError:
            return {
                "enabled": False,
                "provider": None,
                "reason": "langfuse package not installed (pip install langfuse)",
            }
    return {
        "enabled": False,
        "provider": None,
        "reason": "ENABLE_TRACING=1 but no provider key found "
        "(set LANGSMITH_API_KEY or LANGFUSE_PUBLIC_KEY)",
    }


def trace_turn(record: dict, status: dict | None = None) -> None:
    """Send one trajectory turn to the tracing provider.

    No-op unless ``status`` says tracing is enabled. The actual provider
    call is a follow-up: this function currently only marks the seam.
    TODO(follow-up): implement the LangSmith / LangFuse client call here.
    """
    if not status or not status.get("enabled"):
        return
    # TODO(follow-up): forward `record` to the configured provider.
    return None
