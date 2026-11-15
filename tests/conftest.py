"""Test setup: make the repo-root modules importable."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch):
    """Tests opt in with fake transports; never inherit live provider flags."""
    for name in ("USE_REAL_LLM", "ENABLE_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        monkeypatch.delenv(name, raising=False)
