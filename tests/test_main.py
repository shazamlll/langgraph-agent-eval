"""CLI mode claims must follow the selected adapter."""

import main


def test_demo_banner_identifies_real_adapter(monkeypatch, capsys):
    class FakeAdapter:
        model = "test-model"
    class FakeAgent:
        llm = FakeAdapter()
        def __init__(self, **kwargs):
            pass
        def chat(self, *args, **kwargs):
            return "mock reply"
    monkeypatch.setattr(main, "BookstoreAgent", FakeAgent)
    monkeypatch.setattr(main, "DEMO_TURNS", [])
    main.run_demo()
    output = capsys.readouterr().out
    assert "FakeAdapter" in output
    assert "test-model" in output
    assert "stub responder, offline" not in output
