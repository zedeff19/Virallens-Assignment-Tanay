"""Offline tests for main.py: the graph is replaced by a stub."""
from __future__ import annotations

import main as entry


class StubApp:
    def __init__(self, state: dict) -> None:
        self.state, self.questions = state, []

    def invoke(self, payload: dict) -> dict:
        self.questions.append(payload["question"])
        return self.state


def test_question_is_answered_and_printed(monkeypatch, capsys):
    app = StubApp({"final": "41% [Macquarie, p.1]", "query_type": "lookup", "trace": ["classify -> lookup"]})
    monkeypatch.setattr(entry, "build_app", lambda rebuild_index=False: app)

    code = entry.main(["How", "far", "did", "it", "fall?", "--trace"])

    out = capsys.readouterr().out
    assert code == 0
    assert app.questions == ["How far did it fall?"]
    assert "41% [Macquarie, p.1]" in out and "query_type: lookup" in out


def test_no_question_returns_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(entry, "build_app", lambda rebuild_index=False: (_ for _ in ()).throw(AssertionError))

    assert entry.main([]) == 2


def test_missing_api_key_is_one_line_error(monkeypatch, capsys):
    def boom(rebuild_index=False):
        raise RuntimeError("No LLM provider configured: set at least one of GROQ_API_KEY")

    monkeypatch.setattr(entry, "build_app", boom)

    assert entry.main(["q"]) == 1
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_demo_runs_every_sample_question(monkeypatch, capsys):
    app = StubApp({"final": "ok"})
    monkeypatch.setattr(entry, "build_app", lambda rebuild_index=False: app)

    assert entry.main(["--demo"]) == 0
    assert app.questions == entry.DEMO_QUESTIONS
