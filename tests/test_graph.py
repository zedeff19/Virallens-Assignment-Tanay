"""Offline tests for the outer LangGraph shell: stand-in classifier and Editor, no network, no API keys."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.graph import Classification, build_graph, extract_citations

EVIDENCE = {
    "/evidence/macq.md": {
        "content": (
            "# Evidence: macq\nRetriever notes: x\n\n"
            "## [Macquarie, pp.5–7] · macquarie-0012 · Opinion > II\n\nPure omissions are not actionable.\n\n"
            "## [Chiueh ¶40–47] · sec_chiueh-0003 · FACTS\n\nThe concentration policy.\n"
        )
    },
    "/notes/macq.md": {"content": "# Note"},
}


class FakeClassifier:
    def __init__(self, query_type: str = "lookup", error: Exception | None = None) -> None:
        self.query_type, self.error, self.calls = query_type, error, 0

    def invoke(self, messages: Any) -> Classification:
        self.calls += 1
        if self.error:
            raise self.error
        return Classification(query_type=self.query_type, reason="test")  # type: ignore[arg-type]


class FakeEditor:
    """Mimics a compiled agent: `invoke` returns the same shape `ask_editor` reads."""

    def __init__(self, answer: str = "", files: dict | None = None, error: Exception | None = None) -> None:
        self.answer, self.files, self.error, self.payloads = answer, files or {}, error, []

    def invoke(self, payload: dict, config: dict | None = None) -> dict:
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return {"messages": [AIMessage(content=self.answer)], "files": self.files, "todos": []}


def _run(question: str, classifier: FakeClassifier, editor: FakeEditor) -> dict:
    return build_graph(editor_agent=editor, classifier=classifier).invoke({"question": question})


def test_out_of_scope_refuses_without_calling_the_editor():
    editor = FakeEditor(answer="should not be used")
    state = _run("What is the weather in Delhi?", FakeClassifier("out_of_scope"), editor)

    assert editor.payloads == []
    assert "Macquarie v. Moab" in state["final"]
    assert any(line.startswith("refuse") for line in state["trace"])


def test_in_scope_passes_query_type_and_appends_sources():
    editor = FakeEditor("Pure omissions are not actionable [Macquarie, p.5].", EVIDENCE)
    state = _run("What did Macquarie hold?", FakeClassifier("concept"), editor)

    assert "QUERY_TYPE: concept" in editor.payloads[0]["messages"][0].content
    assert state["query_type"] == "concept"
    assert "Sources" in state["final"] and "Macquarie v. Moab" in state["final"]
    assert state["citations"][0]["verified"] is True
    assert "/evidence/macq.md" in state["files"]


def test_classifier_failure_falls_back_to_unknown_and_still_answers():
    editor = FakeEditor("Answer [Macquarie, p.6].", EVIDENCE)
    state = _run("anything", FakeClassifier(error=RuntimeError("all providers down")), editor)

    assert state["query_type"] == "unknown"
    assert len(editor.payloads) == 1
    assert "QUERY_TYPE: unknown" in editor.payloads[0]["messages"][0].content


def test_unverified_citation_is_flagged_not_listed_as_a_source():
    editor = FakeEditor("A claim [Macquarie, p.99] and a fact [Macquarie, p.5].", EVIDENCE)
    state = _run("q", FakeClassifier("lookup"), editor)

    flags = {c["text"]: c["verified"] for c in state["citations"]}
    assert flags == {"[Macquarie, p.99]": False, "[Macquarie, p.5]": True}
    assert "Warning: not found in the retrieved evidence: [Macquarie, p.99]" in state["final"]
    assert "- [Macquarie, p.99]" not in state["final"]


def test_page_and_paragraph_ranges_in_evidence_headings_count():
    files = {p: f["content"] for p, f in EVIDENCE.items()}
    cites = extract_citations("[Macquarie, p.7] [Chiueh ¶45] [Chiueh ¶48] [Last Atlantis, p.1]", files)

    assert [c["verified"] for c in cites] == [True, True, False, False]


def test_editor_failure_is_reported_not_raised():
    editor = FakeEditor(error=RuntimeError("429 daily quota exhausted"))
    state = _run("q", FakeClassifier("compare"), editor)

    assert "RuntimeError" in state["error"]
    assert state["final"].startswith("I could not complete this question")
