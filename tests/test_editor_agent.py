"""Offline wiring test for the Editor: scripted fake models, no network, no API keys.

Checks what the Editor plan promised: the Editor sees only `task`, `read_file` and `write_todos`, the
`task` tool offers exactly the three project subagents (no `general-purpose`), its prompt matches the
subagents' contracts, and a lookup runs retriever -> read_file -> answer without analyst or verifier.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agents import deep_agent, subagents
from src.agents.prompts import (
    ANALYST_DESCRIPTION,
    CORPUS_CARD,
    EDITOR_PROMPT,
    RETRIEVER_DESCRIPTION,
    VERIFIER_DESCRIPTION,
)
from tests.test_retriever_subagent import ScriptedChat, _call

ANSWER = "Macquarie's stock fell 41% [Macquarie, p.1]."
EVIDENCE_PATH = "/evidence/macq-drop.md"
RETRIEVER_REPLY = f"EVIDENCE: {EVIDENCE_PATH} (1 chunks)\nVERDICT: FOUND\nNOTES: [Macquarie, p.1] states the 41% fall."


class RecordingChat(ScriptedChat):
    """ScriptedChat that also keeps the full tool objects, to read their descriptions."""

    seen_tools: list[Any] = []

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        self.seen_tools.append(list(tools))
        return super().bind_tools(tools, **kwargs)


def _name(tool: Any) -> str:
    return tool["function"]["name"] if isinstance(tool, dict) and "function" in tool else tool.name


def _description(tool: Any) -> str:
    return tool["function"]["description"] if isinstance(tool, dict) and "function" in tool else tool.description


def _wire(monkeypatch, editor: ScriptedChat, retriever: ScriptedChat, other: ScriptedChat):
    models = {"retriever": retriever, "analyst": other, "verifier": other}
    monkeypatch.setattr(deep_agent, "get_llm", lambda role: editor)
    monkeypatch.setattr(deep_agent, "get_agent_middleware", lambda role: [])
    monkeypatch.setattr(subagents, "get_llm", lambda role: models[role])
    monkeypatch.setattr(subagents, "get_agent_middleware", lambda role: [])


def test_editor_tools_and_subagents(monkeypatch):
    editor = RecordingChat(script=[AIMessage(content=ANSWER)])
    idle = ScriptedChat(script=[AIMessage(content="")])
    _wire(monkeypatch, editor, idle, idle)

    deep_agent.ask_editor(deep_agent.build_editor_agent(), "How far did Macquarie's stock fall?", "lookup")

    # (a) the Editor can only delegate, read and plan: no write_file / ls / edit_file / glob / grep
    assert set(editor.bound_tools[0]) == {"task", "read_file", "write_todos"}

    # (b) `task` offers our three subagents; deepagents' default general-purpose (all tools) is replaced
    # by the inert stub, so its "access to all tools" description must be gone
    task = next(t for t in editor.seen_tools[0] if _name(t) == "task")
    text = _description(task)
    for name in ("retriever", "analyst", "verifier"):
        assert f"- {name}:" in text
    assert "- general-purpose: Unused." in text
    assert "all tools as the main agent" not in text

    # (c) the todo tool uses the short description, not langchain's ~1K-token default
    todos = next(t for t in editor.seen_tools[0] if _name(t) == "write_todos")
    assert len(_description(todos)) < 300


def test_editor_prompt_matches_subagent_contracts():
    assert CORPUS_CARD in EDITOR_PROMPT
    assert "{corpus_card}" not in EDITOR_PROMPT
    # input fields each subagent expects (see RETRIEVER/ANALYST/VERIFIER descriptions)
    for field in ("TASK_ID", "QUESTION", "DOC_HINT", "SECTION_HINT", "EVIDENCE", "FOCUS", "SIDE", "DRAFT"):
        assert field in EDITOR_PROMPT
    for description, fields in (
        (RETRIEVER_DESCRIPTION, ("TASK_ID", "QUESTION", "DOC_HINT", "SECTION_HINT")),
        (ANALYST_DESCRIPTION, ("TASK_ID", "EVIDENCE", "FOCUS", "SIDE")),
        (VERIFIER_DESCRIPTION, ("DRAFT", "EVIDENCE")),
    ):
        assert all(f in description for f in fields), "subagent description drifted from this test"
    # reply keys it must recognise, and the routing it must follow
    for key in ("VERDICT", "FOUND", "PARTIAL", "NONE", "STATUS", "PASS", "FIX"):
        assert key in EDITOR_PROMPT
    for focus in ("holding", "argument", "timeline"):
        assert focus in EDITOR_PROMPT
    assert "No analyst, no verifier" in EDITOR_PROMPT


def test_lookup_runs_retriever_then_read_file_only(monkeypatch):
    retriever = ScriptedChat(
        script=[
            _call("search_corpus", {"query": "stock price fell 41%", "doc_id": "macquarie"}, "s1"),
            _call("save_evidence", {"task_id": "macq-drop", "chunk_ids": ["macquarie-0000"], "notes": "drop figure"}, "s2"),
            AIMessage(content=RETRIEVER_REPLY),
        ]
    )
    editor = ScriptedChat(
        script=[
            _call(
                "task",
                {
                    "description": "TASK_ID: macq-drop\nQUESTION: How far did Macquarie's stock fall?\nDOC_HINT: macquarie\nSECTION_HINT: none",
                    "subagent_type": "retriever",
                },
                "e1",
            ),
            _call("read_file", {"file_path": EVIDENCE_PATH}, "e2"),
            AIMessage(content=ANSWER),
        ]
    )
    idle = ScriptedChat(script=[AIMessage(content="")])
    _wire(monkeypatch, editor, retriever, idle)

    out = deep_agent.ask_editor(deep_agent.build_editor_agent(), "How far did Macquarie's stock fall?", "lookup")

    assert out["answer"] == ANSWER
    assert EVIDENCE_PATH in out["files"]
    # the Editor read the evidence it was pointed at, and the file really holds the saved passage
    read = next(m for m in out["messages"] if getattr(m, "tool_call_id", None) == "e2")
    assert "41%" in str(read.content)
    # type 1 never reaches the analyst or the verifier
    assert idle.bound_tools == []
    assert not any(p.startswith("/notes/") for p in out["files"])


def test_ask_editor_passes_hint_and_config():
    class FakeAgent:
        def invoke(self, state, config=None):
            self.state, self.config = state, config
            return {"messages": [AIMessage(content="draft"), AIMessage(content=""), AIMessage(content="final")]}

    agent = FakeAgent()
    out = deep_agent.ask_editor(agent, "Timeline of SEC v. Chiueh?", "timeline", thread_id="t1")

    assert agent.state["messages"][0].content == "QUERY_TYPE: timeline\nQUESTION: Timeline of SEC v. Chiueh?"
    assert agent.config["max_concurrency"] == deep_agent.EDITOR_CONFIG["max_concurrency"] == 1
    assert agent.config["recursion_limit"] == deep_agent.EDITOR_CONFIG["recursion_limit"] < 100
    assert agent.config["configurable"] == {"thread_id": "t1"}
    assert out["answer"] == "final"  # last non-empty AI message
    assert json.dumps(out["files"]) == "{}" and out["todos"] == []


def test_ask_editor_normalizes_fullwidth_citation_brackets():
    class FakeAgent:
        def invoke(self, state, config=None):
            return {"messages": [AIMessage(content="It fell 41%.【Macquarie, p.1】")]}

    out = deep_agent.ask_editor(FakeAgent(), "How far did Macquarie's stock fall?", "lookup")
    assert out["answer"] == "It fell 41%.[Macquarie, p.1]"
