"""Offline check that LangSmith run names identify the agent, subagent, delegation, model and tool.

Runs the real Editor graph on scripted models with a `LabelledTracer` whose network calls are
replaced by an in-memory recorder, then reads the names that would have been sent.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tracers.schemas import Run
from langsmith import Client

from src.agents import deep_agent
from src.tracing import LabelledTracer, label_run
from tests.test_editor_agent import ANSWER, EVIDENCE_PATH, RETRIEVER_REPLY, _wire
from tests.test_retriever_subagent import ScriptedChat, _call


@pytest.fixture(autouse=True)
def no_langsmith_network(monkeypatch):
    # RecordingTracer only intercepts LangChain-level runs. The @traceable spans that langchain's agent
    # middleware opens under it are posted through the LangSmith client directly, so without this the
    # test sends traces to api.smith.langchain.com (a 401 without a key, junk traces in your project with one).
    monkeypatch.setattr(Client, "create_run", lambda self, *a, **k: None)
    monkeypatch.setattr(Client, "update_run", lambda self, *a, **k: None)


class RecordingTracer(LabelledTracer):
    # LangChain copies a tracer with `self.__class__(**kwargs)` per child scope, so the recorder is
    # class-level (shared by every copy) and the constructor keeps LangChainTracer's signature.
    runs: list[Run] = []

    def _persist_run_single(self, run: Run) -> None:
        self.runs.append(run)

    def _update_run_single(self, run: Run) -> None:
        pass


def _run(monkeypatch) -> list[Run]:
    retriever = ScriptedChat(
        script=[
            _call("search_corpus", {"query": "stock price fell 41%", "doc_id": "macquarie"}, "s1"),
            _call("save_evidence", {"task_id": "macq-drop", "chunk_ids": ["macquarie-0000"], "notes": "drop"}, "s2"),
            AIMessage(content=RETRIEVER_REPLY),
        ]
    )
    editor = ScriptedChat(
        script=[
            _call(
                "task",
                {"description": "TASK_ID: macq-drop\nQUESTION: q\nDOC_HINT: macquarie\nSECTION_HINT: none", "subagent_type": "retriever"},
                "e1",
            ),
            _call("read_file", {"file_path": EVIDENCE_PATH}, "e2"),
            AIMessage(content=ANSWER),
        ]
    )
    idle = ScriptedChat(script=[AIMessage(content="")])
    _wire(monkeypatch, editor, retriever, idle)

    RecordingTracer.runs = []
    agent = deep_agent.build_editor_agent()
    agent.invoke(
        {"messages": [HumanMessage("QUERY_TYPE: lookup\nQUESTION: q")]},
        config={"run_name": "editor", "callbacks": [RecordingTracer()], "recursion_limit": 40},
    )
    return RecordingTracer.runs


def test_run_names_identify_each_component(monkeypatch):
    names = [r.name for r in _run(monkeypatch)]

    assert names[0] == "AGENT editor"
    assert "TASK -> retriever" in names
    assert "SUBAGENT retriever" in names
    assert "TOOL retriever_search (retriever)" not in names  # scripted tool is search_corpus, not that name
    assert any(n.startswith("TOOL save_evidence") and n.endswith("(retriever)") for n in names)
    assert "TOOL read_file (editor)" in names
    assert "model step (editor)" in names and "model step (retriever)" in names
    assert "tools step (retriever)" in names
    assert sum(n.startswith("LLM ") and n.endswith("(retriever)") for n in names) == 3
    assert sum(n.startswith("LLM ") and n.endswith("(editor)") for n in names) == 3
    # nothing generic is left for the person reading the trace to guess at
    assert not {"model", "tools", "task"} & set(names)


def test_runs_are_tagged_for_filtering(monkeypatch):
    runs = _run(monkeypatch)
    sub = next(r for r in runs if r.name == "SUBAGENT retriever")
    assert {"kind:subagent", "agent:retriever"} <= set(sub.tags)
    delegate = next(r for r in runs if r.name == "TASK -> retriever")
    assert "kind:delegate" in delegate.tags


def test_unrelated_runs_keep_their_name():
    run = Run(
        id="00000000-0000-0000-0000-000000000000",
        name="PatchToolCallsMiddleware.before_agent",
        start_time="2026-01-01T00:00:00Z",
        run_type="chain",
        inputs={},
        serialized={},
    )
    assert label_run(run) is None
