"""Offline wiring test for the analyst and verifier subagents: scripted fake models, no network.

Checks the same risks as the retriever test: each subagent sees exactly its intended tools, a note the
analyst writes is visible in the parent's state after the `task` call, and the parent gets the
subagent's last message as the task result.
"""
from __future__ import annotations

from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import subagents
from src.agents.prompts import ANALYST_DESCRIPTION, VERIFIER_DESCRIPTION
from tests.test_retriever_subagent import ScriptedChat, _call

EVIDENCE_PATH = "/evidence/macq-drop.md"
EVIDENCE = (
    "# Evidence: macq-drop\n"
    "Retriever notes: drop figure\n\n"
    "## [Macquarie, p.1] · macquarie-0000 · Opinion\n"
    "The stock price fell 41%.\n"
)
NOTE_PATH = "/notes/macq-note.md"
NOTE_BODY = "RULE: the stock price fell 41%. [Macquarie, p.1]"
NOTE_ARGS = {"task_id": "macq-note", "focus": "holding", "side": "macquarie", "content": NOTE_BODY}
ANALYST_REPLY = "NOTE: /notes/macq-note.md (holding)\nSTATUS: COMPLETE\nGAPS: none"
VERIFIER_REPLY = 'FIX: 1 issues\n- "fell 14%" | WRONG_NUMBER | the stock price fell 41% [Macquarie, p.1]'


def _task_result(result, call_id: str) -> str:
    msgs = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == call_id]
    assert msgs, f"no tool message for {call_id}"
    return msgs[0].content.strip()


def _build(monkeypatch, *, subagent_name: str, spec, sub_script: list, description: str):
    sub_model = ScriptedChat(script=sub_script)
    parent_model = ScriptedChat(
        script=[
            _call("task", {"description": description, "subagent_type": subagent_name}, "p1"),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr(subagents, "get_llm", lambda role: sub_model)
    monkeypatch.setattr(subagents, "get_agent_middleware", lambda role: [])
    agent = create_deep_agent(model=parent_model, subagents=[spec()])
    return agent, sub_model


def _invoke(agent, question: str):
    return agent.invoke(
        {
            "messages": [HumanMessage(question)],
            "files": {EVIDENCE_PATH: create_file_data(EVIDENCE)},
        }
    )


def test_analyst_tools_note_and_return(monkeypatch):
    agent, sub_model = _build(
        monkeypatch,
        subagent_name="analyst",
        spec=subagents.analyst_subagent,
        sub_script=[
            _call("read_file", {"file_path": EVIDENCE_PATH}, "a1"),
            _call("save_note", NOTE_ARGS, "a2"),
            AIMessage(content=ANALYST_REPLY),
        ],
        description=f"TASK_ID: macq-note\nEVIDENCE: {EVIDENCE_PATH}\nFOCUS: holding\nSIDE: macquarie",
    )
    result = _invoke(agent, "What did Macquarie hold?")

    # (a) exactly read_file and save_note: no write_file / ls / edit_file / glob / grep, and no inherited Editor tools
    assert sub_model.bound_tools, "subagent model was never called"
    assert set(sub_model.bound_tools[0]) == {"read_file", "save_note"}

    # (b) the note saved by the analyst's own save_note is in the parent's state afterwards, with the
    # header written by the tool rather than the model
    files = result["files"]
    assert NOTE_PATH in files
    assert files[NOTE_PATH]["content"].startswith("# Note: macq-note | macquarie | holding\n")
    assert "[Macquarie, p.1]" in files[NOTE_PATH]["content"]
    assert EVIDENCE_PATH in files  # the evidence it read is still there, untouched
    assert files[EVIDENCE_PATH]["content"] == EVIDENCE

    # (c) the parent got the analyst's 3-line final message as the task result
    assert _task_result(result, "p1") == ANALYST_REPLY


def test_analyst_repeated_save_note_is_cut_off(monkeypatch):
    """Regression for the trace where Qwen re-issued the same write 8x and never sent its reply."""
    repeats = [_call("save_note", NOTE_ARGS, f"a{i}") for i in range(2, 9)]
    agent, sub_model = _build(
        monkeypatch,
        subagent_name="analyst",
        spec=subagents.analyst_subagent,
        sub_script=[_call("read_file", {"file_path": EVIDENCE_PATH}, "a1"), *repeats, AIMessage(content=ANALYST_REPLY)],
        description=f"TASK_ID: macq-note\nEVIDENCE: {EVIDENCE_PATH}\nFOCUS: holding\nSIDE: macquarie",
    )
    result = _invoke(agent, "What did Macquarie hold?")

    # read_file, save_note, save_note, then the 3rd save_note is blocked and the run ends
    assert sub_model.cursor == 4
    assert NOTE_PATH in result["files"]  # the first save survives the forced stop
    assert "save_note" in _task_result(result, "p1")


def test_verifier_tools_and_return(monkeypatch):
    agent, sub_model = _build(
        monkeypatch,
        subagent_name="verifier",
        spec=subagents.verifier_subagent,
        sub_script=[
            _call("read_file", {"file_path": EVIDENCE_PATH}, "v1"),
            AIMessage(content=VERIFIER_REPLY),
        ],
        description=f"DRAFT: The stock price fell 14%.\nEVIDENCE: {EVIDENCE_PATH}",
    )
    result = _invoke(agent, "Check this draft.")

    # (a) read-only: read_file and nothing else
    assert sub_model.bound_tools, "subagent model was never called"
    assert set(sub_model.bound_tools[0]) == {"read_file"}

    # (b) it wrote nothing: the files in state are exactly the ones we seeded
    assert set(result["files"]) == {EVIDENCE_PATH}

    # (c) the parent got the verifier's fix list as the task result
    assert _task_result(result, "p1") == VERIFIER_REPLY


def test_specs_are_complete_and_descriptions_short(monkeypatch):
    monkeypatch.setattr(subagents, "get_llm", lambda role: ScriptedChat(script=[AIMessage(content="")]))
    monkeypatch.setattr(subagents, "get_agent_middleware", lambda role: [])
    # `tools` must be an explicit key: a spec without it inherits the parent's tools
    expected_tools = {"analyst": ["save_note"], "verifier": []}
    for name, factory in (("analyst", subagents.analyst_subagent), ("verifier", subagents.verifier_subagent)):
        spec = factory()
        assert spec["name"] == name
        assert [t.name for t in spec["tools"]] == expected_tools[name]
        assert spec["system_prompt"].strip()
    assert 0 < len(ANALYST_DESCRIPTION) <= 450
    assert 0 < len(VERIFIER_DESCRIPTION) <= 450
