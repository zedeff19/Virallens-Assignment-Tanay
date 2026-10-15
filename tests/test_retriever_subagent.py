"""Offline wiring test for the retriever subagent: scripted fake models, no network, no API keys.

Checks what planv3 listed as risks: the subagent only sees the intended tools, and files it writes
are visible in the parent's state after the `task` call returns.
"""
from __future__ import annotations

import json
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agents import subagents

REPLY = "EVIDENCE: /evidence/macq-drop.md (1 chunks)\nVERDICT: FOUND\nNOTES: [Macquarie, p.1] states the 41% fall."


class ScriptedChat(BaseChatModel):
    """Plays back `script` (a list of AIMessage), records the tool names bound on each call."""

    script: list[Any]
    bound_tools: list[list[str]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        names = [t["function"]["name"] if isinstance(t, dict) and "function" in t else getattr(t, "name", str(t)) for t in tools]
        self.bound_tools.append(names)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self.script[min(self.cursor, len(self.script) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _build(monkeypatch):
    sub_model = ScriptedChat(
        script=[
            _call("search_corpus", {"query": "stock price fell 41%", "doc_id": "macquarie"}, "s1"),
            _call("save_evidence", {"task_id": "macq-drop", "chunk_ids": ["macquarie-0000"], "notes": "drop figure"}, "s2"),
            AIMessage(content=REPLY),
        ]
    )
    parent_model = ScriptedChat(
        script=[
            _call(
                "task",
                {
                    "description": "TASK_ID: macq-drop\nQUESTION: How far did Macquarie's stock fall?\nDOC_HINT: macquarie\nSECTION_HINT: none",
                    "subagent_type": "retriever",
                },
                "p1",
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr(subagents, "get_llm", lambda role: sub_model)
    monkeypatch.setattr(subagents, "get_agent_middleware", lambda role: [])
    agent = create_deep_agent(model=parent_model, subagents=[subagents.retriever_subagent()])
    return agent, sub_model


def test_retriever_tools_files_and_return(monkeypatch):
    agent, sub_model = _build(monkeypatch)
    result = agent.invoke({"messages": [HumanMessage("How far did Macquarie's stock fall?")]})

    # (a) the subagent sees exactly the retriever tools: no write_file / ls / edit_file / glob / grep
    assert sub_model.bound_tools, "subagent model was never called"
    assert set(sub_model.bound_tools[0]) == {"search_corpus", "get_neighbors", "find_citations", "save_evidence", "read_file"}

    # (b) the file written inside the subagent is in the parent's state afterwards
    files = result["files"]
    assert "/evidence/macq-drop.md" in files
    body = files["/evidence/macq-drop.md"]["content"]
    assert "[Macquarie, p.1] · macquarie-0000" in body and "41%" in body

    # (c) the parent got the subagent's last message as the task result
    tool_msgs = [m for m in result["messages"] if getattr(m, "tool_call_id", None) == "p1"]
    assert tool_msgs and tool_msgs[0].content.strip() == REPLY

    # the search result the subagent saw is real corpus JSON (tool wiring, not a stub)
    search_msgs = [m for m in sub_model.script if isinstance(m, AIMessage) and m.tool_calls]
    assert json.loads(json.dumps(search_msgs[0].tool_calls[0]["args"]))["doc_id"] == "macquarie"
