"""Deep agent "Editor": plans with a to-do list, delegates to subagents, writes the final answer.

The Editor has only `task`, `read_file` and a short `write_todos`. Searching is the retriever's job, so
the Editor's tool schemas stay at ~1K tokens against Groq's 8K TPM (the deepagents default is ~2.4K).
"""
from __future__ import annotations

import os
from typing import Any

from deepagents import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import tracing_context

from src.agents.prompts import EDITOR_PROMPT, EDITOR_TODO_PROMPT, EDITOR_TODO_TOOL_DESCRIPTION
from src.agents.subagents import analyst_subagent, retriever_subagent, verifier_subagent
from src.llm import get_agent_middleware, get_llm
from src.tracing import LabelledTracer

# `max_concurrency` bounds parallel `task` calls (deepagents has no cap of its own). It is 1 because
# parallel retrievers share one 8K TPM Groq bucket, so running them together only produces 429s; the
# default recursion limit of 9999 would let a looping Editor burn the daily quota.
EDITOR_CONFIG: dict[str, Any] = {"recursion_limit": 40, "max_concurrency": 1}


def _unused_general_purpose() -> SubAgent:
    """Override deepagents' auto-added `general-purpose` subagent with an inert one.

    The default inherits every parent tool and adds ~85 tokens to each Editor call. A spec with the same
    name replaces it; a harness profile that disables it is keyed by provider and would not match the
    fallback providers. No tools and read-only files make a wrong call harmless.
    """
    return {
        "name": "general-purpose",
        "description": "Unused. Never call it; use retriever, analyst or verifier.",
        "system_prompt": "Reply with the single word UNUSED.",
        "tools": [],
        "model": get_llm("verifier"),
        "middleware": [FilesystemMiddleware(tools=["read_file"])],
    }


def build_editor_agent(checkpointer: Any = None):
    """Compiled Editor graph with the retriever, analyst and verifier subagents.

    `FilesystemMiddleware(tools=["read_file"])` replaces deepagents' default (same middleware name), so
    the Editor cannot write or search files itself: evidence arrives only through the subagents.
    """
    return create_deep_agent(
        model=get_llm("editor"),
        system_prompt=EDITOR_PROMPT,
        tools=[],
        subagents=[retriever_subagent(), analyst_subagent(), verifier_subagent(), _unused_general_purpose()],
        middleware=[
            FilesystemMiddleware(tools=["read_file"]),
            TodoListMiddleware(system_prompt=EDITOR_TODO_PROMPT, tool_description=EDITOR_TODO_TOOL_DESCRIPTION),
            *get_agent_middleware("editor"),
        ],
        checkpointer=checkpointer,
        name="editor",
    )


def _text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()


def _normalize_citations(text: str) -> str:
    """gpt-oss models sometimes emit 【label】 citations despite the prompt; the contract is [label]."""
    return text.replace("【", "[").replace("】", "]")


def ask_editor(
    agent,
    question: str,
    query_type: str = "unknown",
    thread_id: str | None = None,
    callbacks: list[Any] | None = None,
) -> dict[str, Any]:
    """Run one question through the Editor.

    Returns `answer` (the last non-empty AI message), `files` (/evidence and /notes), `todos` and the
    full `messages` list for tracing. `thread_id` is required when the agent has a checkpointer;
    `callbacks` are LangChain callback handlers (tracing, call counting).

    Only this call is traced in LangSmith: tracing is switched on here, not globally, so the outer
    classify/format nodes stay out. It needs LANGSMITH_API_KEY; without one nothing is sent.
    """
    config: dict[str, Any] = {
        **EDITOR_CONFIG,
        "run_name": "editor",
        "tags": ["deep-agent"],
        "metadata": {"query_type": query_type, "question": question},
    }
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    tracing = bool(os.getenv("LANGSMITH_API_KEY"))
    handlers = [*(callbacks or [])]
    if tracing:
        handlers.append(LabelledTracer(project_name=os.getenv("LANGSMITH_PROJECT")))
    if handlers:
        config["callbacks"] = handlers
    with tracing_context(enabled=tracing):
        result = agent.invoke(
            {"messages": [HumanMessage(f"QUERY_TYPE: {query_type}\nQUESTION: {question}")]},
            config=config,
        )
    answer = next(
        (t for m in reversed(result["messages"]) if isinstance(m, AIMessage) and (t := _text(m))),
        "",
    )
    return {
        "answer": _normalize_citations(answer),
        "files": result.get("files", {}),
        "todos": result.get("todos", []),
        "messages": result["messages"],
    }
