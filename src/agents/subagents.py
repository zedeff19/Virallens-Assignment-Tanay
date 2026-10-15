"""Subagent specs: retriever, analyst and verifier."""
from __future__ import annotations

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgent
from langchain.agents.middleware import ToolCallLimitMiddleware

from src.agents.evidence import save_evidence
from src.agents.notes import save_note
from src.agents.prompts import (
    ANALYST_DESCRIPTION,
    ANALYST_PROMPT,
    RETRIEVER_DESCRIPTION,
    RETRIEVER_PROMPT,
    VERIFIER_DESCRIPTION,
    VERIFIER_PROMPT,
)
from src.agents.retriever_tools import retriever_search
from src.llm import get_agent_middleware, get_llm
from src.tools import find_citations, get_neighbors

RETRIEVER_TOOLS = [retriever_search, get_neighbors, find_citations, save_evidence]


def retriever_subagent() -> SubAgent:
    """Retriever: one sub-question in, passages saved to /evidence, 3-line verdict out.

    Filesystem tools are cut to `read_file` (mandatory) because the default seven tool schemas cost
    ~2.6K tokens per call against Groq's 8K TPM; evidence is written by `save_evidence` only.
    """
    return {
        "name": "retriever",
        "description": RETRIEVER_DESCRIPTION,
        "system_prompt": RETRIEVER_PROMPT,
        "tools": RETRIEVER_TOOLS,
        "model": get_llm("retriever"),
        "middleware": [FilesystemMiddleware(tools=["read_file"]), *get_agent_middleware("retriever")],
    }


def analyst_subagent() -> SubAgent:
    """Analyst: one evidence file in, one structured note written to /notes, 3-line status out.

    Only `read_file` and `save_note`. `tools` is explicit: a spec without the key would inherit the
    Editor's tools. `save_note` replaces `write_file` because "Updated file ..." gave Qwen no cue to
    stop and it re-issued the same call 8 times; `save_note` returns the exact reply to echo.

    The limit is a hard stop for a model that loops anyway: the 3rd `save_note` call ends the run.
    "end" rather than "continue" because a looping model ignores the error and keeps spending TPM.
    On a forced end the Editor gets the middleware's limit message instead of the 3 lines, but the
    note from the first call is already in /notes/<TASK_ID>.md.
    """
    return {
        "name": "analyst",
        "description": ANALYST_DESCRIPTION,
        "system_prompt": ANALYST_PROMPT,
        "tools": [save_note],
        "model": get_llm("analyst"),
        "middleware": [
            FilesystemMiddleware(tools=["read_file"]),
            ToolCallLimitMiddleware(tool_name="save_note", run_limit=2, exit_behavior="end"),
            *get_agent_middleware("analyst"),
        ],
    }


def verifier_subagent() -> SubAgent:
    """Verifier: draft (inline) + evidence paths in, PASS or a fix list out. Read-only.

    Checks against /evidence only; /notes are derived and must not count as support.
    """
    return {
        "name": "verifier",
        "description": VERIFIER_DESCRIPTION,
        "system_prompt": VERIFIER_PROMPT,
        "tools": [],
        "model": get_llm("verifier"),
        "middleware": [FilesystemMiddleware(tools=["read_file"]), *get_agent_middleware("verifier")],
    }
