"""LangSmith run names that say what ran: agent, subagent, delegation, model call or tool.

LangGraph names every agent's two nodes "model" and "tools", and chat-model runs carry only a class
name, so a deep-agent trace is a wall of identical rows. `LabelledTracer` relabels each run as it
starts, from the metadata LangChain already attaches (`lc_agent_name`, `ls_model_name`):

    AGENT editor                       the Editor graph (root)
    ├ model step (editor)              one pass through the Editor's model node
    │ └ LLM gpt-oss-120b (editor)      the chat-model call itself; a fallback shows its own model name
    ├ tools step (editor)
    │ └ TASK -> retriever              the `task` tool delegating to a subagent
    │   └ SUBAGENT retriever           the subagent's graph, with its own model/tools steps
    │     └ TOOL retriever_search (retriever)

Runs are also tagged `kind:<kind>` and `agent:<name>` for filtering. Names are only cosmetic: nothing
reads them back, and `Trace` (src/trace.py) keys off the tool's real name through its own callbacks.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.schemas import Run

TASK_TOOL = "task"
_NODE_KINDS = {"model": "model step", "tools": "tools step"}
_SUBAGENT_RE = re.compile(r"subagent_type['\"]?\s*[:=]\s*['\"](\w[\w-]*)")


def _subagent_type(run: Run) -> str | None:
    inputs = run.inputs or {}
    if isinstance(inputs.get("subagent_type"), str):
        return inputs["subagent_type"]
    match = _SUBAGENT_RE.search(str(inputs.get("input", "")))
    return match.group(1) if match else None


def label_run(run: Run, parent: Run | None = None) -> tuple[str, str] | None:
    """`(kind, name)` for a run that deserves a clearer name, or None to leave it alone."""
    meta = (run.extra or {}).get("metadata") or {}
    agent = meta.get("lc_agent_name")
    where = f" ({agent})" if agent else ""

    if run.run_type == "tool":
        if run.name == TASK_TOOL:
            return "delegate", f"TASK -> {_subagent_type(run) or '?'}"
        return "tool", f"TOOL {run.name}{where}"

    if run.run_type in ("llm", "chat_model"):
        model = meta.get("ls_model_name") or run.name
        return "llm", f"LLM {model}{where}"

    if run.run_type == "chain":
        if run.name in _NODE_KINDS:
            return "step", f"{_NODE_KINDS[run.name]}{where}"
        # An agent's graph run is named after the agent (`run_name` / `create_agent(name=...)`).
        if agent and run.name == agent:
            delegated = parent is not None and parent.run_type == "tool" and parent.name.startswith("TASK")
            return ("subagent", f"SUBAGENT {agent}") if delegated else ("agent", f"AGENT {agent}")
    return None


class LabelledTracer(LangChainTracer):
    """LangSmith tracer that renames runs with `label_run` before they are sent.

    Passing one in `callbacks=[...]` replaces the default tracer (LangChain only adds its own when no
    `LangChainTracer` is present), so every run is sent once, under the new name.
    """

    def _start_trace(self, run: Run) -> None:
        parent = self.run_map.get(str(run.parent_run_id)) if run.parent_run_id else None
        labelled = label_run(run, parent)
        if labelled:
            kind, name = labelled
            run.name = name
            tags: list[str] = list(run.tags or [])
            tags.append(f"kind:{kind}")
            agent: Any = ((run.extra or {}).get("metadata") or {}).get("lc_agent_name")
            if agent:
                tags.append(f"agent:{agent}")
            run.tags = tags
        super()._start_trace(run)
