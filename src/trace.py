"""Run tracing shared by the LangGraph shell, `main.py` and the live probes.

`Trace` is a LangChain callback handler: pass it in `callbacks=[...]` and read the counters afterwards.
`CITE_RE` matches the citation labels the Editor is told to emit.
"""
from __future__ import annotations

import re

from langchain_core.callbacks import BaseCallbackHandler

# [Label, p.5] / [Label, pp.5-7] / [Label ¶45]: group 1 = label, 2 = first page, 3 = first paragraph
CITE_RE = re.compile(r"\[([^\[\],]+?)(?:,\s*pp?\.\s*(\d+)|\s*¶\s*(\d+))[^\[\]]*\]")


class Trace(BaseCallbackHandler):
    """Counts model responses and records the ordered subagent calls made through `task`."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.llm_errors = 0
        self.subagents: list[str] = []  # subagent_type per task call, in start order
        self.events: list[str] = []  # "start:<type>" and "end:<type>" in order, to check sequencing
        self._runs: dict[object, str] = {}

    def on_llm_end(self, *args, **kwargs) -> None:
        self.llm_calls += 1

    def on_llm_error(self, *args, **kwargs) -> None:
        self.llm_errors += 1

    def on_tool_start(self, serialized, input_str, *, run_id=None, inputs=None, **kwargs) -> None:
        if (serialized or {}).get("name") != "task":
            return
        kind = (inputs or {}).get("subagent_type")
        if kind is None:
            m = re.search(r"subagent_type['\"]?\s*[:=]\s*['\"](\w[\w-]*)", str(input_str))
            kind = m.group(1) if m else "?"
        self.subagents.append(kind)
        self.events.append(f"start:{kind}")
        self._runs[run_id] = kind

    def on_tool_end(self, output, *, run_id=None, **kwargs) -> None:
        if run_id in self._runs:
            self.events.append(f"end:{self._runs[run_id]}")
