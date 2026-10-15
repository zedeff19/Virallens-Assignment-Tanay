"""`save_note`: the analyst's only write path into the virtual filesystem.

Replaces deepagents' `write_file`, whose "Updated file ..." result gave Qwen no cue to stop (it re-issued
an identical call 8 times). Like `save_evidence`, the tool does the bookkeeping: it writes the header,
derives STATUS/GAPS from the body's MISSING lines, and hands the model the exact 3-line reply to echo.
Files live in the DeepAgents state (`files` key), so the Editor and verifier can `read_file` them.
"""
from __future__ import annotations

import re

from deepagents.backends.utils import create_file_data
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

NOTES_DIR = "/notes"
MAX_GAP_WORDS = 30

_MISSING_RE = re.compile(r"^\s*MISSING:\s*(?P<what>.+?)\s*$", re.MULTILINE)


def note_path(task_id: str) -> str:
    """`/notes/<slug>.md`, slugged like `evidence_path` so the Editor's TASK_ID maps to the same name."""
    slug = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-") or "note"
    return f"{NOTES_DIR}/{slug}.md"


def render_note(task_id: str, side: str, focus: str, body: str) -> str:
    header = f"# Note: {task_id.strip()} | {' '.join(side.split())} | {focus.strip()}"
    return f"{header}\n\n{body.strip()}\n"


def status_and_gaps(body: str) -> tuple[str, str]:
    """PARTIAL with the MISSING lines (cut to MAX_GAP_WORDS) when the body has any, else COMPLETE/none."""
    missing = [m.group("what") for m in _MISSING_RE.finditer(body)]
    if not missing:
        return "COMPLETE", "none"
    words = " ".join(missing).split()
    gaps = " ".join(words[:MAX_GAP_WORDS]) + (" ..." if len(words) > MAX_GAP_WORDS else "")
    return "PARTIAL", gaps


@tool
def save_note(task_id: str, focus: str, side: str, content: str, runtime: ToolRuntime) -> Command:
    """Save the structured note to /notes/<task_id>.md. Call once, last, then send the reply it returns.

    Args:
        task_id: The TASK_ID you were given.
        focus: The FOCUS you were given: holding, argument or timeline.
        side: The SIDE you were given (doc_id or label).
        content: The note body only (RULE/POSITION/table sections). Do not write the "# Note:" header line.
    """
    if not content.strip():
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "nothing saved: content is empty. Call save_note again with the note body.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    path = note_path(task_id)
    status, gaps = status_and_gaps(content)
    reply = f"NOTE: {path} ({focus.strip()})\nSTATUS: {status}\nGAPS: {gaps}"
    message = ToolMessage(
        f"saved {path}. Do not call any tool again. Reply now with exactly these 3 lines and nothing else:\n{reply}",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        update={
            "messages": [message],
            "files": {path: create_file_data(render_note(task_id, side, focus, content))},
        }
    )
