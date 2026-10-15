"""`save_evidence`: the retriever's only write path into the virtual filesystem.

The model supplies chunk ids and a short note; the passage text and citation labels are copied
verbatim from chunks.jsonl, so quotes cannot be paraphrased on the way into /evidence and cost no
output tokens. Files live in the DeepAgents state (`files` key), which the `task` tool propagates
back to the parent, so analyst/verifier subagents can `read_file` them.
"""
from __future__ import annotations

import re

from deepagents.backends.utils import create_file_data
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from src.tools import _chunks_by_id

EVIDENCE_DIR = "/evidence"
MAX_CHUNKS = 10

_HEADER_RE = re.compile(r"^## .*? · (?P<chunk_id>[\w\-]+)(?: · .*)?$", re.MULTILINE)


def evidence_path(task_id: str) -> str:
    """`/evidence/<slug>.md`, with the slug restricted to [a-z0-9-] (no path traversal, no spaces)."""
    slug = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-") or "evidence"
    return f"{EVIDENCE_DIR}/{slug}.md"


def _section(chunk: dict) -> str:
    return f"## {chunk['citation']} · {chunk['chunk_id']} · {' > '.join(chunk['section_path'])}\n\n{chunk['text'].strip()}\n"


def render_evidence(task_id: str, notes: str, chunks: list[dict]) -> str:
    head = f"# Evidence: {task_id.strip()}\nRetriever notes: {' '.join(notes.split())}\n"
    return head + "\n" + "\n".join(_section(c) for c in chunks)


def _existing_ids(content: str) -> list[str]:
    return _HEADER_RE.findall(content)


def build_evidence_file(
    task_id: str, chunk_ids: list[str], notes: str, existing: str | None = None
) -> tuple[str, list[str], list[str], list[str]]:
    """Return (file_content, kept_ids, unknown_ids, all_ids_in_file).

    Passing the previous `existing` file content merges: earlier chunks stay, new ones are appended.
    """
    by_id = _chunks_by_id()
    seen: list[str] = _existing_ids(existing) if existing else []
    kept: list[str] = []
    unknown: list[str] = []
    for cid in chunk_ids:
        cid = str(cid).strip()
        if cid in seen or cid in kept:
            continue
        if cid not in by_id:
            unknown.append(cid)
        elif len(seen) + len(kept) < MAX_CHUNKS:
            kept.append(cid)
    all_ids = seen + kept
    if not all_ids:
        return "", kept, unknown, all_ids
    prior_notes = ""
    if existing:
        m = re.search(r"^Retriever notes: (.*)$", existing, re.MULTILINE)
        prior_notes = m.group(1) if m else ""
    merged_notes = " | ".join(n for n in (prior_notes, " ".join(notes.split())) if n)
    return render_evidence(task_id, merged_notes, [by_id[i] for i in all_ids]), kept, unknown, all_ids


@tool
def save_evidence(task_id: str, chunk_ids: list[str], notes: str, runtime: ToolRuntime) -> Command:
    """Save the chosen passages, verbatim, to /evidence/<task_id>.md for the Editor and other subagents.

    Call once, at the end of a retrieval task. Text and citation labels are copied from the corpus by
    chunk_id, so only pass ids you saw in search_corpus / get_neighbors / find_citations results.

    Args:
        task_id: The TASK_ID you were given (short slug, e.g. "macq-drop").
        chunk_ids: chunk_ids of the passages to keep, best first (at most 10).
        notes: One short sentence on why these passages were chosen / what they cover.
    """
    path = evidence_path(task_id)
    prior = (runtime.state.get("files") or {}).get(path)
    existing = prior.get("content") if isinstance(prior, dict) else None
    content, kept, unknown, all_ids = build_evidence_file(task_id, chunk_ids, notes, existing)

    parts = [f"saved {len(all_ids)} chunks to {path}" if all_ids else "nothing saved: no valid chunk_ids"]
    if unknown:
        parts.append(f"unknown ids ignored: {unknown}")
    message = ToolMessage("; ".join(parts), tool_call_id=runtime.tool_call_id)
    update: dict = {"messages": [message]}
    if content:
        update["files"] = {path: create_file_data(content)}
    return Command(update=update)
