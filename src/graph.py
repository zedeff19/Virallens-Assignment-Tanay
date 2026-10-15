"""Outer LangGraph shell: classify -> deep_agent -> format (out-of-scope questions short-circuit to a refusal).

    START -> classify --(out_of_scope)--> refuse ------------> END
                  \\--(anything else)---> deep_agent -> format -> END

`classify` is one small-model call that tags the question with a query type. Off-topic questions end at
`refuse` with zero agent calls, which is what protects the free-tier quota. Everything else goes to the
DeepAgents Editor (`src/agents/deep_agent.py`), which plans, delegates to its subagents and writes the answer.
`format` is deterministic: it checks every citation in the answer against the evidence files the
subagents saved and appends a Sources block.

    app = build_graph()
    state = app.invoke({"question": "How far did Macquarie's stock price fall?"})
    print(state["final"])

`build_graph(editor_agent=..., classifier=...)` accepts stand-ins so the shell can be tested offline.
"""
from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agents.deep_agent import ask_editor
from src.agents.prompts import CORPUS_CARD
from src.ingest.docs import DOCS
from src.trace import CITE_RE, Trace

QueryType = Literal["lookup", "concept", "compare", "connect", "timeline", "unanswerable", "out_of_scope", "unknown"]

# `## [Macquarie, pp.5-7] · macquarie-0012 · ...` / `## [Chiueh ¶40–47] · ...`: groups = label, page lo/hi, para lo/hi.
_HEADING_RE = re.compile(
    r"^## \[([^\[\],]+?)(?:,\s*pp?\.\s*(\d+)(?:\s*[–—-]\s*(\d+))?|\s*¶\s*(\d+)(?:\s*[–—-]\s*(\d+))?)[^\]]*\]",
    re.MULTILINE,
)

CLASSIFIER_PROMPT = f"""\
You route questions for a legal-document question-answering system. Pick exactly one query_type.

CORPUS (doc_id | what it is | cite label)
{CORPUS_CARD}
The final Facebook v. Amalgamated decision, later case history and outcomes are NOT in the corpus.

query_type
- lookup: one fact in one document (a number, a date, a name).
- concept: what a rule, term or doctrine is, or how it is treated across the documents.
- compare: how two sides or documents differ (e.g. Meta vs the U.S.).
- connect: how one document or holding matters to, or applies to, another.
- timeline: dated events, or who is charged with what, in a document.
- unanswerable: about these documents' subject but asks for something they cannot contain: an outcome,
  a final decision, a prediction, or legal advice.
- out_of_scope: has nothing to do with these securities-law documents (weather, coding, recipes, general chat).
- unknown: you cannot tell.

When unsure whether a question is on topic, choose unknown, never out_of_scope. Answer with the
query_type and a one-sentence reason.
"""


class Classification(BaseModel):
    query_type: QueryType = Field(description="the single best routing label")
    reason: str = Field(default="", description="one short sentence")


class RAGState(TypedDict, total=False):
    question: str
    query_type: QueryType
    classifier_note: str
    answer: str  # raw Editor answer
    citations: list[dict[str, Any]]  # {"text", "label", "doc_title", "verified"}
    files: dict[str, str]  # path -> content: /evidence/* and /notes/*
    trace: Annotated[list[str], operator.add]  # every node appends
    error: str | None
    final: str  # what the user sees


def _scope_message() -> str:
    docs = "\n".join(f"  - {d.title} ({d.doc_type})" for d in DOCS.values())
    return (
        "That question is outside what I can answer. I only answer questions about these documents:\n"
        f"{docs}\n"
        "Try a question about securities-fraud claims, Rule 10b-5, risk-factor disclosures or SEC v. Chiueh."
    )


def _file_contents(files: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path, f in files.items():
        content = f.get("content") if isinstance(f, dict) else f
        if isinstance(content, list):  # older deepagents stored a list of lines
            content = "\n".join(content)
        out[path] = content or ""
    return out


def _evidence_ranges(files: dict[str, str]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """(label, "p" | "para") -> page / paragraph ranges that appear in a saved /evidence heading."""
    ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for path, content in files.items():
        if not path.startswith("/evidence/"):
            continue
        for m in _HEADING_RE.finditer(content):
            label = m.group(1).strip()
            if m.group(2):
                key, lo, hi = (label, "p"), int(m.group(2)), int(m.group(3) or m.group(2))
            else:
                key, lo, hi = (label, "para"), int(m.group(4)), int(m.group(5) or m.group(4))
            ranges.setdefault(key, []).append((lo, hi))
    return ranges


def extract_citations(answer: str, files: dict[str, str]) -> list[dict[str, Any]]:
    """Unique citations in `answer`, each marked verified when a saved evidence heading covers it.

    This is a presence check (label plus page / paragraph), not a check that the claim is supported;
    claim checking is the verifier subagent's job.
    """
    ranges = _evidence_ranges(files)
    titles = {d.cite_label: d.title for d in DOCS.values()}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in CITE_RE.finditer(answer):
        text = m.group(0)
        if text in seen:
            continue
        seen.add(text)
        label = m.group(1).strip()
        key, n = ((label, "p"), int(m.group(2))) if m.group(2) else ((label, "para"), int(m.group(3)))
        verified = any(lo <= n <= hi for lo, hi in ranges.get(key, []))
        out.append({"text": text, "label": label, "doc_title": titles.get(label, ""), "verified": verified})
    return out


def render_final(answer: str, citations: list[dict[str, Any]]) -> str:
    parts = [answer.strip()]
    if citations:
        lines = [f"- {c['text']} {c['doc_title']}".rstrip() for c in citations if c["verified"]]
        if lines:
            parts.append("Sources\n" + "\n".join(lines))
        bad = [c["text"] for c in citations if not c["verified"]]
        if bad:
            parts.append("Warning: not found in the retrieved evidence: " + ", ".join(bad))
    return "\n\n".join(parts)


def build_graph(editor_agent: Any = None, classifier: Any = None, checkpointer: Any = None):
    """Compile the outer graph.

    `editor_agent` is a compiled Editor (`build_editor_agent()` when omitted) and `classifier` a runnable
    returning a `Classification` (a rate-limited small model with fallbacks when omitted). Both are built
    once here, not per question. Pass `checkpointer` only if you also pass a `thread_id` in the config.
    """
    if editor_agent is None:
        from src.agents.deep_agent import build_editor_agent

        editor_agent = build_editor_agent()
    if classifier is None:
        from src.llm import get_llm_with_fallbacks

        classifier = get_llm_with_fallbacks("classifier", lambda m: m.with_structured_output(Classification))

    def classify(state: RAGState) -> dict[str, Any]:
        try:
            out = classifier.invoke([SystemMessage(CLASSIFIER_PROMPT), HumanMessage(state["question"])])
            result = out if isinstance(out, Classification) else Classification.model_validate(out)
        except Exception as exc:  # every provider failed or the output was malformed: let the Editor decide
            note = f"classifier failed ({type(exc).__name__}); the Editor decides"
            return {"query_type": "unknown", "classifier_note": note, "trace": [f"classify -> unknown ({note})"]}
        return {
            "query_type": result.query_type,
            "classifier_note": result.reason,
            "trace": [f"classify -> {result.query_type} ({result.reason})"],
        }

    def route_after_classify(state: RAGState) -> str:
        return "refuse" if state["query_type"] == "out_of_scope" else "deep_agent"

    def refuse(state: RAGState) -> dict[str, Any]:
        return {"final": _scope_message(), "trace": ["refuse: out of scope, 0 agent calls"]}

    def deep_agent(state: RAGState) -> dict[str, Any]:
        trace = Trace()
        try:
            out = ask_editor(editor_agent, state["question"], state["query_type"], callbacks=[trace])
        except Exception as exc:  # providers exhausted, recursion limit, ...: report instead of a traceback
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            return {"answer": "", "files": {}, "error": error, "trace": [f"deep_agent: failed ({error})"]}
        summary = (
            f"deep_agent: subagents={trace.subagents} llm_calls={trace.llm_calls} "
            f"failed_attempts={trace.llm_errors} todos={len(out['todos'])}"
        )
        return {"answer": out["answer"], "files": _file_contents(out["files"]), "error": None, "trace": [summary]}

    def format_answer(state: RAGState) -> dict[str, Any]:
        answer = state.get("answer", "")
        if not answer:
            reason = state.get("error") or "the agent returned no answer"
            return {
                "citations": [],
                "final": f"I could not complete this question ({reason}). Please try again.",
                "trace": ["format: no answer"],
            }
        citations = extract_citations(answer, state.get("files", {}))
        unverified = sum(not c["verified"] for c in citations)
        return {
            "citations": citations,
            "final": render_final(answer, citations),
            "trace": [f"format: {len(citations)} citations, {unverified} unverified"],
        }

    graph = StateGraph(RAGState)
    graph.add_node("classify", classify)
    graph.add_node("refuse", refuse)
    graph.add_node("deep_agent", deep_agent)
    graph.add_node("format", format_answer)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_after_classify, {"refuse": "refuse", "deep_agent": "deep_agent"})
    graph.add_edge("deep_agent", "format")
    graph.add_edge("refuse", END)
    graph.add_edge("format", END)
    return graph.compile(checkpointer=checkpointer)
