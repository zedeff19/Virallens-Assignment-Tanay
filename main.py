"""Sample entry point: `python main.py "question"` runs the full pipeline.

    python main.py "How far did Macquarie's stock price fall?"
    python main.py "How do Meta and the U.S. disagree about risk-factor disclosures?" --trace
    python main.py --demo                  # three questions: lookup, unanswerable, out of scope

The function calls a reader should look for, in order:

    ensure_index()         build data/processed/chunks.jsonl and the Chroma + BM25 index when missing
    build_app()            compile the LangGraph shell around the DeepAgents Editor (src/graph.py)
    answer_question(...)   classify -> deep_agent -> format, returns the final graph state

Needs at least one of GROQ_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY (see .env.example).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

DEMO_QUESTIONS = [
    "How far did Macquarie's stock price fall?",
    "Who won Facebook v. Amalgamated Bank at the Supreme Court?",
    "What is the weather in Delhi today?",
]


def ensure_index(force: bool = False) -> None:
    """Make sure the chunks and the search index exist; build whichever is missing. Idempotent."""
    from src.ingest.docs import CHUNKS_PATH
    from src.ingest.index import BM25_PATH, CHROMA_DIR, build_index

    if force or not CHUNKS_PATH.exists():
        from src.ingest.__main__ import main as run_ingest

        print("ingesting PDFs -> data/processed/chunks.jsonl ...", file=sys.stderr)
        if run_ingest() != 0:
            raise RuntimeError("ingest failed; see the output above")
    if force or not BM25_PATH.exists() or not CHROMA_DIR.exists():
        print("building the search index ...", file=sys.stderr)
        build_index()


def build_app(rebuild_index: bool = False):
    """Index (if needed) plus the compiled LangGraph app."""
    ensure_index(force=rebuild_index)
    from src.graph import build_graph

    return build_graph()


def answer_question(app, question: str) -> dict[str, Any]:
    """Run one question through the graph and return the final state."""
    return app.invoke({"question": question})


def _print_result(state: dict[str, Any], show_trace: bool, as_json: bool) -> None:
    if as_json:
        view = {**state, "files": sorted(state.get("files", {}))}
        print(json.dumps(view, ensure_ascii=False, indent=2))
        return
    print(state.get("final", ""))
    if show_trace:
        print("\n--- trace ---")
        print(f"query_type: {state.get('query_type')}")
        for line in state.get("trace", []):
            print(f"  {line}")
        files = sorted(state.get("files", {}))
        if files:
            print("  files: " + ", ".join(files))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the Windows console cannot print the paragraph sign
    ap = argparse.ArgumentParser(description="Multi-agent RAG over five securities-law documents.")
    ap.add_argument("question", nargs="*", help="the question to answer")
    ap.add_argument("--demo", action="store_true", help="run three built-in sample questions")
    ap.add_argument("--trace", action="store_true", help="print query type, subagents and LLM call counts")
    ap.add_argument("--json", action="store_true", help="print the final graph state as JSON")
    ap.add_argument("--rebuild-index", action="store_true", help="re-run ingest and index before answering")
    args = ap.parse_args(argv)

    questions = DEMO_QUESTIONS if args.demo else ([" ".join(args.question)] if args.question else [])
    if not questions or not any(q.strip() for q in questions):
        ap.print_usage(sys.stderr)
        print('error: give a question, or use --demo', file=sys.stderr)
        return 2

    try:
        app = build_app(rebuild_index=args.rebuild_index)
    except RuntimeError as exc:  # e.g. no API key set: one line, no traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1

    failed = False
    for i, question in enumerate(questions):
        if args.demo:
            print(f"{'=' * 72}\nQ{i + 1}: {question}\n{'=' * 72}")
        state = answer_question(app, question)
        failed = failed or bool(state.get("error"))
        _print_result(state, show_trace=args.trace, as_json=args.json)
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()  # LangSmith uploads in the background; do not exit before the last trace is sent
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
