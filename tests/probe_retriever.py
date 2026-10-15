"""Live probes for the retriever subagent alone (uses LLM quota; the SQLite cache makes reruns free).

    python -m tests.probe_retriever            # all probes
    python -m tests.probe_retriever lookup     # probes whose name contains "lookup"

Runs the retriever spec as a stand-alone agent (no Editor), then checks verdict, expected chunk /
document in the evidence file, LLM call count and the 3-line output format.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from src.agents.subagents import retriever_subagent

MAX_LLM_CALLS = 6
PAUSE_S = 20
FORMAT_RE = re.compile(
    r"^EVIDENCE: (?:/evidence/[a-z0-9-]+\.md \(\d+ chunks?\)|none \(0 chunks\))\n"
    r"VERDICT: (FOUND|PARTIAL|NONE)\n"
    r"NOTES: .+$"
)


@dataclass
class Probe:
    name: str
    task_id: str
    question: str
    doc_hint: str = "none"
    section_hint: str = "none"
    verdicts: tuple[str, ...] = ("FOUND",)
    must_contain: tuple[str, ...] = ()  # substrings that must appear in the evidence file
    min_docs: int = 0
    docs_in: set[str] = field(default_factory=set)  # chunk_id prefixes that must appear


PROBES = [
    Probe("lookup", "macq-drop", "How far did Macquarie's stock price fall?", "macquarie", must_contain=("41%",)),
    Probe("concept", "omission-vs-halftruth", "What is the difference between a pure omission and a half-truth under Rule 10b-5(b)?", min_docs=2),
    Probe("exact_string", "item-303", "What does the corpus say about Item 303?", must_contain=("Item 303",)),
    Probe("section_filter", "la-10b5c", "What does the court say about scheme liability under Rule 10b-5(a) and (c)?", "last_atlantis", must_contain=("10b-5",)),
    Probe("citation", "fb-cites-macq", "Where do the Facebook briefs cite Macquarie, and for what?", docs_in={"fb_petitioners", "fb_amicus"}),
    Probe("timeline", "chiueh-policy", "When was the concentration policy adopted or described in the SEC v. Chiueh complaint?", "sec_chiueh", must_contain=("¶",)),
    Probe("unanswerable", "fb-outcome", "Who won Facebook v. Amalgamated Bank at the Supreme Court?", verdicts=("PARTIAL", "NONE")),
    Probe("wrong_hint", "wrong-hint", "How far did Macquarie's stock price fall?", "sec_chiueh", must_contain=("41%",)),
]


class CallCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.llm_calls = 0  # completed model responses (what the agent really used)
        self.llm_errors = 0  # failed attempts: 429s, 413s, ... absorbed by retry/fallback
        self.tool_calls: list[str] = []

    def on_llm_end(self, *args, **kwargs) -> None:
        self.llm_calls += 1

    def on_llm_error(self, *args, **kwargs) -> None:
        self.llm_errors += 1

    def on_tool_start(self, serialized, *args, **kwargs) -> None:
        self.tool_calls.append((serialized or {}).get("name", "?"))


def run_probe(agent, p: Probe) -> tuple[bool, list[str]]:
    task = (
        f"TASK_ID: {p.task_id}\nQUESTION: {p.question}\nDOC_HINT: {p.doc_hint}\nSECTION_HINT: {p.section_hint}"
    )
    counter = CallCounter()
    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(task)]}, config={"callbacks": [counter], "recursion_limit": 25})
    elapsed = time.time() - t0
    out = "\n".join(line.strip() for line in result["messages"][-1].text.strip().splitlines())
    problems: list[str] = []

    if not FORMAT_RE.match(out):
        problems.append("output is not the 3-line format")
    m = re.search(r"^VERDICT: (\w+)$", out, re.MULTILINE)
    verdict = m.group(1) if m else "?"
    if verdict not in p.verdicts:
        problems.append(f"verdict {verdict}, expected one of {p.verdicts}")
    if counter.llm_calls > MAX_LLM_CALLS:
        problems.append(f"{counter.llm_calls} LLM calls > {MAX_LLM_CALLS}")
    n_search = counter.tool_calls.count("search_corpus")
    n_extra = counter.tool_calls.count("get_neighbors") + counter.tool_calls.count("find_citations")
    if n_search > 3 or n_extra > 1:
        problems.append(f"budget exceeded: {n_search} searches, {n_extra} neighbor/citation lookups")

    files = result.get("files", {})
    body = next((f["content"] for path, f in files.items() if path.startswith("/evidence/")), "")
    if "none (0 chunks)" not in out and not body:
        problems.append("claims evidence but no /evidence file was written")
    for s in p.must_contain:
        if s not in body:
            problems.append(f"evidence file missing {s!r}")
    docs = {c.rsplit("-", 1)[0] for c in re.findall(r"· ([\w]+-\d{4}) ·", body)}
    if len(docs) < p.min_docs:
        problems.append(f"only {len(docs)} docs in evidence, expected >= {p.min_docs}")
    if not p.docs_in <= docs:
        problems.append(f"evidence missing docs {sorted(p.docs_in - docs)}")

    print(f"\n[{'ok' if not problems else 'FAIL'}] {p.name}  ({elapsed:.1f}s, {counter.llm_calls} LLM calls + {counter.llm_errors} failed attempts, tools={counter.tool_calls})")
    print("   " + out.replace("\n", "\n   "))
    for pr in problems:
        print(f"   !! {pr}")
    return not problems, problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    spec = retriever_subagent()
    agent = create_agent(
        model=spec["model"],
        tools=spec["tools"],
        system_prompt=spec["system_prompt"],
        middleware=spec["middleware"],
    )
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    probes = [p for p in PROBES if only in p.name]
    passed = 0
    for i, p in enumerate(probes):
        if i:
            time.sleep(PAUSE_S)  # keep the 8K TPM window from carrying over between probes
        try:
            passed += run_probe(agent, p)[0]
        except Exception as exc:  # every provider exhausted: report it, keep going
            print(f"\n[FAIL] {p.name}  all providers failed: {type(exc).__name__}: {str(exc)[:200]}")
    print(f"\n{passed}/{len(probes)} probes passed")
    return 0 if passed == len(probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
