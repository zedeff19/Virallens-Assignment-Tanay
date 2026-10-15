"""Live end-to-end probes for the Editor (uses LLM quota; the SQLite cache makes reruns free).

    python -m tests.probe_editor              # all probes, in order: lookup, compare, connect, ...
    python -m tests.probe_editor compare      # probes whose name contains "compare"

Runs the full Editor with its three subagents and checks the trace the routing table promises: which
subagents fired and in what order, LLM call count, and that every citation in the answer exists in the
evidence the subagents saved. Prints the answer so a person can judge it.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass

from src.agents.deep_agent import ask_editor, build_editor_agent
from src.trace import CITE_RE, Trace

PAUSE_S = 30  # keep the 8K TPM window from carrying over between probes


@dataclass
class Probe:
    name: str
    query_type: str
    question: str
    max_llm_calls: int
    retriever: tuple[int, int]  # (min, max) retriever calls
    analyst: tuple[int, int]
    verifier: tuple[int, int]
    sequential: bool = False  # an analyst must finish before the second retriever starts (type 4)
    must_say: tuple[str, ...] = ()  # regexes the answer must match (case-insensitive)


PROBES = [
    Probe("lookup", "lookup", "How far did Macquarie's stock price fall?", 10, (1, 2), (0, 0), (0, 0), must_say=(r"41\s?(%|percent)",)),
    Probe(
        "compare", "compare", "How do Meta and the U.S. government disagree about risk-factor disclosures in Facebook v. Amalgamated?",
        30, (2, 3), (2, 3), (1, 1),
    ),
    Probe(
        "connect", "connect", "How does the Macquarie decision matter to the Facebook v. Amalgamated case?",
        30, (2, 4), (2, 3), (1, 1), sequential=True,
    ),
    Probe("timeline", "timeline", "Give a timeline of the key events in SEC v. Chiueh.", 24, (1, 2), (1, 1), (1, 1), must_say=(r"¶",)),
    Probe(
        "unanswerable", "unanswerable", "Who won Facebook v. Amalgamated Bank at the Supreme Court?",
        10, (1, 2), (0, 0), (0, 0), must_say=(r"\bnot\b.*\b(in|contain|include|available)",),
    ),
]


def _in_range(n: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= n <= bounds[1]


def run_probe(agent, p: Probe) -> tuple[bool, list[str]]:
    trace = Trace()
    t0 = time.time()
    out = ask_editor(agent, p.question, p.query_type, callbacks=[trace])
    elapsed = time.time() - t0
    problems: list[str] = []
    counts = {k: trace.subagents.count(k) for k in ("retriever", "analyst", "verifier", "general-purpose")}

    for name, bounds in (("retriever", p.retriever), ("analyst", p.analyst), ("verifier", p.verifier)):
        if not _in_range(counts[name], bounds):
            problems.append(f"{counts[name]} {name} calls, expected {bounds[0]}-{bounds[1]}")
    if counts["general-purpose"]:
        problems.append("called the inert general-purpose subagent")
    if trace.llm_calls > p.max_llm_calls:
        problems.append(f"{trace.llm_calls} LLM calls > {p.max_llm_calls}")
    if p.verifier[0] and trace.subagents and trace.subagents[-1] != "verifier":
        problems.append(f"verifier was not the last subagent (order: {trace.subagents})")
    if p.sequential:
        # between the first and the second retriever start there must be a finished analyst
        starts = [i for i, e in enumerate(trace.events) if e == "start:retriever"]
        if len(starts) >= 2 and "end:analyst" not in trace.events[starts[0] : starts[1]]:
            problems.append("hop 2 retriever started before the first analyst finished")

    answer = out["answer"]
    if not answer:
        problems.append("empty answer")
    evidence = "\n".join(f["content"] for path, f in out["files"].items() if path.startswith("/evidence/"))
    if p.retriever[0] and not evidence and "not" not in answer.lower():
        problems.append("no /evidence file saved")
    cites = sorted({m.group(0): m for m in CITE_RE.finditer(answer)}.items())
    if p.query_type != "unanswerable" and not cites:
        problems.append("no [label, p.N] / [label ¶N] citation in the answer")
    for text, m in cites:
        label, page, para = m.group(1).strip(), m.group(2), m.group(3)
        # the first page/paragraph cited must exist under that label in a saved evidence heading
        needle = f"[{label}, p.{page}]" if page else f"[{label} ¶{para}"
        if needle not in evidence:
            problems.append(f"citation {text} not in any saved evidence file (looked for {needle})")
    for pattern in p.must_say:
        if not re.search(pattern, answer, re.IGNORECASE | re.DOTALL):
            problems.append(f"answer does not match /{pattern}/")

    status = "ok" if not problems else "FAIL"
    print(
        f"\n[{status}] {p.name}  ({elapsed:.1f}s, {trace.llm_calls} LLM calls + {trace.llm_errors} failed attempts, "
        f"subagents={trace.subagents}, files={sorted(out['files'])}, todos={len(out['todos'])})"
    )
    print("   " + answer.replace("\n", "\n   "))
    for pr in problems:
        print(f"   !! {pr}")
    return not problems, problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    agent = build_editor_agent()
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    probes = [p for p in PROBES if only in p.name]
    passed = 0
    for i, p in enumerate(probes):
        if i:
            time.sleep(PAUSE_S)
        try:
            passed += run_probe(agent, p)[0]
        except Exception as exc:  # every provider exhausted or the recursion limit hit: report, keep going
            print(f"\n[FAIL] {p.name}  {type(exc).__name__}: {str(exc)[:300]}")
    print(f"\n{passed}/{len(probes)} probes passed")
    return 0 if passed == len(probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
