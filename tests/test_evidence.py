"""Offline tests for src/agents/evidence.py and the retriever prompt/spec pieces."""
from __future__ import annotations

import json
import re

from src.agents import evidence as ev
from src.agents.prompts import CORPUS_CARD, RETRIEVER_PROMPT
from src.agents.retriever_tools import PREVIEW_CHARS, preview, retriever_search
from src.ingest.docs import DOCS
from src.tools import _chunks_by_id, search_corpus

IDS = ["macquarie-0000", "macquarie-0002", "macquarie-0003"]


def test_evidence_path_is_slugged():
    assert ev.evidence_path("Macq Drop") == "/evidence/macq-drop.md"
    assert ev.evidence_path("../../etc/passwd") == "/evidence/etc-passwd.md"
    assert ev.evidence_path("   ") == "/evidence/evidence.md"
    assert ev.evidence_path("a/b\\c") == "/evidence/a-b-c.md"


def test_text_and_citation_are_verbatim():
    content, kept, unknown, all_ids = ev.build_evidence_file("t1", IDS[:2], "why")
    chunks = _chunks_by_id()
    assert kept == all_ids == IDS[:2] and unknown == []
    for cid in IDS[:2]:
        c = chunks[cid]
        assert c["text"].strip() in content
        assert f"## {c['citation']} · {cid} · " in content
    assert content.startswith("# Evidence: t1\nRetriever notes: why\n")


def test_unknown_and_duplicate_ids():
    content, kept, unknown, all_ids = ev.build_evidence_file("t", [IDS[0], IDS[0], "nope-1", " " + IDS[1]], "n")
    assert kept == [IDS[0], IDS[1]]
    assert unknown == ["nope-1"]
    assert content.count("## ") == 2


def test_all_unknown_writes_nothing():
    content, kept, unknown, all_ids = ev.build_evidence_file("t", ["x-1"], "n")
    assert content == "" and kept == [] and all_ids == [] and unknown == ["x-1"]


def test_second_call_merges_not_clobbers():
    first, *_ = ev.build_evidence_file("t", IDS[:2], "first")
    second, kept, _, all_ids = ev.build_evidence_file("t", [IDS[1], IDS[2]], "second", existing=first)
    assert kept == [IDS[2]]
    assert all_ids == IDS
    assert second.count("## ") == 3
    assert "Retriever notes: first | second" in second


def test_cap_applies_across_merges():
    ids = [c for c in _chunks_by_id() if c.startswith("fb_amicus")][:15]
    content, kept, _, all_ids = ev.build_evidence_file("t", ids, "n")
    assert len(all_ids) == ev.MAX_CHUNKS == len(kept)


def test_search_corpus_accepts_null_strings_for_optionals():
    out = search_corpus("stock price fell", doc_id="null", section="", k=2)
    assert "error" not in out and '"hits"' in out
    out = search_corpus("stock price fell", doc_id="None", section="none", k=2)
    assert "error" not in out


def test_corpus_card_and_prompt():
    for doc_id in DOCS:
        assert doc_id in CORPUS_CARD
    assert "{corpus_card}" not in RETRIEVER_PROMPT
    assert "EVIDENCE: none (0 chunks)" in RETRIEVER_PROMPT
    assert re.search(r"^VERDICT: FOUND$", RETRIEVER_PROMPT, re.MULTILINE)


def test_preview_short_text_unchanged_and_long_text_centred_on_query():
    assert preview("short  text\nhere", "anything") == "short text here"
    filler = "Lorem ipsum dolor sit amet. " * 60
    text = filler + "The stock price fell 41% after the announcement. " + filler
    out = preview(text, "stock price fell 41%")
    assert "41%" in out and len(out) <= PREVIEW_CHARS + 6
    assert out.startswith("… ") and out.endswith(" …")


def test_retriever_search_returns_previews_and_passes_errors_through():
    out = json.loads(retriever_search.invoke({"query": "stock price fell", "doc_id": "macquarie"}))
    assert 1 <= len(out["hits"]) <= 4
    full = _chunks_by_id()
    for h in out["hits"]:
        assert set(h) == {"chunk_id", "citation", "doc_id", "section", "preview"}
        assert len(h["preview"]) <= PREVIEW_CHARS + 6 or len(h["preview"]) <= len(full[h["chunk_id"]]["text"])
    err = json.loads(retriever_search.invoke({"query": "x", "doc_id": "bogus"}))
    assert "error" in err and "valid_doc_ids" in err
