"""Ingest pipeline tests: pure cleaning functions on literals + invariants on the real corpus."""
from __future__ import annotations

import pytest

from src.ingest.chunk import split_sentences
from src.ingest.clean import build_hyphen_vocab, join_lines, normalize_text
from src.ingest.docs import DOCS, cite_text
from src.ingest.validate import validate
from src.retrieval.citations import build_citation_index, find_citations
from src.retrieval.tokenize import tokenize


# ----------------------------------------------------------------------------- clean functions
def test_normalize_strips_soft_hyphen_and_hair_space():
    assert normalize_text("Or\xadganization of") == "Organization of"


def test_rule_dash_is_canonicalised():
    assert normalize_text("Rule 10b–5(b)") == "Rule 10b-5(b)"
    assert normalize_text("Rules 10b–5(a) and 10b–5(c)") == "Rules 10b-5(a) and 10b-5(c)"


def test_curly_quotes_and_section_sign_are_kept():
    s = "“[t]o make” 15 U. S. C. §78j(b)"
    assert normalize_text(s) == s


@pytest.fixture(scope="module")
def vocab():
    return build_hyphen_vocab(["a half-truth is", "forward-looking statements", "risk-factor section"])


def test_soft_hyphen_at_line_end_joins_without_hyphen(vocab):
    assert join_lines(["Maritime Or\xad", "ganization formally"], vocab) == "Maritime Organization formally"


def test_typesetting_hyphen_is_dropped(vocab):
    assert join_lines(["an implau-", "sible understanding"], vocab) == "an implausible understanding"


def test_real_compound_keeps_hyphen(vocab):
    assert join_lines(["a half-", "truth"], vocab) == "a half-truth"
    assert join_lines(["the risk-", "factor section"], vocab) == "the risk-factor section"


def test_compound_already_containing_hyphen_keeps_break_hyphen(vocab):
    assert join_lines(["cut-and-", "dry"], vocab) == "cut-and-dry"


def test_citation_numbers_keep_hyphen(vocab):
    assert join_lines(["Rule 10b-", "5(b) makes"], vocab) == "Rule 10b-5(b) makes"
    assert join_lines(["Rule 206(4)-", "8(a)(1)"], vocab) == "Rule 206(4)-8(a)(1)"


def test_en_dash_at_line_end_joins_without_space(vocab):
    assert join_lines(["Rule 10b–", "5(b) makes"], vocab) == "Rule 10b-5(b) makes"


# ------------------------------------------------------------------- sentences / tokens / cites
def test_sentence_splitter_respects_legal_abbreviations():
    text = "See Basic Inc. v. Levinson, 485 U.S. 224, 239 (1988). The Court agreed. Id. at 150. Then it ruled."
    assert split_sentences(text) == [
        "See Basic Inc. v. Levinson, 485 U.S. 224, 239 (1988).",
        "The Court agreed.",
        "Id. at 150.",
        "Then it ruled.",
    ]


def test_tokenizer_keeps_rule_compound_and_parts():
    toks = tokenize("Rule 10b-5(b) and Item 303")
    assert "10b-5(b)" in toks and "10b" in toks and "5" in toks and "item" in toks and "303" in toks


def test_cite_text():
    assert cite_text(5, 5) == "p.5"
    assert cite_text(5, 6) == "pp.5–6"
    assert cite_text(11, 12, 45, 45) == "¶45"
    assert cite_text(11, 12, 45, 52) == "¶¶45–52"


# -------------------------------------------------------------------------- corpus invariants
@pytest.fixture(scope="module")
def corpus():
    from src.ingest.__main__ import run_pipeline

    chunks, sections, pages, count, _ = run_pipeline()
    return chunks, sections, count


def test_no_validation_problems(corpus):
    chunks, _, count = corpus
    assert validate(chunks, count) == []


def test_every_document_is_present(corpus):
    chunks, _, _ = corpus
    assert {c.doc_id for c in chunks} == set(DOCS)


def test_chunk_count_is_in_expected_range(corpus):
    chunks, _, _ = corpus
    assert 150 <= sum(c.searchable for c in chunks) <= 320


def test_toc_and_toa_are_indexed_but_not_searchable(corpus):
    chunks, _, _ = corpus
    non = [c for c in chunks if c.section_kind in ("toc", "toa")]
    assert non and not any(c.searchable for c in non)
    assert all(c.prev_id is None and c.next_id is None for c in non)


def test_overlap_only_inside_a_section_and_never_for_complaint(corpus):
    chunks, _, _ = corpus
    assert all(c.overlap_tokens == 0 for c in chunks if c.doc_id == "sec_chiueh")
    first_of_section = {}
    for c in chunks:
        first_of_section.setdefault((c.doc_id, tuple(c.section_path), c.section_kind), c)
    assert all(c.overlap_tokens == 0 for c in first_of_section.values())


def test_complaint_has_eleven_claim_chunks_with_paragraph_ranges(corpus):
    chunks, _, _ = corpus
    claims = [c for c in chunks if c.doc_id == "sec_chiueh" and "CLAIM FOR RELIEF" in c.header]
    assert len(claims) == 11
    assert all(c.para_start is not None and c.citation.startswith("[Chiueh ¶") for c in claims)


def test_citation_labels(corpus):
    chunks, _, _ = corpus
    mac = next(c for c in chunks if c.doc_id == "macquarie" and c.searchable)
    assert mac.citation.startswith("[Macquarie, p")


def test_citation_index_links_macquarie_but_not_the_2006_last_atlantis(corpus):
    chunks, _, _ = corpus
    index = build_citation_index(chunks)
    macq = find_citations(index, "Macquarie")
    corpus_entry = next(e for e in macq if e["reporter_cite"] == "601 U.S. 257")
    assert corpus_entry["in_corpus_doc_id"] == "macquarie"
    assert {"fb_petitioners", "fb_amicus"} <= set(corpus_entry["doc_ids"])

    omnicare = next(e for e in find_citations(index, "Omnicare") if e["reporter_cite"] == "575 U.S. 175")
    assert {"fb_petitioners", "fb_amicus"} <= set(omnicare["doc_ids"])

    cboe = next(e for e in find_citations(index, "Last Atlantis") if e["reporter_cite"] == "455 F. Supp. 2d 788")
    assert cboe["in_corpus_doc_id"] is None


def test_rule_entries_exist(corpus):
    chunks, _, _ = corpus
    ids = {e["citation_id"] for e in build_citation_index(chunks)}
    assert {"rule:10b-5(b)", "rule:item-303", "rule:section-10(b)"} <= ids
