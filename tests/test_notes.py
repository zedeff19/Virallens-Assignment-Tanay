"""Offline tests for src/agents/notes.py (the analyst's `save_note` tool)."""
from __future__ import annotations

from src.agents import notes
from src.agents.prompts import ANALYST_PROMPT


def test_note_path_is_slugged():
    assert notes.note_path("Macq Note") == "/notes/macq-note.md"
    assert notes.note_path("../../etc/passwd") == "/notes/etc-passwd.md"
    assert notes.note_path("   ") == "/notes/note.md"
    # same slug rule as evidence_path, so the Editor's TASK_ID names both files alike
    from src.agents.evidence import evidence_path

    assert notes.note_path("meta-risk").rsplit("/", 1)[1] == evidence_path("meta-risk").rsplit("/", 1)[1]


def test_render_note_writes_the_header():
    out = notes.render_note(" t1 ", "fb  petitioners", " argument ", "\nPOSITION: x [FB Pet. Br., p.1]\n")
    assert out == "# Note: t1 | fb petitioners | argument\n\nPOSITION: x [FB Pet. Br., p.1]\n"


def test_status_complete_without_missing_lines():
    assert notes.status_and_gaps("POSITION: x\nARGUMENTS:\n1. y") == ("COMPLETE", "none")


def test_status_partial_collects_missing_lines():
    body = "POSITION: x\nMISSING: the other side's reply\n1. y\n  MISSING: the date"
    assert notes.status_and_gaps(body) == ("PARTIAL", "the other side's reply the date")


def test_gaps_are_truncated():
    body = "MISSING: " + " ".join(f"w{i}" for i in range(50))
    status, gaps = notes.status_and_gaps(body)
    assert status == "PARTIAL"
    assert len(gaps.split()) == notes.MAX_GAP_WORDS + 1  # 30 words and the " ..." marker
    assert gaps.endswith("...")


def test_prompt_describes_save_note_not_write_file():
    assert "save_note(" in ANALYST_PROMPT
    assert "write_file" not in ANALYST_PROMPT
