"""Document registry: one `DocSpec` per source PDF, with the per-document cleaning rules.

Everything document-specific lives here so that load/clean/structure/chunk stay generic.
Page numbers are 1-based PDF page indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = ROOT / "index"  # gitignored; rebuilt by `python -m src.ingest.index`
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
CITATIONS_PATH = PROCESSED_DIR / "citations.json"
REPORT_PATH = PROCESSED_DIR / "ingest_report.md"
PREVIEW_PATH = PROCESSED_DIR / "chunks_preview.md"


@dataclass(frozen=True)
class DocSpec:
    doc_id: str
    file: str
    title: str  # used in chunk headers, e.g. "Macquarie v. Moab"
    doc_type: str
    cite_label: str  # used in citations, e.g. "Macquarie" -> [Macquarie, p.5]
    # Full-line regexes (after strip) for running headers/footers/rules to drop.
    header_res: tuple[str, ...] = ()
    # Where the bare printed page number sits on a page.
    page_number_pos: str = "first"  # "first" | "last" | "none"
    # Pages dropped entirely (cover sheets, parties, signature pages, ...).
    drop_pages: frozenset[int] = frozenset()
    # Pages whose content is table of contents / table of authorities (kept, not searchable).
    toc_pages: frozenset[int] = frozenset()
    toa_pages: frozenset[int] = frozenset()
    appendix_from: int | None = None
    # page -> regex; drop every line on that page *before* the first matching line.
    start_at: dict[int, str] = field(default_factory=dict)
    # page -> regex; drop the matching line and every later line on that page.
    stop_at: dict[int, str] = field(default_factory=dict)
    # Paragraph-level boilerplate to drop (regex on the paragraph start).
    drop_paragraph_res: tuple[str, ...] = ()
    # Top-level section split by PDF page, e.g. Macquarie syllabus vs opinion: (first_page, label).
    parts: tuple[tuple[int, str], ...] = ()
    # Heading detection: "bold" (font based) or "centered" (Macquarie bare roman numerals / letters).
    heading_mode: str = "bold"
    # Bold lines at/above this size are not headings (bold block quotes in the petitioners' brief).
    heading_max_size: float | None = None
    # Numbered paragraphs (complaint).
    numbered_paragraphs: bool = False
    # Page label prefix by page (Macquarie syllabus pages collide with opinion pages).
    label_prefix: dict[int, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return DATA_DIR / self.file


_ROMAN_PAGE = r"\(?[ivxIVX]+\)?|\(?\d+a?\)?"

DOCS: dict[str, DocSpec] = {
    "macquarie": DocSpec(
        doc_id="macquarie",
        file="1. Macquarie v. Moab (slip opinion).pdf",
        title="Macquarie v. Moab",
        doc_type="Supreme Court opinion",
        cite_label="Macquarie",
        header_res=(
            r"Cite as: 601 U\. S\. _+ \(2024\)",
            r"Opinion of the Court",
            r"Syllabus",
            r"MACQUARIE INFRASTRUCTURE CORP\. v\.",
            r"MOAB PARTNERS, L\. P\.",
            r"\(Slip Opinion\)",
            r"\(Slip Opinion\) OCTOBER TERM, 2023",
            r"OCTOBER TERM, 2023",
            r"\*",
            r"SUPREME COURT OF THE UNITED STATES",
            r"MACQUARIE INFRASTRUCTURE CORP\. ET AL\. v\.",
            r"MOAB PARTNERS, L\. P\., ET AL\.",
            r"CERTIORARI TO THE UNITED STATES COURT OF APPEALS FOR",
            r"THE SECOND CIRCUIT",
            r"No\. 22.1165",
            r"MACQUARIE INFRASTRUCTURE CORPORATION, ET AL\.,",
            r"PETITIONERS v\. MOAB PARTNERS, L\. P\., ET AL\.",
            r"ON WRIT OF CERTIORARI TO THE UNITED STATES COURT OF",
            r"APPEALS FOR THE SECOND CIRCUIT",
        ),
        page_number_pos="first",
        parts=((1, "Syllabus"), (4, "Opinion")),
        heading_mode="centered",
        drop_paragraph_res=(
            r"NOTE: Where it is feasible, a syllabus",
            r"NOTICE: This opinion is subject to formal revision",
        ),
        label_prefix={1: "Syl. ", 2: "Syl. ", 3: "Syl. "},
    ),
    "fb_petitioners": DocSpec(
        doc_id="fb_petitioners",
        file="2. Facebook v. Amalgamated Bank, petitioners' brief.pdf",
        title="Facebook v. Amalgamated Bank (petitioners' brief)",
        doc_type="Brief for petitioners",
        cite_label="FB Pet. Br.",
        header_res=(_ROMAN_PAGE,),
        page_number_pos="first",
        drop_pages=frozenset({1, 3, 51}),
        toc_pages=frozenset({4, 52}),
        toa_pages=frozenset({5, 6, 7}),
        appendix_from=53,
        start_at={8: r"^OPINIONS? BELOW$"},
        stop_at={50: r"^Respectfully submitted"},
        heading_max_size=11.5,
    ),
    "fb_amicus": DocSpec(
        doc_id="fb_amicus",
        file="3. Facebook v. Amalgamated Bank, U.S. amicus brief.pdf",
        title="Facebook v. Amalgamated Bank (U.S. amicus brief)",
        doc_type="U.S. amicus brief",
        cite_label="US Amicus Br.",
        header_res=(_ROMAN_PAGE, r"OCTOBER 2024"),
        page_number_pos="first",
        drop_pages=frozenset({1}),
        toc_pages=frozenset({3}),
        toa_pages=frozenset({4, 5, 6, 7}),
        start_at={8: r"^INTEREST OF THE UNITED STATES$"},
        stop_at={40: r"^Respectfully submitted"},
        heading_max_size=11.5,
    ),
    "last_atlantis": DocSpec(
        doc_id="last_atlantis",
        file="4. Last Atlantis Capital v. AGS Specialist Partners (district court opinion).pdf",
        title="Last Atlantis Capital v. AGS Specialist Partners",
        doc_type="District court opinion",
        cite_label="Last Atlantis",
        header_res=(r"Case: 1:04-cv-00397 Document #: 759 Filed: 03/26/10 Page \d+ of 20.*", r"\)"),
        page_number_pos="last",
        start_at={1: r"^Presently before me is a motion"},
        stop_at={20: r"^ENTER ORDER"},
    ),
    "sec_chiueh": DocSpec(
        doc_id="sec_chiueh",
        file="5. SEC v. Chiueh (First Amended Complaint).pdf",
        title="SEC v. Chiueh",
        doc_type="SEC complaint",
        cite_label="Chiueh",
        header_res=(r"Case 2:25-cv-01920-BRM-CLW +Document 9 +Filed 04/11/25 +Page \d+ of 51 PageID: \d+",),
        page_number_pos="first",
        drop_pages=frozenset({1, 49, 50, 51}),
        stop_at={48: r"^Dated:"},
        numbered_paragraphs=True,
    ),
}


def cite_text(page_start: int, page_end: int, para_start: int | None = None, para_end: int | None = None) -> str:
    """Canonical citation suffix: 'p.5', 'pp.5–6', '¶45', '¶¶45–52'."""
    if para_start is not None:
        if para_end is None or para_end == para_start:
            return f"¶{para_start}"
        return f"¶¶{para_start}–{para_end}"
    if page_end == page_start:
        return f"p.{page_start}"
    return f"pp.{page_start}–{page_end}"
