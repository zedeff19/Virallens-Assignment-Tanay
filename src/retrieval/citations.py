"""Citation index (regex over case names / reporter cites) backing find_citations().

`build_citation_index(chunks)` returns a list of entries:

    {citation_id, kind ("case" | "rule"), case_name, aliases[], reporter_cite, year,
     doc_ids[], chunk_ids[], count, in_corpus_doc_id}

Sources, in order:
1. full cites in body text:  "Basic Inc. v. Levinson, 485 U. S. 224, 239 (1988)"
2. short cites:              "Basic, 485 U.S. at 231"
3. brief tables of authorities (case + reporter cite + printed pages -> chunks via page labels)
4. rule / item / section references already extracted per chunk (`rule_refs`)

In-corpus linking is keyed on the reporter cite / docket, never on a short name: the 2010
*Last Atlantis* opinion in the corpus is not the 2006 *Last Atlantis v. CBOE* (455 F. Supp. 2d 788)
that it cites.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

_REPORTER = (
    r"U\.\s?S\.|S\.\s?Ct\.|L\.\s?Ed\.\s?2d|F\.\s?[234]d|F\.\s?4th|F\.\s?Supp\.\s?[23]d|F\.\s?Supp\.|"
    r"Fed\.\s?Appx\.|F\.\s?App’x|WL"
)
_WORD = r"[A-Z][\w.&'’\-]*,?"
_LINK = r"(?:of|the|and|for|de|ex|rel\.|&|on|in|at)"
_CASE = (
    rf"(?:In re )?{_WORD}(?: (?:{_WORD}|{_LINK})){{0,8}} v\. {_WORD}(?: (?:{_WORD}|{_LINK})){{0,10}}"
)
_FULL_CITE = re.compile(
    rf"(?P<name>{_CASE}),?\s+(?P<vol>\d{{1,4}})\s+(?P<rep>{_REPORTER})\s+(?P<page>\d{{1,5}})"
    r"(?P<tail>(?:,\s*\d+(?:[–-]\d+)?(?:,\s*n\.\s*\d+)?)*)"
    r"(?:\s*\((?P<paren>[^()]{0,40}?)\))?"
)
_SHORT_CITE = re.compile(
    rf"(?P<short>[A-Z][\w’'\-]+(?: [A-Z][\w’'\-]+)?),\s+(?P<vol>\d{{1,4}})\s+(?P<rep>{_REPORTER})\s+at\s+(?P<pin>\d{{1,5}})"
)
_SIGNALS = {
    "See", "Cf.", "Compare", "With", "Accord", "But", "Contra", "Also", "Thus", "And", "Under", "In", "The",
    "As", "Moreover", "However", "E.g.,", "See,", "Id.", "Ibid.", "Yet", "Because", "Where", "Here", "That",
    "Both", "When", "Although", "Following", "Indeed", "Instead", "Still", "Similarly", "Accordingly", "Further",
    "Section", "Rule", "Item", "Court", "This", "Its", "Other", "Some", "For", "Of", "By", "See,", "Courts",
}
_YEAR = re.compile(r"(\d{4})\s*$")
_TOA_ENTRY = re.compile(
    r"(?P<cite>\d{1,4}\s+(?:%s)\s+\d{1,5})[^.\n]{0,30}?\((?P<paren>[^()]{0,30}\d{4})\)[ .…]*(?P<pages>[\d,\s–\-]+|passim)?" % _REPORTER
)

# In-corpus documents, identified by reporter cite / name. Key on these, not on short names.
IN_CORPUS = [
    {
        "doc_id": "macquarie",
        "reporter": "601 U.S. 257",
        "name_re": re.compile(r"Macquarie Infrastructure Corp(?:oration|\.)?(?:, ET AL\.)? v\. Moab|\bMacquarie\b"),
        "alias": "Macquarie",
    },
    {
        "doc_id": "last_atlantis",
        "reporter": None,
        "name_re": re.compile(r"Last Atlantis Capital LLC v\. AGS Specialist Partners"),
        "alias": "Last Atlantis",
    },
]


def _norm_reporter(rep: str) -> str:
    rep = re.sub(r"\s+", " ", rep.replace("’", "'")).strip()
    rep = re.sub(r"U\. ?S\.", "U.S.", rep)
    rep = re.sub(r"S\. ?Ct\.", "S. Ct.", rep)
    rep = re.sub(r"F\. ?([234])d", r"F.\1d", rep)
    rep = re.sub(r"F\. ?4th", "F.4th", rep)
    rep = re.sub(r"F\. ?Supp\. ?([23])d", r"F. Supp. \1d", rep)
    rep = re.sub(r"Fed\. ?Appx\.", "Fed. Appx.", rep)
    return rep


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" ,")
    if name.startswith("In re "):
        return name
    words = name.split(" ")
    # drop signal words and tails of the previous sentence ("... Item 303. SEC. Macquarie ...")
    while words and (words[0] in _SIGNALS or re.fullmatch(r"(?:[A-Z]{2,}\.|Pp\.|Cir\.|Ct\.|Ibid\.)", words[0])):
        words.pop(0)
    return " ".join(words).strip(" ,")


def _short_alias(name: str) -> str:
    first = name.split(" v. ")[0]
    first = re.sub(r",?\s*(Inc|Corp|Co|Ltd|LLC|L\.P|N\.A)\.?,?$", "", first).strip(" ,")
    return first


def _expand_pages(spec: str | None) -> list[int]:
    pages: list[int] = []
    if not spec:
        return pages
    for part in re.split(r"[,\s]+", spec.strip()):
        m = re.fullmatch(r"(\d+)[–-](\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a < 15:
                pages.extend(range(a, b + 1))
            elif b >= a:
                pages.extend([a, b])
        elif part.isdigit():
            pages.append(int(part))
    return pages


def build_citation_index(chunks: list[Any]) -> list[dict]:
    """`chunks` are `Chunk` objects (or dicts with the same keys)."""

    def g(c: Any, key: str):
        return c[key] if isinstance(c, dict) else getattr(c, key)

    searchable = [c for c in chunks if g(c, "searchable")]
    entries: dict[str, dict] = {}
    by_volrep: dict[tuple[str, str], list[str]] = defaultdict(list)

    def entry(cite: str, name: str | None, year: str | None) -> dict:
        key = f"case:{cite.lower().replace(' ', '-').replace('.', '')}"
        e = entries.get(key)
        if e is None:
            e = entries[key] = {
                "citation_id": key,
                "kind": "case",
                "case_name": name or cite,
                "aliases": [],
                "reporter_cite": cite,
                "year": year,
                "doc_ids": [],
                "chunk_ids": [],
                "count": 0,
                "in_corpus_doc_id": None,
                "_names": Counter(),
            }
            m = re.match(r"(\d+) (.+) (\d+)$", cite)
            if m:
                by_volrep[(m.group(1), m.group(2))].append(key)
        if name:
            e["_names"][name] += 1
        if year and not e["year"]:
            e["year"] = year
        return e

    def add_hit(e: dict, c: Any) -> None:
        if g(c, "chunk_id") not in e["chunk_ids"]:
            e["chunk_ids"].append(g(c, "chunk_id"))
        if g(c, "doc_id") not in e["doc_ids"]:
            e["doc_ids"].append(g(c, "doc_id"))

    # 1) full cites
    for c in searchable:
        for m in _FULL_CITE.finditer(g(c, "text")):
            name = _clean_name(m.group("name"))
            if " v. " not in name and not name.startswith("In re "):
                continue
            cite = f"{m.group('vol')} {_norm_reporter(m.group('rep'))} {m.group('page')}"
            ym = _YEAR.search(m.group("paren") or "")
            e = entry(cite, name, ym.group(1) if ym else None)
            e["count"] += 1
            add_hit(e, c)

    # 2) short cites ("Basic, 485 U.S. at 231")
    for c in searchable:
        for m in _SHORT_CITE.finditer(g(c, "text")):
            rep = _norm_reporter(m.group("rep"))
            cands = by_volrep.get((m.group("vol"), rep), [])
            if len(cands) == 1:
                e = entries[cands[0]]
            elif len(cands) > 1:
                pin = int(m.group("pin"))
                best = max(
                    (k for k in cands if int(entries[k]["reporter_cite"].rsplit(" ", 1)[1]) <= pin),
                    key=lambda k: int(entries[k]["reporter_cite"].rsplit(" ", 1)[1]),
                    default=None,
                )
                if best is None:
                    continue
                e = entries[best]
            else:
                continue
            e["aliases"].append(_clean_name(m.group("short")))
            e["count"] += 1
            add_hit(e, c)

    # 3) tables of authorities: printed pages -> chunks
    offsets: dict[str, int] = {}
    for doc in {g(c, "doc_id") for c in searchable}:
        diffs = Counter(
            g(c, "page_start") - int(g(c, "page_label_start"))
            for c in searchable
            if g(c, "doc_id") == doc and str(g(c, "page_label_start")).isdigit()
        )
        if diffs:
            offsets[doc] = diffs.most_common(1)[0][0]
    for t in (c for c in chunks if g(c, "section_kind") == "toa"):
        flat = re.sub(r"\s+", " ", g(t, "text"))
        for m in _TOA_ENTRY.finditer(flat):
            cite_m = re.match(r"(\d+)\s+(.+?)\s+(\d+)$", m.group("cite"))
            if not cite_m:
                continue
            cite = f"{cite_m.group(1)} {_norm_reporter(cite_m.group(2))} {cite_m.group(3)}"
            key = f"case:{cite.lower().replace(' ', '-').replace('.', '')}"
            e = entries.get(key)
            if e is None:
                # name = text immediately before the reporter cite in the TOA
                before = flat[max(0, m.start() - 160) : m.start()]
                name = _clean_name(re.split(r"\.{3,}|\)\s", before)[-1].strip(" ,"))
                e = entry(cite, name if " v. " in name else None, None)
            doc = g(t, "doc_id")
            off = offsets.get(doc)
            for p in _expand_pages(m.group("pages")):
                if off is None:
                    continue
                pdf_page = p + off
                for c in searchable:
                    if g(c, "doc_id") == doc and g(c, "page_start") <= pdf_page <= g(c, "page_end"):
                        add_hit(e, c)

    # in-corpus linking (reporter cite / docket, not short names) + name mentions
    for spec in IN_CORPUS:
        target = None
        if spec["reporter"]:
            target = entries.get("case:" + spec["reporter"].lower().replace(" ", "-").replace(".", ""))
        if target is None and spec["reporter"]:
            target = entry(spec["reporter"], None, None)
        for c in searchable:
            if g(c, "doc_id") != spec["doc_id"] and spec["name_re"].search(g(c, "text")):
                if target is not None:
                    add_hit(target, c)
        if target is not None:
            target["in_corpus_doc_id"] = spec["doc_id"]
            target["aliases"].append(spec["alias"])

    out: list[dict] = []
    for e in entries.values():
        names = e.pop("_names")
        if names:
            e["case_name"] = max(names, key=lambda n: (len(n), names[n]))
        aliases = {_short_alias(n) for n in names} | set(e["aliases"]) | {e["case_name"]}
        e["aliases"] = sorted(a for a in aliases if a)
        e["chunk_ids"].sort()
        e["doc_ids"].sort()
        out.append(e)

    # rules / items / sections already extracted per chunk
    rules: dict[str, dict] = {}
    for c in searchable:
        for ref in g(c, "rule_refs"):
            r = rules.setdefault(
                ref,
                {
                    "citation_id": f"rule:{ref.lower().replace(' ', '-')}",
                    "kind": "rule",
                    "case_name": ref,
                    "aliases": [ref],
                    "reporter_cite": None,
                    "year": None,
                    "doc_ids": [],
                    "chunk_ids": [],
                    "count": 0,
                    "in_corpus_doc_id": None,
                },
            )
            r["chunk_ids"].append(g(c, "chunk_id"))
            if g(c, "doc_id") not in r["doc_ids"]:
                r["doc_ids"].append(g(c, "doc_id"))
            r["count"] += 1
    out.extend(sorted(rules.values(), key=lambda r: r["citation_id"]))
    out.sort(key=lambda e: (e["kind"], -e["count"], e["citation_id"]))
    return out


def find_citations(index: list[dict], query: str) -> list[dict]:
    """Case-insensitive lookup over names and aliases (used by the find_citations tool)."""
    q = re.sub(r"\s+", " ", query).strip().lower()
    if not q:
        return []
    hits = []
    for e in index:
        hay = [e["case_name"].lower(), (e["reporter_cite"] or "").lower()] + [a.lower() for a in e["aliases"]]
        if any(q == h or q in h for h in hay if h):
            hits.append(e)
    return hits
