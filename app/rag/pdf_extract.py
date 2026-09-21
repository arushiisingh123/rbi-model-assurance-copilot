"""PDF text extraction with page and clause provenance (owner: Nidhi / RAG).

WHAT THIS MODULE DOES
    Reads an RBI source PDF page by page and returns its text **with the
    structure a citation needs**: which page each span of text came from, and
    which numbered clause was in force at that point in the document.

    It answers two questions for any character offset in the extracted text:

        page_for_offset(n)     -> which printed page is this on?
        section_for_offset(n)  -> which clause was this under?

    ``app/rag/chunking.py`` consumes those in Phase B to fill the
    ``page`` / ``section_id`` / ``section_title`` fields ``DocumentChunk``
    already has. This module does not chunk, embed, retrieve, or interpret.

WHY LINE STRUCTURE IS PRESERVED
    RBI Master Directions put the clause number at the START of a line
    ("16. Aspects to be considered in agreement"). Any extraction that
    collapses newlines destroys the only signal that distinguishes a clause
    number from an ordinary number inside a sentence. ``pypdf`` keeps line
    breaks, so they are kept here too: ``PageText.text`` is the page's text
    as extracted, newlines intact.

THE NULL RULE
    A clause identifier is emitted ONLY when the document states it at the
    start of a line in one of the forms below. Everything else yields no
    ``SectionMark``, and a chunk built over that span gets ``section_id
    None``. ``DocumentChunk``'s own docstring gives the reason: a citation
    pointing at "page 1" because no page was known is worse than one that
    admits it has no page -- it looks checkable, and checking it finds the
    wrong thing.

    Nothing here normalises, renumbers or completes a clause identifier.
    What is emitted is what the page prints.

NUMBERING FORMS RECOGNISED (verified against the three Phase A sources)
    "16. Aspects to be considered in agreement"  -> id "16",     title set
    "1.6    (i)  These directions are ..."       -> id "1.6",    title None
    "a) details of the activity being ..."       -> id "16(a)"   (under 16)
    "(b) These Directions shall not apply ..."   -> id "4(b)"    (under 4)
    "Chapter - V" + "Outsourcing Agreement"      -> id "Chapter V", title set

    A sub-item is composed onto the most recent top-level clause on the same
    page-run, which is how the document itself reads. A sub-item appearing
    before any numbered clause is skipped rather than attached to a guess.

KNOWN DIVERGENCE -- READ BEFORE MATCHING AGAINST verified_requirements.py
    The PDF and the RBI website number sub-items DIFFERENTLY for the 2023 IT
    Outsourcing Directions. The website renders clause 16's sub-items as
    16.1, 16.2 ... 16.13; the PDF renders the same sub-items as a), b) ...
    m). They are the same provisions in the same order (m is the 13th
    letter), but this module emits what the PDF prints -- "16(m)" -- and
    never rewrites it to "16.13". Deriving one numbering from the other is a
    mapping neither document states, and inventing it here would put a
    clause reference on screen that the cited source does not contain.
    Reconciling the two numbering schemes is a Phase C decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Separator placed between pages when their text is joined. A newline keeps
# the last line of one page from running into the first line of the next,
# which would otherwise create a line that starts with neither.
PAGE_SEPARATOR = "\n"

# A table-of-contents line: "4. Regulatory requirements ......... 12".
# These carry a real clause number but point at a listing, not the clause, so
# treating one as a clause mark would place a citation on the contents page.
_TOC_LINE = re.compile(r"\.{4,}")

# How many dot-leader lines make a page a contents page. Whole pages are
# rejected rather than individual lines because the leaders sometimes wrap,
# leaving a bare "24. Infor" that looks exactly like a heading. Three is
# comfortably above any incidental ellipsis in body text.
_TOC_PAGE_THRESHOLD = 3

# "16. Aspects to be considered in agreement" / "2. Applicability"
_NUMBERED_HEADING = re.compile(r"^(\d{1,3})\.\s+(\S.*)$")

# "1.6 These directions are concerned with ..." / "16.13 ..."
_DECIMAL_CLAUSE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){1,2})\s+(\S.*)$")

# "(b) These Directions shall not be applicable to:"  and  "a) details of ..."
_SUB_ITEM = re.compile(r"^\(?([a-z]{1,2}|[ivxl]{1,5})\)\s+(\S.*)$")

# "Chapter - V", "Chapter V", "CHAPTER - II", "Part A"
_CHAPTER = re.compile(r"^(Chapter|Part|Section)\s*[-–—]?\s*([IVXLC]+|[A-Z]|\d{1,2})\s*$", re.I)

# A heading's trailing text is only treated as a section TITLE when it reads
# like one: short, and not a sentence. A long line after a clause number is
# the clause's requirement text, not its name.
_MAX_TITLE_CHARS = 60


class PDFExtractionError(Exception):
    """A PDF could not be read, or yielded no usable text."""


@dataclass(frozen=True)
class PageText:
    """One page's extracted text, line structure intact.

    ``page_number`` is 1-based and is the page's position in the PDF, which
    is what a reviewer counts when they open the file. It is NOT the number
    printed in the page header -- RBI PDFs carry front matter, so the two
    differ. Nothing here guesses the printed number.
    """

    page_number: int
    text: str


@dataclass(frozen=True)
class SectionMark:
    """A clause identifier the document states, and where it starts.

    ``section_id`` is verbatim from the page. ``section_title`` is the
    heading text when the line carried one, otherwise ``None`` -- never a
    truncated slice of requirement text.
    """

    section_id: str
    section_title: Optional[str]
    page_number: int
    offset: int


@dataclass(frozen=True)
class ExtractedDocument:
    """A PDF's text plus the provenance needed to cite any part of it."""

    text: str
    pages: tuple[PageText, ...]
    sections: tuple[SectionMark, ...]
    # (page_number, start_offset, end_offset) into ``text``, in page order.
    page_spans: tuple[tuple[int, int, int], ...]
    # Pages identified as a table of contents. Their text is kept -- it is
    # part of the document -- but no clause may be attributed to it, because
    # the clause in force from earlier pages is not the clause a contents
    # LISTING is about. Consumers use this to suppress clause attribution.
    contents_pages: frozenset[int] = frozenset()

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def is_contents_page(self, page_number: int) -> bool:
        """Whether a page is a contents listing rather than substantive text."""
        return page_number in self.contents_pages

    def page_for_offset(self, offset: int) -> Optional[int]:
        """The 1-based page containing ``offset``, or None if out of range.

        Returns None rather than clamping to the first or last page: an
        offset outside the document is a caller bug, and answering it with a
        plausible page number would hide that bug behind a citation.
        """
        for page_number, start, end in self.page_spans:
            if start <= offset < end:
                return page_number
        return None

    def section_for_offset(self, offset: int) -> Optional[SectionMark]:
        """The most recent clause mark at or before ``offset``.

        None when the offset precedes every detected clause -- front matter,
        covering letter and contents pages legitimately have no clause.
        """
        found: Optional[SectionMark] = None
        for mark in self.sections:
            if mark.offset <= offset:
                found = mark
            else:
                break
        return found


def _clean_line(line: str) -> str:
    """Trim a line without altering its words."""
    return line.strip()


def _title_or_none(remainder: str) -> Optional[str]:
    """The heading text of a numbered line, or None if it is body text.

    A numbered line in an RBI Direction is either a heading ("16. Aspects to
    be considered in agreement") or the opening of a requirement ("1.6 These
    directions are concerned with managing risks ..."). Only the first is a
    title. The test is deliberately crude and conservative -- length, and
    the absence of sentence punctuation -- because a wrong title is a wrong
    citation label.
    """
    text = remainder.strip()
    if not text or len(text) > _MAX_TITLE_CHARS:
        return None
    if text.endswith((".", ";", ":", ",")):
        return None
    # A heading does not contain a full stop mid-line.
    if ". " in text:
        return None
    # Nor a comma: "In order to ensure effective management of attendant
    # risks, the" is the opening of a covering-letter paragraph, not a name.
    if "," in text:
        return None
    return text


def _parent_is_plausible(section_id: str, current_integer: Optional[int]) -> bool:
    """Whether a decimal clause continues the document rather than citing one.

    Accepts the clause we are inside (2.6 -> 2.7) and the next one up
    (2.7 -> 3.1.1). Rejects anything further away, which in these documents
    is always a reference to a different instrument's paragraph.
    """
    if current_integer is None:
        return True
    head = section_id.split(".", 1)[0]
    if not head.isdigit():
        return False
    return int(head) in (current_integer, current_integer + 1)


def _next_nonblank(lines, position: int) -> Optional[str]:
    """The next non-blank line after ``position``, stripped, or None.

    Used only for chapter headings, whose name RBI prints on the line below
    the chapter number.
    """
    for line in lines[position + 1 :]:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _is_contents_page(text: str) -> bool:
    """Whether this page is a table of contents.

    Contents pages carry real clause numbers against listing entries, so
    every line on one is a false clause mark. Detected per page because the
    dot leaders sometimes wrap, leaving a bare "24. Infor" that no
    line-level rule can tell from a genuine heading.
    """
    return sum(1 for line in text.splitlines() if _TOC_LINE.search(line)) >= (
        _TOC_PAGE_THRESHOLD
    )


def _detect_mark(
    line: str,
    *,
    page_number: int,
    offset: int,
    current_integer: Optional[int],
    current_parent: Optional[str],
    current_parent_title: Optional[str],
    next_line: Optional[str] = None,
) -> Optional[SectionMark]:
    """A clause mark for this line, or None.

    ``current_integer`` is the whole-number clause currently in force (16)
    and gates decimal cross-references. ``current_parent`` is whatever a
    sub-item should attach to, which may be a decimal ("1.6") -- the two are
    tracked separately so a decimal clause does not reject its own siblings.

    Order matters: decimal clauses are tested before bare numbered headings
    so "1.6" is not read as heading "1".
    """
    if _TOC_LINE.search(line):
        return None

    chapter = _CHAPTER.match(line)
    if chapter:
        label = f"{chapter.group(1).title()} {chapter.group(2).upper()}"
        # RBI prints a chapter's name on the line BELOW its number:
        #   "Chapter - V"
        #   "Outsourcing Agreement"
        # so the title is read from the next line when the caller supplies
        # one and it looks like a heading. Same conservative test as every
        # other title; a line that reads as prose yields None.
        return SectionMark(label, _title_or_none(next_line or ""), page_number, offset)

    decimal = _DECIMAL_CLAUSE.match(line)
    if decimal:
        section_id = decimal.group(1)
        # A decimal whose parent is neither the clause we are currently
        # inside nor the next one is a CROSS-REFERENCE that happens to have
        # wrapped to the start of a line -- "paragraphs 1.5, 1.4 and 1.3 ...
        # of the Annex to RBI circular ...". Accepting it would relabel every
        # following chunk with a clause number from a different instrument.
        #
        # "the next one" has to be allowed because documents advance: 2.7 is
        # followed by 3.1.1. Without it, every clause after the first chapter
        # of the Fraud Risk Directions went undetected. The caller resets the
        # parent at a chapter heading, where numbering legitimately restarts.
        if not _parent_is_plausible(section_id, current_integer):
            return None
        return SectionMark(
            section_id,
            _title_or_none(decimal.group(2)),
            page_number,
            offset,
        )

    heading = _NUMBERED_HEADING.match(line)
    if heading:
        return SectionMark(
            heading.group(1),
            _title_or_none(heading.group(2)),
            page_number,
            offset,
        )

    sub = _SUB_ITEM.match(line)
    if sub and current_parent:
        # Composed onto the clause it sits under, exactly as the page reads.
        # Without a parent clause there is nothing honest to compose onto, so
        # the sub-item is skipped rather than attached to a guess.
        #
        # It INHERITS the parent's title, because that is what the document
        # says: sub-item m) of "16. Aspects to be considered in agreement"
        # sits under that heading and has no heading of its own. The title
        # names the section the text is in; section_id still identifies the
        # sub-item exactly. Nothing is invented -- if the parent had no
        # printed heading, the sub-item has none either.
        return SectionMark(
            f"{current_parent}({sub.group(1)})",
            current_parent_title,
            page_number,
            offset,
        )

    return None


def _is_numbered(section_id: str) -> bool:
    """Whether a detected id is a clause number (not a chapter or sub-item)."""
    return "(" not in section_id and section_id[:1].isdigit()


def _is_chapter(section_id: str) -> bool:
    """Whether a detected id is a chapter/part/section heading."""
    return section_id[:1].isalpha()


def extract_pdf(path: Path | str) -> ExtractedDocument:
    """Extract one PDF's text, pages and clause marks.

    Raises
    ------
    PDFExtractionError
        If the file cannot be opened as a PDF, or every page yields empty
        text (a scanned document with no text layer). An empty extraction is
        an error rather than an empty success: silently indexing nothing
        would leave the corpus looking populated.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PDFExtractionError(
            "pypdf is required to read PDF sources; it is declared in "
            "requirements.txt"
        ) from exc

    pdf_path = Path(path)
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise PDFExtractionError(f"could not open {pdf_path} as a PDF: {exc}") from exc

    pages: list[PageText] = []
    page_spans: list[tuple[int, int, int]] = []
    sections: list[SectionMark] = []
    contents_pages: list[int] = []
    parts: list[str] = []
    cursor = 0
    current_integer: Optional[int] = None
    current_parent: Optional[str] = None
    current_parent_title: Optional[str] = None

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            raise PDFExtractionError(
                f"could not extract text from page {index} of {pdf_path}: {exc}"
            ) from exc

        pages.append(PageText(page_number=index, text=raw))

        start = cursor
        # Walk the page's own lines so each mark gets a real offset. Contents
        # pages are read for text but never for clause marks.
        if _is_contents_page(raw):
            contents_pages.append(index)
        else:
            line_offset = start
            page_lines = raw.splitlines(keepends=True)
            for position, line in enumerate(page_lines):
                stripped = _clean_line(line)
                if stripped:
                    mark = _detect_mark(
                        stripped,
                        page_number=index,
                        offset=line_offset,
                        current_integer=current_integer,
                        current_parent=current_parent,
                        current_parent_title=current_parent_title,
                        next_line=_next_nonblank(page_lines, position),
                    )
                    if mark is not None:
                        sections.append(mark)
                        if _is_chapter(mark.section_id):
                            # Clause numbering restarts at a chapter, so the
                            # previous chapter's number must stop gating.
                            current_integer = None
                        elif _is_numbered(mark.section_id):
                            current_parent = mark.section_id
                            # A sub-item inherits whatever heading its parent
                            # printed -- which may be nothing.
                            current_parent_title = mark.section_title
                            head = mark.section_id.split(".", 1)[0]
                            if head.isdigit():
                                current_integer = int(head)
                line_offset += len(line)

        parts.append(raw)
        cursor = start + len(raw)
        page_spans.append((index, start, cursor))

        parts.append(PAGE_SEPARATOR)
        cursor += len(PAGE_SEPARATOR)

    text = "".join(parts)
    if not text.strip():
        raise PDFExtractionError(
            f"{pdf_path} yielded no extractable text -- it may be a scanned "
            "document with no text layer, which this extractor does not handle"
        )

    return ExtractedDocument(
        text=text,
        pages=tuple(pages),
        sections=tuple(sections),
        page_spans=tuple(page_spans),
        contents_pages=frozenset(contents_pages),
    )


__all__ = [
    "PageText",
    "SectionMark",
    "ExtractedDocument",
    "PDFExtractionError",
    "extract_pdf",
    "PAGE_SEPARATOR",
]
