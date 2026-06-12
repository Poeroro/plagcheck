"""
Document parsers: extract clean text from PDF and DOCX files.
Strips headers, footers, page numbers, and excessive whitespace.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document


# Patterns that look like page numbers / running headers / footers
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MULTI_WS_RE = re.compile(r"[ \t]{2,}")
MULTI_NL_RE = re.compile(r"\n{3,}")


def parse_pdf(path: str | Path) -> str:
    """Extract text from a PDF, page by page. Returns concatenated text."""
    doc = fitz.open(str(path))
    parts: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text:
            parts.append(text)
    doc.close()
    return _clean("\n".join(parts))


def parse_docx(path: str | Path) -> str:
    """Extract paragraph text from a DOCX file."""
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _clean("\n".join(parts))


def parse_txt(path: str | Path) -> str:
    """Read a plain text file."""
    return _clean(Path(path).read_text(encoding="utf-8", errors="ignore"))


def parse_any(path: str | Path) -> str:
    """Dispatch by file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(p)
    if ext == ".docx":
        return parse_docx(p)
    if ext in {".txt", ".md"}:
        return parse_txt(p)
    raise ValueError(f"Unsupported file type: {ext}")


def _clean(text: str) -> str:
    """Normalize whitespace and drop obvious noise."""
    # Drop line that is just a page number
    lines = [ln for ln in text.splitlines() if not PAGE_NUMBER_RE.match(ln)]
    text = "\n".join(lines)
    # Collapse runs of spaces and newlines
    text = MULTI_WS_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def split_paragraphs(text: str, min_len: int = 40) -> list[str]:
    """Split cleaned text into paragraph-sized chunks for matching."""
    paras: list[str] = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if len(chunk) >= min_len:
            paras.append(chunk)
        elif paras and len(chunk) > 0:
            # Merge short tail into previous paragraph
            paras[-1] = paras[-1] + " " + chunk
    return paras


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter. Good enough for ID + EN."""
    # Split on . ! ? followed by space/newline, but keep abbreviation-ish splits conservative
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"“])", text)
    return [s.strip() for s in raw if len(s.strip()) > 15]
