"""
Citation & quote detector: strip text that is legitimately attributed.

Academic writing has three categories of text we should NOT flag as plagiarism:
  1. Direct quotes (in "..." or “...”)
  2. Block quotes (indented or preceded by line break)
  3. Paraphrased text WITH inline citation (e.g. (Smith, 2020))
  4. Bibliographies / reference lists

Detecting these reduces false positives significantly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Regex patterns for academic text
# ---------------------------------------------------------------------------

# Inline citation patterns (English + Indonesian)
# Examples:
#   (Smith, 2020)
#   (Smith & Jones, 2019)
#   (Smith et al., 2018)
#   (Smith, 2020, p. 42)
#   (Smith, 2020; Jones, 2021)
#   (Kementerian Pendidikan, 2022)
#   (Undang-Undang No. 20, 2003)
INLINE_CITATION_RE = re.compile(
    r"""
    \(                                     # opening paren
    (?:[A-Z][A-Za-z\-']+                   # author surname
       (?:(?:\s*(?:&|and|dan)\s*|,\s*)[A-Z][A-Za-z\-']+)*  # more authors
       (?:,?\s*et\s+al\.?)?                # et al.
    )
    (?:,?\s*)?                             # optional comma
    (?:\d{4}[a-z]?)                        # year
    (?:[,:]?\s*(?:p\.|pp\.|hal\.?|h\.)\s*\d+(?:[-–]\d+)?)?  # page
    (?:\s*[;,]\s*                          # multi-cite separator
       (?:[A-Z][A-Za-z\-']+(?:\s+(?:&|and|dan)\s+[A-Z][A-Za-z\-']+)*(?:\s+et\s+al\.?)?
          ,?\s*\d{4}[a-z]?
          (?:[,:]?\s*(?:p\.|pp\.|hal\.?|h\.)\s*\d+(?:[-–]\d+)?)?
       )
    )*
    \)                                     # closing paren
    """,
    re.VERBOSE,
)

# Footnote/endnote markers
FOOTNOTE_REF_RE = re.compile(r"\[\s*\d+\s*\]|\[\s*[A-Z][A-Za-z]+\s*\]")

# Direct quote patterns
# Smart quotes: “...” or "..."
DOUBLE_QUOTE_RE = re.compile(r'["“](.+?)["”]', re.DOTALL)
# Single quotes in some styles for emphasis
SINGLE_QUOTE_RE = re.compile(r"‘(.+?)’", re.DOTALL)

# Block quote indicators (Markdown `> ` or HTML `<blockquote>`)
BLOCKQUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)

# Bibliography section markers
BIBLIO_HEADER_RE = re.compile(
    r"(?im)^\s*("
    r"references?|bibliography|daftar\s+pustaka|dapus|"
    r"bibliografi|DAFTAR\s+PUSTAKA|pustaka"
    r")\s*[:\-]?\s*$"
)

# Numbered reference list (e.g. "1. Smith, A. (2020). Title...")
NUMBERED_REF_RE = re.compile(r"^\s*\[\d+\]\s+[A-Z]|\s+\d+\.\s+[A-Z][a-z]+,\s+[A-Z]\.", re.MULTILINE)

# URL in reference
URL_IN_REF_RE = re.compile(r"https?://\S+|doi\.org/\S+|doi:\s*\S+", re.IGNORECASE)

# Common academic phrases that often wrap citations
CITATION_PHRASE_RE = re.compile(
    r"(?i)according to|menurut|disampaikan oleh|as stated by|"
    r"as reported by|sebagaimana dikemukakan oleh|"
    r"in the work of|dalam karya"
)


@dataclass
class CitationStats:
    """Stats returned by the citation parser for transparency."""
    inline_citations_stripped: int = 0
    direct_quotes_stripped: int = 0
    block_quotes_stripped: int = 0
    footnote_refs_stripped: int = 0
    bibliography_chars_stripped: int = 0
    urls_stripped: int = 0
    original_chars: int = 0
    cleaned_chars: int = 0

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (1 - self.cleaned_chars / self.original_chars) * 100


def strip_citations_and_quotes(text: str) -> tuple[str, CitationStats]:
    """Remove cited/quoted/bibliography text. Return (clean_text, stats)."""
    stats = CitationStats(original_chars=len(text))
    out = text

    # 1. Remove block quotes (entire lines starting with `>`)
    block_count = len(BLOCKQUOTE_LINE_RE.findall(out))
    if block_count:
        out = BLOCKQUOTE_LINE_RE.sub("", out)
        stats.block_quotes_stripped = block_count

    # 2. Remove bibliography sections
    bib_match = BIBLIO_HEADER_RE.search(out)
    if bib_match:
        out = out[: bib_match.start()].rstrip()
        stats.bibliography_chars_stripped = len(text) - len(out)

    # 3. Strip direct quotes
    dq_count = len(DOUBLE_QUOTE_RE.findall(out))
    if dq_count:
        out = DOUBLE_QUOTE_RE.sub("", out)
        stats.direct_quotes_stripped = dq_count

    sq_count = len(SINGLE_QUOTE_RE.findall(out))
    if sq_count:
        out = SINGLE_QUOTE_RE.sub("", out)
        stats.direct_quotes_stripped += sq_count

    # 4. Strip inline citations
    cit_count = len(INLINE_CITATION_RE.findall(out))
    if cit_count:
        out = INLINE_CITATION_RE.sub("", out)
        stats.inline_citations_stripped = cit_count

    # 5. Strip footnote references
    fn_count = len(FOOTNOTE_REF_RE.findall(out))
    if fn_count:
        out = FOOTNOTE_REF_RE.sub("", out)
        stats.footnote_refs_stripped = fn_count

    # 6. Strip URLs (often in references)
    url_count = len(URL_IN_REF_RE.findall(out))
    if url_count:
        out = URL_IN_REF_RE.sub("", out)
        stats.urls_stripped = url_count

    # 7. Numbered references at end of doc
    nr_count = len(NUMBERED_REF_RE.findall(out))
    if nr_count > 2:  # need at least 3 to look like a reference list
        out = NUMBERED_REF_RE.sub("", out)

    # Collapse excessive whitespace
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()

    stats.cleaned_chars = len(out)
    return out, stats


def is_likely_citation_sentence(sentence: str) -> bool:
    """Quick check: is this sentence mostly a citation wrapper?"""
    s = sentence.strip()
    if not s:
        return False
    # Short sentences with author-year pattern are likely citations
    if len(s) < 200 and INLINE_CITATION_RE.search(s):
        return True
    if CITATION_PHRASE_RE.search(s) and INLINE_CITATION_RE.search(s):
        return True
    return False
