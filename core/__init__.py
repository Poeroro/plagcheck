"""PlagCheck — $0 budget plagiarism checker."""
from .parser import parse_any, split_paragraphs, split_sentences
from .fingerprint import make_minhash, estimate_jaccard, build_lsh
from .corpus import Corpus, CorpusDoc
from .engine import PlagEngine
from .report import CheckResult, Match, build_report_html, highlight_pair
from .citations import strip_citations_and_quotes, CitationStats, is_likely_citation_sentence
from .stopwords_id import preprocess_id, ID_STOPWORDS

__all__ = [
    "parse_any",
    "split_paragraphs",
    "split_sentences",
    "make_minhash",
    "estimate_jaccard",
    "build_lsh",
    "Corpus",
    "CorpusDoc",
    "PlagEngine",
    "CheckResult",
    "Match",
    "build_report_html",
    "highlight_pair",
    "strip_citations_and_quotes",
    "CitationStats",
    "is_likely_citation_sentence",
    "preprocess_id",
    "ID_STOPWORDS",
]
