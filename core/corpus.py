"""
Corpus manager: handles local corpus files (text/pdf/docx) and
free external sources (arXiv, OpenAlex, Crossref, Garuda).

Indonesian texts are preprocessed with Sastrawi stemming so word-form
variants ("meneliti" / "penelitian" / "diteliti") match correctly.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .parser import parse_any


# ---------------------------------------------------------------------------
# Indonesian detection
# ---------------------------------------------------------------------------
_ID_FUNCTION_WORDS = frozenset({
    "yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak",
    "pada", "juga", "dari", "ada", "sudah", "telah", "oleh", "ke",
    "akan", "adalah", "sebagai", "antara", "ia", "dia", "mereka",
    "kita", "kami", "saya", "kamu", "tapi", "tetapi", "namun",
    "atau", "bila", "jika", "kalau", "ketika", "saat", "waktu",
    "banyak", "sedikit", "semua", "setiap", "bisa", "dapat", "harus",
    "perlu", "sangat", "amat", "sekali", "telah", "pernah", "selalu",
    "sering", "menurut", "bahwa", "yaitu", "yakni", "oleh",
    "kepada", "terhadap", "mengenai", "tentang", "dalam", "luar",
    "atas", "bawah", "depan", "belakang", "sebelum", "sesudah",
    "setelah", "ketika", "dimana", "kapan", "bagaimana", "mengapa",
    "siapa", "apa", "seperti", "mirip", "hampir", "nyaris",
    "lagi", "masih", "sedang", "sudah", "belum", "akan", "telah",
})


def looks_indonesian(text: str) -> bool:
    """Heuristic: does this text look Indonesian?

    Requires >=6 words AND >=8% of them are Indonesian function words.
    Threshold tuned to avoid false positives on English text while catching
    short Indonesian queries (paragraphs often <20 words).
    """
    if not text:
        return False
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return False
    hits = sum(1 for w in words if w in _ID_FUNCTION_WORDS)
    ratio = hits / len(words)
    return ratio > 0.08


def _preprocess_indonesian(text: str) -> str:
    """Apply Sastrawi stemming + stopword removal for ID text."""
    try:
        from .stopwords_id import tokenize_id, remove_id_stopwords, stem_id_words
        if not looks_indonesian(text):
            return text
        words = tokenize_id(text)
        words = remove_id_stopwords(words)
        words = stem_id_words(words)
        return " ".join(words)
    except Exception:  # noqa: BLE001
        return text


@dataclass
class CorpusDoc:
    """A document (or document chunk) in the comparison corpus."""
    doc_id: str
    title: str
    text: str
    source: str
    url: str = ""
    extra: dict = field(default_factory=dict)
    language: str = ""
    preprocessed: bool = False


class Corpus:
    """In-memory + on-disk corpus for plagiarism comparison."""

    def __init__(self, root: str | Path, preprocess_id: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs: list[CorpusDoc] = []
        self.preprocess_id = preprocess_id
        self._load_local()

    def _load_local(self) -> None:
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".docx", ".md"}:
                try:
                    text = parse_any(p)
                    if not text:
                        continue
                    lang = "id" if looks_indonesian(text) else "en"
                    processed_text = (_preprocess_indonesian(text)
                                     if self.preprocess_id and lang == "id"
                                     else text)
                    self.docs.append(CorpusDoc(
                        doc_id=f"local:{p.relative_to(self.root)}",
                        title=p.stem,
                        text=processed_text,
                        source="local",
                        url=str(p),
                        language=lang,
                        preprocessed=(lang == "id" and self.preprocess_id),
                    ))
                except Exception as e:  # noqa: BLE001
                    print(f"[corpus] skip {p}: {e}")

    def add(self, doc: CorpusDoc) -> None:
        if self.preprocess_id and doc.language == "id" and not doc.preprocessed:
            doc.text = _preprocess_indonesian(doc.text)
            doc.preprocessed = True
        self.docs.append(doc)

    def extend(self, docs: list[CorpusDoc]) -> None:
        for d in docs:
            self.add(d)

    def chunked(self, max_chars: int = 1500) -> list[CorpusDoc]:
        out: list[CorpusDoc] = []
        for d in self.docs:
            if len(d.text) <= max_chars:
                out.append(d)
                continue
            step = max_chars - 200
            chunks = [d.text[i : i + max_chars] for i in range(0, len(d.text), step)]
            for i, c in enumerate(chunks):
                out.append(CorpusDoc(
                    doc_id=f"{d.doc_id}#chunk{i}",
                    title=f"{d.title} [chunk {i + 1}/{len(chunks)}]",
                    text=c,
                    source=d.source,
                    url=d.url,
                    language=d.language,
                    extra={**d.extra, "parent": d.doc_id, "chunk": i},
                    preprocessed=d.preprocessed,
                ))
        return out

    # ------------------------------------------------------------------
    # Free external sources
    # ------------------------------------------------------------------
    def search_arxiv(self, query: str, max_results: int = 10) -> list[CorpusDoc]:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "max_results": str(max_results),
            "sortBy": "relevance",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[arxiv] query failed: {e}")
            return []
        import xml.etree.ElementTree as ET
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        docs: list[CorpusDoc] = []
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", default="", namespaces=ns).strip()
            summary = entry.findtext("a:summary", default="", namespaces=ns).strip()
            link_el = entry.find("a:id", ns)
            link = link_el.text.strip() if link_el is not None else ""
            if not summary:
                continue
            docs.append(CorpusDoc(
                doc_id=f"arxiv:{link or title[:50]}",
                title=title,
                text=summary,
                source="arxiv",
                url=link,
                language="en",
            ))
        return docs

    def search_semanticscholar(self, query: str, max_results: int = 10) -> list[CorpusDoc]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": str(min(max_results, 100)),
            "fields": "title,abstract,url,year,authors",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[s2] query failed: {e}")
            return []
        data = r.json()
        docs: list[CorpusDoc] = []
        for paper in data.get("data", []):
            abstract = paper.get("abstract") or ""
            if not abstract:
                continue
            docs.append(CorpusDoc(
                doc_id=f"s2:{paper.get('paperId', paper.get('title', '')[:50])}",
                title=paper.get("title", "(untitled)"),
                text=abstract,
                source="semanticscholar",
                url=paper.get("url", ""),
                language="en",
                extra={"year": paper.get("year")},
            ))
        return docs

    def export(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(
                [{
                    "doc_id": d.doc_id, "title": d.title, "text": d.text,
                    "source": d.source, "url": d.url, "language": d.language,
                    "preprocessed": d.preprocessed, "extra": d.extra,
                } for d in self.docs],
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.docs)
