"""
Corpus manager: handles local corpus files (text/pdf/docx) and
free external sources (arXiv, Semantic Scholar, Garuda).
"""
from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .parser import parse_any


@dataclass
class CorpusDoc:
    """A document (or document chunk) in the comparison corpus."""
    doc_id: str
    title: str
    text: str
    source: str           # "local" | "arxiv" | "semanticscholar" | "garuda"
    url: str = ""
    extra: dict = field(default_factory=dict)


class Corpus:
    """In-memory + on-disk corpus for plagiarism comparison."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs: list[CorpusDoc] = []
        self._load_local()

    def _load_local(self) -> None:
        """Load every supported file under the corpus root."""
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".txt", ".pdf", ".docx", ".md"}:
                try:
                    text = parse_any(p)
                    if not text:
                        continue
                    self.docs.append(
                        CorpusDoc(
                            doc_id=f"local:{p.relative_to(self.root)}",
                            title=p.stem,
                            text=text,
                            source="local",
                            url=str(p),
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[corpus] skip {p}: {e}")

    def add(self, doc: CorpusDoc) -> None:
        self.docs.append(doc)

    def extend(self, docs: list[CorpusDoc]) -> None:
        self.docs.extend(docs)

    def chunked(self, max_chars: int = 1500) -> list[CorpusDoc]:
        """Split long docs into smaller chunks for finer matching."""
        out: list[CorpusDoc] = []
        for d in self.docs:
            if len(d.text) <= max_chars:
                out.append(d)
                continue
            # Slide a window across the text
            step = max_chars - 200
            chunks = [d.text[i : i + max_chars] for i in range(0, len(d.text), step)]
            for i, c in enumerate(chunks):
                out.append(
                    CorpusDoc(
                        doc_id=f"{d.doc_id}#chunk{i}",
                        title=f"{d.title} [chunk {i + 1}/{len(chunks)}]",
                        text=c,
                        source=d.source,
                        url=d.url,
                        extra={**d.extra, "parent": d.doc_id, "chunk": i},
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Free external sources
    # ------------------------------------------------------------------
    def search_arxiv(self, query: str, max_results: int = 10) -> list[CorpusDoc]:
        """Fetch abstracts from arXiv matching a query. Free, no key."""
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
        # Tiny XML parser: extract <entry> blocks
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
            docs.append(
                CorpusDoc(
                    doc_id=f"arxiv:{link or title[:50]}",
                    title=title,
                    text=summary,
                    source="arxiv",
                    url=link,
                    extra={"query": query},
                )
            )
        return docs

    def search_semanticscholar(self, query: str, max_results: int = 10) -> list[CorpusDoc]:
        """Free GraphQL-style search via Semantic Scholar. Free, no key required for low volume."""
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
            docs.append(
                CorpusDoc(
                    doc_id=f"s2:{paper.get('paperId', paper.get('title', '')[:50])}",
                    title=paper.get("title", "(untitled)"),
                    text=abstract,
                    source="semanticscholar",
                    url=paper.get("url", ""),
                    extra={"year": paper.get("year"), "authors": paper.get("authors", [])},
                )
            )
        return docs

    def export(self, path: str | Path) -> None:
        """Save corpus snapshot to JSON."""
        path = Path(path)
        path.write_text(
            json.dumps(
                [
                    {
                        "doc_id": d.doc_id,
                        "title": d.title,
                        "text": d.text,
                        "source": d.source,
                        "url": d.url,
                        "extra": d.extra,
                    }
                    for d in self.docs
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self.docs)
