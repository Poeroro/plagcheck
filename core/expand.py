"""
Multi-source corpus expander: pulls free academic papers from arXiv,
OpenAlex, Crossref, and Garuda. All free APIs, no keys required.

Each fetched paper is stored as a text file under the corpus dir for
persistent local reuse.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from .corpus import CorpusDoc


USER_AGENT = "DoubleCheck/1.0 (research; contact: admin@tempmeil.xyz)"
RATE_LIMIT_DELAY = 0.3
HTTP_TIMEOUT = 60


def _sleep() -> None:
    time.sleep(RATE_LIMIT_DELAY)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------
def fetch_arxiv(query: str, max_results: int = 50,
                out_dir: Path | None = None) -> list[CorpusDoc]:
    """Fetch arXiv papers via the public export API."""
    url = "http://export.arxiv.org/api/query"
    docs: list[CorpusDoc] = []
    batch = 25
    fetched = 0
    start = 0
    while fetched < max_results:
        params = {
            "search_query": f"all:{query}",
            "max_results": str(min(batch, max_results - fetched)),
            "start": str(start),
            "sortBy": "relevance",
        }
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[arxiv] fetch failed: {e}")
            break
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        batch_count = 0
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
            link_el = entry.find("a:id", ns)
            link = link_el.text.strip() if link_el is not None else ""
            if not summary or not title:
                continue
            text = f"{title}\n\n{summary}"
            doc = CorpusDoc(
                doc_id=f"arxiv:{link or title[:60]}",
                title=title,
                text=text,
                source="arxiv",
                url=link,
            )
            docs.append(doc)
            if out_dir:
                _persist(out_dir, doc)
            batch_count += 1
        if batch_count == 0:
            break
        fetched += batch_count
        start += batch_count
        _sleep()
    return docs


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def fetch_openalex(query: str, max_results: int = 50,
                   out_dir: Path | None = None,
                   language: str = "en") -> list[CorpusDoc]:
    """Fetch OpenAlex works via the public API."""
    base = "https://api.openalex.org/works"
    docs: list[CorpusDoc] = []
    per_page = 25
    page = 1
    fetched = 0
    while fetched < max_results:
        params = {
            "search": query,
            "per_page": str(min(per_page, max_results - fetched)),
            "page": str(page),
        }
        if language:
            params["filter"] = f"language:{language}"
        try:
            r = requests.get(base, params=params, timeout=HTTP_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[openalex] fetch failed: {e}")
            break
        data = r.json()
        batch_count = 0
        for work in data.get("results", []):
            inv = work.get("abstract_inverted_index") or {}
            if not inv:
                continue
            words: list[tuple[int, str]] = []
            for word, positions in inv.items():
                for p in positions:
                    words.append((p, word))
            words.sort()
            abstract = " ".join(w for _, w in words)
            if not abstract:
                continue
            title = work.get("title") or ""
            year = work.get("publication_year")
            authors = ", ".join(
                a.get("author", {}).get("display_name", "")
                for a in (work.get("authorships") or [])[:5]
            )
            url = work.get("doi") or work.get("id", "")
            text = f"{title}\n\nAuthors: {authors}\nYear: {year}\n\n{abstract}"
            doc = CorpusDoc(
                doc_id=f"openalex:{work.get('id', title[:60])}",
                title=title or "(untitled)",
                text=text,
                source="openalex",
                url=url,
                extra={"year": year, "authors": authors},
            )
            docs.append(doc)
            if out_dir:
                _persist(out_dir, doc)
            batch_count += 1
        if batch_count == 0:
            break
        fetched += batch_count
        page += 1
        _sleep()
    return docs


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------
def fetch_crossref(query: str, max_results: int = 50,
                   out_dir: Path | None = None) -> list[CorpusDoc]:
    """Fetch Crossref works. Free, no key."""
    base = "https://api.crossref.org/works"
    docs: list[CorpusDoc] = []
    rows = 25
    offset = 0
    fetched = 0
    while fetched < max_results:
        params = {
            "query": query,
            "rows": str(min(rows, max_results - fetched)),
            "offset": str(offset),
        }
        try:
            r = requests.get(base, params=params, timeout=HTTP_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[crossref] fetch failed: {e}")
            break
        data = r.json()
        items = data.get("message", {}).get("items", [])
        batch_count = 0
        for item in items:
            abstract = item.get("abstract", "")
            if not abstract:
                continue
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()
            if len(abstract) < 100:
                continue
            title = (item.get("title", ["(untitled)"]) or ["(untitled)"])[0]
            url = item.get("URL", item.get("DOI", ""))
            year_list = item.get("issued", {}).get("date-parts", [[None]])
            year = (year_list[0] or [None])[0] if year_list else None
            doc = CorpusDoc(
                doc_id=f"crossref:{item.get('DOI', title[:60])}",
                title=title,
                text=f"{title}\n\n{abstract}",
                source="crossref",
                url=url,
                extra={"year": year, "doi": item.get("DOI", "")},
            )
            docs.append(doc)
            if out_dir:
                _persist(out_dir, doc)
            batch_count += 1
        if batch_count == 0:
            break
        fetched += batch_count
        offset += rows
        _sleep()
    return docs


# ---------------------------------------------------------------------------
# Garuda (Indonesian scientific publications)
# ---------------------------------------------------------------------------
def fetch_garuda(query: str, max_results: int = 30,
                 out_dir: Path | None = None) -> list[CorpusDoc]:
    """Scrape Garuda Kemdikbud."""
    api = "https://garuda.kemdikbud.go.id/api/garuda"
    docs: list[CorpusDoc] = []
    page = 1
    per_page = 10
    fetched = 0
    while fetched < max_results:
        params = {"page": str(page), "per_page": str(per_page), "query": query}
        try:
            r = requests.get(api, params=params, timeout=HTTP_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"[garuda] fetch failed: {e}")
            break
        try:
            data = r.json()
        except json.JSONDecodeError:
            break
        if isinstance(data, dict):
            items = data.get("data") or data.get("docs") or data.get("results") or []
        else:
            items = []
        batch_count = 0
        for it in items:
            abstract = (it.get("abstract") or it.get("abstrak")
                        or it.get("description") or "")
            if not abstract or len(abstract) < 50:
                continue
            title = it.get("title") or "(untitled)"
            url = it.get("url") or it.get("link") or ""
            doc = CorpusDoc(
                doc_id=f"garuda:{it.get('id', title[:60])}",
                title=title,
                text=f"{title}\n\n{abstract}",
                source="garuda",
                url=url,
                extra={"year": it.get("year") or it.get("tahun")},
            )
            docs.append(doc)
            if out_dir:
                _persist(out_dir, doc)
            batch_count += 1
        if batch_count == 0:
            break
        fetched += batch_count
        page += 1
        _sleep()
    return docs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _persist(out_dir: Path, doc: CorpusDoc) -> None:
    """Save a single doc to disk (body-only) for later reuse."""
    safe_title = "".join(c if c.isalnum() else "_" for c in doc.title[:60]) or "untitled"
    p = out_dir / f"{doc.source}__{safe_title}.txt"
    if not p.exists():
        p.write_text(doc.text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------
def expand_corpus(queries: list[str], sources: list[str],
                  max_per_source: int = 50,
                  out_dir: Path | None = None) -> list[CorpusDoc]:
    """Run all enabled sources for all queries. Dedupes by doc_id."""
    seen: set[str] = set()
    out: list[CorpusDoc] = []

    for query in queries:
        if "arxiv" in sources:
            for d in fetch_arxiv(query, max_results=max_per_source, out_dir=out_dir):
                if d.doc_id not in seen:
                    seen.add(d.doc_id)
                    out.append(d)
        if "openalex" in sources:
            for d in fetch_openalex(query, max_results=max_per_source, out_dir=out_dir):
                if d.doc_id not in seen:
                    seen.add(d.doc_id)
                    out.append(d)
        if "crossref" in sources:
            for d in fetch_crossref(query, max_results=max_per_source, out_dir=out_dir):
                if d.doc_id not in seen:
                    seen.add(d.doc_id)
                    out.append(d)
        if "garuda" in sources:
            for d in fetch_garuda(query, max_results=max_per_source, out_dir=out_dir):
                if d.doc_id not in seen:
                    seen.add(d.doc_id)
                    out.append(d)
    return out
