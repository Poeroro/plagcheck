#!/usr/bin/env python3
"""
Garuda (Garba Rujukan Digital) crawler.
Source: https://garuda.kemdiktisaintek.go.id

Hierarchy:
  /area/index/{area_id}?page={n}  -> 10 journals per page
  /journal/view/{journal_id}      -> lists issues
  /journal/view/{journal_id}?issue={name} -> lists articles
  /documents/detail/{doc_id}      -> full abstract + metadata

We crawl:
  1. Pick N subject areas
  2. For each subject: paginate to get M journal IDs
  3. For each journal: get all issues
  4. For each issue: get all article IDs
  5. For each article: fetch abstract, save as txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import requests

BASE = "https://garuda.kemdiktisaintek.go.id"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
STATE_FILE = CORPUS_DIR / ".garuda_state.json"

# Top subject areas for skripsi plagiarism detection
PRIORITY_SUBJECTS = {
    60:  "Computer_Science_IT",
    97:  "Education",
    95:  "Economics",
    137: "Law",
    115: "Engineering",
    185: "Public_Health",
    120: "Environmental_Science",
    190: "Social_Sciences",
    170: "Medicine_Pharmacology",
    150: "Mathematics",
}


class TextExtractor(HTMLParser):
    """Extract clean text from HTML, skip script/style."""
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip = True
        d = dict(attrs)
        if tag == "a" and "href" in d:
            self.parts.append(f"\n[{d['href']}] ")
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def fetch(url: str, *, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout,
                         allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  [fetch ERR] {url}: {e}", file=sys.stderr)
        return None


def get_journals_in_subject(subject_id: int, max_pages: int) -> list[int]:
    """Return journal IDs from /area/index/{subject_id}?page=N"""
    journals: list[int] = []
    for page in range(1, max_pages + 1):
        url = f"{BASE}/area/index/{subject_id}?page={page}"
        html = fetch(url)
        if not html:
            break
        ids = sorted(set(int(x) for x in
                         re.findall(r'href="/journal/view/(\d+)"', html)))
        if not ids:
            break  # no more journals
        new = [j for j in ids if j not in journals]
        if not new:
            break
        journals.extend(new)
        time.sleep(0.5)  # polite
    return journals


def get_issues(journal_id: int) -> list[str]:
    """Return issue query strings (e.g. 'Vol 12 No 2 (2019): Desember')."""
    url = f"{BASE}/journal/view/{journal_id}"
    html = fetch(url)
    if not html:
        return []
    issues: list[str] = []
    for m in re.finditer(r'href="[^"]*issue=([^"&]+)', html):
        issue = unquote(m.group(1))
        if issue not in issues:
            issues.append(issue)
    return issues


def get_article_ids(journal_id: int, issue: str, max_issues: int) -> list[int]:
    """Return document IDs in a journal issue."""
    url = f"{BASE}/journal/view/{journal_id}"
    r = requests.get(url, params={"issue": issue}, headers={"User-Agent": UA},
                     timeout=20)
    if r.status_code != 200:
        return []
    doc_ids = sorted(set(int(x) for x in
                         re.findall(r'href="/documents/detail/(\d+)"', r.text)))
    return doc_ids[:max_issues]


def fetch_article(doc_id: int) -> Optional[dict]:
    """Fetch full article abstract + metadata."""
    url = f"{BASE}/documents/detail/{doc_id}"
    html = fetch(url)
    if not html:
        return None

    p = TextExtractor()
    p.feed(html)
    text = p.get_text()

    # Extract metadata via regex (works on the article-display div)
    def find_meta(pattern: str, default: str = "") -> str:
        m = re.search(pattern, html, re.DOTALL)
        if m:
            # Strip HTML tags
            inner = re.sub(r"<[^>]+>", "", m.group(1)).strip()
            inner = re.sub(r"\s+", " ", inner)
            return inner
        return default

    title = find_meta(r'<xmp>([^<]+)</xmp>\s*</h3>', default="")
    if not title:
        # try first h3 with title
        m = re.search(r'<h3[^>]*>\s*<xmp>([^<]+)</xmp>', html, re.DOTALL)
        if m:
            title = m.group(1).strip()

    # Abstract is in <xmp class="abstract-article">
    abstract = find_meta(r'<xmp class="abstract-article">([^<]+)</xmp>',
                         default="")

    # Authors (in title-article block, look for author links)
    authors: list[str] = []
    for m in re.finditer(r'<a href="/author/view/\d+"><xmp>([^<]+)</xmp>', html):
        authors.append(m.group(1).strip())
    if not authors:
        # fallback: find author-article-afil blocks
        for m in re.finditer(r'<xmp>([A-Z][^<]{2,40})</xmp>\s*</a>\s*<span class="author-article-afil"', html):
            authors.append(m.group(1).strip())

    # Journal name (look for j-title)
    journal = find_meta(r'<a href="/journal/view/\d+"\s*><xmp>([^<]+)</xmp>',
                       default="")

    # Volume/Issue
    vol_issue = ""
    for m in re.finditer(r'<xmp>(Vol[^<]*)</xmp>', html):
        vol_issue = m.group(1).strip()
        break

    # Publish date
    pub_date = ""
    m = re.search(r'Publish Date\s*</h4>\s*<p>([^<]+)</p>', html, re.DOTALL)
    if m:
        pub_date = m.group(1).strip()

    # Subject
    subject = ""
    m = re.search(r'<a class="ui tag label[^"]*" href="/area/index/\d+">([^<]+)</a>', html)
    if m:
        subject = m.group(1).strip()

    return {
        "doc_id": doc_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "journal": journal,
        "volume_issue": vol_issue,
        "publish_date": pub_date,
        "subject": subject,
        "url": f"https://garuda.kemdiktisaintek.go.id/documents/detail/{doc_id}",
    }


def save_article(meta: dict, source_label: str, corpus_dir: Path) -> Optional[Path]:
    """Save article abstract to corpus/ as txt. Returns path or None on skip."""
    if not meta.get("abstract") and not meta.get("title"):
        return None
    # Build doc_id hash to avoid filename collisions
    h = hashlib.md5(f"{meta['doc_id']}-{meta['title']}".encode()).hexdigest()[:10]
    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", meta.get("title", "unknown"))[:60].strip("_")
    fname = f"garuda__{source_label}__{safe_title}_{h}.txt"
    path = corpus_dir / fname
    if path.exists():
        return None  # already saved

    # Save JUST the abstract (no metadata) to avoid Sastrawi preprocessing
    # diluting the embedding with tokenized "authors", "journal", URL parts.
    body = meta.get("abstract", "")
    if not body or len(body.strip()) < 50:
        return None  # skip docs without real abstract

    path.write_text(body, encoding="utf-8")
    return path


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"saved_doc_ids": [], "subjects_done": [], "errors": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="60,97,95,137,115,185",
                    help="Comma-separated subject IDs to crawl")
    ap.add_argument("--max-journals-per-subject", type=int, default=10,
                    help="Limit journals per subject (for sample mode)")
    ap.add_argument("--max-articles", type=int, default=200,
                    help="Total article limit (for sample mode)")
    ap.add_argument("--max-issues-per-journal", type=int, default=3,
                    help="Limit issues per journal (chronological order)")
    ap.add_argument("--articles-per-issue", type=int, default=10,
                    help="Limit articles per issue")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds between requests")
    args = ap.parse_args()

    subject_ids = [int(x) for x in args.subjects.split(",") if x.strip()]
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    saved_ids = set(state.get("saved_doc_ids", []))
    errors = state.get("errors", 0)

    total_saved = 0
    per_subject_count: dict[int, int] = {s: 0 for s in subject_ids}
    queue: list[tuple[int, int, str]] = []  # (subject_id, journal_id, issue)

    # Phase 1: build queue (round-robin journals across subjects so we get
    # diverse coverage, not all-CS like the first run).
    per_subject_journals: dict[int, list[int]] = {}
    per_subject_journal_idx: dict[int, int] = {}
    print(f"=== PHASE 1: Building queue (subjects={subject_ids}) ===")
    for subj in subject_ids:
        label = PRIORITY_SUBJECTS.get(subj, f"subj_{subj}")
        print(f"\n[Subject {subj}: {label}]")
        journals = get_journals_in_subject(subj, max_pages=max(1, args.max_journals_per_subject // 10 + 1))
        journals = journals[:args.max_journals_per_subject]
        per_subject_journals[subj] = journals
        per_subject_journal_idx[subj] = 0
        print(f"  Found {len(journals)} journals")
        time.sleep(args.delay)

    # Round-robin: take 1 journal at a time from each subject
    print(f"\n=== PHASE 1b: Round-robin queue ===")
    while total_saved < args.max_articles:
        any_added = False
        for subj in subject_ids:
            if total_saved >= args.max_articles:
                break
            journals = per_subject_journals.get(subj, [])
            idx = per_subject_journal_idx[subj]
            if idx >= len(journals):
                continue
            journal_id = journals[idx]
            per_subject_journal_idx[subj] = idx + 1
            issues = get_issues(journal_id)[:args.max_issues_per_journal]
            time.sleep(args.delay)
            for issue in issues:
                queue.append((subj, journal_id, issue))
            any_added = True
        if not any_added:
            break

    print(f"\n  Queue size: {len(queue)}")
    print(f"=== PHASE 2: Fetching articles (target={args.max_articles}) ===")

    for i, (subj, journal_id, issue) in enumerate(queue):
        if total_saved >= args.max_articles:
            break
        if errors > 20:
            print("  Too many errors, stopping")
            break

        try:
            doc_ids = get_article_ids(journal_id, issue, args.articles_per_issue)
        except Exception as e:
            print(f"  [ERR] issue list: {e}")
            errors += 1
            continue
        time.sleep(args.delay)

        label = PRIORITY_SUBJECTS.get(subj, f"subj_{subj}")
        for doc_id in doc_ids:
            if doc_id in saved_ids:
                continue
            if total_saved >= args.max_articles:
                break
            print(f"  [{total_saved+1}/{args.max_articles}] doc {doc_id} (j={journal_id}, {label})")
            try:
                meta = fetch_article(doc_id)
            except Exception as e:
                print(f"    [ERR] {e}")
                errors += 1
                continue
            if not meta:
                errors += 1
                continue
            path = save_article(meta, label, CORPUS_DIR)
            if path:
                saved_ids.add(doc_id)
                total_saved += 1
                state["saved_doc_ids"] = list(saved_ids)
                state["errors"] = errors
                save_state(state)
                print(f"    ✓ {meta['title'][:50] if meta['title'] else '(no title)'}...")
            time.sleep(args.delay)

    print(f"\n=== DONE ===")
    print(f"  Saved: {total_saved} articles")
    print(f"  Errors: {errors}")
    print(f"  Corpus dir: {CORPUS_DIR}")


if __name__ == "__main__":
    main()
