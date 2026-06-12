"""Indonesian news corpus crawler.

Pulls article URLs from sitemap.xml of major Indonesian news outlets,
extracts title + body, saves to corpus/ as .txt files for PlagCheck.

Sources (Tier 1 - quick wins):
- Detik.com         (largest ID portal, ~20k+ URLs available)
- Kompas.com        (national newspaper, ~3k URLs)
- CNNIndonesia.com  (TV news, ~2.7k URLs)
- Liputan6.com      (SCTV news portal, ~5k URLs)
- Tribunnews.com    (regional network, ~10k URLs)

Features:
- Polite rate limiting (1.5-3s between requests)
- Resumable: tracks crawled URLs in crawl_state.json
- Deduplication: skips URLs already in corpus/
- Per-source patterns for body extraction
- Saves as `txt__<source>__<hash>.txt` for clean attribution

Usage:
    python scripts/crawl_id_news.py --target 1000           # 1000 per source
    python scripts/crawl_id_news.py --source detik --target 500
    python scripts/crawl_id_news.py --resume                # continue from state
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"
STATE_FILE = Path(__file__).resolve().parent / ".crawl_state.json"

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 PlagCheck/0.5 (research; +https://plagcheck.tempmeil.xyz)"

SOURCES = {
    "detik": {
        "sitemap": "https://www.detik.com/sitemap.xml",
        "delay": 1.5,
        "body_patterns": [
            r'<div[^>]*class="[^"]*detail__body-text[^"]*"[^>]*>(.*?)</div>\s*<',
            r'<article[^>]*>(.*?)</article>',
        ],
        "title_patterns": [
            r'<meta property="og:title" content="([^"]+)"',
            r'<title>([^<]+)</title>',
        ],
    },
    "kompas": {
        "sitemap": "https://www.kompas.com/sitemap.xml",
        "delay": 2.0,
        "body_patterns": [
            r'<div[^>]*class="[^"]*read__content[^"]*"[^>]*>(.*?)</div>\s*<',
            r'<article[^>]*>(.*?)</article>',
        ],
        "title_patterns": [
            r'<meta property="og:title" content="([^"]+)"',
            r'<title>([^<]+)</title>',
        ],
    },
    "cnn": {
        "sitemap": "https://www.cnnindonesia.com/sitemap.xml",
        "delay": 2.0,
        "body_patterns": [
            r'<div[^>]*class="[^"]*detail__body-text[^"]*"[^>]*>(.*?)</div>\s*<',
            r'<article[^>]*>(.*?)</article>',
        ],
        "title_patterns": [
            r'<meta property="og:title" content="([^"]+)"',
            r'<title>([^<]+)</title>',
        ],
    },
    "liputan6": {
        "sitemap": "https://www.liputan6.com/sitemap.xml",
        "delay": 2.0,
        "body_patterns": [
            r'<div[^>]*class="[^"]*article-content-body[^"]*"[^>]*>(.*?)</div>\s*<',
            r'<article[^>]*>(.*?)</article>',
        ],
        "title_patterns": [
            r'<meta property="og:title" content="([^"]+)"',
            r'<title>([^<]+)</title>',
        ],
    },
    "tribun": {
        "sitemap": "https://www.tribunnews.com/sitemap.xml",
        "delay": 2.5,
        "body_patterns": [
            r'<div[^>]*class="[^"]*txt-article[^"]*"[^>]*>(.*?)</div>\s*<',
            r'<article[^>]*>(.*?)</article>',
        ],
        "title_patterns": [
            r'<meta property="og:title" content="([^"]+)"',
            r'<title>([^<]+)</title>',
        ],
    },
}


# ---------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------
def strip_html(html: str) -> str:
    """Remove tags, decode entities, collapse whitespace."""
    # Remove script/style
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode common entities (including numeric)
    import html as html_lib
    text = html_lib.unescape(text)
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_body(html: str, patterns: list[str]) -> str:
    """Extract article body using site-specific patterns."""
    # Remove ad/nav junk before extraction
    junk = re.compile(
        r"(ADVERTISEMENT|SCROLL TO CONTINUE|googletag\.cmd\.push|Baca juga:|Baca juga |"
        r"Simak Video|Advertisement continue reading|Baca :|Lihat juga|Artikel ini telah tayang|"
        r"Artikel ini telah terbit|Simak juga|RELATED:.+)$",
        re.IGNORECASE,
    )
    for pat in patterns:
        m = re.search(pat, html, re.DOTALL)
        if m:
            text = strip_html(m.group(1))
            # Remove junk lines
            text = junk.sub("", text).strip()
            if len(text) > 200:  # minimum article length
                return text
    return ""


def extract_title(html: str, patterns: list[str]) -> str:
    """Extract article title."""
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            title = strip_html(m.group(1))
            # Clean site suffix
            title = re.sub(r"\s*[-|]\s*(Detik|Kompas|CNN Indonesia|Liputan6|Tribun)\s*$", "", title, flags=re.IGNORECASE)
            return title.strip()[:200]
    return ""


# ---------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------
def parse_sitemap(xml: str) -> tuple[list[str], list[str]]:
    """Return (article_urls, sub_sitemap_urls)."""
    clean = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", xml, flags=re.DOTALL)
    locs = re.findall(r"<loc>\s*(\S+?)\s*</loc>", clean)
    articles = []
    subs = []
    for url in locs:
        if url.endswith(".xml"):
            subs.append(url)
        else:
            articles.append(url)
    return articles, subs


def fetch(url: str, timeout: int = 15) -> str | None:
    """HTTP GET with retries."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 200:
                return r.text
            if r.status_code in (403, 429):
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (404, 410):
                return None
            time.sleep(2)
        except Exception as e:
            print(f"  [warn] {url[:60]}: {e}", file=sys.stderr)
            time.sleep(3)
    return None


# ---------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"crawled": {}, "errors": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_duplicate(url: str, source: str) -> bool:
    """Check if URL hash already in corpus."""
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    out = CORPUS_DIR / f"news_{source}_{url_hash}.txt"
    return out.exists()


# ---------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------
def crawl_source(source_name: str, target: int, state: dict) -> int:
    """Crawl up to `target` articles from a single source. Returns count saved."""
    cfg = SOURCES[source_name]
    print(f"\n=== {source_name.upper()} ===", flush=True)
    print(f"  sitemap: {cfg['sitemap']}", flush=True)

    # Phase 1: collect all article URLs
    print("  [1/2] collecting URLs from sitemaps...", flush=True)
    master_xml = fetch(cfg["sitemap"])
    if not master_xml:
        print(f"  [err] failed to fetch master sitemap", flush=True)
        return 0

    direct_articles, sub_sitemaps = parse_sitemap(master_xml)
    print(f"  master sitemap: {len(direct_articles)} direct, {len(sub_sitemaps)} sub-sitemaps", flush=True)

    all_articles = list(direct_articles)
    for sub_url in sub_sitemaps[:30]:  # cap at 30 subs per source
        sub_xml = fetch(sub_url)
        if sub_xml:
            arts, _ = parse_sitemap(sub_xml)
            all_articles.extend(arts)
        time.sleep(0.3)

    # Dedup
    all_articles = list(dict.fromkeys(all_articles))
    # Filter: actual article URLs (skip non-article paths)
    skip_patterns = re.compile(
        r"/(video|foto|infografis|gallery|suara-pembaca|kolom|prokontra|quiz|kuis|podcast|live)/"
        r"|/read/\d+/\d+/\d+$"  # tribun archive pattern
    )
    all_articles = [u for u in all_articles if not skip_patterns.search(u)]
    print(f"  total unique articles: {len(all_articles)}", flush=True)

    # Phase 2: crawl
    print(f"  [2/2] crawling up to {target} articles...", flush=True)
    saved = 0
    fetched = state["crawled"].get(source_name, 0)
    for i, url in enumerate(all_articles):
        if saved >= target:
            break
        if fetched + i < state["crawled"].get(f"{source_name}_idx", 0):
            continue  # already done in a previous run

        if is_duplicate(url, source_name):
            continue

        html = fetch(url)
        if not html:
            state["errors"] = state.get("errors", 0) + 1
            continue

        body = extract_body(html, cfg["body_patterns"])
        if not body:
            continue

        title = extract_title(html, cfg["title_patterns"])
        if not title:
            title = urlparse(url).path.split("/")[-1].replace("-", " ")

        # Save
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        out = CORPUS_DIR / f"news_{source_name}_{url_hash}.txt"
        content = f"{title}\n\n{body}\n\nSource: {url}\n"
        out.write_text(content, encoding="utf-8")
        saved += 1
        if saved % 50 == 0:
            state["crawled"][source_name] = fetched + i + 1
            state["crawled"][f"{source_name}_idx"] = fetched + i + 1
            save_state(state)
            print(f"    saved {saved}/{target} from {source_name} (last: {title[:50]}...)", flush=True)

        time.sleep(cfg["delay"])

    state["crawled"][source_name] = fetched + saved
    state["crawled"][f"{source_name}_idx"] = fetched + saved
    save_state(state)
    print(f"  ✓ {source_name}: saved {saved} new articles", flush=True)
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES.keys()) + ["all"], default="all")
    ap.add_argument("--target", type=int, default=1000, help="articles per source")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    state = load_state()
    if not args.resume:
        # Fresh start
        state = {"crawled": {}, "errors": 0}

    sources = list(SOURCES.keys()) if args.source == "all" else [args.source]
    total_saved = 0
    t0 = time.time()
    for src in sources:
        n = crawl_source(src, args.target, state)
        total_saved += n
    dt = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"  Total new articles: {total_saved}")
    print(f"  Elapsed: {dt/60:.1f} min")
    print(f"  Next: re-run prewarm_cache.py to add new docs to embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
