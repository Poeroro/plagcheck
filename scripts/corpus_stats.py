"""Show PlagCheck corpus composition statistics.

Usage:
    python scripts/corpus_stats.py
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus"


def main() -> int:
    files = [p for p in CORPUS_DIR.iterdir() if p.is_file() and p.suffix == ".txt"]
    if not files:
        print("Corpus is empty.")
        return 1

    # Categorize by source prefix
    by_source: Counter = Counter()
    total_size = 0
    total_words = 0
    id_count = 0
    en_count = 0

    for f in files:
        size = f.stat().st_size
        total_size += size
        words = len(re.findall(r"\w+", f.read_text(encoding="utf-8", errors="ignore")))
        total_words += words

        name = f.stem
        # Categorize
        if name.startswith("news_"):
            parts = name.split("_", 2)
            source = parts[1] if len(parts) > 1 else "news"
            by_source[f"news.{source}"] += 1
        elif name.startswith("arxiv__"):
            by_source["arxiv"] += 1
        elif name.startswith("openalex_") or name.startswith("openalex__"):
            by_source["openalex"] += 1
        elif name.startswith("crossref"):
            by_source["crossref"] += 1
        elif name.startswith("local__"):
            by_source["local"] += 1
        elif name.startswith("cc_") or "commoncrawl" in name:
            by_source["commoncrawl"] += 1
        else:
            by_source["other"] += 1

        # Quick ID detection (Bahasa Indonesia function words)
        text_lower = f.read_text(encoding="utf-8", errors="ignore")[:5000].lower()
        id_words = {"yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak"}
        if sum(1 for w in id_words if w in text_lower) >= 4:
            id_count += 1
        else:
            en_count += 1

    # Report
    print(f"=== PlagCheck Corpus Stats ===\n")
    print(f"Total documents: {len(files):,}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Total words: {total_words / 1_000_000:.2f}M ({total_words:,})")
    print(f"Avg words/doc: {total_words // len(files):,}")
    print(f"")
    print(f"Language: {id_count:,} Indonesian, {en_count:,} English/mixed")
    print(f"")
    print(f"By source:")
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(files)
        print(f"  {source:25s}  {count:6,}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
