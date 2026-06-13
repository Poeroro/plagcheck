#!/usr/bin/env python3
"""Re-write Garuda corpus files to contain ONLY the abstract (no metadata).

The original crawler included title, authors, journal info, source URL
in each file. Sastrawi preprocessing then tokenized the metadata as
Indonesian words (treating "authors", "journal", "https" etc. as content),
which dilutes the abstract embedding and drops self-similarity from ~1.0
to ~0.58.

After running this, re-prewarm_cache.py to rebuild the embedding cache.
"""
from __future__ import annotations

import re
from pathlib import Path

CORPUS = Path("/home/ubuntu/plagcheck/corpus")

def extract_abstract(text: str) -> str:
    """Extract the Abstract section. Returns clean abstract text only."""
    # Find the "Abstract:" header
    m = re.search(r"^Abstract:\s*\n(.*?)$", text, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return text.strip()

rewritten = 0
total = 0
for f in CORPUS.glob("garuda__*.txt"):
    total += 1
    text = f.read_text(encoding="utf-8", errors="replace")
    new_text = extract_abstract(text)
    if new_text != text and len(new_text) > 100:
        f.write_text(new_text, encoding="utf-8")
        rewritten += 1

print(f"Processed: {total}")
print(f"Rewritten: {rewritten}")
