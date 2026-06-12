"""
MinHash fingerprinting for fast near-duplicate detection.
Uses datasketch.MinHash + LSH (Locality-Sensitive Hashing) index.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from datasketch import MinHash, MinHashLSH


# Tunables
SHINGLE_K = 3              # word-level n-gram size (smaller = more lenient)
MINHASH_PERM = 128         # number of permutations
LSH_THRESHOLD = 0.3        # Jaccard threshold for LSH candidate retrieval


WORD_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    """Generate word-level k-shingles as a set."""
    words = WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def make_minhash(text: str, num_perm: int = MINHASH_PERM) -> MinHash:
    """Build a MinHash signature for a text chunk."""
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def estimate_jaccard(a: MinHash, b: MinHash) -> float:
    return a.jaccard(b)


def build_lsh(minhashes: Iterable[MinHash], threshold: float = LSH_THRESHOLD) -> MinHashLSH:
    """Wrap an LSH index around a list of MinHash objects."""
    idx = MinHashLSH(threshold=threshold, num_perm=MINHASH_PERM)
    for i, m in enumerate(minhashes):
        idx.insert(str(i), m)
    return idx


@dataclass
class Fingerprint:
    """Container for a chunk fingerprint."""
    index: int
    text: str
    minhash: MinHash
    shingles: set[str]
