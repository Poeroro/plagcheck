"""Pre-warm ONNX Hybrid cache for the 3452-doc corpus.

Run once after deploy to build the .npz cache. Subsequent semantic checks
load the cache from disk in <1s instead of re-encoding.

Usage:
    python prewarm_cache.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import Corpus, PlagEngine
from core.engine import ONNX_HYBRID_PATH


def main() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] Loading corpus...", flush=True)
    corpus = Corpus(str(ROOT / "corpus"))
    print(f"[{time.strftime('%H:%M:%S')}] Corpus: {len(corpus)} docs", flush=True)

    eng = PlagEngine(corpus)
    print(f"[{time.strftime('%H:%M:%S')}] Loading ONNX encoder...", flush=True)
    t0 = time.time()
    eng.enable_semantic()  # default = minilm-onnx-hybrid
    print(f"[{time.strftime('%H:%M:%S')}] ONNX loaded: {time.time()-t0:.2f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] Building corpus embeddings cache...", flush=True)
    t1 = time.time()
    chunks = corpus.chunked(max_chars=1500)
    print(f"[{time.strftime('%H:%M:%S')}] {len(chunks)} chunks to encode", flush=True)

    emb = eng._get_corpus_embeddings_for_model(
        eng._bi_encoder, "minilm-onnx-hybrid", chunks
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] Done! shape={emb.shape}, "
        f"elapsed={time.time()-t1:.1f}s",
        flush=True,
    )

    cache_dir = ROOT / "corpus" / ".embeddings_cache"
    print(f"[{time.strftime('%H:%M:%S')}] Cache saved to: {cache_dir}", flush=True)
    for f in sorted(cache_dir.glob("*")):
        print(f"  {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
