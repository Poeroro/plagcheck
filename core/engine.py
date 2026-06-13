"""DoubleCheck engine: orchestrator for parse → fingerprint → match → report.

Pipeline:
  1. Parse document
  2. Strip citations/quotes (optional)  ← reduces academic FP
  3. Sliding-window MinHash LSH          ← catches exact + near copies
  4. Semantic cosine (optional)          ← catches paraphrase
  5. Cross-encoder reranking (optional)  ← high-precision rerank
  6. Multi-model ensemble (optional)     ← 2 bi-encoders vote

Encoders support PyTorch (sentence-transformers) and ONNX (Hybrid INT8).
Hybrid uses quantized FFN + FP32 attention: -83% RAM, identical F1.
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path

from .corpus import Corpus, CorpusDoc
from .fingerprint import (
    LSH_THRESHOLD,
    SHINGLE_K,
    _shingles,
    build_lsh,
    estimate_jaccard,
    make_minhash,
)
from .parser import parse_any, split_paragraphs
from .report import CheckResult, Match, build_report_html, highlight_pair

# ONNX model path (Hybrid: FFN INT8, attention FP32, 132MB)
ROOT = Path(__file__).resolve().parent.parent
ONNX_HYBRID_PATH = ROOT / "models" / "onnx" / "minilm-hybrid"
ONNX_FP32_PATH = ROOT / "models" / "onnx" / "minilm-fp32"
ONNX_INT8_PATH = ROOT / "models" / "onnx" / "minilm-int8"


class OnnxEncoder:
    """Lightweight ONNX bi-encoder wrapper.

    Mimics ``sentence_transformers.SentenceTransformer.encode`` enough for
    the engine: takes list[str], returns ``np.ndarray`` shape ``(n, dim)``,
    L2-normalized when ``normalize_embeddings=True``.

    Uses mean pooling over token embeddings (with attention-mask weighting),
    the same pooling SBERT uses for ``paraphrase-multilingual-MiniLM-L12-v2``.
    """

    def __init__(self, model_path: str | Path, name: str = "minilm-onnx-hybrid"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_path = Path(model_path)
        self.name = name
        self.dim: int | None = None

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 3 threads sweet-spot on 4 vCPU (4 has thread contention overhead)
        opts.intra_op_num_threads = 3
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path / "model.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        # Probe output dim once
        inputs = self.tokenizer("probe", return_tensors="np")
        out = self.session.run(None, self._ort_inputs(inputs))[0]
        self.dim = int(out.shape[-1])

    def _ort_inputs(self, pt_inputs) -> dict:
        """Convert transformers BatchEncoding → onnxruntime input dict."""
        out = {}
        for k, v in pt_inputs.items():
            out[k] = v
        return out

    def _mean_pool(self, last_hidden: "np.ndarray", attention_mask: "np.ndarray") -> "np.ndarray":
        """Mean pool with attention-mask weighting. last_hidden: (B, T, H)."""
        import numpy as np
        mask = attention_mask[:, :, None].astype(last_hidden.dtype)
        summed = (last_hidden * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        return summed / counts

    def encode(self, texts, *, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, batch_size: int = 64, **_):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        all_vecs: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=384,  # covers avg corpus; long-tail truncated
                return_tensors="np",
            )
            ort_in = self._ort_inputs(enc)
            last_hidden = self.session.run(None, ort_in)[0]
            pooled = self._mean_pool(last_hidden, enc["attention_mask"])
            all_vecs.append(pooled)

        emb = np.concatenate(all_vecs, axis=0)
        if normalize_embeddings:
            norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
            emb = emb / norms
        return emb

    def __repr__(self) -> str:
        return f"<OnnxEncoder {self.name} dim={self.dim} path={self.model_path}>"


from .corpus import _preprocess_indonesian, looks_indonesian as _looks_id_paragraph  # noqa: E402


# Default thresholds
DEFAULT_NEAR = 0.30
DEFAULT_SEMANTIC = 0.75  # was 0.85 → 0.80 → 0.75. Cross-language paraphrase (ID↔EN) + Garuda ID paraphrase both score ~0.75-0.82.
DEFAULT_CROSS_ENCODER = 0.50

# Default ensemble models (lighter, faster, all multilingual)
# Primary uses ONNX Hybrid (saves 1.2GB RAM, 10x faster load).
# Secondary stays PyTorch mpnet (no ONNX exported; high-quality vote partner).
ENSEMBLE_MODELS = [
    "minilm-onnx-hybrid",                              # ONNX Hybrid INT8 (FFN quant)
    "paraphrase-multilingual-mpnet-base-v2",           # PyTorch mpnet (ensemble partner)
]


def _windowed_jaccard(query_shingles: set[str], corpus_text: str,
                      window_chars: int = 400,
                      step: int = 100) -> tuple[float, str]:
    words = corpus_text.split()
    if not words:
        return 0.0, ""
    cum = [0]
    for w in words:
        cum.append(cum[-1] + len(w) + 1)
    n = len(words)
    best = 0.0
    best_text = ""
    i = 0
    while i < n:
        j = i
        end_char = cum[i] + window_chars
        while j < n and cum[j] < end_char:
            j += 1
        if j - i < SHINGLE_K:
            i += max(1, (j - i) // 2 or 1)
            continue
        win_text = " ".join(words[i:j])
        win_shingles = _shingles(win_text, k=SHINGLE_K)
        if not win_shingles:
            i += 1
            continue
        inter = len(win_shingles & query_shingles)
        union = len(win_shingles | query_shingles)
        if union > 0:
            j_score = inter / union
            if j_score > best:
                best = j_score
                best_text = win_text
        if j >= n:
            break
        next_char = cum[i] + step
        k = i
        while k < n and cum[k] < next_char:
            k += 1
        i = max(i + 1, k)
    return best, best_text


class PlagEngine:
    """DoubleCheck orchestrator with multi-model ensemble support."""

    def __init__(self, corpus: Corpus, *, near_threshold: float = DEFAULT_NEAR,
                 semantic_threshold: float = DEFAULT_SEMANTIC,
                 cross_encoder_threshold: float = DEFAULT_CROSS_ENCODER,
                 cache_dir: Path | None = None,
                 ensemble_models: list[str] | None = None):
        self.corpus = corpus
        self.near_threshold = near_threshold
        self.semantic_threshold = semantic_threshold
        self.cross_encoder_threshold = cross_encoder_threshold

        # Backwards-compat: single model attribute
        self._bi_encoder = None
        # New: multi-model support
        self._bi_encoders: list = []
        self._ensemble_model_names: list[str] = ensemble_models or ENSEMBLE_MODELS

        self._cross_encoder = None

        # Caches (per-model, keyed by model name)
        self._cached_chunks: list[CorpusDoc] | None = None
        self._embeddings_cache: dict[str, object] = {}  # model_name -> np.ndarray

        self.cache_dir = cache_dir or (corpus.root / ".embeddings_cache")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_encoder(self, name: str):
        """Load encoder by name: dispatches to ONNX Hybrid or PyTorch.

        Returns ``(encoder_object, display_name)``. ``display_name`` is the
        string used for cache keys + log lines.
        """
        if name == "minilm-onnx-hybrid" or name.startswith("minilm-onnx-"):
            if not ONNX_HYBRID_PATH.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {ONNX_HYBRID_PATH}. "
                    "Run scripts/export_onnx.py to export."
                )
            print(f"[engine] loading ONNX encoder: {ONNX_HYBRID_PATH}")
            enc = OnnxEncoder(ONNX_HYBRID_PATH, name="minilm-onnx-hybrid")
            return enc, "minilm-onnx-hybrid"

        # PyTorch fallback (mpnet, etc.)
        from sentence_transformers import SentenceTransformer
        print(f"[engine] loading PyTorch encoder: {name}")
        return SentenceTransformer(name), name

    def enable_semantic(self, model_name: str = "minilm-onnx-hybrid") -> None:
        enc, display = self._load_encoder(model_name)
        self._bi_encoder = enc
        self._bi_encoders = [enc]
        self._ensemble_model_names = [display]

    def enable_ensemble(self, model_names: list[str] | None = None) -> None:
        """Load multiple bi-encoders for ensemble voting."""
        names = model_names or self._ensemble_model_names
        self._bi_encoders = []
        self._ensemble_model_names = []
        for n in names:
            try:
                enc, display = self._load_encoder(n)
                self._bi_encoders.append(enc)
                self._ensemble_model_names.append(display)
                print(f"[engine] loaded ensemble model: {display}")
            except Exception as e:  # noqa: BLE001
                print(f"[engine] failed to load {n}: {e}")
        if self._bi_encoders:
            self._bi_encoder = self._bi_encoders[0]

    def enable_cross_encoder(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> None:
        from sentence_transformers import CrossEncoder
        self._cross_encoder = CrossEncoder(model_name)

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------
    def _get_corpus_embeddings_for_model(self, model, model_name: str,
                                          corpus_chunks: list[CorpusDoc]):
        """Get or build cache for a specific model. Supports incremental
        updates: if the cache has SOME of the chunks but not all, only
        encodes the missing ones and merges.
        """
        import numpy as np
        import json as _json

        if model_name in self._embeddings_cache:
            return self._embeddings_cache[model_name]

        current_ids = [c.doc_id for c in corpus_chunks]
        current_id_to_idx = {cid: i for i, cid in enumerate(current_ids)}
        cache_file = self.cache_dir / f"emb_{model_name.replace('/', '_').replace('-', '_')}.npz"
        meta_file = self.cache_dir / f"ids_{model_name.replace('/', '_').replace('-', '_')}.json"

        # Try to load existing cache and incrementally update
        if cache_file.exists() and meta_file.exists():
            try:
                saved_ids = _json.loads(meta_file.read_text(encoding="utf-8"))
                saved_id_to_idx = {sid: i for i, sid in enumerate(saved_ids)}
                # Find which current IDs are missing from cache
                missing = [cid for cid in current_ids if cid not in saved_id_to_idx]
                if not missing and len(saved_ids) == len(current_ids):
                    # Perfect match: load and return
                    data = np.load(cache_file)
                    self._embeddings_cache[model_name] = data["emb"]
                    return data["emb"]
                if missing:
                    # Partial cache: encode only missing chunks
                    print(f"[engine] Cache miss: {len(missing)} new chunks "
                          f"(have {len(saved_ids)}/{len(current_ids)}). "
                          f"Encoding delta only...", flush=True)
                    data = np.load(cache_file)
                    old_emb = data["emb"]
                    # Texts of missing chunks
                    missing_texts = [c.text for c in corpus_chunks
                                     if c.doc_id in missing]
                    new_emb = model.encode(missing_texts, normalize_embeddings=True,
                                           show_progress_bar=False, batch_size=32)
                    # Build new array in current_ids order
                    merged = np.zeros((len(current_ids), old_emb.shape[1]),
                                      dtype=old_emb.dtype)
                    for i, cid in enumerate(current_ids):
                        if cid in saved_id_to_idx:
                            merged[i] = old_emb[saved_id_to_idx[cid]]
                        else:
                            # new_emb index = position in `missing` list
                            new_idx = missing.index(cid)
                            merged[i] = new_emb[new_idx]
                    # Normalize
                    norms = np.linalg.norm(merged, axis=1, keepdims=True).clip(min=1e-12)
                    merged = merged / norms
                    # Save updated cache
                    np.savez_compressed(cache_file, emb=merged)
                    meta_file.write_text(_json.dumps(current_ids, ensure_ascii=False),
                                        encoding="utf-8")
                    self._embeddings_cache[model_name] = merged
                    return merged
                # Else: lengths differ but no missing IDs (probably reordering)
                # Fall through to full re-encode
            except Exception as e:  # noqa: BLE001
                print(f"[engine] cache load failed, full re-encode: {e}", flush=True)

        # Full re-encode (slow path)
        print(f"[engine] Full re-encode of {len(corpus_chunks)} chunks...", flush=True)
        texts = [c.text for c in corpus_chunks]
        emb = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=32)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, emb=emb)
            meta_file.write_text(_json.dumps(current_ids, ensure_ascii=False),
                                encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[engine] cache save failed for {model_name}: {e}", flush=True)

        self._embeddings_cache[model_name] = emb
        return emb

    def invalidate_cache(self) -> None:
        self._cached_chunks = None
        self._embeddings_cache = {}
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Main check
    # ------------------------------------------------------------------
    def check(self, file_path: str | Path, top_k: int = 5,
              use_semantic: bool = False,
              use_cross_encoder: bool = False,
              strip_citations: bool = True,
              min_paragraph_chars: int = 80,
              use_ensemble: bool = False) -> CheckResult:
        started = datetime.utcnow().isoformat(timespec="seconds")
        t0 = time.time()

        raw_text = parse_any(file_path)
        if strip_citations:
            from .citations import strip_citations_and_quotes
            text, cite_stats = strip_citations_and_quotes(raw_text)
        else:
            text = raw_text
            from .citations import CitationStats
            cite_stats = CitationStats(original_chars=len(raw_text), cleaned_chars=len(raw_text))

        # Per-paragraph Indonesian preprocessing (preserves blank-line boundaries
        # so split_paragraphs can detect them — _preprocess_indonesian joins on
        # single space and would otherwise collapse the doc into one mega-paragraph).
        pre_split = split_paragraphs(text, min_len=min_paragraph_chars)
        if pre_split:
            text = "\n\n".join(
                _preprocess_indonesian(p) if _looks_id_paragraph(p) else p
                for p in pre_split
            )
        paragraphs = pre_split or ([text] if text else [])
        if not paragraphs:
            return CheckResult(
                document_name=Path(file_path).name,
                total_paragraphs=0, matches=[], overall_score=0.0,
                flagged_paragraphs=0,
                started_at=started,
                finished_at=datetime.utcnow().isoformat(timespec="seconds"),
                elapsed_seconds=0.0,
                corpus_size=len(self.corpus),
                citation_stats=cite_stats,
            )

        corpus_chunks = self.corpus.chunked(max_chars=1500)
        if not corpus_chunks:
            return CheckResult(
                document_name=Path(file_path).name,
                total_paragraphs=len(paragraphs), matches=[], overall_score=0.0,
                flagged_paragraphs=0,
                started_at=started,
                finished_at=datetime.utcnow().isoformat(timespec="seconds"),
                elapsed_seconds=time.time() - t0,
                corpus_size=0,
                citation_stats=cite_stats,
            )

        corpus_mh = [make_minhash(d.text) for d in corpus_chunks]
        lsh = build_lsh(corpus_mh, threshold=LSH_THRESHOLD)

        # Pre-compute embeddings
        corpus_emb_per_model: dict[str, object] = {}
        if use_ensemble and self._bi_encoders:
            for model, name in zip(self._bi_encoders, self._ensemble_model_names):
                corpus_emb_per_model[name] = self._get_corpus_embeddings_for_model(
                    model, name, corpus_chunks
                )
        elif use_semantic and self._bi_encoder is not None:
            corpus_emb_per_model[self._ensemble_model_names[0]] = \
                self._get_corpus_embeddings_for_model(
                    self._bi_encoder, self._ensemble_model_names[0], corpus_chunks
                )

        matches: list[Match] = []
        flagged_paragraphs = 0

        for para in paragraphs:
            para_shingles = _shingles(para, k=SHINGLE_K)
            para_mh = make_minhash(para)
            candidate_idx: set[int] = set()
            try:
                hits = lsh.query(para_mh)
                for h in hits:
                    try:
                        candidate_idx.add(int(h))
                    except (TypeError, ValueError):
                        pass
            except Exception:  # noqa: BLE001
                candidate_idx = set()
            for ci, cmh in enumerate(corpus_mh):
                if estimate_jaccard(para_mh, cmh) >= LSH_THRESHOLD * 0.7:
                    candidate_idx.add(ci)

            near_hits: list[tuple[float, CorpusDoc, str]] = []
            for ci in candidate_idx:
                score, win_text = _windowed_jaccard(para_shingles, corpus_chunks[ci].text)
                if score >= self.near_threshold:
                    near_hits.append((score, corpus_chunks[ci], win_text))

            # Semantic matching (single or ensemble)
            best: tuple[float, CorpusDoc, str, str] | None = None
            if near_hits:
                near_hits.sort(key=lambda x: -x[0])
                s, d, w = near_hits[0]
                kind = "exact" if s >= 0.9 else "near"
                best = (s, d, w, kind)
            elif corpus_emb_per_model:
                # Use ensemble voting or single model
                if use_ensemble and len(corpus_emb_per_model) > 1:
                    import numpy as np
                    # Get top candidate from each model
                    model_votes: list[tuple[float, int]] = []
                    for name, emb in corpus_emb_per_model.items():
                        model = self._bi_encoders[self._ensemble_model_names.index(name)]
                        para_emb = model.encode([para], normalize_embeddings=True,
                                                show_progress_bar=False)
                        sims = (emb @ para_emb.T).flatten()
                        top_idx = int(np.argmax(sims))
                        top_sim = float(sims[top_idx])
                        model_votes.append((top_sim, top_idx))

                    # Take the candidate that got the most "votes" (consensus)
                    from collections import Counter
                    idx_counter = Counter(idx for _, idx in model_votes)
                    consensus_idx, votes = idx_counter.most_common(1)[0]
                    # Average similarity across models for that idx
                    sims_for_idx = [s for s, idx in model_votes if idx == consensus_idx]
                    avg_sim = sum(sims_for_idx) / len(sims_for_idx)

                    if avg_sim >= self.semantic_threshold:
                        best = (avg_sim, corpus_chunks[consensus_idx], corpus_chunks[consensus_idx].text, "ensemble")
                else:
                    # Single model (backwards-compat)
                    import numpy as np
                    name = list(corpus_emb_per_model.keys())[0]
                    emb = corpus_emb_per_model[name]
                    para_emb = self._bi_encoder.encode([para], normalize_embeddings=True,
                                                       show_progress_bar=False)
                    sims = (emb @ para_emb.T).flatten()
                    order = np.argsort(-sims)[:3]
                    for o in order:
                        s = float(sims[o])
                        if s >= self.semantic_threshold:
                            best = (s, corpus_chunks[o], corpus_chunks[o].text, "semantic")
                            break

            # Cross-encoder reranking
            if use_cross_encoder and self._cross_encoder is not None and best is not None:
                score, doc, win_text, kind = best
                if kind in ("semantic", "near", "ensemble"):
                    rerank_score = self._cross_encoder.predict(
                        [(para, win_text)], convert_to_numpy=True
                    )[0]
                    ce_prob = 1 / (1 + math.exp(-float(rerank_score)))
                    if ce_prob < self.cross_encoder_threshold:
                        best = None
                    else:
                        best = (ce_prob, doc, win_text, "cross-encoder")

            if best is not None:
                score, doc, win_text, kind = best
                flagged_paragraphs += 1
                matches.append(Match(
                    query_text=para,
                    matched_text=win_text or doc.text,
                    score=score,
                    source_title=doc.title,
                    source_url=doc.url,
                    source_id=doc.doc_id,
                    source_type=kind,
                    preview_html=highlight_pair(para, win_text or doc.text),
                ))

        if flagged_paragraphs:
            overall = sum(m.score for m in matches) / flagged_paragraphs
        else:
            overall = 0.0

        finished = datetime.utcnow().isoformat(timespec="seconds")
        return CheckResult(
            document_name=Path(file_path).name,
            total_paragraphs=len(paragraphs),
            matches=sorted(matches, key=lambda m: -m.score)[: top_k * len(paragraphs)],
            overall_score=overall,
            flagged_paragraphs=flagged_paragraphs,
            started_at=started,
            finished_at=finished,
            elapsed_seconds=time.time() - t0,
            corpus_size=len(corpus_chunks),
            citation_stats=cite_stats,
        )

    def report_html(self, result: CheckResult) -> str:
        return build_report_html(result)
