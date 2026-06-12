"""
PlagCheck self-test suite.

Defines a set of test cases with expected outcomes and measures:
  - True Positive rate (recall on plagiarism)
  - False Positive rate (1 - specificity on legitimate text)
  - Precision (matches that are actually plagiarism)
  - F1 score

Run with:
  source venv/bin/activate
  python3 test_accuracy.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core import Corpus, PlagEngine


@dataclass
class TestCase:
    name: str
    file: str
    expect_flagged: bool       # should the engine find plagiarism?
    minhash_can_catch: bool = True  # MinHash LSH literal match (no paraphrase/translate)
    description: str = ""


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Positive cases: documents that should produce matches
POSITIVE_CASES: list[TestCase] = [
    TestCase(
        "exact_copy",
        "samples/exact_copy.txt",
        True,
        minhash_can_catch=True,
        description="One paragraph copied verbatim from a corpus doc",
    ),
    TestCase(
        "indonesian_paraphrase",
        "samples/student_essay.txt",
        True,
        minhash_can_catch=False,  # translated to different language
        description="Indonesian translation of a corpus English doc (semantic only)",
    ),
    TestCase(
        "synthetic_copy",
        "samples/_synthetic_copy.txt",
        True,
        minhash_can_catch=True,
        description="200 words copied verbatim from a corpus arxiv doc",
    ),
]

# Negative cases: legitimate academic text with citations → should NOT match
NEGATIVE_CASES: list[TestCase] = [
    TestCase(
        "indonesian_thesis",
        "samples/sample_thesis.txt",
        False,
        minhash_can_catch=True,
        description="Truly unique Indonesian text about songket weavers",
    ),
    TestCase(
        "synthetic_legit_with_citations",
        "samples/_synthetic_legit.txt",
        False,
        minhash_can_catch=True,
        description="Academic text heavy with citations and bibliography",
    ),
    TestCase(
        "synthetic_unique_id",
        "samples/_synthetic_unique.txt",
        False,
        minhash_can_catch=True,
        description="Completely unique Indonesian text about cats",
    ),
]


def make_synthetic_cases() -> list[TestCase]:
    """Generate throwaway test docs on the fly."""
    samples = Path("samples")
    samples.mkdir(exist_ok=True)

    # Synthetic positive: exact copy of a known corpus paragraph
    arxiv_files = list(Path("corpus").glob("arxiv__*.txt"))
    if arxiv_files:
        first = arxiv_files[0]
        text = first.read_text(encoding="utf-8")
        words = text.split()[:200]
        copied = "COPY TEST\n\n" + " ".join(words)
        Path("samples/_synthetic_copy.txt").write_text(copied, encoding="utf-8")

    # Synthetic negative: a paragraph full of citations
    legitimate = """LEGITIMATE ACADEMIC TEXT

Penelitian sebelumnya menunjukkan berbagai pendekatan dalam machine learning (LeCun et al., 2015). Menurut Smith (2020), deep learning telah mengubah banyak industri. Seperti yang dikatakan oleh Goodfellow (2016, p. 15), "neural networks are universal function approximators". Recent studies (Brown et al., 2020; Garcia, 2019) show a 60% increase in adoption.

REFERENCES
1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
2. Smith, J. (2020). Advances in AI. Journal of AI Research.
3. Goodfellow, I. (2016). Deep Learning. MIT Press.
4. Brown, A. et al. (2020). Industry adoption. Tech Review.
5. Garcia, M. (2019). AI survey. Computing Today.
"""
    Path("samples/_synthetic_legit.txt").write_text(legitimate, encoding="utf-8")

    # Synthetic negative: completely unique Indonesian text
    unique = """BAB I\n\nLatar Belakang Masalah\n\nDi era digital sekarang ini, kucing rumahan semakin menjadi bagian penting dari kehidupan manusia urban. Banyak pemilik kucing yang bekerja dari rumah dan menghabiskan waktu berjam-jam bersama hewan peliharaan mereka. Fenomena ini menarik untuk diteliti lebih lanjut.\n\nTujuan dari penelitian ini adalah untuk memahami pola interaksi antara manusia dan kucing rumahan di daerah Jakarta Selatan. Kami juga ingin mengetahui apakah ada korelasi antara waktu kerja dari rumah dan kebahagiaan pemilik kucing.\n\nManfaat penelitian ini diharapkan dapat memberikan kontribusi bagi psikologi positif dan kesejahteraan hewan peliharaan."""
    Path("samples/_synthetic_unique.txt").write_text(unique, encoding="utf-8")

    return POSITIVE_CASES + NEGATIVE_CASES


def run_suite(engine: PlagEngine, cases: list[TestCase], mode: str) -> dict:
    """Run a full test suite, return accuracy metrics."""
    tp = fp = fn = tn = 0
    results = []

    # Map mode to engine flags
    use_sem = mode in ("semantic", "semantic+ce")
    use_ce = mode == "semantic+ce"

    for tc in cases:
        if not Path(tc.file).exists():
            results.append({"name": tc.name, "skipped": True})
            continue
        try:
            r = engine.check(tc.file, use_semantic=use_sem, use_cross_encoder=use_ce)
        except Exception as e:  # noqa: BLE001
            results.append({"name": tc.name, "error": str(e)})
            continue

        actually_flagged = r.flagged_paragraphs > 0 and r.overall_score > 0.05
        correct = (actually_flagged == tc.expect_flagged)

        if tc.expect_flagged and actually_flagged:
            tp += 1
        elif tc.expect_flagged and not actually_flagged:
            fn += 1
        elif not tc.expect_flagged and actually_flagged:
            fp += 1
        else:
            tn += 1

        results.append({
            "name": tc.name,
            "expected": tc.expect_flagged,
            "got": actually_flagged,
            "correct": correct,
            "score": round(r.overall_score * 100, 1),
            "flagged": f"{r.flagged_paragraphs}/{r.total_paragraphs}",
            "elapsed": round(r.elapsed_seconds, 2),
            "description": tc.description,
        })

    total_pos = tp + fn
    total_neg = tn + fp
    recall = tp / total_pos if total_pos else 0
    specificity = tn / total_neg if total_neg else 0
    fpr = fp / total_neg if total_neg else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "mode": mode,
        "metrics": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(accuracy * 100, 1),
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "specificity": round(specificity * 100, 1),
            "fpr": round(fpr * 100, 1),
            "f1": round(f1 * 100, 1),
        },
        "results": results,
    }


def main() -> None:
    print("=" * 70)
    print("PlagCheck self-test suite — 492-doc corpus, 6 cases")
    print("=" * 70)

    print("\n[corpus] loading...")
    corpus = Corpus("corpus")
    print(f"[corpus] {len(corpus)} docs, {len(corpus.chunked(max_chars=1500))} chunks")

    engine = PlagEngine(corpus)
    make_synthetic_cases()

    # Clear embedding cache to start fresh
    cache = Path("corpus/.embeddings_cache")
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
        print("[cache] cleared")

    all_results = {}

    # --- MINHASH MODE ---
    print("\n--- MODE 1: MINHASH (fast, no semantic, ~1s/doc) ---")
    t0 = time.time()
    mh_results = run_suite(engine, POSITIVE_CASES + NEGATIVE_CASES, "minhash")
    mh_results["elapsed"] = round(time.time() - t0, 1)
    print(f"  took {mh_results['elapsed']}s")
    print_metrics(mh_results["metrics"])
    print_results(mh_results["results"])
    all_results["minhash"] = mh_results

    # --- SEMANTIC MODE ---
    print("\n[semantic] loading model...")
    t0 = time.time()
    engine.enable_semantic()
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("\n--- MODE 2: SEMANTIC (paraphrase detection, ~3-5s/doc) ---")
    t0 = time.time()
    sem_results = run_suite(engine, POSITIVE_CASES + NEGATIVE_CASES, "semantic")
    sem_results["elapsed"] = round(time.time() - t0, 1)
    print(f"  took {sem_results['elapsed']}s")
    print_metrics(sem_results["metrics"])
    print_results(sem_results["results"])
    all_results["semantic"] = sem_results

    # --- SEMANTIC + CROSS-ENCODER MODE ---
    print("\n[cross-encoder] loading...")
    t0 = time.time()
    try:
        engine.enable_cross_encoder()
        print(f"  loaded in {time.time() - t0:.1f}s")

        print("\n--- MODE 3: SEMANTIC + CROSS-ENCODER (high precision, ~5-8s/doc) ---")
        t0 = time.time()
        ce_results = run_suite(engine, POSITIVE_CASES + NEGATIVE_CASES, "semantic+ce")
        ce_results["elapsed"] = round(time.time() - t0, 1)
        print(f"  took {ce_results['elapsed']}s")
        print_metrics(ce_results["metrics"])
        print_results(ce_results["results"])
        all_results["semantic+ce"] = ce_results
    except Exception as e:  # noqa: BLE001
        print(f"  cross-encoder skipped: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'FPR':>6} {'F1':>6}  {'Time':>8}")
    print("-" * 70)
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"{name:<25} {m['accuracy']:>5.1f}% {m['precision']:>5.1f}% "
              f"{m['recall']:>5.1f}% {m['fpr']:>5.1f}% {m['f1']:>5.1f}%  {res['elapsed']:>6.1f}s")

    # Save report
    out = {
        "timestamp": time.time(),
        "corpus_size": len(corpus),
        "corpus_chunks": len(corpus.chunked(max_chars=1500)),
        "modes": all_results,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/accuracy_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[report] saved to reports/accuracy_report.json")


def print_metrics(m: dict) -> None:
    print(f"  Accuracy:    {m['accuracy']:5.1f}%   ({m['tp']+m['tn']}/{m['tp']+m['tn']+m['fp']+m['fn']} correct)")
    print(f"  Precision:   {m['precision']:5.1f}%   (no FP: {m['fp']=})")
    print(f"  Recall:      {m['recall']:5.1f}%   (caught {m['tp']} of {m['tp']+m['fn']} plagiarised)")
    print(f"  Specificity: {m['specificity']:5.1f}%   ({m['fp']} false positives)")
    print(f"  F1 score:    {m['f1']:5.1f}%")


def print_results(results: list[dict]) -> None:
    for r in results:
        if r.get("skipped"):
            print(f"  [SKIP] {r['name']}: file missing")
            continue
        if r.get("error"):
            print(f"  [ERR ] {r['name']}: {r['error']}")
            continue
        ok = "✓" if r["correct"] else "✗"
        exp = "TP" if r["expected"] else "TN"
        got = "TP" if r["got"] else "TN"
        print(f"  [{ok}]  {r['name']:35s} {exp}→{got}  score={r['score']:5.1f}%  "
              f"flagged={r['flagged']:6s}  ({r['elapsed']}s)")


if __name__ == "__main__":
    main()
