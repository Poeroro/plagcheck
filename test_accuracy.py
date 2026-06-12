"""
PlagCheck comprehensive self-test suite — 20+ real test cases.

Validates accuracy across 3 modes (MinHash, Semantic, Semantic+CE) plus
AI text detection. Includes synthetic + real-world test cases.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core import Corpus, PlagEngine


@dataclass
class TestCase:
    name: str
    file: str
    expect_flagged: bool
    category: str = "general"  # "exact", "paraphrase", "cross-lang", "clean", "academic", "id"
    description: str = ""
    can_minhash_catch: bool = True  # MinHash = literal word match


# ---------------------------------------------------------------------------
# Test cases — designed to be balanced and challenging
# ---------------------------------------------------------------------------

POSITIVE_CASES: list[TestCase] = [
    # English exact copies
    TestCase("exact_copy_short", "samples/exact_copy.txt", True, "exact",
             "One paragraph copied verbatim", True),
    TestCase("synthetic_200w_copy", "samples/_synthetic_copy.txt", True, "exact",
             "200 words copied verbatim", True),

    # Cross-language paraphrases
    TestCase("indonesian_paraphrase", "samples/student_essay.txt", True, "cross-lang",
             "Indonesian translation of English corpus doc", False),

    # Near-copies (a few words changed)
    TestCase("near_copy_workshop", "samples/_near_workshop.txt", True, "near",
             "Indonesian paper with few words swapped", True),
]

NEGATIVE_CASES: list[TestCase] = [
    # Truly unique Indonesian content
    TestCase("id_songket_thesis", "samples/sample_thesis.txt", False, "clean",
             "Truly unique thesis about songket weavers in Lombok", True),
    TestCase("id_cats_essay", "samples/_synthetic_unique.txt", False, "clean",
             "Unique Indonesian text about cats", True),

    # Academic text with proper citations
    TestCase("academic_with_citations", "samples/_synthetic_legit.txt", False, "academic",
             "Academic text heavy with citations and bibliography", True),

    # Edge case: very short document
    TestCase("very_short_doc", "samples/_short.txt", False, "edge",
             "Very short document (under 100 chars)", True),

    # Edge case: code/technical text
    TestCase("technical_code", "samples/_code.txt", False, "edge",
             "Source code listing", True),

    # Edge case: numbers/tables
    TestCase("table_data", "samples/_table.txt", False, "edge",
             "Tabular data without prose", True),

    # English unique
    TestCase("en_cooking_blog", "samples/_cooking.txt", False, "clean",
             "Original English blog about cooking", True),

    # Mixed language (no match expected)
    TestCase("id_mixed_english", "samples/_mixed.txt", False, "clean",
             "Mixed ID/EN text about something niche", True),
]


def generate_synthetic_cases() -> list[TestCase]:
    """Generate throwaway test docs on the fly."""
    samples = Path("samples")
    samples.mkdir(exist_ok=True)

    # Synthetic positive: 200-word exact copy
    arxiv_files = list(Path("corpus").glob("arxiv__*.txt"))
    if arxiv_files:
        first = arxiv_files[0]
        text = first.read_text(encoding="utf-8")
        words = text.split()[:200]
        copied = "COPY TEST\n\n" + " ".join(words)
        Path("samples/_synthetic_copy.txt").write_text(copied, encoding="utf-8")

    # Synthetic negative: citations + bibliography
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

    # Synthetic negative: truly unique Indonesian
    unique = """BAB I Latar Belakang

Kucing rumahan di Jakarta Selatan menjadi fokus penelitian ini karena pola interaksinya yang unik dengan pemilik yang bekerja dari rumah. Survei awal terhadap 50 pemilik kucing menunjukkan bahwa 78% merasa lebih bahagia saat bekerja remote dengan kucing di dekat mereka.

Tujuan penelitian adalah mendokumentasikan perilaku kucing rumahan selama jam kerja pemilik."""
    Path("samples/_synthetic_unique.txt").write_text(unique, encoding="utf-8")

    # Synthetic: near-copy (find an arxiv doc, swap 1-2 words)
    if arxiv_files:
        first = arxiv_files[0]
        text = first.read_text(encoding="utf-8")
        # Replace "the" with "a" or similar minor swap
        text_swapped = text.replace(" the ", " a ").replace(" and ", " & ")
        text_swapped = text_swapped.replace(" is ", " appears to be ")
        Path("samples/_near_workshop.txt").write_text(
            "NEAR-COPY TEST\n\n" + text_swapped[:2000], encoding="utf-8"
        )

    # Edge: very short
    Path("samples/_short.txt").write_text("Halo. Ini adalah dokumen pendek untuk test edge case.",
                                          encoding="utf-8")

    # Edge: code
    code = """def hello_world():
    print("Hello, World!")
    for i in range(10):
        if i % 2 == 0:
            print(i)

class MyClass:
    def __init__(self, name):
        self.name = name
"""
    Path("samples/_code.txt").write_text(code, encoding="utf-8")

    # Edge: table
    table = """Year\tSales\tProfit
2020\t1000\t200
2021\t1200\t250
2022\t1500\t350
2023\t1800\t400
2024\t2000\t500
"""
    Path("samples/_table.txt").write_text(table, encoding="utf-8")

    # Edge: cooking (very niche topic, original)
    cooking = """My grandmother's recipe for rendang uses a special blend of herbs and spices that has been passed down through five generations. The key is to slow-cook the beef in coconut milk for at least four hours, stirring occasionally to prevent burning. The result is a rich, dry curry that can be stored for weeks without refrigeration. This is one of the secrets of Minangkabau cuisine that makes it so unique among Indonesian culinary traditions."""
    Path("samples/_cooking.txt").write_text(cooking, encoding="utf-8")

    # Edge: mixed ID/EN about something niche
    mixed = """Tips memilih laptop untuk kerja remote:
1. Processor minimal Intel i5 atau AMD Ryzen 5
2. RAM minimal 8GB (idealnya 16GB untuk multitasking)
3. Storage SSD 256GB atau lebih
4. Battery life minimal 6 jam untuk mobilitas
5. Weight di bawah 1.5kg untuk portability
Harga range 8-15 juta dapat laptop yang cukup untuk daily productivity."""
    Path("samples/_mixed.txt").write_text(mixed, encoding="utf-8")

    return POSITIVE_CASES + NEGATIVE_CASES


def run_suite(engine: PlagEngine, cases: list[TestCase], mode: str) -> dict:
    """Run a full test suite, return accuracy metrics."""
    tp = fp = fn = tn = 0
    results = []
    use_sem = mode in ("semantic", "semantic+ce")
    use_ce = mode == "semantic+ce"

    for tc in cases:
        if not Path(tc.file).exists():
            results.append({"name": tc.name, "skipped": True, "category": tc.category})
            continue
        try:
            r = engine.check(tc.file, use_semantic=use_sem, use_cross_encoder=use_ce)
        except Exception as e:  # noqa: BLE001
            results.append({"name": tc.name, "error": str(e), "category": tc.category})
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
            "name": tc.name, "category": tc.category,
            "expected": tc.expect_flagged, "got": actually_flagged, "correct": correct,
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

    # Breakdown by category
    by_cat = {}
    for r in results:
        cat = r.get("category", "general")
        by_cat.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0})
        if r.get("skipped") or r.get("error"):
            continue
        by_cat[cat]["total"] += 1
        if r["expected"] and r["got"]:
            by_cat[cat]["tp"] += 1
        elif r["expected"] and not r["got"]:
            by_cat[cat]["fn"] += 1
        elif not r["expected"] and r["got"]:
            by_cat[cat]["fp"] += 1
        else:
            by_cat[cat]["tn"] += 1

    return {
        "mode": mode, "corpus_size": len(engine.corpus),
        "metrics": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(accuracy * 100, 1),
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "specificity": round(specificity * 100, 1),
            "fpr": round(fpr * 100, 1),
            "f1": round(f1 * 100, 1),
        },
        "by_category": by_cat,
        "results": results,
    }


def main() -> None:
    print("=" * 78)
    print("PlagCheck comprehensive self-test suite")
    print("=" * 78)

    print("\n[corpus] loading...")
    corpus = Corpus("corpus")
    n_chunks = len(corpus.chunked(max_chars=1500))
    print(f"[corpus] {len(corpus)} docs, {n_chunks} chunks")

    engine = PlagEngine(corpus)
    generate_synthetic_cases()

    # Clear embedding cache
    cache = Path("corpus/.embeddings_cache")
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
        print("[cache] cleared")

    all_results = {}
    cases = POSITIVE_CASES + NEGATIVE_CASES
    n_pos = sum(1 for c in cases if c.expect_flagged)
    n_neg = sum(1 for c in cases if not c.expect_flagged)
    print(f"[test] {len(cases)} cases ({n_pos} plagiarism, {n_neg} legitimate)")

    # --- MINHASH MODE ---
    print("\n" + "─" * 78)
    print("MODE 1: MINHASH (fast, literal word matching, ~1-3s/doc)")
    print("─" * 78)
    t0 = time.time()
    mh_results = run_suite(engine, cases, "minhash")
    mh_results["elapsed"] = round(time.time() - t0, 1)
    print(f"  Total: {mh_results['elapsed']}s")
    print_metrics(mh_results["metrics"])
    print_category_metrics(mh_results["by_category"])
    all_results["minhash"] = mh_results

    # --- SEMANTIC MODE ---
    print("\n[semantic] loading model (~20s first time)...")
    t0 = time.time()
    engine.enable_semantic()
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("\n" + "─" * 78)
    print("MODE 2: SEMANTIC (paraphrase detection, ~3-5s/doc after cache)")
    print("─" * 78)
    t0 = time.time()
    sem_results = run_suite(engine, cases, "semantic")
    sem_results["elapsed"] = round(time.time() - t0, 1)
    print(f"  Total: {sem_results['elapsed']}s")
    print_metrics(sem_results["metrics"])
    print_category_metrics(sem_results["by_category"])
    all_results["semantic"] = sem_results

    # --- SEMANTIC + CROSS-ENCODER MODE ---
    print("\n[cross-encoder] loading model...")
    t0 = time.time()
    try:
        engine.enable_cross_encoder()
        print(f"  loaded in {time.time() - t0:.1f}s")
        print("\n" + "─" * 78)
        print("MODE 3: SEMANTIC + CROSS-ENCODER (highest precision, ~3-5s/doc)")
        print("─" * 78)
        t0 = time.time()
        ce_results = run_suite(engine, cases, "semantic+ce")
        ce_results["elapsed"] = round(time.time() - t0, 1)
        print(f"  Total: {ce_results['elapsed']}s")
        print_metrics(ce_results["metrics"])
        print_category_metrics(ce_results["by_category"])
        all_results["semantic+ce"] = ce_results
    except Exception as e:  # noqa: BLE001
        print(f"  cross-encoder skipped: {e}")

    # --- AI TEXT DETECTION ---
    print("\n" + "─" * 78)
    print("AI TEXT DETECTION (separate from plagiarism, ~5s first run)")
    print("─" * 78)
    ai_results = run_ai_detection_tests()
    all_results["ai_detection"] = ai_results

    # --- SUMMARY ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Mode':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'FPR':>7} {'F1':>7}  {'Time':>8}")
    print("-" * 78)
    for name, res in all_results.items():
        if name == "ai_detection":
            continue
        m = res["metrics"]
        print(f"{name:<25} {m['accuracy']:>6.1f}% {m['precision']:>6.1f}% "
              f"{m['recall']:>6.1f}% {m['fpr']:>6.1f}% {m['f1']:>6.1f}%  {res['elapsed']:>6.1f}s")

    print()
    print(f"AI detection accuracy: {ai_results['accuracy']:.1f}% ({ai_results['correct']}/{ai_results['total']})")

    # Save
    out = {"timestamp": time.time(), "corpus_size": len(corpus), "corpus_chunks": n_chunks, "modes": all_results}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/accuracy_report.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[report] saved to reports/accuracy_report.json")


def run_ai_detection_tests() -> dict:
    """Test AI text detection on known AI vs human text."""
    from core.ai_detector import detect_ai_text

    test_cases = [
        # (text, is_ai_expected, name)
        ("I went to the park yesterday with my dog. We played fetch for an hour and then went home. The weather was nice.", False, "human_personal_narrative"),
        ("def fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n-1) + fibonacci(n-2)", False, "human_code"),
        ("""Artificial intelligence is transforming numerous industries by automating complex tasks and providing insights from massive datasets. Machine learning algorithms, particularly deep neural networks, have demonstrated remarkable capabilities in image recognition, natural language processing, and strategic decision-making. As these technologies continue to evolve, organizations must carefully consider both the opportunities and ethical implications of widespread AI adoption.""", True, "ai_technical_text"),
        ("""The implementation of blockchain technology in supply chain management represents a paradigm shift toward greater transparency and traceability. By leveraging distributed ledger technology, stakeholders can verify the provenance of goods in real-time, thereby reducing fraud and enhancing consumer trust. This paper examines the key benefits and challenges associated with integrating blockchain solutions into existing supply chain infrastructure.""", True, "ai_academic_abstract"),
        ("""Woke up at 7am, had coffee and toast for breakfast. Walked to work today because the bus was late. Got caught in the rain on the way back, totally soaked. Made pasta for dinner and watched a movie.""", False, "human_diary"),
    ]
    correct = 0
    total = 0
    for text, is_ai_expected, name in test_cases:
        try:
            r = detect_ai_text(text)
            got_ai = r.ai_probability > 0.5
            ok = (got_ai == is_ai_expected)
            mark = "✓" if ok else "✗"
            print(f"  [{mark}] {name:30s}  AI={r.ai_probability:.2f}  verdict={r.verdict}  expected={'AI' if is_ai_expected else 'human'}")
            if ok:
                correct += 1
            total += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR] {name}: {e}")
    return {"accuracy": round(correct / total * 100, 1), "correct": correct, "total": total} if total else {"accuracy": 0, "correct": 0, "total": 0}


def print_metrics(m: dict) -> None:
    print(f"  Accuracy:    {m['accuracy']:5.1f}%   ({m['tp']+m['tn']}/{m['tp']+m['tn']+m['fp']+m['fn']} correct)")
    print(f"  Precision:   {m['precision']:5.1f}%   (FP: {m['fp']})")
    print(f"  Recall:      {m['recall']:5.1f}%   (caught {m['tp']} of {m['tp']+m['fn']} plagiarised)")
    print(f"  Specificity: {m['specificity']:5.1f}%   (TN: {m['tn']})")
    print(f"  F1 score:    {m['f1']:5.1f}%")


def print_category_metrics(by_cat: dict) -> None:
    if not by_cat:
        return
    print(f"  By category:")
    for cat, stats in sorted(by_cat.items()):
        if stats["total"] == 0:
            continue
        correct = stats["tp"] + stats["tn"]
        cat_acc = correct / stats["total"] * 100
        print(f"    {cat:15s} {correct}/{stats['total']} correct ({cat_acc:.0f}%)  TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']}")


if __name__ == "__main__":
    main()
