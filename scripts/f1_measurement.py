#!/usr/bin/env python3
"""
Real-world F1 measurement: 20 Garuda papers × 3 plagiarism variations + 10 legit samples.
Outputs results to reports/f1_measurement.csv + summary.
"""
import sys, time, json, random, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/plagcheck')
from core.engine import PlagEngine
from core.corpus import Corpus
from core.parser import split_paragraphs

CORPUS = Path('/home/ubuntu/plagcheck/corpus')
RESULTS = Path('/home/ubuntu/plagcheck/reports')
RESULTS.mkdir(parents=True, exist_ok=True)


def paraphrase(text: str) -> str:
    """Light paraphrase: shuffle sentences + replace 15% of words with synonyms."""
    sentences = text.replace('; ', '.|').replace('. ', '.|').split('.|')
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 2:
        # Shuffle pairs
        for i in range(0, len(sentences) - 1, 2):
            if random.random() < 0.3:
                sentences[i], sentences[i+1] = sentences[i+1], sentences[i]
    out = '. '.join(sentences)
    # Replace some common words with synonyms
    replacements = {
        'study': 'research', 'research': 'study', 'result': 'finding',
        'result': 'outcome', 'show': 'demonstrate', 'demonstrate': 'show',
        'use': 'utilize', 'utilize': 'employ', 'method': 'approach',
        'approach': 'method', 'analysis': 'examination', 'data': 'information',
        'system': 'framework', 'framework': 'system', 'application': 'app',
        'model': 'scheme', 'process': 'procedure', 'important': 'crucial',
        'therefore': 'thus', 'however': 'nevertheless', 'also': 'additionally',
        'because': 'since', 'many': 'numerous', 'showed': 'revealed',
        'found': 'discovered', 'very': 'highly', 'large': 'substantial',
    }
    words = out.split()
    for i, w in enumerate(words):
        wl = w.lower().strip('.,;:')
        if wl in replacements and random.random() < 0.3:
            words[i] = replacements[wl]
    return ' '.join(words)


def light_modify(text: str) -> str:
    """Light modification: insert 'very' / 'also' / remove some punctuation."""
    out = text
    out = out.replace('. ', '. Also, ').replace(', ', ', in fact, ')
    out = out.replace(' the ', ' the particular ').replace(' is ', ' is indeed ')
    return out


def main():
    random.seed(42)
    print(f"[{datetime.now()}] Starting F1 measurement")
    print("=" * 60)

    # Load corpus + engine
    c = Corpus(str(CORPUS))
    eng = PlagEngine(c)
    eng.enable_semantic()  # MUST use semantic for cross-chunk detection
    print(f"Corpus: {len(c.docs)} docs")
    print(f"Encoder: {eng._bi_encoder}")

    # Pick 20 Garuda papers (long enough to have multi-paragraph content)
    garuda_files = list(CORPUS.glob('garuda_*.txt'))
    garuda_files = [f for f in garuda_files if len(f.read_text()) > 1500]
    random.shuffle(garuda_files)
    test_files = garuda_files[:20]
    print(f"Selected {len(test_files)} Garuda papers")

    # For each: 3 variations
    test_cases = []
    for f in test_files:
        text = f.read_text()
        for variant, fn in [('exact', lambda t: t),
                            ('paraphrase', paraphrase),
                            ('light_modify', light_modify)]:
            test_cases.append({
                'source_file': f.name,
                'variant': variant,
                'is_plagiarism': True,  # all derived from corpus
                'text': fn(text),
            })

    # 10 legit samples (NOT from corpus)
    # Use first 1000 chars of random Garuda files NOT in our test set, or short unique content
    for i in range(10):
        # Make up unique Indonesian content about unrelated topics
        legit_topics = [
            "Panduan lengkap berkebun tomat di pekarangan rumah dengan media polybag.",
            "Resep kue lapis legit tradisional khas Palembang dengan keju edam.",
            "Tips memilih kamera mirrorless untuk content creator pemula tahun 2025.",
            "Analisis dampak perubahan iklim terhadap sektor pertanian di Indonesia.",
            "Strategi digital marketing UMKM di marketplace Tokopedia dan Shopee.",
            "Panduan investasi reksa dana untuk pemula dengan modal satu juta rupiah.",
            "Cara merawat ikan cupang halfmoon agar warna cerah dan sirip tidak rusak.",
            "Tutorial membuat website portfolio dengan framework Next.js 14.",
            "Resep sambal terasi buatan rumah yang tahan sampai satu bulan.",
            "Panduan memilih laptop untuk programming dengan budget 15 juta rupiah.",
        ]
        test_cases.append({
            'source_file': f'legit_{i:02d}',
            'variant': 'unique',
            'is_plagiarism': False,
            'text': legit_topics[i],
        })

    print(f"Total test cases: {len(test_cases)}")

    # Run tests
    results = []
    tp = fp = fn = tn = 0
    errors = 0

    for i, tc in enumerate(test_cases, 1):
        # Save to temp file
        tmp = Path(f'/tmp/f1_test_{i:03d}.txt')
        tmp.write_text(tc['text'])
        try:
            t0 = time.time()
            r = eng.check(tmp, use_semantic=True, strip_citations=True,
                          min_paragraph_chars=80)
            elapsed = time.time() - t0
            detected = r.overall_score >= 0.30 or len(r.matches) > 0
            actual = tc['is_plagiarism']

            if actual and detected: tp += 1
            elif actual and not detected: fn += 1
            elif not actual and detected: fp += 1
            else: tn += 1

            results.append({
                'idx': i,
                'source': tc['source_file'],
                'variant': tc['variant'],
                'actual_plagiarism': actual,
                'detected': detected,
                'score': round(r.overall_score, 3),
                'matches': len(r.matches),
                'elapsed_s': round(elapsed, 1),
            })
            status = '✅' if detected == actual else '❌'
            print(f"[{i:3d}/{len(test_cases)}] {status} {tc['variant']:12s} "
                  f"actual={actual} detected={detected} score={r.overall_score:.3f} "
                  f"({elapsed:.1f}s)", flush=True)
        except Exception as e:
            errors += 1
            results.append({
                'idx': i,
                'source': tc['source_file'],
                'variant': tc['variant'],
                'actual_plagiarism': tc['is_plagiarism'],
                'detected': False,
                'score': 0.0,
                'matches': 0,
                'elapsed_s': 0.0,
                'error': str(e)[:200],
            })
            print(f"[{i:3d}/{len(test_cases)}] ⚠️  ERROR: {e}", flush=True)
        finally:
            tmp.unlink(missing_ok=True)

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0

    print()
    print("=" * 60)
    print("📊 F1 MEASUREMENT RESULTS")
    print("=" * 60)
    print(f"Total cases:     {len(test_cases)}")
    print(f"Plagiarism:      {tp + fn} (should detect)")
    print(f"Legit:           {fp + tn} (should NOT detect)")
    print(f"Errors:          {errors}")
    print()
    print(f"✅ True Positives:  {tp}")
    print(f"❌ False Negatives: {fn}  (missed plagiarism)")
    print(f"❌ False Positives: {fp}  (false alarm on legit)")
    print(f"✅ True Negatives:  {tn}")
    print()
    print(f"Precision:       {precision:.3f}")
    print(f"Recall:          {recall:.3f}")
    print(f"F1 Score:        {f1:.3f}")
    print(f"Accuracy:        {accuracy:.3f}")

    # Per-variant breakdown
    print()
    print("Per-variant breakdown:")
    for variant in ['exact', 'paraphrase', 'light_modify', 'unique']:
        v_results = [r for r in results if r['variant'] == variant]
        if not v_results: continue
        v_detected = sum(1 for r in v_results if r['detected'])
        v_actual = sum(1 for r in v_results if r['actual_plagiarism'])
        print(f"  {variant:15s}: {v_detected}/{len(v_results)} detected "
              f"(actual plagiarism: {v_actual}/{len(v_results)})")

    # Save CSV + summary
    csv_path = RESULTS / f'f1_measurement_{datetime.now():%Y%m%d_%H%M%S}.csv'
    with open(csv_path, 'w') as f:
        f.write('idx,source,variant,actual,detected,score,matches,elapsed_s,error\n')
        for r in results:
            err = r.get('error', '')
            f.write(f"{r['idx']},{r['source']},{r['variant']},"
                    f"{r['actual_plagiarism']},{r['detected']},"
                    f"{r['score']},{r['matches']},{r['elapsed_s']},"
                    f"\"{err}\"\n")

    summary_path = RESULTS / 'f1_measurement_latest.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'corpus_size': len(c.docs),
        'total_cases': len(test_cases),
        'errors': errors,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'per_variant': {
            v: {
                'detected': sum(1 for r in results if r['variant'] == v and r['detected']),
                'total': sum(1 for r in results if r['variant'] == v),
            }
            for v in ['exact', 'paraphrase', 'light_modify', 'unique']
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(f"📄 CSV:    {csv_path}")
    print(f"📄 JSON:   {summary_path}")
    return summary


if __name__ == '__main__':
    main()
