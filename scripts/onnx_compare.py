"""Compare: PyTorch FP32 vs ONNX FP32 vs ONNX INT8 vs ONNX Hybrid."""
import os
import sys
import time
import numpy as np
import psutil
from pathlib import Path

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
sys.path.insert(0, '/home/ubuntu/plagcheck')


class PytorchEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def encode(self, texts, batch_size=16):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


class OnnxEmbedder:
    def __init__(self, model_dir):
        from transformers import AutoTokenizer
        import onnxruntime as ort
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(
            os.path.join(model_dir, 'model.onnx'),
            providers=['CPUExecutionProvider']
        )

    def encode(self, texts, batch_size=16):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='np')
            outputs = self.session.run(None, dict(inputs))
            token_embs = outputs[0]
            mask = inputs['attention_mask'][..., None].astype(np.float32)
            summed = (token_embs * mask).sum(axis=1)
            counts = mask.sum(axis=1).clip(min=1e-9)
            embs = summed / counts
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
            all_embs.append(embs)
        return np.vstack(all_embs)


from core.corpus import Corpus
corpus = Corpus('corpus', preprocess_id=True)
chunks = corpus.chunked(max_chars=2000)
chunk_texts = [c.text for c in chunks]
print(f'Corpus: {len(corpus)} docs, {len(chunks)} chunks')

tests = [
    ('exact_copy',   'samples/exact_copy.txt',       True,  0.30),
    ('paraphrase',   'samples/student_essay.txt',    True,  0.30),
    ('synth_copy',   'samples/_synthetic_copy.txt',  True,  0.30),
    ('synth_legit',  'samples/_synthetic_legit.txt', False, 0.30),
    ('synth_unique', 'samples/_synthetic_unique.txt',False, 0.30),
]
texts = []
for name, fp, _, _ in tests:
    texts.append(Path(fp).read_text(encoding='utf-8', errors='ignore')[:1500])

variants = [
    ('PyTorch FP32', PytorchEmbedder, {}),
    ('ONNX FP32',    OnnxEmbedder, {'model_dir': 'models/onnx/minilm-fp32'}),
    ('ONNX INT8',    OnnxEmbedder, {'model_dir': 'models/onnx/minilm-int8'}),
    ('ONNX Hybrid',  OnnxEmbedder, {'model_dir': 'models/onnx/minilm-hybrid'}),
]

results = []
for name, Cls, kwargs in variants:
    print(f'\n=== {name} ===', flush=True)
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    t0 = time.time()
    embedder = Cls(**kwargs)
    t_load = time.time() - t0
    rss_after_load = proc.memory_info().rss

    t0 = time.time()
    chunk_embs = embedder.encode(chunk_texts, batch_size=32)
    t_encode_corpus = time.time() - t0

    t0 = time.time()
    query_embs = embedder.encode(texts, batch_size=8)
    t_encode_queries = time.time() - t0

    sims = query_embs @ chunk_embs.T
    best_scores = sims.max(axis=1)

    rss_final = proc.memory_info().rss
    rss_delta = (rss_final - rss_before) / 1024 / 1024
    del embedder

    results.append({
        'name': name, 't_load': t_load, 't_corpus': t_encode_corpus,
        't_queries': t_encode_queries, 'rss_delta': rss_delta,
        'scores': best_scores,
    })

    for i, (tname, _, expect_flag, thresh) in enumerate(tests):
        score = best_scores[i]
        pred = score > thresh
        ok = 'OK' if pred == expect_flag else 'XX'
        print(f'  [{ok}] {tname:<14} score={score:.3f} pred={"flag" if pred else "ok"}')

print('\n=== COMPARISON ===')
print(f"{'Variant':<15} {'Load':<7} {'Corpus':<8} {'Queries':<9} {'RSS':<8} {'MeanScore':<11} {'F1'}")
for r in results:
    pred_flags = [s > 0.30 for s in r['scores']]
    exp_flags = [t[2] for t in tests]
    tp = sum(1 for p, e in zip(pred_flags, exp_flags) if p and e)
    fp = sum(1 for p, e in zip(pred_flags, exp_flags) if p and not e)
    fn = sum(1 for p, e in zip(pred_flags, exp_flags) if not p and e)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    mean_score = float(np.mean(r['scores']))
    print(f"{r['name']:<15} {r['t_load']:.1f}s   {r['t_corpus']:.1f}s    {r['t_queries']:.2f}s    {r['rss_delta']:+.0f}MB   {mean_score:.3f}      {f1*100:.0f}%")
