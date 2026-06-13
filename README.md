# 🔍 PlagCheck

> **$0-budget plagiarism detection for Indonesian academic papers** — built to be on-par with Turnitin (F1 0.85+) at zero infrastructure cost.

[![Live Demo](https://img.shields.io/badge/Live-plagcheck.tempmeil.xyz-10B981?style=for-the-badge&logo=globe)](https://plagcheck.tempmeil.xyz)
[![Version](https://img.shields.io/badge/version-0.5.1-8B5CF6?style=for-the-badge)](https://github.com/Poeroro/plagcheck/releases)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Corpus](https://img.shields.io/badge/corpus-7.2K_docs-F59E0B?style=for-the-badge)](https://github.com/Poeroro/plagcheck)
[![F1 Score](https://img.shields.io/badge/F1-0.85+_(real_world)-EF4444?style=for-the-badge)](reports/f1_measurement_latest.json)

---

## ✨ What it does

PlagCheck detects plagiarism in **Indonesian** academic documents (skripsi, thesis, paper) by comparing against a 7,000+ document corpus built from:

- 📰 **Indonesian news** (Detik, Kompas, CNN-ID, BBC, Liputan6) — real ID language
- 🎓 **Garuda kemdiktisaintek** — Indonesian academic paper database
- 📚 **Crossref, OpenAlex, arXiv** — international academic literature
- 🇮🇩 **Common Crawl Indonesia** — long-tail ID text

Unlike Turnitin ($30/student/year), PlagCheck is **free, self-hosted, and open source**.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 🚀 **Fast mode** | MinHash LSH + sliding-window Jaccard, 0.1s/doc |
| 🧠 **Semantic mode** | ONNX Hybrid encoder (MiniLM INT8) — catches paraphrasing |
| 🌐 **Cross-language** | 100+ languages via multilingual models |
| 📄 **Multiple formats** | PDF, DOCX, TXT, Markdown |
| 🤖 **AI detection** | Stylometry + perplexity (RoBERTa, GPT-2) |
| 🇮🇩 **Indonesian-aware** | Sastrawi stemming, ID-specific preprocessing |
| 📊 **Reports** | HTML report with highlighted match preview |
| 📚 **Citation stripping** | Removes inline citations, quotes, bibliography before matching |
| 💾 **Incremental cache** | Embedding cache only re-encodes new files, not full corpus |
| 🔒 **Local-only** | No data sent to external APIs |

---

## 🏗️ Architecture

```
┌──────────────┐
│  Web UI      │  FastAPI + Jinja2 templates
│  (FasstAPI)  │  port 8200 → nginx → https://plagcheck.tempmeil.xyz
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│  PlagEngine                          │
│  ┌────────────┐  ┌──────────────┐   │
│  │ MinHash LSH│  │ ONNX Hybrid  │   │  ← cached embeddings
│  │ (primary)  │  │ (semantic)   │   │
│  └────────────┘  └──────────────┘   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  Corpus (7,237 docs)     │
│  • Garuda ID: 3,510      │  ← academic papers
│  • News ID: 275          │
│  • Crossref/EN: 2,000+   │
│  • OpenAlex: 1,200+      │
│  • Other: 252            │
└──────────────────────────┘
```

**Key components:**
- `core/engine.py` — PlagEngine (MinHash + semantic + ensemble)
- `core/corpus.py` — Corpus loading + chunking
- `core/parser.py` — PDF/DOCX/TXT parsing + paragraph splitting
- `core/ai_detector_v2.py` — AI-generated text detection
- `core/citations.py` — Citation/quote stripping
- `prewarm_cache.py` — First-time cache build
- `scripts/crawl_garuda.py` — Garuda academic paper crawler
- `scripts/f1_measurement.py` — F1 score benchmarking

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- 4+ GB RAM
- 5+ GB disk (for models + corpus)

### Install
```bash
git clone https://github.com/Poeroro/plagcheck.git
cd plagcheck
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Export ONNX model (first time)
```bash
venv/bin/python scripts/export_onnx.py
# Output: models/onnx/minilm-hybrid/ (~250MB)
```

### Build embedding cache
```bash
venv/bin/python prewarm_cache.py
# ~30-45 min for 10K corpus, output: corpus/.embeddings_cache/emb_*.npz
```

### Run server
```bash
venv/bin/python web.py
# → http://localhost:8200
```

### Production (systemd + nginx)
See `deploy/` for systemd unit + nginx config.

---

## 📊 Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **F1 score** | **0.85-0.90** | Real-world Garuda test (20 papers × 3 variations) |
| **Precision** | 0.85+ | Low false positive rate |
| **Recall** | 0.80-0.85 | Catches exact + paraphrased |
| **Speed (MinHash)** | 0.1s/doc | Default fast mode |
| **Speed (semantic)** | 18-20s/doc | Cache hit, full corpus scan |
| **RAM** | 1.5 GB | ONNX Hybrid INT8 |
| **Disk** | 800 MB | Models + 12MB cache |

Latest F1 measurement: [`reports/f1_measurement_latest.json`](reports/f1_measurement_latest.json)

---

## 🧪 Test It

```bash
# 4-test quick validation
venv/bin/python -c "
from core.engine import PlagEngine
from core.corpus import Corpus
eng = PlagEngine(Corpus('corpus'))
eng.enable_semantic()
result = eng.check('samples/sample_thesis.txt', use_semantic=True)
print(f'Score: {result.overall_score:.3f}, Matches: {len(result.matches)}')
"
```

Sample test files in `samples/`:
- `sample_thesis.txt` — Indonesian academic writing
- `exact_copy.txt` — Direct plagiarism test
- `student_essay.txt` — Student-level writing
- `_synthetic_*.txt` — Synthetic edge cases

---

## 📂 Project Structure

```
plagcheck/
├── core/                       # Core engine modules
│   ├── engine.py               # PlagEngine (MinHash + semantic)
│   ├── corpus.py               # Corpus loading
│   ├── parser.py               # PDF/DOCX/TXT
│   ├── fingerprint.py          # MinHash + LSH
│   ├── ai_detector_v2.py       # AI text detection
│   ├── citations.py            # Citation stripping
│   ├── stopwords_id.py         # Indonesian stopwords
│   ├── expand.py               # Synonym expansion
│   └── report.py               # HTML report builder
├── corpus/                     # Document corpus (3,737 → 7,237)
│   ├── arxiv_*.txt
│   ├── garuda_*.txt            # Garuda kemdiktisaintek papers
│   ├── news_detik_*.txt
│   ├── news_kompas_*.txt
│   └── .embeddings_cache/      # Cached embeddings
├── scripts/
│   ├── crawl_garuda.py         # Garuda crawler
│   ├── crawl_id_news.py        # News crawler
│   ├── f1_measurement.py       # F1 benchmarking
│   └── export_onnx.py          # ONNX model export
├── templates/                  # Jinja2 templates
│   ├── landing.html
│   ├── results.html
│   └── riwayat.html
├── static/                     # CSS, JS
│   ├── app.css
│   └── app.js
├── samples/                    # Test samples
├── reports/                    # Generated reports
├── models/onnx/                # ONNX models (gitignored)
├── prewarm_cache.py            # First-time cache build
├── web.py                      # FastAPI app
└── requirements.txt
```

---

## 🔬 Roadmap

### v0.5.x (current)
- [x] Garuda academic corpus (3,510 papers)
- [x] Paragraph-preserving Indonesian preprocessing
- [x] Incremental embedding cache
- [x] Mobile-responsive UI
- [x] F1 measurement pipeline

### v0.6.x (next)
- [ ] SINTA author profile integration
- [ ] PDDIKTI mahasiswa data
- [ ] Grammar check (Bahasa Indonesia)
- [ ] Citation generator (citeproc-py)
- [ ] FastAPI Swagger / OpenAPI docs

### v1.0 (target)
- [ ] Tier 3: 50,000+ document corpus
- [ ] AI detection 90%+ accuracy
- [ ] Multi-user auth + API quotas
- [ ] Bilingual landing (ID + EN)
- [ ] Production SaaS deployment guide

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **ML:** PyTorch, ONNX Runtime, sentence-transformers
- **NLP:** Sastrawi (Indonesian stemmer), NLTK
- **Parser:** PyMuPDF (PDF), python-docx
- **Frontend:** Vanilla JS, custom CSS (no framework)
- **Infra:** nginx, systemd, Let's Encrypt
- **Corpus:** Garuda kemdiktisaintek, Common Crawl, OpenAlex, Crossref

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Garuda kemdiktisaintek** — Indonesian academic paper database
- **HuggingFace** — sentence-transformers, ONNX models
- **Sastrawi** — Indonesian NLP library
- **Common Crawl** — open web crawl data

---

## 📞 Contact

- Live: https://plagcheck.tempmeil.xyz
- Issues: https://github.com/Poeroro/plagcheck/issues

**Built with ❤️ for Indonesian academia, on $0 budget.**
