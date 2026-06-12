# PlagCheck — $0 Budget Plagiarism Checker

Built with **$0 infrastructure** (just your existing VPS). Open source libs only.

## What it does
Checks PDF, DOCX, TXT, MD files for plagiarism against a local + free-API corpus.

## Two detection modes
1. **Fast (default)**: MinHash LSH + sliding-window Jaccard over word shingles.
   Catches exact copies and near-copies. ~0.1s per document.
2. **Semantic (optional)**: sentence-transformers paraphrase-multilingual model.
   Catches paraphrases and cross-language translations. ~5–12s + ~470MB first download.

## Stack (all open source, $0)
- Python 3.12 + FastAPI
- PyMuPDF (PDF parse), python-docx (DOCX), datasketch (MinHash/LSH)
- sentence-transformers (semantic) — paraphrase-multilingual-MiniLM-L12-v2
- All running on the existing Contabo VPS

## Layout
```
plagcheck/
├── core/
│   ├── parser.py        # PDF / DOCX / TXT / MD parser
│   ├── fingerprint.py   # MinHash + LSH index
│   ├── corpus.py        # local + arXiv + Semantic Scholar
│   ├── engine.py        # orchestrator
│   └── report.py        # HTML + JSON report
├── corpus/              # local corpus (text files)
├── samples/             # test documents
├── templates/           # HTML UI
├── reports/             # generated reports
├── cli.py               # CLI entry
├── web.py               # FastAPI server
└── venv/                # Python venv
```

## Usage

### Web UI
```
sudo systemctl start plagcheck   # starts on :8200
```
Open http://localhost:8200/ → drag-drop file → click Run Check.

### CLI
```bash
source venv/bin/activate

# Build corpus from arXiv + Semantic Scholar (free)
python3 cli.py build-corpus --query "machine learning" --sources arxiv,s2 --limit 15

# Check a file
python3 cli.py check samples/paper.pdf --out reports/paper.html --json reports/paper.json

# With semantic mode (paraphrase detection)
python3 cli.py check samples/paper.pdf --semantic --out reports/paper.html
```

### API
```bash
# JSON result
curl -X POST http://localhost:8200/api/check -F "file=@paper.pdf" -F "semantic=true"

# Full HTML report
curl -X POST http://localhost:8200/api/report -F "file=@paper.pdf" > report.html

# Health
curl http://localhost:8200/api/health
```

## Performance
- 9-doc corpus, MinHash check: **0.1s per file**
- 9-doc corpus, semantic check: **5-12s per file** (model load ~3s first time)
- RAM: 84MB (MinHash) / ~1.5GB (semantic model loaded)
- Disk: 5.2GB (venv + deps), ~470MB (model cache)

## Thresholds
- `DEFAULT_NEAR = 0.30` — Jaccard similarity for near-duplicate
- `DEFAULT_SEMANTIC = 0.85` — cosine similarity for paraphrase
- `LSH_THRESHOLD = 0.30` — candidate retrieval threshold
- Window size: 400 chars, step: 100 chars (sliding window refinement)

## Roadmap (not implemented yet)
- [ ] Larger corpus (millions of academic papers)
- [ ] Multi-corpus search (parallel arXiv + SINTA + Garuda + Crossref)
- [ ] Citation parser (skip quoted text automatically)
- [ ] Cross-encoder reranking for borderline semantic matches
- [ ] Batch processing (zip upload)
- [ ] Per-paragraph attribution heatmap

## Known limitations
- Semantic model can produce topical false positives (raised threshold to 0.85 helps)
- No citation detection yet (manually quoted text counted as match)
- arXiv rate limit: 1 query / 3s recommended
- Semantic Scholar free tier: 100 req / 5min
