"""Seed a known chunk into the corpus for testing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core.corpus import Corpus, CorpusDoc

c = Corpus("corpus")
c.add(CorpusDoc(
    doc_id="local:test-seed",
    title="Artificial Intelligence in Healthcare",
    text=("Artificial intelligence has revolutionized the healthcare industry "
          "by enabling faster diagnosis, personalized treatment plans, and "
          "improved patient outcomes. Recent advances in deep learning, "
          "particularly convolutional neural networks and transformer "
          "architectures, have made it possible to analyze medical images "
          "with accuracy comparable to expert radiologists. This paper "
          "surveys the major AI applications in clinical settings between "
          "2018 and 2024, including drug discovery, genomics, and robotic "
          "surgery."),
    source="local",
    url="https://example.com/ai-healthcare-2024",
))

# Persist each doc body-only to disk so the next run picks it up
for d in c.docs:
    safe = "".join(ch if ch.isalnum() else "_" for ch in d.title[:50]) or "untitled"
    p = Path("corpus") / f"{d.source}__{safe}.txt"
    if not p.exists():
        p.write_text(d.text, encoding="utf-8")
        print(f"wrote {p}")
print(f"Total corpus: {len(c)}")
