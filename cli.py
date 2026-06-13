#!/usr/bin/env python3
"""DoubleCheck CLI — plagiarism checker.

Usage:
  python3 cli.py build-corpus --query "machine learning" --sources arxiv,s2 --out corpus.txt
  python3 cli.py check file.pdf --out report.html --json report.json
  python3 cli.py web   # start FastAPI server
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure local package import works
sys.path.insert(0, str(Path(__file__).parent))

from core import Corpus, PlagEngine


def cmd_build_corpus(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(out_dir)

    if "arxiv" in args.sources:
        docs = corpus.search_arxiv(args.query, max_results=args.limit)
        print(f"[arxiv] fetched {len(docs)} results")
        corpus.extend(docs)

    if "s2" in args.sources or "semanticscholar" in args.sources:
        docs = corpus.search_semanticscholar(args.query, max_results=args.limit)
        print(f"[s2] fetched {len(docs)} results")
        corpus.extend(docs)

    # Persist each doc body-only to disk so future runs skip the network
    for d in c.docs:
        safe = "".join(c if c.isalnum() else "_" for c in d.title[:60]) or d.doc_id.replace(":", "_")
        path = out_dir / f"{d.source}__{safe}.txt"
        if not path.exists():
            path.write_text(d.text, encoding="utf-8")

    print(f"[corpus] {len(corpus)} documents now in {out_dir}")


def cmd_check(args) -> None:
    file_path = Path(args.file)
    if not file_path.exists():
        sys.exit(f"File not found: {file_path}")

    corpus = Corpus(args.corpus)
    print(f"[corpus] loaded {len(corpus)} local documents")
    engine = PlagEngine(corpus)

    if args.semantic:
        engine.enable_semantic()

    result = engine.check(file_path)

    print()
    print(f"  Document        : {result.document_name}")
    print(f"  Paragraphs      : {result.total_paragraphs}")
    print(f"  Flagged         : {result.flagged_paragraphs}")
    print(f"  Overall score   : {result.overall_score * 100:.1f}%")
    print(f"  Corpus size     : {result.corpus_size} chunks")
    print(f"  Elapsed         : {result.elapsed_seconds:.2f}s")
    print()

    if args.json:
        result.save_json(args.json)
        print(f"[json] saved → {args.json}")
    if args.out:
        html = engine.report_html(result)
        Path(args.out).write_text(html, encoding="utf-8")
        print(f"[html] saved → {args.out}")


def cmd_web(args) -> None:
    import uvicorn
    uvicorn.run("web:app", host=args.host, port=args.port, reload=False)


def main() -> None:
    p = argparse.ArgumentParser(prog="plagcheck", description="$0 plagiarism checker")
    sub = p.add_subparsers(dest="cmd", required=True)

    bc = sub.add_parser("build-corpus", help="Fetch free external corpus")
    bc.add_argument("--query", required=True)
    bc.add_argument("--sources", default="arxiv,s2", help="comma list: arxiv,s2")
    bc.add_argument("--limit", type=int, default=15)
    bc.add_argument("--out", default="corpus")
    bc.set_defaults(func=cmd_build_corpus)

    ck = sub.add_parser("check", help="Check a document")
    ck.add_argument("file")
    ck.add_argument("--corpus", default="corpus")
    ck.add_argument("--out", help="output HTML report path")
    ck.add_argument("--json", help="output JSON report path")
    ck.add_argument("--semantic", action="store_true", help="enable semantic model (slow first run)")
    ck.set_defaults(func=cmd_check)

    web = sub.add_parser("web", help="Start web UI")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8200)
    web.set_defaults(func=cmd_web)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
