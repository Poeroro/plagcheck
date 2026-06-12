"""PlagCheck FastAPI web server.

POST /api/check        upload file, get JSON result
POST /api/report       upload file, get HTML report
GET  /                 upload form (HTML)
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from core import Corpus, PlagEngine

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
REPORT_DIR = ROOT / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PlagCheck", version="0.2.0")

_engine: PlagEngine | None = None
_corpus: Corpus | None = None
_semantic_loaded: bool = False
_cross_encoder_loaded: bool = False


def get_engine() -> PlagEngine:
    global _engine, _corpus
    if _engine is None:
        _corpus = Corpus(ROOT / "corpus")
        _engine = PlagEngine(_corpus)
    return _engine


def ensure_semantic_loaded() -> None:
    global _semantic_loaded
    eng = get_engine()
    if not _semantic_loaded:
        eng.enable_semantic()
        _semantic_loaded = True


def ensure_cross_encoder_loaded() -> None:
    global _cross_encoder_loaded
    eng = get_engine()
    if not _cross_encoder_loaded:
        eng.enable_cross_encoder()
        _cross_encoder_loaded = True


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/api/health")
async def health() -> dict:
    eng = get_engine()
    return {
        "status": "ok",
        "corpus_size": len(eng.corpus),
        "semantic_loaded": _semantic_loaded,
        "cross_encoder_loaded": _cross_encoder_loaded,
    }


@app.post("/api/check")
async def check(
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
) -> dict:
    if not file.filename:
        raise HTTPException(400, "no file")

    safe_name = Path(file.filename).name
    stamp = int(time.time() * 1000)
    saved = UPLOAD_DIR / f"{stamp}_{safe_name}"
    with saved.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        engine = get_engine()
        if semantic:
            ensure_semantic_loaded()
        if cross_encoder:
            ensure_semantic_loaded()
            ensure_cross_encoder_loaded()
        result = engine.check(
            saved,
            use_semantic=semantic,
            use_cross_encoder=cross_encoder,
            strip_citations=strip_citations,
        )
        result.save_json(REPORT_DIR / f"{stamp}_{safe_name}.json")
        return result.to_dict()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"check failed: {e}")


@app.post("/api/report", response_class=HTMLResponse)
async def report(
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
) -> HTMLResponse:
    if not file.filename:
        raise HTTPException(400, "no file")

    safe_name = Path(file.filename).name
    stamp = int(time.time() * 1000)
    saved = UPLOAD_DIR / f"{stamp}_{safe_name}"
    with saved.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    engine = get_engine()
    if semantic:
        ensure_semantic_loaded()
    if cross_encoder:
        ensure_semantic_loaded()
        ensure_cross_encoder_loaded()
    result = engine.check(
        saved,
        use_semantic=semantic,
        use_cross_encoder=cross_encoder,
        strip_citations=strip_citations,
    )
    html = engine.report_html(result)
    (REPORT_DIR / f"{stamp}_{safe_name}.html").write_text(html, encoding="utf-8")
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
