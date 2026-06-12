"""PlagCheck FastAPI web server."""
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

app = FastAPI(title="PlagCheck", version="0.3.0")

_engine: PlagEngine | None = None
_corpus: Corpus | None = None
_semantic_loaded: bool = False
_cross_encoder_loaded: bool = False
_ai_detector_loaded: bool = False


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


def ensure_ai_detector_loaded() -> None:
    global _ai_detector_loaded
    if not _ai_detector_loaded:
        from core.ai_detector import _get_detector
        _get_detector()
        _ai_detector_loaded = True


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
        "ai_detector_loaded": _ai_detector_loaded,
    }


@app.post("/api/check")
async def check(
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
    detect_ai: bool = Form(False),
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
        if detect_ai:
            ensure_ai_detector_loaded()
        result = engine.check(
            saved,
            use_semantic=semantic,
            use_cross_encoder=cross_encoder,
            strip_citations=strip_citations,
        )
        result_dict = result.to_dict()

        # Add AI detection
        if detect_ai:
            try:
                from core.ai_detector import detect_ai_text
                from core.parser import parse_any
                text = parse_any(saved)
                ai = detect_ai_text(text, per_paragraph=False)
                result_dict["ai_detection"] = {
                    "ai_probability": ai.ai_probability,
                    "human_probability": ai.human_probability,
                    "verdict": ai.verdict,
                    "confidence": ai.confidence,
                    "model": ai.model_name,
                }
            except Exception as e:  # noqa: BLE001
                result_dict["ai_detection"] = {"error": str(e)}

        result.save_json(REPORT_DIR / f"{stamp}_{safe_name}.json")
        return result_dict
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"check failed: {e}")


@app.post("/api/report", response_class=HTMLResponse)
async def report(
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
    detect_ai: bool = Form(False),
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
    if detect_ai:
        ensure_ai_detector_loaded()
    result = engine.check(
        saved,
        use_semantic=semantic,
        use_cross_encoder=cross_encoder,
        strip_citations=strip_citations,
    )

    # AI detection
    ai_html = ""
    if detect_ai:
        try:
            from core.ai_detector import detect_ai_text
            from core.parser import parse_any
            text = parse_any(saved)
            ai = detect_ai_text(text, per_paragraph=False)
            color = "hi" if ai.ai_probability > 0.8 else ("mid" if ai.ai_probability > 0.5 else "lo")
            ai_html = f"""
            <div class="cite-box">
              <div class="lbl">AI Text Detection ({ai.model_name})</div>
              <div class="cite-grid">
                <div class="cite-stat"><div class="num">{ai.ai_probability*100:.1f}%</div><div class="lbl">AI probability</div></div>
                <div class="cite-stat"><div class="num">{ai.human_probability*100:.1f}%</div><div class="lbl">Human probability</div></div>
                <div class="cite-stat"><div class="num">{ai.verdict}</div><div class="lbl">verdict</div></div>
                <div class="cite-stat"><div class="num">{ai.confidence}</div><div class="lbl">confidence</div></div>
              </div>
            </div>
            """
        except Exception as e:  # noqa: BLE001
            ai_html = f'<div class="cite-box">AI detection failed: {e}</div>'

    html = engine.report_html(result)
    # Inject AI detection block before matches section
    if ai_html:
        html = html.replace('<h2 style="font-size:16px;margin:0 0 12px">Top Matches',
                            ai_html + '<h2 style="font-size:16px;margin:0 0 12px">Top Matches')
    (REPORT_DIR / f"{stamp}_{safe_name}.html").write_text(html, encoding="utf-8")
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
