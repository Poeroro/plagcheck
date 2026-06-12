"""PlagCheck FastAPI web server — production version."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core import Corpus, PlagEngine

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
REPORT_DIR = ROOT / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

APP_VERSION = "0.4.0"
MAX_UPLOAD_MB = 10

app = FastAPI(title="PlagCheck", version=APP_VERSION)
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

_engine: PlagEngine | None = None
_corpus: Corpus | None = None
_semantic_loaded: bool = False
_ensemble_loaded: bool = False
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


def ensure_ensemble_loaded() -> None:
    global _ensemble_loaded
    eng = get_engine()
    if not _ensemble_loaded:
        eng.enable_ensemble()
        _ensemble_loaded = True


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


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def risk_label(score: float) -> str:
    if score > 0.30:
        return "TINGGI"
    if score > 0.10:
        return "SEDANG"
    return "RENDAH"


def risk_class(score: float) -> str:
    if score > 0.30:
        return "high"
    if score > 0.10:
        return "med"
    return "low"


def pct_class(score: float) -> str:
    if score > 0.70:
        return "high"
    if score > 0.40:
        return "med"
    return "low"


def ai_verdict_class(verdict: str) -> str:
    return {
        "AI-LIKELY": "AI-LIKELY",
        "MIXED": "MIXED",
        "HUMAN": "HUMAN",
    }.get(verdict, "MIXED")


def humanize_ago(iso: str) -> str:
    try:
        ts = datetime.fromisoformat(iso)
        delta = datetime.utcnow() - ts
        secs = int(delta.total_seconds())
        if secs < 60: return f"{secs} detik lalu"
        if secs < 3600: return f"{secs // 60} menit lalu"
        if secs < 86400: return f"{secs // 3600} jam lalu"
        return f"{secs // 86400} hari lalu"
    except Exception:
        return iso


def humanize_time(iso: str) -> str:
    try:
        ts = datetime.fromisoformat(iso)
        return ts.strftime("%d %b %Y · %H:%M")
    except Exception:
        return iso


def build_id_from_path(path: Path) -> str:
    return path.stem  # <timestamp>_<safe_filename>


def list_recent_reports(limit: int = 20) -> list[dict]:
    reports = []
    for path in sorted(REPORT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        report_id = build_id_from_path(path)
        score = data.get("overall_score", 0) * 100
        ai = data.get("ai_detection")
        ai_label = ""
        ai_class = "ai-low"
        if ai and not ai.get("error"):
            ai_pct = ai.get("ai_probability", 0) * 100
            ai_label = f"{ai_pct:.0f}%"
            ai_class = "ai-high" if ai_pct > 60 else "ai-med" if ai_pct > 30 else "ai-low"
        raw_filename = data.get("document_name", "") or "?"
        # Skip orphan/empty entries
        if not raw_filename or raw_filename == "?" or data.get("total_paragraphs", 0) == 0:
            continue
        # Clean up filename: strip "<timestamp>_" prefix
        display_filename = raw_filename
        if "_" in display_filename:
            parts = display_filename.split("_", 1)
            if parts[0].isdigit() and len(parts[0]) >= 10:
                display_filename = parts[1]
        filetype = (raw_filename.split(".")[-1] or "txt").lower()
        if filetype not in ("pdf", "docx", "txt"):
            filetype = "txt"
        reports.append({
            "id": report_id,
            "filename": display_filename,
            "raw_filename": raw_filename,
            "score": f"{score:.1f}",
            "score_class": risk_class(data.get("overall_score", 0)),
            "elapsed": f"{data.get('elapsed_seconds', 0):.1f}",
            "paragraphs": data.get("total_paragraphs", 0),
            "ai_label": ai_label,
            "ai_class": ai_class,
            "filetype": filetype,
            "finished_at_human": humanize_time(data.get("finished_at", "")),
            "ago": humanize_ago(data.get("finished_at", "")),
        })
    return reports[:limit]


def load_report(report_id: str) -> tuple[dict, Path]:
    """Return parsed report data + path. Raises 404 if not found."""
    matches = list(REPORT_DIR.glob(f"{report_id}.*"))
    if not matches:
        raise HTTPException(404, "Report not found")
    json_path = next((m for m in matches if m.suffix == ".json"), None)
    if not json_path:
        raise HTTPException(404, "Report JSON not found")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read report: {e}")
    return data, json_path


def enrich_report(data: dict) -> dict:
    """Add derived UI properties to report data."""
    score = data.get("overall_score", 0)
    data["risk_label"] = risk_label(score)
    data["risk_class"] = risk_class(score)
    for m in data.get("matches", []):
        m["pct_class"] = pct_class(m.get("score", 0))
    ai = data.get("ai_detection")
    if ai and not ai.get("error"):
        ai["verdict_class"] = ai_verdict_class(ai.get("verdict", "MIXED"))
    # Clean up filename for display
    raw = data.get("document_name", "")
    if "_" in raw:
        parts = raw.split("_", 1)
        if parts[0].isdigit() and len(parts[0]) >= 10:
            data["display_name"] = parts[1]
        else:
            data["display_name"] = raw
    else:
        data["display_name"] = raw
    # Compute citation reduction if not present
    cs = data.get("citation_stats") or {}
    if cs and "reduction_pct" not in cs:
        orig = cs.get("original_chars", 0)
        clean = cs.get("cleaned_chars", 0)
        if orig > 0:
            cs["reduction_pct"] = (1 - clean / orig) * 100
        else:
            cs["reduction_pct"] = 0
        data["citation_stats"] = cs
    return data


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    eng = get_engine()
    recent = list_recent_reports(limit=5)
    return templates.TemplateResponse(
        request,
        "landing.html",
        {
            "version": APP_VERSION,
            "corpus_size": f"{len(eng.corpus):,}",
            "recent": recent,
        },
    )


@app.get("/riwayat", response_class=HTMLResponse)
async def riwayat(request: Request) -> HTMLResponse:
    eng = get_engine()
    recent = list_recent_reports(limit=50)
    return templates.TemplateResponse(
        request,
        "riwayat.html",
        {
            "version": APP_VERSION,
            "corpus_size": f"{len(eng.corpus):,}",
            "recent": recent,
            "total_reports": len(recent),
        },
    )


@app.get("/r/{report_id}", response_class=HTMLResponse)
async def report_page(request: Request, report_id: str) -> HTMLResponse:
    data, _ = load_report(report_id)
    data = enrich_report(data)
    total = len(list(REPORT_DIR.glob("*.json")))
    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "version": APP_VERSION,
            "r": data,
            "total_reports": total,
        },
    )


@app.get("/r/{report_id}/json")
async def report_json(report_id: str) -> JSONResponse:
    data, _ = load_report(report_id)
    return JSONResponse(data)


@app.get("/r/{report_id}/html")
async def report_html(report_id: str) -> HTMLResponse:
    data, json_path = load_report(report_id)
    data = enrich_report(data)
    eng = get_engine()
    # Reconstruct a CheckResult for the legacy report builder
    from core.report import CheckResult, CitationStats, Match
    result = CheckResult(
        document_name=data.get("document_name", ""),
        total_paragraphs=data.get("total_paragraphs", 0),
        matches=[Match(**{k: m.get(k) for k in
            ["query_text","matched_text","score","source_title","source_url","source_id","source_type","preview_html"]})
            for m in data.get("matches", [])],
        overall_score=data.get("overall_score", 0),
        flagged_paragraphs=data.get("flagged_paragraphs", 0),
        started_at=data.get("started_at", ""),
        finished_at=data.get("finished_at", ""),
        elapsed_seconds=data.get("elapsed_seconds", 0),
        corpus_size=data.get("corpus_size", 0),
        citation_stats=CitationStats(**{k: v for k, v in (data.get("citation_stats") or {}).items()
            if k in CitationStats.__dataclass_fields__}),
    )
    html = eng.report_html(result)
    return HTMLResponse(
        html,
        headers={"Content-Disposition": f'attachment; filename="{report_id}.html"'},
    )


@app.get("/api/health")
async def health() -> dict:
    eng = get_engine()
    return {
        "status": "ok",
        "corpus_size": len(eng.corpus),
        "semantic_loaded": _semantic_loaded,
        "ensemble_loaded": _ensemble_loaded,
        "cross_encoder_loaded": _cross_encoder_loaded,
        "ai_detector_loaded": _ai_detector_loaded,
    }


@app.post("/api/check")
async def check(
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    ensemble: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
    detect_ai: bool = Form(False),
) -> dict:
    if not file.filename:
        raise HTTPException(400, "no file")

    # Validate file
    safe_name = Path(file.filename).name
    ext = safe_name.split(".")[-1].lower() if "." in safe_name else ""
    if ext not in ("pdf", "docx", "txt"):
        raise HTTPException(400, f"unsupported file type: .{ext}")

    # Read & check size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"file too large (max {MAX_UPLOAD_MB}MB)")

    stamp = int(time.time() * 1000)
    saved = UPLOAD_DIR / f"{stamp}_{safe_name}"
    saved.write_bytes(contents)
    report_id = f"{stamp}_{Path(safe_name).stem}"

    try:
        engine = get_engine()
        if ensemble:
            ensure_ensemble_loaded()
        elif semantic:
            ensure_semantic_loaded()
        if cross_encoder:
            if not ensemble:
                ensure_semantic_loaded()
            ensure_cross_encoder_loaded()
        if detect_ai:
            ensure_ai_detector_loaded()

        result = engine.check(
            saved,
            use_semantic=semantic or ensemble,
            use_cross_encoder=cross_encoder,
            strip_citations=strip_citations,
            use_ensemble=ensemble,
        )

        result_dict = result.to_dict()

        if detect_ai:
            try:
                from core.ai_detector_v2 import detect_ai_enhanced
                from core.parser import parse_any
                text = parse_any(saved)
                ai = detect_ai_enhanced(text, use_perplexity=True, use_stylometry=True)
                result_dict["ai_detection"] = {
                    "ai_probability": ai.ai_probability,
                    "human_probability": ai.human_probability,
                    "verdict": ai.verdict,
                    "confidence": ai.confidence,
                    "model_name": ai.model_name,
                    "signals": ai.signals,
                    "per_paragraph": ai.per_paragraph,
                }
            except Exception as e:  # noqa: BLE001
                result_dict["ai_detection"] = {"error": str(e)}

        # Save JSON report
        report_path = REPORT_DIR / f"{report_id}.json"
        report_path.write_text(
            json.dumps(result_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {"id": report_id, **result_dict}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"check failed: {e}")


@app.post("/api/report", response_class=HTMLResponse)
async def report_legacy(
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    ensemble: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
    detect_ai: bool = Form(False),
) -> HTMLResponse:
    """Legacy endpoint: returns plain HTML report (for backward compat)."""
    payload = await check(file, semantic, ensemble, cross_encoder, strip_citations, detect_ai)
    eng = get_engine()
    from core.report import CheckResult, CitationStats, Match
    result = CheckResult(
        document_name=payload.get("document_name", ""),
        total_paragraphs=payload.get("total_paragraphs", 0),
        matches=[Match(**{k: m.get(k) for k in
            ["query_text","matched_text","score","source_title","source_url","source_id","source_type","preview_html"]})
            for m in payload.get("matches", [])],
        overall_score=payload.get("overall_score", 0),
        flagged_paragraphs=payload.get("flagged_paragraphs", 0),
        started_at=payload.get("started_at", ""),
        finished_at=payload.get("finished_at", ""),
        elapsed_seconds=payload.get("elapsed_seconds", 0),
        corpus_size=payload.get("corpus_size", 0),
        citation_stats=CitationStats(**{k: v for k, v in (payload.get("citation_stats") or {}).items()
            if k in CitationStats.__dataclass_fields__}),
    )
    return HTMLResponse(eng.report_html(result))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
