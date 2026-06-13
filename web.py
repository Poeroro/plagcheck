"""PlagCheck FastAPI web server — production version."""
from __future__ import annotations

import json
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from core import Corpus, PlagEngine
from core.engine import ONNX_HYBRID_PATH

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
REPORT_DIR = ROOT / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

APP_VERSION = "0.6.0"
MAX_UPLOAD_MB = 10

# Per-session privacy: each browser gets its own cookie-stamped session ID
# Reports and uploads are namespaced by this prefix so users can only see
# their own documents. No login required — first visit creates a session.
SESSION_COOKIE = "pc_sid"
SESSION_MAX_AGE = 30 * 24 * 3600  # 30 days
SESSION_PREFIX_LEN = 8  # first 8 alnum chars used in filename prefix (48 bits)

app = FastAPI(title="PlagCheck", version=APP_VERSION)
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


# -----------------------------------------------------------------------
# Session middleware: per-browser cookie-based privacy
# -----------------------------------------------------------------------
class SessionMiddleware(BaseHTTPMiddleware):
    """Attach a session ID to every request via HttpOnly Secure cookie.

    On first visit, a fresh 32-byte URL-safe token is generated and set
    as the pc_sid cookie. The token is also stored on request.state.sid
    so endpoints can read it without re-parsing cookies.
    """

    async def dispatch(self, request: Request, call_next):
        sid = request.cookies.get(SESSION_COOKIE)
        if not _is_valid_sid(sid):
            sid = secrets.token_urlsafe(32)
        request.state.sid = sid
        response = await call_next(request)
        # Always set/refresh the cookie so the session stays alive
        # (response.set_cookie is a no-op if the value matches; the
        # browser updates the expiry either way).
        response.set_cookie(
            key=SESSION_COOKIE,
            value=sid,
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=True,
            path="/",
        )
        return response


# Add session middleware (must come after route definitions to ensure
# it wraps everything, but Starlette wraps in reverse so add before mount)
app.add_middleware(SessionMiddleware)

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
    return path.stem  # <timestamp>_<session_prefix>_<safe_filename>


def _is_valid_sid(sid: str) -> bool:
    """Validate a session ID: at least 16 chars, alnum + - _ only."""
    if not isinstance(sid, str) or len(sid) < 16 or len(sid) > 128:
        return False
    return all(c.isalnum() or c in "-_" for c in sid)


def session_prefix(sid: str) -> str:
    """Filesystem-safe prefix derived from session ID for filename scoping."""
    return "".join(c for c in sid[:SESSION_PREFIX_LEN] if c.isalnum())[:SESSION_PREFIX_LEN]


def get_session_id(request: Request) -> str:
    """Read the session ID set by SessionMiddleware on request.state.

    Endpoints that need the session ID should declare this as a
    dependency. If the middleware did not run (e.g. on a synthetic
    test request), a fresh ID is minted on the fly — but in normal
    production flow it is always populated.
    """
    sid = getattr(request.state, "sid", None)
    if not _is_valid_sid(sid):
        sid = secrets.token_urlsafe(32)
        request.state.sid = sid
    return sid


def list_recent_reports(limit: int = 20, session_id: str = "") -> list[dict]:
    """List reports belonging to the given session only.

    Filenames are scoped as <timestamp>_<session_prefix>_<safe_name>.json,
    so we glob for files starting with the session prefix to enforce
    cross-session isolation at the filesystem level.
    """
    reports = []
    prefix = session_prefix(session_id) if session_id else ""
    if not prefix:
        # No valid session — return nothing rather than leak everything
        return reports
    # Filename format: <timestamp>_<session_prefix>_<name>.json
    # The session prefix is the 2nd underscore-delimited token, so we
    # glob for *_<prefix>_*.json and verify on the loaded result.
    pattern = f"*_{prefix}_*.json"
    for path in sorted(REPORT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
        # Defense in depth: confirm the second token matches
        stem = path.stem
        parts = stem.split("_", 2)
        if len(parts) < 3 or parts[1] != prefix:
            continue
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
        # Clean up filename: strip "<timestamp>_<session_prefix>_" prefix
        display_filename = raw_filename
        if "_" in display_filename:
            parts = raw_filename.split("_", 2)
            # New format: <ts>_<sid8>_<name> — strip first two
            if len(parts) >= 3 and parts[0].isdigit() and len(parts[0]) >= 10:
                display_filename = parts[2]
            # Old format: <ts>_<name> — strip first
            elif len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) >= 10:
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


def load_report(report_id: str, session_id: str = "") -> tuple[dict, Path]:
    """Return parsed report data + path. Raises 404 if not found or
    not owned by the given session (privacy enforcement)."""
    # Defense in depth: reject obvious cross-session access
    prefix = session_prefix(session_id) if session_id else ""
    if not prefix:
        raise HTTPException(404, "Report not found")
    # Filename format: <timestamp>_<session_prefix>_<name>
    # The session prefix is the 2nd underscore-delimited token.
    parts = report_id.split("_", 2)
    if len(parts) < 3 or parts[1] != prefix:
        raise HTTPException(404, "Report not found")
    matches = list(REPORT_DIR.glob(f"{report_id}.*"))
    if not matches:
        raise HTTPException(404, "Report not found")
    json_path = next((m for m in matches if m.suffix == ".json"), None)
    if not json_path:
        raise HTTPException(404, "Report JSON not found")
    # Final filesystem-level check: filename must contain the session prefix
    if prefix not in json_path.name:
        raise HTTPException(404, "Report not found")
    # And the second token must match exactly
    json_parts = json_path.stem.split("_", 2)
    if len(json_parts) < 3 or json_parts[1] != prefix:
        raise HTTPException(404, "Report not found")
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
async def landing(
    request: Request,
    sid: str = Depends(get_session_id),
) -> HTMLResponse:
    eng = get_engine()
    recent = list_recent_reports(limit=5, session_id=sid)
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
async def riwayat(
    request: Request,
    sid: str = Depends(get_session_id),
) -> HTMLResponse:
    eng = get_engine()
    recent = list_recent_reports(limit=50, session_id=sid)
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
async def report_page(
    request: Request,
    report_id: str,
    sid: str = Depends(get_session_id),
) -> HTMLResponse:
    data, _ = load_report(report_id, session_id=sid)
    data = enrich_report(data)
    total = len(list(REPORT_DIR.glob(f"*_{session_prefix(sid)}_*.json")))
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
async def report_json(
    report_id: str,
    sid: str = Depends(get_session_id),
) -> JSONResponse:
    data, _ = load_report(report_id, session_id=sid)
    return JSONResponse(data)


@app.get("/r/{report_id}/html")
async def report_html(
    request: Request,
    report_id: str,
    sid: str = Depends(get_session_id),
) -> HTMLResponse:
    data, json_path = load_report(report_id, session_id=sid)
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
        "version": APP_VERSION,
        "corpus_size": len(eng.corpus),
        "semantic_loaded": _semantic_loaded,
        "ensemble_loaded": _ensemble_loaded,
        "cross_encoder_loaded": _cross_encoder_loaded,
        "ai_detector_loaded": _ai_detector_loaded,
        "primary_encoder": (
            "ONNX Hybrid INT8" if _semantic_loaded and not _ensemble_loaded
            else "ONNX Hybrid INT8 + PyTorch mpnet" if _ensemble_loaded
            else "PyTorch (not yet loaded)"
        ),
        "onnx_model": str(ONNX_HYBRID_PATH.relative_to(ROOT)) if ONNX_HYBRID_PATH.exists() else "missing",
    }


@app.post("/api/check")
async def check(
    request: Request,
    file: UploadFile = File(...),
    semantic: bool = Form(False),
    ensemble: bool = Form(False),
    cross_encoder: bool = Form(False),
    strip_citations: bool = Form(True),
    detect_ai: bool = Form(False),
    sid: str = Depends(get_session_id),
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

    # Scope all paths to the session prefix so other users can't see this upload
    sid_prefix = session_prefix(sid)
    stamp = int(time.time() * 1000)
    # Saved file: <ts>_<sid8>_<safename>  (also acts as the public filename)
    saved = UPLOAD_DIR / f"{stamp}_{sid_prefix}_{safe_name}"
    saved.write_bytes(contents)
    # Report ID: <ts>_<sid8>_<safename.stem>
    report_id = f"{stamp}_{sid_prefix}_{Path(safe_name).stem}"

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
        # Override document_name with our session-scoped filename so the
        # display in /riwayat can strip both the timestamp and session prefix
        result_dict["document_name"] = f"{stamp}_{sid_prefix}_{safe_name}"

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


# -----------------------------------------------------------------------
# Maintenance: periodic cleanup of old session reports
# -----------------------------------------------------------------------
def cleanup_old_reports(max_age_days: int = 7) -> dict:
    """Delete reports and uploaded files older than `max_age_days`.

    Filesystem layout: <timestamp>_<session_prefix>_<name>.[json|txt|pdf|docx].
    The timestamp is the unix-millis at upload time, so we can simply
    parse the first underscore-delimited part and compare to now.
    Returns a stats dict for the caller to log.
    """
    import time as _time
    cutoff_ms = int((_time.time() - max_age_days * 86400) * 1000)
    deleted_reports = 0
    deleted_uploads = 0
    freed_bytes = 0

    for path in REPORT_DIR.glob("*.json"):
        stem = path.stem
        first = stem.split("_", 1)[0] if "_" in stem else ""
        if first.isdigit() and int(first) < cutoff_ms:
            size = path.stat().st_size
            path.unlink()
            deleted_reports += 1
            freed_bytes += size

    for path in UPLOAD_DIR.glob("*"):
        if path.is_file():
            stem = path.stem
            first = stem.split("_", 1)[0] if "_" in stem else ""
            if first.isdigit() and int(first) < cutoff_ms:
                size = path.stat().st_size
                path.unlink()
                deleted_uploads += 1
                freed_bytes += size

    return {
        "deleted_reports": deleted_reports,
        "deleted_uploads": deleted_uploads,
        "freed_mb": round(freed_bytes / 1024 / 1024, 2),
        "max_age_days": max_age_days,
    }


@app.post("/api/admin/cleanup")
async def admin_cleanup(
    max_age_days: int = 7,
) -> dict:
    """Trigger cleanup of old session-scoped reports.

    Idempotent and safe — only touches files older than the cutoff.
    Exposed without auth (admin endpoint) for now; restrict via firewall
    or add token check before production.
    """
    return cleanup_old_reports(max_age_days=max_age_days)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
