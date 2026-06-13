"""
Report generation: turn a MatchResult into human-readable HTML + JSON.
"""
from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Match:
    """One matched chunk with its score and source."""
    query_text: str
    matched_text: str
    score: float
    source_title: str
    source_url: str
    source_id: str
    source_type: str         # "exact" | "near" | "semantic" | "cross-encoder"
    preview_html: str = ""


@dataclass
class CitationStats:
    """Stats from the citation-stripping pass."""
    inline_citations_stripped: int = 0
    direct_quotes_stripped: int = 0
    block_quotes_stripped: int = 0
    footnote_refs_stripped: int = 0
    bibliography_chars_stripped: int = 0
    urls_stripped: int = 0
    original_chars: int = 0
    cleaned_chars: int = 0

    @property
    def reduction_pct(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return (1 - self.cleaned_chars / self.original_chars) * 100


@dataclass
class CheckResult:
    document_name: str
    total_paragraphs: int
    matches: list[Match]
    overall_score: float
    flagged_paragraphs: int
    started_at: str
    finished_at: str
    elapsed_seconds: float
    corpus_size: int
    citation_stats: CitationStats = field(default_factory=CitationStats)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Inline diff highlighter
# ---------------------------------------------------------------------------
def highlight_pair(query: str, matched: str, max_len: int = 800) -> str:
    """Return two HTML strings side-by-side with matching words highlighted."""
    q_words = query[:max_len].split()
    m_words = matched[:max_len].split()
    q_set = {w.lower().strip(".,;:!?\"'()[]") for w in m_words}
    m_set = {w.lower().strip(".,;:!?\"'()[]") for w in q_words}

    def _render(words: list[str], other: set[str]) -> str:
        out = []
        for w in words:
            key = w.lower().strip(".,;:!?\"'()[]")
            if key in other and len(key) > 2:
                out.append(f'<mark>{html.escape(w)}</mark>')
            else:
                out.append(html.escape(w))
        return " ".join(out)

    return (
        '<div class="pair">'
        f'<div class="col"><div class="lbl">DOKUMEN LO</div><div class="txt">{_render(q_words, q_set)}</div></div>'
        f'<div class="col"><div class="lbl">SUMBER</div><div class="txt">{_render(m_words, m_set)}</div></div>'
        '</div>'
    )


def build_report_html(result: CheckResult) -> str:
    """Generate a self-contained HTML report (no external assets)."""
    score_pct = round(result.overall_score * 100, 1)
    flagged = result.flagged_paragraphs
    total = result.total_paragraphs
    cs = result.citation_stats

    css = """
    <style>
      :root { color-scheme: dark; }
      body { font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
             background: #0e1116; color: #d6dde6; margin: 0; padding: 24px; }
      .wrap { max-width: 1100px; margin: 0 auto; }
      h1 { margin: 0 0 4px; font-size: 22px; }
      .sub { color: #8a96a7; margin-bottom: 24px; }
      .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0 28px; }
      .card { background: #161b22; border: 1px solid #232a35; border-radius: 10px; padding: 14px; }
      .card .k { color: #8a96a7; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
      .card .v { font-size: 22px; font-weight: 600; margin-top: 4px; }
      .score { font-size: 38px; font-weight: 700; }
      .score.low  { color: #3fb950; } .score.mid { color: #d29922; } .score.hi { color: #f85149; }
      .bar { height: 6px; background: #232a35; border-radius: 3px; overflow: hidden; margin-top: 8px; }
      .bar > div { height: 100%; background: linear-gradient(90deg, #3fb950 0%, #d29922 50%, #f85149 100%); }
      .match { background: #161b22; border: 1px solid #232a35; border-radius: 10px; padding: 16px; margin: 12px 0; }
      .meta { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
      .pill { background: #21262d; color: #c9d1d9; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
      .pill.hi  { background: #67060c; color: #ffdcd7; }
      .pill.mid { background: #4d3800; color: #f8e3a1; }
      .pill.lo  { background: #0f5132; color: #aff5b4; }
      .src a { color: #58a6ff; text-decoration: none; }
      .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
      .col { background: #0b0e13; border: 1px solid #1f2630; border-radius: 8px; padding: 12px; }
      .lbl { color: #8a96a7; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
      .txt { font-size: 13px; line-height: 1.55; }
      mark { background: #f8514966; color: #ffdcd7; padding: 0 2px; border-radius: 3px; }
      .empty { color: #8a96a7; text-align: center; padding: 36px; }
      .cite-box { background: #0b0e13; border: 1px solid #1f2630; border-radius: 8px; padding: 12px; margin: 12px 0; font-size: 13px; }
      .cite-box .lbl { color: #8a96a7; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
      .cite-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
      .cite-stat .num { font-size: 20px; font-weight: 600; }
      footer { color: #6e7681; font-size: 12px; margin-top: 24px; text-align: center; }
    </style>
    """

    score_class = "hi" if score_pct >= 50 else ("mid" if score_pct >= 20 else "lo")
    pill_class = "hi" if score_pct >= 50 else ("mid" if score_pct >= 20 else "lo")

    if not result.matches:
        body_matches = '<div class="empty">✅ Tidak ditemukan kemiripan signifikan. Dokumen terlihat orisinil terhadap corpus.</div>'
    else:
        rows = []
        for m in result.matches[:100]:
            pct = round(m.score * 100, 1)
            sc = "hi" if pct >= 70 else ("mid" if pct >= 40 else "lo")
            src = ""
            if m.source_url:
                src = f'<a href="{html.escape(m.source_url)}" target="_blank" rel="noopener">link</a>'
            else:
                src = html.escape(m.source_id)
            rows.append(
                f"""
                <div class="match">
                  <div class="meta">
                    <span class="pill {sc}">{pct}% mirip</span>
                    <span class="pill">{m.source_type.upper()}</span>
                    <span class="src"><b>{html.escape(m.source_title[:120])}</b> &nbsp; {src}</span>
                  </div>
                  {m.preview_html}
                </div>
                """
            )
        body_matches = "\n".join(rows)

    cite_html = ""
    if cs.original_chars > 0:
        cite_html = f"""
        <div class="cite-box">
          <div class="lbl">Pre-processing</div>
          <div class="cite-grid">
            <div class="cite-stat"><div class="num">{cs.inline_citations_stripped}</div><div class="lbl">inline citations</div></div>
            <div class="cite-stat"><div class="num">{cs.direct_quotes_stripped}</div><div class="lbl">direct quotes</div></div>
            <div class="cite-stat"><div class="num">{cs.urls_stripped}</div><div class="lbl">URLs</div></div>
            <div class="cite-stat"><div class="num">{cs.reduction_pct:.1f}%</div><div class="lbl">text reduced</div></div>
          </div>
        </div>
        """

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Plagiarism Report — {html.escape(result.document_name)}</title>{css}</head>
<body><div class="wrap">
  <h1>📄 DoubleCheck — Laporan Pengecekan</h1>
  <div class="sub">{html.escape(result.document_name)} &middot; {result.started_at} → {result.finished_at} ({result.elapsed_seconds:.1f}s)</div>

  <div class="grid">
    <div class="card"><div class="k">Overall Similarity</div>
      <div class="score {score_class}">{score_pct}%</div>
      <div class="bar"><div style="width:{min(score_pct, 100)}%"></div></div>
    </div>
    <div class="card"><div class="k">Flagged Paragraphs</div>
      <div class="v">{flagged} / {total}</div>
      <div class="bar"><div style="width:{(flagged/max(total,1))*100:.1f}%"></div></div>
    </div>
    <div class="card"><div class="k">Corpus Size</div>
      <div class="v">{result.corpus_size}</div>
      <div class="bar"><div style="width:100%"></div></div>
    </div>
    <div class="card"><div class="k">Verdict</div>
      <div class="v"><span class="pill {pill_class}">{
        'TINGGI — review manual!' if score_pct >= 50 else
        'SEDANG — periksa sumber' if score_pct >= 20 else
        'RENDAH — terlihat orisinil'
      }</span></div>
    </div>
  </div>
  {cite_html}
  <h2 style="font-size:16px;margin:0 0 12px">Top Matches ({len(result.matches)})</h2>
  {body_matches}
  <footer>Generated by DoubleCheck · {datetime.utcnow().isoformat(timespec='seconds')}Z</footer>
</div></body></html>"""
