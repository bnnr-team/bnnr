"""HTML report renderer for bnnr analyze — dashboard-aligned design.

Produces a self-contained HTML file with:
- Dark theme matching the BNNR dashboard (#080810 bg, #f0a069 accent)
- BNNR logo, confidence legend, and branding footer
- Grouped findings, confusion pair XAI analysis, best/worst per-class gallery
- Per-class XAI quality with component breakdown and formula
- Dataset health with class imbalance bars and embedded thumbnails
- Cross-validation with full metrics table
- Data-specific recommendations with literature references
"""

from __future__ import annotations

import base64
import html as html_mod
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CLASS_COLORS = [
    "#f0a069", "#818cf8", "#22c55e", "#ef4444", "#facc15",
    "#06b6d4", "#e879f9", "#f97316", "#a3e635", "#fb7185",
    "#38bdf8", "#c084fc", "#34d399", "#fbbf24", "#f472b6",
    "#2dd4bf", "#a78bfa", "#4ade80", "#fb923c", "#94a3b8",
]

_LOGO_B64_CACHE: str | None = None

_BNNR_LOGO_SVG_FALLBACK = (
    '<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="40" height="40" rx="10" fill="url(#lg)"/>'
    '<defs><linearGradient id="lg" x1="0" y1="0" x2="40" y2="40">'
    '<stop stop-color="#f0a069"/><stop offset="1" stop-color="#f5b888"/>'
    '</linearGradient></defs>'
    '<text x="20" y="27" text-anchor="middle" font-family="Inter,system-ui,sans-serif" '
    'font-size="20" font-weight="800" fill="#080810">B</text></svg>'
)


def _get_logo_html() -> str:
    """Return an <img> tag with the BNNR logo_dark.png embedded as base64, or SVG fallback."""
    global _LOGO_B64_CACHE  # noqa: PLW0603
    if _LOGO_B64_CACHE is None:
        _LOGO_B64_CACHE = _load_logo_b64()
    if _LOGO_B64_CACHE:
        return f'<img src="data:image/png;base64,{_LOGO_B64_CACHE}" alt="BNNR" style="height:40px;width:auto;" />'
    return _BNNR_LOGO_SVG_FALLBACK


def _load_logo_b64() -> str:
    """Load logo_dark.png from dashboard static, resize to 120px, return base64."""
    candidates = [
        Path(__file__).resolve().parent.parent / "dashboard" / "static" / "logo_dark.png",
        Path(__file__).resolve().parent.parent.parent.parent / "dashboard_web" / "public" / "logo_dark.png",
    ]
    for path in candidates:
        if path.exists():
            try:
                from PIL import Image  # type: ignore[import-untyped]
                pil_img = Image.open(path)
                resample = getattr(Image, "LANCZOS", getattr(Image, "ANTIALIAS", None))
                pil_img = pil_img.resize((120, 120), resample)  # type: ignore[assignment]
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("ascii")
            except ImportError:
                raw = path.read_bytes()
                return base64.b64encode(raw).decode("ascii")
            except Exception:
                logger.debug("Failed to load logo from %s", path)
    return ""


def _css() -> str:
    return """
:root {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  --radius: 12px;
  --transition: 0.2s ease;
  --bg: #080810;
  --bg-subtle: #0d0d18;
  --fg: #f1f5f9;
  --muted: #94a3b8;
  --card: #131322;
  --card-solid: #141423;
  --border: rgba(240, 160, 105, 0.18);
  --border-subtle: rgba(255, 255, 255, 0.06);
  --accent: #f0a069;
  --accent-hover: #f5b888;
  --accent-muted: rgba(240, 160, 105, 0.15);
  --glow: 0 0 20px rgba(240, 160, 105, 0.08), 0 0 40px rgba(240, 160, 105, 0.04);
  --glow-strong: 0 0 20px rgba(240, 160, 105, 0.15), 0 0 60px rgba(240, 160, 105, 0.08);
  --shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
  --green: #22c55e;
  --red: #ef4444;
  --yellow: #facc15;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--fg);
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  overflow-x: hidden;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(240,160,105,0.05) 0%, transparent 60%);
  pointer-events: none;
}
::selection { background: rgba(240,160,105,0.3); color: inherit; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(240,160,105,0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(240,160,105,0.4); }

.page { max-width: 1440px; margin: 0 auto; padding: 0 32px 60px; }
.two-col { display: flex; gap: 28px; }
.main-content { flex: 1; min-width: 0; }

.toc {
  position: sticky; top: 24px; width: 200px; flex-shrink: 0;
  align-self: flex-start; display: none;
}
@media (min-width: 960px) { .toc { display: block; } }
.toc-inner {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--glow);
}
.toc-title {
  font-size: 10px; font-weight: 800; text-transform: uppercase;
  letter-spacing: 1px; color: var(--muted); margin-bottom: 10px;
}
.toc a {
  display: block; font-size: 12px; font-weight: 600; color: var(--muted);
  text-decoration: none; padding: 4px 0; transition: color var(--transition);
}
.toc a:hover, .toc a.active { color: var(--accent); }

.report-header {
  padding: 32px 0 24px; border-bottom: 1px solid var(--border); margin-bottom: 8px;
  display: flex; align-items: center; gap: 14px;
}
.report-header h1 {
  font-size: 28px; font-weight: 800; letter-spacing: -0.5px;
  background: linear-gradient(135deg, #f0a069, #f5b888);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.report-meta {
  font-size: 12px; color: var(--muted); display: flex; gap: 16px; flex-wrap: wrap;
  margin-bottom: 8px;
}
.report-meta span { display: inline-flex; align-items: center; gap: 4px; }

.confidence-legend {
  display: flex; gap: 16px; align-items: center; padding: 8px 16px;
  background: var(--card); border: 1px solid var(--border-subtle);
  border-radius: 8px; margin-bottom: 24px; font-size: 12px; flex-wrap: wrap;
}
.confidence-legend .legend-label { font-weight: 700; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }

.section { margin-bottom: 32px; scroll-margin-top: 24px; }
.section-header {
  display: flex; align-items: center; gap: 10px; margin-bottom: 16px;
  padding-bottom: 8px; border-bottom: 1px solid var(--border-subtle);
}
.section-header h2 {
  font-size: 15px; font-weight: 700; color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.5px;
}
.section-count {
  font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  background: var(--accent-muted); color: var(--accent);
}

.card {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 18px; margin-bottom: 14px; box-shadow: var(--glow);
  transition: border-color var(--transition);
}
.card:hover { border-color: var(--accent); }
.card h3 { font-size: 14px; font-weight: 700; color: var(--accent); margin-bottom: 10px; }

.kpi-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px; margin-bottom: 16px;
}
.kpi-card {
  background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px; text-align: center; box-shadow: var(--glow);
  transition: border-color var(--transition), box-shadow var(--transition);
}
.kpi-card:hover { border-color: var(--accent); box-shadow: var(--glow-strong); }
.kpi-card.best {
  border-color: rgba(240,160,105,0.5);
  background: linear-gradient(135deg, rgba(240,160,105,0.1), rgba(240,160,105,0.03));
  box-shadow: 0 0 16px rgba(240,160,105,0.2);
}
.kpi-value { font-size: 28px; font-weight: 800; color: var(--accent); line-height: 1.1; }
.kpi-label {
  font-size: 10px; color: var(--muted); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.5px;
}

.badge {
  display: inline-flex; align-items: center; font-size: 11px; font-weight: 700;
  padding: 3px 12px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.3px;
}
.badge-ok { background: rgba(34,197,94,0.15); color: #4ade80; }
.badge-warning { background: rgba(240,160,105,0.15); color: #f5b888; }
.badge-critical { background: rgba(239,68,68,0.15); color: #f87171; }

.health-row {
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 14px;
}
.health-bar-track {
  flex: 1; min-width: 120px; max-width: 280px; height: 10px;
  background: rgba(255,255,255,0.06); border-radius: 5px; overflow: hidden;
}
.health-bar-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease; }
.health-score { font-size: 24px; font-weight: 800; color: var(--accent); }

.finding-grid, .rec-grid { display: grid; gap: 10px; }
.finding-card, .rec-card {
  background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px; border-left: 3px solid var(--accent);
  transition: border-color var(--transition), box-shadow var(--transition);
}
.finding-card:hover, .rec-card:hover { box-shadow: var(--glow); }
.finding-card.sev-critical { border-left-color: var(--red); }
.finding-card.sev-high { border-left-color: #f97316; }
.finding-card.sev-medium { border-left-color: var(--yellow); }
.finding-card.sev-low { border-left-color: var(--green); }
.finding-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
.finding-title { font-weight: 700; font-size: 13px; }
.finding-type {
  font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  background: var(--accent-muted); color: var(--accent);
}
.finding-desc { font-size: 13px; color: var(--muted); line-height: 1.55; margin-bottom: 6px; }
.finding-evidence { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
.evidence-tag {
  font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 6px;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle);
  color: var(--muted); font-family: 'SF Mono', SFMono-Regular, Consolas, monospace;
}
.class-tag {
  font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 6px;
  background: rgba(240,160,105,0.1); border: 1px solid rgba(240,160,105,0.25);
  color: var(--accent);
}

.rec-card { border-left-color: var(--accent); }
.rec-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.rec-title { font-weight: 700; font-size: 13px; }
.rec-priority {
  font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px;
  background: rgba(255,255,255,0.06); color: var(--muted);
}
.rec-why { font-size: 13px; color: var(--muted); line-height: 1.5; }
.rec-action { font-size: 12px; color: var(--accent); margin-top: 6px; }

.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 10px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-subtle); }
th {
  font-weight: 700; color: var(--muted); font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.4px; border-bottom: 2px solid var(--border);
}
tr:hover td { background: rgba(240,160,105,0.03); }
.sev-row-critical td { border-left: 3px solid var(--red); }
.sev-row-warning td:first-child { border-left: 3px solid var(--yellow); }

.metric-bar-track {
  width: 60px; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px;
  display: inline-block; vertical-align: middle; margin-left: 6px; overflow: hidden;
}
.metric-bar-fill { height: 100%; border-radius: 3px; display: block; }

.cm-grid { display: inline-grid; gap: 2px; font-size: 13px; margin: 0 auto; }
.cm-corner {
  display: grid; place-items: center; font-size: 9px; font-weight: 700; color: var(--muted);
}
.cm-header {
  text-align: center; padding: 4px 2px; font-size: 10px; font-weight: 700; color: var(--muted);
}
.cm-row-label {
  display: grid; place-items: center; font-size: 10px; font-weight: 700; color: var(--muted);
}
.cm-cell {
  text-align: center; padding: 8px 6px; border-radius: 4px; font-weight: 600;
  font-size: 12px; cursor: default; transition: transform var(--transition); min-width: 44px;
}
.cm-cell:hover { transform: scale(1.1); z-index: 1; position: relative; }

.xai-class-grid {
  display: grid; grid-template-columns: 1fr; gap: 14px;
}
.xai-class-card {
  background: var(--bg-subtle); border: 1px solid var(--border-subtle); border-radius: 10px;
  padding: 14px; transition: border-color var(--transition), box-shadow var(--transition);
}
.xai-class-card:hover { border-color: var(--border); box-shadow: var(--glow); }
.xai-class-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.xai-class-name { font-weight: 700; font-size: 13px; }
.xai-quality-badge {
  font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 8px;
  background: var(--bg-subtle); border: 1px solid var(--border-subtle); color: var(--accent);
}
.xai-quality-bar-track {
  width: 100%; height: 6px; background: rgba(255,255,255,0.06);
  border-radius: 3px; overflow: hidden; margin-bottom: 6px;
}
.xai-quality-bar-fill { height: 100%; border-radius: 3px; }
.xai-flags { display: none; }
.xai-flag {
  font-size: 10px; padding: 1px 6px; border-radius: 4px;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle);
}

.xai-example-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px; margin-top: 12px;
}
.xai-example-card {
  background: var(--bg-subtle); border: 1px solid var(--border-subtle);
  border-radius: 10px; padding: 8px; transition: border-color var(--transition);
}
.xai-example-card:hover { border-color: var(--accent); }
.xai-example-card img {
  width: 100%; border-radius: 8px; display: block; margin-bottom: 6px;
  border: 1px solid var(--border-subtle);
}
.xai-example-card.correct { border-color: rgba(34,197,94,0.4); }
.xai-example-card.wrong { border-color: rgba(239,68,68,0.4); }
.xai-example-meta { font-size: 11px; color: var(--muted); line-height: 1.4; }
.xai-example-meta strong { color: var(--fg); }

.comp-bars { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.comp-bar-item { display: flex; align-items: center; gap: 6px; }
.comp-bar-track {
  height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px;
  overflow: hidden; min-width: 100px; flex: 1;
}
.comp-bar-fill { height: 100%; border-radius: 2px; }
.comp-bar-label { font-size: 10px; color: var(--muted); text-transform: capitalize; letter-spacing: 0.3px; min-width: 72px; }

.dq-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px; margin-bottom: 14px;
}
.dq-stat {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px; text-align: center;
}
.dq-stat-value { font-size: 22px; font-weight: 800; color: var(--accent); }
.dq-stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
.dq-warnings { display: grid; gap: 8px; }
.dq-warning-row {
  display: flex; gap: 10px; padding: 10px 14px; border-radius: 8px;
  border: 1px solid var(--border); align-items: flex-start;
}
.dq-warning-row.dq-critical { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.3); }
.dq-warning-row.dq-warning { background: rgba(240,160,105,0.08); border-color: rgba(240,160,105,0.3); }
.dq-warning-row.dq-info { background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.3); }
.dq-warning-title { font-weight: 700; font-size: 13px; margin-bottom: 2px; }
.dq-warning-msg { font-size: 12px; color: var(--muted); }

.imbalance-chart { margin: 14px 0; }
.imbalance-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.imbalance-label { width: 60px; font-size: 11px; font-weight: 700; text-align: right; color: var(--muted); }
.imbalance-bar { height: 16px; border-radius: 4px; transition: width 0.4s ease; min-width: 2px; }
.imbalance-count { font-size: 11px; color: var(--muted); min-width: 50px; }

.thumb-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px; margin-top: 10px;
}
.thumb-grid img {
  width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px;
  border: 1px solid var(--border-subtle);
}

.cv-global { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.cv-metric {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 16px; text-align: center;
}
.cv-metric-value { font-size: 20px; font-weight: 800; color: var(--accent); }
.cv-metric-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }

.formula-box {
  background: var(--bg-subtle); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px; margin-bottom: 14px;
  font-family: 'SF Mono', SFMono-Regular, Consolas, monospace; font-size: 13px;
  color: var(--accent); text-align: center; line-height: 1.8;
}
.formula-box .formula-var { color: var(--fg); font-weight: 700; }
.formula-box .formula-weight { color: var(--muted); }

.confusion-pair-section { margin-top: 12px; }
.confusion-pair-card {
  background: var(--bg-subtle); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 18px; margin-bottom: 14px;
}
.pair-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.pair-title { font-size: 14px; font-weight: 700; }
.pair-count { font-size: 11px; color: var(--muted); }
.pair-overlays { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
.pair-overlay-card { text-align: center; }
.pair-overlay-card img {
  width: 180px; height: 180px; object-fit: cover; border-radius: 10px;
  border: 3px solid var(--border);
}
.pair-overlay-card.correct img { border-color: rgba(34,197,94,0.7); }
.pair-overlay-card.confused img { border-color: rgba(239,68,68,0.7); }
.pair-overlay-label { font-size: 11px; color: var(--muted); margin-top: 4px; }
.pair-heuristic {
  font-size: 13px; color: var(--muted); line-height: 1.6;
  padding: 10px 14px; background: rgba(240,160,105,0.05);
  border-radius: 8px; border-left: 3px solid var(--accent);
}
.pair-samples { margin-top: 12px; }
.pair-samples-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px;
}
.pair-samples-grid img {
  width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px;
  border: 2px solid rgba(239,68,68,0.4);
}

.footer {
  text-align: center; padding: 32px 0; margin-top: 40px;
  border-top: 1px solid var(--border-subtle); color: var(--muted); font-size: 12px;
}
.footer .brand {
  font-size: 14px; font-weight: 700;
  background: linear-gradient(135deg, #f0a069, #f5b888);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  margin-bottom: 4px;
}
.footer .tagline { font-style: italic; color: var(--accent); margin-top: 4px; font-size: 11px; }

.empty { color: var(--muted); font-size: 13px; font-style: italic; }

@media (max-width: 768px) {
  .page { padding: 0 12px 40px; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
  .kpi-value { font-size: 22px; }
  .cm-cell { min-width: 28px; font-size: 10px; padding: 4px 2px; }
  .cm-header, .cm-row-label { font-size: 9px; }
  .pair-overlays { flex-direction: column; }
  .report-header { flex-direction: column; align-items: flex-start; gap: 8px; }
}
"""


def _esc(s: str) -> str:
    return html_mod.escape(str(s), quote=True)


def _section_open(section_id: str, title: str, count: int | None = None) -> str:
    count_html = f' <span class="section-count">{count}</span>' if count is not None else ""
    return (
        f'<div class="section" id="{_esc(section_id)}">'
        f'<div class="section-header"><h2>{_esc(title)}</h2>{count_html}</div>'
    )


def _section_close() -> str:
    return "</div>"


def _color_for_value(value: float, low: float = 0.0, high: float = 1.0) -> str:
    t = max(0.0, min(1.0, (value - low) / max(high - low, 1e-9)))
    if t < 0.5:
        r, g = 239, int(68 + (250 - 68) * (t * 2))
        b = 68
    else:
        r = int(250 - (250 - 34) * ((t - 0.5) * 2))
        g, b = 197, 94
    return f"rgb({r},{g},{b})"


def _bar_color(value: float) -> str:
    if value >= 0.8:
        return "var(--green)"
    if value >= 0.5:
        return "var(--accent)"
    return "var(--red)"


# ─── KPI ───────────────────────────────────────────────────────────────────

def _kpi_cards(metrics: dict[str, Any], num_classes: int = 0, num_samples: int = 0) -> str:
    if not metrics and not num_classes:
        return ""
    parts: list[str] = []

    ordered = ["accuracy", "f1_macro", "loss", "cohen_kappa", "ece"]
    for key in ordered:
        if key not in metrics:
            continue
        v = metrics[key]
        if not isinstance(v, (int, float)):
            parts.append(f'<div class="kpi-card"><div class="kpi-value">{_esc(str(v))}</div>'
                         f'<div class="kpi-label">{_esc(key)}</div></div>')
            continue
        best = ""
        if key == "accuracy" and v >= 0.9:
            best = " best"
        if key == "loss":
            disp = f"{v:.4f}"
        elif key in ("cohen_kappa", "ece"):
            disp = f"{v:.3f}"
        else:
            disp = f"{v:.1%}"
        if key == "cohen_kappa":
            label = "Cohen\u2019s \u03ba"
        elif key == "ece":
            label = "ECE (top-1)"
        else:
            label = key
        parts.append(f'<div class="kpi-card{best}"><div class="kpi-value">{disp}</div>'
                     f'<div class="kpi-label">{_esc(label)}</div></div>')

    if num_classes:
        parts.append(f'<div class="kpi-card"><div class="kpi-value">{num_classes}</div>'
                     f'<div class="kpi-label">Classes</div></div>')
    if num_samples:
        parts.append(f'<div class="kpi-card"><div class="kpi-value">{num_samples:,}</div>'
                     f'<div class="kpi-label">Samples</div></div>')

    if not parts:
        return ""
    return '<div class="kpi-row">' + "".join(parts) + "</div>"


# ─── Executive Summary ─────────────────────────────────────────────────────

def _executive_block(summary: dict[str, Any]) -> str:
    if not summary:
        return '<p class="empty">No executive summary available.</p>'

    status = summary.get("health_status", "unknown")
    score = summary.get("health_score", 0)
    severity = summary.get("severity", "")
    badge_cls = ("badge-ok" if status == "ok"
                 else "badge-warning" if status == "warning"
                 else "badge-critical")
    bar_color = ("var(--green)" if status == "ok"
                 else "var(--accent)" if status == "warning"
                 else "var(--red)")

    h = '<div class="card">'
    h += '<div class="health-row">'
    h += f'<span class="badge {badge_cls}">{_esc(status.upper())}</span>'
    h += f'<span class="health-score">{score:.0%}</span>'
    h += (f'<div class="health-bar-track">'
          f'<div class="health-bar-fill" style="width:{score*100:.0f}%;background:{bar_color};"></div>'
          f'</div>')
    if severity:
        sev_cls = ("badge-critical" if severity in ("high", "critical")
                   else "badge-warning" if severity == "medium" else "badge-ok")
        h += f' <span class="badge {sev_cls}">Severity: {_esc(severity)}</span>'
    h += '</div></div>'

    key_findings = summary.get("key_findings", [])
    if key_findings:
        h += '<div class="card"><h3>Key Findings</h3><ul style="margin:0 0 0 18px;">'
        for f in key_findings[:5]:
            h += f"<li style='margin-bottom:4px;'>{_esc(f)}</li>"
        h += "</ul></div>"

    actions = summary.get("top_actions", [])
    if actions:
        h += '<div class="card"><h3>Top Actions</h3><ol style="margin:0 0 0 18px;">'
        for a in actions[:5]:
            h += f"<li style='margin-bottom:4px;color:var(--accent);'>{_esc(a)}</li>"
        h += "</ol></div>"

    return h


# ─── Class Diagnostics ─────────────────────────────────────────────────────

def _class_diagnostics_table(diagnostics: list[dict[str, Any]]) -> str:
    if not diagnostics:
        return '<p class="empty">No per-class diagnostics.</p>'

    h = '<div class="table-wrap"><table><thead><tr>'
    h += "<th>#</th><th>Class</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>"
    h += "<th>Cohen\u2019s \u03ba</th><th>Support</th><th>Pred</th><th>Severity</th>"
    h += "</tr></thead><tbody>"

    for i, d in enumerate(diagnostics[:30], 1):
        sev = d.get("severity", "ok")
        row_cls = f' class="sev-row-{sev}"' if sev in ("critical", "warning") else ""
        cid = _esc(str(d.get("class_id", "")))

        def _metric_cell(val: float) -> str:
            pct = f"{val:.1%}"
            bar = (f'<span class="metric-bar-track">'
                   f'<span class="metric-bar-fill" style="width:{val*100:.0f}%;background:{_bar_color(val)};"></span>'
                   f'</span>')
            return f"<td>{pct} {bar}</td>"

        sev_badge = ""
        if sev == "critical":
            sev_badge = '<span class="badge badge-critical">critical</span>'
        elif sev == "warning":
            sev_badge = '<span class="badge badge-warning">warning</span>'
        else:
            sev_badge = '<span class="badge badge-ok">ok</span>'

        kappa = d.get("cohen_kappa", 0)
        kappa_color = _bar_color(max(0, kappa))
        h += f"<tr{row_cls}><td>{i}</td><td><strong>{cid}</strong></td>"
        h += _metric_cell(d.get("accuracy", 0))
        h += _metric_cell(d.get("precision", 0))
        h += _metric_cell(d.get("recall", 0))
        h += _metric_cell(d.get("f1", 0))
        kappa_w = max(0, kappa) * 100
        h += (f'<td>{kappa:.3f} '
              f'<span class="metric-bar-track">'
              f'<span class="metric-bar-fill" style="width:{kappa_w:.0f}%;background:{kappa_color};"></span>'
              f'</span></td>')
        h += f"<td>{d.get('support', 0)}</td>"
        h += f"<td>{d.get('pred_count', 0)}</td>"
        h += f"<td>{sev_badge}</td></tr>"

    h += "</tbody></table></div>"
    return h


def _top_confused_pairs_chart(confusion: dict[str, Any], top_n: int = 8) -> str:
    """Render a horizontal bar chart of the most confused class pairs."""
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    if not isinstance(matrix, list) or not labels_list:
        return ""

    n = len(matrix)
    pairs: list[tuple[str, str, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                true_lbl = str(labels_list[i]) if i < len(labels_list) else str(i)
                pred_lbl = str(labels_list[j]) if j < len(labels_list) else str(j)
                pairs.append((true_lbl, pred_lbl, int(matrix[i][j])))
    if not pairs:
        return ""

    pairs.sort(key=lambda x: -x[2])
    top = pairs[:top_n]
    max_count = top[0][2] if top else 1

    h = '<div class="card"><h3>Most Confused Pairs</h3><div class="imbalance-chart">'
    for true_lbl, pred_lbl, count in top:
        ratio = count / max_count if max_count > 0 else 0
        h += '<div class="imbalance-row">'
        h += f'<div class="imbalance-label" style="width:80px;">{_esc(true_lbl)}\u2192{_esc(pred_lbl)}</div>'
        h += f'<div class="imbalance-bar" style="width:{ratio*100:.0f}%;background:var(--red);"></div>'
        h += f'<div class="imbalance-count">{count:,}\u00d7</div>'
        h += '</div>'
    h += '</div></div>'
    return h


# ─── Findings ──────────────────────────────────────────────────────────────

_CONFIDENCE_LABELS = {"high": "Observed", "medium": "Likely", "low": "Suspected"}
_CONFIDENCE_BADGE_CLS = {"high": "badge-ok", "medium": "badge-warning", "low": "badge-critical"}


def _findings_block(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return '<p class="empty">No findings to report.</p>'
    h = '<div class="finding-grid">'
    for f in findings[:12]:
        sev = f.get("severity", "medium")
        sev_cls = f"sev-{sev}" if sev in ("critical", "high", "medium", "low") else ""
        h += f'<div class="finding-card {sev_cls}">'
        h += '<div class="finding-header">'
        h += f'<span class="finding-title">{_esc(f.get("title", ""))}</span>'

        confidence = f.get("confidence", "medium")
        conf_label = _CONFIDENCE_LABELS.get(confidence, "")
        conf_badge = _CONFIDENCE_BADGE_CLS.get(confidence, "badge-warning")
        if conf_label:
            h += f' <span class="badge {conf_badge}">{_esc(conf_label)}</span>'

        ftype = f.get("finding_type", "")
        if ftype:
            h += f' <span class="finding-type">{_esc(ftype)}</span>'
        h += '</div>'

        desc = f.get("description", "")
        if desc:
            h += f'<div class="finding-desc">{_esc(desc)}</div>'

        class_ids = f.get("class_ids", [])
        if class_ids:
            h += '<div class="finding-evidence">'
            for cid in class_ids[:10]:
                h += f'<span class="class-tag">class {_esc(str(cid))}</span>'
            h += '</div>'

        evidence = f.get("evidence", [])
        if evidence:
            h += '<div class="finding-evidence">'
            for ev in evidence[:5]:
                h += f'<span class="evidence-tag">{_esc(str(ev))}</span>'
            h += '</div>'

        h += '</div>'
    h += '</div>'
    return h


# ─── Confusion Matrix Heatmap ──────────────────────────────────────────────

def _multilabel_per_label_table(confusion: dict[str, Any]) -> str:
    labels_list = confusion.get("labels", [])
    per_label = confusion.get("per_label", [])
    n_samples = confusion.get("n_samples", "")
    if not isinstance(per_label, list) or not labels_list:
        return '<p class="empty">No per-label confusion data.</p>'
    h = f'<p class="muted" style="margin-bottom:8px;">Multi-label (N={_esc(str(n_samples))}) — TP/FP/FN per label.</p>'
    h += '<div class="table-wrap"><table><thead><tr>'
    h += "<th>Label</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>"
    for i, cell in enumerate(per_label):
        if not isinstance(cell, dict):
            continue
        lid = str(labels_list[i]) if i < len(labels_list) else str(i)
        h += f'<tr><td>{_esc(lid)}</td>'
        h += f'<td>{cell.get("tp", 0)}</td>'
        h += f'<td>{cell.get("fp", 0)}</td>'
        h += f'<td>{cell.get("fn", 0)}</td></tr>'
    h += "</tbody></table></div>"
    return h


def _confusion_heatmap(confusion: dict[str, Any]) -> str:
    if confusion.get("type") == "multilabel_per_label":
        return _multilabel_per_label_table(confusion)
    matrix = confusion.get("matrix")
    labels_list = confusion.get("labels", [])
    if not isinstance(matrix, list) or not matrix or not labels_list:
        return '<p class="empty">No confusion matrix data.</p>'

    n = len(matrix)
    flat = [matrix[i][j] for i in range(n) for j in range(n)]
    max_val = max(flat) if flat else 1
    if max_val == 0:
        max_val = 1

    h = '<div style="overflow-x:auto;text-align:center;">'
    cell_w = max(48, min(64, 700 // n))
    grid_w = max(n * cell_w + 80, 400)
    h += f'<div class="cm-grid" style="grid-template-columns:80px repeat({n}, 1fr); max-width:{grid_w}px; margin:0 auto;">'
    h += '<div class="cm-corner">True \\ Pred</div>'
    for j in range(n):
        lbl = str(labels_list[j]) if j < len(labels_list) else str(j)
        h += f'<div class="cm-header">{_esc(lbl)}</div>'

    for i in range(n):
        row_lbl = str(labels_list[i]) if i < len(labels_list) else str(i)
        h += f'<div class="cm-row-label">{_esc(row_lbl)}</div>'
        for j in range(n):
            val = int(matrix[i][j])
            intensity = val / max_val
            if i == j:
                bg = f"rgba(34,197,94,{0.1 + intensity * 0.6:.2f})"
            else:
                bg = f"rgba(239,68,68,{intensity * 0.6:.2f})" if val > 0 else "transparent"
            fg = "var(--fg)" if intensity > 0.3 else "var(--muted)"
            disp = str(val) if val > 0 else ""
            pred_lbl = str(labels_list[j]) if j < len(labels_list) else str(j)
            h += (f'<div class="cm-cell" style="background:{bg};color:{fg};" '
                  f'title="True {_esc(row_lbl)} \u2192 Pred {_esc(pred_lbl)}: {val}">'
                  f'{disp}</div>')

    h += '</div></div>'
    return h


# ─── Confusion Pair XAI Analysis ──────────────────────────────────────────

def _confusion_pair_xai_block(pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return '<p class="empty">No confusion pair analysis available.</p>'

    h = '<div class="confusion-pair-section">'
    for pair in pairs[:3]:
        cls_a = _esc(str(pair.get("class_a", "")))
        cls_b = _esc(str(pair.get("class_b", "")))
        count = pair.get("count", 0)

        h += '<div class="confusion-pair-card">'
        h += '<div class="pair-header">'
        h += f'<span class="pair-title">Class {cls_a} \u2192 Class {cls_b}</span>'
        h += f'<span class="pair-count">{count} confusions</span>'
        h += '</div>'

        mean_correct = pair.get("mean_overlay_correct_a", "")
        mean_confused = pair.get("mean_overlay_confused_ab", "")
        if mean_correct or mean_confused:
            h += '<div class="pair-overlays" style="justify-content:center;">'
            if mean_correct:
                h += '<div class="pair-overlay-card correct">'
                h += (f'<img src="{_esc(mean_correct)}" alt="Mean saliency (correct)" loading="lazy" '
                      f'style="width:240px;height:240px;" />')
                h += f'<div class="pair-overlay-label">Correctly classified as {cls_a} (mean saliency)</div>'
                h += '</div>'
            if mean_confused:
                h += '<div class="pair-overlay-card confused">'
                h += (f'<img src="{_esc(mean_confused)}" alt="Mean saliency (confused)" loading="lazy" '
                      f'style="width:240px;height:240px;" />')
                h += f'<div class="pair-overlay-label">Confused {cls_a}\u2192{cls_b} (mean saliency)</div>'
                h += '</div>'
            h += '</div>'

        desc = pair.get("heuristic_description", "")
        if desc:
            h += f'<div class="pair-heuristic">{_esc(desc)}</div>'

        samples = pair.get("sample_overlays", [])
        if samples:
            h += '<div class="pair-samples">'
            h += f'<div style="font-size:11px;color:var(--muted);margin-bottom:6px;">Individual confused samples ({len(samples)})</div>'
            h += '<div class="pair-samples-grid">'
            for s in samples[:8]:
                overlay = _esc(str(s.get("overlay_path", "")))
                if overlay:
                    h += f'<img src="{overlay}" alt="Confused sample" loading="lazy" />'
            h += '</div></div>'

        h += '</div>'
    h += '</div>'
    return h


# ─── XAI Insights ─────────────────────────────────────────────────────────

def _xai_formula_block() -> str:
    return (
        '<div class="formula-box">'
        '<span class="formula-var">Q</span> = '
        '<span class="formula-weight">0.25</span>\u00b7<span class="formula-var">Acc</span> + '
        '<span class="formula-weight">0.20</span>\u00b7<span class="formula-var">Focus</span> + '
        '<span class="formula-weight">0.15</span>\u00b7<span class="formula-var">Coverage</span> + '
        '<span class="formula-weight">0.15</span>\u00b7<span class="formula-var">Coherence</span> + '
        '<span class="formula-weight">0.10</span>\u00b7<span class="formula-var">Edge</span> + '
        '<span class="formula-weight">0.15</span>\u00b7<span class="formula-var">Consistency</span>'
        '<div style="font-size:11px;color:var(--muted);margin-top:8px;font-family:Inter,system-ui,sans-serif;">'
        'Acc = prediction accuracy on probed samples; '
        'Focus = spatial concentration of attention; '
        'Coverage = fraction of object covered; '
        'Coherence = spatial continuity; '
        'Edge = proportion of attention away from borders; '
        'Consistency = stability across similar inputs.'
        '</div></div>'
    )


def _xai_overview_block(
    summary: dict[str, Any],
    per_class: dict[str, Any],
    best_worst: dict[str, dict[str, list]],
) -> str:
    if not summary and not per_class:
        return '<p class="empty">No XAI analysis was performed.</p>'

    h = _xai_formula_block()

    total_xai_samples = 0
    if per_class:
        for data in per_class.values():
            if isinstance(data, dict):
                total_xai_samples += data.get("sample_count", 0)

    if summary:
        mean_q = summary.get("mean_quality_score")
        if isinstance(mean_q, (int, float)):
            bar_color = _bar_color(mean_q)
            h += '<div class="card">'
            h += f'<h3>Global XAI Quality: {mean_q:.3f}</h3>'
            h += (f'<div class="xai-quality-bar-track" style="max-width:300px;">'
                  f'<div class="xai-quality-bar-fill" style="width:{mean_q*100:.0f}%;background:{bar_color};"></div>'
                  f'</div>')
            h += '<div style="font-size:12px;color:var(--muted);margin-top:6px;">'
            if total_xai_samples:
                h += f'Computed on <strong>{total_xai_samples}</strong> samples. '
                if total_xai_samples < 100:
                    h += 'Small probe set \u2014 treat as indicative, not definitive.'
            h += '</div></div>'

    if not per_class:
        return h

    items: list[tuple[float, str, int, list[str], dict[str, float]]] = []
    for cid, data in per_class.items():
        if isinstance(data, dict):
            q = data.get("mean_quality")
            sc = data.get("sample_count", 0)
            flags = data.get("flags", [])
            comps = data.get("components", {})
        else:
            continue
        if q is None:
            continue
        items.append((float(q), str(cid), sc, flags, comps))

    if not items:
        return h

    items.sort(key=lambda t: t[0])
    h += '<div class="xai-class-grid">'
    for q, cid, sc, flags, comps in items[:12]:
        q_color = _bar_color(q)
        h += f'<div class="xai-class-card" style="border-left:3px solid {q_color};">'

        h += '<div class="xai-class-header">'
        h += f'<span class="xai-class-name">Class {_esc(cid)}</span>'
        h += f'<span class="xai-quality-badge">{q:.3f}</span>'
        if sc:
            h += f'<span style="font-size:11px;color:var(--muted);margin-left:8px;">({sc} samples)</span>'
        h += '</div>'
        h += (f'<div class="xai-quality-bar-track">'
              f'<div class="xai-quality-bar-fill" style="width:{q*100:.0f}%;background:{q_color};"></div>'
              f'</div>')

        if comps:
            comp_names = ["accuracy", "focus", "coverage", "coherence", "edge", "consistency"]
            h += '<div class="comp-bars">'
            for cn in comp_names:
                cv = comps.get(cn, 0.0)
                bc = _bar_color(cv)
                h += '<div class="comp-bar-item">'
                h += f'<div class="comp-bar-label">{cn}</div>'
                h += f'<div class="comp-bar-track"><div class="comp-bar-fill" style="width:{cv*100:.0f}%;background:{bc};"></div></div>'
                h += f'<div style="font-size:10px;color:var(--muted);min-width:32px;text-align:right;">{cv:.2f}</div>'
                h += '</div>'
            h += '</div>'

        bw = best_worst.get(cid, {})
        best_list = bw.get("best", [])
        worst_list = bw.get("worst", [])
        if best_list or worst_list:
            h += '<div class="xai-example-grid">'
            for ex in best_list[:4]:
                overlay = _esc(str(ex.get("overlay_path", "")))
                desc = ex.get("description", "")
                h += '<div class="xai-example-card correct">'
                if overlay:
                    h += f'<img src="{overlay}" loading="lazy" />'
                h += '<div class="xai-example-meta">'
                h += f'<strong style="color:var(--green);">CORRECT</strong> conf={float(ex.get("confidence", 0)):.2f}'
                if desc:
                    h += f'<br>{_esc(desc)}'
                h += '</div></div>'
            for ex in worst_list[:4]:
                overlay = _esc(str(ex.get("overlay_path", "")))
                desc = ex.get("description", "")
                h += '<div class="xai-example-card wrong">'
                if overlay:
                    h += f'<img src="{overlay}" loading="lazy" />'
                h += '<div class="xai-example-meta">'
                h += (f'<strong style="color:var(--red);">WRONG</strong> '
                      f'pred={ex.get("pred_label", "")} conf={float(ex.get("confidence", 0)):.2f}')
                if desc:
                    h += f'<br>{_esc(desc)}'
                h += '</div></div>'
            h += '</div>'

        h += '</div>'
    h += '</div>'
    return h


# ─── XAI Examples (legacy) ────────────────────────────────────────────────

def _xai_examples_block(examples_per_class: dict[str, Any]) -> str:
    if not examples_per_class:
        return ""
    h = ""
    shown = 0
    for cls_id, examples in examples_per_class.items():
        if not examples or shown >= 12:
            break
        h += (f'<div style="margin:18px 0 10px;display:flex;align-items:center;gap:8px;">'
              f'<span style="font-size:14px;font-weight:700;">Class {_esc(str(cls_id))}</span>'
              f'<span class="evidence-tag">{len(examples)} examples</span>'
              f'</div>')
        h += '<div class="xai-example-grid">'
        for ex in examples[:6]:
            overlay = _esc(str(ex.get("overlay_path", ex.get("image_path", ""))))
            is_wrong = ex.get("true_label") != ex.get("pred_label")
            card_cls = "wrong" if is_wrong else ""
            h += f'<div class="xai-example-card {card_cls}">'
            if overlay:
                h += f'<img src="{overlay}" alt="XAI overlay class {_esc(str(cls_id))}" loading="lazy" />'
            h += '<div class="xai-example-meta">'
            wrong_tag = ' <span style="color:var(--red);font-weight:700;">WRONG</span>' if is_wrong else ""
            h += (f'<strong>#{ex.get("index","")}</strong>{wrong_tag}<br>'
                  f'true={ex.get("true_label","")} '
                  f'pred={ex.get("pred_label","")} '
                  f'conf={float(ex.get("confidence",0)):.3f}')
            h += '</div></div>'
        h += '</div>'
        shown += 1
    return h


# ─── Data Quality ──────────────────────────────────────────────────────────

def _data_quality_block(dq: dict[str, Any], true_dist: dict[str, int]) -> str:
    if not dq:
        return '<p class="empty">No data quality analysis was performed.</p>'

    scanned = dq.get("scanned_samples") or dq.get("total_scanned", 0)
    dup_pairs = dq.get("total_duplicate_pairs", 0)
    flagged = dq.get("total_flagged_images", 0)
    dup_groups = dq.get("duplicate_groups", [])
    warnings = dq.get("warnings", [])

    h = '<div class="dq-stats">'
    if scanned:
        h += f'<div class="dq-stat"><div class="dq-stat-value">{scanned:,}</div><div class="dq-stat-label">Scanned</div></div>'
    h += f'<div class="dq-stat"><div class="dq-stat-value">{dup_pairs}</div><div class="dq-stat-label">Duplicate Pairs</div></div>'
    if dup_groups:
        h += f'<div class="dq-stat"><div class="dq-stat-value">{len(dup_groups)}</div><div class="dq-stat-label">Duplicate Groups</div></div>'
    h += f'<div class="dq-stat"><div class="dq-stat-value">{flagged}</div><div class="dq-stat-label">Flagged Images</div></div>'
    h += '</div>'

    if true_dist:
        h += _class_imbalance_chart(true_dist)

    dup_thumbs = dq.get("duplicate_thumbnails", [])
    if dup_thumbs:
        groups_map: dict[str, list[dict[str, str]]] = {}
        ungrouped: list[dict[str, str]] = []
        for thumb in dup_thumbs:
            grp = thumb.get("group", "")
            if grp:
                groups_map.setdefault(grp, []).append(thumb)
            else:
                ungrouped.append(thumb)

        h += '<div class="card"><h3>Near-Duplicate Images</h3>'
        h += ('<p style="font-size:12px;color:var(--muted);margin:0 0 12px 0;">'
              'These images were detected as near-identical by perceptual hashing (dHash). '
              'They may inflate training metrics if present in both train and validation sets.</p>')
        for grp_name, grp_imgs in list(groups_map.items())[:8]:
            h += '<div style="margin-bottom:12px;">'
            h += '<div style="font-size:12px;color:var(--accent);margin-bottom:6px;font-weight:600;">'
            h += f'Duplicate Group ({len(grp_imgs)} images)</div>'
            h += '<div class="thumb-grid">'
            for thumb in grp_imgs[:6]:
                b64 = thumb.get("base64", "")
                if b64:
                    h += f'<img src="data:image/png;base64,{b64}" title="{_esc(thumb.get("filename", ""))}" />'
            h += '</div></div>'
        if ungrouped:
            h += '<div class="thumb-grid">'
            for thumb in ungrouped[:12]:
                b64 = thumb.get("base64", "")
                if b64:
                    h += f'<img src="data:image/png;base64,{b64}" title="{_esc(thumb.get("filename", ""))}" />'
            h += '</div>'
        h += '</div>'
    elif dup_groups:
        h += '<div class="card"><h3>Near-Duplicate Groups</h3>'
        for gi, group in enumerate(dup_groups[:10]):
            size = group.get("size", 0)
            indices = group.get("indices", [])
            h += '<div style="padding:6px 0;font-size:13px;color:var(--fg);">'
            h += f'<span style="color:var(--accent);font-weight:600;">Group {gi+1}</span>'
            h += f' &mdash; {size} images (indices: {", ".join(str(i) for i in indices[:8])}'
            if len(indices) > 8:
                h += f', \u2026+{len(indices)-8} more'
            h += ')</div>'
        h += '</div>'

    flagged_thumbs = dq.get("flagged_thumbnails", [])
    if flagged_thumbs:
        h += '<div class="card"><h3>Flagged Images (Potential Label or Quality Issues)</h3>'
        h += ('<p style="font-size:12px;color:var(--muted);margin:0 0 12px 0;">'
              'These images were flagged for suspicious properties: '
              'zero variance (blank/solid color), extreme dimensions, or unusual pixel distributions. '
              'Inspect manually to confirm if labels are correct.</p>')
        h += '<div class="thumb-grid">'
        for thumb in flagged_thumbs[:16]:
            b64 = thumb.get("base64", "")
            if b64:
                h += f'<img src="data:image/png;base64,{b64}" title="{_esc(thumb.get("filename", ""))}" />'
        h += '</div></div>'

    if warnings:
        h += '<div class="dq-warnings">'
        for w in warnings[:8]:
            msg = w.get("message", "")
            wtype = w.get("type", "info")
            count = w.get("count", 0)
            sev_cls = ("dq-critical" if wtype in ("critical", "error")
                       else "dq-warning" if wtype in ("warning", "high_variance", "tiny_images",
                                                      "zero_variance", "near_duplicates")
                       else "dq-info")
            h += f'<div class="dq-warning-row {sev_cls}">'
            h += '<div>'
            h += f'<div class="dq-warning-title">{_esc(str(wtype))}'
            if count:
                h += f' <span class="evidence-tag">{count} affected</span>'
            h += '</div>'
            if msg:
                h += f'<div class="dq-warning-msg">{_esc(str(msg))}</div>'
            h += '</div></div>'
        h += '</div>'

    h += ('<div style="font-size:11px;color:var(--muted);margin-top:10px;font-style:italic;">'
          'Dataset health uses perceptual hashing (dHash) and statistical heuristics. '
          'False positives possible on small/synthetic datasets.</div>')
    return h


def _class_imbalance_chart(true_dist: dict[str, int]) -> str:
    if not true_dist:
        return ""
    sorted_items = sorted(true_dist.items(), key=lambda x: -x[1])
    max_count = max(true_dist.values()) if true_dist else 1
    mean_count = sum(true_dist.values()) / len(true_dist) if true_dist else 0

    h = '<div class="card"><h3>Class Distribution</h3><div class="imbalance-chart">'
    for cid, count in sorted_items:
        ratio = count / max_count if max_count > 0 else 0
        deviation = (count - mean_count) / mean_count if mean_count > 0 else 0
        if deviation > 0.3:
            color = "var(--accent)"
        elif deviation < -0.3:
            color = "var(--red)"
        else:
            color = "var(--green)"
        h += '<div class="imbalance-row">'
        h += f'<div class="imbalance-label">{_esc(str(cid))}</div>'
        h += f'<div class="imbalance-bar" style="width:{ratio*100:.0f}%;background:{color};"></div>'
        h += f'<div class="imbalance-count">{count:,}</div>'
        h += '</div>'
    h += '</div></div>'
    return h


# ─── Cross-Validation ─────────────────────────────────────────────────────

def _cv_block(cv: dict[str, Any]) -> str:
    if not cv or not cv.get("n_folds"):
        return '<p class="empty">Cross-validation was not run. Use --cv-folds to enable.</p>'

    n_folds = cv.get("n_folds", 0)
    gm = cv.get("global_metrics", {}) or {}
    per_fold = cv.get("per_fold_metrics", []) or []

    h = '<div class="cv-global">'
    h += f'<div class="cv-metric"><div class="cv-metric-value">{n_folds}</div><div class="cv-metric-label">Folds</div></div>'
    if "mean_subset_accuracy" in gm:
        metric_keys = [
            ("mean_f1_macro", "F1 macro"),
            ("mean_subset_accuracy", "Subset acc"),
        ]
    else:
        metric_keys = [
            ("mean_accuracy", "Accuracy"),
            ("mean_precision_macro", "Precision"),
            ("mean_recall_macro", "Recall"),
            ("mean_f1_macro", "F1"),
            ("mean_cohen_kappa", "\u03ba"),
        ]
    for key, label in metric_keys:
        if key in gm:
            std_key = key.replace("mean_", "std_")
            std_val = gm.get(std_key, 0)
            h += (f'<div class="cv-metric"><div class="cv-metric-value">{gm[key]:.3f}</div>'
                  f'<div class="cv-metric-label">{_esc(label)} (\u00b1{std_val:.3f})</div></div>')
    h += '</div>'

    if per_fold:
        h += '<div class="table-wrap"><table><thead><tr>'
        if per_fold and "subset_accuracy" in per_fold[0]:
            h += "<th>Fold</th><th>F1 macro</th><th>Subset acc</th><th>Support</th>"
            h += "</tr></thead><tbody>"
            for fm in per_fold:
                h += f'<tr><td>{fm.get("fold","")}</td>'
                h += f'<td>{fm.get("f1_macro", 0):.3f}</td>'
                h += f'<td>{fm.get("subset_accuracy", 0):.3f}</td>'
                h += f'<td>{fm.get("support",0)}</td></tr>'
        else:
            h += "<th>Fold</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>\u03ba</th><th>Support</th>"
            h += "</tr></thead><tbody>"
            for fm in per_fold:
                acc = fm.get("accuracy", 0)
                h += f'<tr><td>{fm.get("fold","")}</td>'
                h += f'<td>{acc:.3f} '
                h += (f'<span class="metric-bar-track"><span class="metric-bar-fill" '
                      f'style="width:{acc*100:.0f}%;background:{_bar_color(acc)};"></span></span></td>')
                h += f'<td>{fm.get("precision_macro", 0):.3f}</td>'
                h += f'<td>{fm.get("recall_macro", 0):.3f}</td>'
                h += f'<td>{fm.get("f1_macro", 0):.3f}</td>'
                h += f'<td>{fm.get("cohen_kappa", 0):.3f}</td>'
                h += f'<td>{fm.get("support",0)}</td></tr>'
        h += "</tbody></table></div>"
    return h


# ─── Recommendations ──────────────────────────────────────────────────────

def _rec_block(recs_struct: list[dict[str, Any]], recs_legacy: list[str]) -> str:
    if recs_struct:
        h = '<div class="rec-grid">'
        for idx, r in enumerate(recs_struct[:5], 1):
            h += '<div class="rec-card">'
            h += '<div class="rec-header">'
            h += f'<span class="rec-priority" style="min-width:22px;text-align:center;">{idx}</span>'
            h += f'<span class="rec-title">{_esc(r.get("title", ""))}</span>'
            conf = r.get("confidence", "medium")
            conf_label = _CONFIDENCE_LABELS.get(conf, "")
            conf_badge = _CONFIDENCE_BADGE_CLS.get(conf, "badge-warning")
            if conf_label:
                h += f' <span class="badge {conf_badge}">{_esc(conf_label)}</span>'
            h += '</div>'
            why = r.get("why", "")
            if why:
                truncated = why[:400] + "..." if len(why) > 400 else why
                h += f'<div class="rec-why">{_esc(truncated)}</div>'
            ev_run = r.get("evidence_from_run") or []
            if isinstance(ev_run, list) and ev_run:
                h += '<div class="rec-evidence" style="font-size:11px;color:var(--muted);margin-top:4px;">'
                h += "<strong>From this run:</strong> " + _esc("; ".join(str(x) for x in ev_run[:5]))
                h += "</div>"
            action = r.get("action", "")
            if action:
                h += f'<div class="rec-action">\u2192 {_esc(action)}</div>'
            impact = r.get("expected_impact", "")
            if impact:
                h += (f'<div style="font-size:12px;color:var(--green);margin-top:6px;'
                      f'padding:4px 10px;background:rgba(34,197,94,0.08);border-radius:6px;'
                      f'border:1px solid rgba(34,197,94,0.2);">'
                      f'{_esc(impact)}</div>')
            reference = r.get("example_command", "")
            if reference:
                h += (f'<div style="font-size:10px;color:var(--muted);margin-top:4px;'
                      f'font-style:italic;">\U0001f4da {_esc(reference)}</div>')
            h += '</div>'
        h += '</div>'
        return h
    if recs_legacy:
        h = '<div class="rec-grid">'
        for legacy_r in recs_legacy[:5]:
            h += f'<div class="rec-card"><div class="rec-title">{_esc(legacy_r)}</div></div>'
        h += '</div>'
        return h
    return '<p class="empty">No recommendations.</p>'


# ─── Main Renderer ────────────────────────────────────────────────────────

def render_analysis_html(report: Any) -> str:
    """Generate full HTML report aligned with BNNR dashboard design language."""

    schema_version = getattr(report, "schema_version", "0.2.1")
    metrics = getattr(report, "metrics", {}) or {}
    confusion = getattr(report, "confusion", {}) or {}
    summary = getattr(report, "executive_summary", {}) or {}
    analysis_scope = getattr(report, "analysis_scope", {}) or {}
    diagnostics = getattr(report, "class_diagnostics", []) or []
    findings = getattr(report, "findings", []) or []
    xai_summary = getattr(report, "xai_quality_summary", {}) or {}
    xai_per_class = getattr(report, "xai_quality_per_class", {}) or {}
    xai_examples = getattr(report, "xai_examples_per_class", {}) or {}
    data_quality = (
        getattr(report, "data_quality_summary", {})
        or getattr(report, "data_quality_result", {})
        or {}
    )
    cv_results = getattr(report, "cv_results", {}) or {}
    recs_struct = getattr(report, "recommendations_structured", []) or []
    recs_legacy = getattr(report, "recommendations", []) or []
    true_dist = getattr(report, "true_distribution", {}) or {}
    confusion_pair_xai = getattr(report, "confusion_pair_xai", []) or []
    best_worst = getattr(report, "best_worst_examples", {}) or {}

    num_classes = len(true_dist) if true_dist else (len(diagnostics) if diagnostics else 0)
    num_samples = sum(true_dist.values()) if true_dist else 0
    task_label = str(analysis_scope.get("task", "classification")).replace("_", "-")
    scope_reason = analysis_scope.get("reason", "")
    scope_notes = analysis_scope.get("notes", "")

    has_xai = bool(xai_summary or xai_per_class or xai_examples or best_worst)
    has_dq = bool(data_quality)
    has_cv = bool(cv_results and cv_results.get("n_folds"))
    has_confusion_xai = bool(confusion_pair_xai)

    sections: list[tuple[str, str]] = [
        ("exec", "Executive Summary"),
        ("diagnostics", "Class Diagnostics"),
        ("confusion", "Confusion Matrix"),
    ]
    if findings:
        sections.append(("findings", "Findings"))
    if has_confusion_xai:
        sections.append(("confusion-xai", "Confusion Analysis"))
    if has_xai:
        sections.append(("xai", "XAI Insights"))
    if has_dq:
        sections.append(("dq", "Dataset Health"))
    if has_cv:
        sections.append(("cv", "Cross-Validation"))
    sections.append(("recs", "Recommendations"))

    toc = '<nav class="toc"><div class="toc-inner"><div class="toc-title">Contents</div>'
    for sid, label in sections:
        toc += f'<a href="#{sid}">{_esc(label)}</a>'
    toc += '</div></nav>'

    lines: list[str] = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>BNNR Analysis Report</title>",
        "<style>" + _css() + "</style>",
        "</head><body>",
        '<div class="page">',
        # Header with logo
        '<div class="report-header">',
        _get_logo_html(),
        '<div>',
        "<h1>BNNR Analysis Report</h1>",
        '<div class="report-meta">',
        f'<span>v{_esc(schema_version)}</span>',
        f'<span>{_esc(task_label.title())}</span>',
        f'<span>{num_classes} classes \u00b7 {num_samples:,} samples</span>' if num_samples else "",
        '</div></div></div>',
        *(
            [
                '<div class="confidence-legend" style="margin-bottom:12px;background:rgba(234,179,8,0.12);'
                'border:1px solid rgba(234,179,8,0.35);border-radius:8px;padding:10px 14px;">',
                f'<strong>Scope:</strong> {_esc(scope_reason)}',
                "</div>",
            ]
            if scope_reason
            else []
        ),
        *(
            [
                '<div style="font-size:12px;color:var(--muted);margin-bottom:12px;padding:8px 12px;'
                'background:var(--card);border-radius:6px;">',
                f'<strong>Note:</strong> {_esc(scope_notes)}',
                "</div>",
            ]
            if scope_notes
            else []
        ),
        # Confidence legend
        '<div class="confidence-legend">',
        '<span class="legend-label">Confidence levels:</span>',
        '<span class="badge badge-ok">Observed</span> = measured from data',
        '<span class="badge badge-warning">Likely</span> = heuristic signal',
        '<span class="badge badge-critical">Suspected</span> = needs verification',
        '</div>',
        '<div class="two-col">',
        toc,
        '<div class="main-content">',
    ]

    # 1. Executive Summary
    lines.append(_section_open("exec", "Executive Summary"))
    lines.append(_executive_block(summary))
    lines.append(_kpi_cards(metrics, num_classes, num_samples))
    lines.append(_section_close())

    # 2. Class Diagnostics
    lines.append(_section_open("diagnostics", "Class Diagnostics", count=len(diagnostics)))
    lines.append(_class_diagnostics_table(diagnostics))
    lines.append(_top_confused_pairs_chart(confusion))
    lines.append(_section_close())

    # 3. Confusion Matrix
    lines.append(_section_open("confusion", "Confusion Matrix"))
    lines.append(_confusion_heatmap(confusion))
    lines.append(_section_close())

    # 4. Findings (grouped)
    if findings:
        lines.append(_section_open("findings", "Findings", count=len(findings)))
        lines.append(_findings_block(findings))
        lines.append(_section_close())

    # 5. Confusion Analysis (XAI-powered)
    if has_confusion_xai:
        lines.append(_section_open("confusion-xai", "Confusion Analysis", count=len(confusion_pair_xai)))
        lines.append(_confusion_pair_xai_block(confusion_pair_xai))
        lines.append(_section_close())

    # 6. XAI Insights
    if has_xai:
        lines.append(_section_open("xai", "XAI Insights"))
        lines.append(_xai_overview_block(xai_summary, xai_per_class, best_worst))
        if xai_examples and not best_worst:
            lines.append(_xai_examples_block(xai_examples))
        lines.append(_section_close())

    # 7. Dataset Health
    if has_dq:
        lines.append(_section_open("dq", "Dataset Health"))
        lines.append(_data_quality_block(data_quality, true_dist))
        lines.append(_section_close())

    # 8. Cross-Validation
    if has_cv:
        lines.append(_section_open("cv", "Cross-Validation"))
        lines.append(_cv_block(cv_results))
        lines.append(_section_close())

    # 9. Recommendations
    rec_count = min(len(recs_struct), 5) if recs_struct else min(len(recs_legacy), 5)
    lines.append(_section_open("recs", "Recommendations", count=rec_count))
    lines.append(_rec_block(recs_struct, recs_legacy))
    lines.append(_section_close())

    lines.append('</div></div>')

    # Footer
    lines.append(
        '<div class="footer">'
        '<div class="brand">BNNR \u2014 Train \u2192 Explain \u2192 Improve \u2192 Prove</div>'
        '<div class="tagline">Helping your model earn its place in production.</div>'
        f'<div style="margin-top:8px;">Schema v{_esc(schema_version)}</div>'
        '</div>'
    )

    lines.append('</div>')
    lines.append("</body></html>")
    return "\n".join(lines)
