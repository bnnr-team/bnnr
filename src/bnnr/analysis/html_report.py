"""HTML report renderer for bnnr analyze — dashboard-aligned design.

Produces a self-contained HTML file with:
- Dark theme matching the BNNR dashboard (#080810 bg, #f0a069 accent)
- Sticky table of contents with smooth-scroll navigation
- Interactive confusion-matrix heatmap (colored cells)
- Per-class diagnostic bars (recall / precision / F1)
- Interactive cluster scatter plot (per-class colors, JS tooltips)
- Severity-styled finding and recommendation cards
- Professional typography and spacing
"""

from __future__ import annotations

import html as html_mod
import json
from typing import Any

_CLASS_COLORS = [
    "#f0a069", "#818cf8", "#22c55e", "#ef4444", "#facc15",
    "#06b6d4", "#e879f9", "#f97316", "#a3e635", "#fb7185",
    "#38bdf8", "#c084fc", "#34d399", "#fbbf24", "#f472b6",
    "#2dd4bf", "#a78bfa", "#4ade80", "#fb923c", "#94a3b8",
]


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

/* Layout */
.page { max-width: 1200px; margin: 0 auto; padding: 0 24px 60px; }
.two-col { display: flex; gap: 28px; }
.main-content { flex: 1; min-width: 0; }

/* Sticky TOC */
.toc {
  position: sticky;
  top: 24px;
  width: 200px;
  flex-shrink: 0;
  align-self: flex-start;
  display: none;
}
@media (min-width: 960px) { .toc { display: block; } }
.toc-inner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
  box-shadow: var(--glow);
}
.toc-title {
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 10px;
}
.toc a {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  text-decoration: none;
  padding: 4px 0;
  transition: color var(--transition);
}
.toc a:hover, .toc a.active { color: var(--accent); }

/* Header */
.report-header {
  padding: 32px 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 28px;
}
.report-header h1 {
  font-size: 28px;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #f0a069, #f5b888);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 6px;
}
.report-meta {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.report-meta span { display: inline-flex; align-items: center; gap: 4px; }

/* Section */
.section { margin-bottom: 32px; scroll-margin-top: 24px; }
.section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-subtle);
}
.section-header h2 {
  font-size: 15px;
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.section-count {
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--accent-muted);
  color: var(--accent);
}

/* Card */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
  margin-bottom: 14px;
  box-shadow: var(--glow);
  transition: border-color var(--transition);
}
.card:hover { border-color: var(--accent); }
.card h3 {
  font-size: 14px;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 10px;
}

/* KPI Row */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  text-align: center;
  box-shadow: var(--glow);
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
  font-size: 10px;
  color: var(--muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Badges */
.badge {
  display: inline-flex;
  align-items: center;
  font-size: 11px;
  font-weight: 700;
  padding: 3px 12px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
.badge-ok { background: rgba(34,197,94,0.15); color: #4ade80; }
.badge-warning { background: rgba(240,160,105,0.15); color: #f5b888; }
.badge-critical { background: rgba(239,68,68,0.15); color: #f87171; }

/* Health gauge */
.health-row {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.health-bar-track {
  flex: 1;
  min-width: 120px;
  max-width: 280px;
  height: 10px;
  background: rgba(255,255,255,0.06);
  border-radius: 5px;
  overflow: hidden;
}
.health-bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.6s ease;
}
.health-score {
  font-size: 24px;
  font-weight: 800;
  color: var(--accent);
}

/* Finding / Recommendation cards */
.finding-grid, .rec-grid { display: grid; gap: 10px; }
.finding-card, .rec-card {
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  border-left: 3px solid var(--accent);
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
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--accent-muted);
  color: var(--accent);
}
.finding-desc { font-size: 13px; color: var(--muted); line-height: 1.55; margin-bottom: 6px; }
.finding-evidence {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.evidence-tag {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 6px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-subtle);
  color: var(--muted);
  font-family: 'SF Mono', SFMono-Regular, Consolas, monospace;
}
.finding-action {
  font-size: 12px;
  color: var(--accent);
  padding: 6px 10px;
  background: var(--accent-muted);
  border-radius: 6px;
  border: 1px solid var(--border);
  margin-top: 6px;
}
.rec-card { border-left-color: var(--accent); }
.rec-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.rec-title { font-weight: 700; font-size: 13px; }
.rec-priority {
  font-size: 10px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  color: var(--muted);
}
.rec-why { font-size: 13px; color: var(--muted); line-height: 1.5; }
.rec-action {
  font-size: 12px;
  color: var(--accent);
  margin-top: 6px;
}

/* Tables */
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; border-radius: 10px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-subtle); }
th {
  font-weight: 700;
  color: var(--muted);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  border-bottom: 2px solid var(--border);
}
tr:hover td { background: rgba(240,160,105,0.03); }
.sev-row-critical td { border-left: 3px solid var(--red); }
.sev-row-warning td:first-child { border-left: 3px solid var(--yellow); }

/* Metric bar (inline in table) */
.metric-bar-track {
  width: 60px;
  height: 6px;
  background: rgba(255,255,255,0.06);
  border-radius: 3px;
  display: inline-block;
  vertical-align: middle;
  margin-left: 6px;
  overflow: hidden;
}
.metric-bar-fill {
  height: 100%;
  border-radius: 3px;
  display: block;
}

/* Confusion matrix heatmap */
.cm-grid {
  display: inline-grid;
  gap: 2px;
  font-size: 12px;
  margin: 0 auto;
}
.cm-corner {
  display: grid;
  place-items: center;
  font-size: 9px;
  font-weight: 700;
  color: var(--muted);
}
.cm-header {
  text-align: center;
  padding: 4px 2px;
  font-size: 10px;
  font-weight: 700;
  color: var(--muted);
}
.cm-row-label {
  display: grid;
  place-items: center;
  font-size: 10px;
  font-weight: 700;
  color: var(--muted);
}
.cm-cell {
  text-align: center;
  padding: 6px 4px;
  border-radius: 4px;
  font-weight: 600;
  font-size: 11px;
  cursor: default;
  transition: transform var(--transition);
  min-width: 36px;
}
.cm-cell:hover { transform: scale(1.1); z-index: 1; position: relative; }

/* XAI Grid */
.xai-class-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.xai-class-card {
  background: var(--bg-subtle);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 14px;
  transition: border-color var(--transition), box-shadow var(--transition);
}
.xai-class-card:hover { border-color: var(--border); box-shadow: var(--glow); }
.xai-class-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.xai-class-name { font-weight: 700; font-size: 13px; }
.xai-quality-badge {
  font-size: 10px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 8px;
  background: var(--bg-subtle);
  border: 1px solid var(--border-subtle);
  color: var(--accent);
}
.xai-quality-bar-track {
  width: 100%;
  height: 6px;
  background: rgba(255,255,255,0.06);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 6px;
}
.xai-quality-bar-fill { height: 100%; border-radius: 3px; }
.xai-flags {
  font-size: 11px;
  color: var(--muted);
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.xai-flag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 4px;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border-subtle);
}

/* XAI Examples */
.xai-example-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.xai-example-card {
  background: var(--bg-subtle);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 8px;
  transition: border-color var(--transition);
}
.xai-example-card:hover { border-color: var(--accent); }
.xai-example-card img {
  width: 100%;
  border-radius: 8px;
  display: block;
  margin-bottom: 6px;
  border: 1px solid var(--border-subtle);
}
.xai-example-meta {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.4;
}
.xai-example-meta strong { color: var(--fg); }

/* Worst predictions table */
.worst-table tr.wrong td { color: var(--red); }
.worst-conf-high { color: var(--red); font-weight: 700; }
.worst-conf-low { color: var(--muted); }

/* Data quality */
.dq-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.dq-stat {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
  text-align: center;
}
.dq-stat-value { font-size: 22px; font-weight: 800; color: var(--accent); }
.dq-stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
.dq-warnings { display: grid; gap: 8px; }
.dq-warning-row {
  display: flex;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid var(--border);
  align-items: flex-start;
}
.dq-warning-row.dq-critical {
  background: rgba(239,68,68,0.08);
  border-color: rgba(239,68,68,0.3);
}
.dq-warning-row.dq-warning {
  background: rgba(240,160,105,0.08);
  border-color: rgba(240,160,105,0.3);
}
.dq-warning-row.dq-info {
  background: rgba(59,130,246,0.08);
  border-color: rgba(59,130,246,0.3);
}
.dq-warning-title { font-weight: 700; font-size: 13px; margin-bottom: 2px; }
.dq-warning-msg { font-size: 12px; color: var(--muted); }

/* CV table */
.cv-global {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 14px;
}
.cv-metric {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 16px;
  text-align: center;
}
.cv-metric-value { font-size: 20px; font-weight: 800; color: var(--accent); }
.cv-metric-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }

/* Cluster */
.cluster-container {
  position: relative;
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  align-items: flex-start;
}
.cluster-svg-wrap {
  position: relative;
  flex-shrink: 0;
}
.cluster-tooltip {
  display: none;
  position: absolute;
  background: rgba(20,20,35,0.95);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12px;
  box-shadow: var(--shadow);
  pointer-events: none;
  z-index: 100;
  white-space: nowrap;
  color: var(--fg);
}
.cluster-legend {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}
.cluster-legend-item { display: flex; align-items: center; gap: 6px; }
.cluster-legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}
.cluster-legend-label { color: var(--muted); font-weight: 600; }

/* Caveats */
.caveats {
  font-size: 12px;
  color: var(--muted);
  padding: 16px;
  background: var(--bg-subtle);
  border-radius: var(--radius);
  border: 1px solid var(--border-subtle);
  line-height: 1.7;
}

/* Empty state */
.empty { color: var(--muted); font-size: 13px; font-style: italic; }

/* Responsive */
@media (max-width: 768px) {
  .page { padding: 0 12px 40px; }
  .kpi-row { grid-template-columns: repeat(2, 1fr); }
  .kpi-value { font-size: 22px; }
  .cm-cell { min-width: 28px; font-size: 10px; padding: 4px 2px; }
  .cm-header, .cm-row-label { font-size: 9px; }
  .cluster-container { flex-direction: column; }
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
    """Interpolate from red through yellow to green."""
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

    ordered = ["accuracy", "f1_macro", "loss"]
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
        else:
            disp = f"{v:.1%}"
        parts.append(f'<div class="kpi-card{best}"><div class="kpi-value">{disp}</div>'
                     f'<div class="kpi-label">{_esc(key)}</div></div>')

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
                   else "badge-warning" if severity == "medium"
                   else "badge-ok")
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
        h += '<div class="card"><h3>Top Actions</h3><ul style="margin:0 0 0 18px;">'
        for a in actions[:5]:
            h += f"<li style='margin-bottom:4px;color:var(--accent);'>{_esc(a)}</li>"
        h += "</ul></div>"

    return h


# ─── Class Diagnostics ─────────────────────────────────────────────────────

def _class_diagnostics_table(diagnostics: list[dict[str, Any]]) -> str:
    if not diagnostics:
        return '<p class="empty">No per-class diagnostics.</p>'

    h = '<div class="table-wrap"><table><thead><tr>'
    h += "<th>#</th><th>Class</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th>"
    h += "<th>Support</th><th>Pred count</th><th>Severity</th>"
    h += "</tr></thead><tbody>"

    for i, d in enumerate(diagnostics[:30], 1):
        sev = d.get("severity", "ok")
        row_cls = f' class="sev-row-{sev}"' if sev in ("critical", "warning") else ""
        cid = _esc(str(d.get("class_id", "")))
        acc = d.get("accuracy", 0)
        prec = d.get("precision", 0)
        rec = d.get("recall", 0)
        f1 = d.get("f1", 0)
        support = d.get("support", 0)
        pred_count = d.get("pred_count", 0)

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

        h += f"<tr{row_cls}><td>{i}</td><td><strong>{cid}</strong></td>"
        h += _metric_cell(acc) + _metric_cell(prec) + _metric_cell(rec) + _metric_cell(f1)
        h += f"<td>{support}</td><td>{pred_count}</td><td>{sev_badge}</td></tr>"

    h += "</tbody></table></div>"
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
        h += '</div>'

        desc = f.get("description", "")
        if desc:
            h += f'<div class="finding-desc">{_esc(desc)}</div>'

        interp = f.get("interpretation", "")
        if interp:
            h += f'<div class="finding-desc" style="font-style:italic;border-left:2px solid var(--border);padding-left:10px;">{_esc(interp)}</div>'

        evidence = f.get("evidence", [])
        if evidence:
            h += '<div class="finding-evidence">'
            for ev in evidence[:5]:
                h += f'<span class="evidence-tag">{_esc(str(ev))}</span>'
            h += '</div>'

        h += '</div>'
    h += '</div>'
    return h


# ─── Failure Patterns ──────────────────────────────────────────────────────

def _failure_patterns_block(patterns: list[dict[str, Any]]) -> str:
    if not patterns:
        return ""
    h = '<div class="finding-grid">'
    for p in patterns[:15]:
        ptype = p.get("pattern_type", p.get("type", ""))
        desc = p.get("description", "")
        sev = p.get("severity", "medium")
        count = p.get("count", 0)
        sev_cls = f"sev-{sev}" if sev in ("critical", "high", "medium", "low") else ""

        h += f'<div class="finding-card {sev_cls}">'
        h += '<div class="finding-header">'
        h += f'<span class="finding-title">{_esc(desc)}</span>'
        if ptype:
            h += f' <span class="finding-type">{_esc(ptype)}</span>'
        if count:
            h += f' <span class="evidence-tag">count={count}</span>'
        h += '</div>'

        evidence = p.get("evidence", [])
        if evidence:
            h += '<div class="finding-evidence">'
            for ev in evidence[:4]:
                h += f'<span class="evidence-tag">{_esc(str(ev))}</span>'
            h += '</div>'

        meta = p.get("metadata", {})
        if meta:
            tags = [f"{k}={v}" for k, v in meta.items() if not isinstance(v, (dict, list))]
            if tags:
                h += '<div class="finding-evidence">'
                for t in tags[:4]:
                    h += f'<span class="evidence-tag">{_esc(t)}</span>'
                h += '</div>'

        h += '</div>'
    h += '</div>'
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
            h += '</div>'
            why = r.get("why", "")
            if why:
                truncated = why[:300] + "..." if len(why) > 300 else why
                h += f'<div class="rec-why">{_esc(truncated)}</div>'
            action = r.get("action", "")
            if action:
                h += f'<div class="rec-action">{_esc(action)}</div>'
            impact = r.get("expected_impact", "")
            if impact:
                h += (f'<div style="font-size:12px;color:var(--green);margin-top:6px;'
                      f'padding:4px 10px;background:rgba(34,197,94,0.08);border-radius:6px;'
                      f'border:1px solid rgba(34,197,94,0.2);">'
                      f'{_esc(impact)}</div>')
            reference = r.get("example_command", "")
            if reference and not reference.startswith("bnnr"):
                h += (f'<div style="font-size:10px;color:var(--muted);margin-top:4px;'
                      f'font-style:italic;">{_esc(reference)}</div>')
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


# ─── Confusion Matrix Heatmap ──────────────────────────────────────────────

def _confusion_heatmap(confusion: dict[str, Any]) -> str:
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
    h += f'<div class="cm-grid" style="grid-template-columns:60px repeat({n}, 1fr); max-width:{max(n*42+60, 300)}px; margin:0 auto;">'
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
            h += (f'<div class="cm-cell" style="background:{bg};color:{fg};" '
                  f'title="True {_esc(row_lbl)} → Pred {_esc(str(labels_list[j]) if j < len(labels_list) else str(j))}: {val}">'
                  f'{disp}</div>')

    h += '</div></div>'
    return h


# ─── Worst Predictions ────────────────────────────────────────────────────

def _worst_predictions_table(worst: list[dict[str, Any]], n: int = 25) -> str:
    if not worst:
        return '<p class="empty">No worst predictions.</p>'
    subset = worst[:n]
    h = '<div class="table-wrap"><table class="worst-table"><thead><tr>'
    h += "<th>#</th><th>Sample</th><th>True</th><th>Predicted</th><th>Confidence</th><th>Loss</th>"
    h += "</tr></thead><tbody>"
    for i, w in enumerate(subset, 1):
        is_wrong = w.get("true_label") != w.get("pred_label")
        row_cls = ' class="wrong"' if is_wrong else ""
        conf = w.get("confidence", 0)
        conf_cls = "worst-conf-high" if conf >= 0.8 else "worst-conf-low" if conf < 0.5 else ""
        h += f"<tr{row_cls}>"
        h += f"<td>{i}</td>"
        h += f'<td>#{w.get("index", "")}</td>'
        h += f'<td>{w.get("true_label", "")}</td>'
        h += f'<td>{w.get("pred_label", "")}</td>'
        h += f'<td class="{conf_cls}">{conf:.3f}</td>'
        h += f'<td>{w.get("loss", 0):.4f}</td>'
        h += "</tr>"
    h += "</tbody></table></div>"
    return h


# ─── XAI Insights ─────────────────────────────────────────────────────────

def _xai_overview_block(summary: dict[str, Any], per_class: dict[str, Any]) -> str:
    if not summary and not per_class:
        return '<p class="empty">No XAI analysis was performed.</p>'

    h = ""
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
            h += 'Scale 0&ndash;1. Higher means the model focuses on the object, not background/artifacts.'
            if total_xai_samples:
                h += f' Based on a probe set of {total_xai_samples} samples'
                if total_xai_samples < 100:
                    h += ' (small sample &mdash; treat as indicative, not definitive).'
                else:
                    h += '.'
            h += '</div></div>'

    if not per_class:
        return h

    items: list[tuple[float, str, int, list[str]]] = []
    for cid, data in per_class.items():
        if isinstance(data, dict):
            q = data.get("mean_quality")
            sc = data.get("sample_count", 0)
            flags = data.get("flags", [])
        else:
            continue
        if q is None:
            continue
        items.append((float(q), str(cid), sc, flags))

    if not items:
        return h

    items.sort(key=lambda t: t[0])
    h += '<div class="xai-class-grid">'
    for q, cid, sc, flags in items[:12]:
        q_color = _bar_color(q)
        h += f'<div class="xai-class-card" style="border-left:3px solid {q_color};">'
        h += '<div class="xai-class-header">'
        h += f'<span class="xai-class-name">Class {_esc(cid)}</span>'
        h += f'<span class="xai-quality-badge">{q:.3f}</span>'
        h += '</div>'
        h += (f'<div class="xai-quality-bar-track">'
              f'<div class="xai-quality-bar-fill" style="width:{q*100:.0f}%;background:{q_color};"></div>'
              f'</div>')
        h += f'<div class="xai-flags"><span class="xai-flag">samples: {sc}</span>'
        for fl in flags[:3]:
            h += f' <span class="xai-flag">{_esc(str(fl))}</span>'
        h += '</div></div>'
    h += '</div>'
    return h


def _xai_examples_block(examples_per_class: dict[str, Any]) -> str:
    if not examples_per_class:
        return ('<p class="empty">No XAI example overlays were generated. '
                'Run with XAI enabled and an output directory to see saliency overlays.</p>')
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
            border_style = "border-color:rgba(239,68,68,0.4);" if is_wrong else ""
            h += f'<div class="xai-example-card" style="{border_style}">'
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

def _data_quality_block(dq: dict[str, Any]) -> str:
    if not dq:
        return '<p class="empty">No data quality analysis was performed.</p>'

    scanned = dq.get("scanned_samples") or dq.get("total_scanned", 0)
    dup_pairs = dq.get("total_duplicate_pairs", 0)
    flagged = dq.get("total_flagged_images", 0)
    warnings = dq.get("warnings", [])
    summary_text = dq.get("summary", "")

    h = '<div class="dq-stats">'
    if scanned:
        h += f'<div class="dq-stat"><div class="dq-stat-value">{scanned:,}</div><div class="dq-stat-label">Scanned</div></div>'
    h += f'<div class="dq-stat"><div class="dq-stat-value">{dup_pairs}</div><div class="dq-stat-label">Duplicate Pairs</div></div>'
    h += f'<div class="dq-stat"><div class="dq-stat-value">{flagged}</div><div class="dq-stat-label">Flagged Images</div></div>'
    h += '</div>'

    if summary_text:
        h += f'<div class="card" style="font-size:13px;color:var(--muted);">{_esc(summary_text)}</div>'

    if warnings:
        h += '<div class="dq-warnings">'
        for w in warnings[:8]:
            msg = w.get("message", "")
            wtype = w.get("type", "info")
            count = w.get("count", 0)
            sev_cls = ("dq-critical" if wtype in ("critical", "error")
                       else "dq-warning" if wtype in ("warning", "high_variance", "tiny_images", "zero_variance")
                       else "dq-info")
            h += f'<div class="dq-warning-row {sev_cls}">'
            h += '<div>'
            h += f'<div class="dq-warning-title">{_esc(str(wtype))}'
            if count:
                h += f' <span class="evidence-tag">count={count}</span>'
            h += '</div>'
            if msg:
                h += f'<div class="dq-warning-msg">{_esc(str(msg))}</div>'
            h += '</div></div>'
        h += '</div>'
    elif not warnings and scanned:
        h += ('<div style="display:flex;align-items:center;padding:12px 16px;border-radius:8px;'
              'background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.3);'
              'color:var(--fg);font-weight:600;font-size:13px;">No issues detected.</div>')
    h += ('<div style="font-size:11px;color:var(--muted);margin-top:10px;font-style:italic;">'
          'Dataset health checks use perceptual hashing (dHash) for duplicate detection and '
          'statistical heuristics for image quality. False positives are possible, especially '
          'on small or synthetic datasets. Treat warnings as starting points for manual review.'
          '</div>')
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
    for key in ["mean_accuracy", "std_accuracy", "min_accuracy", "max_accuracy"]:
        if key in gm:
            label = key.replace("_", " ").title()
            h += (f'<div class="cv-metric"><div class="cv-metric-value">{gm[key]:.3f}</div>'
                  f'<div class="cv-metric-label">{_esc(label)}</div></div>')
    h += '</div>'

    if per_fold:
        h += '<div class="table-wrap"><table><thead><tr>'
        h += "<th>Fold</th><th>Accuracy</th><th>Support</th>"
        h += "</tr></thead><tbody>"
        for fm in per_fold:
            acc = fm.get("accuracy", 0)
            h += f'<tr><td>{fm.get("fold","")}</td>'
            h += f'<td>{acc:.3f} '
            h += (f'<span class="metric-bar-track"><span class="metric-bar-fill" '
                  f'style="width:{acc*100:.0f}%;background:{_bar_color(acc)};"></span></span></td>')
            h += f'<td>{fm.get("support",0)}</td></tr>'
        h += "</tbody></table></div>"
    return h


# ─── Cluster Visualization ────────────────────────────────────────────────

def _cluster_block(cluster_views: list[dict[str, Any]]) -> str:
    if not cluster_views:
        return '<p class="empty">No cluster visualization generated.</p>'

    view = cluster_views[0]
    points = view.get("points", [])
    if not points:
        return '<p class="empty">No cluster visualization data.</p>'

    unique_true = sorted(set(p.get("true_label", 0) for p in points))
    label_to_color = {lbl: _CLASS_COLORS[i % len(_CLASS_COLORS)] for i, lbl in enumerate(unique_true)}

    points_json = json.dumps([
        {
            "x": p.get("x", 0),
            "y": p.get("y", 0),
            "index": p.get("index", 0),
            "true_label": p.get("true_label", 0),
            "pred_label": p.get("pred_label", 0),
            "confidence": round(p.get("confidence", 0), 3),
        }
        for p in points
    ])
    colors_json = json.dumps({str(k): v for k, v in label_to_color.items()})

    legend = '<div class="cluster-legend" id="cluster-legend">'
    legend += ('<div style="font-size:10px;font-weight:700;color:var(--muted);'
               'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">True Class</div>')
    for lbl in unique_true[:20]:
        c = label_to_color.get(lbl, "#94a3b8")
        legend += (f'<div class="cluster-legend-item" data-class="{lbl}" '
                   f'onclick="toggleClusterClass(this,{lbl})" style="cursor:pointer;">'
                   f'<span class="cluster-legend-dot" style="background:{c};"></span>'
                   f'<span class="cluster-legend-label">Class {_esc(str(lbl))}</span></div>')
    legend += '</div>'

    h = '<div class="cluster-container" style="flex-direction:column;">'
    h += '<div style="display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;">'
    h += '<div class="cluster-svg-wrap" style="flex:1;min-width:0;">'
    h += '<canvas id="cluster-canvas" style="width:100%;height:520px;border-radius:12px;border:1px solid var(--border);background:#020617;cursor:crosshair;"></canvas>'
    h += '<div class="cluster-tooltip" id="cluster-tooltip"></div>'
    h += '</div>'
    h += legend
    h += '</div>'
    h += ('<div style="font-size:12px;color:var(--muted);margin-top:10px;">'
          '2D PCA projection of logits for worst predictions. '
          'Click a class in the legend to filter. Hover points for details.</div>')
    h += '</div>'
    h += f'<script>var _clusterPoints={points_json};var _clusterColors={colors_json};</script>'
    return h


def _cluster_js() -> str:
    return """
<script>
(function() {
  var canvas = document.getElementById('cluster-canvas');
  if (!canvas || typeof _clusterPoints === 'undefined') return;
  var ctx = canvas.getContext('2d');
  var tip = document.getElementById('cluster-tooltip');
  var points = _clusterPoints;
  var colors = _clusterColors;
  var hiddenClasses = {};
  var dpr = window.devicePixelRatio || 1;
  var PAD = 32;

  function resize() {
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function bounds() {
    var xs = points.map(function(p){return p.x;});
    var ys = points.map(function(p){return p.y;});
    return {
      minX: Math.min.apply(null, xs), maxX: Math.max.apply(null, xs),
      minY: Math.min.apply(null, ys), maxY: Math.max.apply(null, ys)
    };
  }

  function toScreen(px, py, b, w, h) {
    var sx = b.maxX - b.minX || 1;
    var sy = b.maxY - b.minY || 1;
    return {
      x: PAD + (px - b.minX) / sx * (w - 2*PAD),
      y: PAD + (b.maxY - py) / sy * (h - 2*PAD)
    };
  }

  function draw() {
    var rect = canvas.getBoundingClientRect();
    var w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(148,163,184,0.1)';
    ctx.lineWidth = 0.5;
    for (var i = 0; i <= 4; i++) {
      var gy = PAD + i * (h - 2*PAD) / 4;
      ctx.beginPath(); ctx.moveTo(PAD, gy); ctx.lineTo(w - PAD, gy); ctx.stroke();
      var gx = PAD + i * (w - 2*PAD) / 4;
      ctx.beginPath(); ctx.moveTo(gx, PAD); ctx.lineTo(gx, h - PAD); ctx.stroke();
    }

    var b = bounds();
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (hiddenClasses[p.true_label]) continue;
      var s = toScreen(p.x, p.y, b, w, h);
      var c = colors[String(p.true_label)] || '#94a3b8';
      var wrongPred = p.true_label !== p.pred_label;

      ctx.beginPath();
      ctx.arc(s.x, s.y, wrongPred ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.globalAlpha = 0.85;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = wrongPred ? 'rgba(239,68,68,0.6)' : 'rgba(255,255,255,0.15)';
      ctx.lineWidth = wrongPred ? 2 : 1;
      ctx.stroke();
    }
  }

  canvas.addEventListener('mousemove', function(e) {
    if (!tip) return;
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    var w = rect.width, h = rect.height;
    var b = bounds();
    var best = null, bestD = 18;
    for (var i = 0; i < points.length; i++) {
      var p = points[i];
      if (hiddenClasses[p.true_label]) continue;
      var s = toScreen(p.x, p.y, b, w, h);
      var d = Math.sqrt((s.x - mx) * (s.x - mx) + (s.y - my) * (s.y - my));
      if (d < bestD) { best = p; bestD = d; }
    }
    if (best) {
      var wrongTag = best.true_label !== best.pred_label
        ? '<span style="color:#ef4444;font-weight:700;"> WRONG</span>' : '';
      tip.innerHTML = '<strong>#' + best.index + '</strong>' + wrongTag + '<br>'
        + 'True: <strong>' + best.true_label + '</strong><br>'
        + 'Pred: <strong>' + best.pred_label + '</strong><br>'
        + 'Conf: ' + best.confidence;
      tip.style.left = (mx + 14) + 'px';
      tip.style.top = (my - 50) + 'px';
      tip.style.display = 'block';
    } else {
      tip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', function() {
    if (tip) tip.style.display = 'none';
  });

  window.toggleClusterClass = function(el, cls) {
    if (hiddenClasses[cls]) {
      delete hiddenClasses[cls];
      el.style.opacity = '1';
    } else {
      hiddenClasses[cls] = true;
      el.style.opacity = '0.3';
    }
    draw();
  };

  window.addEventListener('resize', resize);
  resize();
})();
</script>
"""


# ─── Main Renderer ────────────────────────────────────────────────────────

def render_analysis_html(report: Any) -> str:
    """Generate full HTML report aligned with BNNR dashboard design language."""

    schema_version = getattr(report, "schema_version", "0.2.0")
    metrics = getattr(report, "metrics", {}) or {}
    confusion = getattr(report, "confusion", {}) or {}
    summary = getattr(report, "executive_summary", {}) or {}
    diagnostics = getattr(report, "class_diagnostics", []) or []
    findings = getattr(report, "findings", []) or []
    xai_summary = getattr(report, "xai_quality_summary", {}) or {}
    xai_per_class = getattr(report, "xai_quality_per_class", {}) or {}
    xai_examples = getattr(report, "xai_examples_per_class", {}) or {}
    worst = getattr(report, "worst_predictions", []) or []
    data_quality = getattr(report, "data_quality_summary", {}) or getattr(report, "data_quality_result", {}) or {}
    cv_results = getattr(report, "cv_results", {}) or {}
    cluster_views = getattr(report, "cluster_views", []) or []
    recs_struct = getattr(report, "recommendations_structured", []) or []
    recs_legacy = getattr(report, "recommendations", []) or []
    true_dist = getattr(report, "true_distribution", {}) or {}

    num_classes = len(true_dist) if true_dist else (len(diagnostics) if diagnostics else 0)
    num_samples = sum(true_dist.values()) if true_dist else 0

    has_xai = bool(xai_summary or xai_per_class or xai_examples)
    has_dq = bool(data_quality)
    has_cv = bool(cv_results and cv_results.get("n_folds"))
    has_cluster = bool(cluster_views)

    sections: list[tuple[str, str]] = [
        ("exec", "Executive Summary"),
        ("metrics", "Overview"),
        ("findings", "Findings"),
        ("confusion", "Confusion Matrix"),
        ("diagnostics", "Class Diagnostics"),
    ]
    if has_xai:
        sections.append(("xai", "XAI Insights"))
    if worst:
        sections.append(("worst", "Worst Predictions"))
    if has_cluster:
        sections.append(("cluster", "Error Clusters"))
    if has_dq:
        sections.append(("dq", "Dataset Health"))
    if has_cv:
        sections.append(("cv", "Cross-Validation"))
    sections.append(("recs", "Recommendations"))
    sections.append(("caveats", "Method & Caveats"))

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
        '<div class="report-header">',
        "<h1>BNNR Analysis Report</h1>",
        '<div class="report-meta">',
        f'<span>v{_esc(schema_version)}</span>',
        '<span>Classification</span>',
        f'<span>{num_classes} classes &middot; {num_samples:,} samples</span>' if num_samples else "",
        '</div></div>',
        '<div class="two-col">',
        toc,
        '<div class="main-content">',
    ]

    # 1. Executive Summary
    lines.append(_section_open("exec", "Executive Summary"))
    lines.append(_executive_block(summary))
    lines.append(_section_close())

    # 2. Overview Metrics
    lines.append(_section_open("metrics", "Overview"))
    lines.append(_kpi_cards(metrics, num_classes, num_samples))
    lines.append(_section_close())

    # 3. Findings (the core diagnostic output)
    if findings:
        lines.append(_section_open("findings", "Findings", count=len(findings)))
        lines.append(_findings_block(findings))
        lines.append(_section_close())

    # 4. Confusion Matrix
    lines.append(_section_open("confusion", "Confusion Matrix"))
    lines.append(_confusion_heatmap(confusion))
    lines.append(_section_close())

    # 5. Class Diagnostics
    lines.append(_section_open("diagnostics", "Class Diagnostics", count=len(diagnostics)))
    lines.append(_class_diagnostics_table(diagnostics))
    lines.append(_section_close())

    # 6. XAI Insights (only if data exists)
    if has_xai:
        lines.append(_section_open("xai", "XAI Insights"))
        lines.append(_xai_overview_block(xai_summary, xai_per_class))
        lines.append(_xai_examples_block(xai_examples))
        lines.append(_section_close())

    # 7. Worst Predictions
    if worst:
        lines.append(_section_open("worst", "Worst Predictions", count=min(len(worst), 25)))
        lines.append(_worst_predictions_table(worst))
        lines.append(_section_close())

    # 8. Error Clusters (only if generated)
    if has_cluster:
        lines.append(_section_open("cluster", "Error Clusters"))
        lines.append(_cluster_block(cluster_views))
        lines.append(_section_close())

    # 9. Dataset Health (only if analysis ran)
    if has_dq:
        lines.append(_section_open("dq", "Dataset Health"))
        lines.append(_data_quality_block(data_quality))
        lines.append(_section_close())

    # 10. Cross-Validation (only if folds > 1)
    if has_cv:
        lines.append(_section_open("cv", "Cross-Validation"))
        lines.append(_cv_block(cv_results))
        lines.append(_section_close())

    # 11. Recommendations (capped at 5)
    rec_count = min(len(recs_struct), 5) if recs_struct else min(len(recs_legacy), 5)
    lines.append(_section_open("recs", "Recommendations", count=rec_count))
    lines.append(_rec_block(recs_struct, recs_legacy))
    lines.append(_section_close())

    # 12. Method & Caveats
    lines.append(_section_open("caveats", "Method & Caveats"))
    lines.append(
        '<div class="caveats">'
        "<p><strong>How to read this report.</strong> "
        "Findings are tagged by confidence level: "
        '<span class="badge badge-ok" style="font-size:10px;">Observed</span> = '
        "directly measured from data; "
        '<span class="badge badge-warning" style="font-size:10px;">Likely</span> = '
        "strong heuristic signal; "
        '<span class="badge badge-critical" style="font-size:10px;">Suspected</span> = '
        "requires manual verification.</p>"
        "<p>XAI quality score (0&ndash;1) measures whether saliency maps focus on the object "
        "of interest. Values are computed from a small probe set and should be treated as "
        "indicative rather than definitive.</p>"
        "<p>Cross-validation (when enabled) splits the validation set into k folds and "
        "re-evaluates the same checkpoint. It measures metric stability across data subsets, "
        "not generalization from retraining.</p>"
        "<p>Dataset health checks use perceptual hashing and statistical heuristics. "
        "False positives are possible on small or synthetic datasets.</p>"
        "<p>For training-loop optimization with saliency-guided augmentation, use "
        "<strong>bnnr train</strong> with ICD/AICD.</p>"
        "</div>"
    )
    lines.append(_section_close())

    lines.append('</div></div>')
    lines.append('</div>')

    if has_cluster:
        lines.append(_cluster_js())

    lines.append("</body></html>")
    return "\n".join(lines)
