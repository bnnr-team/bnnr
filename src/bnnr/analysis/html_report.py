"""HTML report renderer for bnnr analyze — dashboard-aligned design."""

from __future__ import annotations

import json
from typing import Any


def _css() -> str:
    """BNNR dashboard-aligned CSS (dark theme tokens)."""
    return """
:root {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  --radius: 12px;
  --bg: #080810;
  --bg-subtle: #0d0d18;
  --fg: #f1f5f9;
  --muted: #94a3b8;
  --card: #131322;
  --border: rgba(240, 160, 105, 0.18);
  --accent: #f0a069;
  --green: #22c55e;
  --red: #ef4444;
  --glow: 0 0 20px rgba(240, 160, 105, 0.08);
}
* { box-sizing: border-box; margin: 0; }
body { background: var(--bg); color: var(--fg); font-size: 14px; line-height: 1.5; padding: 20px; max-width: 1200px; margin: 0 auto; }
.report-header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
.report-header h1 { font-size: 24px; font-weight: 800; color: var(--accent); margin-bottom: 8px; }
.report-meta { font-size: 12px; color: var(--muted); }
.section { margin-bottom: 28px; }
.section h2 { font-size: 16px; font-weight: 700; color: var(--accent); margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; margin-bottom: 14px; box-shadow: var(--glow); }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }
.kpi-card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; text-align: center; }
.kpi-value { font-size: 28px; font-weight: 800; color: var(--accent); line-height: 1.1; }
.kpi-label { font-size: 11px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.badge { display: inline-block; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 999px; margin-right: 8px; margin-bottom: 8px; }
.badge-ok { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
.badge-warning { background: rgba(240, 160, 105, 0.15); color: #f5b888; }
.badge-critical { background: rgba(239, 68, 68, 0.15); color: #f87171; }
.findings-list, .rec-list { display: grid; gap: 10px; }
.finding-card, .rec-card { background: var(--bg-subtle); border: 1px solid var(--border); border-radius: 10px; padding: 14px; border-left: 3px solid var(--accent); }
.finding-card.critical { border-left-color: var(--red); }
.finding-card.warning { border-left-color: var(--accent); }
.finding-title, .rec-title { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
.finding-desc, .rec-why { font-size: 13px; color: var(--muted); margin-bottom: 6px; line-height: 1.5; }
.rec-action { font-size: 13px; color: var(--accent); margin-top: 6px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { font-weight: 600; color: var(--muted); font-size: 11px; text-transform: uppercase; }
.worst-list { display: grid; gap: 8px; }
.worst-item { display: grid; grid-template-columns: auto 1fr auto auto; gap: 12px; align-items: center; padding: 10px; background: var(--bg-subtle); border-radius: 8px; font-size: 13px; }
.caveats { font-size: 12px; color: var(--muted); padding: 16px; background: var(--bg-subtle); border-radius: var(--radius); }
"""


def _section(title: str, content: str) -> str:
    return f'<div class="section"><h2>{title}</h2>{content}</div>'


def _kpi_cards(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    parts = []
    for key in ["accuracy", "f1_macro", "loss"]:
        if key not in metrics:
            continue
        v = metrics[key]
        if isinstance(v, float):
            if key == "loss":
                parts.append(f'<div class="kpi-card"><div class="kpi-value">{v:.4f}</div><div class="kpi-label">{key}</div></div>')
            else:
                parts.append(f'<div class="kpi-card"><div class="kpi-value">{v:.1%}</div><div class="kpi-label">{key}</div></div>')
        else:
            parts.append(f'<div class="kpi-card"><div class="kpi-value">{v}</div><div class="kpi-label">{key}</div></div>')
    if not parts:
        return ""
    return '<div class="kpi-row">' + "".join(parts) + "</div>"


def _executive_block(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    status = summary.get("health_status", "unknown")
    score = summary.get("health_score", 0)
    badge_class = "badge-ok" if status == "ok" else "badge-warning" if status == "warning" else "badge-critical"
    html = f'<div class="card"><span class="badge {badge_class}">{status.upper()}</span>'
    html += f' <strong>Score:</strong> {score:.0%}'
    if summary.get("severity"):
        html += f' &middot; Severity: {summary["severity"]}'
    html += "</div>"
    key = summary.get("key_findings", [])
    if key:
        html += "<div class='card'><strong>Key findings</strong><ul style='margin:10px 0 0 20px;'>"
        for k in key[:5]:
            html += f"<li>{_esc(k)}</li>"
        html += "</ul></div>"
    actions = summary.get("top_actions", [])
    if actions:
        html += "<div class='card'><strong>Top actions</strong><ul style='margin:10px 0 0 20px;'>"
        for a in actions[:5]:
            html += f"<li>{_esc(a)}</li>"
        html += "</ul></div>"
    return html


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _findings_block(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "<p class='muted'>No structured findings.</p>"
    html = "<div class='findings-list'>"
    for f in findings[:15]:
        sev = f.get("severity", "medium")
        cls = "critical" if sev == "critical" else "warning" if sev in ("high", "warning") else ""
        html += f"<div class='finding-card {cls}'>"
        html += f"<div class='finding-title'>{_esc(f.get('title', ''))}</div>"
        html += f"<div class='finding-desc'>{_esc(f.get('description', ''))}</div>"
        if f.get("recommended_action"):
            html += f"<div class='rec-action'>{_esc(f['recommended_action'])}</div>"
        html += "</div>"
    html += "</div>"
    return html


def _rec_block(recs: list[dict[str, Any]], fallback: list[str]) -> str:
    if recs:
        html = "<div class='rec-list'>"
        for r in recs[:15]:
            html += "<div class='rec-card'>"
            html += f"<div class='rec-title'>{_esc(r.get('title', ''))}</div>"
            html += f"<div class='rec-why'>{_esc(r.get('why', ''))}</div>"
            html += f"<div class='rec-action'>{_esc(r.get('action', ''))}</div>"
            html += "</div>"
        html += "</div>"
        return html
    if fallback:
        return "<ul style='margin-left:20px;'>" + "".join(f"<li>{_esc(r)}</li>" for r in fallback) + "</ul>"
    return "<p class='muted'>No recommendations.</p>"


def _class_diagnostics_table(diagnostics: list[dict[str, Any]]) -> str:
    if not diagnostics:
        return "<p class='muted'>No per-class diagnostics.</p>"
    html = "<div class='table-wrap'><table><thead><tr><th>Class</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th><th>Pred count</th><th>Severity</th></tr></thead><tbody>"
    for d in diagnostics[:20]:
        html += f"<tr><td>{_esc(str(d.get('class_id', '')))}</td><td>{d.get('accuracy', 0):.2%}</td><td>{d.get('precision', 0):.2%}</td><td>{d.get('recall', 0):.2%}</td><td>{d.get('f1', 0):.2%}</td><td>{d.get('support', 0)}</td><td>{d.get('pred_count', 0)}</td><td>{_esc(d.get('severity', ''))}</td></tr>"
    html += "</tbody></table></div>"
    return html


def _worst_predictions_table(worst: list[dict[str, Any]], n: int = 20) -> str:
    if not worst:
        return "<p class='muted'>No worst predictions.</p>"
    subset = worst[:n]
    html = "<div class='table-wrap'><table><thead><tr><th>#</th><th>Index</th><th>True</th><th>Pred</th><th>Confidence</th><th>Loss</th></tr></thead><tbody>"
    for i, w in enumerate(subset, 1):
        html += f"<tr><td>{i}</td><td>{w.get('index', '')}</td><td>{w.get('true_label', '')}</td><td>{w.get('pred_label', '')}</td><td>{w.get('confidence', 0):.3f}</td><td>{w.get('loss', 0):.4f}</td></tr>"
    html += "</tbody></table></div>"
    return html


def _confusion_preview(confusion: dict[str, Any]) -> str:
    if not confusion or not confusion.get("matrix"):
        return "<p class='muted'>No confusion matrix.</p>"
    return "<div class='card'><pre style='font-size:11px; overflow:auto; max-height:320px;'>" + _esc(json.dumps(confusion, indent=2)) + "</pre></div>"


def render_analysis_html(report: Any) -> str:
    """Generate full HTML report aligned with BNNR dashboard style."""
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>BNNR Analysis Report</title>",
        "<style>" + _css() + "</style>",
        "</head><body>",
        "<div class='report-header'>",
        "<h1>BNNR Analysis Report</h1>",
        "<div class='report-meta'>Schema v" + getattr(report, "schema_version", "0.2.0") + "</div>",
        "</div>",
    ]

    # Executive summary
    summary = getattr(report, "executive_summary", None) or {}
    lines.append(_section("Executive summary", _executive_block(summary)))

    # KPIs
    lines.append(_section("Overview metrics", _kpi_cards(getattr(report, "metrics", {}))))

    # Class diagnostics
    diag = getattr(report, "class_diagnostics", None) or []
    lines.append(_section("Class diagnostics", _class_diagnostics_table(diag)))

    # Findings
    findings = getattr(report, "findings", None) or []
    lines.append(_section("Findings", _findings_block(findings)))

    # Failure patterns (extended)
    patterns = getattr(report, "failure_patterns_extended", None) or getattr(report, "failure_patterns", None) or []
    if patterns:
        lines.append(_section("Failure patterns", "<div class='card'><pre style='font-size:12px;'>" + _esc(json.dumps(patterns[:20], indent=2)) + "</pre></div>"))

    # XAI
    xai = getattr(report, "xai_quality_summary", None) or {}
    if xai:
        lines.append(_section("XAI quality", "<div class='card'><pre style='font-size:12px;'>" + _esc(json.dumps(xai, indent=2)) + "</pre></div>"))
    xai_diag = getattr(report, "xai_diagnoses", None) or {}
    if xai_diag:
        lines.append(_section("XAI per-class", "<div class='card'><pre style='font-size:11px; max-height:300px; overflow:auto;'>" + _esc(json.dumps(xai_diag, indent=2)) + "</pre></div>"))

    # Worst predictions
    worst = getattr(report, "worst_predictions", None) or []
    lines.append(_section("Worst predictions", _worst_predictions_table(worst)))

    # Confusion
    lines.append(_section("Confusion matrix", _confusion_preview(getattr(report, "confusion", {}))))

    # Recommendations
    recs_struct = getattr(report, "recommendations_structured", None) or []
    recs_legacy = getattr(report, "recommendations", None) or []
    lines.append(_section("Recommendations", _rec_block(recs_struct, recs_legacy)))

    # Caveats
    lines.append(_section("Method & caveats", "<div class='caveats'>This report is generated by <strong>bnnr analyze</strong>. XAI quality score is 0–1; low values suggest background or artefact focus. Recommendations are derived from findings and metrics; apply in your own environment. For full training loop use <strong>bnnr train</strong>.</div>"))

    lines.append("</body></html>")
    return "\n".join(lines)
