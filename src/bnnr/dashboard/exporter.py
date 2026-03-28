"""Static dashboard export helpers and offline report snapshot rendering."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from bnnr.events import load_events, replay_events

# Directory containing bundled static assets (logos, etc.)
_STATIC_DIR = Path(__file__).parent / "static"


def export_dashboard_snapshot(run_dir: Path, out_dir: Path, frontend_dist: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events_file = run_dir / "events.jsonl"
    if not events_file.exists():
        raise FileNotFoundError(f"events.jsonl not found in run directory: {run_dir}")

    # Build state from events
    state = replay_events(load_events(events_file))
    # Always generate a single-file offline dashboard report.
    # This avoids blank-page issues from file:// + module scripts and gives
    # deterministic export UX for end users.
    _copy_logos_to(out_dir)
    (out_dir / "index.html").write_text(
        _standalone_report_html(state, run_dir.name),
        encoding="utf-8",
    )

    # Ensure logos are present in the output directory (for file:// usage)
    _copy_logos_to(out_dir)

    # Write state.json for offline mode (also used when served via HTTP)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(events_file, data_dir / "events.jsonl")
    (data_dir / "state.json").write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    report_json = run_dir / "report.json"
    if report_json.exists():
        shutil.copy2(report_json, data_dir / "report.json")

    # Copy artifacts
    artifacts_src = run_dir / "artifacts"
    artifacts_dst = out_dir / "artifacts"
    if artifacts_src.exists():
        shutil.copytree(artifacts_src, artifacts_dst, dirs_exist_ok=True)
    else:
        artifacts_dst.mkdir(parents=True, exist_ok=True)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "mode": "offline_replay",
                "run_dir": str(run_dir),
                "has_frontend": False,
                "files": {
                    "events": "data/events.jsonl",
                    "state": "data/state.json",
                    "report": "data/report.json" if report_json.exists() else None,
                    "artifacts": "artifacts/",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def _copy_logos_to(dest: Path) -> None:
    """Copy logo PNGs from the bundled static dir into *dest* (if they exist)."""
    for name in ("logo_light.png", "logo_dark.png"):
        src = _STATIC_DIR / name
        dst = dest / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _standalone_report_html(state: dict, run_name: str) -> str:
    """Generate a professional standalone HTML report when no frontend build is available."""

    # Extract key data
    mt = state.get("metrics_timeline", [])
    decisions = state.get("decision_history", [])
    selected_path = state.get("selected_path", ["baseline"])
    branches = state.get("branches", {})
    state.get("xai", [])  # reserved for future use
    sample_timelines = state.get("sample_timelines", {})
    task = state.get("task")
    if not isinstance(task, str):
        run_cfg = state.get("run", {}).get("config", {})
        task = run_cfg.get("task", "classification") if isinstance(run_cfg, dict) else "classification"
    is_multilabel = task == "multilabel"
    is_detection = task == "detection"

    # Best metrics — task-aware primary/secondary metric keys
    if is_detection:
        primary_key = "map_50"
        secondary_key = "map_50_95"
        primary_label = "Best mAP@0.5 ★"
        secondary_label = "Best mAP@[.5:.95] ★"
        final_label = "Final mAP@0.5"
        primary_chart_label = "mAP@0.5 (%)"
        secondary_chart_label = "mAP@[.5:.95] (%)"
        decision_metric_label = "mAP@0.5"
    elif is_multilabel:
        primary_key = "f1_samples"
        secondary_key = "f1_macro"
        primary_label = "Best F1 Samples ★"
        secondary_label = "Best F1 Macro ★"
        final_label = "Final F1 Samples"
        primary_chart_label = "F1 Samples (%)"
        secondary_chart_label = "F1 Macro (%)"
        decision_metric_label = "F1s"
    else:
        primary_key = "accuracy"
        secondary_key = "f1_macro"
        primary_label = "Best Accuracy ★"
        secondary_label = "Best F1 ★"
        final_label = "Final Accuracy"
        primary_chart_label = "Accuracy (%)"
        secondary_chart_label = "F1 (%)"
        decision_metric_label = "Acc"

    best_primary = max((r.get(primary_key, 0) for r in mt), default=0) if mt else 0
    best_secondary = max((r.get(secondary_key, 0) for r in mt), default=0) if mt else 0
    current = mt[-1] if mt else {}
    baseline_entries = [r for r in mt if r.get("branch") == "baseline"]
    baseline_final_primary = baseline_entries[-1].get(primary_key, 0) if baseline_entries else 0

    # Decision rows
    decision_rows = ""
    for d in decisions:
        results = d.get("results", {})
        winner = d.get("selected_branch_label", "")
        winner_metric = results.get(winner, {}).get(primary_key, 0)
        base_metric = d.get(f"baseline_{primary_key}", d.get("baseline_accuracy", 0)) or 0
        delta = winner_metric - base_metric
        delta_class = "pos" if delta > 0 else "neg"
        decision_rows += f"""<tr>
            <td>{d.get('iteration', '?')}</td>
            <td><strong>{winner}</strong></td>
            <td>{winner_metric * 100:.1f}%</td>
            <td class="{delta_class}">{'+' if delta > 0 else ''}{delta * 100:.2f}pp</td>
            <td>{len(results)} evaluated</td>
            <td class="muted">{d.get('decision_reason', '')}</td>
        </tr>"""

    # Metrics chart data (JSON for inline chart)
    chart_data = [
        {
            "label": f"it{r.get('iteration', 0)}/e{r.get('epoch', 0)}",
            "primary": round(r.get(primary_key, 0) * 100, 2),
            "secondary": round(r.get(secondary_key, 0) * 100, 2),
            "loss": round(r.get("loss", 0), 4),
            "branch": r.get("branch", ""),
        }
        for r in mt
    ]

    # Probe samples info
    probe_set = state.get("probe_set", [])
    probe_count = len(probe_set)
    sample_count = sum(len(v) for v in sample_timelines.values())

    # Path display
    path_html = " → ".join(
        f'<span class="chip">{p.split(":")[-1] if ":" in p else p}</span>'
        for p in selected_path
    )

    # Per-class preview table (last available snapshot per class)
    per_class_rows = ""
    per_class_tl = state.get("per_class_timeline", {})
    class_names = state.get("dataset_profile", {}).get("class_names", [])
    if not isinstance(class_names, list) or not class_names:
        run_class_names = state.get("run", {}).get("class_names", [])
        class_names = run_class_names if isinstance(run_class_names, list) else []
    if isinstance(per_class_tl, dict) and per_class_tl:
        def _class_sort_key(raw: object) -> tuple[int, int | str]:
            if isinstance(raw, int):
                return (0, raw)
            if isinstance(raw, str):
                try:
                    return (0, int(raw))
                except ValueError:
                    return (1, raw)
            return (1, str(raw))

        for class_id, rows in sorted(per_class_tl.items(), key=lambda item: _class_sort_key(item[0])):
            if not isinstance(rows, list) or not rows:
                continue
            latest = rows[-1]
            if not isinstance(latest, dict):
                continue

            class_name = str(class_id)
            try:
                class_idx = int(class_id)
                if 0 <= class_idx < len(class_names):
                    class_name = str(class_names[class_idx])
            except (TypeError, ValueError):
                pass

            support = latest.get("support")
            ap = latest.get("ap")
            acc = latest.get("accuracy")
            if is_detection:
                main_value = f"{ap * 100:.1f}%" if isinstance(ap, (int, float)) else "—"
            else:
                main_value = (
                    f"{acc * 100:.1f}%"
                    if isinstance(acc, (int, float))
                    else (f"{ap * 100:.1f}%" if isinstance(ap, (int, float)) else "—")
                )
            support_value = str(support) if support is not None else "—"
            per_class_rows += f"<tr><td>{html.escape(class_name)}</td><td>{main_value}</td><td>{support_value}</td></tr>"

    # Build rich sample cards from sample timelines (original/augmented/xai)
    sample_cards: list[str] = []
    sample_timelines = state.get("sample_timelines", {})
    if isinstance(sample_timelines, dict):
        for sample_id, timeline in sample_timelines.items():
            if not isinstance(timeline, list) or not timeline:
                continue
            entry = timeline[-1]
            if not isinstance(entry, dict):
                continue
            artifacts = entry.get("artifacts", {}) or {}
            if not isinstance(artifacts, dict):
                artifacts = {}
            original = artifacts.get("original")
            augmented = artifacts.get("augmented")
            xai = artifacts.get("xai")
            xai_gt = artifacts.get("xai_gt")
            xai_saliency = artifacts.get("xai_saliency")
            xai_pred = artifacts.get("xai_pred")
            has_split_xai = (
                isinstance(xai_gt, str)
                and xai_gt
                and isinstance(xai_saliency, str)
                and xai_saliency
                and isinstance(xai_pred, str)
                and xai_pred
            )
            if not (original or augmented or xai or has_split_xai):
                continue

            def _img(src: object, label: str) -> str:
                if not isinstance(src, str) or not src:
                    return ""
                esc_src = html.escape(src)
                return (
                    "<div class='img-block'>"
                    f"<img src='./{esc_src}' alt='{html.escape(label)}'/>"
                    f"<div class='caption'>{html.escape(label)}</div>"
                    "</div>"
                )

            def _detection_xai_panels_split(gt: str, sal: str, pred: str) -> str:
                return (
                    "<div class='xai-triptych-row'>"
                    "<div class='xai-triptych-title'>XAI Panels</div>"
                    "<div class='xai-triptych'>"
                    "<div class='img-block'>"
                    f"<img src='./{html.escape(gt)}' alt='XAI GT' class='xai-panel xai-panel-split'/>"
                    "<div class='caption'>GT</div>"
                    "</div>"
                    "<div class='img-block'>"
                    f"<img src='./{html.escape(sal)}' alt='XAI Saliency' class='xai-panel xai-panel-split'/>"
                    "<div class='caption'>Saliency</div>"
                    "</div>"
                    "<div class='img-block'>"
                    f"<img src='./{html.escape(pred)}' alt='XAI Pred+Saliency' class='xai-panel xai-panel-split'/>"
                    "<div class='caption'>Pred + Saliency</div>"
                    "</div>"
                    "</div>"
                    "</div>"
                )

            def _detection_xai_panels_legacy(src: object) -> str:
                if not isinstance(src, str) or not src:
                    return ""
                esc_src = html.escape(src)
                return (
                    "<div class='xai-triptych-row'>"
                    "<div class='xai-triptych-title'>XAI Panels</div>"
                    "<div class='xai-triptych'>"
                    "<div class='img-block'>"
                    f"<img src='./{esc_src}' alt='XAI GT' class='xai-panel xai-panel-gt'/>"
                    "<div class='caption'>GT</div>"
                    "</div>"
                    "<div class='img-block'>"
                    f"<img src='./{esc_src}' alt='XAI Saliency' class='xai-panel xai-panel-saliency'/>"
                    "<div class='caption'>Saliency</div>"
                    "</div>"
                    "<div class='img-block'>"
                    f"<img src='./{esc_src}' alt='XAI Pred+Saliency' class='xai-panel xai-panel-pred'/>"
                    "<div class='caption'>Pred + Saliency</div>"
                    "</div>"
                    "</div>"
                    "</div>"
                )

            info = (
                f"sample={html.escape(str(sample_id))} · "
                f"branch={html.escape(str(entry.get('branch', '')))} · "
                f"it={html.escape(str(entry.get('iteration', '')))} · "
                f"e={html.escape(str(entry.get('epoch', '')))} · "
                f"true={html.escape(str(entry.get('true_class', '')))} · "
                f"pred={html.escape(str(entry.get('predicted_class', '')))}"
            )
            conf = entry.get("confidence")
            conf_txt = f"{conf * 100:.1f}%" if isinstance(conf, (int, float)) else "—"
            main_blocks = f"{_img(original, 'Original')}{_img(augmented, 'Augmented')}"
            if not is_detection:
                main_blocks += _img(xai, "XAI")
            if is_detection:
                if has_split_xai:
                    detection_panels = _detection_xai_panels_split(
                        str(xai_gt), str(xai_saliency), str(xai_pred),
                    )
                else:
                    detection_panels = _detection_xai_panels_legacy(xai)
            else:
                detection_panels = ""
            sample_cards.append(
                "<div class='sample-card'>"
                f"<div class='sample-title'>{info}</div>"
                f"<div class='sample-meta'>confidence={conf_txt}</div>"
                "<div class='img-row'>"
                f"{main_blocks}"
                "</div>"
                f"{detection_panels}"
                "</div>"
            )
            if len(sample_cards) >= 18:
                break

    # Optional textual XAI insights
    xai_insights = state.get("xai_insights_timeline", [])
    insight_rows = ""
    if isinstance(xai_insights, list):
        for row in xai_insights[-12:]:
            if not isinstance(row, dict):
                continue
            cls = row.get("class_name") or row.get("class") or "?"
            txt = row.get("insight") or row.get("summary") or row.get("text") or ""
            if txt:
                insight_rows += (
                    "<tr>"
                    f"<td>{html.escape(str(cls))}</td>"
                    f"<td>{html.escape(str(txt))}</td>"
                    "</tr>"
                )

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>BNNR Report — {run_name}</title>
    <style>
        :root {{
            --bg: #f8fafc; --card: #ffffff; --fg: #0f172a; --muted: #64748b;
            --border: #e2e8f0; --accent: #2563eb; --green: #16a34a; --red: #dc2626;
        }}
        * {{ box-sizing: border-box; margin: 0; }}
        body {{ background: var(--bg); color: var(--fg); font-family: Inter, system-ui, -apple-system, sans-serif; font-size: 14px; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px 60px; }}
        .header {{ text-align: center; padding: 32px 0 24px; border-bottom: 2px solid var(--border); margin-bottom: 24px; }}
        .header .logo {{ height: 64px; margin-bottom: 12px; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; }}
        .header h1 {{ font-size: 28px; font-weight: 800; letter-spacing: -0.5px; }}
        .header .run-id {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
        .header .badge {{ display: inline-block; background: #fde8d0; color: #8b4a25; padding: 2px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 8px; }}
        .section {{ margin-bottom: 24px; }}
        .section h2 {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: var(--fg); display: flex; align-items: center; gap: 8px; }}
        .section h2::before {{ content: ''; width: 4px; height: 20px; background: var(--accent); border-radius: 2px; }}
        .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 14px; }}
        .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 24px; }}
        .kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; text-align: center; }}
        .kpi.best {{ border-color: var(--green); background: linear-gradient(135deg, rgba(22,163,74,0.06), rgba(22,163,74,0.01)); }}
        .kpi-value {{ font-size: 32px; font-weight: 800; color: var(--accent); line-height: 1.1; }}
        .kpi-label {{ font-size: 10px; color: var(--muted); margin-top: 4px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }}
        .path-row {{ display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }}
        .chip {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 4px 14px; font-size: 13px; font-weight: 600; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--border); font-size: 11px; text-transform: uppercase; color: var(--muted); font-weight: 700; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
        .pos {{ color: var(--green); font-weight: 700; }}
        .neg {{ color: var(--red); font-weight: 700; }}
        .muted {{ color: var(--muted); }}
        .chart-wrap {{ width: 100%; overflow-x: auto; }}
        .chart-svg {{ width: 100%; min-width: 720px; height: 280px; border: 1px solid var(--border); border-radius: 8px; background: #fff; }}
        .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); }}
        .sample-grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
        .sample-card {{ border: 1px solid var(--border); border-radius: 10px; padding: 10px; background: #fff; }}
        .sample-title {{ font-size: 12px; font-weight: 700; margin-bottom: 2px; }}
        .sample-meta {{ font-size: 11px; color: var(--muted); margin-bottom: 8px; }}
        .img-row {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 8px; }}
        .img-block img {{ width: 100%; border-radius: 8px; border: 1px solid var(--border); display: block; }}
        .xai-triptych-row {{ margin-top: 8px; }}
        .xai-triptych-title {{ font-size: 11px; font-weight: 700; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.4px; }}
        .xai-triptych {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 8px; }}
        .xai-panel {{ aspect-ratio: 1 / 1; object-fit: cover; }}
        .xai-panel-split {{ object-position: center center; }}
        .xai-panel-gt {{ object-position: left center; }}
        .xai-panel-saliency {{ object-position: center center; }}
        .xai-panel-pred {{ object-position: right center; }}
        .caption {{ font-size: 11px; color: var(--muted); margin-top: 3px; text-align: center; }}
        .legend {{ display:flex; gap:12px; font-size:12px; color:var(--muted); margin-top:8px; }}
        .legend .dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; margin-right:6px; }}
        @media (max-width: 768px) {{
            .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
            .img-row {{ grid-template-columns: 1fr; }}
            .xai-triptych {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="./logo_light.png" alt="BNNR" class="logo" onerror="this.style.display='none'"/>
            <h1>BNNR Training Report</h1>
            <div class="run-id">{run_name}</div>
            <div class="badge">Offline Report</div>
        </div>

        <!-- KPI Overview -->
        <div class="kpi-row">
            <div class="kpi best">
                <div class="kpi-value">{best_primary * 100:.1f}%</div>
                <div class="kpi-label">{primary_label}</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{current.get(primary_key, 0) * 100:.1f}%</div>
                <div class="kpi-label">{final_label}</div>
            </div>
            <div class="kpi best">
                <div class="kpi-value">{best_secondary * 100:.1f}%</div>
                <div class="kpi-label">{secondary_label}</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{len(decisions)}</div>
                <div class="kpi-label">Decisions Made</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{len(branches)}</div>
                <div class="kpi-label">Branches Evaluated</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{len(mt)}</div>
                <div class="kpi-label">Total Epochs</div>
            </div>
        </div>

        <!-- Selected Path -->
        <div class="section">
            <h2>Selected Path</h2>
            <div class="card">
                <div class="path-row">{path_html}</div>
                <p class="muted" style="margin-top: 10px; font-size: 12px;">
                    Baseline {primary_key}: {baseline_final_primary * 100:.1f}% →
                    Best {primary_key}: {best_primary * 100:.1f}%
                    (Δ {'+' if best_primary > baseline_final_primary else ''}{(best_primary - baseline_final_primary) * 100:.2f}pp)
                </p>
            </div>
        </div>

        <!-- Decision History -->
        {f'''<div class="section">
            <h2>Decision History</h2>
            <div class="card" style="overflow-x: auto;">
                <table>
                    <thead><tr>
                        <th>Iter</th><th>Winner</th><th>{decision_metric_label}</th><th>Delta</th><th>Candidates</th><th>Reason</th>
                    </tr></thead>
                    <tbody>{decision_rows}</tbody>
                </table>
            </div>
        </div>''' if decisions else ''}

        <!-- Metrics Chart -->
        <div class="section">
            <h2>Training Metrics</h2>
            <div class="card">
                <div class="chart-wrap">
                    <svg id="metricSvg" class="chart-svg" viewBox="0 0 900 280" preserveAspectRatio="none"></svg>
                </div>
                <div class="legend">
                    <span><span class="dot" style="background:#2563eb"></span>{primary_chart_label}</span>
                    <span><span class="dot" style="background:#16a34a"></span>{secondary_chart_label}</span>
                    <span><span class="dot" style="background:#dc2626"></span>Loss (normalized)</span>
                </div>
            </div>
        </div>

        <!-- Per-class -->
        {f'''<div class="section">
            <h2>Per-Class Snapshot</h2>
            <div class="card" style="overflow-x:auto;">
                <table>
                    <thead><tr><th>Class</th><th>Main metric</th><th>Support</th></tr></thead>
                    <tbody>{per_class_rows}</tbody>
                </table>
            </div>
        </div>''' if per_class_rows else ''}

        <!-- XAI Insights -->
        {f'''<div class="section">
            <h2>XAI Insights</h2>
            <div class="card" style="overflow-x:auto;">
                <table>
                    <thead><tr><th>Class</th><th>Insight</th></tr></thead>
                    <tbody>{insight_rows}</tbody>
                </table>
            </div>
        </div>''' if insight_rows else ''}

        <!-- XAI Samples (if artifacts exist) -->
        <div class="section">
            <h2>Samples & Visual Explanations</h2>
            <div class="card">
                <p class="muted" style="font-size: 12px; margin-bottom: 12px;">
                    {probe_count} probe samples tracked across {sample_count} snapshots.
                    This section is rendered directly from export artifacts for offline use.
                </p>
                <div class="sample-grid">
                    {''.join(sample_cards) if sample_cards else '<div class="muted">No sample visual artifacts were available for this run.</div>'}
                </div>
            </div>
        </div>

        <div class="footer">
            Generated by BNNR Dashboard · Offline Report
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data)};
        const svg = document.getElementById('metricSvg');
        if (svg && chartData.length > 1) {{
            const W = 900, H = 280, pad = 28;
            const plotW = W - pad * 2, plotH = H - pad * 2;
            const n = chartData.length;
            const xs = chartData.map((_, i) => pad + (i * plotW / (n - 1)));
            const pVals = chartData.map(d => Number(d.primary || 0));
            const sVals = chartData.map(d => Number(d.secondary || 0));
            const lVals = chartData.map(d => Number(d.loss || 0));
            const lMax = Math.max(...lVals, 1);
            const lScaled = lVals.map(v => (v / lMax) * 100.0);
            const y = (v) => pad + (100 - v) * plotH / 100.0;

            function poly(vals, color) {{
                const pts = vals.map((v, i) => `${{xs[i]}},${{y(v)}}`).join(' ');
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
                line.setAttribute('fill', 'none');
                line.setAttribute('stroke', color);
                line.setAttribute('stroke-width', '2');
                line.setAttribute('points', pts);
                return line;
            }}

            for (let t=0; t<=5; t++) {{
                const gy = pad + (t * plotH / 5);
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                g.setAttribute('x1', String(pad));
                g.setAttribute('x2', String(W-pad));
                g.setAttribute('y1', String(gy));
                g.setAttribute('y2', String(gy));
                g.setAttribute('stroke', '#e2e8f0');
                g.setAttribute('stroke-width', '1');
                svg.appendChild(g);
            }}

            svg.appendChild(poly(pVals, '#2563eb'));
            svg.appendChild(poly(sVals, '#16a34a'));
            svg.appendChild(poly(lScaled, '#dc2626'));
        }}
    </script>
</body>
</html>"""
