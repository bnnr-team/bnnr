"""Tests for dashboard export and offline snapshot generation."""

from __future__ import annotations

import json

from bnnr.dashboard.exporter import export_dashboard_snapshot


def test_export_dashboard_snapshot_creates_static_bundle(temp_dir) -> None:
    run_dir = temp_dir / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "sequence": 1,
                "run_id": "run_1",
                "timestamp": "2026-01-01T00:00:00Z",
                "type": "run_started",
                "payload": {"run_name": "run_1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "a.txt").write_text("artifact", encoding="utf-8")

    out_dir = temp_dir / "exported"
    exported = export_dashboard_snapshot(run_dir, out_dir)
    assert exported.exists()
    assert (exported / "index.html").exists()
    assert (exported / "data" / "events.jsonl").exists()
    assert (exported / "data" / "state.json").exists()
    assert (exported / "artifacts" / "a.txt").exists()
    assert (exported / "manifest.json").exists()


def test_export_with_frontend_generates_single_index_with_visual_sections(temp_dir) -> None:
    run_dir = temp_dir / "run_2"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "sequence": 1,
                "run_id": "run_2",
                "timestamp": "2026-01-01T00:00:00Z",
                "type": "run_started",
                "payload": {"run_name": "run_2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    frontend_dist = temp_dir / "frontend_dist"
    (frontend_dist / "assets").mkdir(parents=True, exist_ok=True)
    (frontend_dist / "assets" / "main.js").write_text("console.log('ok');", encoding="utf-8")
    (frontend_dist / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head>",
                '<script type="module" src="./assets/main.js"></script>',
                "</head><body><div id='root'></div></body></html>",
            ]
        ),
        encoding="utf-8",
    )

    out_dir = temp_dir / "exported_with_frontend"
    exported = export_dashboard_snapshot(run_dir, out_dir, frontend_dist=frontend_dist)
    assert exported.exists()

    index_html = (exported / "index.html").read_text(encoding="utf-8")
    assert "Samples & Visual Explanations" in index_html
    assert "metricSvg" in index_html
    assert "cdn.jsdelivr.net" not in index_html
    assert "fetch('./data/state.json')" not in index_html
    assert (exported / "offline.html").exists() is False


