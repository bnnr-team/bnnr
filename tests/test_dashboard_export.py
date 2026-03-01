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


def test_export_detection_uses_map_metrics_and_split_xai_panels(temp_dir) -> None:
    run_dir = temp_dir / "run_det"
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "schema_version": "2.1",
            "sequence": 1,
            "run_id": "run_det",
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "run_started",
            "payload": {"run_name": "run_det", "config": {"task": "detection"}},
        },
        {
            "schema_version": "2.1",
            "sequence": 2,
            "run_id": "run_det",
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "epoch_end",
            "payload": {
                "iteration": 0,
                "epoch": 1,
                "branch": "baseline",
                "metrics": {"loss": 1.0, "map_50": 0.12, "map_50_95": 0.06},
            },
        },
        {
            "schema_version": "2.1",
            "sequence": 3,
            "run_id": "run_det",
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "sample_prediction_snapshot",
            "payload": {
                "sample_id": "sample_1",
                "iteration": 0,
                "epoch": 1,
                "branch": "baseline",
                "branch_id": "root:baseline",
                "true_class": 2,
                "predicted_class": 2,
                "confidence": 0.85,
                "artifacts": {
                    "original": "artifacts/sample_1_original.png",
                    "augmented": "artifacts/sample_1_augmented.png",
                    "xai": "artifacts/sample_1_xai.png",
                },
            },
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    out_dir = temp_dir / "exported_det"
    exported = export_dashboard_snapshot(run_dir, out_dir)
    index_html = (exported / "index.html").read_text(encoding="utf-8")

    assert "Best mAP@0.5 ★" in index_html
    assert "Best mAP@[.5:.95] ★" in index_html
    assert "Final mAP@0.5" in index_html
    assert "Best Accuracy ★" not in index_html
    assert "Final Accuracy" not in index_html
    assert "XAI Panels" in index_html
    assert "xai-panel-gt" in index_html
    assert "xai-panel-saliency" in index_html
    assert "xai-panel-pred" in index_html
