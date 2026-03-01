"""Tests for dashboard frontend source expectations used by backend/export."""

from __future__ import annotations

from pathlib import Path


def test_detection_global_xai_view_excludes_gt_overlay() -> None:
    src = Path(__file__).resolve().parents[1] / "dashboard_web" / "src" / "components" / "AugmentationPreview.tsx"
    text = src.read_text(encoding="utf-8")
    assert '<option value="gt">GT Overlay</option>' not in text
    assert '<option value="saliency">Saliency (Global)</option>' in text
    assert '<option value="pred">Pred + Saliency</option>' in text
