"""Tests for dashboard frontend source expectations used by backend/export."""

from __future__ import annotations

from pathlib import Path


def test_augmentation_preview_has_xai_heatmap() -> None:
    """AugmentationPreview shows XAI heatmap for classification."""
    src = Path(__file__).resolve().parents[1] / "dashboard_web" / "src" / "components" / "AugmentationPreview.tsx"
    text = src.read_text(encoding="utf-8")
    assert "XAI Heatmap" in text
