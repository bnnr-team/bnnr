"""Tests for the quick_run high-level API."""

from __future__ import annotations

from bnnr.core import BNNRConfig
from bnnr.quick_run import quick_run


def test_quick_run_smoke(dummy_model, dummy_dataloader, temp_dir) -> None:
    cfg = BNNRConfig(
        m_epochs=1,
        max_iterations=1,
        xai_enabled=False,
        device="cpu",
        checkpoint_dir=temp_dir / "c",
        report_dir=temp_dir / "r",
    )
    result = quick_run(dummy_model, dummy_dataloader, dummy_dataloader, config=cfg)
    assert result.report_json_path.exists()
