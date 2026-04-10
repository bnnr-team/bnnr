"""Opt-in integration: one ``UltralyticsDetectionAdapter.train_step`` on CPU.

Requires ``BNNR_ULTRA_YOLO_SMOKE=1`` (downloads ``yolov8n.pt`` on first run).
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.ultralytics
@pytest.mark.skipif(
    os.environ.get("BNNR_ULTRA_YOLO_SMOKE") != "1",
    reason="Set BNNR_ULTRA_YOLO_SMOKE=1 to run (network + yolov8n.pt).",
)
def test_one_yolo_train_step_finite_loss_cpu() -> None:
    pytest.importorskip("ultralytics")
    from bnnr.detection_adapter import UltralyticsDetectionAdapter

    adapter = UltralyticsDetectionAdapter(model_name="yolov8n.pt", device="cpu", use_amp=False)
    images = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    targets = [
        {
            "boxes": torch.tensor([[4.0, 4.0, 40.0, 40.0]], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[2.0, 2.0, 30.0, 30.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
    ]
    metrics = adapter.train_step((images, targets))
    assert metrics["loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
