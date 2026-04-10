#!/usr/bin/env python3
"""Manual smoke: one YOLO ``train_step`` via BNNR (CPU).

Downloads ``yolov8n.pt`` on first run. Default CI / local runs skip unless:

    BNNR_ULTRA_YOLO_SMOKE=1 python examples/detection/smoke_yolo_bnnr.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if os.environ.get("BNNR_ULTRA_YOLO_SMOKE") != "1":
        print(
            "Skip: set BNNR_ULTRA_YOLO_SMOKE=1 to run one real YOLO train_step "
            "(requires ultralytics, numpy, network for weights).",
        )
        return 0

    import torch

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
    loss = float(metrics["loss"])
    if not (loss > 0 and loss == loss):  # finite and positive
        print("FAIL: unexpected loss", metrics)
        return 1
    print("OK: train_step loss =", loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
