"""Integration: real ``yolov8n.pt`` + many ``UltralyticsDetectionAdapter.train_step`` calls.

Downloads weights once (Ultralytics cache). Catches regressions that only show up
after repeated optimizer steps (non-finite loss, class-id crashes).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ultralytics")


def _random_detection_batch(
    *,
    batch_size: int,
    height: int,
    width: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, list[dict]]:
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(batch_size, 3, height, width, generator=g)
    targets: list[dict] = []
    for _ in range(batch_size):
        n = int(torch.randint(1, 5, (1,), generator=g).item())
        boxes: list[list[float]] = []
        labels: list[int] = []
        for _ in range(n):
            x1 = float(torch.rand(1, generator=g) * (width - 40))
            y1 = float(torch.rand(1, generator=g) * (height - 40))
            x2 = x1 + float(torch.rand(1, generator=g) * 30 + 8)
            y2 = y1 + float(torch.rand(1, generator=g) * 30 + 8)
            x2 = min(x2, float(width - 1))
            y2 = min(y2, float(height - 1))
            boxes.append([x1, y1, x2, y2])
            labels.append(int(torch.randint(0, num_classes, (1,), generator=g).item()))
        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        })
    return images, targets


@pytest.fixture(scope="module")
def yolo_cpu_adapter():
    from bnnr.detection_adapter import UltralyticsDetectionAdapter

    return UltralyticsDetectionAdapter(
        model_name="yolov8n.pt",
        device="cpu",
        use_amp=False,
    )


def test_many_train_steps_all_finite(yolo_cpu_adapter) -> None:
    """25 optimizer steps; every loss must be finite (no skip metrics)."""
    nc = int(getattr(yolo_cpu_adapter.get_model(), "nc", 80))
    for step in range(25):
        images, targets = _random_detection_batch(
            batch_size=2,
            height=320,
            width=320,
            num_classes=nc,
            seed=10_000 + step,
        )
        metrics = yolo_cpu_adapter.train_step((images, targets))
        assert metrics.get("loss_non_finite") is None
        assert metrics.get("loss_yolo_index_error") is None
        loss = float(metrics["loss"])
        assert loss == loss
        assert loss >= 0.0


def test_extreme_class_ids_are_clamped_and_run(yolo_cpu_adapter) -> None:
    """Previously OOB cls caused ``IndexError`` inside Ultralytics TAL."""
    images = torch.rand(2, 3, 320, 320)
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 80.0, 80.0]]), "labels": torch.tensor([10_000])},
        {"boxes": torch.tensor([[20.0, 20.0, 100.0, 100.0]]), "labels": torch.tensor([-5])},
    ]
    m = yolo_cpu_adapter.train_step((images, targets))
    assert m.get("loss_yolo_index_error") is None
    assert float(m["loss"]) == float(m["loss"])
