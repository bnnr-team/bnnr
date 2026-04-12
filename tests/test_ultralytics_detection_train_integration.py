"""Integration: real ``yolov8n.pt`` + many ``UltralyticsDetectionAdapter.train_step`` calls.

Downloads weights once (Ultralytics cache). Catches regressions that only show up
after repeated optimizer steps (non-finite loss, class-id crashes).

**YOLO26** (``yolo26n.pt``, ``@pytest.mark.yolo26``): on GitHub Actions the adapter
fixture **fails** if Ultralytics is too old or weights cannot load, so CI always
exercises BNNR + YOLO26 when this file runs (see ``pytest`` in ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("ultralytics")


def _ultralytics_version_tuple() -> tuple[int, int, int] | None:
    try:
        from importlib.metadata import version as pkg_version

        raw = (pkg_version("ultralytics").split("+", 1)[0]).split(".")[:3]
        parts: list[int] = []
        for p in raw:
            digits = "".join(ch for ch in p if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        while len(parts) < 3:
            parts.append(0)
        return (parts[0], parts[1], parts[2])
    except Exception:
        return None


def _ultralytics_at_least(min_parts: tuple[int, int, int]) -> bool:
    got = _ultralytics_version_tuple()
    return got is not None and got >= min_parts


def _running_on_github_actions() -> bool:
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


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


@pytest.fixture(scope="module")
def yolo26_cpu_adapter():
    """Real ``yolo26n.pt``.

    Local dev: skip if Ultralytics is too old or weights cannot be downloaded.
    GitHub Actions: **fail** so CI never merges without YOLO26 + BNNR compatibility.
    """
    on_ci = _running_on_github_actions()
    if not _ultralytics_at_least((8, 3, 0)):
        msg = "YOLO26 integration needs ultralytics>=8.3.0"
        if on_ci:
            pytest.fail(msg)
        pytest.skip(msg)
    from bnnr.detection_adapter import UltralyticsDetectionAdapter

    try:
        return UltralyticsDetectionAdapter(
            model_name="yolo26n.pt",
            device="cpu",
            use_amp=False,
        )
    except Exception as exc:  # noqa: BLE001
        if on_ci:
            pytest.fail(f"yolo26n.pt must be loadable on CI (network / ultralytics): {exc}")
        pytest.skip(f"yolo26n.pt not available: {exc}")


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
        assert metrics.get("loss_yolo_pred_format_error") is None
        assert metrics.get("loss_yolo_fused_head") is None
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
    assert m.get("loss_yolo_pred_format_error") is None
    assert float(m["loss"]) == float(m["loss"])


@pytest.mark.yolo26
def test_yolo26_one_train_step_finite(yolo26_cpu_adapter) -> None:
    """Regression: YOLO26 ``E2ELoss`` needs training-mode head preds (``one2many['boxes']``)."""
    nc = int(getattr(yolo26_cpu_adapter.get_model(), "nc", 80))
    images, targets = _random_detection_batch(
        batch_size=2,
        height=320,
        width=320,
        num_classes=nc,
        seed=42,
    )
    metrics = yolo26_cpu_adapter.train_step((images, targets))
    assert metrics.get("loss_yolo_index_error") is None
    assert metrics.get("loss_yolo_pred_format_error") is None
    assert metrics.get("loss_yolo_fused_head") is None
    loss = float(metrics["loss"])
    assert loss == loss
    assert loss >= 0.0


@pytest.mark.yolo26
def test_yolo26_multiple_train_steps_finite(yolo26_cpu_adapter) -> None:
    """Several optimizer steps on YOLO26 (regression beyond a single lucky batch)."""
    nc = int(getattr(yolo26_cpu_adapter.get_model(), "nc", 80))
    for step in range(8):
        images, targets = _random_detection_batch(
            batch_size=2,
            height=320,
            width=320,
            num_classes=nc,
            seed=20_000 + step,
        )
        metrics = yolo26_cpu_adapter.train_step((images, targets))
        assert metrics.get("loss_yolo_index_error") is None
        assert metrics.get("loss_yolo_pred_format_error") is None
        assert metrics.get("loss_yolo_fused_head") is None
        loss = float(metrics["loss"])
        assert loss == loss
        assert loss >= 0.0
