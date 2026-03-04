"""Tests for object detection support (PR1 foundation).

Covers:
- BNNRConfig detection fields & validation
- detection_metrics placeholder
- detection_collate_fn
- TorchvisionDetectionAdapter basics
- BNNRTrainer detection-mode smoke test
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.detection_collate import detection_collate_fn, detection_collate_fn_with_index
from bnnr.detection_metrics import calculate_detection_confusion_matrix, calculate_detection_metrics

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


class _FakeDetectionModel(nn.Module):
    """Minimal stand-in that behaves like a torchvision detection model.

    * training mode  → returns loss dict
    * eval mode      → returns list of prediction dicts
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(
        self,
        images: list[Tensor] | Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> list[dict[str, Tensor]] | dict[str, Tensor]:
        if self.training and targets is not None:
            # Return fake losses
            return {
                "loss_classifier": torch.tensor(0.5, requires_grad=True),
                "loss_box_reg": torch.tensor(0.3, requires_grad=True),
            }
        # Eval → return predictions (one per image)
        batch_size = len(images) if isinstance(images, list) else images.shape[0]
        preds = []
        for _ in range(batch_size):
            preds.append(
                {
                    "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                    "labels": torch.tensor([1]),
                    "scores": torch.tensor([0.9]),
                }
            )
        return preds


class _FakeDetectionAdapter:
    """Minimal adapter that satisfies ModelAdapter protocol for detection."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.model = _FakeDetectionModel().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def train_step(
        self, batch: tuple[Tensor, list[dict[str, Tensor]]]
    ) -> dict[str, float]:
        self.model.train()
        images, targets = batch
        images = images.to(self.device)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        self.optimizer.zero_grad()
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        self.optimizer.step()
        return {"loss": float(losses.item())}

    def eval_step(
        self, batch: tuple[Tensor, list[dict[str, Tensor]]]
    ) -> dict[str, float]:
        self.model.eval()
        images, targets = batch
        images = images.to(self.device)
        with torch.no_grad():
            preds = self.model(images)
        metrics = calculate_detection_metrics(preds, targets)
        return metrics

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


class _FakeDetectionDataset(Dataset):
    """Generates random detection samples: (image, target_dict, index)."""

    def __init__(self, n: int = 20, image_size: int = 32) -> None:
        self.n = n
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Tensor], int]:
        img = torch.rand(3, self.image_size, self.image_size)
        num_boxes = torch.randint(1, 4, (1,)).item()
        boxes = torch.rand(num_boxes, 4) * self.image_size
        # Ensure x2 > x1, y2 > y1
        boxes[:, 2] = boxes[:, 0] + torch.abs(boxes[:, 2] - boxes[:, 0]).clamp(min=2)
        boxes[:, 3] = boxes[:, 1] + torch.abs(boxes[:, 3] - boxes[:, 1]).clamp(min=2)
        labels = torch.randint(0, 5, (num_boxes,))
        target = {"boxes": boxes, "labels": labels}
        return img, target, idx


def _det_collate(batch):
    images = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    indices = [b[2] for b in batch]
    return images, targets, indices


@pytest.fixture
def det_config(tmp_path: Path) -> BNNRConfig:
    return BNNRConfig(
        task="detection",
        m_epochs=1,
        max_iterations=1,
        metrics=["mAP@0.5", "loss"],
        selection_metric="loss",
        selection_mode="min",
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        xai_enabled=False,
        device="cpu",
        save_checkpoints=True,
        detection_class_names=["cat", "dog", "car", "person", "bike"],
    )


@pytest.fixture
def det_adapter() -> _FakeDetectionAdapter:
    return _FakeDetectionAdapter(device="cpu")


@pytest.fixture
def det_loaders() -> tuple[DataLoader, DataLoader]:
    train_ds = _FakeDetectionDataset(n=16, image_size=32)
    val_ds = _FakeDetectionDataset(n=8, image_size=32)
    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=_det_collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=_det_collate)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
#  Config tests
# ---------------------------------------------------------------------------


class TestDetectionConfig:
    def test_task_field_default_is_classification(self) -> None:
        cfg = BNNRConfig()
        assert cfg.task == "classification"

    def test_task_field_accepts_detection(self) -> None:
        cfg = BNNRConfig(task="detection")
        assert cfg.task == "detection"

    def test_task_field_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="classification.*detection"):
            BNNRConfig(task="segmentation")

    def test_detection_bbox_format_default(self) -> None:
        cfg = BNNRConfig(task="detection")
        assert cfg.detection_bbox_format == "xyxy"

    def test_detection_bbox_format_accepts_valid(self) -> None:
        for fmt in ("xyxy", "xywh", "cxcywh"):
            cfg = BNNRConfig(task="detection", detection_bbox_format=fmt)
            assert cfg.detection_bbox_format == fmt

    def test_detection_bbox_format_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="xyxy.*xywh.*cxcywh"):
            BNNRConfig(task="detection", detection_bbox_format="pascal_voc")

    def test_detection_score_threshold(self) -> None:
        cfg = BNNRConfig(task="detection", detection_score_threshold=0.7)
        assert cfg.detection_score_threshold == 0.7

    def test_detection_class_names(self) -> None:
        names = ["cat", "dog", "bird"]
        cfg = BNNRConfig(task="detection", detection_class_names=names)
        assert cfg.detection_class_names == names


# ---------------------------------------------------------------------------
#  Detection metrics
# ---------------------------------------------------------------------------


class TestDetectionMetrics:
    def test_returns_dict_with_expected_keys(self) -> None:
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        result = calculate_detection_metrics(preds, targets)
        assert isinstance(result, dict)
        assert "map_50" in result
        assert "map_50_95" in result

    def test_custom_iou_thresholds(self) -> None:
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.9]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        result = calculate_detection_metrics(preds, targets, iou_thresholds=[0.5])
        assert "map_50" in result

    def test_empty_inputs(self) -> None:
        result = calculate_detection_metrics([], [])
        assert isinstance(result, dict)
        assert result["map_50"] == 0.0
        assert result["map_50_95"] == 0.0

    def test_detection_confusion_matrix_shape(self) -> None:
        preds = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]
        confusion = calculate_detection_confusion_matrix(preds, targets, num_classes=4, iou_threshold=0.5)
        assert "labels" in confusion and "matrix" in confusion
        assert len(confusion["labels"]) == 4
        assert len(confusion["matrix"]) == 4
        assert len(confusion["matrix"][0]) == 4


# ---------------------------------------------------------------------------
#  Detection collate
# ---------------------------------------------------------------------------


class TestDetectionCollate:
    def test_collate_basic(self) -> None:
        batch = [
            (
                torch.rand(3, 32, 32),
                {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
            ),
            (
                torch.rand(3, 32, 32),
                {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]), "labels": torch.tensor([2])},
            ),
        ]
        images, targets = detection_collate_fn(batch)
        assert isinstance(images, Tensor)
        assert images.shape[0] == 2
        assert len(targets) == 2
        assert targets[0]["labels"].tolist() == [1]

    def test_collate_variable_box_counts(self) -> None:
        """Each image can have different number of boxes."""
        batch = [
            (
                torch.rand(3, 32, 32),
                {"boxes": torch.rand(3, 4), "labels": torch.tensor([0, 1, 2])},
            ),
            (
                torch.rand(3, 32, 32),
                {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
            ),
        ]
        images, targets = detection_collate_fn(batch)
        assert targets[0]["boxes"].shape[0] == 3
        assert targets[1]["boxes"].shape[0] == 1

    def test_collate_with_index(self) -> None:
        """detection_collate_fn_with_index returns images, targets, indices."""
        batch = [
            (
                torch.rand(3, 32, 32),
                {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])},
                42,
            ),
            (
                torch.rand(3, 32, 32),
                {"boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]), "labels": torch.tensor([2])},
                43,
            ),
        ]
        images, targets, indices = detection_collate_fn_with_index(batch)
        assert isinstance(images, Tensor)
        assert len(targets) == 2
        assert indices.tolist() == [42, 43]


# ---------------------------------------------------------------------------
#  BNNRTrainer detection smoke test
# ---------------------------------------------------------------------------


class TestTrainerDetection:
    def test_is_detection_property(self, det_config: BNNRConfig, det_adapter, det_loaders) -> None:
        train_loader, val_loader = det_loaders
        trainer = BNNRTrainer(
            model=det_adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=[],
            config=det_config,
        )
        assert trainer._is_detection is True

    def test_classification_is_not_detection(self, sample_config: BNNRConfig, model_adapter, dummy_dataloader) -> None:
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        assert trainer._is_detection is False

    def test_detection_run_completes(self, det_config, det_adapter, det_loaders) -> None:
        """Smoke test: full trainer.run() with detection task completes without error."""
        train_loader, val_loader = det_loaders
        trainer = BNNRTrainer(
            model=det_adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=[],
            config=det_config,
        )
        result = trainer.run()
        assert result.best_path is not None
        assert isinstance(result.best_metrics, dict)

    def test_detection_dataset_profile_has_task(self, det_config, det_adapter, det_loaders) -> None:
        """_compute_dataset_profile should include task='detection' and box counts."""
        train_loader, val_loader = det_loaders
        trainer = BNNRTrainer(
            model=det_adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=[],
            config=det_config,
        )
        profile = trainer._compute_dataset_profile()
        assert profile["task"] == "detection"
        assert "total_train_boxes" in profile
        assert "total_val_boxes" in profile
        assert profile["total_train_boxes"] > 0

    def test_detection_eval_class_details(self, det_config, det_adapter, det_loaders) -> None:
        """For detection, _compute_eval_class_details should return per-class AP."""
        train_loader, val_loader = det_loaders
        trainer = BNNRTrainer(
            model=det_adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=[],
            config=det_config,
        )
        class_details, confusion = trainer._compute_eval_class_details()
        # Detection returns per-class AP (class_details) and class-level confusion
        assert isinstance(class_details, dict)
        assert isinstance(confusion, dict)
        if confusion:
            assert "matrix" in confusion


# ---------------------------------------------------------------------------
#  Classification regression – ensure detection changes don't break it
# ---------------------------------------------------------------------------


class TestClassificationRegression:
    """Make sure existing classification workflow still works after detection changes."""

    def test_classification_run_still_works(self, model_adapter, dummy_dataloader, sample_config) -> None:
        from bnnr.augmentations import BasicAugmentation

        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[BasicAugmentation(probability=0.5)],
            config=sample_config,
        )
        result = trainer.run()
        assert result.best_path is not None
        assert "accuracy" in result.best_metrics or "loss" in result.best_metrics

    def test_classification_dataset_profile_no_task_field(self, model_adapter, dummy_dataloader, sample_config) -> None:
        trainer = BNNRTrainer(
            model=model_adapter,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            augmentations=[],
            config=sample_config,
        )
        profile = trainer._compute_dataset_profile()
        assert "task" not in profile  # classification doesn't add task key
        assert "total_train_boxes" not in profile

    def test_config_default_is_still_classification(self) -> None:
        cfg = BNNRConfig()
        assert cfg.task == "classification"
        assert cfg.selection_metric == "accuracy"
