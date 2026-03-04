"""Tests for detection XAI visualisations and events integration."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from torch import Tensor

from bnnr.detection_xai import (
    draw_boxes_on_image,
    generate_detection_saliency,
    overlay_saliency_heatmap,
    save_detection_xai_visualization,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_image() -> np.ndarray:
    """128×128 RGB uint8 image."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture()
def sample_boxes() -> Tensor:
    return torch.tensor([[10, 10, 50, 50], [60, 60, 120, 120]], dtype=torch.float32)


@pytest.fixture()
def sample_labels() -> Tensor:
    return torch.tensor([0, 1], dtype=torch.long)


@pytest.fixture()
def sample_scores() -> Tensor:
    return torch.tensor([0.9, 0.7], dtype=torch.float32)


# ---------------------------------------------------------------------------
#  draw_boxes_on_image
# ---------------------------------------------------------------------------


class TestDrawBoxes:
    def test_basic_draw(self, sample_image: np.ndarray, sample_boxes: Tensor, sample_labels: Tensor) -> None:
        out = draw_boxes_on_image(sample_image, sample_boxes, sample_labels)
        assert out.shape == sample_image.shape
        assert out.dtype == np.uint8

    def test_draw_with_scores(
        self, sample_image: np.ndarray, sample_boxes: Tensor, sample_labels: Tensor, sample_scores: Tensor
    ) -> None:
        out = draw_boxes_on_image(sample_image, sample_boxes, sample_labels, sample_scores)
        assert out.shape == sample_image.shape

    def test_draw_with_class_names(
        self, sample_image: np.ndarray, sample_boxes: Tensor, sample_labels: Tensor
    ) -> None:
        out = draw_boxes_on_image(
            sample_image, sample_boxes, sample_labels, class_names=["cat", "dog"]
        )
        assert out.shape == sample_image.shape

    def test_draw_numpy_inputs(self, sample_image: np.ndarray) -> None:
        boxes_np = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels_np = np.array([0], dtype=np.int64)
        out = draw_boxes_on_image(sample_image, boxes_np, labels_np)
        assert out.shape == sample_image.shape

    def test_draw_no_labels(self, sample_image: np.ndarray, sample_boxes: Tensor) -> None:
        out = draw_boxes_on_image(sample_image, sample_boxes)
        assert out.shape == sample_image.shape

    def test_draw_empty_boxes(self, sample_image: np.ndarray) -> None:
        empty_boxes = torch.zeros(0, 4)
        out = draw_boxes_on_image(sample_image, empty_boxes)
        assert out.shape == sample_image.shape


# ---------------------------------------------------------------------------
#  overlay_saliency_heatmap
# ---------------------------------------------------------------------------


class TestOverlaySaliency:
    def test_basic_overlay(self, sample_image: np.ndarray) -> None:
        saliency = np.random.rand(128, 128).astype(np.float32)
        out = overlay_saliency_heatmap(sample_image, saliency)
        assert out.shape == sample_image.shape
        assert out.dtype == np.uint8

    def test_overlay_different_size(self, sample_image: np.ndarray) -> None:
        """Saliency map gets resized to match image."""
        saliency = np.random.rand(32, 32).astype(np.float32)
        out = overlay_saliency_heatmap(sample_image, saliency)
        assert out.shape == sample_image.shape

    def test_overlay_constant_saliency(self, sample_image: np.ndarray) -> None:
        """All-zero saliency should not crash."""
        saliency = np.zeros((128, 128), dtype=np.float32)
        out = overlay_saliency_heatmap(sample_image, saliency)
        assert out.shape == sample_image.shape

    def test_overlay_3d_saliency(self, sample_image: np.ndarray) -> None:
        """3D saliency (e.g. from squeeze) should be handled."""
        saliency = np.random.rand(1, 128, 128).astype(np.float32)
        out = overlay_saliency_heatmap(sample_image, saliency)
        assert out.shape == sample_image.shape


# ---------------------------------------------------------------------------
#  generate_detection_saliency
# ---------------------------------------------------------------------------


class TestGenerateSaliency:
    def test_basic_generation(self) -> None:
        """Generate saliency from a simple conv model."""
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # Wrap to accept list input (detection-style)
        class DetStyleModel(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone

            def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
                x = torch.stack(images)
                self.backbone(x)
                return [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}] * len(images)

        det_model = DetStyleModel(model)
        images = torch.rand(2, 3, 32, 32)
        target_layers = [model[0]]  # First conv layer

        saliencies = generate_detection_saliency(det_model, images, target_layers, device="cpu")
        assert len(saliencies) == 2
        for sal in saliencies:
            assert sal.shape == (32, 32)
            assert sal.dtype == np.float32

    def test_no_activations(self) -> None:
        """If no activations captured, return zeros."""
        import torch.nn as nn

        class DummyModel(nn.Module):
            def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
                return []

        model = DummyModel()
        images = torch.rand(2, 3, 32, 32)
        # target_layers doesn't exist in model → no activations
        target_layers = [nn.Linear(1, 1)]

        saliencies = generate_detection_saliency(model, images, target_layers, device="cpu")
        assert len(saliencies) == 2
        for sal in saliencies:
            assert sal.shape == (32, 32)
            assert np.allclose(sal, 0.0)


# ---------------------------------------------------------------------------
#  save_detection_xai_visualization
# ---------------------------------------------------------------------------


class TestSaveVisualization:
    def test_save_composite(self, sample_image: np.ndarray, sample_boxes: Tensor, sample_labels: Tensor, tmp_path) -> None:
        saliency = np.random.rand(128, 128).astype(np.float32)
        path = save_detection_xai_visualization(
            sample_image,
            saliency,
            boxes_gt=sample_boxes,
            labels_gt=sample_labels,
            boxes_pred=sample_boxes,
            labels_pred=sample_labels,
            scores_pred=torch.tensor([0.9, 0.7]),
            class_names=["cat", "dog"],
            save_path=tmp_path / "xai_test.png",
            size=256,
        )
        assert path.exists()
        import cv2
        img = cv2.imread(str(path))
        assert img is not None
        # 3 panels side by side, each 256×256
        assert img.shape[1] == 256 * 3
        assert img.shape[0] == 256

    def test_save_without_saliency(self, sample_image: np.ndarray, sample_boxes: Tensor, tmp_path) -> None:
        """Without saliency, panels 2-3 should just show boxes."""
        path = save_detection_xai_visualization(
            sample_image,
            saliency=None,
            boxes_gt=sample_boxes,
            save_path=tmp_path / "no_sal.png",
        )
        assert path.exists()

    def test_save_without_boxes(self, sample_image: np.ndarray, tmp_path) -> None:
        saliency = np.random.rand(128, 128).astype(np.float32)
        path = save_detection_xai_visualization(
            sample_image,
            saliency,
            save_path=tmp_path / "no_boxes.png",
        )
        assert path.exists()

    def test_save_preserves_rgb_colors(self, tmp_path) -> None:
        # Pure red in RGB should remain red after save/load round-trip.
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[..., 0] = 255
        path = save_detection_xai_visualization(
            image=image,
            saliency=None,
            save_path=tmp_path / "rgb_roundtrip.png",
            size=64,
        )
        saved_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert saved_bgr is not None
        saved_rgb = cv2.cvtColor(saved_bgr, cv2.COLOR_BGR2RGB)
        sample_px = saved_rgb[10, 10]
        assert int(sample_px[0]) > 200
        assert int(sample_px[1]) < 30
        assert int(sample_px[2]) < 30


# ---------------------------------------------------------------------------
#  Events replay — detection metrics pass-through
# ---------------------------------------------------------------------------


class TestEventsDetectionMetrics:
    def test_replay_includes_detection_metrics(self) -> None:
        from bnnr.events import replay_events

        events = [
            {
                "type": "run_started",
                "payload": {
                    "run_name": "det_test",
                    "config": {"task": "detection"},
                    "class_names": [],
                },
            },
            {
                "type": "epoch_end",
                "payload": {
                    "iteration": 0,
                    "epoch": 1,
                    "branch": "baseline",
                    "metrics": {
                        "loss": 1.5,
                        "map_50": 0.35,
                        "map_50_95": 0.22,
                    },
                },
            },
        ]

        state = replay_events(events)
        tl = state["metrics_timeline"]
        assert len(tl) == 1
        assert tl[0]["map_50"] == 0.35
        assert tl[0]["map_50_95"] == 0.22
        assert tl[0]["accuracy"] == 0.0  # Not set for detection
        assert tl[0]["loss"] == 1.5

    def test_replay_classification_still_works(self) -> None:
        from bnnr.events import replay_events

        events = [
            {
                "type": "run_started",
                "payload": {
                    "run_name": "cls_test",
                    "config": {"task": "classification"},
                    "class_names": [],
                },
            },
            {
                "type": "epoch_end",
                "payload": {
                    "iteration": 0,
                    "epoch": 1,
                    "branch": "baseline",
                    "metrics": {
                        "loss": 0.5,
                        "accuracy": 0.85,
                        "f1_macro": 0.82,
                    },
                },
            },
        ]

        state = replay_events(events)
        tl = state["metrics_timeline"]
        assert tl[0]["accuracy"] == 0.85
        assert tl[0]["f1_macro"] == 0.82
        assert tl[0]["map_50"] == 0.0  # Not set for classification
        assert tl[0]["map_50_95"] == 0.0

    def test_replay_stores_task_type(self) -> None:
        from bnnr.events import replay_events

        events = [
            {
                "type": "run_started",
                "payload": {
                    "run_name": "det_test",
                    "config": {"task": "detection"},
                },
            },
        ]
        state = replay_events(events)
        assert state.get("task") == "detection"

    def test_replay_detection_metric_units(self) -> None:
        from bnnr.events import replay_events

        events = [
            {
                "type": "run_started",
                "payload": {
                    "run_name": "det_test",
                    "config": {"task": "detection"},
                },
            },
        ]
        state = replay_events(events)
        units = state["run"].get("metric_units", {})
        assert "map_50" in units
        assert "map_50_95" in units


# ---------------------------------------------------------------------------
#  Config auto-detection defaults
# ---------------------------------------------------------------------------


class TestConfigAutoDetection:
    def test_auto_selection_metric(self) -> None:
        from bnnr.core import BNNRConfig

        cfg = BNNRConfig(task="detection")
        assert cfg.selection_metric == "map_50"
        assert "map_50" in cfg.metrics
        assert "map_50_95" in cfg.metrics

    def test_classification_defaults_unchanged(self) -> None:
        from bnnr.core import BNNRConfig

        cfg = BNNRConfig(task="classification")
        assert cfg.selection_metric == "accuracy"
        assert "accuracy" in cfg.metrics

    def test_explicit_override_preserved(self) -> None:
        from bnnr.core import BNNRConfig

        cfg = BNNRConfig(task="detection", selection_metric="map_50_95")
        assert cfg.selection_metric == "map_50_95"
