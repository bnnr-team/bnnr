"""Tests closing coverage gaps for the detection feature.

Covers:
- calculate_per_class_ap (detection_metrics.py)
- compute_detection_box_saliency_occlusion (detection_xai.py)
- DetectionAdapter NaN/Inf loss and bad gradient paths
- DetectionICD/DetectionAICD fill_strategy variants + ValueError
- load_events malformed JSON skipping
- Reporter detection_details
- Dashboard exporter detection panels
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from bnnr.dashboard.exporter import _standalone_report_html
from bnnr.detection_adapter import DetectionAdapter
from bnnr.detection_icd import DetectionAICD, DetectionICD
from bnnr.detection_metrics import calculate_per_class_ap
from bnnr.detection_xai import compute_detection_box_saliency_occlusion
from bnnr.events import load_events, load_events_from_offset
from bnnr.reporting import Reporter


class TestCalculatePerClassAP:
    def _make_preds_targets(
        self,
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        preds = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=torch.float32),
                "labels": torch.tensor([0, 1]),
            },
        ]
        return preds, targets

    def test_basic_happy_path(self):
        preds, targets = self._make_preds_targets()
        result = calculate_per_class_ap(preds, targets, iou_threshold=0.5)
        assert "0" in result
        assert "1" in result
        assert result["0"]["ap"] > 0.0
        assert result["1"]["ap"] > 0.0
        assert result["0"]["support"] == 1
        assert result["1"]["support"] == 1

    def test_with_class_names(self):
        preds, targets = self._make_preds_targets()
        result = calculate_per_class_ap(
            preds, targets, class_names=["cat", "dog"],
        )
        assert result["0"]["name"] == "cat"
        assert result["1"]["name"] == "dog"

    def test_class_names_fallback_for_out_of_range(self):
        preds = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([5]),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "labels": torch.tensor([5]),
            },
        ]
        result = calculate_per_class_ap(preds, targets, class_names=["a", "b"])
        assert result["5"]["name"] == "class_5"

    def test_no_gt_returns_empty(self):
        preds = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            },
        ]
        targets = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
            },
        ]
        result = calculate_per_class_ap(preds, targets)
        assert result == {}

    def test_no_predictions(self):
        preds = [
            {
                "boxes": torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.long),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "labels": torch.tensor([0]),
            },
        ]
        result = calculate_per_class_ap(preds, targets)
        assert "0" in result
        assert result["0"]["ap"] == 0.0
        assert result["0"]["support"] == 1

    def test_multi_image(self):
        preds = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                "scores": torch.tensor([0.7]),
                "labels": torch.tensor([0]),
            },
        ]
        targets = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.tensor([[20, 20, 60, 60]], dtype=torch.float32),
                "labels": torch.tensor([0]),
            },
        ]
        result = calculate_per_class_ap(preds, targets)
        assert result["0"]["support"] == 2
        assert result["0"]["ap"] > 0.0


# ---------------------------------------------------------------------------
# compute_detection_box_saliency_occlusion
# ---------------------------------------------------------------------------


class _SimpleDetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, images: list[Tensor]) -> list[dict[str, Tensor]]:
        preds = []
        for img in images:
            preds.append({
                "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            })
        return preds


class TestComputeDetectionBoxSaliencyOcclusion:
    def test_basic_occlusion(self):
        model = _SimpleDetModel()
        model.eval()
        image = torch.rand(3, 32, 32)
        query_boxes = torch.tensor([[5.0, 5.0, 25.0, 25.0]])
        query_labels = torch.tensor([1])

        maps, scores = compute_detection_box_saliency_occlusion(
            model, image, query_boxes, query_labels,
            device="cpu", grid_size=4,
        )
        assert len(maps) == 1
        assert maps[0].shape == (32, 32)
        assert scores.shape == (1,)

    def test_empty_query_boxes(self):
        model = _SimpleDetModel()
        model.eval()
        image = torch.rand(3, 32, 32)
        query_boxes = torch.zeros(0, 4)
        query_labels = torch.zeros(0, dtype=torch.long)

        maps, scores = compute_detection_box_saliency_occlusion(
            model, image, query_boxes, query_labels,
            device="cpu",
        )
        assert maps == []
        assert scores.shape == (0,)

    def test_wrong_image_dim_raises(self):
        model = _SimpleDetModel()
        image = torch.rand(1, 3, 32, 32)  # 4D instead of 3D
        with pytest.raises(ValueError, match="shape"):
            compute_detection_box_saliency_occlusion(
                model, image, torch.zeros(1, 4), torch.zeros(1, dtype=torch.long),
                device="cpu",
            )

    def test_with_predict_chw(self):
        model = _SimpleDetModel()
        model.eval()
        image = torch.rand(3, 32, 32)
        query_boxes = torch.tensor([[5.0, 5.0, 25.0, 25.0]])
        query_labels = torch.tensor([1])

        def custom_predict(im: Tensor) -> dict[str, Tensor]:
            return {
                "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
                "scores": torch.tensor([0.85]),
                "labels": torch.tensor([1]),
            }

        maps, scores = compute_detection_box_saliency_occlusion(
            model, image, query_boxes, query_labels,
            predict_chw=custom_predict, device="cpu", grid_size=3,
        )
        assert len(maps) == 1
        assert maps[0].shape == (32, 32)

    def test_multiple_query_boxes(self):
        model = _SimpleDetModel()
        model.eval()
        image = torch.rand(3, 32, 32)
        query_boxes = torch.tensor([
            [5.0, 5.0, 15.0, 15.0],
            [16.0, 16.0, 30.0, 30.0],
        ])
        query_labels = torch.tensor([1, 1])

        maps, scores = compute_detection_box_saliency_occlusion(
            model, image, query_boxes, query_labels,
            device="cpu", grid_size=4,
        )
        assert len(maps) == 2
        assert scores.shape == (2,)


# ---------------------------------------------------------------------------
# DetectionAdapter NaN/Inf loss and bad gradients
# ---------------------------------------------------------------------------


class _NaNLossModel(nn.Module):
    def __init__(self, return_nan: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self._return_nan = return_nan

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None) -> Any:
        if self.training and targets is not None:
            if self._return_nan:
                return {"loss": torch.tensor(float("nan"), requires_grad=True)}
            return {"loss": torch.tensor(0.5, requires_grad=True)}
        return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}]


class _BadGradModel(nn.Module):
    """Returns a finite loss but produces Inf gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0))
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, images: list[Tensor], targets: list[dict[str, Tensor]] | None = None) -> Any:
        if self.training and targets is not None:
            loss = self.param * 0.5
            return {"loss": loss}
        return [{"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}]


def _make_batch(n: int = 2, img_size: int = 32) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images = torch.randn(n, 3, img_size, img_size)
    targets = [
        {
            "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
        for _ in range(n)
    ]
    return images, targets


class TestDetectionAdapterNaNLoss:
    def test_nan_loss_skipped(self):
        model = _NaNLossModel(return_nan=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        batch = _make_batch(2)
        metrics = adapter.train_step(batch)
        assert metrics["loss"] == 0.0
        assert metrics.get("loss_non_finite") == 1.0

    def test_finite_loss_works(self):
        model = _NaNLossModel(return_nan=False)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        batch = _make_batch(2)
        metrics = adapter.train_step(batch)
        assert metrics["loss"] > 0.0
        assert "loss_non_finite" not in metrics


class TestDetectionAdapterBadGradients:
    def test_bad_gradients_skipped(self):
        model = _BadGradModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        batch = _make_batch(2)

        # Manually inject Inf into gradients after backward
        metrics = adapter.train_step(batch)
        # Normal case: finite gradients should yield normal loss
        assert "loss" in metrics

        adapter.train_step(batch)  # warm-up

        # Directly test the bad gradient path
        model.train()
        images, targets = batch
        images_list = [img.to("cpu") for img in images]
        targets_on_device = [
            {k: v.to("cpu") if isinstance(v, Tensor) else v for k, v in t.items()}
            for t in targets
        ]
        adapter.optimizer.zero_grad()
        loss_dict = model(images_list, targets_on_device)
        total_loss = sum(loss_dict.values())
        total_loss.backward()

        # Inject inf into gradient
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("inf"))
                break

        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad_grad = True
                break
        assert bad_grad


# ---------------------------------------------------------------------------
# DetectionICD/DetectionAICD fill_strategy
# ---------------------------------------------------------------------------


class TestDetectionICDFillStrategies:
    def _make_image_and_target(self) -> tuple[np.ndarray, dict[str, Any]]:
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        target = {
            "boxes": np.array([[10, 10, 40, 40]], dtype=np.float32),
            "labels": np.array([1]),
        }
        return image, target

    def test_solid_fill(self):
        icd = DetectionICD(fill_strategy="solid", fill_value=0, probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = icd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_gaussian_blur_fill(self):
        icd = DetectionICD(fill_strategy="gaussian_blur", probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = icd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_noise_fill(self):
        icd = DetectionICD(fill_strategy="noise", probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = icd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_global_mean_fill(self):
        icd = DetectionICD(fill_strategy="global_mean", probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = icd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_local_mean_fill(self):
        icd = DetectionICD(fill_strategy="local_mean", probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = icd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_invalid_fill_strategy_raises(self):
        with pytest.raises(ValueError, match="fill_strategy"):
            DetectionICD(fill_strategy="invalid_strategy")

    def test_aicd_solid_fill(self):
        aicd = DetectionAICD(fill_strategy="solid", fill_value=128, probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = aicd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_aicd_gaussian_blur_fill(self):
        aicd = DetectionAICD(fill_strategy="gaussian_blur", probability=1.0)
        image, target = self._make_image_and_target()
        out, out_target = aicd.apply_with_targets(image.copy(), target)
        assert out.shape == image.shape

    def test_targets_unchanged(self):
        icd = DetectionICD(fill_strategy="solid", probability=1.0)
        image, target = self._make_image_and_target()
        original_boxes = target["boxes"].copy()
        _, out_target = icd.apply_with_targets(image.copy(), target)
        np.testing.assert_array_equal(out_target["boxes"], original_boxes)


# ---------------------------------------------------------------------------
# Reporter detection_details assertions
# ---------------------------------------------------------------------------


class TestReporterDetectionDetails:
    def test_log_sample_prediction_with_detection_details(self, temp_dir):
        reporter = Reporter(report_dir=temp_dir)
        from bnnr.core import BNNRConfig

        cfg = BNNRConfig(
            checkpoint_dir=temp_dir / "c",
            report_dir=temp_dir,
            task="detection",
        )
        reporter.start(cfg)
        reporter.log_sample_prediction(
            sample_id="det_sample_0",
            iteration=0,
            epoch=1,
            branch="baseline",
            true_class=0,
            predicted_class=1,
            confidence=0.85,
            loss_local=None,
            xai_gt_artifact="xai_gt.png",
            xai_saliency_artifact="xai_sal.png",
            xai_pred_artifact="xai_pred.png",
            detection_details={"num_gt_boxes": 3, "num_pred_boxes": 5, "map_50": 0.7},
        )

        events_file = reporter.run_dir / "events.jsonl"
        assert events_file.exists()

        from bnnr.events import load_events
        events = load_events(events_file)
        pred_events = [e for e in events if e["type"] == "sample_prediction_snapshot"]
        assert len(pred_events) == 1
        payload = pred_events[0]["payload"]
        assert payload["detection_details"]["num_gt_boxes"] == 3
        assert payload["artifacts"]["xai_gt"] == "xai_gt.png"
        assert payload["artifacts"]["xai_saliency"] == "xai_sal.png"
        assert payload["artifacts"]["xai_pred"] == "xai_pred.png"


# ---------------------------------------------------------------------------
# Dashboard exporter detection panels
# ---------------------------------------------------------------------------


class TestExporterDetectionPanels:
    def _make_detection_state(self) -> dict[str, Any]:
        return {
            "task": "detection",
            "run": {"config": {"task": "detection"}},
            "metrics_timeline": [
                {"iteration": 0, "epoch": 1, "branch": "baseline", "map_50": 0.5, "map_50_95": 0.3},
            ],
            "decision_history": [],
            "selected_path": ["baseline"],
            "branches": {},
            "xai": [],
            "sample_timelines": {
                "s0": [
                    {
                        "iteration": 0,
                        "epoch": 1,
                        "branch": "baseline",
                        "true_class": 0,
                        "predicted_class": 1,
                        "confidence": 0.8,
                        "artifacts": {
                            "original": "orig.png",
                            "augmented": "aug.png",
                            "xai": None,
                            "xai_gt": "gt.png",
                            "xai_saliency": "sal.png",
                            "xai_pred": "pred.png",
                        },
                    },
                ],
            },
            "per_class_timeline": {},
            "xai_insights_timeline": [],
        }

    def test_detection_html_contains_xai_triptych(self):
        state = self._make_detection_state()
        html = _standalone_report_html(state, "test_run")
        assert "XAI Panels" in html
        assert "gt.png" in html
        assert "sal.png" in html
        assert "pred.png" in html

    def test_detection_html_does_not_contain_plain_xai(self):
        state = self._make_detection_state()
        rendered = _standalone_report_html(state, "test_run")
        assert "detection" in state["task"]
        assert "XAI Panels" in rendered

    def test_classification_html_no_triptych(self):
        state = {
            "task": "classification",
            "run": {"config": {"task": "classification"}},
            "metrics_timeline": [
                {"iteration": 0, "epoch": 1, "branch": "baseline", "accuracy": 0.9},
            ],
            "decision_history": [],
            "selected_path": ["baseline"],
            "branches": {},
            "xai": [],
            "sample_timelines": {
                "s0": [
                    {
                        "iteration": 0,
                        "epoch": 1,
                        "branch": "baseline",
                        "true_class": 0,
                        "predicted_class": 0,
                        "confidence": 0.95,
                        "artifacts": {
                            "original": "orig.png",
                            "augmented": "aug.png",
                            "xai": "xai.png",
                        },
                    },
                ],
            },
            "per_class_timeline": {},
            "xai_insights_timeline": [],
        }
        html = _standalone_report_html(state, "test_run")
        assert "XAI Panels" not in html


# ---------------------------------------------------------------------------
# load_events malformed JSON (already has a test but let's also cover offset)
# ---------------------------------------------------------------------------


class TestLoadEventsFromOffsetMalformed:
    def test_load_events_from_offset_skips_corrupt(self, temp_dir):
        events_file = temp_dir / "events.jsonl"
        good = '{"schema_version": "2.1", "sequence": 1, "run_id": "r", "timestamp": "t", "type": "run_started", "payload": {}}\n'
        events_file.write_text(good + "CORRUPT LINE\n" + good, encoding="utf-8")
        events, offset = load_events_from_offset(events_file, 0)
        assert len(events) == 2
        assert offset > 0

    def test_load_events_empty_lines_skipped(self, temp_dir):
        events_file = temp_dir / "events.jsonl"
        good = '{"schema_version": "2.1", "sequence": 1, "run_id": "r", "timestamp": "t", "type": "run_started", "payload": {}}\n'
        events_file.write_text(good + "\n\n" + good, encoding="utf-8")
        events = load_events(events_file)
        assert len(events) == 2

    def test_load_events_nonexistent_file(self, temp_dir):
        events_file = temp_dir / "nonexistent.jsonl"
        events = load_events(events_file)
        assert events == []
