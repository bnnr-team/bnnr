"""Tests for bnnr.detection_adapter — DetectionAdapter lifecycle.

Covers init, train_step, eval_step, epoch_end_eval, epoch_end,
state_dict/load_state_dict, get_model, get_target_layers, and
the _targets_to_device helper.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor, nn

from bnnr.detection_adapter import (
    DetectionAdapter,
    UltralyticsDetectionAdapter,
    _det_images_to_float01,
    _targets_to_device,
)

# ---------------------------------------------------------------------------
# Helpers — tiny detection model that mimics the torchvision contract.
# ---------------------------------------------------------------------------


class _TinyDetectionModel(nn.Module):
    """Minimal detection model stub satisfying train/eval API."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(
        self,
        images: list[Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> Any:
        if self.training and targets is not None:
            # Return loss dict (training mode)
            dummy_loss = torch.tensor(0.5, requires_grad=True)
            return {"loss_classifier": dummy_loss, "loss_box_reg": dummy_loss * 0.3}
        else:
            # Return predictions list (eval mode)
            preds = []
            for img in images:
                preds.append({
                    "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([1]),
                })
            return preds


def _make_batch(n: int = 2, img_size: int = 32) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Create a synthetic detection batch."""
    images = torch.randn(n, 3, img_size, img_size)
    targets = []
    for _ in range(n):
        targets.append({
            "boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        })
    return images, targets


# ---------------------------------------------------------------------------
# _det_images_to_float01
# ---------------------------------------------------------------------------


class TestDetImagesToFloat01:
    def test_scales_clear_0_255_float_batch(self) -> None:
        x = torch.ones(1, 3, 2, 2) * 200.0
        out = _det_images_to_float01(x)
        assert float(out.max()) <= 1.0 + 1e-5
        assert torch.allclose(out, torch.ones_like(out) * (200.0 / 255.0))

    def test_clamps_mild_over_one_without_dividing_by_255(self) -> None:
        x = torch.ones(1, 3, 2, 2) * 1.2
        out = _det_images_to_float01(x)
        assert torch.allclose(out, torch.ones_like(out))


# ---------------------------------------------------------------------------
# _targets_to_device
# ---------------------------------------------------------------------------


class TestTargetsToDevice:
    def test_moves_tensors(self):
        targets = [
            {"boxes": torch.zeros(1, 4), "labels": torch.zeros(1, dtype=torch.long)},
        ]
        result = _targets_to_device(targets, "cpu")
        assert result[0]["boxes"].device.type == "cpu"

    def test_preserves_non_tensors(self):
        targets = [{"boxes": torch.zeros(1, 4), "meta": "hello"}]
        result = _targets_to_device(targets, "cpu")
        assert result[0]["meta"] == "hello"

    def test_empty_list(self):
        result = _targets_to_device([], "cpu")
        assert result == []


# ---------------------------------------------------------------------------
# DetectionAdapter
# ---------------------------------------------------------------------------


class TestDetectionAdapterInit:
    def test_default_init(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        assert adapter.device == "cpu"
        assert adapter.use_amp is False
        assert adapter.score_threshold == 0.05
        assert len(adapter.target_layers) > 0

    def test_custom_target_layers(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(
            model=model, optimizer=opt, device="cpu",
            target_layers=[model.conv],
        )
        assert adapter.target_layers == [model.conv]

    def test_auto_target_layers(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        # Should auto-detect the Conv2d layer
        assert isinstance(adapter.target_layers[0], nn.Conv2d)

    def test_scheduler_stored(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        adapter = DetectionAdapter(
            model=model, optimizer=opt, device="cpu",
            scheduler=sched,
        )
        assert adapter.scheduler is sched


class TestDetectionAdapterTrainStep:
    def test_train_step_returns_loss(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        images, targets = _make_batch(2)
        metrics = adapter.train_step((images, targets))
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] > 0

    def test_train_step_has_component_losses(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        images, targets = _make_batch(1)
        metrics = adapter.train_step((images, targets))
        # Should have loss_xxx keys from the model's loss dict
        loss_keys = [k for k in metrics if k.startswith("loss_")]
        assert len(loss_keys) >= 1


class TestDetectionAdapterEvalStep:
    def test_eval_step_accumulates_preds(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        images, targets = _make_batch(2)
        metrics = adapter.eval_step((images, targets))
        assert "loss" in metrics
        # Predictions should be accumulated
        assert len(adapter._eval_preds) == 2
        assert len(adapter._eval_targets) == 2

    def test_eval_step_filters_by_score(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(
            model=model, optimizer=opt, device="cpu",
            score_threshold=0.95,  # Higher than model's 0.9
        )
        images, targets = _make_batch(1)
        adapter.eval_step((images, targets))
        # All predictions should be filtered out since score=0.9 < threshold=0.95
        assert adapter._eval_preds[0]["boxes"].shape[0] == 0


class TestDetectionAdapterEpochEnd:
    def test_epoch_end_eval_computes_metrics(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        images, targets = _make_batch(2)
        adapter.eval_step((images, targets))
        metrics = adapter.epoch_end_eval()
        assert "map_50" in metrics
        assert "map_50_95" in metrics
        # Accumulators should be reset
        assert len(adapter._eval_preds) == 0
        assert len(adapter._eval_targets) == 0

    def test_epoch_end_eval_empty_returns_zeros(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        metrics = adapter.epoch_end_eval()
        assert metrics["map_50"] == 0.0
        assert metrics["map_50_95"] == 0.0

    def test_epoch_end_eval_snapshots_last_eval(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        images, targets = _make_batch(2)
        adapter.eval_step((images, targets))
        adapter.epoch_end_eval()
        # last_eval_preds/targets should be snapshot
        assert len(adapter.last_eval_preds) == 2
        assert len(adapter.last_eval_targets) == 2

    def test_epoch_end_steps_scheduler(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        adapter = DetectionAdapter(
            model=model, optimizer=opt, device="cpu", scheduler=sched,
        )
        lr_before = opt.param_groups[0]["lr"]
        adapter.epoch_end()
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after < lr_before

    def test_epoch_end_no_scheduler(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        adapter.epoch_end()  # Should not raise


class TestDetectionAdapterStateDict:
    def test_state_dict_contains_model_and_optimizer(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        state = adapter.state_dict()
        assert "model" in state
        assert "optimizer" in state

    def test_state_dict_contains_scheduler(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        adapter = DetectionAdapter(
            model=model, optimizer=opt, device="cpu", scheduler=sched,
        )
        state = adapter.state_dict()
        assert "scheduler" in state

    def test_load_state_dict_roundtrip(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")

        # Do a train step to change weights
        images, targets = _make_batch(1)
        adapter.train_step((images, targets))

        state = adapter.state_dict()

        # Create new adapter and load state
        model2 = _TinyDetectionModel()
        opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)
        adapter2 = DetectionAdapter(model=model2, optimizer=opt2, device="cpu")
        adapter2.load_state_dict(state)

        # Weights should match
        for (k1, v1), (k2, v2) in zip(
            adapter.model.state_dict().items(),
            adapter2.model.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)


class TestDetectionAdapterProtocol:
    def test_get_model(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        assert adapter.get_model() is model

    def test_get_target_layers(self):
        model = _TinyDetectionModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        adapter = DetectionAdapter(model=model, optimizer=opt, device="cpu")
        layers = adapter.get_target_layers()
        assert isinstance(layers, list)
        assert len(layers) > 0


# ---------------------------------------------------------------------------
# UltralyticsDetectionAdapter — mocked YOLO (no weights / GPU)
# ---------------------------------------------------------------------------


def _ultra_fake_inner_model() -> nn.Module:
    m = nn.Conv2d(3, 8, 3, padding=1)

    class _Args:
        box = 7.5

    m.args = _Args()
    return m


def _mock_yolo_results(count: int, *, with_boxes: bool) -> list[MagicMock]:
    out: list[MagicMock] = []
    for _ in range(count):
        r = MagicMock()
        if with_boxes:
            b = MagicMock()
            b.xyxy = torch.tensor([[1.0, 2.0, 10.0, 12.0]])
            b.conf = torch.tensor([0.88])
            b.cls = torch.tensor([3.0])
            r.boxes = b
        else:
            r.boxes = None
        out.append(r)
    return out


@pytest.fixture
def fake_ultralytics_modules() -> Any:
    """Stub ``ultralytics`` so tests run without the optional dependency."""
    import sys
    import types

    saved: dict[str, Any] = {}
    for key in ("ultralytics", "ultralytics.cfg"):
        if key in sys.modules:
            saved[key] = sys.modules[key]

    ultra = types.ModuleType("ultralytics")
    cfg_mod = types.ModuleType("ultralytics.cfg")

    def get_cfg() -> Any:
        ns = types.SimpleNamespace()
        ns.box = 7.5
        return ns

    cfg_mod.get_cfg = get_cfg
    sys.modules["ultralytics.cfg"] = cfg_mod
    sys.modules["ultralytics"] = ultra

    yield ultra

    for key in ("ultralytics", "ultralytics.cfg"):
        sys.modules.pop(key, None)
    for key, mod in saved.items():
        sys.modules[key] = mod


class TestUltralyticsPredictDetectionDicts:
    def test_predict_detection_dicts_one_per_batch_row(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        yolo_inst.model = _ultra_fake_inner_model()
        yolo_inst.predict.return_value = _mock_yolo_results(2, with_boxes=True)
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu")
        images = torch.rand(2, 3, 16, 16)
        preds = adapter.predict_detection_dicts(images)

        assert len(preds) == 2
        assert preds[0]["boxes"].shape == (1, 4)
        assert preds[0]["scores"].shape == (1,)
        assert preds[0]["labels"].shape == (1,)
        assert int(preds[0]["labels"][0].item()) == 3
        yolo_inst.predict.assert_called_once()

    def test_predict_passes_bchw_float01_to_yolo(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        yolo_inst.model = _ultra_fake_inner_model()
        captured: dict[str, Any] = {}

        def _capture_predict(*_a: Any, **kw: Any) -> Any:
            captured["source"] = kw.get("source")
            return _mock_yolo_results(1, with_boxes=True)

        yolo_inst.predict.side_effect = _capture_predict
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu")
        adapter.predict_detection_dicts(torch.rand(1, 3, 16, 16) * 255.0)
        src = captured["source"]
        assert isinstance(src, Tensor)
        assert float(src.max()) <= 1.0 + 1e-4
        assert float(src.min()) >= 0.0

    def test_predict_detection_dicts_none_boxes_empty_tensors(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        yolo_inst.model = _ultra_fake_inner_model()
        yolo_inst.predict.return_value = _mock_yolo_results(1, with_boxes=False)
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu")
        preds = adapter.predict_detection_dicts(torch.rand(1, 3, 8, 8))

        assert len(preds) == 1
        assert preds[0]["boxes"].shape == (0, 4)
        assert preds[0]["scores"].shape == (0,)
        assert preds[0]["labels"].shape == (0,)

    def test_predict_detection_dicts_pads_short_results(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        yolo_inst.model = _ultra_fake_inner_model()
        yolo_inst.predict.return_value = _mock_yolo_results(1, with_boxes=True)
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu")
        preds = adapter.predict_detection_dicts(torch.rand(3, 3, 8, 8))

        assert len(preds) == 3
        assert preds[0]["boxes"].shape[0] == 1
        assert preds[1]["boxes"].shape == (0, 4)
        assert preds[2]["boxes"].shape == (0, 4)

    def test_eval_step_extends_preds_aligned_with_targets(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        yolo_inst.model = _ultra_fake_inner_model()
        yolo_inst.predict.return_value = _mock_yolo_results(2, with_boxes=True)
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu")
        images, targets = _make_batch(2, img_size=16)
        adapter.eval_step((images, targets))

        assert len(adapter._eval_preds) == 2
        assert len(adapter._eval_targets) == 2
        assert adapter._eval_preds[0]["boxes"].numel() > 0


class TestUltralyticsTrainStepBatch:
    def test_loss_receives_float01_and_long_fields(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        inner = _ultra_fake_inner_model()
        captured: dict[str, Any] = {}

        def _loss(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
            captured["batch"] = batch
            dev = batch["img"].device
            t = torch.tensor(1.0, requires_grad=True, device=dev, dtype=torch.float32)
            return t, torch.zeros(3, device=dev, dtype=torch.float32)

        inner.loss = _loss  # type: ignore[method-assign]
        yolo_inst.model = inner
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu", use_amp=True)
        images = torch.rand(2, 3, 32, 32) * 255.0
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 20.0, 20.0]], dtype=torch.float32),
                "labels": torch.tensor([3]),
            },
            {
                "boxes": torch.tensor([[1.0, 1.0, 8.0, 9.0]], dtype=torch.float32),
                "labels": torch.tensor([1]),
            },
        ]
        metrics = adapter.train_step((images, targets))
        assert metrics["loss"] == pytest.approx(1.0)
        b = captured["batch"]
        assert float(b["img"].max()) <= 1.0 + 1e-5
        assert b["cls"].dtype == torch.int64
        assert b["batch_idx"].dtype == torch.int64

    def test_skips_degenerate_boxes_empty_batch(
        self, fake_ultralytics_modules: Any,
    ) -> None:
        mock_yolo_cls = MagicMock()
        yolo_inst = MagicMock()
        inner = _ultra_fake_inner_model()
        called: list[object] = []

        def _loss(batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
            called.append(batch)
            t = torch.tensor(1.0, requires_grad=True)
            return t, torch.zeros(3)

        inner.loss = _loss  # type: ignore[method-assign]
        yolo_inst.model = inner
        mock_yolo_cls.return_value = yolo_inst
        fake_ultralytics_modules.YOLO = mock_yolo_cls

        adapter = UltralyticsDetectionAdapter(model_name="dummy.pt", device="cpu")
        images = torch.rand(2, 3, 16, 16)
        targets = [
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0])},
            {"boxes": torch.tensor([[0.0, 0.0, 0.5, 0.5]]), "labels": torch.tensor([0])},
        ]
        metrics = adapter.train_step((images, targets))
        assert metrics.get("empty_targets") == 1.0
        assert called == []
