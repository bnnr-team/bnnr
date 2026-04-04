"""Extended tests for bnnr.core — uncovered BNNRConfig validators,
BNNRTrainer helper methods, and edge cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.augmentations import BasicAugmentation
from bnnr.core import BNNRConfig, BNNRTrainer

# ---------------------------------------------------------------------------
# BNNRConfig validators
# ---------------------------------------------------------------------------


class TestBNNRConfigValidators:
    def test_invalid_task(self):
        with pytest.raises(ValidationError, match="task"):
            BNNRConfig(task="segmentation")

    def test_invalid_selection_mode(self):
        with pytest.raises(ValidationError, match="selection_mode"):
            BNNRConfig(selection_mode="median")

    def test_invalid_device(self):
        with pytest.raises(ValidationError, match="device"):
            BNNRConfig(device="tpu")

    def test_invalid_report_preview_size(self):
        with pytest.raises(ValidationError):
            BNNRConfig(report_preview_size=0)

    def test_invalid_report_xai_size(self):
        with pytest.raises(ValidationError):
            BNNRConfig(report_xai_size=-5)

    def test_invalid_candidate_pruning_threshold(self):
        with pytest.raises(ValidationError):
            BNNRConfig(candidate_pruning_relative_threshold=0.0)

    def test_invalid_candidate_pruning_warmup(self):
        with pytest.raises(ValidationError):
            BNNRConfig(candidate_pruning_warmup_epochs=0)

    def test_invalid_event_sample_every_epochs(self):
        with pytest.raises(ValidationError):
            BNNRConfig(event_sample_every_epochs=0)

    def test_invalid_event_xai_every_epochs(self):
        with pytest.raises(ValidationError):
            BNNRConfig(event_xai_every_epochs=0)

    def test_invalid_event_min_interval_seconds(self):
        with pytest.raises(ValidationError):
            BNNRConfig(event_min_interval_seconds=-1.0)

    def test_invalid_xai_selection_weight(self):
        with pytest.raises(ValidationError):
            BNNRConfig(xai_selection_weight=1.5)

    def test_invalid_xai_pruning_threshold(self):
        with pytest.raises(ValidationError):
            BNNRConfig(xai_pruning_threshold=-0.1)

    def test_invalid_multilabel_threshold_too_low(self):
        with pytest.raises(ValidationError, match="multilabel_threshold"):
            BNNRConfig(multilabel_threshold=0.0)

    def test_invalid_multilabel_threshold_too_high(self):
        with pytest.raises(ValidationError, match="multilabel_threshold"):
            BNNRConfig(multilabel_threshold=1.0)

    def test_valid_multilabel_config(self):
        cfg = BNNRConfig(
            task="multilabel",
            multilabel_threshold=0.5,
        )
        assert cfg.task == "multilabel"

    def test_report_probe_controls_validation(self):
        with pytest.raises(ValidationError):
            BNNRConfig(report_probe_images_per_class=0)
        with pytest.raises(ValidationError):
            BNNRConfig(report_probe_max_classes=0)

    def test_valid_all_xai_methods(self):
        for method in ("opticam", "gradcam", "craft", "nmf", "nmf_concepts", "real_craft"):
            cfg = BNNRConfig(xai_method=method)
            assert cfg.xai_method == method

# ---------------------------------------------------------------------------
# BNNRTrainer helper methods
# ---------------------------------------------------------------------------


class _TinyCNN(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = self.pool(x).reshape(x.shape[0], -1)
        return self.fc(x)


def _make_loader(n: int = 12, channels: int = 1, size: int = 8, n_classes: int = 3) -> DataLoader:
    torch.manual_seed(0)
    x = torch.rand(n, channels, size, size)
    y = torch.randint(0, n_classes, (n,))
    idx = torch.arange(n)
    return DataLoader(TensorDataset(x, y, idx), batch_size=4)


def _make_trainer(
    n_classes: int = 3,
    **config_overrides: Any,
) -> BNNRTrainer:
    model = _TinyCNN(n_classes=n_classes)
    cfg_kwargs = dict(
        m_epochs=1,
        max_iterations=1,
        device="cpu",
        seed=42,
        xai_enabled=False,
        save_checkpoints=False,
        event_log_enabled=False,
    )
    cfg_kwargs.update(config_overrides)
    cfg = BNNRConfig(**cfg_kwargs)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        target_layers=[model.conv],
        device="cpu",
    )
    train_loader = _make_loader(n_classes=n_classes)
    val_loader = _make_loader(n=8, n_classes=n_classes)
    augs = [BasicAugmentation(probability=0.5, random_state=42)]
    return BNNRTrainer(adapter, train_loader, val_loader, augs, cfg)


class TestBNNRTrainerToJson:
    def test_to_json_returns_valid_json(self):
        import json

        trainer = _make_trainer()
        j = trainer.to_json()
        data = json.loads(j)
        assert "current_iteration" in data
        assert "best_augmentation" in data
        assert "active_augmentations" in data

    def test_to_json_changes_after_run(self):
        import json

        trainer = _make_trainer()
        json.loads(trainer.to_json())  # sanity check
        trainer.run()
        j_after = json.loads(trainer.to_json())
        # After run, there should be a best_augmentation or at least active_augmentations
        assert isinstance(j_after["active_augmentations"], list)


class TestXaiMeanQuality:
    def test_empty_diagnoses_returns_none(self):
        result = BNNRTrainer._xai_mean_quality({})
        assert result is None

    def test_single_class(self):
        diag = {"0": {"quality_score": 0.8, "severity": "ok"}}
        result = BNNRTrainer._xai_mean_quality(diag)
        assert result == pytest.approx(0.8)

    def test_multiple_classes(self):
        diag = {
            "0": {"quality_score": 0.6},
            "1": {"quality_score": 0.9},
            "2": {"quality_score": 0.3},
        }
        result = BNNRTrainer._xai_mean_quality(diag)
        assert result == pytest.approx(0.6)

    def test_no_quality_scores(self):
        diag = {"0": {"severity": "ok"}, "1": {"severity": "warning"}}
        result = BNNRTrainer._xai_mean_quality(diag)
        assert result is None


class TestTensorBatchToPreviewUint8:
    """Test the _tensor_batch_to_preview_uint8 with different value ranges."""

    def test_zero_one_range(self):
        trainer = _make_trainer()
        imgs = torch.rand(2, 3, 8, 8)  # values in [0, 1]
        result = trainer._tensor_batch_to_preview_uint8(imgs)
        assert result.dtype == np.uint8
        assert result.shape == (2, 8, 8, 3)
        assert result.max() <= 255

    def test_zero_255_range(self):
        trainer = _make_trainer()
        imgs = torch.randint(0, 255, (2, 3, 8, 8)).float()
        result = trainer._tensor_batch_to_preview_uint8(imgs)
        assert result.dtype == np.uint8

    def test_negative_range_normalized(self):
        trainer = _make_trainer()
        # Simulate ImageNet-normalized data
        imgs = torch.randn(2, 3, 8, 8) * 2 - 1
        result = trainer._tensor_batch_to_preview_uint8(imgs)
        assert result.dtype == np.uint8

    def test_denormalization_with_config(self):
        trainer = _make_trainer(
            denormalization_mean=[0.485, 0.456, 0.406],
            denormalization_std=[0.229, 0.224, 0.225],
        )
        # Create normalized data
        imgs = torch.randn(1, 3, 8, 8)
        result = trainer._tensor_batch_to_preview_uint8(imgs)
        assert result.dtype == np.uint8

    def test_constant_image_does_not_crash(self):
        trainer = _make_trainer()
        imgs = torch.full((1, 3, 8, 8), 5.0)
        result = trainer._tensor_batch_to_preview_uint8(imgs)
        assert result.dtype == np.uint8

    def test_single_channel_grayscale(self):
        trainer = _make_trainer()
        imgs = torch.rand(1, 1, 8, 8)
        result = trainer._tensor_batch_to_preview_uint8(imgs)
        assert result.dtype == np.uint8
        assert result.shape == (1, 8, 8, 1)


class TestSelectionModeMin:
    """Test that selection_mode='min' works (e.g. when tracking loss)."""

    def test_min_mode_runs(self):
        trainer = _make_trainer(
            selection_metric="loss",
            selection_mode="min",
        )
        result = trainer.run()
        assert result.best_metrics is not None


class TestReEvalBaseline:
    """Test reeval_baseline_per_iteration flag."""

    def test_reeval_baseline_runs(self):
        trainer = _make_trainer(reeval_baseline_per_iteration=True)
        result = trainer.run()
        assert result.best_metrics is not None
