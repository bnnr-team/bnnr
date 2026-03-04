"""Multi-label classification support tests.

Covers:
  1. BNNRConfig accepts task="multilabel" and auto-defaults
  2. SimpleTorchAdapter multilabel train_step / eval_step
  3. calculate_metrics 2D (multi-label) path
  4. calculate_metrics 1D (single-label) regression guard
  5. ICD with multi-hot labels
  6. BNNRTrainer mini-run (multilabel, XAI disabled)
  7. BNNRTrainer mini-run (multilabel, XAI enabled)
  8. Probe initialization with multi-hot labels
  9. Events replay produces correct metric_units for multilabel
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.augmentations import BasicAugmentation
from bnnr.config import validate_config
from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.events import JsonlEventSink, load_events, replay_events
from bnnr.icd import ICD
from bnnr.utils import calculate_metrics
from bnnr.xai_cache import XAICache

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants & helpers
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
IMG_SIZE = 32
N_CLASSES = 6
N_SAMPLES = 24
BATCH_SIZE = 8


class TinyMultilabelCNN(nn.Module):
    """Minimal CNN outputting [B, N_CLASSES] logits."""

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_multilabel_dataset(
    n: int = N_SAMPLES,
    n_classes: int = N_CLASSES,
    img_size: int = IMG_SIZE,
) -> TensorDataset:
    """Create a TensorDataset with multi-hot labels."""
    images = torch.rand(n, 3, img_size, img_size)
    # Random multi-hot labels: at least one class active per sample
    labels = torch.zeros(n, n_classes)
    for i in range(n):
        # Set 1-3 random active labels
        active = torch.randperm(n_classes)[: torch.randint(1, 4, (1,)).item()]
        labels[i, active] = 1.0
    return TensorDataset(images, labels)


def _make_multilabel_loaders(
    n_train: int = N_SAMPLES,
    n_val: int = 12,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    train_ds = _make_multilabel_dataset(n=n_train)
    val_ds = _make_multilabel_dataset(n=n_val)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  1. BNNRConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultilabelConfig:
    def test_config_accepts_multilabel_task(self, tmp_path: Path) -> None:
        cfg = BNNRConfig(
            task="multilabel",
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            device="cpu",
        )
        assert cfg.task == "multilabel"

    def test_config_auto_defaults(self, tmp_path: Path) -> None:
        cfg = BNNRConfig(
            task="multilabel",
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            device="cpu",
        )
        assert cfg.selection_metric == "f1_samples"
        assert "f1_samples" in cfg.metrics
        assert "f1_macro" in cfg.metrics
        assert "accuracy" in cfg.metrics
        assert "loss" in cfg.metrics

    def test_config_rejects_invalid_task(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            BNNRConfig(
                task="invalid_task",
                m_epochs=1,
                max_iterations=1,
                checkpoint_dir=tmp_path / "ckpt",
                report_dir=tmp_path / "report",
                device="cpu",
            )

    def test_config_user_override_preserved(self, tmp_path: Path) -> None:
        cfg = BNNRConfig(
            task="multilabel",
            selection_metric="f1_macro",
            metrics=["f1_macro", "loss"],
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            device="cpu",
        )
        assert cfg.selection_metric == "f1_macro"
        assert cfg.metrics == ["f1_macro", "loss"]

    def test_validate_config_multilabel(self, tmp_path: Path) -> None:
        cfg = BNNRConfig(
            task="multilabel",
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            device="cpu",
        )
        warnings = validate_config(cfg)
        assert not any("selection_metric" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. SimpleTorchAdapter multilabel
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultilabelAdapter:
    def test_train_step_returns_metrics(self) -> None:
        model = TinyMultilabelCNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            multilabel=True,
        )

        images = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        for i in range(BATCH_SIZE):
            active = torch.randperm(N_CLASSES)[:2]
            labels[i, active] = 1.0

        metrics = adapter.train_step((images, labels))
        assert "loss" in metrics
        assert "f1_samples" in metrics
        assert "f1_macro" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)

    def test_eval_step_returns_metrics(self) -> None:
        model = TinyMultilabelCNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            multilabel=True,
        )

        images = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
        labels = torch.zeros(BATCH_SIZE, N_CLASSES)
        for i in range(BATCH_SIZE):
            active = torch.randperm(N_CLASSES)[:2]
            labels[i, active] = 1.0

        metrics = adapter.eval_step((images, labels))
        assert "loss" in metrics
        assert "f1_samples" in metrics
        assert "f1_macro" in metrics
        assert "accuracy" in metrics

    def test_default_eval_metrics_for_multilabel(self) -> None:
        model = TinyMultilabelCNN()
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            device="cpu",
            multilabel=True,
        )
        assert "f1_samples" in adapter.eval_metrics
        assert "f1_macro" in adapter.eval_metrics
        assert "accuracy" in adapter.eval_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  3. calculate_metrics 2D (multi-label path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateMetrics2D:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
        y_pred = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
        metrics = calculate_metrics(y_pred, y_true, metrics=["accuracy", "f1_samples", "f1_macro"])
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_samples"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_zero_predictions(self) -> None:
        y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_pred = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        metrics = calculate_metrics(y_pred, y_true, metrics=["accuracy", "f1_samples"])
        assert metrics["accuracy"] == 0.0
        assert metrics["f1_samples"] == 0.0

    def test_partial_predictions(self) -> None:
        y_true = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
        y_pred = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        metrics = calculate_metrics(y_pred, y_true, metrics=["f1_samples", "precision", "recall"])
        # Each sample has 1 correct of 2 — precision=1.0, recall=0.5
        assert 0.0 < metrics["f1_samples"] < 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5

    def test_hamming_metric(self) -> None:
        y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_pred = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        metrics = calculate_metrics(y_pred, y_true, metrics=["hamming"])
        assert metrics["hamming"] == 1.0  # 1 - hamming_loss, perfect match

    def test_unsupported_metric_raises(self) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="Unsupported metric"):
            calculate_metrics(y_pred, y_true, metrics=["nonexistent"])


# ═══════════════════════════════════════════════════════════════════════════════
#  4. calculate_metrics 1D (regression guard)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateMetrics1D:
    def test_single_label_unchanged(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        metrics = calculate_metrics(y_pred, y_true, metrics=["accuracy", "f1_macro"])
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_single_label_imperfect(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 0, 2, 0, 1])
        metrics = calculate_metrics(y_pred, y_true, metrics=["accuracy", "f1_macro"])
        assert 0.0 < metrics["accuracy"] < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ICD with multi-hot labels
# ═══════════════════════════════════════════════════════════════════════════════


class TestICDMultilabel:
    def test_apply_with_multihot_label(self, tmp_path: Path) -> None:
        model = TinyMultilabelCNN()
        cache = XAICache(tmp_path / "xai_cache")
        icd = ICD(
            model=model,
            target_layers=[model.conv1],
            cache=cache,
            probability=1.0,
            random_state=SEED,
        )
        image = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # Pass a multi-hot label as numpy array
        label = np.array([0, 0, 1, 0, 1, 0])
        result = icd.apply_with_label(image, label=label)
        assert result.shape == image.shape

    def test_batch_with_multihot_labels(self, tmp_path: Path) -> None:
        model = TinyMultilabelCNN()
        cache = XAICache(tmp_path / "xai_cache")
        icd = ICD(
            model=model,
            target_layers=[model.conv1],
            cache=cache,
            probability=1.0,
            random_state=SEED,
        )
        images = np.random.randint(0, 255, (3, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        # 2D multi-hot labels
        labels = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 1]])
        result = icd.apply_batch_with_labels(images, labels)
        assert result.shape == images.shape

    def test_batch_with_scalar_labels_still_works(self, tmp_path: Path) -> None:
        """Regression guard: single-label (1D) labels still work."""
        model = TinyMultilabelCNN()
        cache = XAICache(tmp_path / "xai_cache")
        icd = ICD(
            model=model,
            target_layers=[model.conv1],
            cache=cache,
            probability=1.0,
            random_state=SEED,
        )
        images = np.random.randint(0, 255, (3, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        labels = np.array([0, 1, 2])
        result = icd.apply_batch_with_labels(images, labels)
        assert result.shape == images.shape


# ═══════════════════════════════════════════════════════════════════════════════
#  6. BNNRTrainer mini-run (multilabel, XAI disabled)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultilabelTrainer:
    def test_mini_run_xai_disabled(self, tmp_path: Path) -> None:
        model = TinyMultilabelCNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            multilabel=True,
        )

        config = BNNRConfig(
            task="multilabel",
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            xai_enabled=False,
            device="cpu",
            seed=SEED,
        )

        augmentations = [BasicAugmentation(probability=1.0)]

        train_loader, val_loader = _make_multilabel_loaders()
        trainer = BNNRTrainer(
            config=config,
            model=adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=augmentations,
        )
        result = trainer.run()
        assert result is not None
        assert hasattr(result, "best_metrics")

    # ═══════════════════════════════════════════════════════════════════════
    #  7. BNNRTrainer mini-run (multilabel, XAI enabled)
    # ═══════════════════════════════════════════════════════════════════════

    def test_mini_run_xai_enabled(self, tmp_path: Path) -> None:
        model = TinyMultilabelCNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            target_layers=[model.conv1],
            device="cpu",
            multilabel=True,
        )

        config = BNNRConfig(
            task="multilabel",
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            xai_enabled=True,
            xai_method="opticam",
            device="cpu",
            seed=SEED,
        )

        augmentations = [BasicAugmentation(probability=1.0)]

        train_loader, val_loader = _make_multilabel_loaders()
        trainer = BNNRTrainer(
            config=config,
            model=adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=augmentations,
        )
        result = trainer.run()
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Probe initialization with multi-hot labels
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultilabelProbe:
    def test_probe_distributes_across_active_classes(self, tmp_path: Path) -> None:
        model = TinyMultilabelCNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device="cpu",
            multilabel=True,
        )

        config = BNNRConfig(
            task="multilabel",
            m_epochs=1,
            max_iterations=1,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            xai_enabled=False,
            device="cpu",
            seed=SEED,
            report_probe_images_per_class=2,
            report_probe_max_classes=6,
        )

        augmentations = [BasicAugmentation(probability=1.0)]
        train_loader, val_loader = _make_multilabel_loaders(n_train=24, n_val=24)
        trainer = BNNRTrainer(
            config=config,
            model=adapter,
            train_loader=train_loader,
            val_loader=val_loader,
            augmentations=augmentations,
        )
        trainer._initialize_report_probe_samples()
        assert trainer._report_probe_images is not None
        assert trainer._report_probe_labels is not None
        # Should have probes from multiple classes
        unique_labels = set(trainer._report_probe_labels.numpy().tolist())
        assert len(unique_labels) > 1


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Events replay produces correct metric_units for multilabel
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultilabelEvents:
    def test_replay_events_metric_units(self, tmp_path: Path) -> None:
        events_path = tmp_path / "events.jsonl"
        sink = JsonlEventSink(events_path, run_id="test_ml_run")
        sink.emit(
            "run_started",
            {
                "config": {"task": "multilabel"},
            },
        )
        sink.close()

        events = load_events(events_path)
        state = replay_events(events)
        run = state.get("run", {})
        units = run.get("metric_units", {})
        assert units.get("f1_samples") == "%"
        assert units.get("f1_macro") == "%"
        assert units.get("accuracy") == "%"
        assert units.get("loss") == "unitless"
        assert state.get("task") == "multilabel"

    def test_replay_events_classification_unchanged(self, tmp_path: Path) -> None:
        """Regression guard: classification metric_units are unchanged."""
        events_path = tmp_path / "events.jsonl"
        sink = JsonlEventSink(events_path, run_id="test_cls_run")
        sink.emit(
            "run_started",
            {
                "config": {"task": "classification"},
            },
        )
        sink.close()

        events = load_events(events_path)
        state = replay_events(events)
        run = state.get("run", {})
        units = run.get("metric_units", {})
        assert units.get("accuracy") == "%"
        assert units.get("f1_macro") == "%"
        assert units.get("loss") == "unitless"
        # Should NOT contain multilabel metrics
        assert "f1_samples" not in units
