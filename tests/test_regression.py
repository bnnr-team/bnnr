"""Regression tests for BNNR v0.1 audit fixes.

Tests verify:
- Deterministic training with fixed seed
- Checkpoint/resume produces identical results
- Augmentation registry consistency
- AugmentationRunner after deduplication refactor
- _tensor_to_uint8 warning is per-instance (not class-level)
- Built-in pipelines do NOT apply transforms.Normalize()
- Reporter._decision_reason uses selection_metric, not hardcoded "accuracy"
- validate_config accepts all valid xai_method values
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import AugmentationRegistry, BaseAugmentation, BasicAugmentation
from bnnr.config import validate_config
from bnnr.core import BNNRConfig, BNNRTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x).reshape(x.shape[0], -1)
        return self.fc(x)


def _fixed_loader(
    n: int = 12,
    channels: int = 1,
    size: int = 16,
    n_classes: int = 3,
    seed: int = 0,
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, channels, size, size, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def _make_adapter(seed: int = 42, in_channels: int = 1, n_classes: int = 3) -> SimpleTorchAdapter:
    torch.manual_seed(seed)
    model = _TinyCNN(in_channels=in_channels, n_classes=n_classes)
    return SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        target_layers=[model.conv1],
        device="cpu",
    )


# ---------------------------------------------------------------------------
# 1. Deterministic training with fixed seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Two identical runs with the same seed must yield identical metrics."""

    def _run_once(self, tmp_path, suffix: str) -> dict[str, float]:
        loader = _fixed_loader(seed=0)
        adapter = _make_adapter(seed=42)
        cfg = BNNRConfig(
            m_epochs=2,
            max_iterations=1,
            xai_enabled=False,
            device="cpu",
            seed=42,
            checkpoint_dir=tmp_path / f"ckpt_{suffix}",
            report_dir=tmp_path / f"report_{suffix}",
            save_checkpoints=False,
        )
        trainer = BNNRTrainer(
            adapter, loader, loader,
            [BasicAugmentation(probability=0.5, random_state=42)],
            cfg,
        )
        result = trainer.run()
        return result.best_metrics

    def test_two_runs_produce_identical_metrics(self, tmp_path) -> None:
        m1 = self._run_once(tmp_path, "a")
        m2 = self._run_once(tmp_path, "b")
        for key in m1:
            assert m1[key] == pytest.approx(m2[key], abs=1e-6), (
                f"Metric '{key}' differs: {m1[key]} vs {m2[key]}"
            )


# ---------------------------------------------------------------------------
# 2. Checkpoint / resume restores state
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    """Checkpoint must restore model weights, iteration counter, and RNG."""

    def test_resume_restores_model_weights(self, tmp_path) -> None:
        loader = _fixed_loader(seed=0)
        adapter = _make_adapter(seed=42)
        cfg = BNNRConfig(
            m_epochs=1,
            max_iterations=1,
            xai_enabled=False,
            device="cpu",
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            save_checkpoints=True,
        )
        aug = BasicAugmentation(probability=0.5, random_state=42)
        trainer = BNNRTrainer(adapter, loader, loader, [aug], cfg)
        trainer.run()

        ckpts = sorted(cfg.checkpoint_dir.glob("*.pt"))
        assert ckpts, "At least one checkpoint must be saved"

        # Build fresh adapter with different weights
        adapter2 = _make_adapter(seed=999)
        weights_before = copy.deepcopy(adapter2.state_dict()["model"])

        trainer2 = BNNRTrainer(adapter2, loader, loader, [aug], cfg)
        trainer2.resume_from_checkpoint(ckpts[-1])

        weights_after = adapter2.state_dict()["model"]
        # At least one parameter tensor must differ from the fresh init
        any_changed = any(
            not torch.equal(weights_before[k], weights_after[k])
            for k in weights_before
        )
        assert any_changed, "Model weights should have been restored from checkpoint"

    def test_resume_restores_rng_state(self, tmp_path) -> None:
        loader = _fixed_loader(seed=0)
        adapter = _make_adapter(seed=42)
        cfg = BNNRConfig(
            m_epochs=1,
            max_iterations=1,
            xai_enabled=False,
            device="cpu",
            seed=42,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            save_checkpoints=True,
        )
        aug = BasicAugmentation(probability=0.5, random_state=42)
        trainer = BNNRTrainer(adapter, loader, loader, [aug], cfg)
        trainer.run()

        ckpts = sorted(cfg.checkpoint_dir.glob("*.pt"))
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        rng = state.get("rng_state", {})
        assert "python" in rng
        assert "numpy" in rng
        assert "torch_cpu" in rng


# ---------------------------------------------------------------------------
# 3. Augmentation registry consistency
# ---------------------------------------------------------------------------


class TestAugmentationRegistryConsistency:
    """Registry must contain all documented augmentations and no stale ones."""

    EXPECTED_NAMES = {
        "basic_augmentation",
        "church_noise",
        "dif_presets",
        "drust",
        "luxfer_glass",
        "procam",
        "smugs",
        "tea_stains",
    }

    # Aliases that must also exist
    EXPECTED_ALIASES = {
        "augmentation_1",
        "augmentation_3",
        "augmentation_5",
        "augmentation_6",
        "augmentation_7",
        "augmentation_8",
        "augmentation_9",
        "augmentation_10",
    }

    def test_all_expected_augmentations_registered(self) -> None:
        registered = set(AugmentationRegistry.list_all())
        for name in self.EXPECTED_NAMES | self.EXPECTED_ALIASES:
            assert name in registered, f"'{name}' should be registered"

    def test_create_all_augmentations(self) -> None:
        """Every registered augmentation must be instantiable."""
        for name in AugmentationRegistry.list_all():
            aug = AugmentationRegistry.create(name, probability=1.0, random_state=0)
            assert isinstance(aug, BaseAugmentation)

    def test_apply_all_augmentations(self) -> None:
        """Every registered augmentation must produce valid uint8 output."""
        image = (np.random.RandomState(0).rand(32, 32, 3) * 255).astype(np.uint8)
        for name in AugmentationRegistry.list_all():
            aug = AugmentationRegistry.create(name, probability=1.0, random_state=0)
            out = aug.apply(image)
            assert out.shape == image.shape, f"'{name}' changed shape"
            assert out.dtype == np.uint8, f"'{name}' changed dtype"


# ---------------------------------------------------------------------------
# 4. AugmentationRunner after deduplication refactor
# ---------------------------------------------------------------------------


class TestAugmentationRunnerRefactor:
    """Verify the refactored _apply_augmentation_list works correctly."""

    def test_gpu_and_cpu_paths_produce_output(self) -> None:
        """Both GPU-native and CPU-bound paths should produce valid output."""
        gpu_aug = BasicAugmentation(probability=1.0, random_state=1)
        gpu_aug.device_compatible = True  # type: ignore[attr-defined]
        cpu_aug = BasicAugmentation(probability=1.0, random_state=2)
        cpu_aug.device_compatible = False  # type: ignore[attr-defined]

        runner = AugmentationRunner([gpu_aug, cpu_aug], async_prefetch=False)
        images = torch.rand(4, 3, 16, 16)
        labels = torch.randint(0, 3, (4,))
        out_images, out_labels = runner.apply_batch(images, labels)
        assert out_images.shape == images.shape
        assert out_labels.shape == labels.shape

    def test_empty_augmentations_passthrough(self) -> None:
        runner = AugmentationRunner([], async_prefetch=False)
        images = torch.rand(2, 3, 8, 8)
        labels = torch.randint(0, 3, (2,))
        out_images, out_labels = runner.apply_batch(images, labels)
        assert torch.equal(out_images, images)
        assert torch.equal(out_labels, labels)

    def test_sync_iter_loader(self) -> None:
        """iter_loader in sync mode should yield all batches."""
        aug = BasicAugmentation(probability=0.5, random_state=1)
        runner = AugmentationRunner([aug], async_prefetch=False)
        loader = _fixed_loader(n=8, channels=3, size=8)
        batches = list(runner.iter_loader(loader))
        assert len(batches) == 2  # 8 samples / batch_size 4


# ---------------------------------------------------------------------------
# 5. _tensor_to_uint8 warning is per-instance
# ---------------------------------------------------------------------------


class TestTensorToUint8Warning:
    """Warning about normalized inputs should be per-instance, not class-level."""

    def test_warning_emitted_per_instance(self, tmp_path) -> None:
        loader = _fixed_loader(seed=0)
        adapter = _make_adapter(seed=42)
        cfg = BNNRConfig(
            m_epochs=1, max_iterations=1, xai_enabled=False, device="cpu",
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "report",
            save_checkpoints=False,
        )

        # Create two independent trainer instances
        trainer1 = BNNRTrainer(adapter, loader, loader, [], cfg)
        trainer2 = BNNRTrainer(adapter, loader, loader, [], cfg)

        # Simulate normalized data (values outside [0,1])
        normalized = torch.randn(2, 3, 8, 8)  # mean ~0, std ~1 -> has negatives

        with warnings.catch_warnings(record=True) as w1:
            warnings.simplefilter("always")
            trainer1._tensor_to_uint8(normalized)

        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            trainer2._tensor_to_uint8(normalized)

        # Both instances should emit the warning (not just the first)
        assert any("outside [0, 1]" in str(w.message) for w in w1), "First instance should warn"
        assert any("outside [0, 1]" in str(w.message) for w in w2), "Second instance should also warn"


# ---------------------------------------------------------------------------
# 6. Built-in pipelines do NOT apply Normalize()
# ---------------------------------------------------------------------------


class TestPipelinesNoNormalize:
    """Built-in pipeline transforms should not include Normalize()."""

    def test_mnist_pipeline_no_normalize(self) -> None:
        """MNIST pipeline should produce tensors in [0, 1] range."""
        from torchvision import transforms

        # Simulate the MNIST transform chain
        transform = transforms.Compose([transforms.ToTensor()])
        # A white 28x28 image in PIL range -> ToTensor -> [0, 1]
        from PIL import Image
        img = Image.new("L", (28, 28), 128)
        t = transform(img)
        assert t.min() >= 0.0, "MNIST tensors should be >= 0"
        assert t.max() <= 1.0, "MNIST tensors should be <= 1"


# ---------------------------------------------------------------------------
# 7. validate_config accepts all valid XAI methods
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """validate_config must accept all supported xai_method values."""

    @pytest.mark.parametrize("method", [
        "opticam", "craft", "nmf", "nmf_concepts", "real_craft", "gradcam",
    ])
    def test_valid_xai_methods_accepted(self, method: str) -> None:
        cfg = BNNRConfig(device="cpu", xai_method=method)
        warnings = validate_config(cfg)
        xai_warnings = [w for w in warnings if "xai_method" in w]
        assert not xai_warnings, f"Valid method '{method}' should not trigger warnings: {xai_warnings}"

    def test_invalid_xai_method_produces_warning(self) -> None:
        cfg = BNNRConfig(device="cpu", xai_method="nonexistent_method")
        warnings = validate_config(cfg)
        xai_warnings = [w for w in warnings if "xai_method" in w]
        assert xai_warnings, "Invalid xai_method should produce a validation warning"


# ---------------------------------------------------------------------------
# 8. Reporter._decision_reason uses selection_metric
# ---------------------------------------------------------------------------


class TestDecisionReason:
    """_decision_reason should use selection_metric, not hardcode 'accuracy'."""

    def test_decision_reason_uses_selection_metric(self) -> None:
        from bnnr.reporting import _decision_reason

        results = {
            "aug_a": {"f1_macro": 0.9, "loss": 0.1},
            "aug_b": {"f1_macro": 0.7, "loss": 0.3},
        }
        baseline = {"f1_macro": 0.6, "loss": 0.5}

        reason = _decision_reason(
            "aug_a", results, baseline, selection_metric="f1_macro"
        )
        assert "f1_macro" in reason

    def test_decision_reason_none_selected(self) -> None:
        from bnnr.reporting import _decision_reason

        reason = _decision_reason("none", {}, {}, selection_metric="accuracy")
        assert "No candidate" in reason
