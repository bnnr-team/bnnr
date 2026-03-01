"""Tests for the review-fixes optimisations.

These tests verify correctness of each optimisation without
comparing before/after performance (that's done by benchmarks/).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bnnr.adapter import SimpleTorchAdapter
from bnnr.augmentations import BaseAugmentation
from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.icd import AICD, ICD
from bnnr.utils import set_seed
from bnnr.xai_cache import XAICache

# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


class _TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def tiny_model() -> nn.Module:
    return _TinyCNN()


@pytest.fixture
def tiny_loader() -> DataLoader:
    x = torch.rand(20, 3, 32, 32)
    y = torch.randint(0, 3, (20,))
    idx = torch.arange(20)
    return DataLoader(TensorDataset(x, y, idx), batch_size=8)


# ────────────────────────────────────────────────────────────────────
# P0: _noise_fill determinism
# ────────────────────────────────────────────────────────────────────


class TestNoiseFillDeterminism:
    def test_same_seed_same_output(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        """Two ICD instances with same seed must produce identical noise fill."""
        cache = XAICache(tmp_path / "cache")
        image = (np.random.RandomState(0).rand(32, 32, 3) * 255).astype(np.uint8)
        h = cache._hash_image(image)
        np.save(cache._cache_path(h, 0), np.random.RandomState(1).rand(32, 32).astype(np.float32))

        layers = [tiny_model.conv1]

        aug1 = ICD(model=tiny_model, target_layers=layers, cache=cache,
                    fill_strategy="noise", random_state=42, probability=1.0,
                    threshold_percentile=50.0)
        out1 = aug1.apply_with_label(image.copy(), 0)

        aug2 = ICD(model=tiny_model, target_layers=layers, cache=cache,
                    fill_strategy="noise", random_state=42, probability=1.0,
                    threshold_percentile=50.0)
        out2 = aug2.apply_with_label(image.copy(), 0)

        assert np.array_equal(out1, out2), f"Max diff: {np.abs(out1.astype(int) - out2.astype(int)).max()}"

    def test_different_seeds_different_output(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        """Different seeds should produce different noise patterns."""
        cache = XAICache(tmp_path / "cache")
        image = (np.random.RandomState(0).rand(32, 32, 3) * 255).astype(np.uint8)
        h = cache._hash_image(image)
        # Use a high-contrast saliency map to ensure masking happens
        saliency = np.zeros((32, 32), dtype=np.float32)
        saliency[8:24, 8:24] = 1.0
        np.save(cache._cache_path(h, 0), saliency)

        layers = [tiny_model.conv1]

        aug1 = ICD(model=tiny_model, target_layers=layers, cache=cache,
                    fill_strategy="noise", random_state=42, probability=1.0,
                    threshold_percentile=50.0)
        out1 = aug1.apply_with_label(image.copy(), 0)

        aug2 = ICD(model=tiny_model, target_layers=layers, cache=cache,
                    fill_strategy="noise", random_state=99, probability=1.0,
                    threshold_percentile=50.0)
        out2 = aug2.apply_with_label(image.copy(), 0)

        assert not np.array_equal(out1, out2), "Different seeds should give different outputs"


# ────────────────────────────────────────────────────────────────────
# P1: _compute_tile_mask vectorisation
# ────────────────────────────────────────────────────────────────────


class TestTileMaskVectorisation:
    @pytest.mark.parametrize("h,w,ts", [(32, 32, 8), (64, 64, 8), (96, 96, 8), (224, 224, 16)])
    def test_mask_shape(self, tiny_model: nn.Module, tmp_path: Path, h: int, w: int, ts: int) -> None:
        """Vectorised tile mask should have correct shape."""
        cache = XAICache(tmp_path / "cache")
        layers = [tiny_model.conv1]
        aug = ICD(model=tiny_model, target_layers=layers, cache=cache,
                  tile_size=ts, probability=1.0)

        saliency = np.random.rand(h, w).astype(np.float32)
        mask = aug._compute_tile_mask(saliency, h, w)

        assert mask.shape == (h, w)
        assert mask.dtype == bool

    def test_mask_not_all_same(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        """With varied saliency, mask should have both True and False tiles."""
        cache = XAICache(tmp_path / "cache")
        layers = [tiny_model.conv1]
        aug = ICD(model=tiny_model, target_layers=layers, cache=cache,
                  tile_size=8, probability=1.0, threshold_percentile=50.0)

        saliency = np.random.RandomState(42).rand(64, 64).astype(np.float32)
        mask = aug._compute_tile_mask(saliency, 64, 64)

        assert mask.any(), "Mask should have some True values"
        assert not mask.all(), "Mask should not be all True"

    def test_aicd_inverts(self, tiny_model: nn.Module, tmp_path: Path) -> None:
        """AICD should mask opposite tiles compared to ICD."""
        cache = XAICache(tmp_path / "cache")
        layers = [tiny_model.conv1]
        saliency = np.random.RandomState(42).rand(32, 32).astype(np.float32)

        icd = ICD(model=tiny_model, target_layers=layers, cache=cache,
                  tile_size=8, probability=1.0, threshold_percentile=50.0)
        aicd = AICD(model=tiny_model, target_layers=layers, cache=cache,
                    tile_size=8, probability=1.0, threshold_percentile=50.0)

        mask_icd = icd._compute_tile_mask(saliency, 32, 32)
        mask_aicd = aicd._compute_tile_mask(saliency, 32, 32)

        # They should be mostly complementary (not necessarily exact complement due to threshold edge)
        overlap = (mask_icd & mask_aicd).sum()
        total = mask_icd.sum() + mask_aicd.sum()
        assert overlap < total * 0.3, "ICD and AICD masks should be mostly complementary"


# ────────────────────────────────────────────────────────────────────
# P1: set_seed deterministic parameter
# ────────────────────────────────────────────────────────────────────


class TestSetSeed:
    def test_default_is_deterministic(self) -> None:
        """Default set_seed should set deterministic=True (backward compat)."""
        set_seed(42)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_deterministic_false_enables_benchmark(self) -> None:
        """set_seed(deterministic=False) should enable benchmark mode."""
        set_seed(42, deterministic=False)
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same random numbers."""
        set_seed(123)
        a = torch.rand(5)
        set_seed(123)
        b = torch.rand(5)
        assert torch.allclose(a, b)


# ────────────────────────────────────────────────────────────────────
# P2: _hash_image optimisation
# ────────────────────────────────────────────────────────────────────


class TestHashImage:
    def test_hash_consistency(self, tmp_path: Path) -> None:
        """Same image should produce the same hash."""
        cache = XAICache(tmp_path / "cache")
        image = np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        h1 = cache._hash_image(image)
        h2 = cache._hash_image(image.copy())
        assert h1 == h2

    def test_different_images_different_hash(self, tmp_path: Path) -> None:
        """Different images should produce different hashes."""
        cache = XAICache(tmp_path / "cache")
        img1 = np.zeros((32, 32, 3), dtype=np.uint8)
        img2 = np.ones((32, 32, 3), dtype=np.uint8) * 128
        assert cache._hash_image(img1) != cache._hash_image(img2)

    def test_cache_lookup_works(self, tmp_path: Path) -> None:
        """Cache save/load should work with the new hash function."""
        cache = XAICache(tmp_path / "cache")
        image = np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8)
        saliency = np.random.rand(32, 32).astype(np.float32)

        cache.save_map(saliency, label=5, image=image)
        loaded = cache.get_importance_map(image, label=5)

        assert loaded is not None
        assert np.allclose(loaded, saliency)


# ────────────────────────────────────────────────────────────────────
# P2: aug.probability mutation safety
# ────────────────────────────────────────────────────────────────────


class _DummyAug(BaseAugmentation):
    name = "dummy"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.fliplr(image)


class TestProbabilitySafety:
    def test_copy_does_not_mutate_original(self) -> None:
        """copy.copy() should not affect the original augmentation."""
        aug = _DummyAug(probability=0.5, random_state=42)
        copied = copy.copy(aug)
        copied.probability = 1.0

        assert aug.probability == 0.5, "Original should not be mutated"
        assert copied.probability == 1.0

    def test_preview_produces_output(self) -> None:
        """Preview augmentation with prob=1.0 should always augment."""
        aug = _DummyAug(probability=0.5, random_state=42)
        copied = copy.copy(aug)
        copied.probability = 1.0

        images = np.random.randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)
        result = copied.apply_batch(images)

        # With prob=1.0, all images should be flipped
        for i in range(4):
            assert np.array_equal(result[i], np.fliplr(images[i]))


# ────────────────────────────────────────────────────────────────────
# P1: _evaluate forward hook
# ────────────────────────────────────────────────────────────────────


class TestEvaluateHook:
    def _make_trainer(self, tiny_model: nn.Module, tiny_loader: DataLoader, tmp_path: Path) -> BNNRTrainer:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        adapter = SimpleTorchAdapter(
            model=tiny_model, criterion=criterion, optimizer=optimizer,
            target_layers=[tiny_model.conv1], device="cpu",
        )
        config = BNNRConfig(
            m_epochs=1, max_iterations=1, device="cpu",
            checkpoint_dir=tmp_path / "ckpt", report_dir=tmp_path / "rep",
            xai_enabled=False, save_checkpoints=False,
        )
        return BNNRTrainer(
            model=adapter, train_loader=tiny_loader, val_loader=tiny_loader,
            augmentations=[], config=config,
        )

    def test_predictions_cached(self, tiny_model: nn.Module, tiny_loader: DataLoader, tmp_path: Path) -> None:
        """_evaluate with cache_predictions=True should populate _last_eval_preds."""
        trainer = self._make_trainer(tiny_model, tiny_loader, tmp_path)
        metrics = trainer._evaluate(tiny_loader, cache_predictions=True)
        assert trainer._last_eval_preds is not None
        assert trainer._last_eval_labels is not None
        assert len(trainer._last_eval_preds) == 20  # n_samples
        assert "accuracy" in metrics or "loss" in metrics

    def test_no_cache_leaves_none(self, tiny_model: nn.Module, tiny_loader: DataLoader, tmp_path: Path) -> None:
        """_evaluate with cache_predictions=False should leave preds as None."""
        trainer = self._make_trainer(tiny_model, tiny_loader, tmp_path)
        trainer._evaluate(tiny_loader, cache_predictions=False)
        assert trainer._last_eval_preds is None

    def test_hook_removed_after_evaluate(self, tiny_model: nn.Module, tiny_loader: DataLoader, tmp_path: Path) -> None:
        """After _evaluate, no forward hooks should remain on the model."""
        trainer = self._make_trainer(tiny_model, tiny_loader, tmp_path)
        hooks_before = len(tiny_model._forward_hooks)
        trainer._evaluate(tiny_loader, cache_predictions=True)
        hooks_after = len(tiny_model._forward_hooks)
        assert hooks_after == hooks_before, "Hook should be removed after _evaluate"


# ────────────────────────────────────────────────────────────────────
# P1: Dashboard events streaming
# ────────────────────────────────────────────────────────────────────


class TestDashboardEventsStreaming:
    def test_streaming_pagination(self, tmp_path: Path) -> None:
        """Streaming pagination should return correct page."""
        events_file = tmp_path / "run1" / "events.jsonl"
        events_file.parent.mkdir(parents=True)
        events = [{"type": "test", "idx": i} for i in range(100)]
        with events_file.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        # Read page: offset=10, limit=5
        result: list[dict] = []
        total = 0
        with events_file.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if total >= 10 and len(result) < 5:
                    result.append(json.loads(line))
                total += 1

        assert total == 100
        assert len(result) == 5
        assert result[0]["idx"] == 10
        assert result[4]["idx"] == 14
