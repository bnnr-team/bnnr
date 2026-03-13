"""Tests for bnnr.pipelines — dataset + model pipeline builders.

Covers the _IndexedDataset wrapper, CNN model architectures,
augmentation preset resolution, subset trimming, build dispatcher,
ImageFolder validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from bnnr.core import BNNRConfig
from bnnr.pipelines import (
    _IndexedDataset,
    _maybe_subset,
    _resolve_augmentations,
    build_pipeline,
    list_datasets,
)

# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------


class TestListDatasets:
    def test_returns_list(self):
        ds = list_datasets()
        assert isinstance(ds, list)
        assert len(ds) >= 5

    def test_contains_expected_names(self):
        ds = list_datasets()
        for name in ("mnist", "fashion_mnist", "cifar10", "imagefolder"):
            assert name in ds


# ---------------------------------------------------------------------------
# _IndexedDataset
# ---------------------------------------------------------------------------


class TestIndexedDataset:
    def test_len(self):
        base = TensorDataset(torch.randn(10, 3, 8, 8), torch.arange(10))
        idx_ds = _IndexedDataset(base)
        assert len(idx_ds) == 10

    def test_getitem_returns_triple(self):
        base = TensorDataset(torch.randn(5, 1, 4, 4), torch.arange(5))
        idx_ds = _IndexedDataset(base)
        img, label, index = idx_ds[2]
        assert isinstance(img, Tensor)
        assert label == 2
        assert index == 2

    def test_getitem_first_and_last(self):
        base = TensorDataset(torch.randn(3, 1, 4, 4), torch.tensor([10, 20, 30]))
        idx_ds = _IndexedDataset(base)
        _, label0, idx0 = idx_ds[0]
        _, label2, idx2 = idx_ds[2]
        assert label0 == 10 and idx0 == 0
        assert label2 == 30 and idx2 == 2


# ---------------------------------------------------------------------------
# _maybe_subset
# ---------------------------------------------------------------------------


class TestMaybeSubset:
    def test_no_limit_returns_original(self):
        ds = TensorDataset(torch.randn(20, 3))
        result = _maybe_subset(ds, max_samples=None)
        assert result is ds

    def test_limit_larger_than_dataset(self):
        ds = TensorDataset(torch.randn(5, 3))
        result = _maybe_subset(ds, max_samples=100)
        assert len(result) == 5

    def test_limit_smaller_than_dataset(self):
        ds = TensorDataset(torch.randn(50, 3))
        result = _maybe_subset(ds, max_samples=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _resolve_augmentations
# ---------------------------------------------------------------------------


class TestResolveAugmentations:
    def test_none_returns_empty(self):
        augs = _resolve_augmentations("none", seed=42)
        assert augs == []

    def test_auto_returns_list(self):
        augs = _resolve_augmentations("auto", seed=42)
        assert isinstance(augs, list)
        assert len(augs) > 0

    def test_light_returns_augmentations(self):
        augs = _resolve_augmentations("light", seed=42)
        assert isinstance(augs, list)
        assert len(augs) > 0

    def test_standard_returns_augmentations(self):
        augs = _resolve_augmentations("standard", seed=42)
        assert isinstance(augs, list)
        assert len(augs) > 0

    def test_aggressive_returns_augmentations(self):
        augs = _resolve_augmentations("aggressive", seed=42)
        assert isinstance(augs, list)

    def test_unknown_falls_back_to_auto(self):
        augs = _resolve_augmentations("nonexistent_preset_xyz", seed=42)
        assert isinstance(augs, list)
        assert len(augs) > 0


# ---------------------------------------------------------------------------
# CNN model forward passes
# ---------------------------------------------------------------------------


class TestBuiltinCNNModels:
    def test_mnist_cnn_forward(self):
        from bnnr.pipelines import _MnistCNN

        model = _MnistCNN()
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_cifar_cnn_forward(self):
        from bnnr.pipelines import _CifarCNN

        model = _CifarCNN()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_imagefolder_cnn_forward(self):
        from bnnr.pipelines import _ImageFolderCNN

        model = _ImageFolderCNN(num_classes=5)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5)

    def test_imagefolder_cnn_default_classes(self):
        from bnnr.pipelines import _ImageFolderCNN

        model = _ImageFolderCNN()
        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# build_pipeline dispatcher
# ---------------------------------------------------------------------------


class TestBuildPipelineDispatcher:
    def test_unknown_dataset_raises(self):
        cfg = BNNRConfig()
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_pipeline("nonexistent_dataset_xyz", cfg)

    def test_imagefolder_without_data_path_raises(self):
        cfg = BNNRConfig()
        with pytest.raises(ValueError, match="--data-path is required"):
            build_pipeline("imagefolder", cfg)

    def test_case_insensitive_dataset_name(self):
        cfg = BNNRConfig()
        with pytest.raises(ValueError, match="Unknown dataset"):
            build_pipeline("  UNKNOWN_DATASET  ", cfg)


# ---------------------------------------------------------------------------
# build_imagefolder_pipeline — directory validation
# ---------------------------------------------------------------------------


class TestBuildImagefolderValidation:
    def test_missing_train_dir_raises(self, tmp_path):
        cfg = BNNRConfig()
        with pytest.raises(FileNotFoundError, match="Expected train directory"):
            build_pipeline("imagefolder", cfg, custom_data_path=tmp_path)

    def test_missing_val_dir_raises(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "train" / "classA").mkdir()
        cfg = BNNRConfig()
        with pytest.raises(FileNotFoundError, match="No validation directory"):
            build_pipeline("imagefolder", cfg, custom_data_path=tmp_path)


# ---------------------------------------------------------------------------
# build_mnist_pipeline — with real data (if available, else skip)
# ---------------------------------------------------------------------------


class TestBuildMnistPipeline:
    """Integration test — runs only if MNIST data is already downloaded."""

    def test_build_mnist_pipeline_with_subset(self):
        cfg = BNNRConfig(device="cpu", m_epochs=1, max_iterations=1)
        data_dir = Path("data")
        if not (data_dir / "MNIST").exists():
            pytest.skip("MNIST data not downloaded")
        adapter, train_loader, val_loader, augs = build_pipeline(
            "mnist", cfg, data_dir=data_dir,
            max_train_samples=16, max_val_samples=8,
            augmentation_preset="none",
        )
        # Check loader yields triples (image, label, index)
        batch = next(iter(train_loader))
        assert len(batch) == 3
        images, labels, indices = batch
        assert images.ndim == 4

    def test_build_fashion_mnist_pipeline_with_subset(self):
        cfg = BNNRConfig(device="cpu", m_epochs=1, max_iterations=1)
        data_dir = Path("data")
        if not (data_dir / "FashionMNIST").exists():
            pytest.skip("FashionMNIST data not downloaded")
        adapter, train_loader, val_loader, augs = build_pipeline(
            "fashion_mnist", cfg, data_dir=data_dir,
            max_train_samples=16, max_val_samples=8,
            augmentation_preset="light",
        )
        assert len(augs) > 0


class TestBuildCifar10Pipeline:
    def test_build_cifar10_pipeline_with_subset(self):
        cfg = BNNRConfig(device="cpu", m_epochs=1, max_iterations=1)
        data_dir = Path("data")
        if not (data_dir / "cifar-10-batches-py").exists():
            pytest.skip("CIFAR-10 data not downloaded")
        adapter, train_loader, val_loader, augs = build_pipeline(
            "cifar10", cfg, data_dir=data_dir,
            max_train_samples=16, max_val_samples=8,
            augmentation_preset="none",
        )
        batch = next(iter(train_loader))
        assert len(batch) == 3


