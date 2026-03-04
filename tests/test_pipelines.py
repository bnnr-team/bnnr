"""Tests for bnnr.pipelines — dataset + model pipeline builders.

Covers the _IndexedDataset wrapper, CNN model architectures,
augmentation preset resolution, subset trimming, build dispatcher,
ImageFolder validation, YOLO format helpers, and detection collate.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from bnnr.core import BNNRConfig
from bnnr.pipelines import (
    _detection_collate_fn,
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
        for name in ("mnist", "fashion_mnist", "cifar10", "imagefolder", "coco_mini", "yolo"):
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
# _detection_collate_fn
# ---------------------------------------------------------------------------


class TestDetectionCollateFn:
    def test_basic_collate(self):
        batch = [
            (torch.randn(3, 32, 32), {"boxes": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([0])}, 0),
            (torch.randn(3, 32, 32), {"boxes": torch.tensor([[5, 6, 7, 8]]), "labels": torch.tensor([1])}, 1),
        ]
        images, targets, indices = _detection_collate_fn(batch)
        assert images.shape == (2, 3, 32, 32)
        assert len(targets) == 2
        assert indices == [0, 1]

    def test_empty_boxes(self):
        batch = [
            (torch.randn(3, 16, 16), {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}, 0),
        ]
        images, targets, indices = _detection_collate_fn(batch)
        assert images.shape == (1, 3, 16, 16)
        assert targets[0]["boxes"].shape == (0, 4)


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

    def test_coco_mini_without_data_path_raises(self):
        cfg = BNNRConfig()
        with pytest.raises(ValueError, match="--data-path is required"):
            build_pipeline("coco_mini", cfg)

    def test_yolo_without_data_path_raises(self):
        cfg = BNNRConfig()
        with pytest.raises(ValueError, match="--data-path is required"):
            build_pipeline("yolo", cfg)

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


# ---------------------------------------------------------------------------
# YOLO format helpers
# ---------------------------------------------------------------------------


class TestYoloHelpers:
    def test_resolve_yolo_images_from_entry_dir(self, tmp_path):
        from bnnr.pipelines import _resolve_yolo_images_from_entry

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        (img_dir / "001.jpg").write_bytes(b"\x00")
        (img_dir / "002.png").write_bytes(b"\x00")
        (img_dir / "readme.txt").write_text("not an image")

        paths = _resolve_yolo_images_from_entry(str(img_dir), tmp_path)
        assert len(paths) == 2
        names = {p.name for p in paths}
        assert "001.jpg" in names
        assert "002.png" in names

    def test_resolve_yolo_images_empty_dir(self, tmp_path):
        from bnnr.pipelines import _resolve_yolo_images_from_entry

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        paths = _resolve_yolo_images_from_entry(str(img_dir), tmp_path)
        assert paths == []

    def test_indexed_yolo_detection_label_path(self):
        from bnnr.pipelines import _IndexedYoloDetection

        ds = _IndexedYoloDetection.__new__(_IndexedYoloDetection)
        label_p = ds._label_path(Path("/data/images/train/001.jpg"))
        assert label_p == Path("/data/labels/train/001.txt")

    def test_build_yolo_pipeline_missing_yaml(self, tmp_path):
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="data.yaml"):
            build_pipeline("yolo", cfg, custom_data_path=tmp_path)

    def test_build_yolo_pipeline_invalid_yaml(self, tmp_path):
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text("not_a_valid: yaml_for_yolo\n")
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(ValueError, match="must contain"):
            build_pipeline("yolo", cfg, custom_data_path=yaml_file)

    def test_build_yolo_pipeline_empty_images(self, tmp_path):
        """YOLO data.yaml points to dirs with no images → error."""
        img_train = tmp_path / "images" / "train"
        img_val = tmp_path / "images" / "val"
        img_train.mkdir(parents=True)
        img_val.mkdir(parents=True)

        yaml_content = textwrap.dedent(f"""\
            train: {img_train}
            val: {img_val}
            nc: 3
            names: ['a', 'b', 'c']
        """)
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text(yaml_content)
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="empty sets"):
            build_pipeline("yolo", cfg, custom_data_path=yaml_file)

    def test_indexed_yolo_detection_getitem_no_labels(self, tmp_path):
        """_IndexedYoloDetection.__getitem__ with no label file returns empty boxes."""
        from PIL import Image

        from bnnr.pipelines import _IndexedYoloDetection

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        img_path = img_dir / "test.jpg"
        Image.new("RGB", (32, 32), color="red").save(img_path)

        ds = _IndexedYoloDetection([img_path], image_size=32)
        img_tensor, target, idx = ds[0]
        assert img_tensor.shape == (3, 32, 32)
        assert target["boxes"].shape == (0, 4)
        assert target["labels"].shape == (0,)
        assert idx == 0

    def test_indexed_yolo_detection_getitem_with_labels(self, tmp_path):
        """_IndexedYoloDetection.__getitem__ with a label file returns parsed boxes."""
        from PIL import Image

        from bnnr.pipelines import _IndexedYoloDetection

        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True)
        label_dir = tmp_path / "labels" / "train"
        label_dir.mkdir(parents=True)

        img_path = img_dir / "sample.jpg"
        Image.new("RGB", (64, 64), color="blue").save(img_path)

        label_path = label_dir / "sample.txt"
        # class_id cx cy w h (normalized)
        label_path.write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.8 0.1 0.1\n")

        ds = _IndexedYoloDetection([img_path], image_size=64)
        img_tensor, target, idx = ds[0]
        assert img_tensor.shape == (3, 64, 64)
        assert target["boxes"].shape == (2, 4)
        assert target["labels"].shape == (2,)
        # class 0 → id 1 (reserve 0 for background)
        assert target["labels"][0].item() == 1
        assert target["labels"][1].item() == 2


# ---------------------------------------------------------------------------
# COCO pipeline path validation
# ---------------------------------------------------------------------------


class TestBuildCocoMiniValidation:
    def test_missing_annotations_dir(self, tmp_path):
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="annotations"):
            build_pipeline("coco_mini", cfg, custom_data_path=tmp_path)

    def test_missing_train_dir(self, tmp_path):
        (tmp_path / "annotations").mkdir()
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="train image directory"):
            build_pipeline("coco_mini", cfg, custom_data_path=tmp_path)

    def test_missing_val_dir(self, tmp_path):
        (tmp_path / "annotations").mkdir()
        (tmp_path / "train").mkdir()
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="val image directory"):
            build_pipeline("coco_mini", cfg, custom_data_path=tmp_path)

    def test_missing_train_annotation(self, tmp_path):
        (tmp_path / "annotations").mkdir()
        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="train annotation"):
            build_pipeline("coco_mini", cfg, custom_data_path=tmp_path)

    def test_missing_val_annotation(self, tmp_path):
        (tmp_path / "annotations").mkdir()
        (tmp_path / "train").mkdir()
        (tmp_path / "val").mkdir()
        (tmp_path / "annotations" / "train.json").write_text("{}")
        cfg = BNNRConfig(device="cpu")
        with pytest.raises(FileNotFoundError, match="val annotation"):
            build_pipeline("coco_mini", cfg, custom_data_path=tmp_path)
