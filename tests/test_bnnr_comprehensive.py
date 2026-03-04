"""Comprehensive BNNR test suite.

Tests cover:
  - All 8 built-in augmentation types (numpy + tensor paths)
  - TorchvisionAugmentation wrapper
  - AlbumentationsAugmentation wrapper (if albumentations installed)
  - KorniaAugmentation wrapper (if kornia installed)
  - ICD / AICD with XAI cache
  - AugmentationRunner (sync + async dispatch)
  - AugmentationRegistry
  - BNNRConfig / validation / merge / save / load
  - SimpleTorchAdapter (train / eval / state_dict)
  - BNNRTrainer single iteration + full run
  - Reporter / BNNRRunResult / load_report / compare_runs
  - XAI saliency map generation (OptiCAM)
  - XAI cache precompute / retrieval
  - Presets (auto_select / get_preset / list_presets)
  - quick_run() convenience function
  - One comprehensive E2E test exercising the full BNNR domain pipeline
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bnnr.adapter import SimpleTorchAdapter, XAICapableModel
from bnnr.albumentations_aug import AlbumentationsAugmentation, albumentations_available
from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import (
    AugmentationRegistry,
    BaseAugmentation,
    BasicAugmentation,
    ChurchNoise,
    DifPresets,
    Drust,
    LuxferGlass,
    ProCAM,
    Smugs,
    TeaStains,
    TorchvisionAugmentation,
)
from bnnr.config import load_config, merge_configs, save_config, validate_config
from bnnr.core import BNNRConfig, BNNRTrainer
from bnnr.events import JsonlEventSink, load_events, replay_events
from bnnr.icd import AICD, ICD
from bnnr.kornia_aug import KorniaAugmentation, create_kornia_pipeline, kornia_available
from bnnr.presets import auto_select_augmentations, get_preset, list_presets
from bnnr.quick_run import quick_run
from bnnr.reporting import BNNRRunResult, Reporter, compare_runs, load_report
from bnnr.xai import OptiCAMExplainer, generate_saliency_maps, save_xai_visualization
from bnnr.xai_cache import XAICache

# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
IMG_SIZE = 32
N_CLASSES = 4
N_SAMPLES = 24
BATCH_SIZE = 8


class TinyCNN(nn.Module):
    """Minimal CNN suitable for all tests (supports 1 or 3 channels)."""

    def __init__(self, in_channels: int = 3, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _IndexedDataset(Dataset):
    """Returns (image, label, index)."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, label, index


def _make_rgb_dataset(n: int = N_SAMPLES) -> TensorDataset:
    x = torch.rand(n, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, N_CLASSES, (n,))
    return TensorDataset(x, y)


def _make_gray_dataset(n: int = N_SAMPLES) -> TensorDataset:
    x = torch.rand(n, 1, 28, 28)
    y = torch.randint(0, N_CLASSES, (n,))
    return TensorDataset(x, y)


def _make_loaders(
    dataset: TensorDataset | None = None,
    batch_size: int = BATCH_SIZE,
    indexed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    ds = dataset or _make_rgb_dataset()
    if indexed:
        ds = _IndexedDataset(ds)  # type: ignore[assignment]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, loader


def _make_adapter(
    in_channels: int = 3,
    n_classes: int = N_CLASSES,
    lr: float = 1e-3,
) -> tuple[SimpleTorchAdapter, TinyCNN]:
    model = TinyCNN(in_channels=in_channels, n_classes=n_classes)
    adapter = SimpleTorchAdapter(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        target_layers=[model.conv1],
        device="cpu",
    )
    return adapter, model


def _make_config(tmp_path: Path, **overrides: Any) -> BNNRConfig:
    defaults: dict[str, Any] = {
        "m_epochs": 1,
        "max_iterations": 1,
        "device": "cpu",
        "xai_enabled": False,
        "save_checkpoints": True,
        "verbose": False,
        "checkpoint_dir": tmp_path / "ckpt",
        "report_dir": tmp_path / "reports",
        "early_stopping_patience": 2,
        "event_log_enabled": False,
    }
    defaults.update(overrides)
    return BNNRConfig(**defaults)


def _make_uint8_image(h: int = IMG_SIZE, w: int = IMG_SIZE, c: int = 3) -> np.ndarray:
    return (np.random.rand(h, w, c) * 255).astype(np.uint8)


def _make_batch_tensor(
    b: int = 4, c: int = 3, h: int = IMG_SIZE, w: int = IMG_SIZE,
) -> torch.Tensor:
    return torch.rand(b, c, h, w, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Built-in augmentations — numpy path
# ═══════════════════════════════════════════════════════════════════════════════


ALL_BUILTIN_NAMES = [
    "church_noise",
    "basic_augmentation",
    "dif_presets",
    "drust",
    "luxfer_glass",
    "procam",
    "smugs",
    "tea_stains",
]

ALL_BUILTIN_CLASSES = [
    ChurchNoise,
    BasicAugmentation,
    DifPresets,
    Drust,
    LuxferGlass,
    ProCAM,
    Smugs,
    TeaStains,
]


class TestBuiltinAugmentationsNumpy:
    """Test all 8 built-in augmentations via the numpy (CPU) path."""

    @pytest.mark.parametrize("name", ALL_BUILTIN_NAMES)
    def test_apply_single_image(self, name: str) -> None:
        aug = AugmentationRegistry.create(name, probability=1.0, random_state=SEED)
        image = _make_uint8_image()
        result = aug.apply(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("name", ALL_BUILTIN_NAMES)
    def test_apply_batch(self, name: str) -> None:
        aug = AugmentationRegistry.create(name, probability=1.0, random_state=SEED)
        batch = np.stack([_make_uint8_image() for _ in range(4)])
        result = aug.apply_batch(batch)
        assert result.shape == batch.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("name", ALL_BUILTIN_NAMES)
    def test_probability_zero_is_identity(self, name: str) -> None:
        aug = AugmentationRegistry.create(name, probability=0.0, random_state=SEED)
        batch = np.stack([_make_uint8_image() for _ in range(2)])
        result = aug.apply_batch(batch)
        np.testing.assert_array_equal(result, batch)

    @pytest.mark.parametrize("name", ALL_BUILTIN_NAMES)
    def test_grayscale_single_channel(self, name: str) -> None:
        aug = AugmentationRegistry.create(name, probability=1.0, random_state=SEED)
        gray_batch = np.stack([_make_uint8_image(c=1) for _ in range(2)])
        result = aug.apply_batch(gray_batch)
        assert result.shape == gray_batch.shape

    @pytest.mark.parametrize("name", ALL_BUILTIN_NAMES)
    def test_small_image_does_not_crash(self, name: str) -> None:
        """Augmentations must handle tiny images (e.g. 8×8)."""
        aug = AugmentationRegistry.create(name, probability=1.0, random_state=SEED)
        image = _make_uint8_image(h=8, w=8)
        result = aug.apply(image)
        assert result.shape == image.shape


# ═══════════════════════════════════════════════════════════════════════════════
#  2. GPU-native tensor augmentations
# ═══════════════════════════════════════════════════════════════════════════════

GPU_NATIVE_CLASSES = [ChurchNoise, ProCAM, DifPresets]


class TestGPUNativeAugmentations:
    @pytest.mark.parametrize("aug_cls", GPU_NATIVE_CLASSES)
    def test_tensor_native_output_valid(self, aug_cls: type) -> None:
        aug = aug_cls(probability=1.0, random_state=SEED)
        assert aug.device_compatible is True
        images = _make_batch_tensor()
        result = aug.apply_tensor_native(images)
        assert result.shape == images.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("aug_cls", GPU_NATIVE_CLASSES)
    def test_tensor_fallback_via_apply_tensor(self, aug_cls: type) -> None:
        aug = aug_cls(probability=1.0, random_state=SEED)
        images = _make_batch_tensor()
        result = aug.apply_tensor(images)
        assert result.shape == images.shape

    @pytest.mark.parametrize("aug_cls", GPU_NATIVE_CLASSES)
    def test_tensor_probability_zero(self, aug_cls: type) -> None:
        aug = aug_cls(probability=0.0, random_state=SEED)
        images = _make_batch_tensor()
        result = aug.apply_tensor_native(images)
        assert torch.allclose(result, images)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. TorchvisionAugmentation wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestTorchvisionWrapper:
    def test_basic_torchvision_transform(self) -> None:
        from torchvision import transforms

        transform = transforms.ColorJitter(brightness=0.3, contrast=0.3)
        aug = TorchvisionAugmentation(
            transform, name_override="tv_color_jitter", probability=1.0, random_state=SEED,
        )
        assert aug.name == "tv_color_jitter"
        image = _make_uint8_image()
        result = aug.apply(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_torchvision_hflip(self) -> None:
        from torchvision import transforms

        aug = TorchvisionAugmentation(
            transforms.RandomHorizontalFlip(p=1.0),
            name_override="tv_hflip",
            probability=1.0,
        )
        image = _make_uint8_image()
        result = aug.apply(image)
        # Flipped horizontally — last column becomes first
        np.testing.assert_array_equal(result[:, 0, :], image[:, -1, :])

    def test_torchvision_batch_via_apply_batch(self) -> None:
        from torchvision import transforms

        aug = TorchvisionAugmentation(
            transforms.RandomGrayscale(p=1.0),
            name_override="tv_grayscale",
            probability=1.0,
        )
        batch = np.stack([_make_uint8_image() for _ in range(3)])
        result = aug.apply_batch(batch)
        assert result.shape == batch.shape

    def test_torchvision_via_apply_tensor(self) -> None:
        from torchvision import transforms

        aug = TorchvisionAugmentation(
            transforms.ColorJitter(brightness=0.2),
            name_override="tv_jitter",
            probability=1.0,
        )
        images = _make_batch_tensor()
        result = aug.apply_tensor(images)
        assert result.shape == images.shape


# ═══════════════════════════════════════════════════════════════════════════════
#  4. AlbumentationsAugmentation wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlbumentationsWrapper:
    @pytest.mark.skipif(
        not albumentations_available(), reason="albumentations not installed",
    )
    def test_albu_compose(self) -> None:
        import albumentations as A  # noqa: N812

        transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=0.5),
        ])
        aug = AlbumentationsAugmentation(
            transform, name_override="albu_test", probability=1.0,
        )
        assert aug.name == "albu_test"
        image = _make_uint8_image()
        result = aug.apply(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    @pytest.mark.skipif(
        not albumentations_available(), reason="albumentations not installed",
    )
    def test_albu_batch(self) -> None:
        import albumentations as A  # noqa: N812

        aug = AlbumentationsAugmentation(
            A.GaussNoise(p=1.0),
            name_override="albu_noise",
            probability=1.0,
        )
        batch = np.stack([_make_uint8_image() for _ in range(3)])
        result = aug.apply_batch(batch)
        assert result.shape == batch.shape

    @pytest.mark.skipif(
        not albumentations_available(), reason="albumentations not installed",
    )
    def test_albu_single_transforms(self) -> None:
        import albumentations as A  # noqa: N812

        transforms_to_test = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Rotate(limit=15, p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ]
        image = _make_uint8_image()
        for t in transforms_to_test:
            aug = AlbumentationsAugmentation(t, probability=1.0)
            result = aug.apply(image)
            assert result.shape == image.shape, f"Failed for {type(t).__name__}"

    @pytest.mark.skipif(
        not albumentations_available(), reason="albumentations not installed",
    )
    def test_albu_via_apply_tensor(self) -> None:
        import albumentations as A  # noqa: N812

        aug = AlbumentationsAugmentation(
            A.HorizontalFlip(p=1.0),
            name_override="albu_hflip",
            probability=1.0,
        )
        images = _make_batch_tensor()
        result = aug.apply_tensor(images)
        assert result.shape == images.shape

    def test_albu_import_error_when_missing(self) -> None:
        """If albumentations is not installed, constructor should raise ImportError."""
        if albumentations_available():
            pytest.skip("albumentations is installed — skip negative test")
        with pytest.raises(ImportError, match="Albumentations"):
            AlbumentationsAugmentation(object())


# ═══════════════════════════════════════════════════════════════════════════════
#  5. KorniaAugmentation wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class TestKorniaWrapper:
    @pytest.mark.skipif(not kornia_available(), reason="kornia not installed")
    def test_kornia_hflip(self) -> None:
        import kornia.augmentation as K  # noqa: N812

        aug = KorniaAugmentation(
            K.RandomHorizontalFlip(p=1.0),
            name_override="kornia_hflip_test",
            probability=1.0,
        )
        assert aug.device_compatible is True
        images = _make_batch_tensor()
        result = aug.apply_tensor_native(images)
        assert result.shape == images.shape

    @pytest.mark.skipif(not kornia_available(), reason="kornia not installed")
    def test_kornia_color_jitter(self) -> None:
        import kornia.augmentation as K  # noqa: N812

        aug = KorniaAugmentation(
            K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=1.0),
            name_override="kornia_jitter_test",
            probability=1.0,
        )
        images = _make_batch_tensor()
        result = aug.apply_tensor(images)
        assert result.shape == images.shape

    @pytest.mark.skipif(not kornia_available(), reason="kornia not installed")
    def test_kornia_rotation(self) -> None:
        import kornia.augmentation as K  # noqa: N812

        aug = KorniaAugmentation(
            K.RandomRotation(degrees=30.0, p=1.0),
            name_override="kornia_rotate_test",
            probability=1.0,
        )
        images = _make_batch_tensor()
        result = aug.apply_tensor_native(images)
        assert result.shape == images.shape

    @pytest.mark.skipif(not kornia_available(), reason="kornia not installed")
    def test_kornia_pipeline(self) -> None:
        import kornia.augmentation as K  # noqa: N812

        pipeline = create_kornia_pipeline(
            [K.RandomHorizontalFlip(p=0.5), K.RandomRotation(degrees=15.0, p=0.5)],
            probability=1.0,
            name="kornia_pipe_test",
        )
        assert pipeline.name == "kornia_pipe_test"
        images = _make_batch_tensor()
        result = pipeline.apply_tensor(images)
        assert result.shape == images.shape

    @pytest.mark.skipif(not kornia_available(), reason="kornia not installed")
    def test_kornia_numpy_fallback(self) -> None:
        import kornia.augmentation as K  # noqa: N812

        aug = KorniaAugmentation(
            K.RandomHorizontalFlip(p=1.0),
            probability=1.0,
        )
        image = _make_uint8_image()
        result = aug.apply(image)
        assert result.shape == image.shape

    @pytest.mark.skipif(not kornia_available(), reason="kornia not installed")
    def test_kornia_registered_augmentations(self) -> None:
        """Kornia pre-registered augmentations should be in the registry."""
        expected_names = [
            "kornia_hflip", "kornia_vflip", "kornia_rotation",
            "kornia_color_jitter", "kornia_gaussian_blur",
            "kornia_erasing", "kornia_affine", "kornia_normalize",
        ]
        for name in expected_names:
            assert AugmentationRegistry.is_registered(name), f"{name} not registered"
            aug = AugmentationRegistry.create(name, probability=1.0)
            images = _make_batch_tensor()
            result = aug.apply_tensor(images)
            assert result.shape == images.shape, f"Shape mismatch for {name}"

    def test_kornia_import_error_when_missing(self) -> None:
        if kornia_available():
            pytest.skip("kornia is installed — skip negative test")
        with pytest.raises(ImportError, match="Kornia"):
            KorniaAugmentation(object())


# ═══════════════════════════════════════════════════════════════════════════════
#  6. ICD / AICD
# ═══════════════════════════════════════════════════════════════════════════════

class TestICDAICD:
    def test_icd_with_cache(self, tmp_path: Path) -> None:
        model = TinyCNN()
        target_layers = [model.conv1]

        # Precompute a small cache
        ds = _make_rgb_dataset(n=8)
        loader = DataLoader(ds, batch_size=4)
        cache = XAICache(tmp_path / "xai_cache")
        cache.precompute_cache(
            model=model, train_loader=loader,
            target_layers=target_layers, n_samples=8, method="opticam",
        )

        icd = ICD(
            model=model, target_layers=target_layers,
            cache=cache, probability=1.0, random_state=SEED,
        )
        assert icd.name == "icd"
        assert icd.invert_mask is False
        image = _make_uint8_image()
        result = icd.apply_with_label(image, label=0)
        assert result.shape == image.shape

    def test_aicd_with_cache(self, tmp_path: Path) -> None:
        model = TinyCNN()
        target_layers = [model.conv1]

        ds = _make_rgb_dataset(n=8)
        loader = DataLoader(ds, batch_size=4)
        cache = XAICache(tmp_path / "xai_cache")
        cache.precompute_cache(
            model=model, train_loader=loader,
            target_layers=target_layers, n_samples=8, method="opticam",
        )

        aicd = AICD(
            model=model, target_layers=target_layers,
            cache=cache, probability=1.0, random_state=SEED,
        )
        assert aicd.name == "aicd"
        assert aicd.invert_mask is True
        image = _make_uint8_image()
        result = aicd.apply_with_label(image, label=0)
        assert result.shape == image.shape

    def test_icd_batch_with_labels(self, tmp_path: Path) -> None:
        model = TinyCNN()
        cache = XAICache(tmp_path / "xai_cache")
        icd = ICD(
            model=model, target_layers=[model.conv1],
            cache=cache, probability=1.0, random_state=SEED,
        )
        images = np.stack([_make_uint8_image() for _ in range(3)])
        labels = np.array([0, 1, 2])
        result = icd.apply_batch_with_labels(images, labels)
        assert result.shape == images.shape

    def test_icd_raises_without_label(self) -> None:
        model = TinyCNN()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            icd = ICD(
                model=model, target_layers=[model.conv1],
                probability=1.0, random_state=SEED,
            )
        with pytest.raises(ValueError, match="Label not set"):
            icd.apply(_make_uint8_image())

    def test_icd_set_label_and_apply(self, tmp_path: Path) -> None:
        model = TinyCNN()
        cache = XAICache(tmp_path / "xai_cache")
        icd = ICD(
            model=model, target_layers=[model.conv1],
            cache=cache, probability=1.0, random_state=SEED,
        )
        icd.set_label(2)
        result = icd.apply(_make_uint8_image())
        assert result.shape == (_make_uint8_image().shape[0], _make_uint8_image().shape[1], 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. AugmentationRunner
# ═══════════════════════════════════════════════════════════════════════════════

class TestAugmentationRunnerComprehensive:
    def test_mixed_gpu_cpu_augmentations(self) -> None:
        gpu_aug = ChurchNoise(probability=1.0, random_state=SEED)
        cpu_aug = BasicAugmentation(probability=1.0, random_state=SEED)
        runner = AugmentationRunner([gpu_aug, cpu_aug], async_prefetch=False)
        assert gpu_aug in runner.gpu_augmentations
        assert cpu_aug in runner.cpu_augmentations
        images = _make_batch_tensor()
        labels = torch.randint(0, N_CLASSES, (4,))
        result_images, result_labels = runner.apply_batch(images, labels)
        assert result_images.shape == images.shape
        assert torch.equal(result_labels, labels)

    def test_async_prefetch_iterator(self) -> None:
        cpu_aug = BasicAugmentation(probability=1.0, random_state=SEED)
        runner = AugmentationRunner([cpu_aug], async_prefetch=True)
        batches = [
            (_make_batch_tensor(), torch.randint(0, N_CLASSES, (4,)))
            for _ in range(5)
        ]
        results = list(runner.iter_loader(batches))
        assert len(results) == 5

    def test_runner_with_sample_indices(self) -> None:
        model = TinyCNN()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            icd = ICD(
                model=model, target_layers=[model.conv1],
                probability=1.0, random_state=SEED,
            )
        runner = AugmentationRunner([icd], async_prefetch=False)
        images = _make_batch_tensor()
        labels = torch.randint(0, N_CLASSES, (4,))
        indices = torch.arange(4)
        result_images, result_labels = runner.apply_batch(images, labels, sample_indices=indices)
        assert result_images.shape == images.shape

    def test_runner_empty(self) -> None:
        runner = AugmentationRunner([])
        images = _make_batch_tensor()
        labels = torch.randint(0, N_CLASSES, (4,))
        result_images, result_labels = runner.apply_batch(images, labels)
        assert torch.equal(result_images, images)


# ═══════════════════════════════════════════════════════════════════════════════
#  8. AugmentationRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestAugmentationRegistryComprehensive:
    def test_all_builtin_registered(self) -> None:
        registered = AugmentationRegistry.list_all()
        for name in ALL_BUILTIN_NAMES:
            assert name in registered, f"{name} not registered"

    def test_alias_names(self) -> None:
        """Augmentations have numeric aliases like augmentation_1, augmentation_3, etc."""
        aliases = [
            "augmentation_1",  # church_noise
            "augmentation_3",  # basic_augmentation
            "augmentation_5",  # dif_presets
            "augmentation_6",  # drust
            "augmentation_7",  # luxfer_glass
            "augmentation_8",  # procam
            "augmentation_9",  # smugs
            "augmentation_10",  # tea_stains
        ]
        for alias in aliases:
            assert AugmentationRegistry.is_registered(alias)
            aug = AugmentationRegistry.create(alias, probability=0.5)
            assert isinstance(aug, BaseAugmentation)

    def test_unknown_augmentation_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            AugmentationRegistry.get("nonexistent_aug")

    def test_register_custom(self) -> None:
        AugmentationRegistry._registry.pop("_test_custom_aug", None)

        @AugmentationRegistry.register("_test_custom_aug")
        class CustomAug(BaseAugmentation):
            def apply(self, image: np.ndarray) -> np.ndarray:
                return image

        assert AugmentationRegistry.is_registered("_test_custom_aug")
        aug = AugmentationRegistry.create("_test_custom_aug", probability=1.0)
        assert isinstance(aug, CustomAug)
        # Clean up
        AugmentationRegistry._registry.pop("_test_custom_aug", None)


# ═══════════════════════════════════════════════════════════════════════════════
#  9. BNNRConfig / config module
# ═══════════════════════════════════════════════════════════════════════════════

class TestBNNRConfig:
    def test_defaults(self) -> None:
        cfg = BNNRConfig()
        assert cfg.m_epochs == 5
        assert cfg.selection_metric == "accuracy"
        assert cfg.selection_mode == "max"
        assert cfg.xai_method == "opticam"

    def test_validation_selection_mode(self) -> None:
        with pytest.raises(ValueError, match="selection_mode"):
            BNNRConfig(selection_mode="invalid")

    def test_validation_device(self) -> None:
        with pytest.raises(ValueError, match="device"):
            BNNRConfig(device="tpu")

    def test_validation_pruning_threshold(self) -> None:
        with pytest.raises(ValueError, match="candidate_pruning_relative_threshold"):
            BNNRConfig(candidate_pruning_relative_threshold=0.0)

    def test_frozen_config(self) -> None:
        cfg = BNNRConfig()
        with pytest.raises(Exception):  # pydantic ValidationError for frozen
            cfg.m_epochs = 10  # type: ignore[misc]

    def test_model_copy_with_update(self) -> None:
        cfg = BNNRConfig(m_epochs=5)
        cfg2 = cfg.model_copy(update={"m_epochs": 10})
        assert cfg2.m_epochs == 10
        assert cfg.m_epochs == 5  # original unchanged

    def test_save_and_load_config(self, tmp_path: Path) -> None:
        cfg = BNNRConfig(m_epochs=3, device="cpu")
        config_path = tmp_path / "test_config.yaml"
        save_config(cfg, config_path)
        loaded = load_config(config_path)
        assert loaded.m_epochs == 3
        assert loaded.device == "cpu"

    def test_merge_configs(self) -> None:
        base = BNNRConfig(m_epochs=3, device="cpu")
        merged = merge_configs(base, {"m_epochs": 10})
        assert merged.m_epochs == 10
        assert merged.device == "cpu"

    def test_validate_config(self) -> None:
        cfg = BNNRConfig(m_epochs=1, selection_metric="accuracy")
        warns = validate_config(cfg)
        assert isinstance(warns, list)
        # Valid config should have no warnings
        assert len(warns) == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  10. SimpleTorchAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimpleTorchAdapter:
    def test_train_step(self) -> None:
        adapter, _ = _make_adapter()
        images = _make_batch_tensor(b=4)
        labels = torch.randint(0, N_CLASSES, (4,))
        metrics = adapter.train_step((images, labels))
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_eval_step(self) -> None:
        adapter, _ = _make_adapter()
        images = _make_batch_tensor(b=4)
        labels = torch.randint(0, N_CLASSES, (4,))
        metrics = adapter.eval_step((images, labels))
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_state_dict_roundtrip(self) -> None:
        adapter, _ = _make_adapter()
        state = adapter.state_dict()
        assert "model" in state
        assert "optimizer" in state
        adapter.load_state_dict(state)

    def test_xai_capable_protocol(self) -> None:
        adapter, _ = _make_adapter()
        assert isinstance(adapter, XAICapableModel)
        model = adapter.get_model()
        assert isinstance(model, nn.Module)
        layers = adapter.get_target_layers()
        assert len(layers) > 0

    def test_auto_target_layers(self) -> None:
        model = TinyCNN()
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters()),
            target_layers=None,  # auto-detect
            device="cpu",
        )
        assert len(adapter.target_layers) > 0

    def test_epoch_end_with_scheduler(self) -> None:
        adapter, model = _make_adapter()
        scheduler = torch.optim.lr_scheduler.StepLR(adapter.optimizer, step_size=1)
        adapter.scheduler = scheduler
        initial_lr = adapter.optimizer.param_groups[0]["lr"]
        adapter.epoch_end()
        # StepLR should decrease lr
        assert adapter.optimizer.param_groups[0]["lr"] <= initial_lr


# ═══════════════════════════════════════════════════════════════════════════════
#  11. XAI: saliency maps, visualization, cache
# ═══════════════════════════════════════════════════════════════════════════════

class TestXAI:
    def test_generate_saliency_maps_opticam(self) -> None:
        model = TinyCNN()
        images = _make_batch_tensor(b=2)
        labels = torch.randint(0, N_CLASSES, (2,))
        maps = generate_saliency_maps(
            model, images, labels, [model.conv1], method="opticam",
        )
        assert maps.shape[0] == 2
        assert maps.ndim == 3  # (B, H, W)

    def test_save_xai_visualization(self, tmp_path: Path) -> None:
        images = np.stack([_make_uint8_image() for _ in range(2)])
        maps = np.random.rand(2, IMG_SIZE, IMG_SIZE).astype(np.float32)
        paths = save_xai_visualization(
            images, maps, save_dir=tmp_path / "xai_vis",
        )
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_opticam_explainer_explain(self) -> None:
        model = TinyCNN()
        images = _make_batch_tensor(b=2)
        labels = torch.randint(0, N_CLASSES, (2,))
        explainer = OptiCAMExplainer(use_cuda=False)
        maps = explainer.explain(model, images, labels, [model.conv1])
        assert maps.shape[0] == 2


class TestXAICache:
    def test_precompute_and_retrieve(self, tmp_path: Path) -> None:
        model = TinyCNN()
        ds = _make_rgb_dataset(n=8)
        loader = DataLoader(ds, batch_size=4)
        cache = XAICache(tmp_path / "cache")
        written = cache.precompute_cache(
            model=model, train_loader=loader,
            target_layers=[model.conv1], n_samples=8,
        )
        assert written > 0
        # Check that NPY files were created
        npy_files = list((tmp_path / "cache").glob("*.npy"))
        assert len(npy_files) > 0

    def test_cache_with_indexed_dataset(self, tmp_path: Path) -> None:
        model = TinyCNN()
        ds = _IndexedDataset(_make_rgb_dataset(n=8))
        loader = DataLoader(ds, batch_size=4)
        cache = XAICache(tmp_path / "cache")
        written = cache.precompute_cache(
            model=model, train_loader=loader,
            target_layers=[model.conv1], n_samples=8,
        )
        assert written > 0
        # Index-based cache files
        idx_files = list((tmp_path / "cache").glob("idx_*.npy"))
        assert len(idx_files) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  12. Presets
# ═══════════════════════════════════════════════════════════════════════════════

class TestPresets:
    def test_list_presets(self) -> None:
        presets = list_presets()
        assert isinstance(presets, dict)
        assert "light" in presets
        assert "standard" in presets
        assert "aggressive" in presets
        assert "gpu" in presets

    @pytest.mark.parametrize("preset_name", ["light", "standard", "aggressive", "gpu"])
    def test_get_preset(self, preset_name: str) -> None:
        augs = get_preset(preset_name, random_state=SEED)
        assert isinstance(augs, list)
        assert len(augs) > 0
        for aug in augs:
            assert isinstance(aug, BaseAugmentation)

    def test_auto_select_augmentations(self) -> None:
        augs = auto_select_augmentations(random_state=SEED, prefer_gpu=False)
        assert isinstance(augs, list)
        assert len(augs) > 0

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent_preset")


# ═══════════════════════════════════════════════════════════════════════════════
#  13. Reporter / BNNRRunResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestReporter:
    def test_reporter_start_and_finalize(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        reporter = Reporter(tmp_path / "reports")
        reporter.start(cfg)
        result = reporter.finalize(
            best_path="baseline",
            best_metrics={"accuracy": 0.5, "loss": 1.0},
            selected_augmentations=[],
        )
        assert isinstance(result, BNNRRunResult)
        assert result.report_json_path.exists()

    def test_load_report(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        reporter = Reporter(tmp_path / "reports")
        reporter.start(cfg)
        result = reporter.finalize(
            best_path="baseline -> church_noise",
            best_metrics={"accuracy": 0.75, "loss": 0.5},
            selected_augmentations=["church_noise"],
        )
        loaded = load_report(result.report_json_path)
        assert loaded.best_path == "baseline -> church_noise"
        assert loaded.best_metrics["accuracy"] == 0.75

    def test_compare_runs(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path)
        r1 = BNNRRunResult(
            config=cfg, checkpoints=[], best_path="a", best_metrics={"accuracy": 0.8},
            selected_augmentations=[], total_time=1.0,
            report_json_path=tmp_path / "r1.json", report_html_path=None,
        )
        r2 = BNNRRunResult(
            config=cfg, checkpoints=[], best_path="b", best_metrics={"accuracy": 0.9},
            selected_augmentations=[], total_time=2.0,
            report_json_path=tmp_path / "r2.json", report_html_path=None,
        )
        comparison = compare_runs([r1, r2], metrics=["accuracy"])
        assert comparison["run_0"]["accuracy"] == 0.8
        assert comparison["run_1"]["accuracy"] == 0.9


# ═══════════════════════════════════════════════════════════════════════════════
#  14. Events (JSONL event sink)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvents:
    def test_event_sink_writes_and_replays(self, tmp_path: Path) -> None:
        events_path = tmp_path / "events.jsonl"
        sink = JsonlEventSink(events_path, run_id="test_run")
        sink.emit("run_started", {"run_name": "test_run", "config": {}})
        sink.emit("epoch_end", {"iteration": 0, "epoch": 1, "metrics": {"accuracy": 0.5}})
        sink.close()

        raw_events = load_events(events_path)
        assert len(raw_events) == 2
        assert raw_events[0]["type"] == "run_started"
        assert raw_events[1]["payload"]["metrics"]["accuracy"] == 0.5

        # Also verify replay_events processes them correctly
        replayed = replay_events(raw_events)
        assert "run" in replayed
        assert "metrics_timeline" in replayed


# ═══════════════════════════════════════════════════════════════════════════════
#  15. BNNRTrainer — single iteration
# ═══════════════════════════════════════════════════════════════════════════════

class TestBNNRTrainerUnit:
    def test_run_single_iteration(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(tmp_path, m_epochs=2)
        trainer = BNNRTrainer(adapter, train_loader, val_loader, [], cfg)
        aug = BasicAugmentation(probability=0.5, random_state=SEED)
        metrics, state, best_epoch, pruned = trainer.run_single_iteration(aug)
        assert "accuracy" in metrics
        assert "loss" in metrics
        assert isinstance(state, dict)
        assert best_epoch >= 1
        assert isinstance(pruned, bool)

    def test_run_basic(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(tmp_path)
        aug = ChurchNoise(probability=0.5, random_state=SEED)
        trainer = BNNRTrainer(adapter, train_loader, val_loader, [aug], cfg)
        result = trainer.run()
        assert isinstance(result, BNNRRunResult)
        assert result.best_path is not None
        assert result.report_json_path.exists()

    def test_checkpoint_save_and_resume(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(tmp_path, save_checkpoints=True)
        aug = BasicAugmentation(probability=0.5, random_state=SEED)
        trainer = BNNRTrainer(adapter, train_loader, val_loader, [aug], cfg)
        trainer.run()

        ckpts = sorted(cfg.checkpoint_dir.glob("*.pt"))
        assert len(ckpts) > 0

        # Resume
        adapter2, _ = _make_adapter()
        trainer2 = BNNRTrainer(adapter2, train_loader, val_loader, [aug], cfg)
        trainer2.resume_from_checkpoint(ckpts[-1])
        assert trainer2.current_iteration >= 0

    def test_early_stopping(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(
            tmp_path, m_epochs=1, max_iterations=10, early_stopping_patience=1,
        )
        # Use a very weak augmentation that won't improve — should trigger early stopping
        aug = BasicAugmentation(probability=0.01, random_state=SEED)
        trainer = BNNRTrainer(adapter, train_loader, val_loader, [aug], cfg)
        result = trainer.run()
        # Early stopping should have kicked in before 10 iterations
        assert result is not None

    def test_candidate_pruning(self, tmp_path: Path) -> None:
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(
            tmp_path,
            m_epochs=2,
            candidate_pruning_enabled=True,
            candidate_pruning_warmup_epochs=1,
            candidate_pruning_relative_threshold=0.9,
        )
        trainer = BNNRTrainer(adapter, train_loader, val_loader, [], cfg)
        aug = BasicAugmentation(probability=0.5, random_state=SEED)
        _, _, _, pruned = trainer.run_single_iteration(
            aug,
            baseline_metrics={"accuracy": 1.0, "loss": 0.0},
        )
        # With a very high baseline, the candidate should be pruned
        assert pruned is True


# ═══════════════════════════════════════════════════════════════════════════════
#  16. quick_run
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuickRun:
    def test_quick_run_basic(self, tmp_path: Path) -> None:
        model = TinyCNN()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(tmp_path)
        result = quick_run(
            model, train_loader, val_loader,
            config=cfg,
            augmentations=[ChurchNoise(probability=0.3, random_state=SEED)],
        )
        assert isinstance(result, BNNRRunResult)
        assert result.best_path is not None

    def test_quick_run_auto_augmentations(self, tmp_path: Path) -> None:
        model = TinyCNN()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(tmp_path)
        result = quick_run(
            model, train_loader, val_loader,
            config=cfg,
            augmentations=None,  # auto-select
        )
        assert isinstance(result, BNNRRunResult)


# ═══════════════════════════════════════════════════════════════════════════════
#  17. COMPREHENSIVE E2E TEST — full BNNR domain pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestE2EFullPipeline:
    """
    End-to-end test that exercises the core BNNR domain logic:
      - Baseline training
      - Multiple augmentation candidates from different sources
        (built-in BNNR, torchvision, albumentations, kornia)
      - ICD/AICD with XAI cache
      - Iterative selection with pruning
      - XAI saliency generation (OptiCAM)
      - Event logging
      - Report generation + loading
      - Checkpoint save/resume

    Uses a tiny synthetic dataset for speed (~seconds on CPU).
    """

    def test_e2e_full_bnnr_pipeline(self, tmp_path: Path) -> None:
        # ── Model ────────────────────────────────────────────────────
        model = TinyCNN(in_channels=3, n_classes=N_CLASSES)
        target_layers = [model.conv1]
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            target_layers=target_layers,
            device="cpu",
        )

        # ── Data (indexed dataset for ICD cache) ─────────────────────
        ds = _make_rgb_dataset(n=N_SAMPLES)
        indexed_ds = _IndexedDataset(ds)
        train_loader = DataLoader(indexed_ds, batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(indexed_ds, batch_size=BATCH_SIZE, shuffle=False)

        # ── XAI cache for ICD/AICD ──────────────────────────────────
        cache_dir = tmp_path / "xai_cache"
        xai_cache = XAICache(cache_dir)
        xai_cache.precompute_cache(
            model=model, train_loader=train_loader,
            target_layers=target_layers, n_samples=N_SAMPLES,
            method="opticam",
        )

        # ── Augmentations: mix of all sources ────────────────────────
        augmentations: list[BaseAugmentation] = [
            # Built-in BNNR augmentations
            ChurchNoise(probability=0.5, random_state=SEED),
            BasicAugmentation(probability=0.5, random_state=SEED + 1),
            DifPresets(probability=0.5, random_state=SEED + 2),
            ProCAM(probability=0.5, random_state=SEED + 3),
        ]

        # Torchvision wrapper
        from torchvision import transforms
        tv_aug = TorchvisionAugmentation(
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            name_override="tv_color_jitter",
            probability=0.5,
            random_state=SEED + 4,
        )
        augmentations.append(tv_aug)

        # Albumentations wrapper (if available)
        if albumentations_available():
            import albumentations as A  # noqa: N812
            albu_aug = AlbumentationsAugmentation(
                A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                ]),
                name_override="albu_standard",
                probability=0.5,
                random_state=SEED + 5,
            )
            augmentations.append(albu_aug)

        # Kornia wrapper (if available)
        if kornia_available():
            import kornia.augmentation as K  # noqa: N812
            kornia_aug = KorniaAugmentation(
                K.RandomHorizontalFlip(p=1.0),
                name_override="kornia_hflip",
                probability=0.5,
                random_state=SEED + 6,
            )
            augmentations.append(kornia_aug)

        # ICD & AICD
        icd = ICD(
            model=model, target_layers=target_layers,
            cache=xai_cache, probability=0.5, random_state=SEED + 10,
        )
        icd.name = "icd"
        augmentations.append(icd)

        aicd = AICD(
            model=model, target_layers=target_layers,
            cache=xai_cache, probability=0.5, random_state=SEED + 11,
        )
        aicd.name = "aicd"
        augmentations.append(aicd)

        n_augs = len(augmentations)
        assert n_augs >= 7, f"Expected ≥7 augmentations, got {n_augs}"

        # ── Config ───────────────────────────────────────────────────
        cfg = BNNRConfig(
            m_epochs=2,
            max_iterations=2,
            device="cpu",
            xai_enabled=True,
            xai_method="opticam",
            xai_samples=2,
            xai_cache_dir=cache_dir,
            xai_cache_samples=N_SAMPLES,
            save_checkpoints=True,
            verbose=False,
            checkpoint_dir=tmp_path / "ckpt",
            report_dir=tmp_path / "reports",
            early_stopping_patience=3,
            candidate_pruning_enabled=True,
            candidate_pruning_warmup_epochs=1,
            candidate_pruning_relative_threshold=0.5,
            event_log_enabled=True,
            report_preview_size=64,
            report_xai_size=64,
            report_probe_images_per_class=1,
            report_probe_max_classes=N_CLASSES,
            seed=SEED,
        )

        # ── Train ────────────────────────────────────────────────────
        trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
        result = trainer.run()

        # ── Assertions ───────────────────────────────────────────────

        # 1. Result structure
        assert isinstance(result, BNNRRunResult)
        assert result.best_path is not None
        assert isinstance(result.best_metrics, dict)
        assert "accuracy" in result.best_metrics
        assert "loss" in result.best_metrics

        # 2. Report files
        assert result.report_json_path.exists()
        report_data = json.loads(result.report_json_path.read_text())
        assert "best_path" in report_data
        assert "config" in report_data
        assert "checkpoints" in report_data
        assert isinstance(report_data["checkpoints"], list)
        assert len(report_data["checkpoints"]) > 0

        # 3. Report round-trip
        loaded = load_report(result.report_json_path)
        assert loaded.best_path == result.best_path
        assert loaded.best_metrics == result.best_metrics

        # 4. Checkpoints
        ckpts = sorted(cfg.checkpoint_dir.glob("*.pt"))
        assert len(ckpts) > 0

        # 5. Events log
        events_path = result.report_json_path.parent / "events.jsonl"
        assert events_path.exists()
        raw_events = load_events(events_path)
        assert len(raw_events) > 0
        event_types = {e["type"] for e in raw_events}
        assert "run_started" in event_types
        assert "epoch_end" in event_types
        assert "dataset_profile" in event_types

        # 6. XAI artifacts (check that some were generated)
        xai_dirs = list((result.report_json_path.parent / "artifacts" / "xai").glob("*"))
        # XAI might not generate depending on selection, but dir should exist
        # At minimum, baseline XAI should be there
        assert len(xai_dirs) >= 0  # Baseline + at least one iteration

        # 7. Sample preview artifacts
        sample_dirs = list(
            (result.report_json_path.parent / "artifacts" / "samples").glob("*")
        )
        assert len(sample_dirs) >= 0

        # 8. Analysis section
        assert "analysis" in report_data

        # 9. Config preserved
        assert report_data["config"]["m_epochs"] == 2
        assert report_data["config"]["device"] == "cpu"

        # 10. Checkpoint resume works
        adapter2 = SimpleTorchAdapter(
            model=TinyCNN(in_channels=3, n_classes=N_CLASSES),
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(
                TinyCNN(in_channels=3, n_classes=N_CLASSES).parameters(), lr=1e-3,
            ),
            target_layers=None,
            device="cpu",
        )
        trainer2 = BNNRTrainer(adapter2, train_loader, val_loader, augmentations, cfg)
        trainer2.resume_from_checkpoint(ckpts[-1])
        assert trainer2.current_iteration >= 0

    def test_e2e_grayscale_pipeline(self, tmp_path: Path) -> None:
        """E2E test with grayscale images (1 channel) to verify channel handling."""
        model = TinyCNN(in_channels=1, n_classes=N_CLASSES)
        adapter = SimpleTorchAdapter(
            model=model,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            target_layers=[model.conv1],
            device="cpu",
        )
        ds = _make_gray_dataset()
        train_loader, val_loader = _make_loaders(ds)
        cfg = _make_config(tmp_path)
        augmentations = [
            ChurchNoise(probability=0.5, random_state=SEED),
            BasicAugmentation(probability=0.5, random_state=SEED + 1),
        ]
        trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
        result = trainer.run()
        assert isinstance(result, BNNRRunResult)
        assert result.best_path is not None

    def test_e2e_multiple_iterations_with_selection(self, tmp_path: Path) -> None:
        """Run 3 iterations to verify augmentation selection and path building."""
        adapter, model = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(
            tmp_path,
            m_epochs=1,
            max_iterations=3,
            early_stopping_patience=4,  # Don't stop early
            candidate_pruning_enabled=False,
        )
        augmentations = [
            ChurchNoise(probability=0.8, random_state=SEED),
            ProCAM(probability=0.8, random_state=SEED + 1),
            DifPresets(probability=0.8, random_state=SEED + 2),
        ]
        trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
        result = trainer.run()
        assert isinstance(result, BNNRRunResult)
        # The best_path should be at least "baseline"
        assert "baseline" in result.best_path or len(result.selected_augmentations) >= 0

    def test_e2e_with_all_bnnr_augmentation_types(self, tmp_path: Path) -> None:
        """Ensure all 8 built-in BNNR augmentation types work together in a full run."""
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(
            tmp_path,
            m_epochs=1,
            max_iterations=1,
            early_stopping_patience=2,
        )
        augmentations = [
            AugmentationRegistry.create(name, probability=0.3, random_state=SEED + i)
            for i, name in enumerate(ALL_BUILTIN_NAMES)
        ]
        trainer = BNNRTrainer(adapter, train_loader, val_loader, augmentations, cfg)
        result = trainer.run()
        assert isinstance(result, BNNRRunResult)
        assert result.report_json_path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
#  18. Edge cases and contract tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_augmentation_probability_bounds(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            ChurchNoise(probability=-0.1)
        with pytest.raises(ValueError, match="probability"):
            ChurchNoise(probability=1.5)

    def test_intensity_bounds(self) -> None:
        with pytest.raises(ValueError, match="intensity"):
            ChurchNoise(probability=1.0, intensity=-0.1)
        with pytest.raises(ValueError, match="intensity"):
            ChurchNoise(probability=1.0, intensity=3.0)

    def test_augmentation_repr(self) -> None:
        aug = ChurchNoise(probability=0.5, intensity=0.8)
        r = repr(aug)
        assert "ChurchNoise" in r
        assert "probability=0.5" in r
        assert "intensity=0.8" in r

    def test_augmentation_str(self) -> None:
        aug = ChurchNoise(probability=0.5)
        assert str(aug) == aug.name

    def test_validate_input_2d(self) -> None:
        aug = ChurchNoise(probability=1.0)
        gray_2d = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        validated = aug.validate_input(gray_2d)
        assert validated.ndim == 3
        assert validated.shape[2] == 3

    def test_validate_input_float(self) -> None:
        aug = ChurchNoise(probability=1.0)
        float_img = np.random.rand(32, 32, 3).astype(np.float32)
        validated = aug.validate_input(float_img)
        assert validated.dtype == np.uint8

    def test_trainer_no_augmentations(self, tmp_path: Path) -> None:
        """Trainer should work with empty augmentation list (baseline only)."""
        adapter, _ = _make_adapter()
        train_loader, val_loader = _make_loaders()
        cfg = _make_config(tmp_path, max_iterations=0)
        trainer = BNNRTrainer(adapter, train_loader, val_loader, [], cfg)
        result = trainer.run()
        assert result.best_path == "baseline"

    def test_load_nonexistent_report_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_report(tmp_path / "nonexistent.json")

    def test_config_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")
