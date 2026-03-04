"""Tests for GPU-native augmentation paths and AugmentationRunner."""

from __future__ import annotations

import pytest
import torch

from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import (
    BasicAugmentation,
    ChurchNoise,
    DifPresets,
    ProCAM,
)


def _make_batch(b: int = 4, c: int = 3, h: int = 32, w: int = 32) -> torch.Tensor:
    """Create a random BCHW float32 tensor in [0, 1]."""
    return torch.rand(b, c, h, w, dtype=torch.float32)


class TestGPUNativeAugmentations:
    """Verify that augmentations with device_compatible=True produce valid tensors."""

    @pytest.mark.parametrize(
        "aug_cls",
        [ChurchNoise, ProCAM, DifPresets],
    )
    def test_gpu_augmentation_output_shape(self, aug_cls: type) -> None:
        aug = aug_cls(probability=1.0, random_state=42)
        assert aug.device_compatible is True
        images = _make_batch()
        result = aug.apply_tensor_native(images)
        assert result.shape == images.shape
        assert result.dtype == images.dtype

    @pytest.mark.parametrize(
        "aug_cls",
        [ChurchNoise, ProCAM, DifPresets],
    )
    def test_gpu_augmentation_output_range(self, aug_cls: type) -> None:
        aug = aug_cls(probability=1.0, random_state=42)
        images = _make_batch()
        result = aug.apply_tensor_native(images)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize(
        "aug_cls",
        [ChurchNoise, ProCAM, DifPresets],
    )
    def test_gpu_augmentation_modifies_input(self, aug_cls: type) -> None:
        aug = aug_cls(probability=1.0, random_state=42)
        images = _make_batch()
        result = aug.apply_tensor_native(images)
        # Augmentation should change at least some values
        assert not torch.allclose(result, images, atol=1e-6)

    @pytest.mark.parametrize(
        "aug_cls",
        [ChurchNoise, ProCAM, DifPresets],
    )
    def test_gpu_augmentation_probability_zero_passthrough(self, aug_cls: type) -> None:
        aug = aug_cls(probability=0.0, random_state=42)
        images = _make_batch()
        result = aug.apply_tensor_native(images)
        assert torch.allclose(result, images)

    def test_gpu_augmentation_intensity_blend(self) -> None:
        aug = ChurchNoise(probability=1.0, intensity=0.5, random_state=42)
        images = _make_batch()
        result = aug.apply_tensor_native(images)
        # Should be closer to original than full-intensity
        full_aug = ChurchNoise(probability=1.0, intensity=1.0, random_state=42)
        full_result = full_aug.apply_tensor_native(images)
        diff_partial = (result - images).abs().mean()
        diff_full = (full_result - images).abs().mean()
        assert diff_partial < diff_full


class TestAugmentationRunner:
    """Test the AugmentationRunner sync and async dispatch."""

    def test_sync_dispatch(self) -> None:
        augs = [ChurchNoise(probability=1.0, random_state=42)]
        runner = AugmentationRunner(augs, async_prefetch=False)
        images = _make_batch()
        labels = torch.zeros(4, dtype=torch.long)
        result_images, result_labels = runner.apply_batch(images, labels)
        assert result_images.shape == images.shape
        assert torch.equal(result_labels, labels)

    def test_sync_dispatch_with_cpu_aug(self) -> None:
        augs = [BasicAugmentation(probability=1.0, random_state=42)]
        runner = AugmentationRunner(augs, async_prefetch=False)
        images = _make_batch()
        labels = torch.zeros(4, dtype=torch.long)
        result_images, result_labels = runner.apply_batch(images, labels)
        assert result_images.shape == images.shape

    def test_async_iter_loader(self) -> None:
        """Test async prefetch with a simple data loader."""
        augs = [BasicAugmentation(probability=1.0, random_state=42)]
        runner = AugmentationRunner(augs, async_prefetch=True)

        # Simulate a DataLoader
        batches = [(_make_batch(), torch.zeros(4, dtype=torch.long)) for _ in range(5)]
        results = list(runner.iter_loader(batches))
        assert len(results) == 5
        for images, labels in results:
            assert images.shape == (4, 3, 32, 32)
            assert labels.shape == (4,)

    def test_sync_iter_loader_no_cpu_augs(self) -> None:
        """When all augs are GPU-native, iter_loader should still work (sync path)."""
        augs = [ChurchNoise(probability=1.0, random_state=42)]
        runner = AugmentationRunner(augs, async_prefetch=True)

        batches = [(_make_batch(), torch.zeros(4, dtype=torch.long)) for _ in range(3)]
        results = list(runner.iter_loader(batches))
        assert len(results) == 3

    def test_runner_splits_gpu_and_cpu_augs(self) -> None:
        gpu_aug = ChurchNoise(probability=1.0, random_state=42)
        cpu_aug = BasicAugmentation(probability=1.0, random_state=42)
        runner = AugmentationRunner([gpu_aug, cpu_aug])
        assert gpu_aug in runner.gpu_augmentations
        assert cpu_aug in runner.cpu_augmentations

    def test_runner_empty_augmentations(self) -> None:
        runner = AugmentationRunner([])
        images = _make_batch()
        labels = torch.zeros(4, dtype=torch.long)
        result_images, result_labels = runner.apply_batch(images, labels)
        assert torch.equal(result_images, images)
        assert torch.equal(result_labels, labels)


class TestApplyTensorFallback:
    """Test the apply_tensor fallback for CPU-bound augmentations."""

    def test_cpu_aug_apply_tensor_works(self) -> None:
        aug = BasicAugmentation(probability=1.0, random_state=42)
        assert aug.device_compatible is False
        images = _make_batch()
        result = aug.apply_tensor(images)
        assert result.shape == images.shape
        assert result.dtype == images.dtype
