"""Tests for GPU-native augmentation paths and AugmentationRunner."""

from __future__ import annotations

import pytest
import torch

from bnnr.augmentation_runner import AugmentationRunner
from bnnr.augmentations import (
    BaseAugmentation,
    BasicAugmentation,
    ChurchNoise,
    DifPresets,
    ProCAM,
)


def _make_batch(b: int = 4, c: int = 3, h: int = 32, w: int = 32) -> torch.Tensor:
    """Create a random BCHW float32 tensor in [0, 1]."""
    return torch.rand(b, c, h, w, dtype=torch.float32)


class _OrderRecorder(BaseAugmentation):
    """Records the order in which it is applied; CPU or GPU per flag."""

    def __init__(self, tag: str, order: list, *, device_compatible: bool) -> None:
        super().__init__(probability=1.0, name_override=tag)
        self._tag = tag
        self._order = order
        self.device_compatible = device_compatible

    def apply(self, image):  # type: ignore[override]
        return image

    def apply_tensor(self, images):  # type: ignore[override]
        if not self.device_compatible:
            raise NotImplementedError
        self._order.append(self._tag)
        return images

    def apply_batch(self, np_images):  # type: ignore[override]
        self._order.append(self._tag)
        return np_images


class _IndexRecordingGPUAug(BaseAugmentation):
    """GPU-native, index-aware aug that records the sample_indices it receives."""

    device_compatible = True

    def __init__(self, seen: list) -> None:
        super().__init__(probability=1.0, name_override="idx_gpu")
        self._seen = seen

    def apply(self, image):  # type: ignore[override]
        return image

    def apply_batch_with_labels(self, np_images, np_labels, sample_indices=None):
        self._seen.append(sample_indices)
        return np_images


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

    def test_apply_batch_respects_list_order(self) -> None:
        """apply_batch must apply augs in list order, not GPU-then-CPU."""
        labels = torch.zeros(4, dtype=torch.long)

        order_cg: list = []
        cpu = _OrderRecorder("cpu", order_cg, device_compatible=False)
        gpu = _OrderRecorder("gpu", order_cg, device_compatible=True)
        AugmentationRunner([cpu, gpu], async_prefetch=False).apply_batch(
            _make_batch(), labels
        )
        assert order_cg == ["cpu", "gpu"]

        order_gc: list = []
        cpu2 = _OrderRecorder("cpu", order_gc, device_compatible=False)
        gpu2 = _OrderRecorder("gpu", order_gc, device_compatible=True)
        AugmentationRunner([gpu2, cpu2], async_prefetch=False).apply_batch(
            _make_batch(), labels
        )
        assert order_gc == ["gpu", "cpu"]

    def test_interleaved_list_uses_sync_path_in_order(self) -> None:
        """An interleaved CPU/GPU list falls back to the sync path, in order."""
        order: list = []
        cpu_a = _OrderRecorder("cpu_a", order, device_compatible=False)
        gpu = _OrderRecorder("gpu", order, device_compatible=True)
        cpu_b = _OrderRecorder("cpu_b", order, device_compatible=False)
        runner = AugmentationRunner([cpu_a, gpu, cpu_b], async_prefetch=True)
        assert runner._cpu_then_gpu is False
        batches = [(_make_batch(), torch.zeros(4, dtype=torch.long))]
        list(runner.iter_loader(batches))
        assert order == ["cpu_a", "gpu", "cpu_b"]

    def test_async_gpu_stage_receives_sample_indices(self) -> None:
        """In the async path, the GPU stage must get real sample_indices."""
        import numpy as np

        seen: list = []
        cpu_aug = BasicAugmentation(probability=1.0, random_state=0)  # CPU → async
        gpu_idx = _IndexRecordingGPUAug(seen)  # GPU, index-aware, runs after
        runner = AugmentationRunner([cpu_aug, gpu_idx], async_prefetch=True)
        assert runner._cpu_then_gpu is True

        batches = [
            (_make_batch(), torch.zeros(4, dtype=torch.long), torch.arange(0, 4)),
            (_make_batch(), torch.zeros(4, dtype=torch.long), torch.arange(4, 8)),
        ]
        list(runner.iter_loader(batches))

        assert len(seen) == 2
        assert all(idx is not None for idx in seen)  # not hash-keyed
        assert np.array_equal(seen[0], np.array([0, 1, 2, 3]))


class TestApplyTensorFallback:
    """Test the apply_tensor fallback for CPU-bound augmentations."""

    def test_cpu_aug_apply_tensor_works(self) -> None:
        aug = BasicAugmentation(probability=1.0, random_state=42)
        assert aug.device_compatible is False
        images = _make_batch()
        result = aug.apply_tensor(images)
        assert result.shape == images.shape
        assert result.dtype == images.dtype
