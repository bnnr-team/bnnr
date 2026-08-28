"""Tests for GPU-native augmentation paths and AugmentationRunner."""

from __future__ import annotations

import numpy as np
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
from bnnr.training.image_utils import NormalisedInputError


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


class TestRunnerNormalisedBatches:
    """The runner used to carry the copy of the converter without any check."""

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def _normalise(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.STD).view(1, 3, 1, 1)
        return (images - mean) / std

    def test_normalised_batch_raises_without_stats(self) -> None:
        aug = BasicAugmentation(probability=1.0, random_state=0)
        aug.device_compatible = False  # type: ignore[attr-defined]
        runner = AugmentationRunner([aug], async_prefetch=False)
        images = self._normalise(_make_batch())

        with pytest.raises(NormalisedInputError):
            runner.apply_batch(images, torch.zeros(4, dtype=torch.long))

    def test_normalised_batch_survives_with_stats(self) -> None:
        """With stats the batch stays in its own convention and keeps its spread."""
        aug = BasicAugmentation(probability=1.0, random_state=0)
        aug.device_compatible = False  # type: ignore[attr-defined]
        runner = AugmentationRunner(
            [aug], async_prefetch=False, denorm_mean=self.MEAN, denorm_std=self.STD
        )
        images = self._normalise(_make_batch())

        out, _ = runner.apply_batch(images, torch.zeros(4, dtype=torch.long))

        assert out.shape == images.shape
        # Clipping to [0, 1] would have collapsed the normalised range.
        assert float(out.min()) < -0.5
        assert float(out.max()) > 0.5
        assert torch.isfinite(out).all()


class TestChurchNoiseModes:
    """CPU and tensor paths must run the same transform (FIX-0-3)."""

    @staticmethod
    def _plan_for(**kwargs):
        """Redraw the plan a tensor-path call would use for the first image.

        ``apply_tensor_native`` consumes one draw for the probability check and
        one to seed the torch generator before it asks for a plan, so mirror
        that here.
        """
        aug = ChurchNoise(probability=1.0, **kwargs)
        aug._rnd.random()
        aug._rnd.randrange(0, 2**63 - 1)
        return aug._regional_plan(64, 64)

    def test_regional_is_the_default_on_both_paths(self) -> None:
        assert ChurchNoise().noise_mode == "regional"

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="noise_mode"):
            ChurchNoise(noise_mode="sometimes")

    def test_num_lines_affects_the_tensor_path(self) -> None:
        """num_lines used to be ignored once a tensor path was available."""
        _, few = self._plan_for(random_state=7, num_lines=1)
        _, many = self._plan_for(random_state=7, num_lines=3)
        assert len(few) == 2
        assert len(many) > len(few)

    def test_tensor_regional_noise_follows_the_per_region_plan(self) -> None:
        """Each region gets its own standard deviation, as on the numpy path."""
        aug = ChurchNoise(probability=1.0, random_state=11, num_lines=1)
        regions, plan = self._plan_for(random_state=11, num_lines=1)

        images = torch.full((1, 3, 64, 64), 0.5)
        out = aug.apply_tensor_native(images)
        diff = (out - images)[0, 0].numpy()

        for region, std, _kind in plan:
            observed = float(diff[regions == region].std())
            assert observed == pytest.approx(std / 255.0, rel=0.15)

    def test_numpy_regional_noise_follows_the_per_region_plan(self) -> None:
        aug = ChurchNoise(probability=1.0, random_state=11, num_lines=1)
        plan_aug = ChurchNoise(probability=1.0, random_state=11, num_lines=1)
        regions, plan = plan_aug._regional_plan(64, 64)

        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = aug.apply(image)
        diff = out.astype(np.float32)[:, :, 0] - 128.0

        for region, std, _kind in plan:
            observed = float(diff[regions == region].std())
            assert observed == pytest.approx(std, rel=0.15)

    def test_uniform_mode_has_one_std_everywhere(self) -> None:
        aug = ChurchNoise(probability=1.0, random_state=3, noise_mode="uniform")
        images = torch.full((1, 3, 64, 64), 0.5)
        diff = (aug.apply_tensor_native(images) - images)[0, 0].numpy()

        halves = [float(diff[:32].std()), float(diff[32:].std())]
        assert halves[0] == pytest.approx(halves[1], rel=0.12)

    def test_regional_mode_varies_across_the_image(self) -> None:
        """The property uniform mode cannot have, on the path that used to lack it."""
        aug = ChurchNoise(probability=1.0, random_state=11, num_lines=1)
        regions, plan = self._plan_for(random_state=11, num_lines=1)
        stds = sorted(std for _region, std, _kind in plan)
        planned_ratio = stds[-1] / stds[0]
        assert planned_ratio > 1.05  # this seed does draw different stds per region

        images = torch.full((1, 3, 64, 64), 0.5)
        diff = (aug.apply_tensor_native(images) - images)[0, 0].numpy()
        observed = sorted(float(diff[regions == region].std()) for region, _s, _k in plan)
        assert observed[-1] / observed[0] == pytest.approx(planned_ratio, rel=0.2)

    def test_tensor_path_is_deterministic_for_a_seed(self) -> None:
        images = torch.full((2, 3, 32, 32), 0.5)
        first = ChurchNoise(probability=1.0, random_state=5).apply_tensor_native(images)
        second = ChurchNoise(probability=1.0, random_state=5).apply_tensor_native(images)
        assert torch.equal(first, second)

    def test_both_paths_produce_comparable_noise_magnitude(self) -> None:
        """The two paths are the same transform, so their noise scale matches."""
        cpu_out = ChurchNoise(probability=1.0, random_state=2, num_lines=3).apply(
            np.full((96, 96, 3), 128, dtype=np.uint8)
        )
        cpu_std = float((cpu_out.astype(np.float32)[:, :, 0] - 128.0).std()) / 255.0

        images = torch.full((1, 3, 96, 96), 0.5)
        tensor_out = ChurchNoise(
            probability=1.0, random_state=2, num_lines=3
        ).apply_tensor_native(images)
        tensor_std = float((tensor_out - images)[0, 0].numpy().std())

        assert tensor_std == pytest.approx(cpu_std, rel=0.35)
