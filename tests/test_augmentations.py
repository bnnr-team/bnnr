"""Tests for augmentation registration and behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bnnr.augmentations import AugmentationRegistry, BaseAugmentation, TorchvisionAugmentation


class _DummyAug(BaseAugmentation):
    name = "dummy"

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image


def test_augmentation_registry_register() -> None:
    AugmentationRegistry._registry.pop("dummy_aug", None)

    @AugmentationRegistry.register("dummy_aug")
    class DummyAug(_DummyAug):
        pass

    assert AugmentationRegistry.is_registered("dummy_aug")
    assert AugmentationRegistry.get("dummy_aug") is DummyAug


def test_augmentation_registry_list_all() -> None:
    names = AugmentationRegistry.list_all()
    assert "basic_augmentation" in names
    assert "church_noise" in names
    assert "augmentation_1" in names
    assert "augmentation_10" in names
    # Removed augmentations must not be registered.
    assert "compress_blur" not in names
    assert "memory_killer" not in names
    assert "augmentation_2" not in names
    assert "augmentation_4" not in names


def test_registered_augmentations_apply_shape() -> None:
    image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    for name in [
        "church_noise",
        "basic_augmentation",
        "dif_presets",
        "drust",
        "luxfer_glass",
        "procam",
        "smugs",
        "tea_stains",
    ]:
        aug = AugmentationRegistry.create(name, probability=1.0, random_state=42)
        out = aug.apply(image)
        assert out.shape == image.shape
        assert out.dtype == np.uint8


def test_probability_in_apply_batch() -> None:
    image = (np.random.rand(2, 32, 32, 3) * 255).astype(np.uint8)
    aug = AugmentationRegistry.create("basic_augmentation", probability=0.0, random_state=1)
    out = aug.apply_batch(image)
    assert np.all(out == image)


def test_grayscale_batch_shape_preserved() -> None:
    image = (np.random.rand(2, 28, 28, 1) * 255).astype(np.uint8)
    aug = AugmentationRegistry.create("augmentation_3", probability=1.0, random_state=1)
    out = aug.apply_batch(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_grayscale_church_noise_shape_preserved_with_blending() -> None:
    image = (np.random.rand(2, 28, 28, 1) * 255).astype(np.uint8)
    aug = AugmentationRegistry.create(
        "augmentation_1",
        probability=1.0,
        random_state=1,
        intensity=0.5,
    )
    out = aug.apply_batch(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_registry_warns_for_cpu_bound_augmentation() -> None:
    AugmentationRegistry._cpu_warning_emitted = False
    with pytest.warns(RuntimeWarning, match="CPU-bound"):
        _ = AugmentationRegistry.create("basic_augmentation", probability=1.0, random_state=1)


def test_device_compatible_apply_tensor_native_path() -> None:
    class _TensorOnlyAug(BaseAugmentation):
        name = "tensor_only"
        device_compatible = True

        def apply(self, image: np.ndarray) -> np.ndarray:
            return image

        def apply_tensor_native(self, images: torch.Tensor) -> torch.Tensor:
            return images + 1.0

    aug = _TensorOnlyAug(probability=1.0, random_state=1)
    images = torch.zeros(2, 3, 4, 4)
    out = aug.apply_tensor(images)
    assert torch.allclose(out, torch.ones_like(images))


def test_intensity_blending() -> None:
    """Intensity < 1.0 should blend augmented output with original."""
    image = (np.random.rand(2, 32, 32, 3) * 255).astype(np.uint8)
    # intensity=0.0 means no augmentation effect at all.
    aug = AugmentationRegistry.create("church_noise", probability=1.0, random_state=42, intensity=0.0)
    out = aug.apply_batch(image)
    assert np.allclose(out, image, atol=1)  # allow rounding tolerance


def test_intensity_validation() -> None:
    with pytest.raises(ValueError, match="intensity"):
        _DummyAug(probability=1.0, intensity=-0.5)
    with pytest.raises(ValueError, match="intensity"):
        _DummyAug(probability=1.0, intensity=3.0)


def test_name_override() -> None:
    aug = AugmentationRegistry.create("procam", probability=0.5, name_override="procam_v2")
    assert aug.name == "procam_v2"


def test_torchvision_augmentation() -> None:
    """TorchvisionAugmentation should wrap a PIL callable and produce valid output."""
    from PIL import ImageFilter

    class _SimpleBlur:
        def __call__(self, pil_image):
            return pil_image.filter(ImageFilter.GaussianBlur(radius=1))

    aug = TorchvisionAugmentation(_SimpleBlur(), name_override="pil_blur", probability=1.0, random_state=1)
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    out = aug.apply(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert aug.name == "pil_blur"
