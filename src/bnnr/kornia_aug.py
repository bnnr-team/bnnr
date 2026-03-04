"""GPU-native augmentation wrappers using Kornia.

This module provides a ``KorniaAugmentation`` wrapper that adapts
Kornia augmentation modules to the BNNR ``BaseAugmentation`` interface,
enabling fully GPU-native augmentation pipelines.

Requires the ``[gpu]`` extra::

    pip install bnnr[gpu]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import Tensor

from bnnr.augmentations import AugmentationRegistry, BaseAugmentation

logger = logging.getLogger(__name__)

_KORNIA_AVAILABLE = False
try:
    import kornia.augmentation as K  # noqa: N812

    _KORNIA_AVAILABLE = True
except ImportError:
    K = None  # type: ignore[assignment]


def kornia_available() -> bool:
    """Return True if kornia is installed."""
    return _KORNIA_AVAILABLE


class KorniaAugmentation(BaseAugmentation):
    """Wrap a ``kornia.augmentation.AugmentationBase2D`` as a BNNR augmentation.

    The augmentation operates entirely on GPU tensors (BCHW, float32,
    in [0, 1]).  When ``apply_batch`` is called with numpy arrays, the
    wrapper transparently converts to/from tensors.

    Parameters
    ----------
    kornia_transform:
        A Kornia augmentation instance (e.g. ``K.RandomHorizontalFlip(p=0.5)``).
    name_override:
        Human-readable name for display / registry.
    probability:
        Per-batch probability.  Note that Kornia transforms also have
        their own internal ``p`` parameter – make sure to set it on the
        Kornia transform itself.
    """

    name: str = "kornia"
    device_compatible: bool = True

    def __init__(
        self,
        kornia_transform: Any,
        probability: float = 1.0,
        intensity: float = 1.0,
        name_override: str | None = None,
        random_state: int | None = None,
    ) -> None:
        if not _KORNIA_AVAILABLE:
            raise ImportError(
                "Kornia is required for KorniaAugmentation. "
                "Install it with: pip install bnnr[gpu]"
            )
        super().__init__(
            probability=probability,
            intensity=intensity,
            name_override=name_override or f"kornia_{type(kornia_transform).__name__}",
            random_state=random_state,
        )
        self._transform = kornia_transform

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single HWC uint8 numpy image."""
        image = self.validate_input(image)
        # Convert to BCHW float32 tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        augmented = self._transform(tensor)
        # Convert back to HWC uint8
        result = (augmented.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
        return result

    def apply_tensor_native(self, images: Tensor) -> Tensor:
        """Apply augmentation directly on BCHW float32 tensors (GPU-native)."""
        if self._rnd.random() > self.probability:
            return images
        augmented = self._transform(images)
        if self.intensity < 1.0:
            augmented = images * (1.0 - self.intensity) + augmented * self.intensity
        return augmented

    def apply_tensor(self, images: Tensor) -> Tensor:
        """Override to use GPU-native path directly."""
        return self.apply_tensor_native(images)


def create_kornia_pipeline(
    transforms: list[Any],
    probability: float = 1.0,
    name: str = "kornia_pipeline",
) -> KorniaAugmentation:
    """Create a single BNNR augmentation from a list of Kornia transforms.

    Parameters
    ----------
    transforms:
        List of Kornia augmentation instances to compose.
    probability:
        Per-batch application probability.
    name:
        Name for the composed augmentation.

    Returns
    -------
    KorniaAugmentation wrapping a ``K.AugmentationSequential``.
    """
    if not _KORNIA_AVAILABLE:
        raise ImportError(
            "Kornia is required for create_kornia_pipeline. "
            "Install it with: pip install bnnr[gpu]"
        )
    pipeline = K.AugmentationSequential(*transforms, data_keys=["input"])
    return KorniaAugmentation(
        kornia_transform=pipeline,
        probability=probability,
        name_override=name,
    )


# ---------------------------------------------------------------------------
# Pre-built GPU-native augmentations using Kornia
# ---------------------------------------------------------------------------


def _register_kornia_augmentations() -> None:
    """Register GPU-native Kornia augmentations if Kornia is available.

    Called at module load time. If Kornia is not installed, no augmentations
    are registered and no error is raised.
    """
    if not _KORNIA_AVAILABLE:
        return

    @AugmentationRegistry.register("kornia_hflip")
    class KorniaHFlip(KorniaAugmentation):
        name = "kornia_hflip"

        def __init__(self, probability: float = 0.5, **kwargs: Any) -> None:
            super().__init__(
                kornia_transform=K.RandomHorizontalFlip(p=1.0),
                probability=probability,
                name_override="kornia_hflip",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_vflip")
    class KorniaVFlip(KorniaAugmentation):
        name = "kornia_vflip"

        def __init__(self, probability: float = 0.5, **kwargs: Any) -> None:
            super().__init__(
                kornia_transform=K.RandomVerticalFlip(p=1.0),
                probability=probability,
                name_override="kornia_vflip",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_rotation")
    class KorniaRotation(KorniaAugmentation):
        name = "kornia_rotation"

        def __init__(self, probability: float = 0.5, degrees: float = 30.0, **kwargs: Any) -> None:
            super().__init__(
                kornia_transform=K.RandomRotation(degrees=degrees, p=1.0),
                probability=probability,
                name_override="kornia_rotation",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_color_jitter")
    class KorniaColorJitter(KorniaAugmentation):
        name = "kornia_color_jitter"

        def __init__(
            self,
            probability: float = 0.5,
            brightness: float = 0.2,
            contrast: float = 0.2,
            saturation: float = 0.2,
            hue: float = 0.1,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                kornia_transform=K.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    p=1.0,
                ),
                probability=probability,
                name_override="kornia_color_jitter",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_gaussian_blur")
    class KorniaGaussianBlur(KorniaAugmentation):
        name = "kornia_gaussian_blur"

        def __init__(
            self,
            probability: float = 0.5,
            kernel_size: tuple[int, int] = (5, 5),
            sigma: tuple[float, float] = (0.1, 2.0),
            **kwargs: Any,
        ) -> None:
            super().__init__(
                kornia_transform=K.RandomGaussianBlur(
                    kernel_size=kernel_size,
                    sigma=sigma,
                    p=1.0,
                ),
                probability=probability,
                name_override="kornia_gaussian_blur",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_erasing")
    class KorniaErasing(KorniaAugmentation):
        name = "kornia_erasing"

        def __init__(self, probability: float = 0.5, **kwargs: Any) -> None:
            super().__init__(
                kornia_transform=K.RandomErasing(p=1.0),
                probability=probability,
                name_override="kornia_erasing",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_affine")
    class KorniaAffine(KorniaAugmentation):
        name = "kornia_affine"

        def __init__(
            self,
            probability: float = 0.5,
            degrees: float = 15.0,
            translate: tuple[float, float] = (0.1, 0.1),
            scale: tuple[float, float] = (0.9, 1.1),
            **kwargs: Any,
        ) -> None:
            super().__init__(
                kornia_transform=K.RandomAffine(
                    degrees=degrees,
                    translate=translate,
                    scale=scale,
                    p=1.0,
                ),
                probability=probability,
                name_override="kornia_affine",
                **kwargs,
            )

    @AugmentationRegistry.register("kornia_normalize")
    class KorniaNormalize(KorniaAugmentation):
        name = "kornia_normalize"

        def __init__(
            self,
            mean: tuple[float, ...] = (0.485, 0.456, 0.406),
            std: tuple[float, ...] = (0.229, 0.224, 0.225),
            probability: float = 1.0,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                kornia_transform=K.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std),
                    p=1.0,
                ),
                probability=probability,
                name_override="kornia_normalize",
                **kwargs,
            )


# Auto-register at import time
_register_kornia_augmentations()
