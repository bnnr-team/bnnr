"""Albumentations integration for BNNR.

Wraps Albumentations transforms as BNNR ``BaseAugmentation`` instances,
allowing them to participate in BNNR's iterative augmentation selection.

Requires the ``[albumentations]`` extra::

    pip install bnnr[albumentations]

Example::

    import albumentations as A
    from bnnr.albumentations_aug import AlbumentationsAugmentation

    aug = AlbumentationsAugmentation(
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ]),
        name_override="albu_standard",
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bnnr.augmentations import BaseAugmentation

logger = logging.getLogger(__name__)

_ALBUMENTATIONS_AVAILABLE = False
try:
    import albumentations as A  # noqa: N812

    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None  # type: ignore[assignment]


def albumentations_available() -> bool:
    """Return True if albumentations is installed."""
    return _ALBUMENTATIONS_AVAILABLE


class AlbumentationsAugmentation(BaseAugmentation):
    """Wrap an Albumentations transform as a BNNR augmentation.

    The Albumentations transform is applied per-image on CPU (HWC uint8).
    Probability is controlled by the BNNR ``probability`` parameter;
    the internal Albumentations ``p`` values are respected as-is.

    Parameters
    ----------
    transform:
        An Albumentations ``Compose`` or single transform instance.
    name_override:
        Human-readable name for display / registry.
    probability:
        Per-image probability of applying this augmentation (BNNR-level).
    """

    name: str = "albumentations"
    device_compatible: bool = False  # Albumentations is CPU-only

    def __init__(
        self,
        transform: Any,
        probability: float = 1.0,
        intensity: float = 1.0,
        name_override: str | None = None,
        random_state: int | None = None,
    ) -> None:
        if not _ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "Albumentations is required for AlbumentationsAugmentation. "
                "Install it with: pip install bnnr[albumentations]"
            )
        super().__init__(
            probability=probability,
            intensity=intensity,
            name_override=name_override or "albumentations",
            random_state=random_state,
        )
        self._transform = transform

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the Albumentations transform to a single HWC uint8 image."""
        image = self.validate_input(image)
        result = self._transform(image=image)
        augmented: np.ndarray = result["image"]
        if augmented.ndim == 2:
            augmented = np.stack([augmented] * 3, axis=-1)
        return augmented.astype(np.uint8)


__all__ = [
    "AlbumentationsAugmentation",
    "albumentations_available",
]
