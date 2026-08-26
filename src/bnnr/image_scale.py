"""Batch value-range conventions shared by every image conversion in BNNR.

Batches reach BNNR in one of three conventions, and the augmentations only
understand the first two:

``unit``
    float in [0, 1]
``byte``
    float carrying [0, 255] values
``normalised``
    float after ``transforms.Normalize()``, which BNNR must undo before
    augmenting and redo afterwards

This module is deliberately free of BNNR imports so that both
``bnnr.augmentations`` and ``bnnr.training.image_utils`` can depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

# Tolerances are expressed as a fraction of the convention's own scale, so the
# same 1 % undershoot allowance covers interpolation ringing in both.
_UNIT_MAX = 1.05
_BYTE_MAX = 260.0
_UNDERSHOOT = 0.01

# A "byte" batch whose brightest pixel sits below this is indistinguishable
# from a normalised batch by range alone: 32/255 is a 12 %-grey image, while a
# normalised batch reaching 32 would need a std near 0.03, which no standard
# normalisation uses. Inside that band we refuse rather than guess, because
# guessing wrong destroys the image silently.
_BYTE_MIN_PEAK = 32.0


class NormalisedInputError(ValueError):
    """Raised when a batch cannot be converted to uint8 without destroying it."""


@dataclass(frozen=True)
class BatchScale:
    """The convention a batch of images is expressed in.

    ``mean`` and ``std`` are populated only for ``kind="normalised"`` and are
    the values needed to undo the normalisation.
    """

    kind: Literal["unit", "byte", "normalised"]
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None


def detect_batch_scale(
    images: Tensor | np.ndarray,
    *,
    denorm_mean: list[float] | tuple[float, ...] | None = None,
    denorm_std: list[float] | tuple[float, ...] | None = None,
) -> BatchScale:
    """Classify *images* into one of the three conventions.

    An in-range batch is classified from its range alone, so configuring
    ``denormalization_mean`` / ``denormalization_std`` for reporting never
    changes how a plain [0, 1] or [0, 255] batch is treated. The stats are
    consulted only once the range rules out both, which is exactly the case
    that used to be silently clipped.

    Raises
    ------
    NormalisedInputError
        When the range fits neither convention and no denormalisation stats
        were configured.
    """
    lo = float(images.min())
    hi = float(images.max())

    if lo >= -_UNDERSHOOT and hi <= _UNIT_MAX:
        return BatchScale("unit")
    if lo >= -_UNDERSHOOT * 255.0 and _BYTE_MIN_PEAK <= hi <= _BYTE_MAX:
        return BatchScale("byte")

    if denorm_mean is not None and denorm_std is not None:
        return BatchScale(
            "normalised",
            tuple(float(m) for m in denorm_mean),
            tuple(float(s) for s in denorm_std),
        )

    raise NormalisedInputError(
        f"BNNR received a batch with values in [{lo:.3f}, {hi:.3f}], which is "
        "neither [0, 1] nor [0, 255]. This usually means transforms.Normalize() "
        "was applied before BNNR augmentations, which operate on unnormalised "
        "uint8 images. Either remove Normalize() from your DataLoader "
        "transforms and rely on BatchNorm in the model, or set "
        "denormalization_mean and denormalization_std in the BNNR config so "
        "BNNR can undo and redo the normalisation around each augmentation."
    )


def denorm_arrays(scale: BatchScale, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate and materialise the per-channel mean/std of a normalised scale."""
    mean = np.asarray(scale.mean, dtype=np.float32)
    std = np.asarray(scale.std, dtype=np.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise NormalisedInputError(
            f"denormalization_mean/std have {mean.size}/{std.size} entries but the "
            f"batch has {channels} channels."
        )
    if not np.all(std > 0):
        raise NormalisedInputError("denormalization_std entries must all be positive.")
    return mean, std


def _channel_stats(scale: BatchScale, images: Tensor) -> tuple[Tensor, Tensor]:
    mean_np, std_np = denorm_arrays(scale, int(images.shape[1]))
    mean = torch.as_tensor(mean_np, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std_np, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return mean, std


def to_unit(images: Tensor, scale: BatchScale) -> Tensor:
    """Convert a BCHW batch from *scale* into the [0, 1] convention."""
    if scale.kind == "unit":
        return images
    if scale.kind == "byte":
        return images / 255.0
    mean, std = _channel_stats(scale, images)
    return (images * std + mean).clamp(0.0, 1.0)


def from_unit(images: Tensor, scale: BatchScale) -> Tensor:
    """Convert a BCHW batch in [0, 1] back into *scale*."""
    if scale.kind == "unit":
        return images
    if scale.kind == "byte":
        return images * 255.0
    mean, std = _channel_stats(scale, images)
    return (images - mean) / std


__all__ = [
    "BatchScale",
    "NormalisedInputError",
    "denorm_arrays",
    "detect_batch_scale",
    "from_unit",
    "to_unit",
]
