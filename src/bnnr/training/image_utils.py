"""Tensor ↔ uint8 conversion utilities for training pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from bnnr.image_scale import (
    _UNIT_MAX,
    BatchScale,
    NormalisedInputError,
    detect_batch_scale,
)
from bnnr.image_scale import (
    denorm_arrays as _denorm_arrays,
)
from bnnr.utils import lazy_cv2 as cv2

__all__ = [
    "BatchScale",
    "NormalisedInputError",
    "det_uint8_batch_to_float01",
    "detect_batch_scale",
    "resize_saliency_batch",
    "tensor_batch_to_preview_uint8",
    "tensor_to_uint8",
    "uint8_to_tensor",
]


def tensor_to_uint8(
    images: Tensor,
    *,
    scale: BatchScale | None = None,
    denorm_mean: list[float] | tuple[float, ...] | None = None,
    denorm_std: list[float] | tuple[float, ...] | None = None,
    warn_context: Any | None = None,
) -> np.ndarray:
    """Convert a (B, C, H, W) float tensor to a (B, H, W, C) uint8 array.

    Pass *scale* when the caller also has to invert the conversion, so both
    directions agree on the convention instead of re-deriving it from data
    that the augmentation has since changed. When omitted it is detected via
    :func:`detect_batch_scale`, which raises rather than clipping a batch it
    cannot represent.

    *warn_context* is accepted for backward compatibility and is unused: the
    case it used to warn about now raises :class:`NormalisedInputError`.
    """
    np_images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    if scale is None:
        scale = detect_batch_scale(np_images, denorm_mean=denorm_mean, denorm_std=denorm_std)

    if scale.kind == "unit":
        return np.clip(np_images * 255.0, 0.0, 255.0).astype("uint8")  # type: ignore[no-any-return]
    if scale.kind == "byte":
        return np.clip(np_images, 0.0, 255.0).astype("uint8")  # type: ignore[no-any-return]

    mean, std = _denorm_arrays(scale, np_images.shape[-1])
    denormalised = np.clip(np_images * std + mean, 0.0, 1.0)
    return (denormalised * 255.0).astype("uint8")  # type: ignore[no-any-return]


def uint8_to_tensor(
    np_images: np.ndarray,
    *,
    ref_batch: Tensor,
    scale: BatchScale | None = None,
) -> Tensor:
    """Convert (B, H, W, C) uint8 back to (B, C, H, W) float tensor.

    With *scale* the conversion is the exact inverse of :func:`tensor_to_uint8`,
    including re-applying the normalisation. Without it the legacy heuristic on
    ``ref_batch`` is used, which cannot express the normalised convention.
    """
    t = torch.as_tensor(np_images, dtype=ref_batch.dtype, device=ref_batch.device)
    t = t.permute(0, 3, 1, 2)

    if scale is None:
        if ref_batch.max() <= _UNIT_MAX:
            t = t / 255.0
        return t

    if scale.kind == "byte":
        return t
    t = t / 255.0
    if scale.kind == "unit":
        return t

    mean_np, std_np = _denorm_arrays(scale, t.shape[1])
    mean = torch.as_tensor(mean_np, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std_np, dtype=t.dtype, device=t.device).view(1, -1, 1, 1)
    return (t - mean) / std


def det_uint8_batch_to_float01(np_images: np.ndarray, *, ref_batch: Tensor) -> Tensor:
    """Uint8 HWC batch -> BCHW float32 in [0, 1] (detection / YOLO contract)."""
    t = torch.as_tensor(np_images, dtype=torch.uint8, device=ref_batch.device)
    t = t.permute(0, 3, 1, 2).to(dtype=torch.float32)
    return (t / 255.0).clamp(0.0, 1.0)


def tensor_batch_to_preview_uint8(
    images: Tensor,
    denorm_mean: list[float] | None = None,
    denorm_std: list[float] | None = None,
) -> np.ndarray:
    """Convert tensor batch to preview-quality uint8, with optional denormalisation."""
    np_images: np.ndarray = images.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    out: np.ndarray = np.zeros_like(np_images, dtype=np.uint8)

    for idx in range(np_images.shape[0]):
        sample = np_images[idx]
        converted: np.ndarray | None = None

        if float(sample.min()) >= 0.0 and float(sample.max()) <= 1.0:
            converted = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
        elif float(sample.min()) >= 0.0 and float(sample.max()) <= 255.0:
            converted = np.clip(sample, 0, 255).astype(np.uint8)
        else:
            if denorm_mean is not None and denorm_std is not None:
                mean = np.array(denorm_mean, dtype=np.float32)
                std = np.array(denorm_std, dtype=np.float32)
                denorm = sample * std + mean
                denorm = np.clip(denorm, 0.0, 1.0)
                if float(denorm.max() - denorm.min()) > 1e-6:
                    converted = (denorm * 255.0).astype(np.uint8)

        if converted is None:
            min_val = float(sample.min())
            max_val = float(sample.max())
            if max_val - min_val < 1e-8:
                converted = np.zeros_like(sample, dtype=np.uint8)
            else:
                normed = (sample - min_val) / (max_val - min_val)
                converted = np.clip(normed * 255.0, 0, 255).astype(np.uint8)

        out[idx] = converted
    return out  # type: ignore[no-any-return]


def resize_saliency_batch(maps: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (B, H, W) saliency batch to *target_h* x *target_w*."""
    if maps.ndim != 3:
        return maps
    if maps.shape[1] == target_h and maps.shape[2] == target_w:
        return maps
    resized: list[np.ndarray] = []
    for sal in maps:
        resized_map = cv2.resize(sal.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_map.astype(np.float32))
    return np.stack(resized, axis=0)  # type: ignore[no-any-return]
