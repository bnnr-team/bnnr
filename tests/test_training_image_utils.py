"""Unit tests for training.image_utils."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bnnr.training.image_utils import (
    BatchScale,
    NormalisedInputError,
    det_uint8_batch_to_float01,
    detect_batch_scale,
    resize_saliency_batch,
    tensor_batch_to_preview_uint8,
    tensor_to_uint8,
    uint8_to_tensor,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalise(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _denormalise(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return images * std + mean


def test_tensor_to_uint8_from_01() -> None:
    images = torch.rand(2, 3, 8, 8)
    out = tensor_to_uint8(images)
    assert out.dtype == np.uint8
    assert out.shape == (2, 8, 8, 3)


def test_uint8_to_tensor_roundtrip() -> None:
    ref = torch.rand(2, 3, 4, 4)
    np_u8 = (ref.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    back = uint8_to_tensor(np_u8, ref_batch=ref)
    assert back.shape == ref.shape
    assert back.max() <= 1.05


def test_det_uint8_batch_to_float01() -> None:
    ref = torch.zeros(1, 3, 4, 4)
    np_u8 = np.ones((1, 4, 4, 3), dtype=np.uint8) * 255
    out = det_uint8_batch_to_float01(np_u8, ref_batch=ref)
    assert out.shape == (1, 3, 4, 4)
    assert float(out.max()) <= 1.0


def test_resize_saliency_batch() -> None:
    maps = np.ones((2, 4, 4), dtype=np.float32) * 0.5
    resized = resize_saliency_batch(maps, 8, 8)
    assert resized.shape == (2, 8, 8)


def test_tensor_batch_to_preview_uint8() -> None:
    images = torch.rand(1, 3, 16, 16)
    out = tensor_batch_to_preview_uint8(images)
    assert out.dtype == np.uint8
    assert out.shape == (1, 16, 16, 3)


# ---------------------------------------------------------------------------
# Batch-convention detection (FIX-0-1)
# ---------------------------------------------------------------------------


def test_detect_unit_batch() -> None:
    assert detect_batch_scale(torch.rand(2, 3, 8, 8)).kind == "unit"


def test_detect_byte_batch() -> None:
    images = torch.rand(2, 3, 8, 8) * 255.0
    assert detect_batch_scale(images).kind == "byte"


def test_normalised_batch_raises_without_stats() -> None:
    normalised = _normalise(torch.rand(2, 3, 8, 8))
    with pytest.raises(NormalisedInputError) as excinfo:
        detect_batch_scale(normalised)
    message = str(excinfo.value)
    # The error has to name both ways out, not just one.
    assert "Normalize()" in message
    assert "denormalization_mean" in message


def test_normalised_batch_detected_with_stats() -> None:
    normalised = _normalise(torch.rand(2, 3, 8, 8))
    scale = detect_batch_scale(
        normalised, denorm_mean=IMAGENET_MEAN, denorm_std=IMAGENET_STD
    )
    assert scale.kind == "normalised"
    assert scale.mean == tuple(IMAGENET_MEAN)


def test_configured_stats_do_not_hijack_a_unit_batch() -> None:
    """Stats set for reporting must not change how an in-range batch is treated."""
    scale = detect_batch_scale(
        torch.rand(2, 3, 8, 8), denorm_mean=IMAGENET_MEAN, denorm_std=IMAGENET_STD
    )
    assert scale.kind == "unit"


def test_normalised_roundtrip_is_lossless_to_one_step() -> None:
    original = torch.rand(2, 3, 8, 8)
    normalised = _normalise(original)
    scale = detect_batch_scale(
        normalised, denorm_mean=IMAGENET_MEAN, denorm_std=IMAGENET_STD
    )

    as_uint8 = tensor_to_uint8(normalised, scale=scale)
    assert as_uint8.dtype == np.uint8
    # A destroyed batch collapses onto 0/255; a denormalised one keeps its spread.
    assert 10 < int(as_uint8.mean()) < 245

    back = uint8_to_tensor(as_uint8, ref_batch=normalised, scale=scale)
    assert back.shape == normalised.shape
    assert torch.allclose(_denormalise(back), original, atol=1.0 / 255.0)


def test_dark_byte_batch_is_not_destroyed() -> None:
    """A [0, 255] batch whose peak is below 200 used to be clipped into [0, 1]."""
    images = torch.rand(1, 3, 8, 8) * 150.0
    out = tensor_to_uint8(images)
    assert int(out.max()) > 100


def test_unit_batch_overshoot_does_not_wrap_around() -> None:
    """1.04 * 255 overflows uint8; it must clamp to white, not wrap to black."""
    images = torch.full((1, 3, 4, 4), 1.04)
    out = tensor_to_uint8(images)
    assert int(out.min()) == 255


def test_unit_batch_undershoot_does_not_wrap_around() -> None:
    images = torch.full((1, 3, 4, 4), -0.005)
    out = tensor_to_uint8(images)
    assert int(out.max()) == 0


def test_uint8_to_tensor_without_scale_keeps_legacy_heuristic() -> None:
    ref = torch.full((1, 3, 4, 4), 200.0)
    u8 = np.full((1, 4, 4, 3), 255, dtype=np.uint8)
    assert float(uint8_to_tensor(u8, ref_batch=ref).max()) > 1.5


def test_denorm_stats_channel_mismatch_raises() -> None:
    images = _normalise(torch.rand(1, 3, 4, 4))
    scale = BatchScale("normalised", (0.5, 0.5), (0.5, 0.5))
    with pytest.raises(NormalisedInputError):
        tensor_to_uint8(images, scale=scale)
