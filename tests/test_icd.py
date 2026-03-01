"""Tests for ICD and AICD augmentation behavior."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from bnnr.icd import AICD, ICD
from bnnr.xai_cache import XAICache


def _dummy_model_and_layers(dummy_model):
    return dummy_model, [dummy_model.conv1]


# ---------------------------------------------------------------------------
# Cache-hit path (tile-based masking with real cached maps)
# ---------------------------------------------------------------------------


def test_icd_apply_with_label_uses_cache(dummy_model, temp_dir) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    cache = XAICache(temp_dir / "cache")
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    h = cache._hash_image(image)
    np.save(cache._cache_path(h, 1), np.random.rand(32, 32).astype(np.float32))

    aug = ICD(model=model, target_layers=layers, cache=cache, random_state=42)
    out = aug.apply_with_label(image, 1)
    assert out.shape == image.shape


def test_aicd_apply_with_label(dummy_model) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        aug = AICD(model=model, target_layers=layers, cache=None, random_state=42)
    aug.set_label(1)
    assert aug._current_label == 1


# ---------------------------------------------------------------------------
# Grayscale + resized map
# ---------------------------------------------------------------------------


def test_icd_apply_batch_with_labels_grayscale_and_resized_map(dummy_model, temp_dir) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    cache = XAICache(temp_dir / "cache_idx")
    # Simulate cached CRAFT-like lower-resolution map for indexed sample.
    np.save(cache._index_cache_path(7, 1), np.random.rand(14, 14).astype(np.float32))

    aug = ICD(model=model, target_layers=layers, cache=cache, random_state=42, probability=1.0)
    images = (np.random.rand(1, 28, 28, 1) * 255).astype(np.uint8)
    labels = np.array([1], dtype=np.int64)
    sample_indices = np.array([7], dtype=np.int64)

    out = aug.apply_batch_with_labels(images, labels, sample_indices=sample_indices)
    assert out.shape == images.shape


# ---------------------------------------------------------------------------
# Fill strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["gaussian_blur", "local_mean", "global_mean", "noise", "solid"])
def test_fill_strategies(dummy_model, temp_dir, strategy) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    cache = XAICache(temp_dir / f"cache_{strategy}")
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    h = cache._hash_image(image)
    np.save(cache._cache_path(h, 0), np.random.rand(32, 32).astype(np.float32))

    aug = ICD(
        model=model, target_layers=layers, cache=cache,
        fill_strategy=strategy, random_state=42, probability=1.0,
    )
    out = aug.apply_with_label(image, 0)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_invalid_fill_strategy(dummy_model) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    with pytest.raises(ValueError, match="fill_strategy must be one of"):
        ICD(model=model, target_layers=layers, fill_strategy="invalid")


# ---------------------------------------------------------------------------
# Tile size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tile_size", [4, 8, 16])
def test_tile_sizes(dummy_model, temp_dir, tile_size) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    cache = XAICache(temp_dir / f"cache_ts{tile_size}")
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    h = cache._hash_image(image)
    np.save(cache._cache_path(h, 0), np.random.rand(32, 32).astype(np.float32))

    aug = ICD(
        model=model, target_layers=layers, cache=cache,
        tile_size=tile_size, random_state=42, probability=1.0,
    )
    out = aug.apply_with_label(image, 0)
    assert out.shape == image.shape


def test_invalid_tile_size(dummy_model) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    with pytest.raises(ValueError, match="tile_size must be >= 1"):
        ICD(model=model, target_layers=layers, tile_size=0)


# ---------------------------------------------------------------------------
# Backward compat: mask_value deprecation
# ---------------------------------------------------------------------------


def test_mask_value_deprecation(dummy_model, temp_dir) -> None:
    model, layers = _dummy_model_and_layers(dummy_model)
    cache = XAICache(temp_dir / "cache_dep")
    with pytest.warns(DeprecationWarning, match="mask_value is deprecated"):
        aug = ICD(model=model, target_layers=layers, cache=cache, mask_value=128)
    assert aug.fill_strategy == "solid"
    assert aug.fill_value == 128


# ---------------------------------------------------------------------------
# Cache miss → online computation (no random rectangle)
# ---------------------------------------------------------------------------


def test_cache_miss_computes_online(dummy_model, temp_dir) -> None:
    """When cache exists but misses, ICD should compute saliency online,
    not fall back to a random rectangle."""
    model, layers = _dummy_model_and_layers(dummy_model)
    cache = XAICache(temp_dir / "cache_miss")
    # Empty cache — every image will miss
    aug = ICD(
        model=model, target_layers=layers, cache=cache,
        probability=1.0, random_state=42,
    )
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    out = aug.apply_with_label(image, 0)
    assert out.shape == image.shape
    # Verify the miss counter was incremented
    assert aug._cache_miss_count >= 1
