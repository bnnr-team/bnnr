"""Tests for XAI cache read/write behavior."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bnnr.xai_cache import XAICache


def test_xai_cache_save_and_get(temp_dir) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    image = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    h = cache._hash_image(image)
    path = cache._cache_path(h, 1)
    np.save(path, np.ones((32, 32), dtype=np.float32))
    out = cache.get_importance_map(image, 1)
    assert out is not None
    assert out.shape == (32, 32)


def test_xai_cache_get_by_sample_index(temp_dir) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    index_path = cache._index_cache_path(7, 2)
    np.save(index_path, np.ones((16, 16), dtype=np.float32))

    image = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    out = cache.get_importance_map(image, 2, sample_index=7)
    assert out is not None
    assert out.shape == (16, 16)


def test_precompute_cache_skips_when_existing_entries_cover_target(temp_dir, dummy_model, monkeypatch) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    for idx in range(3):
        np.save(cache.cache_dir / f"existing_{idx}.npy", np.zeros((4, 4), dtype=np.float32))

    x = torch.rand(3, 3, 8, 8)
    y = torch.randint(0, 2, (3,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    called = {"value": False}

    def _fake_generate(*_args, **_kwargs):
        called["value"] = True
        return np.zeros((3, 8, 8), dtype=np.float32)

    monkeypatch.setattr("bnnr.xai_cache.generate_saliency_maps", _fake_generate)
    written = cache.precompute_cache(
        model=dummy_model,
        train_loader=loader,
        target_layers=[dummy_model.conv1],
        n_samples=3,
        method="opticam",
        force_recompute=False,
        show_progress=False,
    )

    assert written == 0
    assert called["value"] is False
