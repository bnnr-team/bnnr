"""Tests for XAI cache read/write behavior."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bnnr.xai_cache import (
    _MANIFEST_SCHEMA,
    XAICache,
    _model_state_hash,
    _safe_dataset_size,
    _safe_image_shape,
)


def _write_matching_manifest(
    cache: XAICache, loader: DataLoader, method: str, model: torch.nn.Module
) -> None:
    """Mark the cache as produced by *method*/*model* on *loader* so it is reusable."""
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "method": method,
        "dataset_size": _safe_dataset_size(loader),
        "image_shape": _safe_image_shape(loader),
        "model_hash": _model_state_hash(model),
    }
    cache._manifest_path().write_text(json.dumps(manifest), encoding="utf-8")


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


def test_save_map_index_keyed_persists(temp_dir) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    cache.save_map(np.ones((8, 8), dtype=np.float32), label=1, sample_index=3)
    assert cache._index_cache_path(3, 1).exists()


def test_save_map_hash_only_does_not_persist(temp_dir) -> None:
    """Without a sample_index, save_map must not write (no unbounded growth)."""
    cache = XAICache(temp_dir / "xai_cache")
    before = list(cache.cache_dir.glob("*.npy"))
    # Different images, all hash-only — must add zero files.
    for _ in range(5):
        image = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
        cache.save_map(np.ones((8, 8), dtype=np.float32), label=0, image=image)
    after = list(cache.cache_dir.glob("*.npy"))
    assert len(after) == len(before) == 0


def test_trim_to_max_mb_evicts_oldest(temp_dir) -> None:
    import os

    cache = XAICache(temp_dir / "xai_cache")
    # 8 maps of ~1 MB each (512x512 float32 = 1 MiB), mtimes 1000..1007.
    arr = np.ones((512, 512), dtype=np.float32)
    paths = []
    for i in range(8):
        p = cache._index_cache_path(i, 0)
        np.save(p, arr)
        os.utime(p, (1000 + i, 1000 + i))  # oldest = i0
        paths.append(p)

    evicted = cache.trim_to_max_mb(3)  # keep ~3 newest
    assert evicted >= 4
    total_mb = sum(p.stat().st_size for p in cache.cache_dir.glob("*.npy")) / (1024 * 1024)
    assert total_mb <= 3
    # The newest map (index 7) must survive; the oldest (index 0) must be gone.
    assert cache._index_cache_path(7, 0).exists()
    assert not cache._index_cache_path(0, 0).exists()


def test_trim_to_max_mb_disabled_when_zero(temp_dir) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    np.save(cache._index_cache_path(0, 0), np.ones((256, 256), dtype=np.float32))
    assert cache.trim_to_max_mb(0) == 0
    assert cache._index_cache_path(0, 0).exists()


def test_precompute_cache_skips_when_existing_entries_cover_target(temp_dir, dummy_model, monkeypatch) -> None:
    cache = XAICache(temp_dir / "xai_cache")

    x = torch.rand(3, 3, 8, 8)
    y = torch.randint(0, 2, (3,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    # A matching manifest marks the existing entries as same-provenance, so the
    # count-based fast-skip is allowed to reuse them.
    _write_matching_manifest(cache, loader, "opticam", dummy_model)
    for idx in range(3):
        np.save(cache.cache_dir / f"existing_{idx}.npy", np.zeros((4, 4), dtype=np.float32))

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


def test_precompute_writes_manifest(temp_dir, dummy_model, monkeypatch) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    x = torch.rand(2, 3, 8, 8)
    y = torch.randint(0, 2, (2,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    monkeypatch.setattr(
        "bnnr.xai_cache.generate_saliency_maps",
        lambda *_a, **_k: np.zeros((2, 8, 8), dtype=np.float32),
    )
    cache.precompute_cache(
        model=dummy_model,
        train_loader=loader,
        target_layers=[dummy_model.conv1],
        n_samples=2,
        method="opticam",
    )

    manifest = json.loads(cache._manifest_path().read_text(encoding="utf-8"))
    assert manifest["schema"] == _MANIFEST_SCHEMA
    assert manifest["method"] == "opticam"
    assert manifest["dataset_size"] == 2
    assert manifest["image_shape"] == [3, 8, 8]
    assert manifest["model_hash"] == _model_state_hash(dummy_model)


def test_precompute_invalidates_on_model_change(temp_dir, dummy_model, monkeypatch) -> None:
    """A different model sharing the cache dir must invalidate stale maps."""
    cache = XAICache(temp_dir / "xai_cache")
    x = torch.rand(3, 3, 8, 8)
    y = torch.randint(0, 2, (3,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    # Matching manifest + maps left by the model at its current weights.
    _write_matching_manifest(cache, loader, "opticam", dummy_model)
    for idx in range(3):
        np.save(cache.cache_dir / f"idx_{idx}_0.npy", np.zeros((4, 4), dtype=np.float32))

    # Same method/dataset/shape, but the weights changed -> stale maps.
    with torch.no_grad():
        next(dummy_model.parameters()).add_(1.0)

    called = {"value": False}

    def _fake_generate(*_args, **_kwargs):
        called["value"] = True
        return np.zeros((3, 8, 8), dtype=np.float32)

    monkeypatch.setattr("bnnr.xai_cache.generate_saliency_maps", _fake_generate)
    cache.precompute_cache(
        model=dummy_model,
        train_loader=loader,
        target_layers=[dummy_model.conv1],
        n_samples=3,
        method="opticam",
    )

    assert called["value"] is True  # recomputed instead of reusing stale maps
    manifest = json.loads(cache._manifest_path().read_text(encoding="utf-8"))
    assert manifest["model_hash"] == _model_state_hash(dummy_model)


def test_precompute_invalidates_on_manifest_mismatch(temp_dir, dummy_model, monkeypatch) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    x = torch.rand(3, 3, 8, 8)
    y = torch.randint(0, 2, (3,))
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    # Stale maps left by a run that used a different XAI method.
    _write_matching_manifest(cache, loader, "craft", dummy_model)
    for idx in range(3):
        np.save(cache.cache_dir / f"idx_{idx}_0.npy", np.zeros((4, 4), dtype=np.float32))

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
    )

    # Method changed (craft -> opticam): stale maps were dropped and recomputed.
    assert called["value"] is True
    assert written == 3
    assert json.loads(cache._manifest_path().read_text(encoding="utf-8"))["method"] == "opticam"


def test_precompute_reuses_on_manifest_match(temp_dir, dummy_model, monkeypatch) -> None:
    cache = XAICache(temp_dir / "xai_cache")
    x = torch.rand(3, 3, 8, 8)
    y = torch.zeros(3, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=3)

    monkeypatch.setattr(
        "bnnr.xai_cache.generate_saliency_maps",
        lambda *_a, **_k: np.zeros((3, 8, 8), dtype=np.float32),
    )
    first = cache.precompute_cache(
        model=dummy_model, train_loader=loader, target_layers=[dummy_model.conv1],
        n_samples=3, method="opticam",
    )
    assert first == 3

    called = {"value": False}

    def _fake_generate(*_args, **_kwargs):
        called["value"] = True
        return np.zeros((3, 8, 8), dtype=np.float32)

    monkeypatch.setattr("bnnr.xai_cache.generate_saliency_maps", _fake_generate)
    second = cache.precompute_cache(
        model=dummy_model, train_loader=loader, target_layers=[dummy_model.conv1],
        n_samples=3, method="opticam",
    )
    # Same provenance: nothing recomputed.
    assert second == 0
    assert called["value"] is False
