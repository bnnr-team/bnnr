"""Disk-backed saliency cache utilities for ICD and XAI workflows."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bnnr.xai import BaseExplainer, generate_saliency_maps

logger = logging.getLogger(__name__)

# Prefer xxhash (much faster) if available, else fall back to sha256.
# Benchmarks show sha256 is faster than md5 on modern CPUs with
# hardware SHA extensions, so we keep sha256 as default fallback.
try:
    import xxhash as _xxhash

    def _fast_hash(data: bytes) -> str:
        return _xxhash.xxh64(data).hexdigest()
except ImportError:
    def _fast_hash(data: bytes) -> str:  # type: ignore[misc]
        return hashlib.sha256(data).hexdigest()


class XAICache:
    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_image(self, image: np.ndarray) -> str:
        return _fast_hash(image.tobytes())

    def _cache_path(self, image_hash: str, label: int) -> Path:
        return self.cache_dir / f"{image_hash}_{label}.npy"

    def _index_cache_path(self, sample_index: int, label: int) -> Path:
        return self.cache_dir / f"idx_{sample_index}_{label}.npy"

    # ------------------------------------------------------------------
    # Lazy-caching: persist a single saliency map computed on-the-fly
    # ------------------------------------------------------------------

    def save_map(
        self,
        saliency: np.ndarray,
        label: int,
        sample_index: int | None = None,
        image: np.ndarray | None = None,
    ) -> None:
        """Persist a saliency map so future lookups are instant (lazy caching).

        At least one of *sample_index* or *image* must be provided so we can
        derive a cache key.
        """
        if sample_index is not None:
            path = self._index_cache_path(sample_index, label)
        elif image is not None:
            image_hash = self._hash_image(image)
            path = self._cache_path(image_hash, label)
        else:
            return  # nothing to key on — silently skip
        np.save(path, saliency.astype(np.float32))

    # ------------------------------------------------------------------
    # Batch-level helpers for precompute fast-skip
    # ------------------------------------------------------------------

    def _batch_all_cached(
        self,
        labels: torch.Tensor,
        sample_indices: torch.Tensor | None,
        count: int,
    ) -> bool:
        """Return ``True`` if every item in the batch already has a cache file.

        Only works reliably when *sample_indices* are available (index-based
        caching).  Falls back to ``False`` for hash-based caching because we
        would need the actual image bytes to compute the hash.
        """
        if sample_indices is None:
            return False
        for idx in range(count):
            lab = labels[idx]
            label = int(lab.item()) if lab.ndim == 0 else int(lab.argmax().item())
            path = self._index_cache_path(int(sample_indices[idx].item()), label)
            if not path.exists():
                return False
        return True

    # ------------------------------------------------------------------
    # Precompute
    # ------------------------------------------------------------------

    def precompute_cache(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        target_layers: list[nn.Module],
        explainer: BaseExplainer | None = None,
        n_samples: int = 0,
        method: str = "opticam",
        force_recompute: bool = False,
        show_progress: bool = False,
    ) -> int:
        """Pre-compute saliency maps and persist them to disk.

        Parameters
        ----------
        n_samples:
            Number of samples to cache.  When ``<= 0`` the **entire** dataset
            exposed by *train_loader* is cached (i.e. iterate until the loader
            is exhausted).
        """
        model.eval()
        written = 0
        processed = 0
        skipped_batches = 0

        cache_all = n_samples <= 0

        if not force_recompute and not cache_all:
            existing = len(list(self.cache_dir.glob("*.npy")))
            processed = min(existing, n_samples)
            if processed >= n_samples:
                return 0

        # For "cache all" we don't know total upfront — use None for tqdm
        total = None if cache_all else n_samples
        progress = tqdm(
            total=total,
            initial=processed,
            desc="Precomputing XAI cache",
            leave=False,
            disable=not show_progress,
        )

        try:
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, sample_indices = batch
                else:
                    images, labels = batch
                    sample_indices = None

                batch_size = images.shape[0]

                # ── Fast-skip: skip entire batch if all cache files exist ──
                if not force_recompute and self._batch_all_cached(
                    labels, sample_indices, batch_size
                ):
                    skipped_batches += 1
                    processed += batch_size
                    progress.update(batch_size)
                    if not cache_all and processed >= n_samples:
                        break
                    continue

                with torch.no_grad():
                    images = images.to(next(model.parameters()).device)
                    labels = labels.to(images.device)

                if explainer is None:
                    maps = generate_saliency_maps(model, images, labels, target_layers, method=method)
                else:
                    maps = explainer.explain(model, images, labels, target_layers)
                for idx in range(batch_size):
                    img_np = images[idx].detach().cpu().permute(1, 2, 0).numpy()
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255.0).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)

                    lab = labels[idx]
                    label = int(lab.item()) if lab.ndim == 0 else int(lab.argmax().item())
                    if sample_indices is not None:
                        sample_index = int(sample_indices[idx])
                        path = self._index_cache_path(sample_index, label)
                    else:
                        image_hash = self._hash_image(img_np)
                        path = self._cache_path(image_hash, label)

                    if force_recompute or not path.exists():
                        np.save(path, maps[idx].astype(np.float32))
                        written += 1
                    processed += 1
                    progress.update(1)
                    if not cache_all and processed >= n_samples:
                        break
            if skipped_batches > 0:
                logger.info(
                    "XAI cache: skipped %d already-cached batches (no recomputation)",
                    skipped_batches,
                )
            return written
        finally:
            progress.close()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_importance_map(self, image: np.ndarray, label: int, sample_index: int | None = None) -> np.ndarray | None:
        if sample_index is not None:
            indexed = self._index_cache_path(sample_index, label)
            if indexed.exists():
                return np.load(indexed)

        image_hash = self._hash_image(image)
        path = self._cache_path(image_hash, label)
        if not path.exists():
            return None
        return np.load(path)
