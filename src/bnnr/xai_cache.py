"""Disk-backed saliency cache utilities for ICD and XAI workflows."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bnnr.xai import BaseExplainer, generate_saliency_maps

logger = logging.getLogger(__name__)

# Manifest describing what produced the cached maps. A mismatch (different XAI
# method, dataset size, or image shape) invalidates the cache so stale saliency
# maps are never reused, e.g. across runs that share an explicit cache dir.
_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_SCHEMA = 2


def _safe_dataset_size(loader: DataLoader) -> int | None:
    """Best-effort dataset length; ``None`` when the loader has no sized dataset."""
    try:
        return int(len(loader.dataset))  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _safe_image_shape(loader: DataLoader) -> list[int] | None:
    """Best-effort image shape from the first sample; ``None`` if unavailable."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        sample = dataset[0]
    except Exception:
        return None
    img = sample[0] if isinstance(sample, (list, tuple)) else sample
    shape = getattr(img, "shape", None)
    if shape is None:
        return None
    return [int(x) for x in shape]


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


def _model_state_hash(model: nn.Module) -> str:
    """Cheap, deterministic fingerprint of a model's weights.

    Each parameter is sampled (up to 64 evenly spaced values plus a global sum)
    so the cost stays low for large models while still changing whenever the
    weights change. Used so the XAI cache invalidates when a *different* model
    writes into the same cache dir, not only on a different method/dataset/shape.
    """
    parts: list[bytes] = []
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name]
        parts.append(name.encode())
        if not hasattr(tensor, "detach"):
            parts.append(repr(tensor).encode())
            continue
        flat = tensor.detach().to(torch.float64).reshape(-1).cpu()
        n = int(flat.numel())
        parts.append(str(tuple(tensor.shape)).encode())
        if n == 0:
            continue
        idx = torch.linspace(0, n - 1, steps=min(n, 64)).long()
        parts.append(flat[idx].numpy().tobytes())
        parts.append(repr(float(flat.sum())).encode())
    return _fast_hash(b"".join(parts))


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
    # Manifest-based invalidation
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.cache_dir / _MANIFEST_FILENAME

    def _load_manifest(self) -> dict | None:
        path = self._manifest_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _ensure_cache_consistent(
        self,
        *,
        method: str,
        dataset_size: int | None,
        image_shape: list[int] | None,
        force_recompute: bool,
        model_hash: str | None = None,
    ) -> None:
        """Drop cached maps whose provenance differs from the current request.

        Writes a manifest of ``{method, dataset_size, image_shape, model_hash}``.
        When an existing manifest does not match (or ``force_recompute`` is set),
        all ``*.npy`` maps are removed so they are recomputed from the current
        model/method/dataset instead of silently reused. ``model_hash`` lets a
        different model sharing the same cache dir invalidate stale saliency even
        when method, dataset size and image shape are identical.
        """
        desired = {
            "schema": _MANIFEST_SCHEMA,
            "method": method,
            "dataset_size": dataset_size,
            "image_shape": image_shape,
            "model_hash": model_hash,
        }
        if not force_recompute and self._load_manifest() == desired:
            return
        for npy in self.cache_dir.glob("*.npy"):
            npy.unlink(missing_ok=True)
        self._manifest_path().write_text(json.dumps(desired), encoding="utf-8")

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

        Only *index-keyed* maps are persisted. Hash-keyed persistence was
        removed: lazy saves hit on a cache miss, which (outside precompute)
        means the image is already augmented, so a hash key never re-hits and
        the cache grows unboundedly. The *image* argument is accepted for
        backward compatibility but is no longer used for persistence.
        """
        if sample_index is None:
            return  # only index-keyed maps persist (no unbounded hash growth)
        path = self._index_cache_path(sample_index, label)
        np.save(path, saliency.astype(np.float32))

    def trim_to_max_mb(self, max_mb: int) -> int:
        """Evict oldest cache maps (LRU by mtime) until under *max_mb*.

        Returns the number of ``.npy`` files deleted. ``max_mb <= 0`` disables
        the cap (no-op). The manifest is never evicted.
        """
        if max_mb <= 0:
            return 0
        max_bytes = max_mb * 1024 * 1024
        npy_files = list(self.cache_dir.glob("*.npy"))
        sizes = {p: p.stat().st_size for p in npy_files}
        total = sum(sizes.values())
        if total <= max_bytes:
            return 0
        # Oldest first (smallest mtime). Delete until under the cap.
        npy_files.sort(key=lambda p: p.stat().st_mtime)
        evicted = 0
        for path in npy_files:
            if total <= max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            total -= sizes[path]
            evicted += 1
        return evicted

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

        # Invalidate stale maps (different method/dataset/image shape) before any
        # fast-skip so we never reuse another run's saliency.
        self._ensure_cache_consistent(
            method=method,
            dataset_size=_safe_dataset_size(train_loader),
            image_shape=_safe_image_shape(train_loader),
            force_recompute=force_recompute,
            model_hash=_model_state_hash(model),
        )

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
