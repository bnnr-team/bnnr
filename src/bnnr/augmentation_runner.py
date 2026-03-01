"""Augmentation dispatch engine with optional async prefetch.

Provides a single entry point for applying augmentations to batches,
automatically dispatching GPU-native augmentations on-device and
optionally prefetching CPU-bound augmentations in a background thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Iterator
from queue import Empty, Queue
from typing import Any

import numpy as np
import torch
from torch import Tensor

from bnnr.augmentations import BaseAugmentation

logger = logging.getLogger("bnnr.augmentation_runner")


class AugmentationRunner:
    """Dispatch augmentations to GPU or CPU paths with optional async prefetch.

    When ``async_prefetch=True`` and there are CPU-bound augmentations, the
    runner overlaps CPU augmentation of batch N+1 with GPU training of batch N.

    Can be used in two ways:

    1. **Sync dispatch**: call ``apply_batch(images, labels)`` directly.
    2. **Async iterator**: wrap a DataLoader and iterate::

           runner = AugmentationRunner(augs, async_prefetch=True)
           for images, labels in runner.iter_loader(train_loader):
               ...

    Parameters
    ----------
    augmentations : list[BaseAugmentation]
        Augmentations to apply.
    async_prefetch : bool
        If True, CPU-bound augmentations are applied in a background thread.
    prefetch_queue_size : int
        Max number of prefetched batches to keep in memory.
    """

    def __init__(
        self,
        augmentations: list[BaseAugmentation],
        async_prefetch: bool = True,
        prefetch_queue_size: int = 2,
    ) -> None:
        self.augmentations = augmentations
        self.async_prefetch = async_prefetch
        self.prefetch_queue_size = prefetch_queue_size

        # Split augmentations into GPU-native and CPU-bound
        self.gpu_augmentations = [a for a in augmentations if a.device_compatible]
        self.cpu_augmentations = [a for a in augmentations if not a.device_compatible]

        self._prefetch_queue: Queue[tuple[Tensor, Tensor] | None] = Queue(
            maxsize=prefetch_queue_size
        )
        self._prefetch_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._worker_exception: BaseException | None = None

    def apply_batch(
        self,
        images: Tensor,
        labels: Tensor,
        sample_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply all augmentations to a batch (synchronous).

        GPU-native augmentations are applied directly on the tensor.
        CPU-bound augmentations go through the numpy fallback path.
        """
        images = self._apply_gpu_augmentations(images, labels, sample_indices)
        images = self._apply_cpu_augmentations(images, labels, sample_indices)
        return images, labels

    def _apply_augmentation_list(
        self,
        augmentations: list[BaseAugmentation],
        images: Tensor,
        labels: Tensor,
        sample_indices: Tensor | None = None,
    ) -> Tensor:
        """Apply a list of augmentations to a batch of images.

        For each augmentation, tries (in order):
        1. ``apply_batch_with_labels`` (label-aware path, e.g. ICD/AICD)
        2. ``apply_tensor`` (GPU-native tensor path)
        3. ``apply_batch`` (numpy uint8 fallback)
        """
        for aug in augmentations:
            if hasattr(aug, "apply_batch_with_labels"):
                np_images = _tensor_to_uint8(images)
                np_labels = labels.detach().cpu().numpy()
                np_indices = (
                    sample_indices.detach().cpu().numpy()
                    if sample_indices is not None
                    else None
                )
                aug_images = aug.apply_batch_with_labels(
                    np_images, np_labels, sample_indices=np_indices
                )
                images = _uint8_to_tensor(aug_images, ref_batch=images)
            else:
                try:
                    images = aug.apply_tensor(images)
                except NotImplementedError:
                    np_images = _tensor_to_uint8(images)
                    aug_images = aug.apply_batch(np_images)
                    images = _uint8_to_tensor(aug_images, ref_batch=images)
        return images

    def _apply_gpu_augmentations(
        self,
        images: Tensor,
        labels: Tensor,
        sample_indices: Tensor | None = None,
    ) -> Tensor:
        """Apply GPU-native augmentations directly on tensors."""
        return self._apply_augmentation_list(
            self.gpu_augmentations, images, labels, sample_indices
        )

    def _apply_cpu_augmentations(
        self,
        images: Tensor,
        labels: Tensor,
        sample_indices: Tensor | None = None,
    ) -> Tensor:
        """Apply CPU-bound augmentations via numpy fallback."""
        return self._apply_augmentation_list(
            self.cpu_augmentations, images, labels, sample_indices
        )

    # ------------------------------------------------------------------
    # Async prefetch iterator
    # ------------------------------------------------------------------

    def iter_loader(
        self,
        data_loader: Iterable[Any],
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over a DataLoader, applying augmentations with optional async prefetch.

        When ``async_prefetch=True`` and there are CPU-bound augmentations,
        batch N+1 is augmented in a background thread while batch N trains.

        Yields
        ------
        tuple[Tensor, Tensor]
            (augmented_images, labels)
        """
        if not self.async_prefetch or not self.cpu_augmentations:
            # Sync path: just apply augmentations inline
            for raw_batch in data_loader:
                images, labels, sample_indices = _unpack_batch(raw_batch)
                images, labels = self.apply_batch(images, labels, sample_indices)
                yield images, labels
            return

        # Async path: CPU augmentations run in background thread
        self._stop_event.clear()
        self._worker_exception = None

        # Clear any stale items from previous runs
        while not self._prefetch_queue.empty():
            try:
                self._prefetch_queue.get_nowait()
            except Empty:
                break

        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(data_loader,),
            daemon=True,
        )
        self._prefetch_thread.start()

        try:
            while True:
                if self._worker_exception is not None:
                    raise self._worker_exception  # type: ignore[misc]

                batch = self._prefetch_queue.get()
                if batch is None:
                    # Worker is done
                    if self._worker_exception is not None:
                        raise self._worker_exception  # type: ignore[misc]
                    break

                images, labels = batch
                # GPU augmentations are applied here (main thread, on-device)
                images = self._apply_gpu_augmentations(images, labels)
                yield images, labels
        finally:
            self._stop_event.set()
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                self._prefetch_thread.join(timeout=5)

    def _prefetch_worker(self, data_loader: Iterable[Any]) -> None:
        """Background thread: apply CPU augmentations and enqueue results."""
        try:
            for raw_batch in data_loader:
                if self._stop_event.is_set():
                    break
                images, labels, sample_indices = _unpack_batch(raw_batch)
                # Apply CPU augmentations in this background thread
                images = self._apply_cpu_augmentations(images, labels, sample_indices)
                self._prefetch_queue.put((images, labels))
        except Exception as exc:
            self._worker_exception = exc
            logger.exception("AugmentationRunner prefetch worker failed")
        finally:
            self._prefetch_queue.put(None)  # sentinel


def _unpack_batch(
    raw_batch: Any,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Unpack a DataLoader batch into (images, labels, sample_indices)."""
    if isinstance(raw_batch, (list, tuple)) and len(raw_batch) == 3:
        return raw_batch[0], raw_batch[1], raw_batch[2]
    if isinstance(raw_batch, (list, tuple)) and len(raw_batch) == 2:
        return raw_batch[0], raw_batch[1], None
    raise ValueError(f"Unexpected batch format: {type(raw_batch)}")


def _tensor_to_uint8(images: Tensor) -> np.ndarray:
    """Convert a (B, C, H, W) float tensor to a (B, H, W, C) uint8 array."""
    np_images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    lo, hi = float(np_images.min()), float(np_images.max())

    if lo < -0.01 or (hi > 1.05 and hi < 200):
        np_images = np.clip(np_images, 0.0, 1.0)

    if hi <= 1.05:
        return (np_images * 255.0).astype("uint8")  # type: ignore[no-any-return]
    return np_images.astype("uint8")  # type: ignore[no-any-return]


def _uint8_to_tensor(np_images: np.ndarray, *, ref_batch: Tensor) -> Tensor:
    """Convert (B, H, W, C) uint8 back to (B, C, H, W) float tensor."""
    t = torch.as_tensor(np_images, dtype=ref_batch.dtype, device=ref_batch.device)
    t = t.permute(0, 3, 1, 2)
    if ref_batch.max() <= 1.05:
        t = t / 255.0
    return t


__all__ = ["AugmentationRunner"]
