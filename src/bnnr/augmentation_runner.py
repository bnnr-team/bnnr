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

from torch import Tensor

from bnnr.augmentations import BaseAugmentation
from bnnr.image_scale import BatchScale, detect_batch_scale, from_unit, to_unit
from bnnr.training.image_utils import tensor_to_uint8, uint8_to_tensor

# The convention every tensor-path augmentation implementation expects.
_UNIT_SCALE = BatchScale("unit")

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
    denorm_mean, denorm_std : list[float] | None
        Per-channel normalisation statistics of the incoming batches. Supply
        them when the DataLoader applies ``transforms.Normalize()``: the runner
        undoes the normalisation before augmenting and redoes it afterwards.
        Without them a normalised batch raises instead of being corrupted.
    """

    def __init__(
        self,
        augmentations: list[BaseAugmentation],
        async_prefetch: bool = True,
        prefetch_queue_size: int = 2,
        denorm_mean: list[float] | None = None,
        denorm_std: list[float] | None = None,
    ) -> None:
        self.augmentations = augmentations
        self.async_prefetch = async_prefetch
        self.prefetch_queue_size = prefetch_queue_size
        self.denorm_mean = denorm_mean
        self.denorm_std = denorm_std

        # Split augmentations into GPU-native and CPU-bound
        self.gpu_augmentations = [a for a in augmentations if a.device_compatible]
        self.cpu_augmentations = [a for a in augmentations if not a.device_compatible]

        # The async split (CPU augs in a worker thread, GPU augs on the main
        # thread) only preserves the user's order when every CPU aug precedes
        # every GPU aug in the list. When the list interleaves them, we fall
        # back to the in-order sync path so results never depend on the split.
        self._cpu_then_gpu = list(augmentations) == self.cpu_augmentations + self.gpu_augmentations

        self._prefetch_queue: Queue[tuple[Tensor, Tensor, Tensor | None] | None] = Queue(
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

        Augmentations run strictly in the user-provided list order; each is
        dispatched to its GPU-native tensor path or numpy fallback per call
        (see ``_apply_augmentation_list``). Order is preserved even for mixed
        CPU/GPU lists, unlike the async split.
        """
        images = self._apply_augmentation_list(
            self.augmentations, images, labels, sample_indices
        )
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
        scale: BatchScale | None = None
        for aug in augmentations:
            if hasattr(aug, "apply_batch_with_labels"):
                scale = scale or self._batch_scale(images)
                np_images = tensor_to_uint8(images, scale=scale)
                np_labels = labels.detach().cpu().numpy()
                np_indices = (
                    sample_indices.detach().cpu().numpy()
                    if sample_indices is not None
                    else None
                )
                aug_images = aug.apply_batch_with_labels(
                    np_images, np_labels, sample_indices=np_indices
                )
                images = uint8_to_tensor(aug_images, ref_batch=images, scale=scale)
            else:
                # Every tensor-path implementation, including third-party ones,
                # assumes [0, 1]. Hand it that and convert back, so a normalised
                # or [0, 255] batch is not truncated inside the augmentation.
                scale = scale or self._batch_scale(images)
                unit = to_unit(images, scale)
                try:
                    # No scale argument: the batch is already in the convention
                    # apply_tensor would detect, and third-party overrides that
                    # predate the keyword keep working unchanged.
                    unit = aug.apply_tensor(unit)
                except NotImplementedError:
                    np_images = tensor_to_uint8(unit, scale=_UNIT_SCALE)
                    aug_images = aug.apply_batch(np_images)
                    unit = uint8_to_tensor(aug_images, ref_batch=unit, scale=_UNIT_SCALE)
                images = from_unit(unit, scale)
        return images

    def _batch_scale(self, images: Tensor) -> BatchScale:
        """Detect the batch convention once, before any augmentation changes it."""
        return detect_batch_scale(
            images, denorm_mean=self.denorm_mean, denorm_std=self.denorm_std
        )

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
        if not self.async_prefetch or not self.cpu_augmentations or not self._cpu_then_gpu:
            # Sync path: apply augmentations inline in strict list order. Used
            # when async is disabled, there are no CPU augs to offload, or the
            # list interleaves CPU/GPU augs (where the split would reorder them).
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
                exc = self._worker_exception
                if exc is not None:
                    if isinstance(exc, BaseException):
                        raise exc
                    raise RuntimeError(
                        f"Prefetch worker failed with non-exception payload: {type(exc).__name__}"
                    )

                batch = self._prefetch_queue.get()
                if batch is None:
                    # Worker is done
                    exc = self._worker_exception
                    if exc is not None:
                        if isinstance(exc, BaseException):
                            raise exc
                        raise RuntimeError(
                            f"Prefetch worker failed with non-exception payload: {type(exc).__name__}"
                        )
                    break

                images, labels, sample_indices = batch
                # GPU augmentations are applied here (main thread, on-device).
                # sample_indices are threaded through so index-aware GPU augs
                # (e.g. cached ICD) key on the sample index, not an image hash.
                images = self._apply_gpu_augmentations(images, labels, sample_indices)
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
                self._prefetch_queue.put((images, labels, sample_indices))
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


__all__ = ["AugmentationRunner"]
