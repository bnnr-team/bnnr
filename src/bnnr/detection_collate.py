"""Custom collate function for detection datasets.

Standard PyTorch DataLoader's ``default_collate`` cannot handle
variable-length targets (different number of boxes per image). This
module provides ``detection_collate_fn`` that stacks images into a
tensor but keeps targets as a list of dicts.
"""

from __future__ import annotations

import torch
from torch import Tensor


def detection_collate_fn(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Collate detection samples into a batch.

    Parameters
    ----------
    batch : list[tuple[Tensor, dict]]
        Each element is ``(image, target)`` where image is a ``Tensor[C, H, W]``
        and target is a dict with ``boxes`` and ``labels`` (and optionally
        ``image_id``, ``area``, ``iscrowd``).

    Returns
    -------
    images : Tensor[B, C, H, W]
        Stacked images.
    targets : list[dict[str, Tensor]]
        List of target dicts (one per image), unchanged.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def detection_collate_fn_with_index(
    batch: list[tuple[Tensor, dict[str, Tensor], int]],
) -> tuple[Tensor, list[dict[str, Tensor]], Tensor]:
    """Collate detection samples with sample indices.

    Same as ``detection_collate_fn`` but also returns the per-sample
    index tensor (for ICD cache compatibility).
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, targets, indices


__all__ = [
    "detection_collate_fn",
    "detection_collate_fn_with_index",
]
