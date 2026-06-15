"""Batch-level augmentation application for classification and detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from bnnr.augmentations import BaseAugmentation
from bnnr.utils import lazy_cv2 as cv2

if TYPE_CHECKING:
    from bnnr.trainer import BNNRTrainer


def apply_augmentation_to_batch(
    trainer: BNNRTrainer,
    batch: Any,
    augmentation: BaseAugmentation,
    sample_indices: Tensor | None = None,
) -> Any:
    """Apply *augmentation* to a training/eval batch (classification or detection)."""
    if trainer._is_detection:
        images, targets = batch
        targets_mode = trainer.config.detection_targets_mode
        can_apply_with_targets = hasattr(augmentation, "apply_with_targets")
        if targets_mode == "bbox_aware" and not can_apply_with_targets:
            can_apply_with_targets = False
        if targets_mode == "image_only":
            can_apply_with_targets = False

        if can_apply_with_targets:
            np_images = trainer._tensor_to_uint8(images)
            aug_images_list = []
            aug_targets_list = []
            ref_h = int(images.shape[2])
            ref_w = int(images.shape[3])
            for idx in range(np_images.shape[0]):
                aug_img, aug_tgt = augmentation.apply_with_targets(  # type: ignore[attr-defined]
                    np_images[idx], targets[idx],
                )
                if aug_img.shape[0] != ref_h or aug_img.shape[1] != ref_w:
                    src_h, src_w = int(aug_img.shape[0]), int(aug_img.shape[1])
                    sx = ref_w / max(src_w, 1)
                    sy = ref_h / max(src_h, 1)
                    aug_img = cv2.resize(aug_img, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

                    boxes_any = aug_tgt.get("boxes")
                    if isinstance(boxes_any, Tensor):
                        boxes_t = boxes_any.clone().to(dtype=torch.float32)
                        if boxes_t.numel() > 0:
                            boxes_t[:, [0, 2]] *= sx
                            boxes_t[:, [1, 3]] *= sy
                            boxes_t[:, [0, 2]] = boxes_t[:, [0, 2]].clamp_(0, ref_w)
                            boxes_t[:, [1, 3]] = boxes_t[:, [1, 3]].clamp_(0, ref_h)
                        aug_tgt["boxes"] = boxes_t
                    elif boxes_any is not None:
                        boxes_np = np.asarray(boxes_any, dtype=np.float32).copy()
                        if boxes_np.size > 0:
                            boxes_np[:, [0, 2]] *= sx
                            boxes_np[:, [1, 3]] *= sy
                            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0, ref_w)
                            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0, ref_h)
                        aug_tgt["boxes"] = boxes_np

                aug_images_list.append(aug_img)
                aug_targets_list.append(aug_tgt)
            aug_images_np = np.stack(aug_images_list, axis=0)
            return trainer._det_uint8_batch_to_float01(aug_images_np, ref_batch=images), aug_targets_list

        try:
            images = augmentation.apply_tensor(images)
            return images, targets
        except (NotImplementedError, RuntimeError, TypeError):
            np_images = trainer._tensor_to_uint8(images)
            aug_images = augmentation.apply_batch(np_images)
            return trainer._det_uint8_batch_to_float01(aug_images, ref_batch=images), targets

    images, labels = batch
    if hasattr(augmentation, "apply_batch_with_labels"):
        np_images = trainer._tensor_to_uint8(images)
        np_labels = labels.detach().cpu().numpy()
        np_indices = sample_indices.detach().cpu().numpy() if sample_indices is not None else None
        aug_images = augmentation.apply_batch_with_labels(
            np_images, np_labels, sample_indices=np_indices,
        )
        return trainer._uint8_to_tensor(aug_images, ref_batch=images), labels

    try:
        images = augmentation.apply_tensor(images)
        return images, labels
    except (NotImplementedError, RuntimeError, TypeError):
        np_images = trainer._tensor_to_uint8(images)
        aug_images = augmentation.apply_batch(np_images)
        return trainer._uint8_to_tensor(aug_images, ref_batch=images), labels
