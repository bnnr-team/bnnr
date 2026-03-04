"""Bbox-aware augmentations for object detection.

Provides ``BboxAwareAugmentation`` — a ``BaseAugmentation`` subclass that
augments *both* images and bounding box targets.  The trainer detects
these via ``hasattr(aug, 'apply_with_targets')`` and dispatches
accordingly in ``_apply_augmentation_to_batch``.

Two flavours are provided:

1. **AlbumentationsBboxAugmentation** – wraps any Albumentations pipeline
   that has ``BboxParams`` configured.
2. **Pure-NumPy geometric augmentations** (``DetectionHorizontalFlip``,
   ``DetectionVerticalFlip``, ``DetectionRandomRotate90``,
   ``DetectionRandomScale``) – lightweight, zero extra dependencies.

Additionally, two detection-specific composite augmentations are included:

3. **MosaicAugmentation** – 4-image mosaic (à la YOLOv4).
4. **DetectionMixUp** – alpha-blend two images with combined targets.
"""

from __future__ import annotations

import abc
import logging
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor

from bnnr.augmentations import BaseAugmentation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  BboxAwareAugmentation base
# ---------------------------------------------------------------------------


class BboxAwareAugmentation(BaseAugmentation, abc.ABC):
    """Augmentation that transforms both images and bounding boxes.

    Subclasses must implement ``apply_with_targets``.  The inherited
    ``apply`` method is also available for image-only use (boxes are
    dropped).

    Target dict format::

        {
            "boxes": np.ndarray | Tensor  # [N, 4] xyxy
            "labels": np.ndarray | Tensor  # [N]
        }
    """

    name: str = "bbox_aware"

    @abc.abstractmethod
    def apply_with_targets(
        self,
        image: np.ndarray,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply augmentation to image *and* bounding boxes.

        Parameters
        ----------
        image : np.ndarray
            HWC uint8 image.
        target : dict
            Must contain ``boxes`` (N,4 xyxy) and ``labels`` (N,).

        Returns
        -------
        (augmented_image, augmented_target)
        """
        ...

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Image-only fallback (boxes are not updated)."""
        dummy_target = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int64),
        }
        aug_image, _ = self.apply_with_targets(image, dummy_target)
        return aug_image


# ---------------------------------------------------------------------------
#  Helper: convert boxes between numpy / Tensor and clip to image
# ---------------------------------------------------------------------------

def _ensure_numpy_boxes(boxes: Any) -> np.ndarray:
    """Convert boxes (Tensor or ndarray) to float32 ndarray."""
    if isinstance(boxes, Tensor):
        result: np.ndarray = boxes.detach().cpu().numpy().astype(np.float32)
        return result
    out: np.ndarray = np.asarray(boxes, dtype=np.float32)
    return out


def _ensure_numpy_labels(labels: Any) -> np.ndarray:
    if isinstance(labels, Tensor):
        result: np.ndarray = labels.detach().cpu().numpy().astype(np.int64)
        return result
    out: np.ndarray = np.asarray(labels, dtype=np.int64)
    return out


def _clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Clip xyxy boxes to image boundaries."""
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def _filter_small_boxes(
    boxes: np.ndarray, labels: np.ndarray, min_area: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove boxes with area < min_area after clipping."""
    if len(boxes) == 0:
        return boxes, labels
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    keep = areas >= min_area
    return boxes[keep], labels[keep]


def _restore_target_types(
    target: dict[str, Any],
    boxes: np.ndarray,
    labels: np.ndarray,
    was_tensor: bool,
) -> dict[str, Any]:
    """Re-wrap boxes/labels into the same type as the original target."""
    out = dict(target)
    if was_tensor:
        out["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        out["labels"] = torch.as_tensor(labels, dtype=torch.int64)
    else:
        out["boxes"] = boxes
        out["labels"] = labels
    return out


# ---------------------------------------------------------------------------
#  Pure-NumPy geometric augmentations
# ---------------------------------------------------------------------------


class DetectionHorizontalFlip(BboxAwareAugmentation):
    """Random horizontal flip of image + boxes."""

    name: str = "det_hflip"

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability:
            return image, target

        h, w = image.shape[:2]
        was_tensor = isinstance(target.get("boxes"), Tensor)
        boxes = _ensure_numpy_boxes(target["boxes"]).copy()
        labels = _ensure_numpy_labels(target["labels"]).copy()

        flipped = cv2.flip(image, 1)  # horizontal

        if len(boxes) > 0:
            new_x1 = w - boxes[:, 2]
            new_x2 = w - boxes[:, 0]
            boxes[:, 0] = new_x1
            boxes[:, 2] = new_x2
            boxes = _clip_boxes(boxes, h, w)
            boxes, labels = _filter_small_boxes(boxes, labels)

        return flipped, _restore_target_types(target, boxes, labels, was_tensor)


class DetectionVerticalFlip(BboxAwareAugmentation):
    """Random vertical flip of image + boxes."""

    name: str = "det_vflip"

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability:
            return image, target

        h, w = image.shape[:2]
        was_tensor = isinstance(target.get("boxes"), Tensor)
        boxes = _ensure_numpy_boxes(target["boxes"]).copy()
        labels = _ensure_numpy_labels(target["labels"]).copy()

        flipped = cv2.flip(image, 0)  # vertical

        if len(boxes) > 0:
            new_y1 = h - boxes[:, 3]
            new_y2 = h - boxes[:, 1]
            boxes[:, 1] = new_y1
            boxes[:, 3] = new_y2
            boxes = _clip_boxes(boxes, h, w)
            boxes, labels = _filter_small_boxes(boxes, labels)

        return flipped, _restore_target_types(target, boxes, labels, was_tensor)


class DetectionRandomRotate90(BboxAwareAugmentation):
    """Random 90° rotation (0/90/180/270) of image + boxes."""

    name: str = "det_rotate90"

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability:
            return image, target

        k = self._rnd.choice([0, 1, 2, 3])  # number of 90° rotations
        if k == 0:
            return image, target

        h, w = image.shape[:2]
        was_tensor = isinstance(target.get("boxes"), Tensor)
        boxes = _ensure_numpy_boxes(target["boxes"]).copy()
        labels = _ensure_numpy_labels(target["labels"]).copy()

        rotated = np.rot90(image, k=k).copy()

        if len(boxes) > 0:
            for _ in range(k):
                # Each 90° CW rotation: (x1, y1, x2, y2) → (y1, W-x2, y2, W-x1)
                # where W is the current width before rotation
                _, cur_w = h, w
                new_boxes = np.empty_like(boxes)
                new_boxes[:, 0] = boxes[:, 1]         # new x1 = old y1
                new_boxes[:, 1] = cur_w - boxes[:, 2]  # new y1 = W - old x2
                new_boxes[:, 2] = boxes[:, 3]          # new x2 = old y2
                new_boxes[:, 3] = cur_w - boxes[:, 0]  # new y2 = W - old x1
                boxes = new_boxes
                # After 90° rotation, dimensions swap
                h, w = w, h

            boxes = _clip_boxes(boxes, h, w)
            boxes, labels = _filter_small_boxes(boxes, labels)

        return rotated, _restore_target_types(target, boxes, labels, was_tensor)


class DetectionRandomScale(BboxAwareAugmentation):
    """Random scale (resize) image + boxes.

    Parameters
    ----------
    scale_range : tuple[float, float]
        Min and max scale factors.
    """

    name: str = "det_scale"

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.8, 1.2),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scale_range = scale_range

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability:
            return image, target

        h, w = image.shape[:2]
        was_tensor = isinstance(target.get("boxes"), Tensor)
        boxes = _ensure_numpy_boxes(target["boxes"]).copy()
        labels = _ensure_numpy_labels(target["labels"]).copy()

        sx = self._rnd.uniform(*self.scale_range)
        sy = self._rnd.uniform(*self.scale_range)

        new_w, new_h = int(w * sx), int(h * sy)
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy
            boxes = _clip_boxes(boxes, new_h, new_w)
            boxes, labels = _filter_small_boxes(boxes, labels)

        return scaled, _restore_target_types(target, boxes, labels, was_tensor)


# ---------------------------------------------------------------------------
#  Albumentations BboxParams wrapper
# ---------------------------------------------------------------------------


class AlbumentationsBboxAugmentation(BboxAwareAugmentation):
    """Wraps an Albumentations pipeline with ``BboxParams`` for detection.

    Example::

        import albumentations as A
        from bnnr.detection_augmentations import AlbumentationsBboxAugmentation

        aug = AlbumentationsBboxAugmentation(
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["labels"],
                    min_visibility=0.3,
                ),
            ),
            name_override="albu_det_standard",
        )
    """

    name: str = "albu_bbox"

    def __init__(
        self,
        transform: Any,
        probability: float = 1.0,
        intensity: float = 1.0,
        name_override: str | None = None,
        random_state: int | None = None,
    ) -> None:
        try:
            import albumentations  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Albumentations is required for AlbumentationsBboxAugmentation. "
                "Install it with: pip install albumentations"
            ) from exc

        super().__init__(
            probability=probability,
            intensity=intensity,
            name_override=name_override or "albu_bbox",
            random_state=random_state,
        )
        self._transform = transform

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability:
            return image, target

        was_tensor = isinstance(target.get("boxes"), Tensor)
        boxes = _ensure_numpy_boxes(target["boxes"])
        labels = _ensure_numpy_labels(target["labels"])

        # Albumentations expects boxes as list of [x1, y1, x2, y2]
        bboxes_list = boxes.tolist() if len(boxes) > 0 else []
        labels_list = labels.tolist() if len(labels) > 0 else []

        result = self._transform(
            image=image,
            bboxes=bboxes_list,
            labels=labels_list,
        )

        aug_image = result["image"]
        aug_boxes = np.array(result["bboxes"], dtype=np.float32).reshape(-1, 4)
        aug_labels = np.array(result["labels"], dtype=np.int64)

        if aug_image.ndim == 2:
            aug_image = np.stack([aug_image] * 3, axis=-1)

        return aug_image.astype(np.uint8), _restore_target_types(
            target, aug_boxes, aug_labels, was_tensor,
        )


# ---------------------------------------------------------------------------
#  MosaicAugmentation (4-image composite)
# ---------------------------------------------------------------------------


class MosaicAugmentation(BboxAwareAugmentation):
    """4-image Mosaic augmentation (à la YOLOv4).

    Creates a 2×2 mosaic from the current image and 3 additional images
    provided via ``set_pool()``.  Each image occupies a random quadrant
    with a random center offset.

    Must call ``set_pool(images, targets)`` before training to register
    the extra images.  The trainer should call this once per epoch with
    a shuffled subset of the dataset.
    """

    name: str = "det_mosaic"

    def __init__(
        self,
        output_size: tuple[int, int] = (640, 640),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.output_size = output_size  # (H, W)
        self._pool_images: list[np.ndarray] = []
        self._pool_targets: list[dict[str, Any]] = []

    def set_pool(
        self,
        images: list[np.ndarray],
        targets: list[dict[str, Any]],
    ) -> None:
        """Register pool of extra images/targets for mosaic composition."""
        self._pool_images = images
        self._pool_targets = targets

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability or len(self._pool_images) < 3:
            return image, target

        was_tensor = isinstance(target.get("boxes"), Tensor)
        out_h, out_w = self.output_size

        # Random mosaic center
        cx = self._rnd.randint(out_w // 4, 3 * out_w // 4)
        cy = self._rnd.randint(out_h // 4, 3 * out_h // 4)

        # Select 3 extra images
        extra_indices = self._rnd.sample(range(len(self._pool_images)), k=3)
        imgs = [image] + [self._pool_images[i] for i in extra_indices]
        tgts = [target] + [self._pool_targets[i] for i in extra_indices]

        mosaic = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
        all_boxes: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        # Quadrant placements: (slice for mosaic, crop from source)
        placements = [
            (0, 0, cy, cx),          # top-left
            (0, cx, cy, out_w),      # top-right
            (cy, 0, out_h, cx),      # bottom-left
            (cy, cx, out_h, out_w),  # bottom-right
        ]

        for idx, (y1, x1, y2, x2) in enumerate(placements):
            qh, qw = y2 - y1, x2 - x1
            if qh <= 0 or qw <= 0:
                continue

            src_img = imgs[idx]
            src_h, src_w = src_img.shape[:2]

            # Resize source to fit quadrant
            resized = cv2.resize(src_img, (qw, qh), interpolation=cv2.INTER_LINEAR)
            mosaic[y1:y2, x1:x2] = resized

            # Scale and offset boxes
            src_boxes = _ensure_numpy_boxes(tgts[idx]["boxes"]).copy()
            src_labels = _ensure_numpy_labels(tgts[idx]["labels"]).copy()

            if len(src_boxes) > 0:
                # Scale from source image coords to quadrant coords
                scale_x = qw / max(src_w, 1)
                scale_y = qh / max(src_h, 1)
                src_boxes[:, [0, 2]] = src_boxes[:, [0, 2]] * scale_x + x1
                src_boxes[:, [1, 3]] = src_boxes[:, [1, 3]] * scale_y + y1
                src_boxes = _clip_boxes(src_boxes, out_h, out_w)
                src_boxes, src_labels = _filter_small_boxes(src_boxes, src_labels)
                all_boxes.append(src_boxes)
                all_labels.append(src_labels)

        if all_boxes:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
        else:
            final_boxes = np.zeros((0, 4), dtype=np.float32)
            final_labels = np.zeros((0,), dtype=np.int64)

        return mosaic, _restore_target_types(target, final_boxes, final_labels, was_tensor)


# ---------------------------------------------------------------------------
#  DetectionMixUp
# ---------------------------------------------------------------------------


class DetectionMixUp(BboxAwareAugmentation):
    """Alpha-blend two detection images with combined targets.

    Like ``MosaicAugmentation``, call ``set_pool()`` first.
    """

    name: str = "det_mixup"

    def __init__(
        self,
        alpha_range: tuple[float, float] = (0.3, 0.7),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha_range = alpha_range
        self._pool_images: list[np.ndarray] = []
        self._pool_targets: list[dict[str, Any]] = []

    def set_pool(
        self,
        images: list[np.ndarray],
        targets: list[dict[str, Any]],
    ) -> None:
        self._pool_images = images
        self._pool_targets = targets

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability or len(self._pool_images) < 1:
            return image, target

        was_tensor = isinstance(target.get("boxes"), Tensor)
        h, w = image.shape[:2]

        # Pick a random partner
        partner_idx = self._rnd.randint(0, len(self._pool_images) - 1)
        partner_img = self._pool_images[partner_idx]
        partner_tgt = self._pool_targets[partner_idx]

        # Resize partner to same size
        partner_resized = cv2.resize(partner_img, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha = self._rnd.uniform(*self.alpha_range)
        mixed = cv2.addWeighted(image, alpha, partner_resized, 1.0 - alpha, 0)

        # Combine boxes from both images
        boxes1 = _ensure_numpy_boxes(target["boxes"]).copy()
        labels1 = _ensure_numpy_labels(target["labels"]).copy()
        boxes2 = _ensure_numpy_boxes(partner_tgt["boxes"]).copy()
        labels2 = _ensure_numpy_labels(partner_tgt["labels"]).copy()

        # Scale partner boxes to target image size
        p_h, p_w = partner_img.shape[:2]
        if len(boxes2) > 0:
            boxes2[:, [0, 2]] *= w / max(p_w, 1)
            boxes2[:, [1, 3]] *= h / max(p_h, 1)
            boxes2 = _clip_boxes(boxes2, h, w)

        if len(boxes1) > 0 and len(boxes2) > 0:
            final_boxes = np.concatenate([boxes1, boxes2], axis=0)
            final_labels = np.concatenate([labels1, labels2], axis=0)
        elif len(boxes1) > 0:
            final_boxes, final_labels = boxes1, labels1
        elif len(boxes2) > 0:
            final_boxes, final_labels = boxes2, labels2
        else:
            final_boxes = np.zeros((0, 4), dtype=np.float32)
            final_labels = np.zeros((0,), dtype=np.int64)

        return mixed, _restore_target_types(target, final_boxes, final_labels, was_tensor)


# ---------------------------------------------------------------------------
#  Detection augmentation presets
# ---------------------------------------------------------------------------

def get_detection_preset(
    preset_name: str = "standard",
    random_state: int | None = None,
) -> list[BboxAwareAugmentation]:
    """Return a list of bbox-aware augmentations for the given preset.

    Presets
    -------
    light
        Horizontal flip only.
    standard
        Flips, rotation, scale, Albumentations color jitter (if available).
    aggressive
        All of standard plus mosaic and mixup.
    """
    preset_name = preset_name.lower().strip()

    if preset_name == "light":
        return [
            DetectionHorizontalFlip(probability=0.5, random_state=random_state),
        ]

    if preset_name == "standard":
        augs: list[BboxAwareAugmentation] = [
            DetectionHorizontalFlip(probability=0.5, random_state=random_state),
            DetectionVerticalFlip(probability=0.5, random_state=random_state),
            DetectionRandomRotate90(probability=0.5, random_state=random_state),
            DetectionRandomScale(
                scale_range=(0.85, 1.15),
                probability=0.5,
                random_state=random_state,
            ),
        ]
        # Add Albumentations color jitter if available
        try:
            import albumentations as alb

            augs.append(
                AlbumentationsBboxAugmentation(
                    alb.Compose(
                        [
                            alb.RandomBrightnessContrast(p=0.5),
                            alb.HueSaturationValue(p=0.5),
                        ],
                        bbox_params=alb.BboxParams(
                            format="pascal_voc",
                            label_fields=["labels"],
                            min_visibility=0.2,
                        ),
                    ),
                    name_override="det_color_jitter",
                    probability=0.5,
                    random_state=random_state,
                )
            )
        except ImportError:
            pass
        return augs

    if preset_name == "aggressive":
        augs = get_detection_preset("standard", random_state=random_state)
        augs.extend([
            MosaicAugmentation(
                output_size=(640, 640),
                probability=0.5,
                random_state=random_state,
            ),
            DetectionMixUp(
                alpha_range=(0.3, 0.7),
                probability=0.5,
                random_state=random_state,
            ),
        ])
        return augs

    raise ValueError(
        f"Unknown detection preset: '{preset_name}'. "
        f"Available: light, standard, aggressive"
    )


__all__ = [
    "BboxAwareAugmentation",
    "AlbumentationsBboxAugmentation",
    "DetectionHorizontalFlip",
    "DetectionVerticalFlip",
    "DetectionRandomRotate90",
    "DetectionRandomScale",
    "MosaicAugmentation",
    "DetectionMixUp",
    "get_detection_preset",
]
