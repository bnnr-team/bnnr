"""Detection-aware ICD / AICD.

Uses bounding box regions as importance priors: pixels inside boxes are
considered *important*, pixels outside are *background*.  This creates a
tile-based mask using the same fill strategies as the classification ICD.

Two variants:

- **DetectionICD**: masks the most important tiles (object regions) to
  force the model to learn from context.
- **DetectionAICD**: masks background tiles (least important) to
  reduce noisy context and focus training on objects.

For full saliency-based detection ICD (using XAI per-box gradients),
see PR3 plans.

The augmentations implement ``apply_with_targets`` and are therefore
compatible with ``BNNRTrainer``'s detection path.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from bnnr.detection_augmentations import (
    BboxAwareAugmentation,
    _ensure_numpy_boxes,
)

logger = logging.getLogger(__name__)

_VALID_FILL_STRATEGIES = frozenset(
    {"gaussian_blur", "local_mean", "global_mean", "noise", "solid"}
)


class _DetectionBaseICD(BboxAwareAugmentation):
    """Base for detection-aware ICD / AICD.

    Builds a saliency-like map from bounding boxes: each pixel inside a box
    receives a saliency score of 1.0, outside 0.0.  Tile-based masking and
    fill strategies are then applied identically to the classification ICD.
    """

    invert_mask: bool = False

    def __init__(
        self,
        threshold_percentile: float = 70.0,
        tile_size: int = 8,
        fill_strategy: str = "gaussian_blur",
        fill_value: int = 0,
        blur_kernel_ratio: float = 0.15,
        min_box_area: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold_percentile = threshold_percentile
        self.tile_size = max(1, tile_size)
        if fill_strategy not in _VALID_FILL_STRATEGIES:
            raise ValueError(
                f"fill_strategy must be one of {sorted(_VALID_FILL_STRATEGIES)}, "
                f"got {fill_strategy!r}"
            )
        self.fill_strategy = fill_strategy
        self.fill_value = fill_value
        self.blur_kernel_ratio = blur_kernel_ratio
        self.min_box_area = min_box_area

    # ------------------------------------------------------------------
    #  Box-derived saliency map
    # ------------------------------------------------------------------

    @staticmethod
    def _boxes_to_saliency(
        boxes: np.ndarray, h: int, w: int,
    ) -> np.ndarray:
        """Create a saliency map from xyxy boxes.

        Each pixel inside any box gets 1.0, outside gets 0.0.
        Overlapping boxes simply stay at 1.0.
        """
        saliency: np.ndarray = np.zeros((h, w), dtype=np.float32)
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            saliency[y1:y2, x1:x2] = 1.0
        return saliency

    # ------------------------------------------------------------------
    #  Tile mask (same logic as classification ICD)
    # ------------------------------------------------------------------

    def _compute_tile_mask(
        self, saliency_map: np.ndarray, image_h: int, image_w: int,
    ) -> np.ndarray:
        ts = self.tile_size
        if saliency_map.shape[0] != image_h or saliency_map.shape[1] != image_w:
            saliency_map = cv2.resize(
                saliency_map, (image_w, image_h), interpolation=cv2.INTER_LINEAR,
            )

        n_rows = max(1, image_h // ts)
        n_cols = max(1, image_w // ts)

        tile_scores = np.zeros((n_rows, n_cols), dtype=np.float32)
        for r in range(n_rows):
            y0, y1 = r * ts, min((r + 1) * ts, image_h)
            for c in range(n_cols):
                x0, x1 = c * ts, min((c + 1) * ts, image_w)
                tile_scores[r, c] = saliency_map[y0:y1, x0:x1].mean()

        flat = tile_scores.ravel()
        if self.invert_mask:
            # AICD: mask low-saliency tiles (keep important ones)
            percentile = 100.0 - self.threshold_percentile
            thr = np.percentile(flat, percentile) if len(flat) > 0 else 0.0
            # Use <= so that tiles with exactly the threshold value (often 0)
            # also get masked.  Without this, sparse saliency maps produce
            # a threshold of 0 and no tiles are masked.
            tile_mask = tile_scores <= thr
        else:
            # ICD: mask high-saliency tiles
            thr = np.percentile(flat, self.threshold_percentile) if len(flat) > 0 else 1.0
            tile_mask = tile_scores >= thr

        pixel_mask: np.ndarray = np.zeros((image_h, image_w), dtype=bool)
        for r in range(n_rows):
            y0, y1 = r * ts, min((r + 1) * ts, image_h)
            for c in range(n_cols):
                if tile_mask[r, c]:
                    x0, x1 = c * ts, min((c + 1) * ts, image_w)
                    pixel_mask[y0:y1, x0:x1] = True

        return pixel_mask

    # ------------------------------------------------------------------
    #  Fill strategies (delegated from classification ICD)
    # ------------------------------------------------------------------

    def _apply_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        if not mask.any():
            return out
        strategy = self.fill_strategy
        if strategy == "gaussian_blur":
            return self._gaussian_blur_fill(out, mask)
        if strategy == "local_mean":
            return self._global_mean_fill(out, mask)
        if strategy == "global_mean":
            return self._global_mean_fill(out, mask)
        if strategy == "noise":
            return self._noise_fill(out, mask)
        return self._solid_fill(out, mask)

    def _gaussian_blur_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        k_size = max(3, int(max(h, w) * self.blur_kernel_ratio))
        if k_size % 2 == 0:
            k_size += 1
        blurred = cv2.GaussianBlur(image, (k_size, k_size), 0)
        image[mask] = blurred[mask]
        return image

    def _global_mean_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mean_colour = image.mean(axis=(0, 1)).astype(image.dtype)
        image[mask] = mean_colour
        return image

    def _noise_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        n_masked = int(mask.sum())
        channels = image.shape[2] if image.ndim == 3 else 1
        means = image.mean(axis=(0, 1))
        stds = image.std(axis=(0, 1)).clip(1.0)
        noise = np.empty((n_masked, channels), dtype=image.dtype)
        for ch in range(channels):
            noise[:, ch] = np.clip(
                np.array([self._rnd.gauss(float(means[ch]), float(stds[ch])) for _ in range(n_masked)]),
                0, 255,
            ).astype(image.dtype)
        image[mask] = noise
        return image

    def _solid_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image[mask] = self.fill_value
        return image

    # ------------------------------------------------------------------
    #  Main apply method
    # ------------------------------------------------------------------

    def apply_with_targets(
        self, image: np.ndarray, target: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        image = self.validate_input(image)
        if self._rnd.random() > self.probability:
            return image, target

        h, w = image.shape[:2]
        boxes = _ensure_numpy_boxes(target["boxes"])

        # Build saliency map from boxes
        saliency = self._boxes_to_saliency(boxes, h, w)
        mask = self._compute_tile_mask(saliency, h, w)
        out = self._apply_fill(image, mask)

        # Targets are unchanged (we only augment the image)
        return out, target


class DetectionICD(_DetectionBaseICD):
    """Detection ICD — masks high-saliency (object) tiles.

    Forces the model to learn from context/background, improving
    robustness to occlusion and co-occurring object bias.
    """

    name: str = "det_icd"
    invert_mask: bool = False


class DetectionAICD(_DetectionBaseICD):
    """Detection Anti-ICD — masks low-saliency (background) tiles.

    Focuses training on object regions by removing distracting
    background context.
    """

    name: str = "det_aicd"
    invert_mask: bool = True


__all__ = [
    "DetectionICD",
    "DetectionAICD",
]
