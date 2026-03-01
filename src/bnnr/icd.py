"""ICD and AICD augmentation implementations for classification workflows."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from bnnr.augmentations import BaseAugmentation
from bnnr.xai import generate_saliency_maps
from bnnr.xai_cache import XAICache

logger = logging.getLogger(__name__)

_VALID_FILL_STRATEGIES = frozenset(
    {"gaussian_blur", "local_mean", "global_mean", "noise", "solid"}
)


class _BaseICD(BaseAugmentation):
    """Base class for Intelligent Coarse Dropout (ICD) and Anti-ICD (AICD).

    Uses XAI saliency maps to create a *tile-based* mask over the most (ICD)
    or least (AICD) salient regions of the image.  The masked area is filled
    with a configurable strategy (blurred original, local mean, noise, etc.)
    instead of a hard black rectangle.
    """

    invert_mask: bool = False
    device_compatible: bool = True

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[nn.Module],
        threshold_percentile: float = 70.0,
        explainer: str = "opticam",
        use_cuda: bool = True,
        cache: XAICache | None = None,
        # --- new tile / fill params ---
        tile_size: int = 8,
        fill_strategy: str = "gaussian_blur",
        fill_value: int = 0,
        blur_kernel_ratio: float = 0.15,
        # --- deprecated compat ---
        mask_value: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.target_layers = target_layers
        self.threshold_percentile = threshold_percentile
        self.explainer = explainer
        self.use_cuda = use_cuda
        self.cache = cache
        self._current_label: int | None = None
        self._cache_miss_count: int = 0

        # Tile size
        if tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        self.tile_size = tile_size

        # Fill strategy
        if fill_strategy not in _VALID_FILL_STRATEGIES:
            raise ValueError(
                f"fill_strategy must be one of {sorted(_VALID_FILL_STRATEGIES)}, "
                f"got {fill_strategy!r}"
            )
        self.fill_strategy = fill_strategy
        self.fill_value = fill_value
        self.blur_kernel_ratio = blur_kernel_ratio

        # Backward-compat: mask_value → solid fill
        if mask_value is not None:
            warnings.warn(
                "mask_value is deprecated. Use fill_strategy='solid' and "
                "fill_value=<color> instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.fill_strategy = "solid"
            self.fill_value = mask_value

        if self.cache is None:
            warnings.warn(
                f"{self.__class__.__name__} initialized without XAICache. "
                "Saliency maps will be computed online and can be very slow.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Tensor entry-point (delegates to numpy path)
    # ------------------------------------------------------------------

    def apply_tensor_native(self, images: torch.Tensor) -> torch.Tensor:
        """GPU-native ICD/AICD masking using cached saliency maps.

        Falls back to CPU path if cache is unavailable or labels are not set.
        """
        return self.apply_tensor(images)

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def set_label(self, label: int) -> None:
        self._current_label = int(label)

    # ------------------------------------------------------------------
    # Saliency map retrieval
    # ------------------------------------------------------------------

    def _compute_online(self, image: np.ndarray, label: int) -> np.ndarray:
        """Compute saliency map on-the-fly for a single image (slow)."""
        tensor = (
            torch.as_tensor(image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        labels = torch.as_tensor([label], dtype=torch.long)
        device = next(self.model.parameters()).device
        tensor = tensor.to(device)
        labels = labels.to(device)
        maps = generate_saliency_maps(
            self.model, tensor, labels, self.target_layers, method=self.explainer
        )
        return maps[0]

    def _compute_online_batch(
        self, images: list[np.ndarray], labels: list[int],
    ) -> list[np.ndarray]:
        """Compute saliency maps for a whole batch in ONE forward pass.

        This is dramatically faster than calling ``_compute_online`` per image
        because GPU utilisation is maximised (1 forward pass instead of N).
        """
        if not images:
            return []
        tensors = [
            torch.as_tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)
            for img in images
        ]
        batch = torch.stack(tensors)
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        label_tensor = label_tensor.to(device)
        maps = generate_saliency_maps(
            self.model, batch, label_tensor, self.target_layers, method=self.explainer,
        )
        return [maps[i] for i in range(len(images))]

    def _get_map(
        self, image: np.ndarray, label: int, sample_index: int | None = None
    ) -> np.ndarray:
        if self.cache is not None:
            cached = self.cache.get_importance_map(
                image, label, sample_index=sample_index
            )
            if cached is not None:
                return cached
            # Cache miss — compute online AND save for next epoch (lazy caching)
            self._cache_miss_count += 1
            if self._cache_miss_count <= 5 or self._cache_miss_count % 100 == 0:
                logger.warning(
                    "%s: XAI cache miss #%d — computing saliency online "
                    "(will be cached for subsequent epochs).",
                    self.__class__.__name__,
                    self._cache_miss_count,
                )
            saliency = self._compute_online(image, label)
            self.cache.save_map(
                saliency, label, sample_index=sample_index, image=image,
            )
            return saliency

        # No cache at all — always compute online (nothing to save to)
        return self._compute_online(image, label)

    # ------------------------------------------------------------------
    # Tile-based mask construction
    # ------------------------------------------------------------------

    def _compute_tile_mask(
        self, saliency_map: np.ndarray, image_h: int, image_w: int
    ) -> np.ndarray:
        """Build a binary (H, W) mask from tile-averaged saliency scores.

        For ICD  (invert_mask=False): mask tiles with highest saliency.
        For AICD (invert_mask=True):  mask tiles with lowest saliency.
        """
        ts = self.tile_size
        # Resize saliency to image dims if needed
        if saliency_map.shape[0] != image_h or saliency_map.shape[1] != image_w:
            saliency_map = cv2.resize(
                saliency_map.astype(np.float32),
                (image_w, image_h),
                interpolation=cv2.INTER_LINEAR,
            )

        n_rows = max(1, image_h // ts)
        n_cols = max(1, image_w // ts)

        # Vectorised tile scoring: crop to tile-aligned region, reshape, mean
        crop_h = n_rows * ts
        crop_w = n_cols * ts
        cropped = saliency_map[:crop_h, :crop_w].astype(np.float32)
        tile_scores = cropped.reshape(n_rows, ts, n_cols, ts).mean(axis=(1, 3))

        # Threshold
        flat = tile_scores.ravel()
        if self.invert_mask:
            # AICD: mask low-saliency tiles (keep important ones)
            percentile = 100.0 - self.threshold_percentile
            thr = np.percentile(flat, percentile)
            tile_mask = tile_scores < thr
        else:
            # ICD: mask high-saliency tiles
            thr = np.percentile(flat, self.threshold_percentile)
            tile_mask = tile_scores > thr

        # Vectorised expand: repeat each tile decision to pixel level
        pixel_mask = np.repeat(
            np.repeat(tile_mask, ts, axis=0), ts, axis=1
        )
        # Pad back to original image size if tiles don't evenly divide
        if pixel_mask.shape[0] < image_h or pixel_mask.shape[1] < image_w:
            full_mask = np.zeros((image_h, image_w), dtype=bool)
            full_mask[:pixel_mask.shape[0], :pixel_mask.shape[1]] = pixel_mask
            pixel_mask = full_mask

        return pixel_mask

    # ------------------------------------------------------------------
    # Fill strategies
    # ------------------------------------------------------------------

    def _apply_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Dispatch to the configured fill strategy."""
        out = image.copy()
        if not mask.any():
            return out

        strategy = self.fill_strategy
        if strategy == "gaussian_blur":
            return self._gaussian_blur_fill(out, mask)
        if strategy == "local_mean":
            return self._local_mean_fill(out, mask)
        if strategy == "global_mean":
            return self._global_mean_fill(out, mask)
        if strategy == "noise":
            return self._noise_fill(out, mask)
        # solid
        return self._solid_fill(out, mask)

    def _gaussian_blur_fill(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Replace masked pixels with a heavily blurred version of the original."""
        h, w = image.shape[:2]
        k_size = max(3, int(max(h, w) * self.blur_kernel_ratio))
        if k_size % 2 == 0:
            k_size += 1
        blurred = cv2.GaussianBlur(image, (k_size, k_size), 0)
        image[mask] = blurred[mask]
        return image

    def _local_mean_fill(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Fill each masked tile with the mean colour of its non-masked neighbours."""
        ts = self.tile_size
        h, w = image.shape[:2]
        n_rows = max(1, h // ts)
        n_cols = max(1, w // ts)

        # Build a tile-level mask (True = masked)
        tile_masked = np.zeros((n_rows, n_cols), dtype=bool)
        for r in range(n_rows):
            y0, y1 = r * ts, min((r + 1) * ts, h)
            for c in range(n_cols):
                x0, x1 = c * ts, min((c + 1) * ts, w)
                tile_masked[r, c] = mask[y0:y1, x0:x1].any()

        for r in range(n_rows):
            y0, y1 = r * ts, min((r + 1) * ts, h)
            for c in range(n_cols):
                if not tile_masked[r, c]:
                    continue
                x0, x1 = c * ts, min((c + 1) * ts, w)
                # Collect colours from neighbouring non-masked tiles
                neighbour_pixels: list[np.ndarray] = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n_rows and 0 <= nc < n_cols:
                            if not tile_masked[nr, nc]:
                                ny0, ny1 = nr * ts, min((nr + 1) * ts, h)
                                nx0, nx1 = nc * ts, min((nc + 1) * ts, w)
                                neighbour_pixels.append(
                                    image[ny0:ny1, nx0:nx1].reshape(-1, image.shape[2])
                                )
                if neighbour_pixels:
                    mean_colour = np.concatenate(neighbour_pixels, axis=0).mean(
                        axis=0
                    )
                else:
                    # All neighbours are masked — fall back to global mean
                    mean_colour = image.mean(axis=(0, 1))
                image[y0:y1, x0:x1][mask[y0:y1, x0:x1]] = mean_colour.astype(
                    image.dtype
                )

        return image

    def _global_mean_fill(
        self, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Fill masked pixels with the global mean colour of the image."""
        mean_colour = image.mean(axis=(0, 1)).astype(image.dtype)
        image[mask] = mean_colour
        return image

    def _noise_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill masked pixels with random noise matching per-channel stats."""
        n_masked = int(mask.sum())
        channels = image.shape[2] if image.ndim == 3 else 1
        means = image.mean(axis=(0, 1))
        stds = image.std(axis=(0, 1)).clip(1.0)  # avoid zero-std
        noise = np.empty((n_masked, channels), dtype=image.dtype)
        for ch in range(channels):
            noise[:, ch] = np.clip(
                np.array(
                    [self._rnd.gauss(float(means[ch]), float(stds[ch])) for _ in range(n_masked)]
                ),
                0,
                255,
            ).astype(image.dtype)
        image[mask] = noise
        return image

    def _solid_fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill masked pixels with a solid colour value."""
        image[mask] = self.fill_value
        return image

    # ------------------------------------------------------------------
    # Helpers for grayscale output conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_original_channels(
        result: np.ndarray, original_ndim: int, original_channels: int,
    ) -> np.ndarray:
        """Convert an RGB result back to match the original image layout."""
        if original_channels == 1:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            return gray if original_ndim == 2 else gray[..., None]
        return result

    # ------------------------------------------------------------------
    # Main apply methods
    # ------------------------------------------------------------------

    def apply_with_label(
        self, image: np.ndarray, label: int | np.ndarray, sample_index: int | None = None
    ) -> np.ndarray:
        # Multi-label support: convert multi-hot vector to dominant class
        if isinstance(label, np.ndarray) and label.ndim >= 1 and label.size > 1:
            label = int(np.argmax(label))
        else:
            label = int(label)
        original_ndim = image.ndim
        original_channels = 1 if image.ndim == 2 else image.shape[-1]
        image = self.validate_input(image)
        h, w = image.shape[:2]

        imp = self._get_map(image, label, sample_index=sample_index)
        mask = self._compute_tile_mask(imp, h, w)
        out = self._apply_fill(image, mask)

        return self._to_original_channels(out, original_ndim, original_channels)

    def apply_batch_with_labels(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        sample_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        out = images.copy()

        # ── Phase 1: Decide which images to augment & check cache ─────
        augment_meta: list[tuple[int, np.ndarray, int, int | None, int, int]] = []
        cached_maps: dict[int, np.ndarray] = {}
        uncached: list[tuple[int, np.ndarray, int, int | None]] = []

        for idx in range(images.shape[0]):
            if self._rnd.random() > self.probability:
                continue
            # Multi-label support: convert multi-hot vector to dominant class
            raw_label = labels[idx]
            label = int(raw_label) if raw_label.ndim == 0 else int(np.argmax(raw_label))
            sample_index = (
                int(sample_indices[idx]) if sample_indices is not None else None
            )
            original_ndim = images[idx].ndim
            original_channels = 1 if images[idx].ndim == 2 else images[idx].shape[-1]
            image = self.validate_input(images[idx])
            augment_meta.append(
                (idx, image, label, sample_index, original_ndim, original_channels)
            )

            # Try cache
            if self.cache is not None:
                hit = self.cache.get_importance_map(
                    image, label, sample_index=sample_index
                )
                if hit is not None:
                    cached_maps[idx] = hit
                    continue

            uncached.append((idx, image, label, sample_index))

        # ── Phase 2: Batch-compute uncached saliency maps in ONE pass ─
        if uncached:
            uncached_images = [item[1] for item in uncached]
            uncached_labels = [item[2] for item in uncached]
            batch_maps = self._compute_online_batch(uncached_images, uncached_labels)

            for (u_idx, u_image, u_label, u_sample_index), smap in zip(
                uncached, batch_maps
            ):
                cached_maps[u_idx] = smap
                # Lazy-save to cache for next epoch
                if self.cache is not None:
                    self.cache.save_map(
                        smap, u_label,
                        sample_index=u_sample_index,
                        image=u_image,
                    )

            n_miss = len(uncached)
            self._cache_miss_count += n_miss
            if self._cache_miss_count <= 5 or self._cache_miss_count % 100 == 0:
                logger.warning(
                    "%s: batch cache miss (%d images) — computed & cached for "
                    "subsequent epochs.",
                    self.__class__.__name__,
                    n_miss,
                )

        # ── Phase 3: Apply tile mask + fill ────────────────────────────
        for idx, image, label, sample_index, original_ndim, original_channels in augment_meta:
            h, w = image.shape[:2]
            smap = cached_maps[idx]
            mask = self._compute_tile_mask(smap, h, w)
            result = self._apply_fill(image, mask)
            out[idx] = self._to_original_channels(
                result, original_ndim, original_channels
            )

        return out

    def apply(self, image: np.ndarray) -> np.ndarray:
        if self._current_label is None:
            raise ValueError("Label not set. Use set_label() or apply_with_label().")
        return self.apply_with_label(image, self._current_label)


class ICD(_BaseICD):
    """Intelligent Coarse Dropout — masks the most salient tiles."""

    name = "icd"
    invert_mask = False


class AICD(_BaseICD):
    """Anti-ICD — masks the least salient tiles (keeps important regions)."""

    name = "aicd"
    invert_mask = True
