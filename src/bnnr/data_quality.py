"""Data-quality utilities: duplicate detection (dHash) and image sanity checks.

All functions operate on **torch Tensors** (``[C, H, W]`` or ``[B, C, H, W]``)
so they integrate with the existing DataLoader pipeline without extra I/O.

Computational budget
--------------------
* dHash computation: ~0.05 ms / image  (resize 9×8 + compare)
* Quality checks:    ~0.02 ms / image  (std, mean, nan/inf)
* Duplicate search:  O(n²) hamming comparisons — capped at *MAX_SAMPLES*

For a 10 k-image dataset the total overhead is < 2 s.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

#: Maximum number of images to scan (avoids O(n²) blowup on huge datasets).
MAX_SAMPLES: int = 10_000

#: Hash size in bits — 8×8 = 64 bit difference hash.
_HASH_SIZE: int = 8

#: Hamming-distance threshold below which two hashes are "near-duplicate".
DUPLICATE_HAMMING_THRESHOLD: int = 10

#: Image standard-deviation (after normalisation) below which we flag
#  "zero / near-zero variance" (solid-colour or blank).
ZERO_VARIANCE_THRESHOLD: float = 0.01

#: If an image is smaller than this on *either* spatial axis (before any
#  DataLoader resize) we flag it.  Because the DataLoader may already resize
#  we also check if mean ≈ 0 and std ≈ 0 as a proxy.
TINY_IMAGE_MIN_SIDE: int = 16

#: Thumbnail size (px) for saved duplicate/flagged images.
_THUMBNAIL_SIZE: int = 128

#: Max duplicate groups to save images for (keeps artifact dir small).
_MAX_GROUPS_WITH_IMAGES: int = 10

#: Max images per duplicate group to save.
_MAX_IMAGES_PER_GROUP: int = 8

#: Max quality-flagged images to save thumbnails for.
_MAX_FLAGGED_IMAGES_SAVED: int = 20


# ---------------------------------------------------------------------------
#  dHash  (difference hash)
# ---------------------------------------------------------------------------

def compute_dhash_batch(images: torch.Tensor) -> np.ndarray:
    """Compute 64-bit difference hashes for a batch of images.

    Parameters
    ----------
    images : Tensor  [B, C, H, W]
        Batch of images (any value range — only relative ordering matters).

    Returns
    -------
    np.ndarray of ``np.uint64``  shape ``(B,)``
    """
    with torch.no_grad():
        # 1. Convert to single-channel (mean across channels)
        if images.shape[1] > 1:
            gray = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            gray = images

        # 2. Resize to (_HASH_SIZE+1, _HASH_SIZE) → (9, 8) using bilinear
        resized = F.interpolate(
            gray.float(),
            size=(_HASH_SIZE, _HASH_SIZE + 1),
            mode="bilinear",
            align_corners=False,
        )  # [B, 1, 8, 9]

        # 3. Horizontal gradient: compare pixel[col] > pixel[col+1]
        diff = resized[:, 0, :, :-1] > resized[:, 0, :, 1:]  # [B, 8, 8]

        # 4. Pack 64 bools → uint64
        flat = diff.reshape(-1, 64).cpu().numpy().astype(np.uint8)  # [B, 64]

        powers = (1 << np.arange(64, dtype=np.uint64))  # [64]
        hashes = (flat.astype(np.uint64) * powers[None, :]).sum(axis=1)

    return hashes


def find_near_duplicates(
    hashes: np.ndarray,
    indices: np.ndarray,
    threshold: int = DUPLICATE_HAMMING_THRESHOLD,
) -> list[tuple[int, int, int]]:
    """Find pairs of near-duplicate images by Hamming distance.

    Parameters
    ----------
    hashes : np.ndarray of uint64  shape ``(N,)``
    indices : np.ndarray of int    shape ``(N,)``
        Original dataset indices corresponding to each hash.
    threshold : int
        Maximum Hamming distance to consider a pair "duplicate".

    Returns
    -------
    List of ``(idx_a, idx_b, hamming_distance)`` tuples,
    sorted by distance ascending.
    """
    n = len(hashes)
    pairs: list[tuple[int, int, int]] = []

    # For large N, process in chunks to limit memory
    chunk = 2048
    for i_start in range(0, n, chunk):
        i_end = min(i_start + chunk, n)
        block_a = hashes[i_start:i_end].astype(np.uint64)  # [chunk]

        for j_start in range(i_start, n, chunk):
            j_end = min(j_start + chunk, n)
            # Skip lower-triangle blocks we've already processed
            if j_start < i_start:
                continue
            block_b = hashes[j_start:j_end].astype(np.uint64)  # [chunk2]

            # Pairwise XOR → popcount
            xor = block_a[:, None] ^ block_b[None, :]  # [chunk, chunk2]

            # Vectorised popcount via lookup table
            dist = _popcount_2d(xor)

            # Find pairs within threshold (only upper triangle when same block)
            within = dist <= threshold
            if i_start == j_start:
                # Exclude diagonal + lower triangle to avoid self-pairs and mirrors
                within &= np.triu(np.ones(within.shape, dtype=bool), k=1)
            rows, cols = np.where(within)

            for r, c in zip(rows, cols):
                idx_a = int(indices[i_start + r])
                idx_b = int(indices[j_start + c])
                pairs.append((idx_a, idx_b, int(dist[r, c])))

    pairs.sort(key=lambda t: t[2])
    return pairs


def _popcount_2d(arr: np.ndarray) -> np.ndarray:
    """Vectorised popcount for a 2-D uint64 array."""
    # Kernighan-style is slow in numpy; use byte-level LUT instead.
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    total = np.zeros(arr.shape, dtype=np.int32)
    val = arr.copy()
    for _ in range(8):  # 8 bytes in uint64
        total += lut[(val & 0xFF).astype(np.uint8)]
        val >>= 8
    return total


# ---------------------------------------------------------------------------
#  Thumbnail helpers
# ---------------------------------------------------------------------------

def _tensor_to_thumbnail(image: torch.Tensor, size: int = _THUMBNAIL_SIZE) -> np.ndarray:
    """Convert a single [C, H, W] tensor to a [size, size, 3] uint8 numpy array.

    Handles normalised tensors (ImageNet-style), [0,1], and [0,255] ranges.
    """
    with torch.no_grad():
        img = image.detach().cpu().float()

        # [C, H, W] → [1, C, H, W] → resize → [C, size, size]
        resized = F.interpolate(
            img.unsqueeze(0), size=(size, size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # [C, H, W] → [H, W, C]
        hwc = resized.permute(1, 2, 0).numpy()

        # Handle grayscale → RGB
        if hwc.shape[2] == 1:
            hwc = np.repeat(hwc, 3, axis=2)

        # Auto-detect range and convert to [0, 255]
        vmin, vmax = float(hwc.min()), float(hwc.max())
        if vmin >= 0.0 and vmax <= 1.0:
            out = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
        elif vmin >= 0.0 and vmax <= 255.0:
            out = np.clip(hwc, 0, 255).astype(np.uint8)
        else:
            # Normalised data — rescale to [0, 255]
            if vmax - vmin < 1e-8:
                out = np.zeros_like(hwc, dtype=np.uint8)
            else:
                normed = (hwc - vmin) / (vmax - vmin)
                out = np.clip(normed * 255.0, 0, 255).astype(np.uint8)

    return out


def _save_thumbnail(image_np: np.ndarray, path: Path) -> Path:
    """Save a [H, W, 3] uint8 array as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return path


# ---------------------------------------------------------------------------
#  Image quality checks
# ---------------------------------------------------------------------------

def check_image_quality_batch(
    images: torch.Tensor,
    batch_offset: int,
) -> list[dict[str, Any]]:
    """Run quality checks on a batch of images.

    Parameters
    ----------
    images : Tensor  [B, C, H, W]
    batch_offset : int
        Index of the first image in this batch within the full dataset
        (used for reporting indices).

    Returns
    -------
    List of dicts, one per flagged image::

        {"index": int, "issues": ["zero_variance", ...]}
    """
    issues: list[dict[str, Any]] = []
    b = images.shape[0]

    with torch.no_grad():
        imgs = images.float()

        # Per-image statistics
        flat = imgs.reshape(b, -1)
        stds = flat.std(dim=1)           # [B]
        means = flat.mean(dim=1)         # [B]
        has_nan = torch.isnan(imgs.reshape(b, -1)).any(dim=1)   # [B]
        has_inf = torch.isinf(imgs.reshape(b, -1)).any(dim=1)   # [B]

    stds_np = stds.cpu().numpy()
    means_np = means.cpu().numpy()
    nan_np = has_nan.cpu().numpy()
    inf_np = has_inf.cpu().numpy()

    for i in range(b):
        sample_issues: list[str] = []

        if nan_np[i]:
            sample_issues.append("contains_nan")
        if inf_np[i]:
            sample_issues.append("contains_inf")
        if stds_np[i] < ZERO_VARIANCE_THRESHOLD:
            sample_issues.append("zero_variance")
        # All-black or all-white (common corruption artefact)
        if means_np[i] < 0.005 and stds_np[i] < 0.01:
            sample_issues.append("near_black")
        elif means_np[i] > 0.995 and stds_np[i] < 0.01:
            sample_issues.append("near_white")

        if sample_issues:
            issues.append({
                "index": batch_offset + i,
                "issues": sample_issues,
            })

    return issues


# ---------------------------------------------------------------------------
#  High-level API: run all checks on a DataLoader
# ---------------------------------------------------------------------------

def run_data_quality_analysis(
    train_loader: torch.utils.data.DataLoader,
    *,
    max_samples: int = MAX_SAMPLES,
    save_dir: Path | None = None,
    run_dir: Path | None = None,
    duplicate_threshold: int = DUPLICATE_HAMMING_THRESHOLD,
) -> dict[str, Any]:
    """Run duplicate detection + image quality checks.

    Designed to be called once before training.  Caps at *max_samples* to
    bound compute time.

    Parameters
    ----------
    save_dir : Path | None
        When provided, thumbnails of flagged images are saved here as PNGs
        and relative artifact paths are included in the result dict.
    run_dir : Path | None
        The reporter run directory.  Artifact paths in the output are made
        relative to this path so the dashboard ``/artifacts/`` endpoint can
        serve them.

    Returns
    -------
    dict
        Ready to merge into the ``dataset_profile`` payload.  Keys:

        * ``data_quality.warnings`` — list of warning dicts
        * ``data_quality.duplicate_groups`` — list of duplicate pair groups
        * ``data_quality.summary`` — human-readable summary string
    """
    all_hashes: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    all_quality_issues: list[dict[str, Any]] = []
    # Keep lightweight thumbnails for flagged-image preview.
    # Key = dataset index, value = [H, W, 3] uint8 numpy thumbnail.
    thumbnail_cache: dict[int, np.ndarray] = {}
    processed = 0
    sample_cursor = 0

    for raw_batch in train_loader:
        if processed >= max_samples:
            break

        # Unpack batch (classification/multilabel: image, label, index)
        if len(raw_batch) == 3:
            images, _, _ = raw_batch
        else:
            images, _ = raw_batch

        if images.ndim != 4:
            sample_cursor += images.shape[0]
            continue

        batch_size = images.shape[0]
        remaining = max_samples - processed
        if batch_size > remaining:
            images = images[:remaining]
            batch_size = remaining

        # --- dHash ---
        hashes = compute_dhash_batch(images)
        indices = np.arange(sample_cursor, sample_cursor + batch_size)
        all_hashes.append(hashes)
        all_indices.append(indices)

        # --- Quality checks ---
        batch_issues = check_image_quality_batch(images, sample_cursor)
        all_quality_issues.extend(batch_issues)

        # --- Stash thumbnails for all images so we can preview duplicates
        #     and quality-flagged samples later.  Bounded by max_samples.
        if save_dir is not None:
            for i in range(batch_size):
                global_idx = sample_cursor + i
                thumbnail_cache[global_idx] = _tensor_to_thumbnail(images[i])

        processed += batch_size
        sample_cursor += batch_size

    # --- Aggregate hashes and find duplicates ---
    duplicate_pairs: list[tuple[int, int, int]] = []
    if all_hashes:
        combined_hashes = np.concatenate(all_hashes)
        combined_indices = np.concatenate(all_indices)
        duplicate_pairs = find_near_duplicates(
            combined_hashes, combined_indices,
            threshold=duplicate_threshold,
        )

    # --- Group duplicates into clusters ---
    duplicate_groups = _group_duplicate_pairs(duplicate_pairs)

    # --- Save thumbnails and enrich groups/warnings with artifact paths ---
    if save_dir is not None and (duplicate_groups or all_quality_issues):
        _save_flagged_thumbnails(
            duplicate_groups=duplicate_groups,
            quality_issues=all_quality_issues,
            thumbnail_cache=thumbnail_cache,
            save_dir=save_dir,
            run_dir=run_dir,
        )

    # Free thumbnail cache — no longer needed
    thumbnail_cache.clear()

    # --- Build warnings list ---
    warnings = _build_warnings(all_quality_issues, duplicate_groups, processed)

    # --- Summary ---
    total_warnings = sum(w["count"] for w in warnings)
    if total_warnings == 0:
        summary = f"All {processed:,} scanned images passed quality checks."
    else:
        summary = f"Found {total_warnings} issue(s) across {processed:,} scanned images."

    return {
        "data_quality": {
            "scanned_samples": processed,
            "warnings": warnings,
            "duplicate_groups": duplicate_groups[:50],  # cap for payload size
            "total_duplicate_pairs": len(duplicate_pairs),
            "total_flagged_images": len(all_quality_issues),
            "summary": summary,
        },
    }


def _save_flagged_thumbnails(
    *,
    duplicate_groups: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]],
    thumbnail_cache: dict[int, np.ndarray],
    save_dir: Path,
    run_dir: Path | None,
) -> None:
    """Save thumbnail PNGs for duplicate groups + quality-flagged images.

    Mutates *duplicate_groups* and *quality_issues* in-place to add
    ``image_paths`` lists with artifact-relative paths.
    """
    def _rel(p: Path) -> str:
        """Make path relative to run_dir for the dashboard artifact endpoint."""
        if run_dir is not None:
            try:
                return str(p.relative_to(run_dir)).replace("\\", "/")
            except ValueError:
                pass
        return str(p).replace("\\", "/")

    # --- Duplicate group thumbnails ---
    dup_dir = save_dir / "duplicates"
    for gi, group in enumerate(duplicate_groups[:_MAX_GROUPS_WITH_IMAGES]):
        group_dir = dup_dir / f"group_{gi}"
        paths: list[str] = []
        for idx in group["indices"][:_MAX_IMAGES_PER_GROUP]:
            thumb = thumbnail_cache.get(idx)
            if thumb is None:
                continue
            p = _save_thumbnail(thumb, group_dir / f"idx_{idx}.png")
            paths.append(_rel(p))
        if paths:
            group["image_paths"] = paths

    # --- Quality-issue thumbnails ---
    issue_dir = save_dir / "flagged"
    saved_count = 0
    # Build a mapping from index → issue entry for quick lookup
    for entry in quality_issues:
        if saved_count >= _MAX_FLAGGED_IMAGES_SAVED:
            break
        idx = entry["index"]
        thumb = thumbnail_cache.get(idx)
        if thumb is None:
            continue
        p = _save_thumbnail(thumb, issue_dir / f"idx_{idx}.png")
        entry["image_path"] = _rel(p)
        saved_count += 1


def _group_duplicate_pairs(
    pairs: list[tuple[int, int, int]],
) -> list[dict[str, Any]]:
    """Group duplicate pairs into connected clusters using union-find."""
    if not pairs:
        return []

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    min_dist: dict[tuple[int, int], int] = {}
    for idx_a, idx_b, dist in pairs:
        union(idx_a, idx_b)
        key = (min(idx_a, idx_b), max(idx_a, idx_b))
        if key not in min_dist or dist < min_dist[key]:
            min_dist[key] = dist

    groups_map: dict[int, list[int]] = {}
    all_nodes = set()
    for a, b, _ in pairs:
        all_nodes.add(a)
        all_nodes.add(b)
    for node in all_nodes:
        root = find(node)
        groups_map.setdefault(root, []).append(node)

    result: list[dict[str, Any]] = []
    for members in groups_map.values():
        members = sorted(set(members))
        if len(members) < 2:
            continue
        result.append({
            "indices": members,
            "size": len(members),
        })

    result.sort(key=lambda g: -g["size"])
    return result


def _build_warnings(
    quality_issues: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
    scanned: int,
) -> list[dict[str, Any]]:
    """Aggregate quality issues into typed warning entries."""
    warnings: list[dict[str, Any]] = []

    # --- Duplicates ---
    total_dup_images = sum(g["size"] for g in duplicate_groups)
    if duplicate_groups:
        warnings.append({
            "type": "near_duplicates",
            "severity": "warning",
            "count": total_dup_images,
            "groups": len(duplicate_groups),
            "message": (
                f"{total_dup_images} images in {len(duplicate_groups)} "
                f"near-duplicate group(s). Duplicates can bias training "
                f"and inflate metrics."
            ),
        })

    # --- Aggregate quality issues by type ---
    issue_counts: dict[str, list[int]] = {}
    issue_images: dict[str, list[str]] = {}
    for entry in quality_issues:
        img_path = entry.get("image_path")
        for issue_type in entry["issues"]:
            issue_counts.setdefault(issue_type, []).append(entry["index"])
            if img_path:
                issue_images.setdefault(issue_type, []).append(img_path)

    issue_meta: dict[str, tuple[str, str]] = {
        "contains_nan": (
            "critical",
            "images contain NaN values — likely data corruption or broken "
            "preprocessing pipeline.",
        ),
        "contains_inf": (
            "critical",
            "images contain Inf values — likely data corruption or broken "
            "preprocessing pipeline.",
        ),
        "zero_variance": (
            "warning",
            "images have near-zero variance (solid colour or blank). "
            "These provide no learning signal and may degrade model quality.",
        ),
        "near_black": (
            "info",
            "images are almost entirely black. Verify they are not "
            "corrupt or improperly exposed.",
        ),
        "near_white": (
            "info",
            "images are almost entirely white. Verify they are not "
            "corrupt or improperly exposed.",
        ),
    }

    for issue_type, indices in issue_counts.items():
        severity, description = issue_meta.get(
            issue_type, ("info", f"flagged for '{issue_type}'.")
        )
        warning_entry: dict[str, Any] = {
            "type": issue_type,
            "severity": severity,
            "count": len(indices),
            "indices": indices[:20],  # cap for payload size
            "message": f"{len(indices)} {description}",
        }
        paths = issue_images.get(issue_type)
        if paths:
            warning_entry["image_paths"] = paths[:20]
        warnings.append(warning_entry)

    # --- Sort: critical → warning → info ---
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    warnings.sort(key=lambda w: severity_order.get(w["severity"], 9))

    return warnings
