"""Shape statistics for a saliency map, for the attention-regime diagnosis.

BNNR already computes a saliency map for every image and then keeps only which
pixels to mask. The choice between ICD and AICD depends on *where the model is
already looking*, which is measurable from the same maps with no annotation.

Two things have to be pinned down or the numbers do not transfer between runs:

**Resolution.** Normalised entropy of a 7x7 OptiCAM map upsampled to 224x224 is
not the entropy of the native 7x7 map: the upsampling interpolates mass into
neighbouring pixels and raises the entropy. Every statistic here is therefore
computed at one declared resolution, recorded on the result as ``resolution``.
A threshold calibrated at one resolution means nothing at another.

**Perturbation fill.** ``perturbation_shift`` has to overwrite pixels with
something, and the choice is exactly the free parameter T21 is measuring. It is
frozen at ``gaussian_blur`` here and recorded as ``perturbation_fill`` so the
calibration is not confounded by it. The correlation is Spearman, which is
rank-based, so an explainer that rescales its maps does not move the statistic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from bnnr.utils import lazy_cv2 as cv2
from bnnr.xai_analysis import analyze_saliency_map

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn

    from bnnr.xai import BaseExplainer

__all__ = [
    "DEFAULT_RESOLUTION",
    "PERTURBATION_FILL",
    "SaliencyStats",
    "aggregate_saliency_stats",
    "perturbation_shift",
    "saliency_stats_from_map",
]

#: Resolution every statistic is computed at unless the caller declares another.
#: Native CAM grids differ per backbone (7x7 for ResNet-50 at 224, 14x14 for a
#: ViT/16); resampling to a common grid is what makes the numbers comparable.
DEFAULT_RESOLUTION = (14, 14)

#: Frozen fill for :func:`perturbation_shift`. Not configurable on purpose, see
#: the module docstring.
PERTURBATION_FILL = "gaussian_blur"

_BLUR_KERNEL_RATIO = 0.1


@dataclass(frozen=True)
class SaliencyStats:
    """Shape of one saliency map, or the aggregate over a batch of them.

    Attributes
    ----------
    concentration
        ``1 - entropy / log2(H*W)``, on the grid named by ``resolution``. 1 is
        all mass in a single cell, 0 is uniform. This is the normalisation that
        makes entropy comparable across grid sizes; it does not make it
        comparable across *interpolations* of the same map, hence the recorded
        resolution.
    gini
        Gini coefficient of the map values. 0 uniform, 1 all mass in one cell.
        Concentration and gini both measure inequality but disagree on the
        shape of the tail, which is why the diagnosis gets to see both.
    border_mass
        Fraction of total activation inside a 15%-wide border strip. High
        values are the signature of a model attending to padding or framing
        artefacts rather than to the object.
    resolution
        ``(H, W)`` the three statistics above were computed on.
    perturbation_shift
        ``1 - rho(map_before, map_after)`` with Spearman ``rho``, where the
        perturbation blurs the complement of the top-k saliency. ``None`` on a
        result from :func:`saliency_stats_from_map`, which never runs a model.
    perturbation_fill
        Fill used for that perturbation, or ``None`` when it was not run.
    n_maps
        Number of maps behind these numbers. 1 for a single map.
    """

    concentration: float
    gini: float
    border_mass: float
    resolution: tuple[int, int]
    perturbation_shift: float | None = None
    perturbation_fill: str | None = None
    n_maps: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for run records and report artifacts."""
        return asdict(self)


def _resample(map_2d: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resample a 2-D map to ``resolution`` (H, W), preserving nothing but shape."""
    h, w = resolution
    if map_2d.shape == (h, w):
        return map_2d.astype(np.float64, copy=False)
    # INTER_AREA both downsamples without aliasing and, on upsampling, degrades
    # to bilinear, which is what the CAM libraries themselves use.
    resized = cv2.resize(
        map_2d.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_AREA
    )
    return np.asarray(resized, dtype=np.float64)


def saliency_stats_from_map(
    map_2d: np.ndarray,
    *,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> SaliencyStats:
    """Compute the three model-free statistics of one ``[H, W]`` saliency map.

    The map is resampled to ``resolution`` first, so two backbones with
    different native CAM grids produce comparable numbers.

    Entropy, gini and the border strip come from
    :func:`bnnr.xai_analysis.analyze_saliency_map`, which is the one
    implementation of them in the library.

    An all-zero map has no shape to speak of; it comes back as concentration 0,
    gini 0, border_mass 0 rather than as a division by zero.
    """
    if map_2d.ndim != 2:
        raise ValueError(f"saliency map must be 2-D, got shape {map_2d.shape}")

    grid = _resample(map_2d, resolution)
    h, w = grid.shape
    raw = analyze_saliency_map(grid)

    max_entropy = float(np.log2(h * w)) if h * w > 1 else 0.0
    if max_entropy > 0.0 and float(grid.sum()) > 1e-8:
        concentration = 1.0 - raw["entropy"] / max_entropy
    else:
        concentration = 0.0

    return SaliencyStats(
        concentration=float(np.clip(concentration, 0.0, 1.0)),
        gini=float(raw["gini"]),
        border_mass=float(raw["edge_ratio"]),
        resolution=(h, w),
        n_maps=1,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tied ranks, the convention Spearman is defined against."""
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.float64)

    # Average the ranks within each run of equal values. A constant map is all
    # one tie group, which is what makes the correlation undefined below.
    sorted_vals = flat[order]
    start = 0
    for i in range(1, flat.size + 1):
        if i == flat.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho between two maps, 0.0 when either is constant.

    A constant map has zero rank variance, so rho is undefined. Returning 0.0
    reports "no monotone relationship", which sends ``perturbation_shift`` to
    1.0: a map that carries no ordering did not survive the perturbation in any
    usable sense.
    """
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(float(ra @ ra) * float(rb @ rb)))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(float(ra @ rb) / denom, -1.0, 1.0))


def _blur_complement(images: Tensor, maps: np.ndarray, top_k: float) -> Tensor:
    """Blur everything outside the top-k saliency of each image.

    Keeping the top-k and destroying the rest is the deletion-style probe: if
    the model was genuinely reading the highlighted region, its explanation
    should barely move.
    """
    import torch
    import torchvision.transforms.functional as tv_functional

    b, _, h, w = images.shape
    k_size = max(3, int(max(h, w) * _BLUR_KERNEL_RATIO))
    if k_size % 2 == 0:
        k_size += 1
    blurred = tv_functional.gaussian_blur(images, [k_size, k_size])

    keep = np.empty((b, h, w), dtype=bool)
    for i in range(b):
        grid = _resample(maps[i], (h, w))
        flat = grid.ravel()
        n_keep = max(1, int(round(top_k * flat.size)))
        # argpartition on the negated map puts the n_keep largest first.
        idx = np.argpartition(-flat, n_keep - 1)[:n_keep]
        flat_keep = np.zeros(flat.size, dtype=bool)
        flat_keep[idx] = True
        keep[i] = flat_keep.reshape(h, w)

    keep_t = torch.as_tensor(keep, device=images.device).unsqueeze(1)
    return torch.where(keep_t, images, blurred)


def perturbation_shift(
    model: nn.Module,
    explainer: BaseExplainer,
    images: Tensor,
    labels: Tensor,
    target_layers: list[nn.Module],
    *,
    top_k: float = 0.2,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
) -> tuple[list[float], np.ndarray]:
    """How far each explanation moves when the unattended region is destroyed.

    Returns ``(shifts, maps_before)``: one ``1 - rho`` per image in ``images``,
    and the unperturbed maps, so a caller that also wants
    :func:`saliency_stats_from_map` does not pay for a second explainer pass.

    ``top_k`` is the fraction of pixels kept intact. The complement is filled
    with ``gaussian_blur`` (:data:`PERTURBATION_FILL`); this is deliberately not
    a parameter, see the module docstring.

    A shift near 0 means the explanation is unchanged by removing everything it
    did not point at, which is the signature of a model reading a genuine local
    feature. A shift near 1 means the explanation was an artefact of the
    context.
    """
    if not 0.0 < top_k <= 1.0:
        raise ValueError(f"top_k must be in (0, 1], got {top_k}")

    maps_before = np.asarray(explainer.explain(model, images, labels, target_layers))
    perturbed = _blur_complement(images, maps_before, top_k)
    maps_after = np.asarray(explainer.explain(model, perturbed, labels, target_layers))

    shifts: list[float] = []
    for before, after in zip(maps_before, maps_after):
        rho = _spearman(_resample(before, resolution), _resample(after, resolution))
        shifts.append(float(np.clip(1.0 - rho, 0.0, 2.0)))
    return shifts, maps_before


def aggregate_saliency_stats(
    stats: list[SaliencyStats],
    *,
    shifts: list[float] | None = None,
) -> SaliencyStats:
    """Reduce per-image stats to one batch-level :class:`SaliencyStats`.

    The reduction is the **median**, not the mean, and this is a decision worth
    stating: saliency statistics on a real dataset are skewed by a handful of
    images the explainer fails on (a flat map, a map dominated by one artefact).
    A mean lets those images move a threshold; a median does not.

    ``resolution`` must agree across all inputs, since mixing resolutions is the
    exact failure the recorded field exists to catch. ``shifts`` is the optional
    per-image output of :func:`perturbation_shift`, reduced the same way.
    """
    if not stats:
        raise ValueError("aggregate_saliency_stats needs at least one SaliencyStats")

    resolutions = {s.resolution for s in stats}
    if len(resolutions) > 1:
        raise ValueError(
            f"cannot aggregate stats computed at different resolutions: {sorted(resolutions)}"
        )

    shift = float(np.median(shifts)) if shifts else None
    return SaliencyStats(
        concentration=float(np.median([s.concentration for s in stats])),
        gini=float(np.median([s.gini for s in stats])),
        border_mass=float(np.median([s.border_mass for s in stats])),
        resolution=resolutions.pop(),
        perturbation_shift=shift,
        perturbation_fill=PERTURBATION_FILL if shifts else None,
        n_maps=sum(s.n_maps for s in stats),
    )
